"""Power model: per-server and per-rack power draw based on GPU utilisation.

Improvements over v1:
- Time-varying PUE (depends on outside temperature and facility load)
- Non-linear GPU power curve (power doesn't scale perfectly linearly with util)
"""

import math
from dataclasses import dataclass, field

from dc_sim.config import SimConfig


@dataclass
class ServerPowerState:
    """Power state for a single server."""

    server_id: str
    rack_id: int
    gpu_utilisation: float
    gpu_power_draw_w: float
    total_power_draw_w: float
    power_cap_pct: float | None = None  # None = no cap, else 0-100


@dataclass
class RackPowerState:
    """Power state for a rack."""

    rack_id: int
    total_power_kw: float
    pdu_utilisation_pct: float
    servers: list[ServerPowerState] = field(default_factory=list)


@dataclass
class FacilityPowerState:
    """Aggregate facility power state."""

    it_power_kw: float
    total_power_kw: float
    pue: float
    headroom_kw: float
    power_cap_exceeded: bool
    racks: list[RackPowerState] = field(default_factory=list)


class PowerModel:
    """Calculates power draw from GPU utilisation."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.facility = config.facility
        self.power_cfg = config.power

    def _gpu_power_curve(self, utilisation: float) -> float:
        """Non-linear GPU power: GPUs draw ~40% TDP at idle,
        then scale superlinearly toward 100% TDP at full util.

        This models real GPU behaviour where power rises faster
        at high utilisation due to voltage/frequency scaling.
        """
        idle_fraction = 0.05  # 5% TDP at zero utilisation (fans, memory)
        if utilisation <= 0.0:
            return idle_fraction * self.power_cfg.gpu_tdp_watts
        # Cubic curve: rises slowly at low util, steeply at high util
        active_power = (0.3 * utilisation + 0.7 * utilisation ** 2) * self.power_cfg.gpu_tdp_watts
        return (idle_fraction + (1.0 - idle_fraction) * (active_power / self.power_cfg.gpu_tdp_watts)) * self.power_cfg.gpu_tdp_watts

    def compute_dynamic_pue(self, it_power_kw: float, ambient_temp_c: float = 22.0) -> float:
        """PUE varies with load and ambient temperature.

        At low load: PUE is higher (fixed overhead dominates)
        At high load: PUE approaches the configured base
        At high ambient temps: PUE increases (chillers work harder)
        """
        base_pue = self.power_cfg.pue_overhead_factor

        # Load factor: PUE is worse at low utilisation
        max_it = self.power_cfg.facility_power_cap_kw / base_pue
        load_fraction = min(1.0, it_power_kw / max(1.0, max_it))
        # At 10% load, PUE is ~20% worse; at 100% load, it's at base
        load_penalty = 0.2 * (1.0 - load_fraction) ** 2

        # Ambient temp: every degree above 22°C adds ~0.005 to PUE
        temp_penalty = max(0.0, (ambient_temp_c - 22.0) * 0.005)

        return base_pue + load_penalty + temp_penalty

    def compute(
        self,
        server_gpu_utilisation: dict[str, float],
        throttled_racks: set[int],
        server_power_caps: dict[str, float],
        server_max_util_override: dict[str, float] | None = None,
        rack_power_multiplier: dict[int, float] | None = None,
        ambient_temp_c: float = 22.0,
    ) -> FacilityPowerState:
        """
        Compute power state from server GPU utilisations.
        server_gpu_utilisation: server_id -> 0.0-1.0
        throttled_racks: rack IDs that are thermally throttled (cap util at 0.5)
        server_power_caps: server_id -> power cap percentage (0-100)
        server_max_util_override: server_id -> max util (e.g. 0.3 for GPU degraded)
        rack_power_multiplier: rack_id -> multiplier (e.g. 1.2 for PDU spike)
        ambient_temp_c: current ambient temperature for dynamic PUE
        """
        server_max_util_override = server_max_util_override or {}
        rack_power_multiplier = rack_power_multiplier or {}
        racks: list[RackPowerState] = []
        total_it_power_w = 0.0

        for rack_id in range(self.facility.num_racks):
            rack_servers: list[ServerPowerState] = []
            rack_power_w = 0.0

            for srv_idx in range(self.facility.servers_per_rack):
                server_id = f"rack-{rack_id}-srv-{srv_idx}"
                raw_util = server_gpu_utilisation.get(server_id, 0.05)

                # Thermal throttling caps at 50%
                if rack_id in throttled_racks:
                    raw_util = min(raw_util, 0.5)

                # GPU degraded caps at 30%
                max_util = server_max_util_override.get(server_id)
                if max_util is not None:
                    raw_util = min(raw_util, max_util)

                # Power cap (e.g. throttle_gpu action)
                power_cap = server_power_caps.get(server_id)
                if power_cap is not None:
                    raw_util = raw_util * (power_cap / 100.0)

                gpu_power = self._gpu_power_curve(raw_util) * self.facility.gpus_per_server
                total_server = self.power_cfg.server_base_power_watts + gpu_power
                rack_power_w += total_server

                rack_servers.append(
                    ServerPowerState(
                        server_id=server_id,
                        rack_id=rack_id,
                        gpu_utilisation=raw_util,
                        gpu_power_draw_w=gpu_power,
                        total_power_draw_w=total_server,
                        power_cap_pct=power_cap,
                    )
                )

            mult = rack_power_multiplier.get(rack_id, 1.0)
            rack_power_kw = (rack_power_w / 1000.0) * mult
            pdu_util = (rack_power_kw / self.power_cfg.pdu_capacity_kw) * 100.0
            racks.append(
                RackPowerState(
                    rack_id=rack_id,
                    total_power_kw=rack_power_kw,
                    pdu_utilisation_pct=pdu_util,
                    servers=rack_servers,
                )
            )
            total_it_power_w += rack_power_w * mult

        it_power_kw = total_it_power_w / 1000.0
        pue = self.compute_dynamic_pue(it_power_kw, ambient_temp_c)
        total_power_kw = it_power_kw * pue
        headroom = self.power_cfg.facility_power_cap_kw - total_power_kw
        cap_exceeded = total_power_kw > self.power_cfg.facility_power_cap_kw

        return FacilityPowerState(
            it_power_kw=it_power_kw,
            total_power_kw=total_power_kw,
            pue=pue,
            headroom_kw=headroom,
            power_cap_exceeded=cap_exceeded,
            racks=racks,
        )
