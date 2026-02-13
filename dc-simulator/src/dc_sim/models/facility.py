"""Top-level facility model composing thermal, power, workload, and carbon."""

from dataclasses import dataclass, field

from dc_sim.clock import SimulationClock
from dc_sim.config import SimConfig
from dc_sim.models.carbon import CarbonModel, CarbonState
from dc_sim.models.power import FacilityPowerState
from dc_sim.models.thermal import FacilityThermalState
from dc_sim.models.workload import WorkloadQueue
from dc_sim.models.power import PowerModel
from dc_sim.models.thermal import ThermalModel


@dataclass
class FacilityState:
    """Snapshot of entire facility state after a tick."""

    current_time: float
    tick_count: int
    thermal: FacilityThermalState
    power: FacilityPowerState
    carbon: CarbonState
    workload_pending: int
    workload_running: int
    workload_completed: int
    sla_violations: int


class Facility:
    """Orchestrates thermal, power, workload, and carbon models."""

    def __init__(
        self,
        config: SimConfig,
        clock: SimulationClock,
        workload_queue: WorkloadQueue | None = None,
        rng_seed: int = 42,
    ):
        self.config = config
        self.clock = clock
        self.power_model = PowerModel(config)
        self.thermal_model = ThermalModel(config, rng_seed=rng_seed)
        self.carbon_model = CarbonModel(config, rng_seed=rng_seed)
        self.workload_queue = workload_queue or WorkloadQueue(config)

        self._server_power_caps: dict[str, float] = {}
        self._crac_setpoints: dict[int, float] = {}
        self._cooling_capacity_factor: dict[int, float] = {}
        self._last_thermal = FacilityThermalState()

    def step(
        self,
        cooling_capacity_factor: dict[int, float] | None = None,
        server_max_util_override: dict[str, float] | None = None,
        rack_power_multiplier: dict[int, float] | None = None,
    ) -> FacilityState:
        """
        Advance simulation by one tick.
        Order: workload -> power -> thermal -> carbon.
        """
        # Use provided cooling factor or default (all 1.0)
        if cooling_capacity_factor is None:
            cooling_capacity_factor = {r: 1.0 for r in range(self.config.facility.num_racks)}

        # 1. Workload: arrivals, scheduling, completion, GPU utilisation
        server_gpu_util = self.workload_queue.step(self.clock.current_time)

        # 2. Thermal throttling from previous state (we don't have it in first tick)
        throttled_racks = set()
        for rack in getattr(self, "_last_thermal", FacilityThermalState()).racks:
            if rack.throttled:
                throttled_racks.add(rack.rack_id)

        # Get ambient temp from previous thermal state for power model
        ambient_temp = getattr(self._last_thermal, "ambient_temp_c", self.config.thermal.ambient_temp_c)

        # 3. Power (now with ambient temp for dynamic PUE)
        power_state = self.power_model.compute(
            server_gpu_utilisation=server_gpu_util,
            throttled_racks=throttled_racks,
            server_power_caps=self._server_power_caps,
            server_max_util_override=server_max_util_override,
            rack_power_multiplier=rack_power_multiplier,
            ambient_temp_c=ambient_temp,
        )

        # 4. Thermal: need rack power and cooling factor
        rack_power = {r.rack_id: r.total_power_kw for r in power_state.racks}
        thermal_state = self.thermal_model.step(
            rack_power_kw=rack_power,
            cooling_capacity_factor=cooling_capacity_factor,
            tick_interval_s=self.clock.tick_interval_s,
            sim_time=self.clock.current_time,
        )
        self._last_thermal = thermal_state

        # 5. Carbon and cost
        carbon_state = self.carbon_model.step(
            sim_time=self.clock.current_time,
            total_power_kw=power_state.total_power_kw,
            tick_interval_s=self.clock.tick_interval_s,
        )

        return FacilityState(
            current_time=self.clock.current_time,
            tick_count=self.clock.tick_count,
            thermal=thermal_state,
            power=power_state,
            carbon=carbon_state,
            workload_pending=len(self.workload_queue.pending),
            workload_running=len(self.workload_queue.running),
            workload_completed=len(self.workload_queue.completed),
            sla_violations=len(self.workload_queue.get_sla_violations()),
        )

    def set_server_power_cap(self, server_id: str, power_cap_pct: float | None) -> None:
        """Set GPU power cap for a server (None to clear)."""
        if power_cap_pct is None:
            self._server_power_caps.pop(server_id, None)
        else:
            self._server_power_caps[server_id] = power_cap_pct

    def reset(self) -> None:
        """Reset all models to initial state."""
        self.workload_queue.reset()
        self.thermal_model.reset()
        self.carbon_model.reset()
        self._server_power_caps.clear()
        self._last_thermal = FacilityThermalState()
