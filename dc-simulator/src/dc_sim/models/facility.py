"""Top-level facility model composing thermal, power, workload, carbon, GPU, network, storage, cooling."""

from dataclasses import dataclass, field

from dc_sim.clock import SimulationClock
from dc_sim.config import SimConfig
from dc_sim.models.carbon import CarbonModel, CarbonState
from dc_sim.models.cooling import CoolingModel, FacilityCoolingState
from dc_sim.models.gpu import GpuModel, FacilityGpuState
from dc_sim.models.network import NetworkModel, FacilityNetworkState
from dc_sim.models.power import FacilityPowerState, PowerModel
from dc_sim.models.storage import StorageModel, FacilityStorageState
from dc_sim.models.thermal import FacilityThermalState, ThermalModel
from dc_sim.models.workload import WorkloadQueue


@dataclass
class FacilityState:
    """Snapshot of entire facility state after a tick."""

    current_time: float
    tick_count: int
    thermal: FacilityThermalState
    power: FacilityPowerState
    carbon: CarbonState
    gpu: FacilityGpuState = field(default_factory=FacilityGpuState)
    network: FacilityNetworkState = field(default_factory=FacilityNetworkState)
    storage: FacilityStorageState = field(default_factory=FacilityStorageState)
    cooling: FacilityCoolingState = field(default_factory=FacilityCoolingState)
    workload_pending: int = 0
    workload_running: int = 0
    workload_completed: int = 0
    sla_violations: int = 0


class Facility:
    """Orchestrates thermal, power, workload, carbon, GPU, network, storage, and cooling models."""

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
        self.gpu_model = GpuModel(config, rng_seed=rng_seed)
        self.network_model = NetworkModel(config, rng_seed=rng_seed)
        self.storage_model = StorageModel(config, rng_seed=rng_seed)
        self.cooling_model = CoolingModel(config, rng_seed=rng_seed)
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
        network_partition_racks: set[int] | None = None,
    ) -> FacilityState:
        """
        Advance simulation by one tick.
        Order: workload -> power -> thermal -> GPU -> network -> storage -> cooling -> carbon.
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

        # 5. Per-GPU telemetry
        thermal_inlets = {r.rack_id: r.inlet_temp_c for r in thermal_state.racks}
        gpu_state = self.gpu_model.step(
            server_gpu_utilisation=server_gpu_util,
            thermal_rack_inlets=thermal_inlets,
            throttled_racks=throttled_racks,
            running_jobs=list(self.workload_queue.running),
            sim_time=self.clock.current_time,
        )

        # 6. Network traffic
        network_state = self.network_model.step(
            server_gpu_utilisation=server_gpu_util,
            running_jobs=list(self.workload_queue.running),
            network_partition_racks=network_partition_racks,
            sim_time=self.clock.current_time,
        )

        # 7. Storage I/O
        storage_state = self.storage_model.step(
            server_gpu_utilisation=server_gpu_util,
            running_jobs=list(self.workload_queue.running),
            sim_time=self.clock.current_time,
            tick_interval_s=self.clock.tick_interval_s,
        )

        # 8. Cooling system
        total_it_heat = power_state.it_power_kw  # All IT power becomes heat
        crac_failed = self.cooling_model.get_failed_units()
        cooling_state = self.cooling_model.step(
            total_it_heat_kw=total_it_heat,
            ambient_temp_c=ambient_temp,
            crac_setpoints=self._crac_setpoints,
            crac_failed_units=crac_failed,
            sim_time=self.clock.current_time,
        )

        # 9. Carbon and cost
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
            gpu=gpu_state,
            network=network_state,
            storage=storage_state,
            cooling=cooling_state,
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
        self.gpu_model.reset()
        self.network_model.reset()
        self.storage_model.reset()
        self.cooling_model.reset()
        self._server_power_caps.clear()
        self._last_thermal = FacilityThermalState()
