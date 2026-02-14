"""Per-GPU metrics model: temperature, memory, clock speed, ECC errors, PCIe bandwidth.

Models realistic NVIDIA-style GPU telemetry at individual device granularity.
Each server has `gpus_per_server` GPUs, each tracked independently.
"""

import math
from dataclasses import dataclass, field

from dc_sim.config import SimConfig


@dataclass
class GpuState:
    """Telemetry snapshot for a single GPU device."""

    gpu_id: str  # e.g. "rack-0-srv-1-gpu-2"
    server_id: str
    rack_id: int
    # Core metrics
    sm_utilisation_pct: float = 0.0  # Streaming multiprocessor util (0-100)
    mem_utilisation_pct: float = 0.0  # Memory controller util (0-100)
    gpu_temp_c: float = 35.0  # Junction temperature
    mem_temp_c: float = 30.0  # HBM/GDDR temperature
    power_draw_w: float = 15.0  # Board power
    # Clocks
    sm_clock_mhz: int = 210  # Current SM clock (can boost or throttle)
    mem_clock_mhz: int = 1215  # Current memory clock
    # Memory
    mem_used_mib: int = 0
    mem_total_mib: int = 81920  # 80 GiB HBM3 (H100-class)
    # Errors
    ecc_sbe_count: int = 0  # Single-bit ECC correctable
    ecc_dbe_count: int = 0  # Double-bit ECC uncorrectable
    # PCIe
    pcie_tx_gbps: float = 0.0  # PCIe transmit throughput
    pcie_rx_gbps: float = 0.0  # PCIe receive throughput
    # NVLink (inter-GPU fabric)
    nvlink_tx_gbps: float = 0.0
    nvlink_rx_gbps: float = 0.0
    # Fan / thermal throttle
    fan_speed_pct: float = 30.0  # 0-100
    thermal_throttle: bool = False
    power_throttle: bool = False


@dataclass
class ServerGpuState:
    """Aggregate GPU state for one server."""

    server_id: str
    rack_id: int
    gpus: list[GpuState] = field(default_factory=list)
    total_gpu_power_w: float = 0.0
    avg_gpu_temp_c: float = 35.0
    total_mem_used_mib: int = 0
    total_mem_total_mib: int = 0


@dataclass
class FacilityGpuState:
    """Facility-wide GPU telemetry."""

    servers: list[ServerGpuState] = field(default_factory=list)
    total_gpus: int = 0
    healthy_gpus: int = 0
    throttled_gpus: int = 0
    ecc_error_gpus: int = 0  # GPUs with any DBE
    avg_gpu_temp_c: float = 35.0
    avg_sm_util_pct: float = 0.0
    total_gpu_mem_used_mib: int = 0
    total_gpu_mem_total_mib: int = 0


class GpuModel:
    """Simulates per-GPU telemetry based on workload utilisation and thermal state.

    Realistic behaviours modelled:
    - GPU temperature rises non-linearly with SM utilisation
    - Memory temperature correlates with memory utilisation
    - Clock speeds boost at low temps, throttle at high temps
    - Fan speed ramps with temperature
    - ECC errors accumulate probabilistically (rare)
    - PCIe/NVLink bandwidth scales with workload type
    - Memory allocation correlates with job type (training > inference)
    - Power/thermal throttling at extreme conditions
    """

    # NVIDIA H100-class reference parameters
    BASE_SM_CLOCK_MHZ = 1410
    BOOST_SM_CLOCK_MHZ = 1980
    BASE_MEM_CLOCK_MHZ = 1593  # HBM3
    GPU_TDP_W = 300  # Will be overridden from config
    MEM_TOTAL_MIB = 81920  # 80 GiB HBM3
    PCIE_MAX_GBPS = 64.0  # PCIe Gen5 x16
    NVLINK_MAX_GBPS = 450.0  # NVLink 4.0 per direction

    # Thermal model for GPU die
    AMBIENT_TO_IDLE_OFFSET = 13.0  # GPU idle is ~13°C above inlet
    TEMP_PER_UTIL_FACTOR = 0.55  # °C per % utilisation
    MEM_TEMP_OFFSET = -5.0  # Memory runs slightly cooler than die
    THERMAL_THROTTLE_TEMP = 83.0  # °C — GPU starts throttling
    SHUTDOWN_TEMP = 95.0  # °C — GPU would shut down
    FAN_RAMP_THRESHOLD = 50.0  # °C — fans start ramping above idle

    # ECC error rates (per GPU per tick)
    SBE_RATE_PER_TICK = 0.0005  # ~0.05% chance per tick
    DBE_RATE_PER_TICK = 0.00002  # ~0.002% — very rare

    def __init__(self, config: SimConfig, rng_seed: int = 42):
        self.config = config
        self.GPU_TDP_W = config.power.gpu_tdp_watts
        self._rng = __import__("numpy").random.default_rng(rng_seed + 300)

        # Persistent state: ECC error accumulators per GPU
        self._ecc_sbe: dict[str, int] = {}
        self._ecc_dbe: dict[str, int] = {}

    def step(
        self,
        server_gpu_utilisation: dict[str, float],
        thermal_rack_inlets: dict[int, float],
        throttled_racks: set[int],
        running_jobs: list | None = None,
        sim_time: float = 0.0,
    ) -> FacilityGpuState:
        """Compute per-GPU telemetry for all GPUs in the facility.

        Args:
            server_gpu_utilisation: server_id -> average GPU util (0.0-1.0)
            thermal_rack_inlets: rack_id -> inlet temp (°C)
            throttled_racks: set of rack IDs that are thermally throttled
            running_jobs: list of running Job objects (for memory/bandwidth estimation)
            sim_time: current simulation time in seconds
        """
        facility = self.config.facility
        servers: list[ServerGpuState] = []
        all_temps = []
        all_utils = []
        total_mem_used = 0
        total_mem_total = 0
        total_gpus = 0
        healthy = 0
        throttled_count = 0
        ecc_error_count = 0

        # Build per-server job type map for memory/bandwidth estimation
        server_job_types: dict[str, str] = {}
        if running_jobs:
            for job in running_jobs:
                for srv in getattr(job, "assigned_servers", []):
                    server_job_types[srv] = getattr(job, "job_type", "batch")

        for rack_id in range(facility.num_racks):
            inlet_temp = thermal_rack_inlets.get(rack_id, 22.0)
            is_throttled_rack = rack_id in throttled_racks

            for srv_idx in range(facility.servers_per_rack):
                server_id = f"rack-{rack_id}-srv-{srv_idx}"
                avg_util = server_gpu_utilisation.get(server_id, 0.05)
                job_type = server_job_types.get(server_id, "batch")

                gpu_states = []
                srv_total_power = 0.0
                srv_total_mem = 0
                srv_temps = []

                for gpu_idx in range(facility.gpus_per_server):
                    gpu_id = f"{server_id}-gpu-{gpu_idx}"
                    total_gpus += 1

                    # Per-GPU util varies slightly from server average
                    noise = self._rng.normal(0, 0.02)
                    gpu_util = max(0.0, min(1.0, avg_util + noise))
                    sm_pct = gpu_util * 100.0

                    # ── Temperature ──
                    # Non-linear: rises faster at high util
                    base_temp = inlet_temp + self.AMBIENT_TO_IDLE_OFFSET
                    heat_rise = (self.TEMP_PER_UTIL_FACTOR * sm_pct
                                 + 0.003 * sm_pct ** 1.5)
                    # Small per-GPU jitter
                    jitter = self._rng.normal(0, 0.8)
                    gpu_temp = base_temp + heat_rise + jitter

                    mem_temp = gpu_temp + self.MEM_TEMP_OFFSET
                    # Memory-bound workloads warm HBM more
                    if job_type == "training":
                        mem_temp += 3.0

                    # ── Throttling ──
                    thermal_thr = gpu_temp >= self.THERMAL_THROTTLE_TEMP
                    if thermal_thr or is_throttled_rack:
                        sm_pct = min(sm_pct, 50.0)
                        gpu_util = sm_pct / 100.0
                        throttled_count += 1

                    # ── Power ──
                    idle_power = 0.05 * self.GPU_TDP_W
                    active_power = (0.3 * gpu_util + 0.7 * gpu_util ** 2) * self.GPU_TDP_W
                    gpu_power = idle_power + (1.0 - 0.05) * active_power
                    power_thr = gpu_power >= 0.95 * self.GPU_TDP_W
                    if power_thr:
                        gpu_power = 0.95 * self.GPU_TDP_W

                    # ── Clocks ──
                    # Boost at low-mid temps, throttle at high
                    if gpu_temp < 70:
                        clock_frac = 1.0
                    elif gpu_temp < self.THERMAL_THROTTLE_TEMP:
                        clock_frac = 1.0 - (gpu_temp - 70) / (self.THERMAL_THROTTLE_TEMP - 70) * 0.15
                    else:
                        clock_frac = 0.7  # Hard throttle
                    sm_clock = int(self.BASE_SM_CLOCK_MHZ
                                   + (self.BOOST_SM_CLOCK_MHZ - self.BASE_SM_CLOCK_MHZ) * clock_frac * gpu_util)
                    mem_clock = self.BASE_MEM_CLOCK_MHZ  # Memory clock is usually fixed

                    # ── Memory allocation ──
                    if gpu_util < 0.01:
                        mem_used = int(self.MEM_TOTAL_MIB * 0.01)  # ~800 MiB driver overhead
                    elif job_type == "training":
                        # Training uses 60-95% memory (model + optimizer + activations)
                        mem_frac = 0.6 + 0.35 * gpu_util
                        mem_used = int(self.MEM_TOTAL_MIB * mem_frac)
                    elif job_type == "inference":
                        # Inference uses 20-50% (model weights + KV cache)
                        mem_frac = 0.2 + 0.3 * gpu_util
                        mem_used = int(self.MEM_TOTAL_MIB * mem_frac)
                    else:  # batch
                        mem_frac = 0.3 + 0.4 * gpu_util
                        mem_used = int(self.MEM_TOTAL_MIB * mem_frac)

                    mem_util = (mem_used / self.MEM_TOTAL_MIB) * 100.0

                    # ── Fan speed ──
                    if gpu_temp < self.FAN_RAMP_THRESHOLD:
                        fan_pct = 30.0  # Minimum idle speed
                    else:
                        fan_pct = 30.0 + 70.0 * ((gpu_temp - self.FAN_RAMP_THRESHOLD)
                                                   / (self.THERMAL_THROTTLE_TEMP - self.FAN_RAMP_THRESHOLD))
                    fan_pct = min(100.0, max(30.0, fan_pct))

                    # ── PCIe bandwidth ──
                    # Scales with utilisation; training jobs use more DMA
                    pcie_base = gpu_util * self.PCIE_MAX_GBPS * 0.4
                    if job_type == "training":
                        pcie_base *= 1.5  # AllReduce gradient syncs
                    pcie_tx = min(self.PCIE_MAX_GBPS, pcie_base * (0.9 + self._rng.random() * 0.2))
                    pcie_rx = min(self.PCIE_MAX_GBPS, pcie_base * (0.9 + self._rng.random() * 0.2))

                    # ── NVLink bandwidth ──
                    # Only active when multi-GPU jobs run on same server
                    nvlink_tx, nvlink_rx = 0.0, 0.0
                    if job_type == "training" and gpu_util > 0.1:
                        # Training uses NVLink for tensor parallelism / allreduce
                        nvlink_frac = gpu_util * 0.5
                        nvlink_tx = nvlink_frac * self.NVLINK_MAX_GBPS * (0.85 + self._rng.random() * 0.3)
                        nvlink_rx = nvlink_frac * self.NVLINK_MAX_GBPS * (0.85 + self._rng.random() * 0.3)
                        nvlink_tx = min(self.NVLINK_MAX_GBPS, nvlink_tx)
                        nvlink_rx = min(self.NVLINK_MAX_GBPS, nvlink_rx)

                    # ── ECC errors ──
                    # Initialise counters if new GPU
                    if gpu_id not in self._ecc_sbe:
                        self._ecc_sbe[gpu_id] = 0
                        self._ecc_dbe[gpu_id] = 0

                    # Probability increases with temperature
                    temp_factor = 1.0 + max(0, (gpu_temp - 70) * 0.02)
                    if self._rng.random() < self.SBE_RATE_PER_TICK * temp_factor:
                        self._ecc_sbe[gpu_id] += 1
                    if self._rng.random() < self.DBE_RATE_PER_TICK * temp_factor:
                        self._ecc_dbe[gpu_id] += 1

                    sbe = self._ecc_sbe[gpu_id]
                    dbe = self._ecc_dbe[gpu_id]
                    if dbe > 0:
                        ecc_error_count += 1

                    if not thermal_thr and not power_thr:
                        healthy += 1

                    gpu_state = GpuState(
                        gpu_id=gpu_id,
                        server_id=server_id,
                        rack_id=rack_id,
                        sm_utilisation_pct=round(sm_pct, 1),
                        mem_utilisation_pct=round(mem_util, 1),
                        gpu_temp_c=round(gpu_temp, 1),
                        mem_temp_c=round(mem_temp, 1),
                        power_draw_w=round(gpu_power, 1),
                        sm_clock_mhz=sm_clock,
                        mem_clock_mhz=mem_clock,
                        mem_used_mib=mem_used,
                        mem_total_mib=self.MEM_TOTAL_MIB,
                        ecc_sbe_count=sbe,
                        ecc_dbe_count=dbe,
                        pcie_tx_gbps=round(pcie_tx, 2),
                        pcie_rx_gbps=round(pcie_rx, 2),
                        nvlink_tx_gbps=round(nvlink_tx, 2),
                        nvlink_rx_gbps=round(nvlink_rx, 2),
                        fan_speed_pct=round(fan_pct, 1),
                        thermal_throttle=thermal_thr,
                        power_throttle=power_thr,
                    )
                    gpu_states.append(gpu_state)
                    srv_total_power += gpu_power
                    srv_total_mem += mem_used
                    srv_temps.append(gpu_temp)
                    all_temps.append(gpu_temp)
                    all_utils.append(sm_pct)

                srv_state = ServerGpuState(
                    server_id=server_id,
                    rack_id=rack_id,
                    gpus=gpu_states,
                    total_gpu_power_w=round(srv_total_power, 1),
                    avg_gpu_temp_c=round(sum(srv_temps) / max(1, len(srv_temps)), 1),
                    total_mem_used_mib=srv_total_mem,
                    total_mem_total_mib=self.MEM_TOTAL_MIB * facility.gpus_per_server,
                )
                servers.append(srv_state)
                total_mem_used += srv_total_mem
                total_mem_total += self.MEM_TOTAL_MIB * facility.gpus_per_server

        return FacilityGpuState(
            servers=servers,
            total_gpus=total_gpus,
            healthy_gpus=healthy,
            throttled_gpus=throttled_count,
            ecc_error_gpus=ecc_error_count,
            avg_gpu_temp_c=round(sum(all_temps) / max(1, len(all_temps)), 1) if all_temps else 35.0,
            avg_sm_util_pct=round(sum(all_utils) / max(1, len(all_utils)), 1) if all_utils else 0.0,
            total_gpu_mem_used_mib=total_mem_used,
            total_gpu_mem_total_mib=total_mem_total,
        )

    def reset(self) -> None:
        """Clear persistent ECC counters."""
        self._ecc_sbe.clear()
        self._ecc_dbe.clear()
