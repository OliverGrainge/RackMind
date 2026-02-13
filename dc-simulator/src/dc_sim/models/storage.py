"""Storage I/O model: IOPS, throughput, latency for shared storage fabric.

Models a realistic NVMe-oF (NVMe over Fabrics) shared storage system with:
- Per-rack NVMe storage shelves
- Read/write IOPS based on workload type
- Throughput (GB/s) with bandwidth saturation
- Latency modelling with queue depth effects
- Storage capacity and utilisation tracking
"""

import math
from dataclasses import dataclass, field

from dc_sim.config import SimConfig


@dataclass
class RackStorageState:
    """Storage telemetry for a single rack's local NVMe shelf."""

    rack_id: int
    # IOPS
    read_iops: int = 0
    write_iops: int = 0
    total_iops: int = 0
    max_iops: int = 1_000_000  # NVMe shelf IOPS capacity
    # Throughput
    read_throughput_gbps: float = 0.0
    write_throughput_gbps: float = 0.0
    max_throughput_gbps: float = 25.0  # NVMe-oF shelf bandwidth
    # Latency
    avg_read_latency_us: float = 80.0  # NVMe baseline ~80μs
    avg_write_latency_us: float = 20.0  # NVMe write ~20μs
    p99_read_latency_us: float = 200.0
    # Capacity
    used_tb: float = 0.0
    total_tb: float = 30.0  # Per-rack NVMe capacity
    utilisation_pct: float = 0.0
    # Health
    drive_health_pct: float = 100.0  # TBW (Terabytes Written) wear indicator
    queue_depth: int = 0


@dataclass
class FacilityStorageState:
    """Facility-wide storage telemetry."""

    racks: list[RackStorageState] = field(default_factory=list)
    # Aggregates
    total_read_iops: int = 0
    total_write_iops: int = 0
    total_read_throughput_gbps: float = 0.0
    total_write_throughput_gbps: float = 0.0
    total_used_tb: float = 0.0
    total_capacity_tb: float = 0.0
    avg_read_latency_us: float = 80.0
    avg_write_latency_us: float = 20.0


class StorageModel:
    """Simulates per-rack NVMe storage I/O based on workload type.

    Key behaviours:
    - Training jobs: large sequential reads (dataset), periodic writes (checkpoints)
    - Inference jobs: small random reads (model weights at startup), minimal writes
    - Batch jobs: heavy mixed I/O (data ingest + result export)
    - IOPS and throughput scale with GPU utilisation
    - Latency degrades under high queue depth (Little's Law)
    - Drive wear accumulates over time from writes
    """

    # NVMe shelf parameters
    MAX_IOPS = 1_000_000  # Per-rack NVMe shelf
    MAX_THROUGHPUT_GBPS = 25.0
    BASE_READ_LATENCY_US = 80.0
    BASE_WRITE_LATENCY_US = 20.0
    CAPACITY_PER_RACK_TB = 30.0

    # I/O profiles per server at 100% util (IOPS)
    TRAINING_READ_IOPS = 50_000  # Large sequential reads
    TRAINING_WRITE_IOPS = 5_000  # Checkpoint writes
    INFERENCE_READ_IOPS = 8_000  # Mostly cached, some KV writes
    INFERENCE_WRITE_IOPS = 500
    BATCH_READ_IOPS = 30_000  # Data ingest
    BATCH_WRITE_IOPS = 15_000  # Result export

    # Throughput profiles per server at 100% util (Gbps)
    TRAINING_READ_GBPS = 3.0  # Large block sequential
    TRAINING_WRITE_GBPS = 0.5  # Checkpoint bursts
    INFERENCE_READ_GBPS = 0.3  # Small block random
    INFERENCE_WRITE_GBPS = 0.05
    BATCH_READ_GBPS = 2.0  # Mixed block sizes
    BATCH_WRITE_GBPS = 1.0

    def __init__(self, config: SimConfig, rng_seed: int = 42):
        self.config = config
        self._rng = __import__("numpy").random.default_rng(rng_seed + 500)
        # Persistent: cumulative writes per rack (for drive wear)
        self._cumulative_writes_tb: dict[int, float] = {}
        # Storage used per rack (grows slowly)
        self._used_tb: dict[int, float] = {}

    def step(
        self,
        server_gpu_utilisation: dict[str, float],
        running_jobs: list | None = None,
        sim_time: float = 0.0,
        tick_interval_s: float = 60.0,
    ) -> FacilityStorageState:
        """Compute storage I/O state for current tick."""
        facility = self.config.facility

        # Build per-server job type map
        server_job_types: dict[str, str] = {}
        if running_jobs:
            for job in running_jobs:
                for srv in getattr(job, "assigned_servers", []):
                    server_job_types[srv] = getattr(job, "job_type", "batch")

        rack_states: list[RackStorageState] = []
        total_r_iops = 0
        total_w_iops = 0
        total_r_tp = 0.0
        total_w_tp = 0.0
        total_used = 0.0
        total_cap = 0.0
        all_r_lat = []
        all_w_lat = []

        for rack_id in range(facility.num_racks):
            if rack_id not in self._cumulative_writes_tb:
                self._cumulative_writes_tb[rack_id] = 0.0
                self._used_tb[rack_id] = self._rng.uniform(5.0, 15.0)  # Pre-populated

            rack_r_iops = 0
            rack_w_iops = 0
            rack_r_tp = 0.0
            rack_w_tp = 0.0

            for srv_idx in range(facility.servers_per_rack):
                server_id = f"rack-{rack_id}-srv-{srv_idx}"
                util = server_gpu_utilisation.get(server_id, 0.0)
                job_type = server_job_types.get(server_id, "idle")

                if util < 0.01 or job_type == "idle":
                    # Idle: minimal background I/O
                    rack_r_iops += 100
                    rack_w_iops += 10
                    rack_r_tp += 0.01
                    rack_w_tp += 0.001
                    continue

                noise = 1.0 + self._rng.normal(0, 0.05)

                if job_type == "training":
                    rack_r_iops += int(self.TRAINING_READ_IOPS * util * noise)
                    rack_w_iops += int(self.TRAINING_WRITE_IOPS * util * noise)
                    rack_r_tp += self.TRAINING_READ_GBPS * util * noise
                    rack_w_tp += self.TRAINING_WRITE_GBPS * util * noise
                elif job_type == "inference":
                    rack_r_iops += int(self.INFERENCE_READ_IOPS * util * noise)
                    rack_w_iops += int(self.INFERENCE_WRITE_IOPS * util * noise)
                    rack_r_tp += self.INFERENCE_READ_GBPS * util * noise
                    rack_w_tp += self.INFERENCE_WRITE_GBPS * util * noise
                else:  # batch
                    rack_r_iops += int(self.BATCH_READ_IOPS * util * noise)
                    rack_w_iops += int(self.BATCH_WRITE_IOPS * util * noise)
                    rack_r_tp += self.BATCH_READ_GBPS * util * noise
                    rack_w_tp += self.BATCH_WRITE_GBPS * util * noise

            # Cap at shelf limits
            rack_total_iops = min(self.MAX_IOPS, rack_r_iops + rack_w_iops)
            if rack_r_iops + rack_w_iops > 0:
                r_frac = rack_r_iops / (rack_r_iops + rack_w_iops)
            else:
                r_frac = 0.5
            rack_r_iops = int(rack_total_iops * r_frac)
            rack_w_iops = rack_total_iops - rack_r_iops

            rack_total_tp = rack_r_tp + rack_w_tp
            if rack_total_tp > self.MAX_THROUGHPUT_GBPS:
                scale = self.MAX_THROUGHPUT_GBPS / rack_total_tp
                rack_r_tp *= scale
                rack_w_tp *= scale

            # Queue depth estimation (Little's Law: QD = λ * W)
            iops_rate = rack_total_iops
            qd = max(1, int(iops_rate * self.BASE_READ_LATENCY_US / 1_000_000))
            qd = min(1024, qd)  # Cap at NVMe queue depth limit

            # Latency model (degrades with queue depth)
            # NVMe latency roughly: base + k * ln(queue_depth)
            qd_factor = 1.0 + 0.3 * math.log(max(1, qd))
            iops_pressure = min(1.0, rack_total_iops / self.MAX_IOPS)
            congestion_factor = 1.0 / (1.0 - min(0.95, iops_pressure * 0.9))

            r_lat = self.BASE_READ_LATENCY_US * qd_factor * congestion_factor
            w_lat = self.BASE_WRITE_LATENCY_US * qd_factor * congestion_factor
            p99_r_lat = r_lat * 2.5  # P99 is ~2.5x average for NVMe

            # Track cumulative writes for drive wear
            writes_this_tick_tb = (rack_w_tp * tick_interval_s) / (8 * 1000)  # Gbps * s → TB
            self._cumulative_writes_tb[rack_id] += writes_this_tick_tb

            # Drive health degrades with writes (100 PB endurance per rack)
            endurance_pb = 100.0
            cumulative_pb = self._cumulative_writes_tb[rack_id] / 1000.0
            drive_health = max(0.0, 100.0 * (1.0 - cumulative_pb / endurance_pb))

            # Storage used grows slowly with write activity
            self._used_tb[rack_id] = min(
                self.CAPACITY_PER_RACK_TB * 0.95,
                self._used_tb[rack_id] + writes_this_tick_tb * 0.001)  # Only 0.1% is new data

            used_tb = self._used_tb[rack_id]
            utilisation = (used_tb / self.CAPACITY_PER_RACK_TB) * 100.0

            rack_states.append(RackStorageState(
                rack_id=rack_id,
                read_iops=rack_r_iops,
                write_iops=rack_w_iops,
                total_iops=rack_total_iops,
                max_iops=self.MAX_IOPS,
                read_throughput_gbps=round(rack_r_tp, 2),
                write_throughput_gbps=round(rack_w_tp, 2),
                max_throughput_gbps=self.MAX_THROUGHPUT_GBPS,
                avg_read_latency_us=round(r_lat, 1),
                avg_write_latency_us=round(w_lat, 1),
                p99_read_latency_us=round(p99_r_lat, 1),
                used_tb=round(used_tb, 2),
                total_tb=self.CAPACITY_PER_RACK_TB,
                utilisation_pct=round(utilisation, 1),
                drive_health_pct=round(drive_health, 1),
                queue_depth=qd,
            ))

            total_r_iops += rack_r_iops
            total_w_iops += rack_w_iops
            total_r_tp += rack_r_tp
            total_w_tp += rack_w_tp
            total_used += used_tb
            total_cap += self.CAPACITY_PER_RACK_TB
            all_r_lat.append(r_lat)
            all_w_lat.append(w_lat)

        return FacilityStorageState(
            racks=rack_states,
            total_read_iops=total_r_iops,
            total_write_iops=total_w_iops,
            total_read_throughput_gbps=round(total_r_tp, 2),
            total_write_throughput_gbps=round(total_w_tp, 2),
            total_used_tb=round(total_used, 2),
            total_capacity_tb=round(total_cap, 2),
            avg_read_latency_us=round(sum(all_r_lat) / max(1, len(all_r_lat)), 1),
            avg_write_latency_us=round(sum(all_w_lat) / max(1, len(all_w_lat)), 1),
        )

    def reset(self) -> None:
        """Clear persistent state."""
        self._cumulative_writes_tb.clear()
        self._used_tb.clear()
