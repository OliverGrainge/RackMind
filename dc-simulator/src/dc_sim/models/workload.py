"""Workload model: job queue with stochastic arrivals and naive scheduler.

Improvements over v1:
- Three job types: training, inference, batch — each with distinct GPU, duration, priority profiles
- Training: long-running, high GPU, medium priority
- Inference: short, low GPU, high priority (latency-sensitive)
- Batch: medium duration, variable GPU, low priority (cost-sensitive, can be deferred)
"""

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum

from dc_sim.config import SimConfig


class JobType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    BATCH = "batch"


# Profiles per job type: (gpu_range, duration_range_s, priority_range, sla_range_s, gpu_util)
JOB_PROFILES = {
    JobType.TRAINING: {
        "gpu_range": (4, 16),
        "duration_range_s": (3600, 14400),  # 1-4 hours
        "priority_range": (2, 4),
        "sla_range_s": (1800.0, 7200.0),
        "gpu_util": 0.92,  # High sustained utilisation
    },
    JobType.INFERENCE: {
        "gpu_range": (1, 2),
        "duration_range_s": (60, 600),  # 1-10 minutes
        "priority_range": (4, 5),
        "sla_range_s": (30.0, 300.0),  # Tight SLA
        "gpu_util": 0.6,  # Bursty, lower average
    },
    JobType.BATCH: {
        "gpu_range": (2, 8),
        "duration_range_s": (600, 7200),  # 10 min - 2 hours
        "priority_range": (1, 3),
        "sla_range_s": (3600.0, 14400.0),  # Relaxed SLA
        "gpu_util": 0.85,
    },
}

# Arrival weight: inference jobs arrive more frequently than training
JOB_TYPE_WEIGHTS = {
    JobType.TRAINING: 0.2,
    JobType.INFERENCE: 0.5,
    JobType.BATCH: 0.3,
}


@dataclass
class Job:
    """A single workload job."""

    job_id: str
    name: str
    gpu_requirement: int
    priority: int  # 1 (low) to 5 (critical)
    duration_s: int
    submitted_at: float
    started_at: float | None = None
    completed_at: float | None = None
    assigned_servers: list[str] = field(default_factory=list)
    status: str = "queued"  # queued, running, completed, failed, preempted
    sla_deadline_s: float = 600.0
    sla_violated: bool = False
    job_type: str = "batch"
    gpu_util_target: float = 0.9  # Target GPU utilisation when running


class WorkloadQueue:
    """Job queue with pending, running, and completed jobs."""

    def __init__(self, config: SimConfig, rng=None):
        self.config = config
        self.facility = config.facility
        self.workload_cfg = config.workload
        self.rng = rng
        self.pending: list[Job] = []
        self.running: list[Job] = []
        self.completed: list[Job] = []
        self._server_gpu_utilisation: dict[str, float] = {}
        self._init_server_utilisation()

    def _init_server_utilisation(self) -> None:
        """Initialise utilisation map for all servers."""
        for rack_id in range(self.facility.num_racks):
            for srv_idx in range(self.facility.servers_per_rack):
                server_id = f"rack-{rack_id}-srv-{srv_idx}"
                self._server_gpu_utilisation[server_id] = 0.05  # Idle

    def _get_rng(self):
        if self.rng is None:
            import numpy as np
            return np.random.default_rng(self.config.rng_seed)
        return self.rng

    def _pick_job_type(self, rng) -> JobType:
        """Weighted random selection of job type."""
        types = list(JOB_TYPE_WEIGHTS.keys())
        weights = [JOB_TYPE_WEIGHTS[t] for t in types]
        idx = rng.choice(len(types), p=weights)
        return types[idx]

    def _server_gpus_available(self) -> dict[str, int]:
        """Return available GPU slots per server (server_id -> count)."""
        slots: dict[str, int] = {}
        for rack_id in range(self.facility.num_racks):
            for srv_idx in range(self.facility.servers_per_rack):
                server_id = f"rack-{rack_id}-srv-{srv_idx}"
                slots[server_id] = self.facility.gpus_per_server
        for job in self.running:
            for srv in job.assigned_servers:
                slots[srv] -= 1  # Simplified: 1 GPU per server assignment
        return slots

    def _find_placement(self, gpu_req: int) -> list[str] | None:
        """First-fit: find servers with enough GPU slots. Returns server IDs."""
        slots = self._server_gpus_available()
        assigned: list[str] = []
        needed = gpu_req
        for server_id in sorted(slots.keys()):
            if needed <= 0:
                break
            avail = slots[server_id]
            take = min(needed, avail)
            if take > 0:
                for _ in range(take):
                    assigned.append(server_id)
                needed -= take
        return assigned if needed == 0 else None

    def step(self, current_time: float) -> dict[str, float]:
        """
        Advance workload by one tick. Returns server_id -> gpu_utilisation.
        """
        rng = self._get_rng()

        # 1. Arrivals: Poisson process, P(at least 1) = 1 - exp(-lambda * tick)
        rate = 1.0 / self.workload_cfg.mean_job_arrival_interval_s
        tick_s = self.config.clock.tick_interval_s
        prob_arrival = 1 - math.exp(-rate * tick_s) if rate > 0 else 0
        if rng.random() < prob_arrival:
            job_type = self._pick_job_type(rng)
            profile = JOB_PROFILES[job_type]

            gpu_lo, gpu_hi = profile["gpu_range"]
            max_gpus = self.facility.num_racks * self.facility.servers_per_rack * self.facility.gpus_per_server
            gpu_req = int(rng.integers(gpu_lo, min(gpu_hi + 1, max_gpus)))
            gpu_req = max(1, gpu_req)

            dur_lo, dur_hi = profile["duration_range_s"]
            dur = int(rng.integers(dur_lo, dur_hi + 1))

            pri_lo, pri_hi = profile["priority_range"]
            priority = int(rng.integers(pri_lo, pri_hi + 1))

            sla_lo, sla_hi = profile["sla_range_s"]
            sla = float(rng.uniform(sla_lo, sla_hi))

            job_id = str(uuid.uuid4())
            name = f"{job_type.value}-{job_id[:8]}"
            self.pending.append(
                Job(
                    job_id=job_id,
                    name=name,
                    gpu_requirement=gpu_req,
                    priority=priority,
                    duration_s=dur,
                    submitted_at=current_time,
                    sla_deadline_s=sla,
                    status="queued",
                    job_type=job_type.value,
                    gpu_util_target=profile["gpu_util"],
                )
            )

        # 2. SLA check for pending
        for job in self.pending:
            wait = current_time - job.submitted_at
            if wait >= job.sla_deadline_s and not job.sla_violated:
                job.sla_violated = True

        # 3. Scheduling: first-fit by priority
        self.pending.sort(key=lambda j: -j.priority)
        for job in list(self.pending):
            if job.status != "queued":
                continue
            placement = self._find_placement(job.gpu_requirement)
            if placement:
                job.assigned_servers = placement
                job.started_at = current_time
                job.status = "running"
                self.pending.remove(job)
                self.running.append(job)

        # 4. Completion
        for job in list(self.running):
            if job.started_at is None:
                continue
            elapsed = current_time - job.started_at
            if elapsed >= job.duration_s:
                job.completed_at = current_time
                job.status = "completed"
                self.running.remove(job)
                self.completed.append(job)

        # 5. Update GPU utilisation: avg across GPUs on each server
        self._init_server_utilisation()
        gpu_count_per_server: dict[str, int] = {}
        gpu_util_sum_per_server: dict[str, float] = {}
        for rack_id in range(self.facility.num_racks):
            for srv_idx in range(self.facility.servers_per_rack):
                srv = f"rack-{rack_id}-srv-{srv_idx}"
                gpu_count_per_server[srv] = self.facility.gpus_per_server
                gpu_util_sum_per_server[srv] = 0.05 * self.facility.gpus_per_server  # Idle baseline
        for job in self.running:
            util = job.gpu_util_target
            for srv in job.assigned_servers:
                gpu_util_sum_per_server[srv] = gpu_util_sum_per_server.get(srv, 0) - 0.05 + util
        for srv in gpu_count_per_server:
            n = gpu_count_per_server[srv]
            s = gpu_util_sum_per_server.get(srv, 0.05 * n)
            self._server_gpu_utilisation[srv] = min(1.0, s / n)

        return dict(self._server_gpu_utilisation)

    def get_job(self, job_id: str) -> Job | None:
        """Find job by ID in any queue."""
        for job in self.pending + self.running + self.completed:
            if job.job_id == job_id:
                return job
        return None

    def migrate_job(self, job_id: str, target_rack_id: int) -> bool:
        """Move a running job to a different rack. Returns success."""
        job = self.get_job(job_id)
        if not job or job.status != "running":
            return False
        # Find servers in target rack with enough capacity
        target_servers = [f"rack-{target_rack_id}-srv-{i}" for i in range(self.facility.servers_per_rack)]
        slots = self._server_gpus_available()
        for srv in job.assigned_servers:
            slots[srv] += 1  # Free old slots
        assigned = []
        needed = job.gpu_requirement
        for srv in target_servers:
            if needed <= 0:
                break
            avail = slots.get(srv, self.facility.gpus_per_server)
            take = min(needed, avail)
            for _ in range(take):
                assigned.append(srv)
            needed -= take
        if needed == 0:
            job.assigned_servers = assigned
            return True
        return False

    def preempt_job(self, job_id: str, mark_as_failed: bool = False) -> bool:
        """Preempt a running job. Returns success."""
        job = self.get_job(job_id)
        if not job or job.status != "running":
            return False
        job.status = "failed" if mark_as_failed else "preempted"
        self.running.remove(job)
        self.completed.append(job)
        return True

    def get_sla_violations(self) -> list[Job]:
        """Jobs that violated SLA (queued too long)."""
        return [j for j in self.pending + self.completed if j.sla_violated]

    def reset(self) -> None:
        """Reset queue state."""
        self.pending.clear()
        self.running.clear()
        self.completed.clear()
        self._init_server_utilisation()
