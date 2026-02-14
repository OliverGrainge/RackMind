"""Tests for the workload model."""

import uuid

import numpy as np
import pytest

from dc_sim.config import SimConfig
from dc_sim.models.workload import Job, WorkloadQueue


def test_job_moves_pending_to_running():
    """A submitted job should move from pending to running when capacity is available."""
    config = SimConfig()
    config.workload.mean_job_arrival_interval_s = 1e9  # No random arrivals
    rng = np.random.default_rng(42)
    queue = WorkloadQueue(config, rng=rng)

    job = Job(
        job_id=str(uuid.uuid4()),
        name="test-job",
        gpu_requirement=1,
        priority=5,
        duration_s=1000,
        submitted_at=0,
        sla_deadline_s=600,
    )
    queue.pending.append(job)

    queue.step(0)
    assert job in queue.running
    assert job.started_at == 0
    assert len(job.assigned_servers) >= 1


def test_job_completes_after_duration():
    """A job should move to completed after its duration elapses."""
    config = SimConfig()
    config.workload.mean_job_arrival_interval_s = 1e9
    rng = np.random.default_rng(42)
    queue = WorkloadQueue(config, rng=rng)

    job = Job(
        job_id=str(uuid.uuid4()),
        name="test-job",
        gpu_requirement=1,
        priority=5,
        duration_s=100,
        submitted_at=0,
        sla_deadline_s=600,
    )
    queue.pending.append(job)
    queue.step(0)
    assert job in queue.running

    for t in range(60, 200, 60):
        queue.step(t)

    assert job in queue.completed
    assert job.status == "completed"


def test_sla_violation_flagged():
    """SLA violations should be flagged when queue wait exceeds deadline."""
    config = SimConfig()
    config.workload.mean_job_arrival_interval_s = 1e9
    config.facility.num_racks = 1
    config.facility.servers_per_rack = 1
    config.facility.gpus_per_server = 1
    rng = np.random.default_rng(42)
    queue = WorkloadQueue(config, rng=rng)

    job = Job(
        job_id=str(uuid.uuid4()),
        name="test-job",
        gpu_requirement=1,
        priority=1,
        duration_s=1000,
        submitted_at=0,
        sla_deadline_s=120,
    )
    queue.pending.append(job)

    job2 = Job(
        job_id=str(uuid.uuid4()),
        name="blocker",
        gpu_requirement=1,
        priority=5,
        duration_s=1000,
        submitted_at=0,
        sla_deadline_s=600,
    )
    queue.pending.append(job2)
    queue.pending.sort(key=lambda j: -j.priority)

    queue.step(0)
    queue.step(60)
    queue.step(120)
    queue.step(180)

    violations = queue.get_sla_violations()
    assert any(j.job_id == job.job_id for j in violations)


def test_preempt_frees_slots():
    """Preempting a job should free its GPU slots and mark as preempted."""
    config = SimConfig()
    config.workload.mean_job_arrival_interval_s = 1e9
    rng = np.random.default_rng(42)
    queue = WorkloadQueue(config, rng=rng)

    job = Job(
        job_id=str(uuid.uuid4()),
        name="test-job",
        gpu_requirement=1,
        priority=1,
        duration_s=1000,
        submitted_at=0,
        sla_deadline_s=600,
    )
    queue.pending.append(job)
    queue.step(0)
    assert job in queue.running
    slots_before = sum(1 for s in queue._server_gpus_available().values() if s < config.facility.gpus_per_server)

    ok = queue.preempt_job(job.job_id)
    assert ok
    assert job.status == "preempted"
    assert job in queue.completed
    slots = queue._server_gpus_available()
    total_avail = sum(slots.values())
    assert total_avail == config.facility.num_racks * config.facility.servers_per_rack * config.facility.gpus_per_server
