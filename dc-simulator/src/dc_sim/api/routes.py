"""REST API routes for the data centre simulator."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dc_sim.telemetry import facility_state_to_dict

router = APIRouter()

# Simulator instance - set by main.py
_simulator: Any = None


def set_simulator(sim: Any) -> None:
    """Inject the simulator instance."""
    global _simulator
    _simulator = sim


def get_sim() -> Any:
    if _simulator is None:
        raise HTTPException(500, "Simulator not initialised")
    return _simulator


# --- Request/Response schemas ---


class MigrateWorkloadRequest(BaseModel):
    job_id: str
    target_rack_id: int


class AdjustCoolingRequest(BaseModel):
    rack_id: int
    setpoint_c: float


class ThrottleGpuRequest(BaseModel):
    server_id: str
    power_cap_pct: float


class PreemptJobRequest(BaseModel):
    job_id: str


class ResolveFailureRequest(BaseModel):
    failure_id: str


class InjectFailureRequest(BaseModel):
    type: str
    target: str
    duration_s: int | None = None


# --- Telemetry endpoints ---


@router.get("/status")
def get_status() -> dict:
    """Full current FacilityState snapshot."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        sim.tick(1)
        state = sim.telemetry.get_latest()
    return facility_state_to_dict(state)


@router.get("/thermal")
def get_thermal() -> dict:
    """All rack thermal states."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet - run a tick")
    return {
        "racks": [
            {
                "rack_id": r.rack_id,
                "inlet_temp_c": r.inlet_temp_c,
                "outlet_temp_c": r.outlet_temp_c,
                "heat_generated_kw": r.heat_generated_kw,
                "throttled": r.throttled,
                "humidity_pct": r.humidity_pct,
                "delta_t_c": r.delta_t_c,
            }
            for r in state.thermal.racks
        ],
        "ambient_temp_c": state.thermal.ambient_temp_c,
        "avg_humidity_pct": state.thermal.avg_humidity_pct,
    }


@router.get("/thermal/{rack_id}")
def get_thermal_rack(rack_id: int) -> dict:
    """Single rack thermal state."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    for r in state.thermal.racks:
        if r.rack_id == rack_id:
            return {
                "rack_id": r.rack_id,
                "inlet_temp_c": r.inlet_temp_c,
                "outlet_temp_c": r.outlet_temp_c,
                "heat_generated_kw": r.heat_generated_kw,
                "throttled": r.throttled,
                "humidity_pct": r.humidity_pct,
                "delta_t_c": r.delta_t_c,
            }
    raise HTTPException(404, f"Rack {rack_id} not found")


@router.get("/power")
def get_power() -> dict:
    """Facility power summary."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    p = state.power
    return {
        "it_power_kw": p.it_power_kw,
        "total_power_kw": p.total_power_kw,
        "pue": p.pue,
        "headroom_kw": p.headroom_kw,
        "power_cap_exceeded": p.power_cap_exceeded,
    }


@router.get("/power/{rack_id}")
def get_power_rack(rack_id: int) -> dict:
    """Single rack power state."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    for r in state.power.racks:
        if r.rack_id == rack_id:
            return {
                "rack_id": r.rack_id,
                "total_power_kw": r.total_power_kw,
                "pdu_utilisation_pct": r.pdu_utilisation_pct,
            }
    raise HTTPException(404, f"Rack {rack_id} not found")


@router.get("/carbon")
def get_carbon() -> dict:
    """Current carbon and cost state."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    c = state.carbon
    return {
        "carbon_intensity_gco2_kwh": c.carbon_intensity_gco2_kwh,
        "carbon_rate_gco2_s": c.carbon_rate_gco2_s,
        "cumulative_carbon_kg": c.cumulative_carbon_kg,
        "electricity_price_gbp_kwh": c.electricity_price_gbp_kwh,
        "cost_rate_gbp_h": c.cost_rate_gbp_h,
        "cumulative_cost_gbp": c.cumulative_cost_gbp,
    }


@router.get("/workload/queue")
def get_workload_queue() -> dict:
    """All pending jobs."""
    sim = get_sim()
    jobs = [
        {
            "job_id": j.job_id,
            "name": j.name,
            "gpu_requirement": j.gpu_requirement,
            "priority": j.priority,
            "status": j.status,
            "job_type": j.job_type,
        }
        for j in sim.workload_queue.pending
    ]
    return {"pending": jobs}


@router.get("/workload/running")
def get_workload_running() -> dict:
    """All running jobs."""
    sim = get_sim()
    jobs = [
        {
            "job_id": j.job_id,
            "name": j.name,
            "gpu_requirement": j.gpu_requirement,
            "assigned_servers": j.assigned_servers,
            "started_at": j.started_at,
            "job_type": j.job_type,
        }
        for j in sim.workload_queue.running
    ]
    return {"running": jobs}


@router.get("/workload/completed")
def get_workload_completed(last_n: int = 10) -> dict:
    """Recent completed jobs."""
    sim = get_sim()
    jobs = sim.workload_queue.completed[-last_n:]
    return {
        "completed": [
            {
                "job_id": j.job_id,
                "name": j.name,
                "status": j.status,
                "completed_at": j.completed_at,
                "job_type": j.job_type,
            }
            for j in jobs
        ]
    }


@router.get("/workload/sla_violations")
def get_sla_violations() -> dict:
    """Jobs that missed SLA."""
    sim = get_sim()
    jobs = sim.workload_queue.get_sla_violations()
    return {
        "sla_violations": [
            {
                "job_id": j.job_id,
                "name": j.name,
                "submitted_at": j.submitted_at,
                "sla_deadline_s": j.sla_deadline_s,
                "job_type": j.job_type,
            }
            for j in jobs
        ]
    }


@router.get("/failures/active")
def get_failures_active() -> dict:
    """Currently active failures."""
    sim = get_sim()
    failures = sim.failure_engine.get_active_failures()
    return {
        "active": [
            {
                "failure_id": f.failure_id,
                "type": f.failure_type,
                "target": f.target,
                "started_at": f.started_at,
                "effect": f.effect,
            }
            for f in failures
        ]
    }


@router.get("/telemetry/history")
def get_telemetry_history(last_n: int = 60) -> dict:
    """Last N ticks of full state."""
    sim = get_sim()
    entries = sim.telemetry.get_last_n(last_n)
    return {
        "history": [
            {"timestamp": t, "state": facility_state_to_dict(s)} for t, s in entries
        ]
    }


@router.get("/audit")
def get_audit_log(last_n: int = 50) -> dict:
    """Recent audit log entries (actions taken on the simulator)."""
    sim = get_sim()
    return {"entries": sim.audit_log.get_last_n(last_n)}


# --- Action endpoints ---


@router.post("/actions/migrate_workload")
def migrate_workload(req: MigrateWorkloadRequest) -> dict:
    """Move a running job to a different rack."""
    sim = get_sim()
    ok = sim.facility.workload_queue.migrate_job(req.job_id, req.target_rack_id)
    result = "ok" if ok else "not_found"
    sim.audit_log.record(
        timestamp=sim.clock.current_time,
        action="migrate_workload",
        params={"job_id": req.job_id, "target_rack_id": req.target_rack_id},
        result=result,
    )
    if not ok:
        raise HTTPException(404, f"Job {req.job_id} not found or not running")
    return {"ok": True, "job_id": req.job_id, "target_rack_id": req.target_rack_id}


@router.post("/actions/adjust_cooling")
def adjust_cooling(req: AdjustCoolingRequest) -> dict:
    """Change CRAC setpoint for a zone (rack)."""
    sim = get_sim()
    sim.facility._crac_setpoints[req.rack_id] = req.setpoint_c
    sim.audit_log.record(
        timestamp=sim.clock.current_time,
        action="adjust_cooling",
        params={"rack_id": req.rack_id, "setpoint_c": req.setpoint_c},
    )
    return {"ok": True, "rack_id": req.rack_id, "setpoint_c": req.setpoint_c}


@router.post("/actions/throttle_gpu")
def throttle_gpu(req: ThrottleGpuRequest) -> dict:
    """Limit GPU power on a server."""
    sim = get_sim()
    sim.facility.set_server_power_cap(req.server_id, req.power_cap_pct)
    sim.audit_log.record(
        timestamp=sim.clock.current_time,
        action="throttle_gpu",
        params={"server_id": req.server_id, "power_cap_pct": req.power_cap_pct},
    )
    return {"ok": True, "server_id": req.server_id, "power_cap_pct": req.power_cap_pct}


@router.post("/actions/preempt_job")
def preempt_job(req: PreemptJobRequest) -> dict:
    """Kill a low-priority job to free resources."""
    sim = get_sim()
    ok = sim.facility.workload_queue.preempt_job(req.job_id)
    result = "ok" if ok else "not_found"
    sim.audit_log.record(
        timestamp=sim.clock.current_time,
        action="preempt_job",
        params={"job_id": req.job_id},
        result=result,
    )
    if not ok:
        raise HTTPException(404, f"Job {req.job_id} not found or not running")
    return {"ok": True, "job_id": req.job_id}


@router.post("/actions/resolve_failure")
def resolve_failure(req: ResolveFailureRequest) -> dict:
    """Simulate repair of a failure."""
    sim = get_sim()
    ok = sim.failure_engine.resolve(req.failure_id)
    result = "ok" if ok else "not_found"
    sim.audit_log.record(
        timestamp=sim.clock.current_time,
        action="resolve_failure",
        params={"failure_id": req.failure_id},
        result=result,
    )
    if not ok:
        raise HTTPException(404, f"Failure {req.failure_id} not found")
    return {"ok": True, "failure_id": req.failure_id}


# --- Simulation control ---


@router.post("/sim/tick")
def sim_tick(n: int = 1) -> dict:
    """Advance simulation by n ticks."""
    sim = get_sim()
    states = sim.tick(n)
    latest = states[-1] if states else None
    return {
        "ticks_advanced": n,
        "current_time": sim.clock.current_time,
        "tick_count": sim.clock.tick_count,
        "elapsed": sim.clock.elapsed_human_readable,
    }


@router.post("/sim/run")
def sim_run(tick_interval_s: float = 0.5) -> dict:
    """Start continuous simulation loop (ticks in background)."""
    sim = get_sim()
    ok = sim.start_continuous(tick_interval_real_s=tick_interval_s)
    return {
        "ok": ok,
        "running": sim.is_running,
        "message": "Started" if ok else "Already running",
    }


@router.post("/sim/pause")
def sim_pause() -> dict:
    """Pause continuous simulation loop."""
    sim = get_sim()
    ok = sim.stop_continuous()
    return {
        "ok": ok,
        "running": sim.is_running,
        "message": "Paused" if ok else "Was not running",
    }


@router.get("/sim/status")
def sim_status() -> dict:
    """Whether continuous simulation is running."""
    sim = get_sim()
    return {"running": sim.is_running, "tick_count": sim.clock.tick_count}


@router.post("/sim/reset")
def sim_reset() -> dict:
    """Reset to initial state."""
    sim = get_sim()
    sim.reset()
    return {"ok": True}


@router.post("/sim/inject_failure")
def sim_inject_failure(req: InjectFailureRequest) -> dict:
    """Manually inject a failure."""
    sim = get_sim()
    sim.failure_engine.set_current_time(sim.clock.current_time)
    failures = sim.failure_engine.inject(req.type, req.target, req.duration_s)
    if not failures:
        raise HTTPException(400, f"Unknown failure type: {req.type}")
    sim.audit_log.record(
        timestamp=sim.clock.current_time,
        action="inject_failure",
        params={"type": req.type, "target": req.target, "duration_s": req.duration_s},
    )
    return {"ok": True, "failure_id": failures[0].failure_id}


@router.get("/sim/config")
def sim_config() -> dict:
    """Return current SimConfig."""
    sim = get_sim()
    c = sim.config
    return {
        "facility": c.facility.model_dump(),
        "thermal": c.thermal.model_dump(),
        "power": c.power.model_dump(),
        "workload": c.workload.model_dump(),
        "clock": c.clock.model_dump(),
    }
