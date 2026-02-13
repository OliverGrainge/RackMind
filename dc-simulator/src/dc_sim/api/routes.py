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


# ── GPU endpoints ──────────────────────────────────────────


@router.get("/gpu")
def get_gpu_summary() -> dict:
    """Facility-wide GPU summary."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    g = state.gpu
    return {
        "total_gpus": g.total_gpus,
        "healthy_gpus": g.healthy_gpus,
        "throttled_gpus": g.throttled_gpus,
        "ecc_error_gpus": g.ecc_error_gpus,
        "avg_gpu_temp_c": g.avg_gpu_temp_c,
        "avg_sm_util_pct": g.avg_sm_util_pct,
        "total_gpu_mem_used_mib": g.total_gpu_mem_used_mib,
        "total_gpu_mem_total_mib": g.total_gpu_mem_total_mib,
    }


@router.get("/gpu/{server_id}")
def get_gpu_server(server_id: str) -> dict:
    """Per-GPU telemetry for a specific server."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    for srv in state.gpu.servers:
        if srv.server_id == server_id:
            return {
                "server_id": srv.server_id,
                "rack_id": srv.rack_id,
                "total_gpu_power_w": srv.total_gpu_power_w,
                "avg_gpu_temp_c": srv.avg_gpu_temp_c,
                "total_mem_used_mib": srv.total_mem_used_mib,
                "total_mem_total_mib": srv.total_mem_total_mib,
                "gpus": [
                    {
                        "gpu_id": gpu.gpu_id,
                        "sm_utilisation_pct": gpu.sm_utilisation_pct,
                        "mem_utilisation_pct": gpu.mem_utilisation_pct,
                        "gpu_temp_c": gpu.gpu_temp_c,
                        "mem_temp_c": gpu.mem_temp_c,
                        "power_draw_w": gpu.power_draw_w,
                        "sm_clock_mhz": gpu.sm_clock_mhz,
                        "mem_clock_mhz": gpu.mem_clock_mhz,
                        "mem_used_mib": gpu.mem_used_mib,
                        "mem_total_mib": gpu.mem_total_mib,
                        "ecc_sbe_count": gpu.ecc_sbe_count,
                        "ecc_dbe_count": gpu.ecc_dbe_count,
                        "pcie_tx_gbps": gpu.pcie_tx_gbps,
                        "pcie_rx_gbps": gpu.pcie_rx_gbps,
                        "nvlink_tx_gbps": gpu.nvlink_tx_gbps,
                        "nvlink_rx_gbps": gpu.nvlink_rx_gbps,
                        "fan_speed_pct": gpu.fan_speed_pct,
                        "thermal_throttle": gpu.thermal_throttle,
                        "power_throttle": gpu.power_throttle,
                    }
                    for gpu in srv.gpus
                ],
            }
    raise HTTPException(404, f"Server {server_id} not found")


# ── Network endpoints ──────────────────────────────────────


@router.get("/network")
def get_network_summary() -> dict:
    """Facility-wide network summary."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    n = state.network
    return {
        "total_east_west_gbps": n.total_east_west_gbps,
        "total_north_south_gbps": n.total_north_south_gbps,
        "total_rdma_gbps": n.total_rdma_gbps,
        "avg_fabric_latency_us": n.avg_fabric_latency_us,
        "total_packet_loss_pct": n.total_packet_loss_pct,
        "total_crc_errors": n.total_crc_errors,
        "racks": [
            {
                "rack_id": r.rack_id,
                "ingress_gbps": r.ingress_gbps,
                "egress_gbps": r.egress_gbps,
                "intra_rack_gbps": r.intra_rack_gbps,
                "tor_utilisation_pct": r.tor_utilisation_pct,
                "avg_latency_us": r.avg_latency_us,
                "p99_latency_us": r.p99_latency_us,
                "packet_loss_pct": r.packet_loss_pct,
                "rdma_tx_gbps": r.rdma_tx_gbps,
                "rdma_rx_gbps": r.rdma_rx_gbps,
                "active_ports": r.active_ports,
                "total_ports": r.total_ports,
            }
            for r in n.racks
        ],
        "spine_links": [
            {
                "src_rack_id": s.src_rack_id,
                "dst_rack_id": s.dst_rack_id,
                "bandwidth_gbps": s.bandwidth_gbps,
                "utilisation_pct": s.utilisation_pct,
                "latency_us": s.latency_us,
            }
            for s in n.spine_links
        ],
    }


@router.get("/network/{rack_id}")
def get_network_rack(rack_id: int) -> dict:
    """Single rack network state."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    for r in state.network.racks:
        if r.rack_id == rack_id:
            return {
                "rack_id": r.rack_id,
                "ingress_gbps": r.ingress_gbps,
                "egress_gbps": r.egress_gbps,
                "intra_rack_gbps": r.intra_rack_gbps,
                "tor_utilisation_pct": r.tor_utilisation_pct,
                "avg_latency_us": r.avg_latency_us,
                "p99_latency_us": r.p99_latency_us,
                "packet_loss_pct": r.packet_loss_pct,
                "crc_errors": r.crc_errors,
                "rdma_tx_gbps": r.rdma_tx_gbps,
                "rdma_rx_gbps": r.rdma_rx_gbps,
                "active_ports": r.active_ports,
                "total_ports": r.total_ports,
            }
    raise HTTPException(404, f"Rack {rack_id} not found")


# ── Storage endpoints ──────────────────────────────────────


@router.get("/storage")
def get_storage_summary() -> dict:
    """Facility-wide storage summary."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    s = state.storage
    return {
        "total_read_iops": s.total_read_iops,
        "total_write_iops": s.total_write_iops,
        "total_read_throughput_gbps": s.total_read_throughput_gbps,
        "total_write_throughput_gbps": s.total_write_throughput_gbps,
        "total_used_tb": s.total_used_tb,
        "total_capacity_tb": s.total_capacity_tb,
        "avg_read_latency_us": s.avg_read_latency_us,
        "avg_write_latency_us": s.avg_write_latency_us,
        "racks": [
            {
                "rack_id": r.rack_id,
                "read_iops": r.read_iops,
                "write_iops": r.write_iops,
                "total_iops": r.total_iops,
                "max_iops": r.max_iops,
                "read_throughput_gbps": r.read_throughput_gbps,
                "write_throughput_gbps": r.write_throughput_gbps,
                "avg_read_latency_us": r.avg_read_latency_us,
                "avg_write_latency_us": r.avg_write_latency_us,
                "p99_read_latency_us": r.p99_read_latency_us,
                "used_tb": r.used_tb,
                "total_tb": r.total_tb,
                "utilisation_pct": r.utilisation_pct,
                "drive_health_pct": r.drive_health_pct,
                "queue_depth": r.queue_depth,
            }
            for r in s.racks
        ],
    }


@router.get("/storage/{rack_id}")
def get_storage_rack(rack_id: int) -> dict:
    """Single rack storage state."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    for r in state.storage.racks:
        if r.rack_id == rack_id:
            return {
                "rack_id": r.rack_id,
                "read_iops": r.read_iops,
                "write_iops": r.write_iops,
                "total_iops": r.total_iops,
                "read_throughput_gbps": r.read_throughput_gbps,
                "write_throughput_gbps": r.write_throughput_gbps,
                "avg_read_latency_us": r.avg_read_latency_us,
                "avg_write_latency_us": r.avg_write_latency_us,
                "p99_read_latency_us": r.p99_read_latency_us,
                "used_tb": r.used_tb,
                "total_tb": r.total_tb,
                "drive_health_pct": r.drive_health_pct,
                "queue_depth": r.queue_depth,
            }
    raise HTTPException(404, f"Rack {rack_id} not found")


# ── Cooling endpoints ──────────────────────────────────────


@router.get("/cooling")
def get_cooling() -> dict:
    """Facility cooling system state."""
    sim = get_sim()
    state = sim.telemetry.get_latest()
    if state is None:
        raise HTTPException(404, "No state yet")
    c = state.cooling
    return {
        "total_cooling_output_kw": c.total_cooling_output_kw,
        "total_cooling_capacity_kw": c.total_cooling_capacity_kw,
        "cooling_load_pct": c.cooling_load_pct,
        "cop": c.cop,
        "cooling_power_kw": c.cooling_power_kw,
        "chw_plant_supply_temp_c": c.chw_plant_supply_temp_c,
        "chw_plant_return_temp_c": c.chw_plant_return_temp_c,
        "chw_plant_delta_t_c": c.chw_plant_delta_t_c,
        "pump_power_kw": c.pump_power_kw,
        "pump_flow_rate_lps": c.pump_flow_rate_lps,
        "cooling_tower": {
            "condenser_supply_temp_c": c.cooling_tower.condenser_supply_temp_c,
            "condenser_return_temp_c": c.cooling_tower.condenser_return_temp_c,
            "wet_bulb_temp_c": c.cooling_tower.wet_bulb_temp_c,
            "approach_temp_c": c.cooling_tower.approach_temp_c,
            "fan_speed_pct": c.cooling_tower.fan_speed_pct,
            "heat_rejection_kw": c.cooling_tower.heat_rejection_kw,
        },
        "crac_units": [
            {
                "unit_id": u.unit_id,
                "supply_air_temp_c": u.supply_air_temp_c,
                "return_air_temp_c": u.return_air_temp_c,
                "fan_speed_pct": u.fan_speed_pct,
                "airflow_cfm": u.airflow_cfm,
                "chw_supply_temp_c": u.chw_supply_temp_c,
                "chw_return_temp_c": u.chw_return_temp_c,
                "chw_flow_rate_lps": u.chw_flow_rate_lps,
                "cooling_output_kw": u.cooling_output_kw,
                "cooling_capacity_kw": u.cooling_capacity_kw,
                "load_pct": u.load_pct,
                "operational": u.operational,
                "fault_code": u.fault_code,
            }
            for u in c.crac_units
        ],
    }


# ── Workload endpoints ─────────────────────────────────────


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
