"""Evaluation API endpoints for benchmarking agent performance."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dc_sim.evaluation import (
    SCENARIOS,
    Evaluator,
    EvaluationResult,
    FailureInjection,
    ScenarioConfig,
    ScenarioDefinition,
    SessionManager,
    _baseline_cache,
    run_scenario,
)

eval_router = APIRouter(prefix="/eval", tags=["evaluation"])

_simulator: Any = None
_session_manager: SessionManager | None = None


def set_eval_simulator(sim: Any) -> None:
    """Inject the simulator instance."""
    global _simulator, _session_manager
    _simulator = sim
    _session_manager = SessionManager(sim)


def _get_sim() -> Any:
    if _simulator is None:
        raise HTTPException(500, "Simulator not initialised")
    return _simulator


def _get_session_mgr() -> SessionManager:
    if _session_manager is None:
        raise HTTPException(500, "Session manager not initialised")
    return _session_manager


# ── Scenario & scoring endpoints ─────────────────────────────


@eval_router.get("/scenarios")
def list_scenarios() -> dict:
    """List all available evaluation scenarios with full details."""
    return {
        "scenarios": [
            {
                "scenario_id": s.scenario_id,
                "name": s.name,
                "description": s.description,
                "duration_ticks": s.duration_ticks,
                "duration_hours": s.duration_ticks * 60 / 3600,
                "rng_seed": s.rng_seed,
                "failure_count": len(s.failure_injections),
                "failure_injections": [fi.model_dump() for fi in s.failure_injections],
                "mean_job_arrival_interval_s": s.workload_overrides.mean_job_arrival_interval_s,
            }
            for s in SCENARIOS.values()
        ]
    }


@eval_router.post("/run/{scenario_id}")
def run_eval(scenario_id: str, mode: str = "agent") -> dict:
    """Run an evaluation scenario to completion.

    Query params:
        mode: "agent" (default) or "baseline"
    """
    if scenario_id not in SCENARIOS:
        raise HTTPException(404, f"Unknown scenario: {scenario_id}")

    sim = _get_sim()
    scenario = SCENARIOS[scenario_id]

    # For baseline mode, no agent callback
    is_baseline = mode == "baseline"
    result = run_scenario(sim, scenario, agent_callback=None)
    result.run_type = "baseline" if is_baseline else "agent"

    # Cache baseline results
    if is_baseline:
        _baseline_cache[scenario_id] = result

    return result.to_dict()


@eval_router.get("/score")
def get_live_score(scenario_id: str = "steady_state") -> dict:
    """Score the current live telemetry without resetting.

    Uses the specified scenario definition for normalisation references.
    """
    if scenario_id not in SCENARIOS:
        raise HTTPException(404, f"Unknown scenario: {scenario_id}")

    sim = _get_sim()
    scenario = SCENARIOS[scenario_id]

    evaluator = Evaluator(sim, scenario)
    result = evaluator.compute()
    result.run_type = "live"

    resp = result.to_dict()
    resp["ticks_available"] = len(list(sim.telemetry._buffer))
    resp["note"] = "Scored from live telemetry; no scenario was run"
    return resp


@eval_router.get("/baseline/{scenario_id}")
def get_baseline(scenario_id: str) -> dict:
    """Get or compute a baseline (no-agent) score for a scenario."""
    if scenario_id not in SCENARIOS:
        raise HTTPException(404, f"Unknown scenario: {scenario_id}")

    # Return cached if available
    if scenario_id in _baseline_cache:
        return _baseline_cache[scenario_id].to_dict()

    # Compute baseline
    sim = _get_sim()
    scenario = SCENARIOS[scenario_id]
    result = run_scenario(sim, scenario, agent_callback=None)
    result.run_type = "baseline"
    _baseline_cache[scenario_id] = result
    return result.to_dict()


# ── Session endpoints (step-by-step control) ──────────────────


@eval_router.post("/session/start/{scenario_id}")
def session_start(scenario_id: str, agent_name: str = "unnamed") -> dict:
    """Start a step-by-step evaluation session."""
    mgr = _get_session_mgr()
    try:
        info = mgr.start(scenario_id, agent_name)
    except ValueError as e:
        msg = str(e)
        if "already active" in msg:
            raise HTTPException(409, msg)
        elif "Unknown scenario" in msg:
            raise HTTPException(404, msg)
        else:
            raise HTTPException(400, msg)
    return info


@eval_router.post("/session/step")
def session_step() -> dict:
    """Advance the evaluation session by one tick."""
    mgr = _get_session_mgr()
    try:
        return mgr.step()
    except ValueError as e:
        raise HTTPException(400, str(e))


@eval_router.post("/session/end")
def session_end() -> dict:
    """End the session, compute scores, and return results."""
    mgr = _get_session_mgr()
    try:
        result = mgr.end()
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result.to_dict()


@eval_router.get("/session/status")
def session_status() -> dict:
    """Get current session status."""
    mgr = _get_session_mgr()
    return mgr.get_status()


# ── Agent registry endpoints ─────────────────────────────────


@eval_router.get("/agents")
def list_agents() -> dict:
    """List all registered agent names."""
    from agents import AGENT_REGISTRY

    return {
        "agents": [
            {"name": name, "class": type(agent).__name__}
            for name, agent in AGENT_REGISTRY.items()
        ]
    }


class FailureInjectionRequest(BaseModel):
    at_tick: int
    failure_type: str
    target: str
    duration_s: int | None = None


class RunAgentRequest(BaseModel):
    agent_name: str
    scenario_id: str
    # Optional overrides to customise the scenario
    duration_ticks: int | None = None
    rng_seed: int | None = None
    mean_job_arrival_interval_s: float | None = None
    failure_injections: list[FailureInjectionRequest] | None = None


def _build_scenario_override(
    base_scenario_id: str,
    duration_ticks: int | None = None,
    rng_seed: int | None = None,
    mean_job_arrival_interval_s: float | None = None,
    failure_injections: list[FailureInjectionRequest] | None = None,
) -> ScenarioDefinition | None:
    """Build a modified ScenarioDefinition from a base scenario + overrides.

    Returns None if no overrides are provided (use the base as-is).
    """
    has_overrides = any(
        v is not None
        for v in [duration_ticks, rng_seed, mean_job_arrival_interval_s, failure_injections]
    )
    if not has_overrides:
        return None

    base = SCENARIOS[base_scenario_id]
    return ScenarioDefinition(
        scenario_id=base.scenario_id,
        name=base.name,
        description=base.description + " (custom overrides)",
        duration_ticks=duration_ticks if duration_ticks is not None else base.duration_ticks,
        rng_seed=rng_seed if rng_seed is not None else base.rng_seed,
        failure_injections=[
            FailureInjection(**fi.model_dump()) for fi in failure_injections
        ]
        if failure_injections is not None
        else list(base.failure_injections),
        workload_overrides=ScenarioConfig(
            mean_job_arrival_interval_s=(
                mean_job_arrival_interval_s
                if mean_job_arrival_interval_s is not None
                else base.workload_overrides.mean_job_arrival_interval_s
            )
        ),
    )


@eval_router.post("/run-agent")
def run_agent(req: RunAgentRequest) -> dict:
    """Run a registered agent against a scenario.

    Executes the full scenario in-process: start → step loop → end.
    Records the result to the leaderboard CSV.
    Supports optional scenario overrides for custom parameters.
    """
    from agents import AGENT_REGISTRY
    from dc_sim.runner import AgentRunner

    if req.agent_name not in AGENT_REGISTRY:
        raise HTTPException(404, f"Unknown agent: {req.agent_name}")
    if req.scenario_id not in SCENARIOS:
        raise HTTPException(404, f"Unknown scenario: {req.scenario_id}")

    sim = _get_sim()
    agent = AGENT_REGISTRY[req.agent_name]
    runner = AgentRunner(agent, sim)

    scenario_override = _build_scenario_override(
        req.scenario_id,
        req.duration_ticks,
        req.rng_seed,
        req.mean_job_arrival_interval_s,
        req.failure_injections,
    )

    result = runner.run(req.scenario_id, record=True, scenario_override=scenario_override)
    return result


# ── Baseline endpoint ────────────────────────────────────────


class RunBaselineRequest(BaseModel):
    scenario_id: str
    # Optional overrides (same as RunAgentRequest)
    duration_ticks: int | None = None
    rng_seed: int | None = None
    mean_job_arrival_interval_s: float | None = None
    failure_injections: list[FailureInjectionRequest] | None = None


@eval_router.post("/run-baseline")
def run_baseline_endpoint(req: RunBaselineRequest) -> dict:
    """Run a scenario with no agent (baseline) and record to the leaderboard.

    This is the 'no-agent' comparison run. The result is recorded to the
    leaderboard with agent_name="baseline" so it appears alongside agent runs.
    """
    from dc_sim.leaderboard import record_result

    if req.scenario_id not in SCENARIOS:
        raise HTTPException(404, f"Unknown scenario: {req.scenario_id}")

    sim = _get_sim()
    base_scenario = SCENARIOS[req.scenario_id]

    scenario_override = _build_scenario_override(
        req.scenario_id,
        req.duration_ticks,
        req.rng_seed,
        req.mean_job_arrival_interval_s,
        req.failure_injections,
    )
    scenario = scenario_override if scenario_override else base_scenario

    result = run_scenario(sim, scenario, agent_callback=None)
    result.run_type = "baseline"

    # Cache for EVAL tab baseline comparison
    _baseline_cache[req.scenario_id] = result

    result_dict = result.to_dict()

    # Record to leaderboard
    record_result("baseline", req.scenario_id, result_dict)

    return result_dict


# ── Leaderboard endpoints ────────────────────────────────────


@eval_router.get("/leaderboard")
def get_leaderboard() -> dict:
    """Return the full leaderboard as JSON."""
    from dc_sim.leaderboard import load_leaderboard

    df = load_leaderboard()
    return {"entries": df.to_dict(orient="records")}


class SubmitResultRequest(BaseModel):
    agent_name: str
    scenario_id: str
    result: dict


@eval_router.post("/leaderboard/submit")
def submit_result(req: SubmitResultRequest) -> dict:
    """Submit a result to the leaderboard CSV."""
    from dc_sim.leaderboard import record_result

    run_id = record_result(req.agent_name, req.scenario_id, req.result)
    return {"ok": True, "run_id": run_id}
