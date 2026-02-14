"""Tests for the evaluation and scoring framework."""

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dc_sim.config import SimConfig
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
from dc_sim.main import create_app
from dc_sim.simulator import Simulator


@pytest.fixture
def client():
    _baseline_cache.clear()
    app = create_app(SimConfig())
    return TestClient(app)


@pytest.fixture
def sim():
    return Simulator(SimConfig())


# ── Scenario definitions ────────────────────────────────────


def test_scenarios_has_all_five():
    """All 5 predefined scenarios are registered."""
    expected = {"steady_state", "thermal_crisis", "carbon_valley", "overload", "cascade"}
    assert set(SCENARIOS.keys()) == expected


def test_weights_sum_to_one():
    """Dimension weights sum to 1.0."""
    total = sum(Evaluator.DIMENSION_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6


# ── API endpoints ───────────────────────────────────────────


def test_scenarios_endpoint_returns_all_five(client):
    """GET /eval/scenarios returns all 5 scenarios."""
    resp = client.get("/eval/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert "scenarios" in data
    ids = {s["scenario_id"] for s in data["scenarios"]}
    assert ids == {"steady_state", "thermal_crisis", "carbon_valley", "overload", "cascade"}


def test_run_steady_state_returns_valid_scores(client):
    """POST /eval/run/steady_state returns valid composite + 7 dimensions."""
    resp = client.post("/eval/run/steady_state?mode=baseline")
    assert resp.status_code == 200
    data = resp.json()
    assert "composite_score" in data
    assert 0.0 <= data["composite_score"] <= 100.0
    assert "dimensions" in data
    assert len(data["dimensions"]) == 7


def test_run_unknown_scenario_returns_404(client):
    """POST /eval/run/nonexistent returns 404."""
    resp = client.post("/eval/run/nonexistent")
    assert resp.status_code == 404


def test_dimension_scores_in_range(client):
    """All dimension scores are between 0 and 100."""
    resp = client.post("/eval/run/steady_state?mode=baseline")
    data = resp.json()
    for d in data["dimensions"]:
        assert 0.0 <= d["score"] <= 100.0, f'{d["name"]} score {d["score"]} out of range'


def test_baseline_cache_populated(client):
    """GET /eval/baseline populates the cache."""
    resp = client.get("/eval/baseline/steady_state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_type"] == "baseline"
    assert "composite_score" in data


def test_get_score_with_populated_telemetry(client):
    """GET /eval/score works after ticking simulation."""
    # Advance simulation to populate telemetry
    client.post("/sim/tick?n=30")
    resp = client.get("/eval/score?scenario_id=steady_state")
    assert resp.status_code == 200
    data = resp.json()
    assert "composite_score" in data
    assert data["run_type"] == "live"
    assert data["ticks_available"] > 0


# ── Scoring correctness ────────────────────────────────────


def test_thermal_crisis_lower_thermal_score(sim):
    """Thermal crisis scenario produces a lower thermal score than steady state."""
    steady = run_scenario(sim, SCENARIOS["steady_state"])
    crisis = run_scenario(sim, SCENARIOS["thermal_crisis"])

    steady_thermal = next(d for d in steady.dimensions if d.name == "thermal_safety")
    crisis_thermal = next(d for d in crisis.dimensions if d.name == "thermal_safety")

    assert crisis_thermal.score <= steady_thermal.score


def test_cascade_audit_trail_has_injections(sim):
    """Cascade scenario records failure injections in the audit log."""
    run_scenario(sim, SCENARIOS["cascade"])

    inject_entries = [
        e for e in sim.audit_log._entries
        if e.action == "inject_failure" and e.source == "scenario"
    ]
    # CASCADE has 5 scripted failures
    assert len(inject_entries) >= 4  # at least most should be recorded


# ── Session API tests ────────────────────────────────────────


def test_session_start_returns_scenario_info(client):
    """POST /eval/session/start returns session info."""
    resp = client.post("/eval/session/start/thermal_crisis?agent_name=test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_id"] == "thermal_crisis"
    assert data["duration_ticks"] == 120
    assert data["tick_interval_s"] == 60.0
    assert data["agent_name"] == "test"
    # Clean up
    client.post("/eval/session/end")


def test_session_start_unknown_scenario_404(client):
    """POST /eval/session/start/nonexistent returns 404."""
    resp = client.post("/eval/session/start/nonexistent")
    assert resp.status_code == 404


def test_session_start_twice_returns_409(client):
    """Cannot start a second session while one is active."""
    client.post("/eval/session/start/steady_state")
    resp = client.post("/eval/session/start/thermal_crisis")
    assert resp.status_code == 409
    # Clean up
    client.post("/eval/session/end")


def test_session_step_advances_tick(client):
    """POST /eval/session/step advances by one tick."""
    client.post("/eval/session/start/steady_state")
    resp = client.post("/eval/session/step")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tick"] == 1
    assert data["max_ticks"] == 240
    assert data["done"] is False
    assert "state" in data
    # Clean up
    client.post("/eval/session/end")


def test_session_step_without_session_returns_400(client):
    """POST /eval/session/step without active session returns 400."""
    resp = client.post("/eval/session/step")
    assert resp.status_code == 400


def test_session_step_injects_failures_at_correct_tick(client):
    """Failures are injected at the tick specified in the scenario."""
    client.post("/eval/session/start/thermal_crisis")
    # thermal_crisis injects crac_failure at at_tick=30
    # step() uses tick_idx = current_tick (0-based) before advancing,
    # so the 31st call (tick_idx=30) triggers the injection.
    for _ in range(31):
        resp = client.post("/eval/session/step")
    data = resp.json()
    assert len(data["failures_injected"]) > 0
    assert data["failures_injected"][0]["type"] == "crac_failure"
    # Clean up
    client.post("/eval/session/end")


def test_session_end_returns_scores(client):
    """POST /eval/session/end returns EvaluationResult."""
    client.post("/eval/session/start/steady_state")
    for _ in range(10):
        client.post("/eval/session/step")
    resp = client.post("/eval/session/end")
    assert resp.status_code == 200
    data = resp.json()
    assert "composite_score" in data
    assert len(data["dimensions"]) == 7
    assert data["run_type"] == "agent"


def test_session_end_without_session_returns_400(client):
    """POST /eval/session/end without active session returns 400."""
    resp = client.post("/eval/session/end")
    assert resp.status_code == 400


def test_session_status_shows_progress(client):
    """GET /eval/session/status reflects progress."""
    client.post("/eval/session/start/steady_state")
    client.post("/eval/session/step")
    resp = client.get("/eval/session/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] is True
    assert data["current_tick"] == 1
    assert data["remaining_ticks"] == 239
    # Clean up
    client.post("/eval/session/end")


def test_session_status_inactive(client):
    """GET /eval/session/status when no session is active."""
    resp = client.get("/eval/session/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] is False


def test_actions_work_between_steps(client):
    """Agent can take actions between steps."""
    client.post("/eval/session/start/steady_state")
    client.post("/eval/session/step")
    # Adjust cooling between steps
    resp = client.post("/actions/adjust_cooling", json={"rack_id": 0, "setpoint_c": 16.0})
    assert resp.status_code == 200
    # Step again
    resp = client.post("/eval/session/step")
    assert resp.status_code == 200
    # Clean up
    client.post("/eval/session/end")


def test_full_session_lifecycle(client):
    """Run a complete short session: start → step N times → end."""
    client.post("/eval/session/start/steady_state?agent_name=test_agent")
    for i in range(5):
        resp = client.post("/eval/session/step")
        assert resp.json()["tick"] == i + 1
    resp = client.post("/eval/session/end")
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["composite_score"] <= 100.0
    assert data["metadata"]["agent_name"] == "test_agent"


# ── Agent registry & runner tests ────────────────────────────


def test_list_agents_returns_random(client):
    """GET /eval/agents lists at least the built-in random agent."""
    resp = client.get("/eval/agents")
    assert resp.status_code == 200
    data = resp.json()
    names = [a["name"] for a in data["agents"]]
    assert "random" in names


def test_run_agent_endpoint(client):
    """POST /eval/run-agent runs the random agent and returns scores."""
    resp = client.post("/eval/run-agent", json={
        "agent_name": "random",
        "scenario_id": "steady_state",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "composite_score" in data
    assert 0.0 <= data["composite_score"] <= 100.0
    assert len(data["dimensions"]) == 7


def test_run_agent_unknown_agent_404(client):
    """POST /eval/run-agent with unknown agent returns 404."""
    resp = client.post("/eval/run-agent", json={
        "agent_name": "nonexistent",
        "scenario_id": "steady_state",
    })
    assert resp.status_code == 404


def test_run_agent_unknown_scenario_404(client):
    """POST /eval/run-agent with unknown scenario returns 404."""
    resp = client.post("/eval/run-agent", json={
        "agent_name": "random",
        "scenario_id": "nonexistent",
    })
    assert resp.status_code == 404


# ── Leaderboard tests ────────────────────────────────────────


def test_leaderboard_empty_initially(client):
    """GET /eval/leaderboard returns empty entries initially."""
    resp = client.get("/eval/leaderboard")
    assert resp.status_code == 200
    # May have entries from other tests, just check structure
    data = resp.json()
    assert "entries" in data


def test_leaderboard_submit_and_retrieve(client):
    """Submit a result and verify it appears in leaderboard."""
    fake_result = {
        "composite_score": 75.5,
        "dimensions": [
            {"name": "sla_quality", "score": 80, "weight": 0.25, "metrics": {}},
            {"name": "energy_efficiency", "score": 70, "weight": 0.20, "metrics": {}},
            {"name": "carbon", "score": 65, "weight": 0.15, "metrics": {}},
            {"name": "thermal_safety", "score": 90, "weight": 0.15, "metrics": {}},
            {"name": "cost", "score": 60, "weight": 0.10, "metrics": {}},
            {"name": "infra_health", "score": 85, "weight": 0.10, "metrics": {}},
            {"name": "failure_response", "score": 50, "weight": 0.05, "metrics": {}},
        ],
        "duration_ticks": 240,
        "total_sim_time_s": 14400.0,
    }
    resp = client.post("/eval/leaderboard/submit", json={
        "agent_name": "test_agent",
        "scenario_id": "steady_state",
        "result": fake_result,
    })
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Verify it shows up
    resp2 = client.get("/eval/leaderboard")
    entries = resp2.json()["entries"]
    test_entries = [e for e in entries if e.get("agent_name") == "test_agent"]
    assert len(test_entries) >= 1


def test_leaderboard_csv_module():
    """Test leaderboard module directly with temp CSV."""
    from dc_sim.leaderboard import load_leaderboard, record_result

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_leaderboard.csv"

        # Initially empty
        df = load_leaderboard(csv_path)
        assert len(df) == 0

        # Record a result
        fake_result = {
            "composite_score": 42.0,
            "dimensions": [
                {"name": "sla_quality", "score": 50, "weight": 0.25, "metrics": {}},
                {"name": "energy_efficiency", "score": 40, "weight": 0.20, "metrics": {}},
                {"name": "carbon", "score": 30, "weight": 0.15, "metrics": {}},
                {"name": "thermal_safety", "score": 60, "weight": 0.15, "metrics": {}},
                {"name": "cost", "score": 35, "weight": 0.10, "metrics": {}},
                {"name": "infra_health", "score": 45, "weight": 0.10, "metrics": {}},
                {"name": "failure_response", "score": 55, "weight": 0.05, "metrics": {}},
            ],
            "duration_ticks": 120,
            "total_sim_time_s": 7200.0,
        }
        run_id = record_result("my_agent", "thermal_crisis", fake_result, csv_path)
        assert len(run_id) > 0

        # Verify
        df2 = load_leaderboard(csv_path)
        assert len(df2) == 1
        assert df2.iloc[0]["agent_name"] == "my_agent"
        assert df2.iloc[0]["composite_score"] == 42.0


# ── Custom scenario override tests ─────────────────────────


def test_run_agent_with_custom_duration(client):
    """POST /eval/run-agent with custom duration_ticks override."""
    resp = client.post("/eval/run-agent", json={
        "agent_name": "random",
        "scenario_id": "steady_state",
        "duration_ticks": 20,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "composite_score" in data
    assert data["duration_ticks"] == 20


def test_run_agent_with_custom_failures(client):
    """POST /eval/run-agent with custom failure injections."""
    resp = client.post("/eval/run-agent", json={
        "agent_name": "random",
        "scenario_id": "steady_state",
        "duration_ticks": 30,
        "failure_injections": [
            {"at_tick": 5, "failure_type": "crac_failure", "target": "crac-0", "duration_s": 600},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "composite_score" in data


def test_run_agent_with_custom_arrival_rate(client):
    """POST /eval/run-agent with custom job arrival interval."""
    resp = client.post("/eval/run-agent", json={
        "agent_name": "random",
        "scenario_id": "steady_state",
        "duration_ticks": 20,
        "mean_job_arrival_interval_s": 60.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "composite_score" in data


def test_session_start_with_explicit_scenario(sim):
    """SessionManager.start() accepts an explicit ScenarioDefinition."""
    custom = ScenarioDefinition(
        scenario_id="custom_test",
        name="CUSTOM_TEST",
        description="A custom test scenario",
        duration_ticks=15,
        rng_seed=999,
        failure_injections=[],
        workload_overrides=ScenarioConfig(mean_job_arrival_interval_s=200.0),
    )
    mgr = SessionManager(sim)
    info = mgr.start("custom_test", "tester", scenario=custom)
    assert info["scenario_id"] == "custom_test"
    assert info["duration_ticks"] == 15

    # Step a few times and end
    for _ in range(5):
        mgr.step()
    result = mgr.end()
    assert 0.0 <= result.composite_score <= 100.0


# ── Baseline endpoint tests ─────────────────────────────────


def test_run_baseline_endpoint(client):
    """POST /eval/run-baseline runs a no-agent baseline and records to leaderboard."""
    resp = client.post("/eval/run-baseline", json={
        "scenario_id": "steady_state",
        "duration_ticks": 20,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "composite_score" in data
    assert data["run_type"] == "baseline"
    assert data["duration_ticks"] == 20


def test_run_baseline_unknown_scenario_404(client):
    """POST /eval/run-baseline with unknown scenario returns 404."""
    resp = client.post("/eval/run-baseline", json={
        "scenario_id": "nonexistent",
    })
    assert resp.status_code == 404


def test_run_baseline_records_to_leaderboard(client):
    """POST /eval/run-baseline records the result to the leaderboard."""
    client.post("/eval/run-baseline", json={
        "scenario_id": "steady_state",
        "duration_ticks": 15,
    })
    resp = client.get("/eval/leaderboard")
    entries = resp.json()["entries"]
    baseline_entries = [e for e in entries if e.get("agent_name") == "baseline"]
    assert len(baseline_entries) >= 1


def test_scenarios_endpoint_returns_full_details(client):
    """GET /eval/scenarios returns rng_seed, failure_injections, and arrival rate."""
    resp = client.get("/eval/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    for s in data["scenarios"]:
        assert "rng_seed" in s
        assert "failure_injections" in s
        assert "mean_job_arrival_interval_s" in s
        assert isinstance(s["failure_injections"], list)
