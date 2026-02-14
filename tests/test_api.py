"""Tests for the REST API."""

import pytest
from fastapi.testclient import TestClient

from dc_sim.api.routes import set_simulator
from dc_sim.config import SimConfig
from dc_sim.main import create_app
from dc_sim.simulator import Simulator


@pytest.fixture
def client():
    app = create_app(SimConfig())
    return TestClient(app)


def test_get_status_returns_valid_json(client):
    """GET /status returns valid JSON matching FacilityState schema."""
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "current_time" in data
    assert "tick_count" in data
    assert "thermal" in data
    assert "power" in data
    assert "workload_pending" in data
    assert "workload_running" in data
    assert "workload_completed" in data
    assert "sla_violations" in data
    assert "racks" in data["thermal"]


def test_post_tick_advances_clock(client):
    """POST /sim/tick advances the clock."""
    r1 = client.get("/status")
    t1 = r1.json()["current_time"]

    client.post("/sim/tick?n=10")

    r2 = client.get("/status")
    t2 = r2.json()["current_time"]
    assert t2 > t1


def test_migrate_workload_valid_returns_200(client):
    """POST /actions/migrate_workload with valid job ID returns 200."""
    client.post("/sim/tick?n=30")
    running = client.get("/workload/running").json()["running"]
    if not running:
        pytest.skip("No running jobs - run more ticks or lower arrival interval")

    job_id = running[0]["job_id"]
    resp = client.post("/actions/migrate_workload", json={"job_id": job_id, "target_rack_id": 3})
    assert resp.status_code == 200


def test_migrate_workload_invalid_returns_404(client):
    """POST /actions/migrate_workload with invalid job ID returns 404."""
    resp = client.post(
        "/actions/migrate_workload",
        json={"job_id": "nonexistent-job-id", "target_rack_id": 3},
    )
    assert resp.status_code == 404
