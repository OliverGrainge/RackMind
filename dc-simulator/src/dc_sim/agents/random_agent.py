"""Random agent: takes simple random actions each tick. Useful as a baseline."""

from __future__ import annotations

import random

from dc_sim.agents.base import AgentAction, BaseAgent


class RandomAgent(BaseAgent):
    """Baseline agent that takes random actions.

    Each tick:
    - Resolves any active failures
    - Randomly adjusts cooling for one rack
    - Occasionally preempts lowest-priority job under pressure
    """

    name = "random"

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def act(self, state: dict) -> list[AgentAction]:
        actions: list[AgentAction] = []

        # ── Always resolve active failures ────────────────
        failures = state.get("failures", [])
        for f in failures:
            actions.append(AgentAction(
                action_type="resolve_failure",
                params={"failure_id": f["failure_id"]},
            ))

        # ── Randomly adjust cooling for one rack ─────────
        thermal = state.get("thermal", {})
        racks = thermal.get("racks", [])
        if racks:
            rack = self._rng.choice(racks)
            rack_id = rack["rack_id"]
            temp = rack["inlet_temp_c"]
            # Lower setpoint if hot, raise if cool
            if temp > 33:
                setpoint = self._rng.uniform(14.0, 16.0)
            elif temp < 24:
                setpoint = self._rng.uniform(19.0, 22.0)
            else:
                setpoint = self._rng.uniform(16.0, 20.0)
            actions.append(AgentAction(
                action_type="adjust_cooling",
                params={"rack_id": rack_id, "setpoint_c": round(setpoint, 1)},
            ))

        # ── Preempt lowest-priority job if queue is long ──
        pending = state.get("workload_pending", 0)
        running = state.get("workload_running", 0)
        if pending > 5 and running > 0 and self._rng.random() < 0.3:
            # We need running job IDs — use the jobs from state if available
            running_jobs = state.get("running_jobs", [])
            if running_jobs:
                # Pick lowest priority job
                lowest = min(running_jobs, key=lambda j: j.get("priority", 3))
                actions.append(AgentAction(
                    action_type="preempt_job",
                    params={"job_id": lowest["job_id"]},
                ))

        return actions

    def on_session_start(self, session_info: dict) -> None:
        # Reset RNG for reproducibility
        self._rng = random.Random(42)
