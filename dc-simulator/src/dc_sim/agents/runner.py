"""Agent runner: executes an agent against an evaluation scenario in-process."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dc_sim.agents.base import AgentAction, BaseAgent
from dc_sim.evaluation import SessionManager
from dc_sim.leaderboard import record_result

if TYPE_CHECKING:
    from dc_sim.evaluation import ScenarioDefinition
    from dc_sim.simulator import Simulator


def _execute_action(sim: Simulator, action: AgentAction) -> bool:
    """Execute a single agent action directly on the simulator.

    Returns True if the action was executed successfully.
    """
    t = action.action_type
    p = action.params

    try:
        if t == "migrate_workload":
            ok = sim.facility.workload_queue.migrate_job(
                p["job_id"], p["target_rack_id"]
            )
            sim.audit_log.record(
                timestamp=sim.clock.current_time,
                action="migrate_workload",
                params=p,
                result="ok" if ok else "not_found",
                source="agent",
            )
            return ok

        elif t == "adjust_cooling":
            sim.facility._crac_setpoints[p["rack_id"]] = p["setpoint_c"]
            sim.audit_log.record(
                timestamp=sim.clock.current_time,
                action="adjust_cooling",
                params=p,
                source="agent",
            )
            return True

        elif t == "throttle_gpu":
            sim.facility.set_server_power_cap(p["server_id"], p["power_cap_pct"])
            sim.audit_log.record(
                timestamp=sim.clock.current_time,
                action="throttle_gpu",
                params=p,
                source="agent",
            )
            return True

        elif t == "preempt_job":
            ok = sim.facility.workload_queue.preempt_job(p["job_id"])
            sim.audit_log.record(
                timestamp=sim.clock.current_time,
                action="preempt_job",
                params=p,
                result="ok" if ok else "not_found",
                source="agent",
            )
            return ok

        elif t == "resolve_failure":
            ok = sim.failure_engine.resolve(p["failure_id"])
            sim.audit_log.record(
                timestamp=sim.clock.current_time,
                action="resolve_failure",
                params=p,
                result="ok" if ok else "not_found",
                source="agent",
            )
            return ok

        else:
            return False

    except (KeyError, TypeError):
        return False


class AgentRunner:
    """Runs an agent against the simulator using SessionManager.

    Operates in-process (no HTTP). The agent's act() method receives the
    full facility state dict each tick and returns AgentActions that are
    executed directly on the Simulator.
    """

    def __init__(self, agent: BaseAgent, sim: Simulator) -> None:
        self.agent = agent
        self.sim = sim

    def run(
        self,
        scenario_id: str,
        record: bool = True,
        scenario_override: ScenarioDefinition | None = None,
    ) -> dict:
        """Run the agent against a single scenario.

        Args:
            scenario_id: Which scenario to run.
            record: If True, append result to the leaderboard CSV.
            scenario_override: Optional explicit ScenarioDefinition.
                When provided, this is used instead of looking up by
                *scenario_id*.

        Returns:
            EvaluationResult as dict.
        """
        mgr = SessionManager(self.sim)
        info = mgr.start(scenario_id, self.agent.name, scenario=scenario_override)
        self.agent.on_session_start(info)

        while True:
            step_result = mgr.step()
            state = step_result["state"]

            # Agent decides actions
            actions = self.agent.act(state)

            # Execute each action
            for action in actions:
                _execute_action(self.sim, action)

            if step_result["done"]:
                break

        # End session and get scores
        eval_result = mgr.end()
        result_dict = eval_result.to_dict()
        self.agent.on_session_end(result_dict)

        # Record to leaderboard
        if record:
            record_result(self.agent.name, scenario_id, result_dict)

        return result_dict

    def run_all(self, record: bool = True) -> list[dict]:
        """Run the agent against ALL scenarios.

        Returns list of EvaluationResult dicts.
        """
        from dc_sim.evaluation import SCENARIOS

        results = []
        for scenario_id in SCENARIOS:
            result = self.run(scenario_id, record=record)
            results.append(result)
        return results
