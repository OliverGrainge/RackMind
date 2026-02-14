"""Base agent interface. Subclass BaseAgent to create a new agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentAction:
    """A single action the agent wants to take.

    Attributes:
        action_type: One of "migrate_workload", "adjust_cooling",
            "throttle_gpu", "preempt_job", "resolve_failure".
        params: Action-specific parameters (see below).

    Parameter schemas by action_type:
        migrate_workload:  {"job_id": str, "target_rack_id": int}
        adjust_cooling:    {"rack_id": int, "setpoint_c": float}
        throttle_gpu:      {"server_id": str, "power_cap_pct": float}
        preempt_job:       {"job_id": str}
        resolve_failure:   {"failure_id": str}
    """

    action_type: str
    params: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for DC simulator agents.

    To create a new agent:
        1. Subclass BaseAgent
        2. Set `name` as a class attribute (used in leaderboard)
        3. Implement `act()` — return a list of AgentActions
        4. Register in `src/dc_sim/agents/__init__.py`

    Example::

        class MyAgent(BaseAgent):
            name = "my_smart_agent"

            def act(self, state: dict) -> list[AgentAction]:
                actions = []
                # Read state, decide what to do
                if state.get("thermal", {}).get("racks"):
                    for rack in state["thermal"]["racks"]:
                        if rack["inlet_temp_c"] > 35:
                            actions.append(AgentAction(
                                action_type="adjust_cooling",
                                params={"rack_id": rack["rack_id"], "setpoint_c": 15.0},
                            ))
                return actions
    """

    name: str = "unnamed"

    @abstractmethod
    def act(self, state: dict) -> list[AgentAction]:
        """Decide what actions to take given the current facility state.

        Args:
            state: Full facility state dict (same format as GET /status response).
                Contains keys: thermal, power, carbon, gpu, network, storage,
                cooling, workload_pending, workload_running, sla_violations,
                current_time, tick_count, plus failures (active list).

        Returns:
            List of AgentActions to execute this tick. Can be empty.
        """
        ...

    def on_session_start(self, session_info: dict) -> None:
        """Called when an evaluation session starts. Override for setup."""

    def on_session_end(self, result: dict) -> None:
        """Called when an evaluation session ends with scores. Override for cleanup."""
