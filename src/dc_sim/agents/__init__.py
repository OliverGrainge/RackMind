"""Agent registry.

To add a new agent:
    1. Create a new file in this directory (e.g. my_agent.py)
    2. Subclass BaseAgent and implement act()
    3. Import and register it below

Example::

    from dc_sim.agents.my_agent import MyAgent
    register_agent(MyAgent())
"""

from __future__ import annotations

from dc_sim.agents.base import AgentAction, BaseAgent

AGENT_REGISTRY: dict[str, BaseAgent] = {}


def register_agent(agent: BaseAgent) -> None:
    """Register an agent so it can be selected from the dashboard."""
    AGENT_REGISTRY[agent.name] = agent


# ── Built-in agents ──────────────────────────────────────────

from dc_sim.agents.random_agent import RandomAgent  # noqa: E402

register_agent(RandomAgent())

__all__ = [
    "AGENT_REGISTRY",
    "AgentAction",
    "BaseAgent",
    "register_agent",
]
