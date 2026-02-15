"""LLM-based agent using LangChain tool calling to control the simulation.

Requires: pip install langchain-openai langchain-core
Set OPENAI_API_KEY in your environment.
"""

from __future__ import annotations

import json
import os

from agents.base import AgentAction, BaseAgent


def _format_state_for_prompt(state: dict) -> str:
    """Format facility state into a readable string for the LLM."""
    # Select key metrics to keep prompt manageable
    parts = []
    parts.append(f"Tick: {state.get('tick_count')} | Time: {state.get('current_time')}")

    thermal = state.get("thermal", {})
    if racks := thermal.get("racks"):
        temps = [f"R{r['rack_id']}:{r['inlet_temp_c']:.1f}C" for r in racks[:8]]
        parts.append(f"Rack temps: {', '.join(temps)}")
        hot = [r for r in racks if r.get("inlet_temp_c", 0) > 35]
        if hot:
            parts.append(f"WARNING: Hot racks {[r['rack_id'] for r in hot]} > 35C")

    power = state.get("power", {})
    if power:
        parts.append(f"IT power: {power.get('total_it_kw', 0):.0f} kW, PUE: {power.get('pue', 0):.2f}")

    carbon = state.get("carbon", {})
    if carbon:
        parts.append(
            f"Carbon: {carbon.get('intensity_gco2_kwh', 0):.0f} gCO2/kWh, "
            f"cumulative: {carbon.get('cumulative_kg', 0):.1f} kg"
        )

    parts.append(
        f"Workload: {state.get('workload_pending')} pending, "
        f"{state.get('workload_running')} running, "
        f"{state.get('sla_violations')} SLA violations"
    )

    failures = state.get("failures", [])
    if failures:
        parts.append(
            f"Active failures: {[(f['failure_id'], f['type']) for f in failures]}"
        )

    running_jobs = state.get("running_jobs", [])
    if running_jobs:
        jobs_str = ", ".join(
            f"{j['job_id']}(p{j.get('priority', 3)})" for j in running_jobs[:10]
        )
        parts.append(f"Running jobs: {jobs_str}")

    return "\n".join(parts)


def _create_tools():
    """Create LangChain tools for simulator actions."""
    from langchain_core.tools import tool

    @tool
    def adjust_cooling(rack_id: int, setpoint_c: float) -> str:
        """Adjust the cooling setpoint for a rack. Use when inlet temp > 35C (cool down)
        or < 22C (save energy by raising setpoint). rack_id 0-7, setpoint 14-24 C."""
        return json.dumps({"action_type": "adjust_cooling", "params": {"rack_id": rack_id, "setpoint_c": setpoint_c}})

    @tool
    def migrate_workload(job_id: str, target_rack_id: int) -> str:
        """Move a running job to a different rack. Use to relieve hot racks or rebalance.
        target_rack_id 0-7. Get job_id from running_jobs."""
        return json.dumps({"action_type": "migrate_workload", "params": {"job_id": job_id, "target_rack_id": target_rack_id}})

    @tool
    def throttle_gpu(server_id: str, power_cap_pct: float) -> str:
        """Cap GPU power on a server (0-100). Use to reduce heat from a hot rack.
        server_id format: rack-N-srv-M e.g. rack-0-srv-0."""
        return json.dumps({"action_type": "throttle_gpu", "params": {"server_id": server_id, "power_cap_pct": power_cap_pct}})

    @tool
    def preempt_job(job_id: str) -> str:
        """Kill a running job to free GPU resources. Use when pending queue is long
        and you need to make room for higher-priority work. Prefer low-priority jobs."""
        return json.dumps({"action_type": "preempt_job", "params": {"job_id": job_id}})

    @tool
    def resolve_failure(failure_id: str) -> str:
        """Resolve an active failure. Always resolve failures when present.
        failure_id from the failures list in state."""
        return json.dumps({"action_type": "resolve_failure", "params": {"failure_id": failure_id}})

    return [adjust_cooling, migrate_workload, throttle_gpu, preempt_job, resolve_failure]


class LLMAgent(BaseAgent):
    """Agent that uses an LLM with tool calling to control the data centre.

    The LLM observes facility state each tick and chooses actions via LangChain tools.
    Requires OPENAI_API_KEY. Set OPENAI_LLM_MODEL env var to override model (default: gpt-4o-mini).
    """

    name = "llm_agent"

    def __init__(self, model: str | None = None):
        self.model_name = model or os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")
        self._tools = _create_tools()
        self._llm = None  # Lazy init to avoid import at module load

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.1,
            ).bind_tools(self._tools)
        return self._llm

    def act(self, state: dict) -> list[AgentAction]:
        """Format state, call LLM with tools, parse tool calls into AgentActions."""
        try:
            llm = self._get_llm()
        except ImportError as e:
            raise ImportError(
                "LLM agent requires: pip install langchain-openai langchain-core"
            ) from e

        if not os.environ.get("OPENAI_API_KEY"):
            return []

        state_str = _format_state_for_prompt(state)

        system = """You are a data centre operations agent. Each tick you receive facility state.
Your goal: minimise SLA violations, keep racks cool (inlet < 35C), resolve failures quickly,
and optimise for energy/carbon when safe. Use the tools to take actions. You may call multiple tools per tick.
Be decisive: if racks are hot, cool them; if failures exist, resolve them; if the queue is backed up, consider preempting low-priority jobs."""

        user = f"Current facility state:\n{state_str}\n\nWhat actions do you take this tick?"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        response = llm.invoke(messages)
        actions: list[AgentAction] = []

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "")
                args = getattr(tc, "args", None) or (tc.get("args") if isinstance(tc, dict) else {}) or {}
                try:
                    # Map tool name to action_type (they match)
                    action_type = name
                    if action_type in (
                        "adjust_cooling",
                        "migrate_workload",
                        "throttle_gpu",
                        "preempt_job",
                        "resolve_failure",
                    ):
                        actions.append(AgentAction(action_type=action_type, params=dict(args)))
                except Exception:
                    pass  # Skip malformed tool calls

        return actions

    def on_session_start(self, session_info: dict) -> None:
        self._llm = None  # Reset to re-bind tools for fresh session
