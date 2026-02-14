# DC Simulator

A high-fidelity, physics-based data centre simulator for developing and benchmarking AI agents. Train agents to optimise real-world operations — thermal management, power efficiency, workload scheduling, failure response, and carbon reduction — in a safe, repeatable environment with a standardised API and automated evaluation framework.


## Demo

<!-- Replace with your actual demo recording (GIF or MP4) -->
![Dashboard Demo](media/demo.gif)

*The monitoring dashboard provides real-time visibility across thermal, power, GPU, network, storage, cooling, carbon, and workload systems.*

## Architecture

```
dc-simulator/
├── src/
│   ├── agents/                 # Your agents live here
│   │   ├── base.py             # BaseAgent ABC + AgentAction dataclass
│   │   ├── random_agent.py     # Example baseline agent
│   │   └── __init__.py         # Agent registry
│   └── dc_sim/                 # Simulator platform (don't modify)
│       ├── api/                # REST API (FastAPI)
│       ├── models/             # Physics models (thermal, power, GPU, ...)
│       ├── simulator.py        # Simulation orchestrator
│       ├── evaluation.py       # Scoring framework (7 dimensions, 5 scenarios)
│       ├── runner.py           # Agent <-> simulator integration
│       └── ...
├── dashboard.py                # Streamlit monitoring dashboard
├── run.py                      # Single-command launcher
├── tests/                      # Test suite
└── SIMULATOR_GUIDE.md          # Full API and theory documentation
```

**Separation of concerns:**
- **`src/agents/`** — what you create and modify. Your custom agents, the base class, and the registry.
- **`src/dc_sim/`** — the simulator platform. Physics models, API, evaluation, scoring. You shouldn't need to touch this.
- **`run.py`** — starts everything with one command.

## What It Models

| System | Details |
|--------|---------|
| **Thermal** | Hot-aisle/cold-aisle rack temps, heat recirculation, non-linear cooling, time-varying ambient |
| **Power** | Non-linear GPU power curves, per-server/rack/facility draw, dynamic PUE |
| **GPU** | 128 H100-class GPUs: junction/HBM temp, SM utilisation, memory, clocks, ECC errors, PCIe/NVLink |
| **Network** | Leaf-spine fabric, ToR switches, RDMA/RoCE, M/M/1 queuing latency, packet loss |
| **Storage** | NVMe-oF shared storage: IOPS, throughput, latency (Little's Law), drive health/wear |
| **Cooling** | CRAC units, chilled water loop, cooling tower, COP efficiency, pump power |
| **Carbon** | UK-realistic grid carbon intensity (140-280 gCO2/kWh daily cycle), electricity spot pricing |
| **Workload** | Training / inference / batch jobs, Poisson arrivals, priority scheduling, SLA deadlines |
| **Failures** | CRAC degradation/failure, GPU degradation, PDU spikes, network partitions |

## Quick Start

```bash
git clone <repo-url>
cd dc-simulator
pip install -e .

# Launch API server + dashboard
python run.py
```

The API starts at [http://127.0.0.1:8000](http://127.0.0.1:8000) (docs at [/docs](http://127.0.0.1:8000/docs)).
The dashboard opens at [http://localhost:8501](http://localhost:8501).

```bash
python run.py --api-only         # API server only
python run.py --dashboard-only   # Dashboard only (API must be running)
python run.py --port 8080        # Custom API port
```

## Building an Agent

### Step 1 — Create your agent file

Create `src/agents/my_agent.py`:

```python
from agents.base import AgentAction, BaseAgent


class MyAgent(BaseAgent):
    name = "my_agent"

    def act(self, state: dict) -> list[AgentAction]:
        """Called every tick (60 s). Inspect state, return actions."""
        actions = []

        # --- Resolve any active failures immediately -----------------
        for f in state.get("failures", []):
            actions.append(AgentAction(
                action_type="resolve_failure",
                params={"failure_id": f["failure_id"]},
            ))

        # --- Thermal management: cool down hot racks -----------------
        for rack in state.get("thermal", {}).get("racks", []):
            if rack["inlet_temp_c"] > 35:
                actions.append(AgentAction(
                    action_type="adjust_cooling",
                    params={"rack_id": rack["rack_id"], "setpoint_c": 15.0},
                ))
            elif rack["inlet_temp_c"] < 22:
                # Save energy — raise setpoint when it's cool
                actions.append(AgentAction(
                    action_type="adjust_cooling",
                    params={"rack_id": rack["rack_id"], "setpoint_c": 21.0},
                ))

        # --- Workload management: preempt if overloaded --------------
        if state.get("workload_pending", 0) > 10:
            running_jobs = state.get("running_jobs", [])
            # Drop the lowest priority job to free capacity
            low = [j for j in running_jobs if j.get("priority", 3) <= 1]
            for job in low[:1]:
                actions.append(AgentAction(
                    action_type="preempt_job",
                    params={"job_id": job["job_id"]},
                ))

        return actions
```

### Step 2 — Register it

Add two lines to `src/agents/__init__.py`:

```python
from agents.my_agent import MyAgent
register_agent(MyAgent())
```

### Step 3 — Run it

**From the dashboard** — open the EVAL tab, select your agent and a scenario, then click RUN AGENT.

**From the API:**

```bash
curl -X POST http://127.0.0.1:8000/eval/run-agent \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "my_agent", "scenario_id": "thermal_crisis"}'
```

Results are automatically recorded to the leaderboard and visible in the dashboard.

### The `state` dict

Every tick your `act()` method receives the full facility state:

```python
{
    "current_time": "2025-01-01T00:05:00",
    "tick_count": 5,
    "thermal": {"racks": [{"rack_id": 0, "inlet_temp_c": 28.3, ...}, ...]},
    "power": {"total_it_kw": 312.5, "pue": 1.31, ...},
    "carbon": {"intensity_gco2_kwh": 195.0, "cumulative_kg": 42.1, ...},
    "gpu": {"servers": [{"server_id": "srv-00", "utilisation": 0.85, ...}, ...]},
    "cooling": {"crac_units": [...], "chiller_kw": 45.2, ...},
    "network": {"switches": [...], "avg_latency_us": 12.5, ...},
    "storage": {"drives": [...], "total_iops": 850000, ...},
    "workload_pending": 3,
    "workload_running": 24,
    "running_jobs": [{"job_id": "j-0042", "priority": 3, ...}, ...],
    "sla_violations": 0,
    "failures": [{"failure_id": "f-001", "type": "crac_degraded", ...}]
}
```

### Available Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `adjust_cooling` | `rack_id: int`, `setpoint_c: float` | Change CRAC cooling setpoint for a rack (14-24 C) |
| `migrate_workload` | `job_id: str`, `target_rack_id: int` | Move a running job to a different rack |
| `throttle_gpu` | `server_id: str`, `power_cap_pct: float` | Limit GPU power (0.0-1.0) |
| `preempt_job` | `job_id: str` | Kill a running job to free resources |
| `resolve_failure` | `failure_id: str` | Manually resolve an active failure |

### Lifecycle Hooks

Override these optional methods for setup/teardown:

```python
class MyAgent(BaseAgent):
    name = "my_agent"

    def on_session_start(self, session_info: dict) -> None:
        """Called before the first tick. Use for initialisation."""
        self.history = []

    def on_session_end(self, result: dict) -> None:
        """Called after the last tick with final scores."""
        print(f"Score: {result['composite_score']:.1f}")

    def act(self, state: dict) -> list[AgentAction]:
        self.history.append(state)
        # ... your logic ...
        return []
```

### LLM-Based Agents

For LLM tool-calling agents (e.g. using LangChain), define tools inside your agent class that map to `AgentAction` instances. Call the LLM in `act()` and return the resulting actions. The five simulator actions map directly to LangChain tools:

```python
from langchain_core.tools import tool
from agents.base import AgentAction, BaseAgent


class LLMAgent(BaseAgent):
    name = "llm_agent"

    def __init__(self):
        self.tools = [self.cool_rack, self.resolve, self.preempt]
        # Bind tools to your LLM of choice
        # self.llm = ChatAnthropic(...).bind_tools(self.tools)

    @staticmethod
    @tool
    def cool_rack(rack_id: int, setpoint_c: float) -> str:
        """Lower the cooling setpoint for an overheating rack."""
        return f"adjust_cooling|{rack_id}|{setpoint_c}"

    @staticmethod
    @tool
    def resolve(failure_id: str) -> str:
        """Resolve an active failure."""
        return f"resolve_failure|{failure_id}"

    @staticmethod
    @tool
    def preempt(job_id: str) -> str:
        """Preempt a low-priority job to free resources."""
        return f"preempt_job|{job_id}"

    def act(self, state: dict) -> list[AgentAction]:
        # Format state into a prompt, call LLM, parse tool calls
        # into AgentAction objects and return them.
        ...
```

## Evaluation & Scoring

Agents are scored across 7 dimensions, weighted into a composite score (0-100):

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| SLA Quality | 25% | Job completion rate, queue wait times |
| Energy Efficiency | 20% | PUE, power utilisation |
| Carbon | 15% | Emissions relative to workload |
| Thermal Safety | 15% | Temperature violations, throttling events |
| Cost | 10% | Electricity cost efficiency |
| Infrastructure Health | 10% | Equipment health, utilisation balance |
| Failure Response | 5% | Time-to-resolve, SLA impact during failures |

### Scenarios

| Scenario | Duration | Description |
|----------|----------|-------------|
| `steady_state` | 240 ticks (4h) | Normal operations — baseline efficiency test |
| `thermal_crisis` | 120 ticks (2h) | CRAC failure at t=30min — failure detection and workload migration |
| `carbon_valley` | 1440 ticks (24h) | Full day cycle — carbon-aware scheduling challenge |
| `overload` | 120 ticks (2h) | 3x job arrival rate — scheduling under pressure |
| `cascade` | 120 ticks (2h) | 5 sequential failures — multi-failure triage |

Custom scenarios can be configured from the dashboard's EVAL tab (duration, arrival rate, failure injections).

## Configuration

Set `DC_SIM_CONFIG` to a YAML file path, or use defaults. See `config.yaml.example` for all options.

```bash
DC_SIM_CONFIG=config.yaml python run.py
```

## Full Documentation

For the complete API reference, physics model details, and evaluation methodology, see **[SIMULATOR_GUIDE.md](SIMULATOR_GUIDE.md)**.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
