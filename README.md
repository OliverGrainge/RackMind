# RackMind

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
│   │   ├── random_agent.py     # Baseline agent
│   │   ├── llm_agent.py       # LLM agent (LangChain tools)
│   │   └── __init__.py        # Agent registry
│   └── dc_sim/                 # Simulator platform (don't modify)
│       ├── api/                # REST API (FastAPI)
│       ├── models/             # Physics models (thermal, power, GPU, ...)
│       ├── simulator.py       # Simulation orchestrator
│       ├── evaluation.py      # Scoring framework (7 dimensions, 5 scenarios)
│       ├── runner.py          # Agent <-> simulator integration
│       └── ...
├── dashboard.py               # Streamlit monitoring dashboard
├── run.py                     # Single-command launcher
└── tests/                     # Test suite
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

# Optional: for the LLM agent, install with extras
pip install -e ".[llm]"

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

### Step 1 — Install LLM dependencies and create your agent

Install the LangChain extras (for the built-in LLM agent) or create your own:

```bash
pip install -e ".[llm]"
```

Create `src/agents/llm_agent.py` — an LLM-based agent that uses LangChain tool calling to control the simulation:

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from agents.base import AgentAction, BaseAgent


# Define a LangChain tool that the LLM can call to control the simulator
@tool
def adjust_cooling(rack_id: int, setpoint_c: float) -> str:
    """Adjust the cooling setpoint for a rack. Use when inlet temp > 35C to cool down,
    or when < 22C to save energy. rack_id 0-7, setpoint 14-24 C."""
    return f"adjust_cooling|{rack_id}|{setpoint_c}"


@tool
def resolve_failure(failure_id: str) -> str:
    """Resolve an active failure. Always resolve failures when present."""
    return f"resolve_failure|{failure_id}"


class LLMAgent(BaseAgent):
    name = "llm_agent"

    def __init__(self):
        self.tools = [adjust_cooling, resolve_failure]
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).bind_tools(self.tools)

    def act(self, state: dict) -> list[AgentAction]:
        """Format state, call LLM with tools, parse tool calls into AgentActions."""
        state_str = self._format_state(state)
        messages = [
            {"role": "system", "content": "You are a DC ops agent. Keep racks cool, resolve failures. Use the tools."},
            {"role": "user", "content": f"State:\n{state_str}\n\nWhat actions do you take?"},
        ]
        response = self.llm.invoke(messages)
        actions = []
        for tc in getattr(response, "tool_calls", []) or []:
            name = getattr(tc, "name", None) or tc.get("name", "")
            args = getattr(tc, "args", None) or tc.get("args", {}) or {}
            if name in ("adjust_cooling", "resolve_failure"):
                actions.append(AgentAction(action_type=name, params=dict(args)))
        return actions

    def _format_state(self, state: dict) -> str:
        racks = state.get("thermal", {}).get("racks", [])
        temps = ", ".join(f"R{r['rack_id']}:{r['inlet_temp_c']:.1f}C" for r in racks) if racks else "N/A"
        failures = state.get("failures", [])
        return f"Rack temps: {temps}\nFailures: {[(f['failure_id'], f['type']) for f in failures]}"
```

### Step 2 — Register it

Add to `src/agents/__init__.py`:

```python
from agents.llm_agent import LLMAgent
register_agent(LLMAgent())
```

*(The LLM agent is already registered if you installed with `.[llm]`; it's available as `llm_agent`.)*

### Step 3 — Run it

Set your OpenAI API key, then run:

```bash
export OPENAI_API_KEY=sk-...
```

**From the dashboard** — open the EVAL tab, select `llm_agent` and a scenario, then click RUN AGENT.

**From the API:**

```bash
curl -X POST http://127.0.0.1:8000/eval/run-agent \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "llm_agent", "scenario_id": "thermal_crisis"}'
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
| `throttle_gpu` | `server_id: str`, `power_cap_pct: float` | Limit GPU power (0-100%) |
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
        return []  # your logic
```

### Alternative: Rule-Based Agents

For agents without an LLM, implement `act()` with explicit logic:

```python
from agents.base import AgentAction, BaseAgent

class MyAgent(BaseAgent):
    name = "my_agent"

    def act(self, state: dict) -> list[AgentAction]:
        actions = []
        for f in state.get("failures", []):
            actions.append(AgentAction("resolve_failure", {"failure_id": f["failure_id"]}))
        for rack in state.get("thermal", {}).get("racks", []):
            if rack["inlet_temp_c"] > 35:
                actions.append(AgentAction("adjust_cooling", {"rack_id": rack["rack_id"], "setpoint_c": 15.0}))
        return actions
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

---

## How It Works

The DC Simulator is a **discrete time-step simulation** of a GPU data centre. It models ten interconnected systems:

- **Thermal** — rack inlet/outlet temperatures, humidity, hot-aisle recirculation, and CRAC cooling
- **Power** — per-server and per-rack power draw with non-linear GPU power curves and dynamic PUE
- **GPU** — per-GPU telemetry (temperature, utilisation, memory, clocks, ECC errors, PCIe/NVLink bandwidth)
- **Network** — leaf-spine fabric with per-rack ToR switches, east-west/north-south traffic, RDMA, latency modelling
- **Storage** — NVMe-oF shared storage with IOPS, throughput, latency, capacity tracking, and drive wear
- **Cooling** — CRAC units, chilled water loop, cooling tower, COP efficiency, pump power
- **Workload** — a queue of typed jobs (training, inference, batch) with stochastic arrivals, priority scheduling, and SLA tracking
- **Carbon** — time-varying grid carbon intensity and electricity spot pricing (UK-realistic profiles)
- **Failures** — random and manual failure injection (CRAC, GPU, PDU, network)
- **Audit** — append-only log of every action taken, for attestation and evaluation

It runs entirely in memory (no real hardware). Each "tick" advances simulated time by a configurable interval (default 60 seconds). The simulation exposes a **REST API** so an LLM agent or human operator can observe state and take actions.

### Simulation Flow (What Happens Each Tick)

On each tick, the simulator runs in this fixed order:

```
 1. Advance the clock (+60 simulated seconds)
 2. Failure engine: probabilistically inject new failures; expire old ones
 3. Network partition: fail jobs on affected racks
 4. Workload: new job arrivals, schedule pending jobs, complete finished jobs
 5. Power: compute power from GPU utilisation (with throttling, power caps, dynamic PUE)
 6. Thermal: update rack temperatures from heat vs cooling (with recirculation, humidity)
 7. GPU: compute per-GPU telemetry (temps, clocks, memory, ECC, bandwidth)
 8. Network: compute per-rack traffic, RDMA, latency, spine utilisation
 9. Storage: compute per-rack I/O (IOPS, throughput, latency, capacity, drive wear)
10. Cooling: compute CRAC unit states, chilled water loop, cooling tower, COP
11. Carbon: compute carbon emissions and electricity cost for this tick
12. Append full state to telemetry ring buffer
```

The ordering matters: **Workload -> Power -> Thermal -> GPU -> Network -> Storage -> Cooling -> Carbon**. Jobs set GPU utilisation, which determines power draw, which determines heat generation, which determines temperature. GPU telemetry then uses per-rack temperatures and workload utilisation to derive per-GPU metrics. Network and storage I/O are driven by the active workload mix. Cooling state is derived from the thermal load. Carbon accumulates from total facility power (IT + cooling). Thermal throttling (inlet >= 40C) then caps GPU utilisation at 50% on the *next* tick, creating a stabilising feedback loop.

### Theory and Metrics

#### Thermal Model

Real data centres use a **hot-aisle/cold-aisle** layout. CRAC (Computer Room Air Conditioning) units push cold air into a raised floor plenum. Cold air enters racks at the front (the "cold aisle"), absorbs heat from servers, and exits at the back (the "hot aisle"). Containment directs this hot exhaust back to the CRAC return.

The simulator models this with a simplified energy balance per rack:

```
net_heat = heat_generated + recirculation_heat - heat_removed
temp_delta = net_heat * thermal_mass_coefficient * (tick_interval / 60)
new_inlet = previous_inlet + temp_delta
```

**Key metrics:**

| Metric | Unit | What it means |
|---|---|---|
| **Inlet temperature** | C | Air temperature entering the rack at the cold aisle. ASHRAE recommends 18-27C for reliable operation. Above 35C is a warning; above 40C triggers thermal throttling. |
| **Outlet temperature** | C | Air temperature leaving the rack at the hot aisle. Typically inlet + 10-20C depending on load. Calculated as `inlet + (heat_kw * 5)`. |
| **Delta-T** | C | Outlet minus inlet. Higher delta-T means more heat is being extracted from the air, indicating higher server load. |
| **Heat generated** | kW | Thermal power emitted by the servers in this rack. Equal to the rack's IT power draw (all electrical energy becomes heat). |
| **Throttled** | bool | Whether the rack has hit the critical inlet temperature (default 40C). When throttled, GPU utilisation is hard-capped at 50%, reducing both performance and heat output. |
| **Humidity** | % RH | Relative humidity at the rack. ASHRAE recommends 20-80% RH. Too low risks static discharge; too high risks condensation. Humidity drops when heat is high (hot air has lower relative humidity) and rises with active cooling. |
| **Ambient temperature** | C | Outside air temperature, which varies on a daily cycle (+/- 4C swing, peaking mid-afternoon). Higher ambient makes the CRAC work harder (it rejects heat to outside air), reducing effective cooling capacity. |

**Hot-aisle recirculation:** In practice, some hot exhaust air leaks around rack containment and mixes with cold-aisle supply air. The simulator models this: each rack receives ~8% of its neighbours' exhaust heat. Racks in the middle of a row run slightly warmer than racks at the ends.

**Non-linear cooling efficiency:** Cooling degrades at high temperatures (above 30C inlet, each degree reduces efficiency by 2%) and high humidity (above 60% RH). Additionally, when outdoor ambient temperature rises, the CRAC's ability to reject heat decreases.

#### Power Model

GPU power draw is the dominant factor in data centre energy consumption. The simulator models this at the server level:

```
server_power = base_power + gpu_power(utilisation) * num_gpus
rack_power = sum(server_power for each server) * pdu_multiplier
facility_it_power = sum(rack_power for each rack)
facility_total_power = it_power * PUE
```

**Non-linear GPU power curve:** Real GPUs do not scale power linearly with utilisation. At idle they draw ~5% of TDP (fans, memory controllers). As utilisation increases, power rises superlinearly due to voltage/frequency scaling:

```
gpu_power(util) = (idle_fraction + (1 - idle_fraction) * (0.3*util + 0.7*util^2)) * TDP
```

This means a GPU at 50% utilisation draws significantly less than 50% of peak power --- closer to 35-40%.

**Key metrics:**

| Metric | Unit | What it means |
|---|---|---|
| **IT Power** | kW | Total electrical power consumed by IT equipment (servers, GPUs, switches). This is the "useful" power that does compute work. |
| **Total Power** | kW | Total facility power including cooling, lighting, UPS losses. Equal to IT power multiplied by PUE. |
| **PUE** | ratio | Power Usage Effectiveness. Total facility power divided by IT power. A PUE of 1.4 means for every 1 kW of compute, 0.4 kW goes to overhead (cooling, etc). Industry average is ~1.58; hyperscalers achieve 1.1-1.2. |
| **Headroom** | kW | Facility power cap minus current total power. Positive means within budget; negative means the cap is exceeded. |
| **PDU utilisation** | % | How much of a rack's Power Distribution Unit capacity is being used. Exceeding 80% is a warning; 100% risks a PDU trip. |
| **GPU utilisation** | 0-1 | Fraction of GPU compute capacity in use. 0.05 = idle; 0.9 = heavy training workload. |

**Dynamic PUE:** In reality, PUE is not constant. The simulator varies PUE based on two factors:

1. **Load fraction:** At low IT load, fixed overhead (lighting, UPS losses, baseline cooling) dominates, pushing PUE higher. At full load, PUE approaches the configured base (default 1.4).
2. **Ambient temperature:** Every degree above 22C adds ~0.005 to PUE because chillers must work harder to reject heat.

#### Workload Model

The workload model simulates a realistic job mix arriving at the data centre:

**Job types:**

| Type | GPU range | Duration | Priority | SLA | GPU utilisation |
|---|---|---|---|---|---|
| **Training** | 4-16 GPUs | 1-4 hours | 2-4 (medium-high) | 30 min - 2 hours | 92% (sustained) |
| **Inference** | 1-2 GPUs | 1-10 min | 4-5 (high-critical) | 30 sec - 5 min | 60% (bursty) |
| **Batch** | 2-8 GPUs | 10 min - 2 hours | 1-3 (low-medium) | 1-4 hours | 85% |

**Arrival process:** Jobs arrive following a Poisson process with a configurable mean interval (default 5 minutes). The probability of at least one arrival per tick is `1 - exp(-rate * tick_interval)`. Arrival type is weighted: 50% inference, 30% batch, 20% training.

**Scheduling:** A simple first-fit priority scheduler. Pending jobs are sorted by descending priority and placed on the first available GPU slots. This is intentionally naive --- a good agent should be able to improve upon it by preempting low-priority batch jobs to make room for urgent inference requests.

**Key metrics:**

| Metric | What it means |
|---|---|
| **Pending** | Jobs waiting in the queue for GPU resources. High pending count means the cluster is oversubscribed. |
| **Running** | Jobs currently executing on GPUs. |
| **Completed** | Total jobs that have finished (includes successful, failed, and preempted). |
| **SLA violations** | Jobs whose queue wait time exceeded their SLA deadline before they started running. This is the primary measure of service quality. |

#### Carbon Model

The simulator models time-varying grid carbon intensity and electricity pricing using UK-realistic profiles.

**Carbon intensity** (grams CO2 per kWh) represents how much carbon dioxide is emitted per unit of electricity generated. It varies with the grid's generation mix:

- **Night (01:00-05:00):** ~140 gCO2/kWh --- wind and nuclear dominate
- **Morning (08:00-10:00):** ~200 gCO2/kWh --- gas peakers ramp up for demand
- **Afternoon peak (14:00-16:00):** ~260-280 gCO2/kWh --- gas generation at maximum
- **Evening (18:00-22:00):** ~220 gCO2/kWh --- demand tapering off

The profile follows a sinusoidal daily pattern with Gaussian noise (std dev 5 gCO2/kWh).

**Electricity price** (GBP per kWh) follows a double-peak pattern reflecting demand:

- **Night trough (01:00-05:00):** ~0.10 GBP/kWh --- cheapest electricity
- **Morning peak (07:00-09:00):** ~0.23 GBP/kWh --- commuter/industrial demand
- **Mid-day (11:00-15:00):** ~0.15 GBP/kWh --- moderate
- **Evening peak (17:00-19:00):** ~0.21 GBP/kWh --- residential demand surge
- **Late evening (21:00-00:00):** ~0.13 GBP/kWh --- declining

**Key metrics:**

| Metric | Unit | What it means |
|---|---|---|
| **Carbon intensity** | gCO2/kWh | Current grid carbon intensity. An agent can reduce carbon by deferring batch jobs to low-carbon periods (e.g. overnight when wind generation is high). |
| **Carbon rate** | gCO2/s | Instantaneous carbon emission rate = (carbon_intensity * total_power_kw) / 3600. |
| **Cumulative carbon** | kg CO2 | Total carbon emitted since simulation start. The key metric for carbon attribution. |
| **Electricity price** | GBP/kWh | Current spot price. An agent can reduce cost by shifting flexible workloads to off-peak periods. |
| **Cost rate** | GBP/hour | Instantaneous electricity cost = price * total_power_kw. |
| **Cumulative cost** | GBP | Total electricity cost since simulation start. |

**Carbon-aware scheduling:** An intelligent agent could exploit the ~2x daily variation in carbon intensity by:
- Running latency-insensitive batch/training jobs during low-carbon overnight windows
- Accepting higher SLA risk for low-priority jobs in exchange for significant carbon savings
- Balancing power consumption to avoid peak pricing periods

#### GPU Model

The GPU model produces per-GPU telemetry for every GPU in the facility (default: 128 GPUs across 32 servers). It simulates NVIDIA H100-class accelerators with realistic thermal, power, and performance characteristics.

**Key metrics (per GPU):**

| Metric | Unit | What it means |
|---|---|---|
| **sm_utilisation_pct** | % | Streaming multiprocessor utilisation. 0% = idle, 90%+ = heavy compute (training). |
| **mem_utilisation_pct** | % | Memory controller utilisation, correlated with but not identical to SM util. |
| **gpu_temp_c** | C | GPU junction temperature. Normal: 30-75C. Warning: 75-83C. Throttle: >83C. |
| **mem_temp_c** | C | HBM memory temperature. Typically 5-10C above junction. >95C risks data corruption. |
| **power_draw_w** | W | Instantaneous GPU board power. TDP is 300W; actual range is ~15W (idle) to 300W (full load). |
| **sm_clock_mhz** | MHz | Current SM clock frequency. Base: 1095 MHz. Boost: up to 1980 MHz. Drops under thermal/power throttling. |
| **mem_clock_mhz** | MHz | HBM memory clock. Base: 1593 MHz. |
| **mem_used_mib** | MiB | GPU memory allocated. Total is 81920 MiB (80 GiB) per H100. Training jobs use 60-95%. |
| **mem_total_mib** | MiB | Total available GPU memory (81920 MiB). |
| **ecc_sbe_count** | count | Cumulative single-bit ECC errors (correctable). Rising count is a wear indicator. |
| **ecc_dbe_count** | count | Cumulative double-bit ECC errors (uncorrectable). Any DBE is a serious hardware concern. |
| **pcie_tx_gbps** / **pcie_rx_gbps** | Gbps | PCIe Gen5 x16 bandwidth (max ~64 Gbps). |
| **nvlink_tx_gbps** / **nvlink_rx_gbps** | Gbps | NVLink bandwidth (max ~450 Gbps). Heavy during multi-GPU training. |
| **fan_speed_pct** | % | GPU fan speed. 30% at idle, 100% at thermal throttle. |
| **thermal_throttle** | bool | Whether the GPU is thermally throttling (junction > 83C). |
| **power_throttle** | bool | Whether the GPU is power-limited (drawing at TDP). |

#### Network Model

The network model simulates a **leaf-spine data centre fabric** with Top-of-Rack (ToR) switches. **Latency** uses an M/M/1 queuing model: `latency = base_latency / (1 - utilisation)`. **Packet loss** is effectively zero below 80% ToR utilisation; above 80%, loss increases quadratically.

**Key metrics (per rack):** ingress_gbps, egress_gbps, tor_utilisation_pct, avg_latency_us, p99_latency_us, packet_loss_pct, rdma_tx_gbps, rdma_rx_gbps.

#### Storage Model

The storage model simulates **NVMe-oF shared storage** — each rack has a local NVMe shelf (default 100 TB, ~800K max IOPS). **Latency** increases with queue depth via Little's Law. **Drive health** tracks write endurance; below 90% is a warning.

**Key metrics (per rack):** read_iops, write_iops, total_iops, avg_read_latency_us, avg_write_latency_us, used_tb, total_tb, drive_health_pct.

#### Cooling Model

The cooling model simulates **CRAC units**, **chilled water loop**, and **cooling tower**. **COP (Coefficient of Performance)** varies from ~2.0 (hot ambient, high load) to ~6.0 (cool ambient, low load).

**Key metrics:** supply_air_temp_c, return_air_temp_c, cooling_output_kw, cooling_capacity_kw, load_pct, operational, cop, cooling_power_kw.

#### Failure Engine

| Failure type | Target | Effect | Default duration |
|---|---|---|---|
| `crac_degraded` | `crac-0` or `crac-1` | 50% cooling for that CRAC's racks | 10-30 min |
| `crac_failure` | `crac-0` or `crac-1` | 0% cooling (total loss) for that CRAC's racks | 5-15 min |
| `gpu_degraded` | `rack-{r}-srv-{s}` | GPU stuck at 30% max utilisation | Until manually resolved |
| `pdu_spike` | `rack-{r}` | +20% power draw on that rack | 5 min |
| `network_partition` | `rack-{r}` | All jobs on that rack fail immediately | Instant |

**CRAC zones:** CRAC unit 0 cools racks 0-3; CRAC unit 1 cools racks 4-7. Each tick has a ~0.5% chance of random failure injection.

#### Audit Log

Every action taken through the API is recorded in an append-only audit log with timestamp, action, params, result, and source — enabling post-hoc evaluation and attestation.

---

## API Reference

Base URL: `http://127.0.0.1:8000` (when running locally). Interactive Swagger docs at `/docs`.

### Simulation Control

| Method | Endpoint | Description |
|---|---|---|
| POST | `/sim/tick?n=10` | Advance simulation by `n` ticks |
| POST | `/sim/run?tick_interval_s=0.5` | Start continuous auto-tick in background |
| POST | `/sim/pause` | Stop continuous auto-tick |
| GET | `/sim/status` | Whether continuous sim is running + tick count |
| POST | `/sim/reset` | Reset to initial state (clears all history) |
| POST | `/sim/inject_failure` | Inject a failure (see body below) |
| GET | `/sim/config` | Current configuration parameters |

**Inject failure body:**
```json
{"type": "crac_failure", "target": "crac-0", "duration_s": 600}
```
`duration_s` is optional; defaults depend on failure type.

### Telemetry (Read State)

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/status` | Full facility state snapshot |
| GET | `/thermal` | All rack thermal states + ambient temp + humidity |
| GET | `/thermal/{rack_id}` | Single rack thermal state |
| GET | `/power` | Facility power summary (IT, total, PUE, headroom) |
| GET | `/power/{rack_id}` | Single rack power state |
| GET | `/gpu` | Facility-wide GPU summary |
| GET | `/gpu/{server_id}` | Full per-GPU telemetry for a specific server |
| GET | `/network` | Facility-wide network summary + per-rack ToR + spine links |
| GET | `/network/{rack_id}` | Single rack network state |
| GET | `/storage` | Facility-wide storage summary + per-rack NVMe shelf data |
| GET | `/storage/{rack_id}` | Single rack storage state |
| GET | `/cooling` | Full cooling system state (COP, CHW loop, CRAC units, cooling tower) |
| GET | `/carbon` | Carbon intensity, price, cumulative emissions and cost |
| GET | `/workload/queue` | Pending jobs |
| GET | `/workload/running` | Running jobs |
| GET | `/workload/completed?last_n=10` | Recent completed jobs |
| GET | `/workload/sla_violations` | Jobs that missed their SLA |
| GET | `/failures/active` | Currently active failures |
| GET | `/telemetry/history?last_n=60` | Last N ticks for time-series analysis |
| GET | `/audit?last_n=50` | Recent audit log entries |

### Actions (Agent/Operator)

All actions are recorded in the audit log.

| Method | Endpoint | Body | What it does |
|---|---|---|
| POST | `/actions/migrate_workload` | `{"job_id": "...", "target_rack_id": 3}` | Move a running job to a different rack |
| POST | `/actions/adjust_cooling` | `{"rack_id": 2, "setpoint_c": 16.0}` | Lower the CRAC setpoint for a zone |
| POST | `/actions/throttle_gpu` | `{"server_id": "rack-0-srv-2", "power_cap_pct": 70}` | Cap a server's GPU power |
| POST | `/actions/preempt_job` | `{"job_id": "..."}` | Kill a running job to free GPU resources |
| POST | `/actions/resolve_failure` | `{"failure_id": "..."}` | Manually repair a failure |

### Exploring the API (curl examples)

```bash
# Simulate 1 hour (60 ticks x 60s)
curl -X POST "http://127.0.0.1:8000/sim/tick?n=60"

# Full status
curl http://127.0.0.1:8000/status

# Carbon and cost
curl http://127.0.0.1:8000/carbon

# GPU fleet and per-server detail
curl http://127.0.0.1:8000/gpu
curl http://127.0.0.1:8000/gpu/rack-0-srv-0

# Network and storage
curl http://127.0.0.1:8000/network
curl http://127.0.0.1:8000/storage
curl http://127.0.0.1:8000/cooling

# Inject CRAC failure and observe
curl -X POST "http://127.0.0.1:8000/sim/inject_failure" \
  -H "Content-Type: application/json" \
  -d '{"type": "crac_failure", "target": "crac-0"}'

# Migrate a job to relieve a hot rack
curl -X POST "http://127.0.0.1:8000/actions/migrate_workload" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "<job-id>", "target_rack_id": 5}'

# Check audit trail
curl http://127.0.0.1:8000/audit?last_n=10
```

---

## Configuration

Set `DC_SIM_CONFIG` to a YAML file path, or use defaults:

```bash
DC_SIM_CONFIG=config.yaml python run.py
```

Example `config.yaml`:

```yaml
facility:
  num_racks: 8
  servers_per_rack: 4
  gpus_per_server: 4

thermal:
  ambient_temp_c: 22.0
  crac_setpoint_c: 18.0
  crac_cooling_capacity_kw: 50.0
  thermal_mass_coefficient: 0.3
  max_safe_inlet_temp_c: 35.0
  critical_inlet_temp_c: 40.0
  crac_units: 2

power:
  gpu_tdp_watts: 300
  server_base_power_watts: 200
  pdu_capacity_kw: 20.0
  facility_power_cap_kw: 120.0
  pue_overhead_factor: 1.4

workload:
  mean_job_arrival_interval_s: 300

clock:
  tick_interval_s: 60

rng_seed: 42
```

See `config.yaml.example` for all options.

---

## Python Usage (Without API)

You can drive the simulator programmatically:

```python
from dc_sim.simulator import Simulator

sim = Simulator()
sim.tick(60)  # 1 hour

state = sim.telemetry.get_latest()

# Thermal and power
print(f"Rack 0 inlet:   {state.thermal.racks[0].inlet_temp_c:.1f}C")
print(f"IT power:        {state.power.it_power_kw:.1f} kW")
print(f"PUE:             {state.power.pue:.2f}")

# GPU telemetry
print(f"GPU fleet:       {state.gpu.healthy_gpus}/{state.gpu.total_gpus} healthy")
print(f"Avg GPU temp:    {state.gpu.avg_gpu_temp_c:.1f}C")

# Carbon and cost
print(f"Carbon emitted:  {state.carbon.cumulative_carbon_kg:.1f} kg CO2")
print(f"Running jobs:    {state.workload_running}")
```

---

## Designed For

This simulator is built to support an **LLM-based agentic system** that will:

1. **Observe** telemetry via the GET endpoints (thermal, power, GPU, network, storage, cooling, carbon, workload, failures)
2. **Reason** about the current state (is anything overheating? are GPUs throttling? is network congested? is carbon high? are SLAs at risk?)
3. **Act** via the POST action endpoints (migrate, throttle, adjust cooling, preempt, resolve)
4. **Learn** from outcomes tracked in the audit log and telemetry history

The agent's goal is to **optimise across multiple objectives simultaneously**:

- Minimise SLA violations (service quality)
- Minimise energy cost (electricity pricing)
- Minimise carbon emissions (carbon attribution)
- Maintain thermal safety (avoid GPU/rack throttling)
- Maintain network health (low latency, no packet loss)
- Manage storage capacity and drive health
- Maximise cooling efficiency (high COP)
- Respond to failures (resilience)

These objectives sometimes conflict --- for example, running all jobs immediately minimises SLA violations but maximises cost and carbon. A good agent learns to defer flexible workloads to low-cost, low-carbon periods while keeping latency-sensitive inference jobs running at all times.

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
