# Phase 1: Simulated Data Centre Environment

## Objective

Build a Python-based data centre simulator that produces realistic time-series telemetry for thermal, power, and workload systems. This simulator will serve as the environment for an LLM-based agent in later phases. It must expose telemetry via a FastAPI REST API and support failure injection.

---

## Project Structure

```
dc-simulator/
├── pyproject.toml
├── README.md
├── src/
│   └── dc_sim/
│       ├── __init__.py
│       ├── main.py              # FastAPI app entrypoint
│       ├── config.py            # All tuneable parameters as a dataclass
│       ├── clock.py             # Simulation clock (discrete time-step)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── thermal.py       # Thermal model
│       │   ├── power.py         # Power model
│       │   ├── workload.py      # Workload queue and scheduler
│       │   └── facility.py      # Top-level facility combining all models
│       ├── failures.py          # Failure injection engine
│       ├── api/
│       │   ├── __init__.py
│       │   └── routes.py        # All REST endpoints
│       └── telemetry.py         # In-memory telemetry ringbuffer and logger
└── tests/
    ├── test_thermal.py
    ├── test_power.py
    ├── test_workload.py
    └── test_api.py
```

---

## Dependencies

```
python >= 3.11
fastapi
uvicorn
pydantic >= 2.0
numpy
```

No GPU required. No ML libraries. No heavy frameworks. Keep it simple.

---

## Configuration (`config.py`)

Create a `SimConfig` Pydantic `BaseModel` (or dataclass) with the following tuneable parameters. All values below are sensible defaults — make them overridable via a `config.yaml` or environment variables.

```yaml
facility:
  num_racks: 8
  servers_per_rack: 4
  gpus_per_server: 4

thermal:
  ambient_temp_c: 22.0
  crac_setpoint_c: 18.0
  crac_cooling_capacity_kw: 50.0
  thermal_mass_coefficient: 0.3    # how quickly temps respond to load changes
  max_safe_inlet_temp_c: 35.0
  critical_inlet_temp_c: 40.0     # triggers thermal throttling

power:
  gpu_tdp_watts: 300               # per GPU
  server_base_power_watts: 200     # per server (CPU, memory, fans)
  pdu_capacity_kw: 20.0            # per rack
  facility_power_cap_kw: 120.0
  pue_overhead_factor: 1.4         # PUE = total facility power / IT power

workload:
  mean_job_arrival_interval_s: 300  # Poisson arrival
  job_duration_range_s: [600, 7200]
  gpu_requirement_range: [1, 8]

clock:
  tick_interval_s: 60               # simulation time per tick
  realtime_factor: 0.0              # 0 = as fast as possible, 1.0 = real-time
```

---

## Simulation Clock (`clock.py`)

Implement a discrete time-step simulation clock.

### Requirements

- Maintain a `current_time: float` in simulated seconds since epoch (start at 0).
- `tick()` advances `current_time` by `tick_interval_s`.
- If `realtime_factor > 0`, `tick()` sleeps for `tick_interval_s * realtime_factor` real seconds (this allows an agent to interact in near-real-time later).
- Expose `current_time`, `tick_count`, and `elapsed_human_readable` (e.g. `"02:15:00"`).
- The clock should be injectable as a dependency, not a global.

---

## Thermal Model (`models/thermal.py`)

Simulates rack-level inlet and outlet temperatures using a simplified energy balance.

### Data Structures

```python
@dataclass
class RackThermalState:
    rack_id: int
    inlet_temp_c: float
    outlet_temp_c: float
    heat_generated_kw: float
```

### Physics (simplified)

On each tick, for each rack:

1. **Heat generated** = sum of power draw across all servers in the rack (from power model).
2. **Heat removed** = `crac_cooling_capacity_kw / num_racks` (uniform distribution, unless a CRAC unit has failed — see failures).
3. **Net heat** = heat generated − heat removed.
4. **Temperature delta** = `net_heat * thermal_mass_coefficient * tick_interval_s / 60`.
5. **New inlet temp** = `previous_inlet_temp + temperature_delta`. Clamp to `[ambient_temp_c, 60.0]`.
6. **Outlet temp** = `inlet_temp + (heat_generated_kw * 5.0)` (rough approximation: ~5°C rise per kW of heat through a rack).

### Thermal throttling

If `inlet_temp >= critical_inlet_temp_c`, set a `throttled: bool = True` flag on the rack. Throttled racks cap GPU utilisation at 50%. The agent (in phase 2) should ideally intervene before this happens.

---

## Power Model (`models/power.py`)

Simulates per-server and per-rack power draw based on GPU utilisation.

### Data Structures

```python
@dataclass
class ServerPowerState:
    server_id: str          # e.g. "rack-0-srv-2"
    gpu_utilisation: float  # 0.0 - 1.0
    gpu_power_draw_w: float
    total_power_draw_w: float

@dataclass
class RackPowerState:
    rack_id: int
    total_power_kw: float
    pdu_utilisation_pct: float  # total_power / pdu_capacity

@dataclass
class FacilityPowerState:
    it_power_kw: float
    total_power_kw: float       # it_power * pue_overhead_factor
    pue: float
    headroom_kw: float          # facility_power_cap - total_power
    power_cap_exceeded: bool
```

### Logic

For each server on each tick:

1. `gpu_power_draw = gpu_utilisation * gpu_tdp_watts * num_gpus_on_server`
2. `total_server_power = server_base_power_watts + gpu_power_draw`
3. If rack is thermally throttled, cap `gpu_utilisation` at 0.5 before calculating.
4. Aggregate up to rack, then to facility.
5. `pue = total_facility_power / it_power` — in practice, use `pue_overhead_factor` as a constant multiplier (a simplification).

---

## Workload Model (`models/workload.py`)

Simulates a job queue with stochastic arrivals and a naive scheduler.

### Data Structures

```python
@dataclass
class Job:
    job_id: str               # UUID
    name: str                 # human-readable, e.g. "llm-finetune-37"
    gpu_requirement: int      # number of GPUs needed
    priority: int             # 1 (low) to 5 (critical)
    duration_s: int           # how long it runs (simulated seconds)
    submitted_at: float       # sim clock time
    started_at: float | None
    completed_at: float | None
    assigned_servers: list[str]
    status: str               # "queued", "running", "completed", "failed", "preempted"
    sla_deadline_s: float     # max acceptable queue wait time

class WorkloadQueue:
    pending: list[Job]
    running: list[Job]
    completed: list[Job]
```

### Logic

On each tick:

1. **Arrivals**: With probability derived from `mean_job_arrival_interval_s` (Poisson process), generate a new job with random GPU requirement, duration, and priority. Add to `pending`.
2. **Scheduling**: Naive first-fit. Iterate through `pending` (sorted by priority descending), find servers with available GPU slots, assign and move to `running`. If no capacity, job stays queued.
3. **Completion**: Check all `running` jobs. If `current_time - started_at >= duration_s`, move to `completed` and free the GPU slots.
4. **SLA tracking**: If a queued job exceeds `sla_deadline_s` without starting, flag it as an SLA violation (don't remove it, just track).
5. **GPU utilisation update**: Running jobs set GPU utilisation on their assigned servers. Idle GPUs have utilisation 0.05 (idle draw). This feeds into the power model.

---

## Facility Model (`models/facility.py`)

The top-level model that composes thermal, power, and workload.

### Requirements

- Holds instances of the thermal, power, and workload models.
- `step()` method calls workload → power → thermal in order each tick (workload determines GPU util → power depends on util → thermal depends on power).
- Holds a `FacilityState` snapshot after each tick containing all sub-model states.
- Passes the simulation clock to all sub-models.

---

## Failure Injection (`failures.py`)

Random and scriptable failures to test agent resilience.

### Failure Types

| Failure | Effect | Duration |
|---------|--------|----------|
| `crac_degraded` | One CRAC unit drops to 50% cooling capacity for affected racks | 10–30 min |
| `crac_failure` | One CRAC unit provides 0% cooling | 5–15 min |
| `gpu_degraded` | A specific GPU on a server is stuck at 30% max util | Until manually "repaired" via API |
| `pdu_spike` | A rack's power draw gets a +20% spike for 5 min | 5 min |
| `network_partition` | A rack's jobs fail and must be rescheduled | Instant |

### Interface

```python
class FailureEngine:
    def __init__(self, config, rng_seed: int = 42):
        ...

    def tick(self, current_time: float, facility_state) -> list[ActiveFailure]:
        """Probabilistically inject failures. Return list of newly activated failures."""
        ...

    def inject(self, failure_type: str, target: str, duration_s: int | None = None):
        """Manually inject a specific failure (used by API for testing)."""
        ...

    def get_active_failures(self) -> list[ActiveFailure]:
        ...

    def resolve(self, failure_id: str):
        """Manually resolve a failure (simulates repair)."""
        ...
```

Random failure probability per tick should be low (~0.5% per tick per rack) so the simulation doesn't become chaotic, but high enough that a 4-hour simulated run sees 2–3 failures.

---

## Telemetry Ringbuffer (`telemetry.py`)

Store the last N ticks of telemetry for API queries and time-series analysis.

### Requirements

- In-memory ringbuffer (use `collections.deque(maxlen=1000)`).
- Each entry is a timestamped `FacilityState` snapshot (serialisable to JSON).
- Support queries: `get_latest()`, `get_range(start_time, end_time)`, `get_last_n(n)`.
- Optionally write to a JSONL file for offline analysis.

---

## REST API (`api/routes.py`)

Expose the simulation state via FastAPI. These endpoints are what the LLM agent will call as tools in Phase 2.

### Endpoints

#### Telemetry (GET)

| Endpoint | Returns |
|----------|---------|
| `GET /status` | Full current `FacilityState` snapshot |
| `GET /thermal` | All rack thermal states |
| `GET /thermal/{rack_id}` | Single rack thermal state |
| `GET /power` | Facility power summary |
| `GET /power/{rack_id}` | Single rack power state |
| `GET /workload/queue` | All pending jobs |
| `GET /workload/running` | All running jobs |
| `GET /workload/completed?last_n=10` | Recent completed jobs |
| `GET /workload/sla_violations` | Jobs that missed SLA |
| `GET /failures/active` | Currently active failures |
| `GET /telemetry/history?last_n=60` | Last N ticks of full state |

#### Actions (POST) — these are what the agent will call

| Endpoint | Body | Effect |
|----------|------|--------|
| `POST /actions/migrate_workload` | `{ "job_id": "...", "target_rack_id": 3 }` | Move a running job to a different rack |
| `POST /actions/adjust_cooling` | `{ "rack_id": 2, "setpoint_c": 16.0 }` | Change CRAC setpoint for a zone |
| `POST /actions/throttle_gpu` | `{ "server_id": "rack-0-srv-2", "power_cap_pct": 70 }` | Limit GPU power on a server |
| `POST /actions/preempt_job` | `{ "job_id": "..." }` | Kill a low-priority job to free resources |
| `POST /actions/resolve_failure` | `{ "failure_id": "..." }` | Simulate repair of a failure |

#### Simulation Control (POST)

| Endpoint | Effect |
|----------|--------|
| `POST /sim/tick` | Advance one tick manually (useful for step-by-step agent testing) |
| `POST /sim/tick?n=10` | Advance N ticks |
| `POST /sim/run` | Start continuous simulation loop |
| `POST /sim/pause` | Pause continuous loop |
| `POST /sim/reset` | Reset to initial state |
| `POST /sim/inject_failure` | `{ "type": "crac_failure", "target": "rack-3" }` |
| `GET /sim/config` | Return current SimConfig |

---

## Testing Requirements

Write tests using `pytest`. Minimum coverage:

### `test_thermal.py`

- A rack with high GPU util and no cooling should see rising inlet temps over 10 ticks.
- A rack with zero load should converge toward ambient temperature.
- Thermal throttling should activate when inlet temp exceeds critical threshold.

### `test_power.py`

- Server power draw at 100% GPU util should equal `server_base_power + (gpu_tdp * num_gpus)`.
- Facility power should equal sum of all server power × PUE factor.
- `headroom_kw` should be negative when power cap is exceeded.

### `test_workload.py`

- A submitted job should move from `pending` to `running` when capacity is available.
- A job should move to `completed` after its duration elapses.
- SLA violations should be flagged when queue wait exceeds deadline.
- Preempting a job should free its GPU slots and mark it as `preempted`.

### `test_api.py`

- `GET /status` returns valid JSON matching the `FacilityState` schema.
- `POST /sim/tick` advances the clock.
- `POST /actions/migrate_workload` with a valid job ID returns 200.
- `POST /actions/migrate_workload` with an invalid job ID returns 404.

---

## Acceptance Criteria

The simulator is complete when:

1. `uvicorn dc_sim.main:app` starts without errors.
2. `POST /sim/tick?n=60` advances 60 ticks (simulating 1 hour) and `GET /status` returns a plausible facility state with non-trivial thermal, power, and workload data.
3. Injecting a `crac_failure` on rack-0 via the API causes rack-0's inlet temperature to rise over subsequent ticks.
4. Submitting jobs faster than capacity causes the queue to grow and SLA violations to appear.
5. `POST /actions/migrate_workload` successfully moves a running job from a hot rack to a cooler one, and subsequent ticks show the hot rack cooling down.
6. All tests pass: `pytest tests/ -v` with 0 failures.
7. Telemetry history endpoint returns time-series data suitable for plotting.

---

## Non-Goals (Do Not Build)

- No frontend / dashboard (this comes later, possibly Phase 3).
- No persistence / database — everything is in-memory.
- No authentication on the API.
- No Docker containerisation yet.
- No real hardware integration.
- No ML or LLM code — that is Phase 2.

---

## Notes for the Agent

- Start by implementing `config.py` and `clock.py` — these are dependencies for everything else.
- Then build models in order: `power.py` → `thermal.py` → `workload.py` → `facility.py`. Power and thermal are tightly coupled, but power is simpler so start there.
- Build the API last — it's just a thin layer over the facility model.
- Use type hints everywhere. Use Pydantic models for all API request/response schemas.
- Use a single `numpy.random.Generator` instance seeded from config for reproducibility.
- Keep each file under 200 lines if possible. Prefer clarity over cleverness.