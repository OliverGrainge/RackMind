# Data Centre Simulator: How It Works & How to Use It

This document explains the theory behind the simulator, what each metric means, and how to use the system effectively.

---

## What Is It?

The DC Simulator is a **discrete time-step simulation** of a GPU data centre. It models six interconnected systems:

- **Thermal** --- rack inlet/outlet temperatures, humidity, hot-aisle recirculation, and CRAC cooling
- **Power** --- per-server and per-rack power draw with non-linear GPU power curves and dynamic PUE
- **Workload** --- a queue of typed jobs (training, inference, batch) with stochastic arrivals, priority scheduling, and SLA tracking
- **Carbon** --- time-varying grid carbon intensity and electricity spot pricing (UK-realistic profiles)
- **Failures** --- random and manual failure injection (CRAC, GPU, PDU, network)
- **Audit** --- append-only log of every action taken, for attestation and evaluation

It runs entirely in memory (no real hardware). Each "tick" advances simulated time by a configurable interval (default 60 seconds). The simulation exposes a **REST API** so an LLM agent or human operator can observe state and take actions.

---

## Simulation Flow (What Happens Each Tick)

On each tick, the simulator runs in this fixed order:

```
1. Advance the clock (+60 simulated seconds)
2. Failure engine: probabilistically inject new failures; expire old ones
3. Network partition: fail jobs on affected racks
4. Workload: new job arrivals, schedule pending jobs, complete finished jobs
5. Power: compute power from GPU utilisation (with throttling, power caps, dynamic PUE)
6. Thermal: update rack temperatures from heat vs cooling (with recirculation, humidity)
7. Carbon: compute carbon emissions and electricity cost for this tick
8. Append full state to telemetry ring buffer
```

The ordering matters: **Workload -> Power -> Thermal -> Carbon**. Jobs set GPU utilisation, which determines power draw, which determines heat generation, which determines temperature. Thermal throttling (inlet >= 40C) then caps GPU utilisation at 50% on the *next* tick, creating a stabilising feedback loop.

---

## Theory and Metrics

### Thermal Model

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

### Power Model

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

### Workload Model

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

### Carbon Model

Empati's core mission involves carbon attribution for AI infrastructure. The simulator models time-varying grid carbon intensity and electricity pricing using UK-realistic profiles.

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

### Failure Engine

Failures test the agent's ability to detect and respond to infrastructure problems.

| Failure type | Target | Effect | Default duration |
|---|---|---|---|
| `crac_degraded` | `crac-0` or `crac-1` | 50% cooling for that CRAC's racks | 10-30 min |
| `crac_failure` | `crac-0` or `crac-1` | 0% cooling (total loss) for that CRAC's racks | 5-15 min |
| `gpu_degraded` | `rack-{r}-srv-{s}` | GPU stuck at 30% max utilisation | Until manually resolved |
| `pdu_spike` | `rack-{r}` | +20% power draw on that rack | 5 min |
| `network_partition` | `rack-{r}` | All jobs on that rack fail immediately | Instant |

**CRAC zones:** CRAC unit 0 cools racks 0-3; CRAC unit 1 cools racks 4-7. A CRAC failure affects an entire zone (4 racks), making it a serious event.

**Random injection:** Each tick has a ~0.5% chance of generating a random failure (crac_degraded, pdu_spike, or network_partition). This means a typical 4-hour simulation will see 2-3 random failures.

**Failure cascades:** A CRAC failure causes temperatures to rise. If temperatures reach the throttle threshold, GPU utilisation drops, which reduces throughput, which may cause SLA violations. A good agent should detect the CRAC failure early and migrate workloads to the unaffected zone before throttling kicks in.

### Audit Log

Every action taken through the API (migrate, throttle, adjust cooling, preempt, resolve failure, inject failure) is recorded in an append-only audit log. Each entry contains:

- **Timestamp:** Simulated time when the action was taken
- **Action:** Which action endpoint was called
- **Params:** The request parameters
- **Result:** "ok" or an error description
- **Source:** "api" (default), extensible to "agent" or "operator" in Phase 2

The audit log enables:
- Post-hoc evaluation of agent decisions
- Comparison of different agent strategies on the same scenario
- Financial-grade attestation of what actions were taken and when

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
| GET | `/status` | Full facility state snapshot (thermal + power + carbon + workload) |
| GET | `/thermal` | All rack thermal states + ambient temp + humidity |
| GET | `/thermal/{rack_id}` | Single rack thermal state |
| GET | `/power` | Facility power summary (IT, total, PUE, headroom) |
| GET | `/power/{rack_id}` | Single rack power state |
| GET | `/carbon` | Carbon intensity, price, cumulative emissions and cost |
| GET | `/workload/queue` | Pending jobs (with job type) |
| GET | `/workload/running` | Running jobs (with job type and server assignments) |
| GET | `/workload/completed?last_n=10` | Recent completed jobs |
| GET | `/workload/sla_violations` | Jobs that missed their SLA |
| GET | `/failures/active` | Currently active failures |
| GET | `/telemetry/history?last_n=60` | Last N ticks for time-series analysis |
| GET | `/audit?last_n=50` | Recent audit log entries |

### Actions (Agent/Operator)

All actions are recorded in the audit log.

| Method | Endpoint | Body | What it does |
|---|---|---|---|
| POST | `/actions/migrate_workload` | `{"job_id": "...", "target_rack_id": 3}` | Move a running job to a different rack. Use this to relieve thermal hotspots or rebalance load. |
| POST | `/actions/adjust_cooling` | `{"rack_id": 2, "setpoint_c": 16.0}` | Lower the CRAC setpoint for a zone (more cooling but more energy). Default is 18C; lower values increase cooling capacity by up to 20%. |
| POST | `/actions/throttle_gpu` | `{"server_id": "rack-0-srv-2", "power_cap_pct": 70}` | Cap a server's GPU power to a percentage. Useful for reducing heat on specific servers. |
| POST | `/actions/preempt_job` | `{"job_id": "..."}` | Kill a running job to free GPU resources. Typically used to preempt low-priority batch jobs when urgent inference requests are queued. |
| POST | `/actions/resolve_failure` | `{"failure_id": "..."}` | Manually repair a failure (e.g. restart a degraded GPU). |

---

## Quick Start

### 1. Install and run

```bash
cd dc-simulator
pip install -e .
uvicorn dc_sim.main:app --reload
```

Open http://127.0.0.1:8000/docs for interactive API documentation.

### 2. Start the dashboard

With the API running, start the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard shows thermal heatmaps, power breakdowns, carbon/cost gauges, workload tables, failure status, audit log, and time-series charts across four tabs (Power & PUE, Temperatures, Carbon, Cost).

### 3. Advance time and inspect state

```bash
# Simulate 1 hour (60 ticks x 60s)
curl -X POST "http://127.0.0.1:8000/sim/tick?n=60"

# Full status (thermal + power + carbon + workload)
curl http://127.0.0.1:8000/status

# Just carbon and cost
curl http://127.0.0.1:8000/carbon
```

### 4. Trigger a failure and observe the response

```bash
# Inject CRAC failure (racks 0-3 lose all cooling)
curl -X POST "http://127.0.0.1:8000/sim/inject_failure" \
  -H "Content-Type: application/json" \
  -d '{"type": "crac_failure", "target": "crac-0"}'

# Advance 10 ticks and watch temperatures rise
curl -X POST "http://127.0.0.1:8000/sim/tick?n=10"

# Check rack 0 inlet temperature (should be rising)
curl http://127.0.0.1:8000/thermal/0
```

### 5. Move a job to relieve a hot rack

```bash
# Get a running job ID
curl http://127.0.0.1:8000/workload/running

# Migrate it to a cooler rack in the unaffected zone
curl -X POST "http://127.0.0.1:8000/actions/migrate_workload" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "<job-id-from-above>", "target_rack_id": 5}'
```

### 6. Check the audit trail

```bash
# See what actions have been taken
curl http://127.0.0.1:8000/audit?last_n=10
```

### 7. Monitor carbon and cost over time

```bash
# Get last 60 ticks of history (includes carbon data)
curl http://127.0.0.1:8000/telemetry/history?last_n=60
```

---

## Configuration

Set `DC_SIM_CONFIG` to the path of a YAML config file:

```bash
export DC_SIM_CONFIG=config.yaml
uvicorn dc_sim.main:app
```

Example `config.yaml`:

```yaml
facility:
  num_racks: 8
  servers_per_rack: 4
  gpus_per_server: 4

thermal:
  ambient_temp_c: 22.0       # Base outside temperature
  crac_setpoint_c: 18.0      # CRAC supply air temperature
  crac_cooling_capacity_kw: 50.0
  thermal_mass_coefficient: 0.3
  max_safe_inlet_temp_c: 35.0  # Warning threshold
  critical_inlet_temp_c: 40.0  # Throttling threshold
  crac_units: 2

power:
  gpu_tdp_watts: 300          # Per-GPU thermal design power
  server_base_power_watts: 200
  pdu_capacity_kw: 20.0
  facility_power_cap_kw: 120.0
  pue_overhead_factor: 1.4    # Base PUE (actual PUE varies dynamically)

workload:
  mean_job_arrival_interval_s: 300  # Job every ~5 min on average

clock:
  tick_interval_s: 60         # 1 tick = 60 simulated seconds

rng_seed: 42                  # For reproducible simulations
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
print(f"Rack 0 inlet:   {state.thermal.racks[0].inlet_temp_c:.1f}C")
print(f"Rack 0 humidity: {state.thermal.racks[0].humidity_pct:.0f}% RH")
print(f"IT power:        {state.power.it_power_kw:.1f} kW")
print(f"PUE:             {state.power.pue:.2f}")
print(f"Carbon emitted:  {state.carbon.cumulative_carbon_kg:.1f} kg CO2")
print(f"Electricity cost: {state.carbon.cumulative_cost_gbp:.2f} GBP")
print(f"Running jobs:    {state.workload_running}")
print(f"SLA violations:  {state.sla_violations}")
```

---

## Designed For

This simulator is built to support an **LLM-based agentic system** that will:

1. **Observe** telemetry via the GET endpoints (thermal, power, carbon, workload, failures)
2. **Reason** about the current state (is anything overheating? is carbon high? are SLAs at risk?)
3. **Act** via the POST action endpoints (migrate, throttle, adjust cooling, preempt, resolve)
4. **Learn** from outcomes tracked in the audit log and telemetry history

The agent's goal is to **optimise across multiple objectives simultaneously**:

- Minimise SLA violations (service quality)
- Minimise energy cost (electricity pricing)
- Minimise carbon emissions (carbon attribution)
- Maintain thermal safety (avoid throttling)
- Respond to failures (resilience)

These objectives sometimes conflict --- for example, running all jobs immediately minimises SLA violations but maximises cost and carbon. A good agent learns to defer flexible workloads to low-cost, low-carbon periods while keeping latency-sensitive inference jobs running at all times.

The same API can be used for manual testing, dashboards, automated evaluation scenarios, or multi-agent coordination experiments.
