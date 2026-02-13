# Work Package 1: Simulated Data Centre Agent Platform

**Duration:** 4–6 weeks  
**Effort:** ~20 hrs/week  

## Objective
Build a working LLM-based agentic system that monitors, reasons about, and makes decisions on simulated data centre telemetry — proving out the core architecture before you ever touch a real facility.

## Rationale
You don't have a data centre, but Empati needs you building agentic systems and data centre optimisation from day one. The right move is to build a realistic simulation environment and a real agent stack on top of it. This is genuinely useful — the agent code, tool interfaces, and decision logic transfer directly once real sensor data is available.

## Deliverables

### Phase 1: Simulated Data Centre Environment (Week 1–2)
Build a lightweight Python simulator that emits realistic time-series telemetry:

- **Thermal model:** Rack inlet/outlet temps, CRAC units, hot/cold aisle differentials. Use simplified thermodynamic equations — heat generated proportional to GPU load, cooling as a function of airflow and setpoint.  
- **Power model:** Per-server power draw (CPU/GPU utilisation → watts via empirical curves), PDU-level aggregation, PUE calculation.  
- **Workload model:** A queue of jobs (training runs, inference batches) with resource requirements, priorities, and SLAs. Jobs arrive stochastically.  
- **Failure injection:** Random GPU degradation, cooling partial failures, power spikes.  

This runs on your desktop. No GPU needed — it's a discrete event / time-step simulation. Use something like SimPy or just a custom tick loop. Expose telemetry via a simple API (FastAPI or just an in-memory stream).

### Phase 2: Agent Architecture (Week 2–4)
Build an LLM-based agent that consumes the simulated telemetry and acts on it:

**Tool-use agent framework:**  
Use LangChain/LangGraph or build a minimal tool-calling loop yourself (recommended — you'll learn more and Empati likely cares about this). The agent should have access to:

- `get_thermal_status()` — current temps across racks  
- `get_power_status()` — current draw, PUE, headroom  
- `get_workload_queue()` — pending and running jobs  
- `migrate_workload(job_id, target_rack)` — move a job  
- `adjust_cooling(zone, setpoint)` — change CRAC setpoint  
- `throttle_gpu(rack_id, power_cap)` — reduce GPU power limit  
- `alert(severity, message)` — raise an alert  

**Decision loop:**  
The agent runs on a periodic cycle (e.g., every simulated 5 minutes). It receives a telemetry summary, reasons about whether intervention is needed, and calls tools if so.

**LLM backend:**  
Use a local model on your desktop GPU (Llama 3 8B via Ollama, or Mistral 7B) for development iteration. Use Claude/GPT-4 API for evaluation runs. This keeps costs near zero during dev.

**Structured output:**  
The agent should produce structured JSON decisions, not just free text. This is critical for connecting to operational workflows later.

### Phase 3: Evaluation & Optimisation Scenarios (Week 4–6)
Build 3–4 concrete scenarios and evaluate the agent:

- **Thermal runaway:** A cooling unit degrades. Does the agent detect rising temps and redistribute load before thermal throttling kicks in?  
- **Power budget constraint:** Facility approaches power cap. Does the agent intelligently shed low-priority workloads?  
- **Workload scheduling:** Given a queue of mixed-priority jobs, does the agent pack them efficiently across racks while respecting thermal/power constraints?  
- **Carbon-aware scheduling (stretch):** Feed in a mock grid carbon intensity signal. Does the agent defer flexible workloads to low-carbon periods?  

For each scenario, log the agent's reasoning chain and actions. Build a simple scoring framework: energy cost, SLA violations, thermal exceedances, carbon intensity of compute.

## Infrastructure Plan

| Resource        | Use                                                            |
|-----------------|----------------------------------------------------------------|
| Desktop GPU     | Running local LLM for agent dev/testing                         |
| Slurm cluster   | Running batch evaluation across scenarios, parameter sweeps     |
| AWS node (opt.) | Only if you need a larger model (70B) for comparison evals      |

## Key Technical Decisions to Make Early

- **Build vs. framework:** I'd recommend building a minimal agent loop (~200 lines) rather than adopting a heavy framework. You'll understand tool-calling, context management, and retry logic deeply, which is exactly what this role values.  
- **Simulation fidelity:** Don't over-engineer the physics. The point is plausible telemetry that exercises the agent's reasoning. You can calibrate against real data later.  
- **Observability:** Log everything — every LLM call, every tool invocation, every decision. Build a simple dashboard (Streamlit or a basic React page) to replay agent sessions. This becomes your demo artifact.

## What This Sets You Up For
Once Empati gives you access to real sensor data or a real facility, you swap the simulator for a real telemetry ingestion layer. The agent, tools, and evaluation framework carry over directly. You've also demonstrated the core competency the role asks for: building LLM-based agentic systems, not just using them.
