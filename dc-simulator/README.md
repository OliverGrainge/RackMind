# DC Simulator

A Python-based data centre simulator that produces realistic time-series telemetry for thermal, power, carbon, workload, and failure systems. Designed as the environment for LLM-based agentic AI systems that optimise data centre operations.

**For a full explanation of the theory, metrics, and API, see [SIMULATOR_GUIDE.md](SIMULATOR_GUIDE.md).**

## What it models

- **Thermal** --- hot-aisle/cold-aisle rack temperatures, humidity, heat recirculation between adjacent racks, non-linear cooling efficiency, time-varying ambient temperature
- **Power** --- non-linear GPU power curves, per-server/rack/facility power draw, dynamic PUE that varies with load and ambient temperature
- **Carbon & Cost** --- UK-realistic time-varying grid carbon intensity (140-280 gCO2/kWh daily cycle) and electricity spot pricing (double-peak morning/evening pattern)
- **Workload** --- three job types (training, inference, batch) with distinct GPU, duration, priority, and SLA profiles; Poisson arrival process; first-fit priority scheduling
- **Failures** --- CRAC degradation/failure, GPU degradation, PDU spikes, network partitions; random injection + manual API control
- **Audit** --- append-only log of every action for attestation and evaluation

## Installation

```bash
cd dc-simulator
pip install -e .
```

## Running

```bash
uvicorn dc_sim.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API docs.

### Dashboard

With the API running, start the monitoring dashboard:

```bash
streamlit run dashboard.py
```

The dashboard provides thermal heatmaps, power breakdowns, carbon/cost gauges, workload tables, failure status, an audit log, and time-series history across four tabs (Power & PUE, Temperatures, Carbon, Cost).

## Quick examples

### Advance simulation and get full status
```bash
curl -X POST "http://127.0.0.1:8000/sim/tick?n=60"
curl http://127.0.0.1:8000/status
```

### Check carbon and electricity cost
```bash
curl http://127.0.0.1:8000/carbon
```

### Inject a CRAC failure
```bash
curl -X POST "http://127.0.0.1:8000/sim/inject_failure" \
  -H "Content-Type: application/json" \
  -d '{"type": "crac_failure", "target": "crac-0"}'
```

### Migrate a job to a cooler rack
```bash
curl -X POST "http://127.0.0.1:8000/actions/migrate_workload" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "<job-id>", "target_rack_id": 5}'
```

### Check the audit trail
```bash
curl http://127.0.0.1:8000/audit?last_n=10
```

## Configuration

Set `DC_SIM_CONFIG` to a YAML file path, or use defaults. See [SIMULATOR_GUIDE.md](SIMULATOR_GUIDE.md#configuration) for the full config schema and `config.yaml.example` for all options.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
