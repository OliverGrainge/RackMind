# DC Simulator

A Python-based data centre simulator that produces realistic time-series telemetry for thermal, power, GPU, network, storage, cooling, carbon, workload, and failure systems. Designed as the environment for LLM-based agentic AI systems that optimise data centre operations.

**For a full explanation of the theory, metrics, and API, see [SIMULATOR_GUIDE.md](SIMULATOR_GUIDE.md).**

## What it models

- **Thermal** --- hot-aisle/cold-aisle rack temperatures, humidity, heat recirculation between adjacent racks, non-linear cooling efficiency, time-varying ambient temperature
- **Power** --- non-linear GPU power curves, per-server/rack/facility power draw, dynamic PUE that varies with load and ambient temperature
- **GPU** --- per-GPU telemetry for 128 H100-class GPUs: junction/HBM temperature, SM utilisation, memory allocation, clock speeds (boost/throttle), power draw, ECC error tracking (SBE/DBE), PCIe/NVLink bandwidth, fan speed
- **Network** --- leaf-spine data centre fabric with per-rack ToR switches, east-west/north-south traffic, RDMA/RoCE for GPU-to-GPU, M/M/1 queuing latency model, packet loss, spine link utilisation, CRC error counters
- **Storage** --- NVMe-oF shared storage per rack: read/write IOPS, throughput, latency with queue depth modelling (Little's Law), capacity tracking, drive health/wear lifecycle
- **Cooling** --- CRAC units (supply/return air temps, fan speed, airflow, CHW temps), chilled water loop, cooling tower (wet-bulb approach, heat rejection), COP efficiency varying with ambient conditions, pump power
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

The dashboard provides eight tabs: Overview (fleet health, power, workload, GPU/network/storage/cooling summaries), Infrastructure (rack heatmap, node telemetry), Fleet (time-series charts for power, PUE, temperatures), GPU (per-GPU detail tables), Network (ToR switch and spine link telemetry), Storage (NVMe shelf IOPS, latency, capacity), Cooling (CRAC units, cooling tower, COP), and Carbon (emissions and cost tracking with time-series charts).

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

### GPU fleet health and per-server detail
```bash
curl http://127.0.0.1:8000/gpu
curl http://127.0.0.1:8000/gpu/rack-0-srv-0
```

### Network fabric and storage I/O
```bash
curl http://127.0.0.1:8000/network
curl http://127.0.0.1:8000/storage
```

### Cooling plant state (CRAC units, CHW, cooling tower)
```bash
curl http://127.0.0.1:8000/cooling
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
