# Data Centre Simulator: How It Works & How to Use It

This document explains the theory behind the simulator, what each metric means, and how to use the system effectively.

---

## What Is It?

The DC Simulator is a **discrete time-step simulation** of a GPU data centre. It models ten interconnected systems:

- **Thermal** --- rack inlet/outlet temperatures, humidity, hot-aisle recirculation, and CRAC cooling
- **Power** --- per-server and per-rack power draw with non-linear GPU power curves and dynamic PUE
- **GPU** --- per-GPU telemetry (temperature, utilisation, memory, clocks, ECC errors, PCIe/NVLink bandwidth)
- **Network** --- leaf-spine fabric with per-rack ToR switches, east-west/north-south traffic, RDMA, latency modelling
- **Storage** --- NVMe-oF shared storage with IOPS, throughput, latency, capacity tracking, and drive wear
- **Cooling** --- CRAC units, chilled water loop, cooling tower, COP efficiency, pump power
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
 7. GPU: compute per-GPU telemetry (temps, clocks, memory, ECC, bandwidth)
 8. Network: compute per-rack traffic, RDMA, latency, spine utilisation
 9. Storage: compute per-rack I/O (IOPS, throughput, latency, capacity, drive wear)
10. Cooling: compute CRAC unit states, chilled water loop, cooling tower, COP
11. Carbon: compute carbon emissions and electricity cost for this tick
12. Append full state to telemetry ring buffer
```

The ordering matters: **Workload -> Power -> Thermal -> GPU -> Network -> Storage -> Cooling -> Carbon**. Jobs set GPU utilisation, which determines power draw, which determines heat generation, which determines temperature. GPU telemetry then uses per-rack temperatures and workload utilisation to derive per-GPU metrics. Network and storage I/O are driven by the active workload mix. Cooling state is derived from the thermal load. Carbon accumulates from total facility power (IT + cooling). Thermal throttling (inlet >= 40C) then caps GPU utilisation at 50% on the *next* tick, creating a stabilising feedback loop.

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

### GPU Model

The GPU model produces per-GPU telemetry for every GPU in the facility (default: 128 GPUs across 32 servers). It simulates NVIDIA H100-class accelerators with realistic thermal, power, and performance characteristics.

**How it works:** Each tick, the GPU model iterates over every server and every GPU within that server. For each GPU, it uses the server's rack-level thermal state and workload utilisation to derive:

- **Junction temperature:** Base temperature is the rack inlet temp, plus a non-linear rise with utilisation (idle GPUs sit ~5C above ambient; fully loaded GPUs reach 80-90C). A small random offset per GPU simulates manufacturing variation.
- **HBM memory temperature:** Typically 5-10C above junction temperature, since HBM stacks sit directly on the GPU die.
- **Clock speeds:** The SM clock starts at the base clock (1095 MHz for H100) and boosts up to 1980 MHz. Boost headroom decreases as temperature rises. Above 83C junction temp, the GPU enters thermal throttling and clocks drop to ~60% of base.
- **Power draw:** Follows the same non-linear curve as the power model (idle ~5% TDP, superlinear rise with utilisation), then adjusted by the actual clock ratio.
- **Memory allocation:** Depends on job type. Training jobs allocate 60-95% of HBM (80 GiB total); inference jobs allocate 20-50%; idle GPUs still use ~5% for driver overhead.
- **ECC errors:** Single-bit errors (SBE) and double-bit errors (DBE) accumulate probabilistically. The base rate is very low (~1e-7 per tick), but increases 3x when junction temp exceeds 85C. SBEs are correctable; DBEs indicate potential hardware failure.
- **PCIe bandwidth:** TX/RX bandwidth scales with utilisation. Training jobs generate heavy NVLink traffic (up to 450 Gbps bidirectional); inference jobs use mainly PCIe for host communication.
- **Fan speed:** Scales linearly from 30% (idle) to 100% (thermal throttle).

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
| **pcie_tx_gbps** | Gbps | PCIe Gen5 x16 transmit bandwidth (max ~64 Gbps). |
| **pcie_rx_gbps** | Gbps | PCIe Gen5 x16 receive bandwidth. |
| **nvlink_tx_gbps** | Gbps | NVLink transmit bandwidth (max ~450 Gbps). Heavy during multi-GPU training. |
| **nvlink_rx_gbps** | Gbps | NVLink receive bandwidth. |
| **fan_speed_pct** | % | GPU fan speed. 30% at idle, 100% at thermal throttle. |
| **thermal_throttle** | bool | Whether the GPU is thermally throttling (junction > 83C). |
| **power_throttle** | bool | Whether the GPU is power-limited (drawing at TDP). |

**Aggregate metrics (per server and facility):**

| Metric | Unit | What it means |
|---|---|---|
| **total_gpus** | count | Total GPUs in the facility (default 128). |
| **healthy_gpus** | count | GPUs not in thermal or power throttle. |
| **throttled_gpus** | count | GPUs currently thermal-throttling. |
| **ecc_error_gpus** | count | GPUs with any double-bit ECC error (potential RMA candidates). |
| **avg_gpu_temp_c** | C | Mean junction temperature across all GPUs. |
| **avg_sm_util_pct** | % | Mean SM utilisation across all GPUs. |
| **total_gpu_mem_used_mib** | MiB | Total GPU memory in use facility-wide. |
| **total_gpu_mem_total_mib** | MiB | Total GPU memory available facility-wide. |

### Network Model

The network model simulates a **leaf-spine data centre fabric** --- the standard topology used in modern GPU clusters. Each rack has a Top-of-Rack (ToR) switch connecting to spine switches that provide inter-rack connectivity.

**Topology:** Each rack's ToR switch has a fixed number of ports (default 48) split between server downlinks and spine uplinks. Spine switches interconnect all racks, and any rack can reach any other rack in exactly two hops (server -> ToR -> spine -> ToR -> server).

**Traffic classes:**

| Traffic type | Direction | Driven by | Typical bandwidth |
|---|---|---|---|
| **East-west** | Rack-to-rack | Multi-rack training jobs (gradient sync) | 20-40 Gbps per server |
| **North-south** | DC-to-external | Inference serving (API traffic) | 5-10 Gbps per server |
| **RDMA/RoCE** | GPU-to-GPU | NVLink/InfiniBand for collective comms | 40 Gbps per server (training) |
| **Intra-rack** | Within rack | Local I/O, NVMe-oF storage | 5-15 Gbps per server |

**Latency model:** The simulator uses an **M/M/1 queuing model** to compute per-rack fabric latency:

```
latency = base_latency / (1 - utilisation)
```

Where `base_latency` is the unloaded fabric latency (~2 us for intra-rack, ~5 us for inter-rack). As ToR utilisation approaches 100%, latency grows non-linearly toward infinity --- this models real-world congestion. P99 latency is estimated as 3x average latency.

**Packet loss:** Below 80% ToR utilisation, packet loss is effectively zero. Above 80%, loss increases quadratically:

```
loss_pct = 0.5 * ((util - 0.8) / 0.2)^2    (for util > 80%)
```

This models TCP buffer overflows and switch queue drops under congestion.

**Key metrics (per rack):**

| Metric | Unit | What it means |
|---|---|---|
| **ingress_gbps** | Gbps | Total traffic entering the rack from the fabric. |
| **egress_gbps** | Gbps | Total traffic leaving the rack to the fabric. |
| **intra_rack_gbps** | Gbps | Traffic staying within the rack (server-to-server, storage). |
| **tor_utilisation_pct** | % | ToR switch port utilisation. >80% causes congestion; >95% causes packet loss. |
| **avg_latency_us** | us | Average fabric latency for packets transiting this rack's ToR. |
| **p99_latency_us** | us | 99th percentile latency (estimated as ~3x average). Key SLA metric for inference. |
| **packet_loss_pct** | % | Percentage of packets dropped due to congestion. >0.1% is a problem for RDMA workloads. |
| **rdma_tx_gbps** | Gbps | RDMA transmit bandwidth (GPU-to-GPU traffic). High during distributed training. |
| **rdma_rx_gbps** | Gbps | RDMA receive bandwidth. |
| **crc_errors** | count | Cumulative CRC errors on the ToR switch. Rising count indicates cable or optic degradation. |
| **active_ports** | count | Number of ToR switch ports currently active. |
| **total_ports** | count | Total ToR switch ports (default 48). |

**Aggregate metrics (facility-wide):**

| Metric | Unit | What it means |
|---|---|---|
| **total_east_west_gbps** | Gbps | Total inter-rack (lateral) traffic across the fabric. |
| **total_north_south_gbps** | Gbps | Total external-facing traffic. |
| **total_rdma_gbps** | Gbps | Total RDMA/RoCE traffic for GPU collective communications. |
| **avg_fabric_latency_us** | us | Mean fabric latency across all racks. |
| **total_packet_loss_pct** | % | Facility-wide average packet loss. |
| **total_crc_errors** | count | Sum of CRC errors across all ToR switches. |

**Spine link metrics:**

| Metric | Unit | What it means |
|---|---|---|
| **src_rack_id / dst_rack_id** | id | The two racks connected by this spine link. |
| **bandwidth_gbps** | Gbps | Current traffic on this spine link. |
| **utilisation_pct** | % | What fraction of the link's capacity (100 Gbps) is in use. |
| **latency_us** | us | Current latency on this link (increases with utilisation). |

### Storage Model

The storage model simulates **NVMe-oF (NVMe over Fabrics) shared storage** --- the standard architecture for AI/ML clusters where each rack has a local NVMe shelf that provides high-performance block storage for training data, model checkpoints, and inference caches.

**How it works:** Each rack has a shared NVMe storage shelf (default 100 TB capacity, ~800K max IOPS). The I/O profile is driven by the active workload mix on that rack:

| Job type | Read IOPS | Write IOPS | Read throughput | Write throughput | Pattern |
|---|---|---|---|---|---|
| **Training** | 15K/server | 3K/server | 6 Gbps | 1.5 Gbps | Large sequential reads (dataset), periodic checkpoint writes |
| **Inference** | 5K/server | 500/server | 1 Gbps | 0.1 Gbps | Small random reads (model weights, KV cache) |
| **Batch** | 20K/server | 8K/server | 8 Gbps | 3 Gbps | Heavy mixed I/O (ETL, data processing) |

**Latency model:** Storage latency increases with queue depth, following **Little's Law**:

```
base_read_latency = 80 us    (NVMe baseline)
base_write_latency = 20 us   (write-back cache)
queue_depth = total_iops * (avg_latency / 1e6)
actual_latency = base_latency * (1 + 0.5 * queue_depth / max_queue_depth)
```

As the storage shelf fills with concurrent I/O requests, latency rises. P99 read latency is estimated as 3x average.

**Drive wear and capacity:**

- Each rack's NVMe shelf has a write endurance of ~100 PB (petabytes). Cumulative writes are tracked across the simulation.
- **Drive health** is calculated as `100 * (1 - cumulative_writes / endurance)`. Below 90% health is a warning; below 70% indicates the drives are approaching end-of-life.
- Storage used grows slowly as jobs write checkpoints and logs (~1 GB per tick per active server).

**Key metrics (per rack):**

| Metric | Unit | What it means |
|---|---|---|
| **read_iops** | ops/s | Current read I/O operations per second. |
| **write_iops** | ops/s | Current write I/O operations per second. |
| **total_iops** | ops/s | Combined read + write IOPS. |
| **max_iops** | ops/s | Maximum IOPS the NVMe shelf can sustain (default 800K). |
| **read_throughput_gbps** | Gbps | Read data throughput. |
| **write_throughput_gbps** | Gbps | Write data throughput. |
| **avg_read_latency_us** | us | Average read latency. Baseline ~80us; rises with queue depth. |
| **avg_write_latency_us** | us | Average write latency. Baseline ~20us (write-back cache). |
| **p99_read_latency_us** | us | 99th percentile read latency (~3x average). |
| **used_tb** | TB | Storage capacity currently in use. |
| **total_tb** | TB | Total storage capacity per rack (default 100 TB). |
| **utilisation_pct** | % | Percentage of storage capacity used. |
| **drive_health_pct** | % | Remaining drive endurance. 100% = new; <70% = nearing end-of-life. |
| **queue_depth** | count | Current I/O queue depth (concurrent requests). High QD increases latency. |

**Aggregate metrics (facility-wide):**

| Metric | Unit | What it means |
|---|---|---|
| **total_read_iops** | ops/s | Sum of read IOPS across all racks. |
| **total_write_iops** | ops/s | Sum of write IOPS across all racks. |
| **total_read_throughput_gbps** | Gbps | Sum of read throughput across all racks. |
| **total_write_throughput_gbps** | Gbps | Sum of write throughput across all racks. |
| **total_used_tb** | TB | Total storage used facility-wide. |
| **total_capacity_tb** | TB | Total storage capacity facility-wide. |
| **avg_read_latency_us** | us | Mean read latency across all racks. |
| **avg_write_latency_us** | us | Mean write latency across all racks. |

### Cooling Model

The cooling model simulates the **mechanical cooling plant** that removes heat from the data centre. It models three layers: CRAC (Computer Room Air Conditioning) units, a chilled water (CHW) loop, and an outdoor cooling tower.

**Architecture:**

```
[Servers] --> hot air --> [CRAC units] --> chilled water --> [CHW plant] --> [Cooling tower] --> outdoor air
```

1. **CRAC units** (default 2) sit on the data hall floor. They draw in hot return air from the hot aisle, pass it over chilled water coils, and blow cold supply air into the raised floor plenum. CRAC-0 serves racks 0-3; CRAC-1 serves racks 4-7.

2. **Chilled water (CHW) loop** circulates water between the CRAC units and the central chiller plant. Supply water enters the CRACs at ~7C and returns at ~12C (the delta-T depends on heat load).

3. **Cooling tower** rejects the collected heat to the outdoor atmosphere. Its efficiency depends on the outdoor wet-bulb temperature.

**COP (Coefficient of Performance):** COP is the ratio of cooling output to electrical input --- it measures cooling efficiency:

```
COP = cooling_output_kw / cooling_power_kw
```

A COP of 4.0 means for every 1 kW of electricity consumed by the cooling plant, 4 kW of heat is removed. COP varies with conditions:

- **Best case (~6.0):** Cool ambient, low load, cooling tower highly effective
- **Typical (~4.0-4.5):** Moderate ambient and load
- **Worst case (~2.0):** Hot ambient (>35C), high load, cooling tower struggling

The simulator derives COP from outdoor conditions: `COP = base_cop * (1 - 0.02 * max(0, ambient - 22)) * (1 + 0.1 * max(0, 22 - ambient))`, clamped between 2.0 and 6.0.

**Key metrics (per CRAC unit):**

| Metric | Unit | What it means |
|---|---|---|
| **unit_id** | id | CRAC unit identifier (0 or 1). |
| **supply_air_temp_c** | C | Cold air temperature blown into the raised floor. Target: ~15C. |
| **return_air_temp_c** | C | Hot air temperature returning from the hot aisle. Typically 30-45C under load. |
| **fan_speed_pct** | % | CRAC fan speed. Scales with cooling demand. 100% = maximum airflow. |
| **airflow_cfm** | CFM | Volume of air moved per minute. Maximum ~12000 CFM per CRAC. |
| **chw_supply_temp_c** | C | Chilled water entering the CRAC coils. ~7C from the plant. |
| **chw_return_temp_c** | C | Chilled water leaving the CRAC coils (warmed by absorbing rack heat). |
| **chw_flow_rate_lps** | L/s | Chilled water flow rate through this CRAC. |
| **cooling_output_kw** | kW | Actual heat being removed by this CRAC right now. |
| **cooling_capacity_kw** | kW | Maximum heat this CRAC can remove. |
| **load_pct** | % | Fraction of cooling capacity in use (output / capacity). |
| **operational** | bool | Whether this CRAC is operational. `false` during a CRAC failure. |
| **fault_code** | string | Empty if healthy; contains fault description during failures. |

**Cooling tower metrics:**

| Metric | Unit | What it means |
|---|---|---|
| **wet_bulb_temp_c** | C | Outdoor wet-bulb temperature. Lower = better cooling tower performance. |
| **condenser_supply_temp_c** | C | Condenser water entering the cooling tower (~28C). |
| **condenser_return_temp_c** | C | Condenser water returning from the tower (cooled, ~23-25C). |
| **approach_temp_c** | C | Difference between condenser return and wet-bulb. Lower approach = more efficient tower. |
| **fan_speed_pct** | % | Tower fan speed. Scales with heat rejection demand. |
| **heat_rejection_kw** | kW | Total heat being rejected to the atmosphere. |

**Facility-wide cooling metrics:**

| Metric | Unit | What it means |
|---|---|---|
| **cop** | ratio | Coefficient of Performance for the entire cooling plant. |
| **total_cooling_output_kw** | kW | Total heat being removed from the data hall. |
| **total_cooling_capacity_kw** | kW | Maximum cooling the plant can deliver. |
| **cooling_load_pct** | % | How much of total cooling capacity is in use. >90% is critical. |
| **cooling_power_kw** | kW | Total electrical power consumed by cooling (CRACs + tower + pumps). |
| **chw_plant_supply_temp_c** | C | Chilled water supply temperature from the plant. |
| **chw_plant_return_temp_c** | C | Chilled water return temperature to the plant. |
| **chw_plant_delta_t_c** | C | Temperature difference across the CHW loop (return - supply). |
| **pump_power_kw** | kW | Electrical power consumed by CHW circulation pumps. |
| **pump_flow_rate_lps** | L/s | Total chilled water flow rate. |

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
| GET | `/status` | Full facility state snapshot (thermal + power + gpu + network + storage + cooling + carbon + workload) |
| GET | `/thermal` | All rack thermal states + ambient temp + humidity |
| GET | `/thermal/{rack_id}` | Single rack thermal state |
| GET | `/power` | Facility power summary (IT, total, PUE, headroom) |
| GET | `/power/{rack_id}` | Single rack power state |
| GET | `/gpu` | Facility-wide GPU summary (total, healthy, throttled, ECC errors, avg temp/util, memory) |
| GET | `/gpu/{server_id}` | Full per-GPU telemetry for a specific server (22 fields per GPU) |
| GET | `/network` | Facility-wide network summary + per-rack ToR switch data + spine link data |
| GET | `/network/{rack_id}` | Single rack network state (ToR switch telemetry) |
| GET | `/storage` | Facility-wide storage summary + per-rack NVMe shelf data |
| GET | `/storage/{rack_id}` | Single rack storage state (IOPS, throughput, latency, capacity, health) |
| GET | `/cooling` | Full cooling system state (COP, CHW loop, CRAC units array, cooling tower) |
| GET | `/carbon` | Carbon intensity, price, cumulative emissions and cost |
| GET | `/workload/queue` | Pending jobs (with job type) |
| GET | `/workload/running` | Running jobs (with job type and server assignments) |
| GET | `/workload/completed?last_n=10` | Recent completed jobs |
| GET | `/workload/sla_violations` | Jobs that missed their SLA |
| GET | `/failures/active` | Currently active failures |
| GET | `/telemetry/history?last_n=60` | Last N ticks for time-series analysis |
| GET | `/audit?last_n=50` | Recent audit log entries |

**GPU endpoint details:**

`GET /gpu` returns aggregate metrics:
```json
{
  "total_gpus": 128,
  "healthy_gpus": 124,
  "throttled_gpus": 4,
  "ecc_error_gpus": 0,
  "avg_gpu_temp_c": 52.3,
  "avg_sm_util_pct": 68.5,
  "total_gpu_mem_used_mib": 5242880,
  "total_gpu_mem_total_mib": 10485760
}
```

`GET /gpu/{server_id}` (e.g. `/gpu/rack-0-srv-0`) returns per-GPU telemetry:
```json
{
  "server_id": "rack-0-srv-0",
  "rack_id": 0,
  "total_gpu_power_w": 1020.0,
  "avg_gpu_temp_c": 71.2,
  "gpus": [
    {
      "gpu_id": "rack-0-srv-0-gpu-0",
      "sm_utilisation_pct": 92.0,
      "mem_utilisation_pct": 85.0,
      "gpu_temp_c": 73.5,
      "mem_temp_c": 81.2,
      "power_draw_w": 265.0,
      "sm_clock_mhz": 1830,
      "mem_clock_mhz": 1593,
      "mem_used_mib": 73728,
      "mem_total_mib": 81920,
      "ecc_sbe_count": 0,
      "ecc_dbe_count": 0,
      "pcie_tx_gbps": 12.5,
      "pcie_rx_gbps": 8.3,
      "nvlink_tx_gbps": 180.0,
      "nvlink_rx_gbps": 175.0,
      "fan_speed_pct": 72.0,
      "thermal_throttle": false,
      "power_throttle": false
    }
  ]
}
```

**Network endpoint details:**

`GET /network` returns facility summary, per-rack ToR data, and spine links:
```json
{
  "total_east_west_gbps": 156.2,
  "total_north_south_gbps": 42.0,
  "total_rdma_gbps": 120.5,
  "avg_fabric_latency_us": 8.3,
  "total_packet_loss_pct": 0.001,
  "total_crc_errors": 0,
  "racks": [
    {
      "rack_id": 0,
      "ingress_gbps": 22.5,
      "egress_gbps": 18.3,
      "intra_rack_gbps": 8.2,
      "tor_utilisation_pct": 45.0,
      "avg_latency_us": 6.5,
      "p99_latency_us": 19.5,
      "packet_loss_pct": 0.0,
      "rdma_tx_gbps": 15.0,
      "rdma_rx_gbps": 14.8,
      "active_ports": 32,
      "total_ports": 48
    }
  ],
  "spine_links": [
    {"src_rack_id": 0, "dst_rack_id": 1, "bandwidth_gbps": 12.5, "utilisation_pct": 12.5, "latency_us": 5.7}
  ]
}
```

**Storage endpoint details:**

`GET /storage` returns facility summary and per-rack NVMe shelf data:
```json
{
  "total_read_iops": 320000,
  "total_write_iops": 85000,
  "total_read_throughput_gbps": 128.0,
  "total_write_throughput_gbps": 34.0,
  "total_used_tb": 245.3,
  "total_capacity_tb": 800.0,
  "avg_read_latency_us": 95.0,
  "avg_write_latency_us": 25.0,
  "racks": [
    {
      "rack_id": 0,
      "read_iops": 45000,
      "write_iops": 12000,
      "total_iops": 57000,
      "max_iops": 800000,
      "read_throughput_gbps": 18.0,
      "write_throughput_gbps": 4.8,
      "avg_read_latency_us": 88.0,
      "avg_write_latency_us": 22.0,
      "p99_read_latency_us": 264.0,
      "used_tb": 32.5,
      "total_tb": 100.0,
      "utilisation_pct": 32.5,
      "drive_health_pct": 99.8,
      "queue_depth": 12
    }
  ]
}
```

**Cooling endpoint details:**

`GET /cooling` returns CRAC units, CHW loop, and cooling tower state:
```json
{
  "total_cooling_output_kw": 45.0,
  "total_cooling_capacity_kw": 100.0,
  "cooling_load_pct": 45.0,
  "cop": 4.2,
  "cooling_power_kw": 10.7,
  "chw_plant_supply_temp_c": 7.0,
  "chw_plant_return_temp_c": 12.3,
  "chw_plant_delta_t_c": 5.3,
  "pump_power_kw": 2.1,
  "pump_flow_rate_lps": 18.5,
  "cooling_tower": {
    "condenser_supply_temp_c": 28.0,
    "condenser_return_temp_c": 23.5,
    "wet_bulb_temp_c": 18.0,
    "approach_temp_c": 5.5,
    "fan_speed_pct": 42.0,
    "heat_rejection_kw": 55.7
  },
  "crac_units": [
    {
      "unit_id": 0,
      "supply_air_temp_c": 15.2,
      "return_air_temp_c": 32.5,
      "fan_speed_pct": 65.0,
      "airflow_cfm": 7800,
      "chw_supply_temp_c": 7.0,
      "chw_return_temp_c": 13.1,
      "chw_flow_rate_lps": 9.5,
      "cooling_output_kw": 24.0,
      "cooling_capacity_kw": 50.0,
      "load_pct": 48.0,
      "operational": true,
      "fault_code": ""
    }
  ]
}
```

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

# Full status (thermal + power + gpu + network + storage + cooling + carbon + workload)
curl http://127.0.0.1:8000/status

# Just carbon and cost
curl http://127.0.0.1:8000/carbon

# GPU fleet health and per-server detail
curl http://127.0.0.1:8000/gpu
curl http://127.0.0.1:8000/gpu/rack-0-srv-0

# Network fabric and per-rack ToR switch
curl http://127.0.0.1:8000/network
curl http://127.0.0.1:8000/network/0

# Storage I/O and per-rack NVMe shelf
curl http://127.0.0.1:8000/storage
curl http://127.0.0.1:8000/storage/0

# Cooling plant (CRAC units, CHW loop, cooling tower)
curl http://127.0.0.1:8000/cooling
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

# Thermal and power
print(f"Rack 0 inlet:   {state.thermal.racks[0].inlet_temp_c:.1f}C")
print(f"Rack 0 humidity: {state.thermal.racks[0].humidity_pct:.0f}% RH")
print(f"IT power:        {state.power.it_power_kw:.1f} kW")
print(f"PUE:             {state.power.pue:.2f}")

# GPU telemetry
print(f"GPU fleet:       {state.gpu.healthy_gpus}/{state.gpu.total_gpus} healthy")
print(f"Avg GPU temp:    {state.gpu.avg_gpu_temp_c:.1f}C")
print(f"Avg SM util:     {state.gpu.avg_sm_util_pct:.1f}%")
# Per-GPU detail
gpu0 = state.gpu.servers[0].gpus[0]
print(f"GPU-0 clock:     {gpu0.sm_clock_mhz} MHz, ECC SBE: {gpu0.ecc_sbe_count}")

# Network
print(f"East-west:       {state.network.total_east_west_gbps:.1f} Gbps")
print(f"Fabric latency:  {state.network.avg_fabric_latency_us:.1f} us")

# Storage
print(f"Read IOPS:       {state.storage.total_read_iops:.0f}")
print(f"Storage used:    {state.storage.total_used_tb:.1f} / {state.storage.total_capacity_tb:.0f} TB")

# Cooling
print(f"COP:             {state.cooling.cop:.2f}")
print(f"Cooling load:    {state.cooling.cooling_load_pct:.0f}%")

# Carbon and cost
print(f"Carbon emitted:  {state.carbon.cumulative_carbon_kg:.1f} kg CO2")
print(f"Electricity cost: {state.carbon.cumulative_cost_gbp:.2f} GBP")

# Workload
print(f"Running jobs:    {state.workload_running}")
print(f"SLA violations:  {state.sla_violations}")
```

---

## Designed For

This simulator is built to support an **LLM-based agentic system** that will:

1. **Observe** telemetry via the GET endpoints (thermal, power, GPU, network, storage, cooling, carbon, workload, failures)
2. **Reason** about the current state (is anything overheating? are GPUs throttling? is network congested? is carbon high? are SLAs at risk? are drives wearing out?)
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

These objectives sometimes conflict --- for example, running all jobs immediately minimises SLA violations but maximises cost and carbon. A good agent learns to defer flexible workloads to low-cost, low-carbon periods while keeping latency-sensitive inference jobs running at all times. The rich telemetry from GPU, network, storage, and cooling systems provides the observability needed for fine-grained optimisation decisions.

The same API can be used for manual testing, dashboards, automated evaluation scenarios, or multi-agent coordination experiments.
