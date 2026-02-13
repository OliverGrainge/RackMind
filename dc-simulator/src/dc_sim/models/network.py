"""Network traffic model: per-rack and inter-rack bandwidth, latency, packet loss.

Simulates a realistic leaf-spine data centre fabric with:
- Per-rack ToR (Top-of-Rack) switch metrics
- Inter-rack spine fabric bandwidth
- East-west (server-to-server) vs north-south (external) traffic
- Latency modelling with congestion effects
- Packet loss under high load
- RDMA/RoCE traffic for GPU-to-GPU communication
"""

import math
from dataclasses import dataclass, field

from dc_sim.config import SimConfig


@dataclass
class RackNetworkState:
    """Network telemetry for a single rack's ToR switch."""

    rack_id: int
    # Throughput (Gbps)
    ingress_gbps: float = 0.0  # Traffic entering the rack
    egress_gbps: float = 0.0  # Traffic leaving the rack
    intra_rack_gbps: float = 0.0  # East-west within rack
    # Capacity
    tor_link_capacity_gbps: float = 100.0  # Uplink to spine
    tor_utilisation_pct: float = 0.0  # Uplink utilisation
    # Latency
    avg_latency_us: float = 5.0  # Microseconds (intra-rack baseline)
    p99_latency_us: float = 15.0  # Tail latency
    # Errors
    packet_loss_pct: float = 0.0
    crc_errors: int = 0
    # RDMA traffic (GPU-to-GPU over RoCE v2)
    rdma_tx_gbps: float = 0.0
    rdma_rx_gbps: float = 0.0
    # Active ports
    active_ports: int = 0
    total_ports: int = 48  # Typical ToR switch


@dataclass
class SpineLinkState:
    """State of a spine fabric link between two racks."""

    src_rack_id: int
    dst_rack_id: int
    bandwidth_gbps: float = 0.0
    capacity_gbps: float = 400.0  # Spine link capacity
    utilisation_pct: float = 0.0
    latency_us: float = 2.0  # Extra hop latency


@dataclass
class FacilityNetworkState:
    """Facility-wide network telemetry."""

    racks: list[RackNetworkState] = field(default_factory=list)
    spine_links: list[SpineLinkState] = field(default_factory=list)
    # Aggregates
    total_east_west_gbps: float = 0.0  # Inter-server total
    total_north_south_gbps: float = 0.0  # External/internet total
    total_rdma_gbps: float = 0.0  # GPU fabric total
    avg_fabric_latency_us: float = 5.0  # Average cross-rack latency
    total_packet_loss_pct: float = 0.0
    total_crc_errors: int = 0


class NetworkModel:
    """Simulates data centre network traffic based on workload and GPU activity.

    Key behaviours:
    - Training jobs generate heavy RDMA traffic (AllReduce gradient sync)
    - Inference jobs generate moderate north-south traffic (client requests)
    - Batch jobs generate storage I/O traffic
    - Latency increases non-linearly with utilisation (queuing theory)
    - Packet loss emerges at high congestion (>80% utilisation)
    - Inter-rack traffic depends on multi-rack job placements
    """

    # Physical constants
    BASE_INTRA_RACK_LATENCY_US = 2.0  # Switch hop latency
    BASE_INTER_RACK_LATENCY_US = 5.0  # Through spine
    TOR_UPLINK_GBPS = 100.0  # ToR uplink capacity
    SPINE_LINK_GBPS = 400.0  # Spine link capacity
    SERVER_NIC_GBPS = 100.0  # Per-server NIC (100GbE)
    PORTS_PER_TOR = 48

    # Traffic generation rates per server (at 100% GPU util)
    TRAINING_RDMA_GBPS_PER_SERVER = 40.0  # Heavy gradient sync
    INFERENCE_NS_GBPS_PER_SERVER = 8.0  # Client request/response
    BATCH_STORAGE_GBPS_PER_SERVER = 15.0  # Data loading
    IDLE_TRAFFIC_GBPS = 0.1  # Management, heartbeats

    def __init__(self, config: SimConfig, rng_seed: int = 42):
        self.config = config
        self._rng = __import__("numpy").random.default_rng(rng_seed + 400)
        self._crc_errors: dict[int, int] = {}  # Persistent per rack

    def step(
        self,
        server_gpu_utilisation: dict[str, float],
        running_jobs: list | None = None,
        network_partition_racks: set[int] | None = None,
        sim_time: float = 0.0,
    ) -> FacilityNetworkState:
        """Compute network state for current tick.

        Args:
            server_gpu_utilisation: server_id -> GPU util (0-1)
            running_jobs: list of running Job objects
            network_partition_racks: racks isolated by network failure
            sim_time: current simulation time
        """
        facility = self.config.facility
        partition_racks = network_partition_racks or set()

        # Build per-server job type map
        server_job_types: dict[str, str] = {}
        # Track which racks have jobs spanning multiple racks (inter-rack traffic)
        rack_to_job_racks: dict[int, set[int]] = {r: set() for r in range(facility.num_racks)}

        if running_jobs:
            for job in running_jobs:
                job_racks = set()
                for srv in getattr(job, "assigned_servers", []):
                    server_job_types[srv] = getattr(job, "job_type", "batch")
                    # Extract rack ID from server_id
                    parts = srv.split("-")
                    if len(parts) >= 2:
                        r_id = int(parts[1])
                        job_racks.add(r_id)
                # Mark multi-rack jobs
                if len(job_racks) > 1:
                    for r_id in job_racks:
                        rack_to_job_racks[r_id].update(job_racks - {r_id})

        rack_states: list[RackNetworkState] = []
        spine_traffic: dict[tuple[int, int], float] = {}  # (src, dst) -> Gbps
        total_ew = 0.0
        total_ns = 0.0
        total_rdma = 0.0
        total_crc = 0

        for rack_id in range(facility.num_racks):
            if rack_id not in self._crc_errors:
                self._crc_errors[rack_id] = 0

            is_partitioned = rack_id in partition_racks

            rack_ingress = 0.0
            rack_egress = 0.0
            rack_intra = 0.0
            rack_rdma_tx = 0.0
            rack_rdma_rx = 0.0
            active_ports = 0

            for srv_idx in range(facility.servers_per_rack):
                server_id = f"rack-{rack_id}-srv-{srv_idx}"
                util = server_gpu_utilisation.get(server_id, 0.0)
                job_type = server_job_types.get(server_id, "idle")

                if is_partitioned:
                    # Partitioned racks have no network traffic
                    continue

                if util < 0.01:
                    # Idle server: just management traffic
                    rack_intra += self.IDLE_TRAFFIC_GBPS
                    active_ports += 1
                    continue

                active_ports += 1

                if job_type == "training":
                    # Training: heavy RDMA intra-rack, some inter-rack
                    rdma_bw = self.TRAINING_RDMA_GBPS_PER_SERVER * util
                    rack_rdma_tx += rdma_bw * 0.5
                    rack_rdma_rx += rdma_bw * 0.5
                    rack_intra += rdma_bw * 0.7  # 70% stays in rack
                    inter_rack_bw = rdma_bw * 0.3  # 30% crosses spine

                    # Distribute inter-rack traffic
                    partner_racks = rack_to_job_racks.get(rack_id, set())
                    if partner_racks:
                        per_partner = inter_rack_bw / len(partner_racks)
                        for partner in partner_racks:
                            key = (min(rack_id, partner), max(rack_id, partner))
                            spine_traffic[key] = spine_traffic.get(key, 0) + per_partner
                        rack_egress += inter_rack_bw

                    # Small amount of storage traffic for checkpointing
                    rack_egress += 2.0 * util
                    total_rdma += rdma_bw

                elif job_type == "inference":
                    # Inference: north-south client traffic
                    ns_bw = self.INFERENCE_NS_GBPS_PER_SERVER * util
                    rack_ingress += ns_bw * 0.6  # Requests in
                    rack_egress += ns_bw * 0.4  # Responses out
                    rack_intra += ns_bw * 0.2  # Internal routing
                    total_ns += ns_bw

                else:  # batch
                    # Batch: storage I/O dominant
                    storage_bw = self.BATCH_STORAGE_GBPS_PER_SERVER * util
                    rack_ingress += storage_bw * 0.7  # Data reads
                    rack_egress += storage_bw * 0.3  # Results writes
                    rack_intra += storage_bw * 0.1

            # Add noise
            noise_factor = 1.0 + self._rng.normal(0, 0.03)
            rack_ingress *= noise_factor
            rack_egress *= noise_factor

            total_ew += rack_intra
            total_traffic = rack_ingress + rack_egress
            tor_util = (total_traffic / self.TOR_UPLINK_GBPS) * 100.0 if not is_partitioned else 0

            # Latency model (M/M/1 queue approximation)
            rho = min(0.95, tor_util / 100.0)  # Utilisation factor
            if rho > 0.01:
                avg_latency = self.BASE_INTRA_RACK_LATENCY_US / (1.0 - rho)
                p99_latency = avg_latency * (1.0 + 2.3 * rho)  # Tail amplification
            else:
                avg_latency = self.BASE_INTRA_RACK_LATENCY_US
                p99_latency = self.BASE_INTRA_RACK_LATENCY_US * 1.5

            # Packet loss at high utilisation
            if rho > 0.8:
                pkt_loss = (rho - 0.8) * 5.0  # 0-1% loss at 80-100%
            else:
                pkt_loss = 0.0
            pkt_loss = min(2.0, pkt_loss)

            # CRC errors (rare, environmental)
            if self._rng.random() < 0.001:
                self._crc_errors[rack_id] += self._rng.integers(1, 5)
            crc = self._crc_errors[rack_id]
            total_crc += crc

            rack_states.append(RackNetworkState(
                rack_id=rack_id,
                ingress_gbps=round(rack_ingress, 2),
                egress_gbps=round(rack_egress, 2),
                intra_rack_gbps=round(rack_intra, 2),
                tor_link_capacity_gbps=self.TOR_UPLINK_GBPS,
                tor_utilisation_pct=round(min(100, tor_util), 1),
                avg_latency_us=round(avg_latency, 1),
                p99_latency_us=round(p99_latency, 1),
                packet_loss_pct=round(pkt_loss, 3),
                crc_errors=crc,
                rdma_tx_gbps=round(rack_rdma_tx, 2),
                rdma_rx_gbps=round(rack_rdma_rx, 2),
                active_ports=active_ports,
                total_ports=self.PORTS_PER_TOR,
            ))

        # Spine link states
        spine_links = []
        all_latencies = []
        for (src, dst), bw in spine_traffic.items():
            util_pct = (bw / self.SPINE_LINK_GBPS) * 100.0
            rho = min(0.95, util_pct / 100.0)
            link_latency = self.BASE_INTER_RACK_LATENCY_US / (1.0 - max(0.01, rho))
            all_latencies.append(link_latency)
            spine_links.append(SpineLinkState(
                src_rack_id=src,
                dst_rack_id=dst,
                bandwidth_gbps=round(bw, 2),
                capacity_gbps=self.SPINE_LINK_GBPS,
                utilisation_pct=round(min(100, util_pct), 1),
                latency_us=round(link_latency, 1),
            ))

        avg_fabric_latency = (sum(all_latencies) / len(all_latencies)) if all_latencies else self.BASE_INTER_RACK_LATENCY_US
        total_pkt_loss = sum(r.packet_loss_pct for r in rack_states) / max(1, len(rack_states))

        return FacilityNetworkState(
            racks=rack_states,
            spine_links=spine_links,
            total_east_west_gbps=round(total_ew, 2),
            total_north_south_gbps=round(total_ns, 2),
            total_rdma_gbps=round(total_rdma, 2),
            avg_fabric_latency_us=round(avg_fabric_latency, 1),
            total_packet_loss_pct=round(total_pkt_loss, 4),
            total_crc_errors=total_crc,
        )

    def reset(self) -> None:
        """Clear persistent counters."""
        self._crc_errors.clear()
