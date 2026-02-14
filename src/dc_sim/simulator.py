"""Main simulation orchestration: facility, failures, telemetry, audit."""

import threading
import time

from dc_sim.clock import SimulationClock
from dc_sim.config import SimConfig
from dc_sim.failures import FailureEngine
from dc_sim.models.facility import Facility
from dc_sim.models.workload import WorkloadQueue
from dc_sim.telemetry import AuditLog, TelemetryBuffer


class Simulator:
    """Orchestrates the facility, failure engine, telemetry, and audit log."""

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self.clock = SimulationClock(
            tick_interval_s=self.config.clock.tick_interval_s,
            realtime_factor=self.config.clock.realtime_factor,
        )
        rng = __import__("numpy").random.default_rng(self.config.rng_seed)
        self.workload_queue = WorkloadQueue(self.config, rng=rng)
        self.facility = Facility(
            config=self.config,
            clock=self.clock,
            workload_queue=self.workload_queue,
            rng_seed=self.config.rng_seed,
        )
        self.failure_engine = FailureEngine(self.config, rng_seed=self.config.rng_seed)
        self.telemetry = TelemetryBuffer(maxlen=1000)
        self.audit_log = AuditLog(maxlen=5000)
        self._running = False
        self._run_thread: threading.Thread | None = None
        self._tick_interval_real_s: float = 0.5  # Real seconds between ticks when running

    def _run_loop(self) -> None:
        """Background thread: tick repeatedly while _running."""
        while self._running:
            self.tick(1)
            time.sleep(self._tick_interval_real_s)

    def start_continuous(self, tick_interval_real_s: float = 0.5) -> bool:
        """Start continuous simulation. Returns False if already running."""
        if self._running:
            return False
        self._tick_interval_real_s = tick_interval_real_s
        self._running = True
        self._run_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._run_thread.start()
        return True

    def stop_continuous(self) -> bool:
        """Stop continuous simulation. Returns False if not running."""
        if not self._running:
            return False
        self._running = False
        if self._run_thread:
            self._run_thread.join(timeout=2)
            self._run_thread = None
        return True

    @property
    def is_running(self) -> bool:
        return self._running

    def tick(self, n: int = 1) -> list:
        """Advance simulation by n ticks. Returns list of states."""
        states = []
        for _ in range(n):
            self.clock.tick(1)
            self.failure_engine.set_current_time(self.clock.current_time)

            # Random failure injection (may add network partitions)
            self.failure_engine.tick(self.clock.current_time, self.telemetry.get_latest())

            # Apply network partition: fail jobs on affected racks
            partition_racks = self.failure_engine.get_network_partition_racks()
            for rack_id in partition_racks:
                for job in list(self.workload_queue.running):
                    if job.assigned_servers and job.assigned_servers[0].startswith(f"rack-{rack_id}-"):
                        self.workload_queue.preempt_job(job.job_id, mark_as_failed=True)

            cooling = self.failure_engine.get_cooling_capacity_factors()
            # Adjust cooling by CRAC setpoint (lower setpoint = more cooling)
            default_setpoint = self.config.thermal.crac_setpoint_c
            for rack_id in range(self.config.facility.num_racks):
                sp = self.facility._crac_setpoints.get(rack_id)
                if sp is not None:
                    scale = 1.0 + (default_setpoint - sp) * 0.03
                    cooling[rack_id] = cooling.get(rack_id, 1.0) * max(0.8, min(1.2, scale))
            gpu_degraded = self.failure_engine.get_gpu_degraded_servers()
            server_max_util = {s: 0.3 for s in gpu_degraded} if gpu_degraded else None
            rack_mult = {}
            for rack_id in range(self.config.facility.num_racks):
                mult = self.failure_engine.get_pdu_spike_factor(rack_id)
                if mult != 1.0:
                    rack_mult[rack_id] = mult

            state = self.facility.step(
                cooling_capacity_factor=cooling,
                server_max_util_override=server_max_util,
                rack_power_multiplier=rack_mult if rack_mult else None,
                network_partition_racks=partition_racks if partition_racks else None,
            )
            self.telemetry.append(state)
            states.append(state)
        return states

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._running = False
        if self._run_thread:
            self._run_thread.join(timeout=2)
        self.clock = SimulationClock(
            tick_interval_s=self.config.clock.tick_interval_s,
            realtime_factor=self.config.clock.realtime_factor,
        )
        rng = __import__("numpy").random.default_rng(self.config.rng_seed)
        self.workload_queue = WorkloadQueue(self.config, rng=rng)
        self.facility = Facility(
            config=self.config,
            clock=self.clock,
            workload_queue=self.workload_queue,
            rng_seed=self.config.rng_seed,
        )
        self.failure_engine = FailureEngine(self.config, rng_seed=self.config.rng_seed)
        self.telemetry = TelemetryBuffer(maxlen=1000)
        self.audit_log = AuditLog(maxlen=5000)
