"""Failure injection engine for testing agent resilience."""

from dataclasses import dataclass, field
from enum import Enum
import uuid

from dc_sim.config import SimConfig


class FailureType(str, Enum):
    """Supported failure types."""

    CRAC_DEGRADED = "crac_degraded"
    CRAC_FAILURE = "crac_failure"
    GPU_DEGRADED = "gpu_degraded"
    PDU_SPIKE = "pdu_spike"
    NETWORK_PARTITION = "network_partition"


@dataclass
class ActiveFailure:
    """A currently active failure."""

    failure_id: str
    failure_type: str
    target: str  # e.g. "crac-0", "rack-3", "rack-0-srv-2"
    started_at: float
    duration_s: float | None  # None = until manually resolved (e.g. gpu_degraded)
    effect: str = ""


class FailureEngine:
    """Manages failure injection and effects."""

    def __init__(self, config: SimConfig, rng_seed: int = 42):
        self.config = config
        self.rng_seed = rng_seed
        self._rng = __import__("numpy").random.default_rng(rng_seed)
        self._active: dict[str, ActiveFailure] = {}
        self._crac_racks = self._compute_crac_racks()

    def _compute_crac_racks(self) -> dict[int, list[int]]:
        """Map CRAC unit ID to list of rack IDs it cools."""
        num_racks = self.config.facility.num_racks
        crac_units = self.config.thermal.crac_units
        racks_per_crac = max(1, num_racks // crac_units)
        result: dict[int, list[int]] = {}
        for crac_id in range(crac_units):
            start = crac_id * racks_per_crac
            end = min(start + racks_per_crac, num_racks)
            result[crac_id] = list(range(start, end))
        return result

    def tick(self, current_time: float, facility_state: object) -> list[ActiveFailure]:
        """
        Probabilistically inject failures. Return newly activated failures.
        ~0.5% per tick per rack so 4-hour run sees 2-3 failures.
        """
        newly_activated: list[ActiveFailure] = []
        num_racks = self.config.facility.num_racks
        prob_per_rack = 0.005

        if self._rng.random() < prob_per_rack * num_racks:
            rack_id = int(self._rng.integers(0, num_racks))
            failure_type_str = self._rng.choice(
                ["crac_degraded", "pdu_spike", "network_partition"]
            )
            if failure_type_str == "crac_degraded":
                crac_id = rack_id % max(1, len(self._crac_racks))
                crac_id = min(crac_id, len(self._crac_racks) - 1)
                target = f"crac-{crac_id}"
                duration = int(self._rng.integers(600, 1800))  # 10-30 min
            elif failure_type_str == "pdu_spike":
                target = f"rack-{rack_id}"
                duration = 300  # 5 min
            else:
                target = f"rack-{rack_id}"
                duration = 0  # Instant effect
            failures = self.inject(failure_type_str, target, duration)
            newly_activated.extend(failures)

        # Check for expired failures
        to_remove = []
        for fid, f in self._active.items():
            if f.duration_s is not None:
                if current_time - f.started_at >= f.duration_s:
                    to_remove.append(fid)
        for fid in to_remove:
            del self._active[fid]

        return newly_activated

    def inject(
        self,
        failure_type: str,
        target: str,
        duration_s: int | None = None,
    ) -> list[ActiveFailure]:
        """Manually inject a failure. Returns list of created failures."""
        failure_id = str(uuid.uuid4())
        current_time = getattr(self, "_current_time", 0.0)

        if failure_type == FailureType.CRAC_DEGRADED.value:
            if duration_s is None:
                duration_s = 1200  # 20 min default
            f = ActiveFailure(
                failure_id=failure_id,
                failure_type=failure_type,
                target=target,
                started_at=current_time,
                duration_s=float(duration_s),
                effect="50% cooling capacity",
            )
        elif failure_type == FailureType.CRAC_FAILURE.value:
            if duration_s is None:
                duration_s = 600  # 10 min default
            f = ActiveFailure(
                failure_id=failure_id,
                failure_type=failure_type,
                target=target,
                started_at=current_time,
                duration_s=float(duration_s),
                effect="0% cooling capacity",
            )
        elif failure_type == FailureType.GPU_DEGRADED.value:
            f = ActiveFailure(
                failure_id=failure_id,
                failure_type=failure_type,
                target=target,
                started_at=current_time,
                duration_s=None,
                effect="GPU stuck at 30% max util",
            )
        elif failure_type == FailureType.PDU_SPIKE.value:
            duration_s = duration_s or 300
            f = ActiveFailure(
                failure_id=failure_id,
                failure_type=failure_type,
                target=target,
                started_at=current_time,
                duration_s=float(duration_s),
                effect="+20% power draw",
            )
        elif failure_type == FailureType.NETWORK_PARTITION.value:
            f = ActiveFailure(
                failure_id=failure_id,
                failure_type=failure_type,
                target=target,
                started_at=current_time,
                duration_s=0.0,
                effect="Jobs on rack fail",
            )
        else:
            return []

        self._active[failure_id] = f
        return [f]

    def set_current_time(self, t: float) -> None:
        """Update current time for inject()."""
        self._current_time = t

    def get_cooling_capacity_factor(self, rack_id: int) -> float:
        """Get cooling factor for a rack (0.0-1.0) based on active CRAC failures."""
        factor = 1.0
        crac_units = self.config.thermal.crac_units
        racks_per_crac = max(1, self.config.facility.num_racks // crac_units)
        crac_id = rack_id // racks_per_crac
        crac_id = min(crac_id, crac_units - 1)

        for f in self._active.values():
            if f.failure_type == FailureType.CRAC_FAILURE.value and f.target == f"crac-{crac_id}":
                factor = 0.0
            elif f.failure_type == FailureType.CRAC_DEGRADED.value and f.target == f"crac-{crac_id}":
                factor = min(factor, 0.5)
        return factor

    def get_cooling_capacity_factors(self) -> dict[int, float]:
        """Get cooling factor for all racks."""
        result = {}
        for rack_id in range(self.config.facility.num_racks):
            result[rack_id] = self.get_cooling_capacity_factor(rack_id)
        return result

    def get_pdu_spike_factor(self, rack_id: int) -> float:
        """Get power spike multiplier for a rack (1.0 or 1.2)."""
        for f in self._active.values():
            if f.failure_type == FailureType.PDU_SPIKE.value and f.target == f"rack-{rack_id}":
                return 1.2
        return 1.0

    def get_network_partition_racks(self) -> set[int]:
        """Racks with active network partition (jobs should fail)."""
        result = set()
        for f in self._active.values():
            if f.failure_type == FailureType.NETWORK_PARTITION.value and f.target.startswith("rack-"):
                try:
                    result.add(int(f.target.split("-")[1]))
                except (IndexError, ValueError):
                    pass
        return result

    def get_gpu_degraded_servers(self) -> set[str]:
        """Servers with degraded GPU (30% max util)."""
        return {f.target for f in self._active.values() if f.failure_type == FailureType.GPU_DEGRADED.value}

    def get_active_failures(self) -> list[ActiveFailure]:
        """Return all active failures."""
        return list(self._active.values())

    def resolve(self, failure_id: str) -> bool:
        """Manually resolve a failure. Returns True if found and removed."""
        if failure_id in self._active:
            del self._active[failure_id]
            return True
        return False
