"""In-memory telemetry ringbuffer, audit log, and history queries."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from dc_sim.models.facility import FacilityState


def facility_state_to_dict(state: FacilityState) -> dict[str, Any]:
    """Serialize FacilityState to JSON-serialisable dict."""
    result: dict[str, Any] = {
        "current_time": state.current_time,
        "tick_count": state.tick_count,
        "workload_pending": state.workload_pending,
        "workload_running": state.workload_running,
        "workload_completed": state.workload_completed,
        "sla_violations": state.sla_violations,
    }
    result["thermal"] = {
        "racks": [
            {
                "rack_id": r.rack_id,
                "inlet_temp_c": r.inlet_temp_c,
                "outlet_temp_c": r.outlet_temp_c,
                "heat_generated_kw": r.heat_generated_kw,
                "throttled": r.throttled,
                "humidity_pct": r.humidity_pct,
                "delta_t_c": r.delta_t_c,
            }
            for r in state.thermal.racks
        ],
        "ambient_temp_c": state.thermal.ambient_temp_c,
        "avg_humidity_pct": state.thermal.avg_humidity_pct,
    }
    result["power"] = {
        "it_power_kw": state.power.it_power_kw,
        "total_power_kw": state.power.total_power_kw,
        "pue": state.power.pue,
        "headroom_kw": state.power.headroom_kw,
        "power_cap_exceeded": state.power.power_cap_exceeded,
        "racks": [
            {
                "rack_id": r.rack_id,
                "total_power_kw": r.total_power_kw,
                "pdu_utilisation_pct": r.pdu_utilisation_pct,
            }
            for r in state.power.racks
        ],
    }
    result["carbon"] = {
        "carbon_intensity_gco2_kwh": state.carbon.carbon_intensity_gco2_kwh,
        "carbon_rate_gco2_s": state.carbon.carbon_rate_gco2_s,
        "cumulative_carbon_kg": state.carbon.cumulative_carbon_kg,
        "electricity_price_gbp_kwh": state.carbon.electricity_price_gbp_kwh,
        "cost_rate_gbp_h": state.carbon.cost_rate_gbp_h,
        "cumulative_cost_gbp": state.carbon.cumulative_cost_gbp,
    }
    return result


@dataclass
class AuditEntry:
    """Record of an action taken (by agent or operator)."""

    timestamp: float
    action: str  # e.g. "migrate_workload", "adjust_cooling"
    params: dict[str, Any] = field(default_factory=dict)
    result: str = "ok"  # "ok" or error message
    source: str = "api"  # "api", "agent", "operator"


class AuditLog:
    """Append-only log of all actions taken on the simulator."""

    def __init__(self, maxlen: int = 5000):
        self._entries: deque[AuditEntry] = deque(maxlen=maxlen)

    def record(
        self,
        timestamp: float,
        action: str,
        params: dict[str, Any] | None = None,
        result: str = "ok",
        source: str = "api",
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=timestamp,
            action=action,
            params=params or {},
            result=result,
            source=source,
        )
        self._entries.append(entry)
        return entry

    def get_last_n(self, n: int = 50) -> list[dict[str, Any]]:
        entries = list(self._entries)[-n:]
        return [
            {
                "timestamp": e.timestamp,
                "action": e.action,
                "params": e.params,
                "result": e.result,
                "source": e.source,
            }
            for e in entries
        ]

    def get_all(self) -> list[dict[str, Any]]:
        return self.get_last_n(len(self._entries))

    def clear(self) -> None:
        self._entries.clear()


class TelemetryBuffer:
    """Ring buffer of timestamped facility state snapshots."""

    def __init__(self, maxlen: int = 1000, log_path: str | None = None):
        self._buffer: deque[tuple[float, FacilityState]] = deque(maxlen=maxlen)
        self._log_path = log_path

    def append(self, state: FacilityState) -> None:
        """Append a state snapshot."""
        self._buffer.append((state.current_time, state))
        if self._log_path:
            self._write_to_file(state)

    def _write_to_file(self, state: FacilityState) -> None:
        """Append state to JSONL file."""
        import json
        data = facility_state_to_dict(state)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def get_latest(self) -> FacilityState | None:
        """Return the most recent state."""
        if not self._buffer:
            return None
        return self._buffer[-1][1]

    def get_last_n(self, n: int) -> list[tuple[float, FacilityState]]:
        """Return the last n (time, state) pairs."""
        return list(self._buffer)[-n:]

    def get_range(
        self, start_time: float, end_time: float
    ) -> list[tuple[float, FacilityState]]:
        """Return states within the time range."""
        return [(t, s) for t, s in self._buffer if start_time <= t <= end_time]
