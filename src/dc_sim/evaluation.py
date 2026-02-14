"""Evaluation framework: scenarios, scoring, and benchmarking for agent performance."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from dc_sim.config import SimConfig
    from dc_sim.models.facility import FacilityState
    from dc_sim.simulator import Simulator


# ── Data structures ───────────────────────────────────────────


class FailureInjection(BaseModel):
    """A scripted failure to inject at a specific simulation tick."""

    at_tick: int
    failure_type: str
    target: str
    duration_s: int | None = None


class ScenarioConfig(BaseModel):
    """Workload overrides for a scenario."""

    mean_job_arrival_interval_s: float = 300.0


class ScenarioDefinition(BaseModel):
    """A named, reproducible evaluation scenario."""

    scenario_id: str
    name: str
    description: str
    duration_ticks: int
    rng_seed: int
    failure_injections: list[FailureInjection] = Field(default_factory=list)
    workload_overrides: ScenarioConfig = Field(default_factory=ScenarioConfig)


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension (0-100)."""

    name: str
    score: float
    weight: float
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class EvaluationResult:
    """Full evaluation result after running or scoring a scenario."""

    scenario_id: str
    run_type: str  # "agent" or "baseline"
    composite_score: float
    dimensions: list[DimensionScore] = field(default_factory=list)
    duration_ticks: int = 0
    total_sim_time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_type": self.run_type,
            "composite_score": self.composite_score,
            "dimensions": [
                {
                    "name": d.name,
                    "score": round(d.score, 2),
                    "weight": d.weight,
                    "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in d.metrics.items()},
                    "notes": d.notes,
                }
                for d in self.dimensions
            ],
            "duration_ticks": self.duration_ticks,
            "total_sim_time_s": self.total_sim_time_s,
            "metadata": self.metadata,
        }


# ── Predefined scenarios ──────────────────────────────────────

STEADY_STATE = ScenarioDefinition(
    scenario_id="steady_state",
    name="STEADY_STATE",
    description="4 hours of normal operation with baseline arrival rate. No scripted failures. Tests general operational efficiency.",
    duration_ticks=240,
    rng_seed=42,
    failure_injections=[],
    workload_overrides=ScenarioConfig(mean_job_arrival_interval_s=300.0),
)

THERMAL_CRISIS = ScenarioDefinition(
    scenario_id="thermal_crisis",
    name="THERMAL_CRISIS",
    description="2 hours with CRAC unit 0 failing at t=30min for 45 minutes. Racks 0-3 lose all cooling. Tests failure detection and workload migration.",
    duration_ticks=120,
    rng_seed=123,
    failure_injections=[
        FailureInjection(at_tick=30, failure_type="crac_failure", target="crac-0", duration_s=2700),
    ],
    workload_overrides=ScenarioConfig(mean_job_arrival_interval_s=300.0),
)

CARBON_VALLEY = ScenarioDefinition(
    scenario_id="carbon_valley",
    name="CARBON_VALLEY",
    description="24-hour full day cycle. Tests carbon-aware scheduling — can the agent shift flexible work to overnight low-carbon periods?",
    duration_ticks=1440,
    rng_seed=77,
    failure_injections=[],
    workload_overrides=ScenarioConfig(mean_job_arrival_interval_s=300.0),
)

OVERLOAD = ScenarioDefinition(
    scenario_id="overload",
    name="OVERLOAD",
    description="2 hours at 3x normal job arrival rate. Tests scheduling under pressure, preemption decisions, and SLA triage.",
    duration_ticks=120,
    rng_seed=55,
    failure_injections=[],
    workload_overrides=ScenarioConfig(mean_job_arrival_interval_s=100.0),
)

CASCADE = ScenarioDefinition(
    scenario_id="cascade",
    name="CASCADE",
    description="2 hours with 5 sequential failures (CRAC degraded, GPU degraded, PDU spike, network partition, CRAC failure). Tests multi-failure triage.",
    duration_ticks=120,
    rng_seed=99,
    failure_injections=[
        FailureInjection(at_tick=10, failure_type="crac_degraded", target="crac-0", duration_s=1200),
        FailureInjection(at_tick=25, failure_type="gpu_degraded", target="rack-2-srv-1", duration_s=None),
        FailureInjection(at_tick=40, failure_type="pdu_spike", target="rack-4", duration_s=300),
        FailureInjection(at_tick=60, failure_type="network_partition", target="rack-3", duration_s=0),
        FailureInjection(at_tick=80, failure_type="crac_failure", target="crac-1", duration_s=1800),
    ],
    workload_overrides=ScenarioConfig(mean_job_arrival_interval_s=300.0),
)

SCENARIOS: dict[str, ScenarioDefinition] = {
    s.scenario_id: s
    for s in [STEADY_STATE, THERMAL_CRISIS, CARBON_VALLEY, OVERLOAD, CASCADE]
}

# ── Baseline cache ────────────────────────────────────────────

_baseline_cache: dict[str, EvaluationResult] = {}


# ── Normalisation helpers ─────────────────────────────────────


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _norm(value: float, target: float, worst: float) -> float:
    """Normalise: target → 100, worst → 0, clamped."""
    if abs(worst - target) < 1e-9:
        return 100.0 if value <= target else 0.0
    return _clamp(100.0 - 100.0 * (value - target) / (worst - target))


# ── Evaluator ─────────────────────────────────────────────────


class Evaluator:
    """Computes scores from telemetry history, workload queue, and audit log."""

    DIMENSION_WEIGHTS: dict[str, float] = {
        "sla_quality": 0.25,
        "energy_efficiency": 0.20,
        "carbon": 0.15,
        "thermal_safety": 0.15,
        "cost": 0.10,
        "infra_health": 0.10,
        "failure_response": 0.05,
    }

    def __init__(self, sim: Simulator, scenario: ScenarioDefinition) -> None:
        self.sim = sim
        self.scenario = scenario
        self._states: list[FacilityState] = [s for _, s in sim.telemetry._buffer]
        self._all_jobs = (
            list(sim.workload_queue.pending)
            + list(sim.workload_queue.running)
            + list(sim.workload_queue.completed)
        )
        self._audit = list(sim.audit_log._entries)
        self._config = sim.config

    def compute(self) -> EvaluationResult:
        dims = [
            self._score_sla(),
            self._score_energy(),
            self._score_carbon(),
            self._score_thermal(),
            self._score_cost(),
            self._score_infra(),
            self._score_failure_response(),
        ]
        composite = sum(d.score * d.weight for d in dims)
        return EvaluationResult(
            scenario_id=self.scenario.scenario_id,
            run_type="agent",
            composite_score=round(composite, 2),
            dimensions=dims,
            duration_ticks=len(self._states),
            total_sim_time_s=self.sim.clock.current_time,
        )

    # ── Dimension scorers ─────────────────────────────────

    def _score_sla(self) -> DimensionScore:
        w = self.DIMENSION_WEIGHTS["sla_quality"]
        completed_ok = [j for j in self._all_jobs if j.status == "completed"]
        total_submitted = len(self._all_jobs)
        violated = [j for j in self._all_jobs if j.sla_violated]

        violation_rate = len(violated) / max(1, total_submitted)
        sla_score = _clamp(100.0 - violation_rate * 200.0)

        completion_rate = len(completed_ok) / max(1, total_submitted)
        completion_score = completion_rate * 100.0

        wait_times = [
            j.started_at - j.submitted_at
            for j in completed_ok
            if j.started_at is not None
        ]
        avg_wait = sum(wait_times) / max(1, len(wait_times)) if wait_times else 0.0
        wait_score = _norm(avg_wait, 300.0, 3600.0)

        score = 0.5 * sla_score + 0.3 * completion_score + 0.2 * wait_score

        return DimensionScore(
            name="sla_quality",
            score=round(_clamp(score), 2),
            weight=w,
            metrics={
                "violation_rate_pct": violation_rate * 100,
                "completion_rate_pct": completion_rate * 100,
                "avg_queue_wait_s": avg_wait,
                "jobs_submitted": total_submitted,
                "jobs_completed": len(completed_ok),
                "jobs_violated": len(violated),
            },
        )

    def _score_energy(self) -> DimensionScore:
        w = self.DIMENSION_WEIGHTS["energy_efficiency"]
        if not self._states:
            return DimensionScore(name="energy_efficiency", score=50.0, weight=w)

        pues = [s.power.pue for s in self._states]
        avg_pue = sum(pues) / len(pues)
        pue_score = _norm(avg_pue, 1.2, 2.0)

        tick_s = self._config.clock.tick_interval_s
        total_kwh = sum(s.power.total_power_kw * (tick_s / 3600) for s in self._states)
        completed_count = len([j for j in self._all_jobs if j.status == "completed"])
        kwh_per_job = total_kwh / max(1, completed_count)
        kwh_score = _norm(kwh_per_job, 5.0, 50.0)

        gpu_utils = [s.gpu.avg_sm_util_pct for s in self._states]
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
        gpu_util_score = _clamp(avg_gpu_util)

        score = 0.4 * pue_score + 0.3 * kwh_score + 0.3 * gpu_util_score

        return DimensionScore(
            name="energy_efficiency",
            score=round(_clamp(score), 2),
            weight=w,
            metrics={
                "avg_pue": avg_pue,
                "total_kwh": total_kwh,
                "kwh_per_job": kwh_per_job,
                "avg_gpu_util_pct": avg_gpu_util,
            },
        )

    def _score_carbon(self) -> DimensionScore:
        w = self.DIMENSION_WEIGHTS["carbon"]
        if not self._states:
            return DimensionScore(name="carbon", score=50.0, weight=w)

        last = self._states[-1]
        total_carbon_kg = last.carbon.cumulative_carbon_kg
        tick_s = self._config.clock.tick_interval_s
        duration_h = self.scenario.duration_ticks * tick_s / 3600
        reference_carbon_kg = duration_h * 100.0 * 200.0 / 1000.0
        carbon_score = _norm(total_carbon_kg, 0.0, reference_carbon_kg)

        # gCO2 per GPU-hour of useful work
        gpu_hours = sum(
            s.gpu.healthy_gpus * (s.gpu.avg_sm_util_pct / 100.0) * (tick_s / 3600)
            for s in self._states
        )
        carbon_per_gpu_h = (total_carbon_kg * 1000.0) / max(0.001, gpu_hours)
        efficiency_score = _norm(carbon_per_gpu_h, 500.0, 5000.0)

        # Carbon awareness: higher GPU util during low-carbon vs high-carbon
        low_carbon = [s.gpu.avg_sm_util_pct for s in self._states if s.carbon.carbon_intensity_gco2_kwh < 180]
        high_carbon = [s.gpu.avg_sm_util_pct for s in self._states if s.carbon.carbon_intensity_gco2_kwh >= 250]
        if low_carbon and high_carbon:
            low_avg = sum(low_carbon) / len(low_carbon)
            high_avg = sum(high_carbon) / len(high_carbon)
            delta = (low_avg - high_avg) / 100.0
            awareness_score = _clamp(50.0 + delta * 50.0)
        else:
            awareness_score = 50.0

        score = 0.4 * carbon_score + 0.35 * efficiency_score + 0.25 * awareness_score

        return DimensionScore(
            name="carbon",
            score=round(_clamp(score), 2),
            weight=w,
            metrics={
                "total_carbon_kg": total_carbon_kg,
                "reference_carbon_kg": reference_carbon_kg,
                "carbon_per_gpu_hour_g": carbon_per_gpu_h,
                "carbon_awareness_delta_pct": (
                    (sum(low_carbon) / len(low_carbon) - sum(high_carbon) / len(high_carbon))
                    if low_carbon and high_carbon
                    else 0.0
                ),
            },
        )

    def _score_thermal(self) -> DimensionScore:
        w = self.DIMENSION_WEIGHTS["thermal_safety"]
        if not self._states:
            return DimensionScore(name="thermal_safety", score=100.0, weight=w)

        num_racks = self._config.facility.num_racks
        total_rack_ticks = num_racks * len(self._states)
        throttled_count = sum(
            1
            for s in self._states
            for r in s.thermal.racks
            if r.throttled
        )
        throttled_frac = throttled_count / max(1, total_rack_ticks)
        throttle_score = _clamp(100.0 - throttled_frac * 500.0)

        safe_temp = self._config.thermal.max_safe_inlet_temp_c
        crit_temp = self._config.thermal.critical_inlet_temp_c
        peak_inlet = max(
            r.inlet_temp_c
            for s in self._states
            for r in s.thermal.racks
        )
        if peak_inlet <= safe_temp:
            peak_score = 100.0
        elif peak_inlet >= crit_temp:
            peak_score = 0.0
        else:
            peak_score = 100.0 * (crit_temp - peak_inlet) / (crit_temp - safe_temp)

        thermal_event_ticks = sum(
            1
            for s in self._states
            if any(r.inlet_temp_c > safe_temp for r in s.thermal.racks)
        )
        event_rate = thermal_event_ticks / max(1, len(self._states))
        event_score = _clamp(100.0 - event_rate * 300.0)

        score = 0.4 * throttle_score + 0.35 * peak_score + 0.25 * event_score

        return DimensionScore(
            name="thermal_safety",
            score=round(_clamp(score), 2),
            weight=w,
            metrics={
                "throttled_fraction_pct": throttled_frac * 100,
                "peak_inlet_temp_c": peak_inlet,
                "thermal_event_ticks": thermal_event_ticks,
                "thermal_event_rate_pct": event_rate * 100,
            },
        )

    def _score_cost(self) -> DimensionScore:
        w = self.DIMENSION_WEIGHTS["cost"]
        if not self._states:
            return DimensionScore(name="cost", score=50.0, weight=w)

        last = self._states[-1]
        total_cost = last.carbon.cumulative_cost_gbp
        tick_s = self._config.clock.tick_interval_s
        duration_h = self.scenario.duration_ticks * tick_s / 3600
        reference_cost = duration_h * 100.0 * 0.20
        cost_score = _norm(total_cost, 0.0, reference_cost)

        completed_count = len([j for j in self._all_jobs if j.status == "completed"])
        cost_per_job = total_cost / max(1, completed_count)
        cpj_score = _norm(cost_per_job, 0.50, 5.0)

        # Price awareness: higher IT load during cheap periods
        cheap = [s.power.it_power_kw for s in self._states if s.carbon.electricity_price_gbp_kwh < 0.12]
        expensive = [s.power.it_power_kw for s in self._states if s.carbon.electricity_price_gbp_kwh > 0.20]
        if cheap and expensive:
            cheap_avg = sum(cheap) / len(cheap)
            expensive_avg = sum(expensive) / len(expensive)
            awareness = (cheap_avg - expensive_avg) / max(1.0, expensive_avg)
            price_score = _clamp(50.0 + awareness * 50.0)
        else:
            price_score = 50.0

        score = 0.45 * cost_score + 0.30 * cpj_score + 0.25 * price_score

        return DimensionScore(
            name="cost",
            score=round(_clamp(score), 2),
            weight=w,
            metrics={
                "total_cost_gbp": total_cost,
                "reference_cost_gbp": reference_cost,
                "cost_per_job_gbp": cost_per_job,
                "price_awareness_delta_kw": (
                    (sum(cheap) / len(cheap) - sum(expensive) / len(expensive))
                    if cheap and expensive
                    else 0.0
                ),
            },
        )

    def _score_infra(self) -> DimensionScore:
        w = self.DIMENSION_WEIGHTS["infra_health"]
        if not self._states:
            return DimensionScore(name="infra_health", score=100.0, weight=w)

        ecc_vals = [s.gpu.ecc_error_gpus for s in self._states]
        avg_ecc = sum(ecc_vals) / len(ecc_vals)
        ecc_score = _clamp(100.0 - avg_ecc * 10.0)

        loss_vals = [s.network.total_packet_loss_pct for s in self._states]
        avg_loss = sum(loss_vals) / len(loss_vals)
        packet_score = _clamp(100.0 - avg_loss * 1000.0)

        crc_vals = [s.network.total_crc_errors for s in self._states]
        avg_crc = sum(crc_vals) / len(crc_vals)
        crc_score = _clamp(100.0 - avg_crc * 5.0)

        # Storage drive health — use the last state (drive health only decreases)
        last = self._states[-1]
        rack_healths = [r.drive_health_pct for r in last.storage.racks]
        avg_health = sum(rack_healths) / max(1, len(rack_healths)) if rack_healths else 100.0
        storage_score = _clamp(avg_health)

        score = 0.30 * ecc_score + 0.30 * packet_score + 0.20 * crc_score + 0.20 * storage_score

        return DimensionScore(
            name="infra_health",
            score=round(_clamp(score), 2),
            weight=w,
            metrics={
                "avg_ecc_error_gpus": avg_ecc,
                "avg_packet_loss_pct": avg_loss,
                "avg_crc_errors": avg_crc,
                "avg_drive_health_pct": avg_health,
            },
        )

    def _score_failure_response(self) -> DimensionScore:
        w = self.DIMENSION_WEIGHTS["failure_response"]
        scripted = self.scenario.failure_injections
        tick_s = self._config.clock.tick_interval_s

        if not scripted:
            return DimensionScore(
                name="failure_response",
                score=100.0,
                weight=w,
                metrics={"scripted_failures": 0, "mean_ttr_s": 0, "unresolved": 0},
                notes="No scripted failures in this scenario.",
            )

        # Build injection time -> failure_id mapping from audit log
        inject_entries = [e for e in self._audit if e.action == "inject_failure" and e.source == "scenario"]
        resolve_entries = [e for e in self._audit if e.action == "resolve_failure"]

        # Match injections to resolutions by failure_id
        inject_map: dict[str, float] = {}
        for e in inject_entries:
            fid = e.params.get("failure_id", "")
            if fid:
                inject_map[fid] = e.timestamp

        response_times: list[float] = []
        resolved_ids: set[str] = set()
        for e in resolve_entries:
            fid = e.params.get("failure_id", "")
            if fid in inject_map and fid not in resolved_ids:
                response_times.append(e.timestamp - inject_map[fid])
                resolved_ids.add(fid)

        expected = len(scripted)
        unresolved = expected - len(response_times)

        if response_times:
            mean_ttr = sum(response_times) / len(response_times)
            ttr_score = _norm(mean_ttr, 300.0, 3600.0)
        else:
            mean_ttr = 0.0
            ttr_score = 0.0 if expected > 0 else 100.0

        # Penalty for unresolved failures
        unresolved_penalty = (unresolved / max(1, expected)) * 50.0
        ttr_score = _clamp(ttr_score - unresolved_penalty)

        # SLA violations during failure windows
        violation_count = 0
        for fi in scripted:
            inject_time = fi.at_tick * tick_s
            dur = fi.duration_s if fi.duration_s else 3600
            end_time = inject_time + dur
            for j in self._all_jobs:
                if j.sla_violated and inject_time <= j.submitted_at <= end_time:
                    violation_count += 1

        failure_sla_score = _clamp(100.0 - violation_count * 20.0)

        score = 0.7 * ttr_score + 0.3 * failure_sla_score

        return DimensionScore(
            name="failure_response",
            score=round(_clamp(score), 2),
            weight=w,
            metrics={
                "scripted_failures": expected,
                "mean_ttr_s": mean_ttr,
                "unresolved": unresolved,
                "violations_during_failures": violation_count,
            },
        )


# ── Scenario runner ───────────────────────────────────────────


def run_scenario(
    sim: Simulator,
    scenario: ScenarioDefinition,
    agent_callback: Callable[[FacilityState], None] | None = None,
) -> EvaluationResult:
    """Reset sim, run a scenario to completion, return scores.

    Args:
        sim: The Simulator instance.
        scenario: Which scenario to run.
        agent_callback: If provided, called after each tick with the new state.
            Pass None for a no-agent baseline run.

    Returns:
        EvaluationResult with composite and per-dimension scores.
    """
    # Save original config
    original_config = sim.config

    # Build scenario config
    cfg = sim.config.model_copy(deep=True)
    cfg.rng_seed = scenario.rng_seed
    cfg.workload.mean_job_arrival_interval_s = scenario.workload_overrides.mean_job_arrival_interval_s

    # Reset with scenario config
    sim.config = cfg
    sim.reset()

    # Pre-sort failure injections by tick
    injections_by_tick: dict[int, list[FailureInjection]] = {}
    for fi in scenario.failure_injections:
        injections_by_tick.setdefault(fi.at_tick, []).append(fi)

    # Run
    for tick_idx in range(scenario.duration_ticks):
        # Inject scripted failures
        if tick_idx in injections_by_tick:
            for fi in injections_by_tick[tick_idx]:
                sim.failure_engine.set_current_time(sim.clock.current_time)
                failures = sim.failure_engine.inject(fi.failure_type, fi.target, fi.duration_s)
                if failures:
                    sim.audit_log.record(
                        timestamp=sim.clock.current_time,
                        action="inject_failure",
                        params={
                            "type": fi.failure_type,
                            "target": fi.target,
                            "duration_s": fi.duration_s,
                            "failure_id": failures[0].failure_id,
                        },
                        source="scenario",
                    )

        states = sim.tick(1)
        if agent_callback and states:
            agent_callback(states[-1])

    # Score
    evaluator = Evaluator(sim, scenario)
    result = evaluator.compute()
    result.run_type = "agent" if agent_callback else "baseline"

    # Restore original config
    sim.config = original_config

    return result


# ── Session management ───────────────────────────────────────


@dataclass
class SessionState:
    """Mutable state for an active evaluation session."""

    scenario_id: str
    scenario: ScenarioDefinition
    agent_name: str
    current_tick: int
    max_ticks: int
    injections_by_tick: dict[int, list[FailureInjection]]
    original_config: Any  # SimConfig
    started_at_real: float
    active: bool = True


class SessionManager:
    """Manages step-by-step evaluation sessions for external agents.

    Single-tenant: only one session can be active at a time.
    """

    def __init__(self, sim: Simulator) -> None:
        self.sim = sim
        self._session: SessionState | None = None

    @property
    def active(self) -> bool:
        return self._session is not None and self._session.active

    def start(
        self,
        scenario_id: str,
        agent_name: str = "unnamed",
        scenario: ScenarioDefinition | None = None,
    ) -> dict:
        """Start a new evaluation session.

        Args:
            scenario_id: ID of a predefined scenario (used for lookup if
                ``scenario`` is not provided).
            agent_name: Name of the agent being evaluated.
            scenario: Optional explicit ScenarioDefinition. When provided,
                this is used directly instead of looking up by *scenario_id*.

        Returns a session info dict.
        Raises ValueError if a session is already active or scenario unknown.
        """
        if self.active:
            raise ValueError("A session is already active. End it first.")
        if scenario is None:
            if scenario_id not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario_id}")
            scenario = SCENARIOS[scenario_id]
        if self.sim.is_running:
            raise ValueError("Stop continuous simulation before starting a session.")

        # Save original config
        original_config = self.sim.config

        # Build scenario config
        cfg = self.sim.config.model_copy(deep=True)
        cfg.rng_seed = scenario.rng_seed
        cfg.workload.mean_job_arrival_interval_s = (
            scenario.workload_overrides.mean_job_arrival_interval_s
        )

        # Reset sim with scenario config
        self.sim.config = cfg
        self.sim.reset()

        # Pre-sort failure injections by tick
        injections_by_tick: dict[int, list[FailureInjection]] = {}
        for fi in scenario.failure_injections:
            injections_by_tick.setdefault(fi.at_tick, []).append(fi)

        self._session = SessionState(
            scenario_id=scenario_id,
            scenario=scenario,
            agent_name=agent_name,
            current_tick=0,
            max_ticks=scenario.duration_ticks,
            injections_by_tick=injections_by_tick,
            original_config=original_config,
            started_at_real=_time.time(),
        )

        return {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "duration_ticks": scenario.duration_ticks,
            "tick_interval_s": cfg.clock.tick_interval_s,
            "failure_count": len(scenario.failure_injections),
            "agent_name": agent_name,
        }

    def step(self) -> dict:
        """Advance one tick. Returns step result dict.

        Injects scripted failures before ticking (same order as run_scenario).
        Raises ValueError if no active session or already complete.
        """
        if not self.active:
            raise ValueError("No active session")
        session = self._session
        assert session is not None

        if session.current_tick >= session.max_ticks:
            raise ValueError("Session already completed all ticks")

        tick_idx = session.current_tick
        injected: list[dict] = []

        # Inject scripted failures for this tick
        if tick_idx in session.injections_by_tick:
            for fi in session.injections_by_tick[tick_idx]:
                self.sim.failure_engine.set_current_time(self.sim.clock.current_time)
                failures = self.sim.failure_engine.inject(
                    fi.failure_type, fi.target, fi.duration_s
                )
                if failures:
                    self.sim.audit_log.record(
                        timestamp=self.sim.clock.current_time,
                        action="inject_failure",
                        params={
                            "type": fi.failure_type,
                            "target": fi.target,
                            "duration_s": fi.duration_s,
                            "failure_id": failures[0].failure_id,
                        },
                        source="scenario",
                    )
                    injected.append({
                        "failure_id": failures[0].failure_id,
                        "type": fi.failure_type,
                        "target": fi.target,
                        "effect": failures[0].effect,
                    })

        # Advance simulation
        states = self.sim.tick(1)
        session.current_tick += 1
        done = session.current_tick >= session.max_ticks

        # Build state dict
        from dc_sim.telemetry import facility_state_to_dict

        state_dict = facility_state_to_dict(states[-1]) if states else {}

        # Add failure info and running jobs to state for agent convenience
        active_failures = [
            {
                "failure_id": f.failure_id,
                "type": f.failure_type,
                "target": f.target,
                "effect": f.effect,
            }
            for f in self.sim.failure_engine.get_active_failures()
        ]
        state_dict["failures"] = active_failures

        # Add running job details for agent decision-making
        running_jobs = [
            {
                "job_id": j.job_id,
                "name": j.name,
                "gpu_requirement": j.gpu_requirement,
                "priority": j.priority,
                "job_type": j.job_type,
                "assigned_servers": j.assigned_servers,
            }
            for j in self.sim.workload_queue.running
        ]
        state_dict["running_jobs"] = running_jobs

        return {
            "tick": session.current_tick,
            "max_ticks": session.max_ticks,
            "done": done,
            "sim_time_s": self.sim.clock.current_time,
            "failures_injected": injected,
            "state": state_dict,
            "active_failures": active_failures,
        }

    def end(self) -> EvaluationResult:
        """End the session and compute scores.

        Restores the original simulator config.
        Raises ValueError if no active session.
        """
        if not self.active:
            raise ValueError("No active session")
        session = self._session
        assert session is not None

        # Score
        evaluator = Evaluator(self.sim, session.scenario)
        result = evaluator.compute()
        result.run_type = "agent"
        result.metadata["agent_name"] = session.agent_name
        result.metadata["session_ticks_completed"] = session.current_tick
        result.metadata["session_ticks_total"] = session.max_ticks
        result.metadata["session_elapsed_real_s"] = _time.time() - session.started_at_real

        # Restore original config
        self.sim.config = session.original_config

        # Clear session
        self._session = None

        return result

    def get_status(self) -> dict:
        """Get current session status."""
        if not self.active or self._session is None:
            return {
                "active": False,
                "scenario_id": None,
                "agent_name": None,
                "current_tick": 0,
                "max_ticks": 0,
                "remaining_ticks": 0,
                "progress_pct": 0.0,
                "sim_time_s": 0.0,
                "elapsed_real_s": 0.0,
            }
        session = self._session
        return {
            "active": True,
            "scenario_id": session.scenario_id,
            "agent_name": session.agent_name,
            "current_tick": session.current_tick,
            "max_ticks": session.max_ticks,
            "remaining_ticks": session.max_ticks - session.current_tick,
            "progress_pct": round(
                100.0 * session.current_tick / max(1, session.max_ticks), 1
            ),
            "sim_time_s": self.sim.clock.current_time,
            "elapsed_real_s": round(_time.time() - session.started_at_real, 2),
        }
