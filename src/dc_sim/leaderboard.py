"""Leaderboard: CSV-backed results store for agent evaluation runs."""

from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Default CSV path (relative to project root)
_DEFAULT_CSV = Path(__file__).resolve().parent.parent.parent / "results" / "leaderboard.csv"

COLUMNS = [
    "run_id",
    "timestamp",
    "agent_name",
    "scenario_id",
    "composite_score",
    "sla_quality",
    "energy_efficiency",
    "carbon",
    "thermal_safety",
    "cost",
    "infra_health",
    "failure_response",
    "duration_ticks",
    "total_sim_time_s",
]

DIMENSION_NAMES = [
    "sla_quality",
    "energy_efficiency",
    "carbon",
    "thermal_safety",
    "cost",
    "infra_health",
    "failure_response",
]


def _ensure_csv(csv_path: Path | None = None) -> Path:
    """Create CSV with headers if it doesn't exist."""
    path = csv_path or _DEFAULT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
    return path


def record_result(
    agent_name: str,
    scenario_id: str,
    eval_result: dict[str, Any],
    csv_path: Path | None = None,
) -> str:
    """Append an evaluation result to the leaderboard CSV.

    Args:
        agent_name: Name of the agent.
        scenario_id: Scenario that was run.
        eval_result: EvaluationResult.to_dict() output.
        csv_path: Optional override for CSV path.

    Returns:
        The generated run_id.
    """
    path = _ensure_csv(csv_path)
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Extract per-dimension scores
    dim_scores: dict[str, float] = {}
    for d in eval_result.get("dimensions", []):
        dim_scores[d["name"]] = d["score"]

    row = [
        run_id,
        timestamp,
        agent_name,
        scenario_id,
        round(eval_result.get("composite_score", 0), 2),
    ]
    # Add dimension scores in canonical order
    for dim_name in DIMENSION_NAMES:
        row.append(round(dim_scores.get(dim_name, 0), 2))
    row.append(eval_result.get("duration_ticks", 0))
    row.append(round(eval_result.get("total_sim_time_s", 0), 2))

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return run_id


def load_leaderboard(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the leaderboard CSV as a DataFrame.

    Returns an empty DataFrame with correct columns if file is missing.
    """
    path = csv_path or _DEFAULT_CSV
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def get_best_scores(
    scenario_id: str | None = None,
    csv_path: Path | None = None,
) -> pd.DataFrame:
    """Get best composite score per agent (optionally filtered by scenario).

    Returns DataFrame with columns: agent_name, scenario_id, composite_score, ...
    """
    df = load_leaderboard(csv_path)
    if df.empty:
        return df

    if scenario_id:
        df = df[df["scenario_id"] == scenario_id]

    if df.empty:
        return df

    # Best composite per agent per scenario
    idx = df.groupby(["agent_name", "scenario_id"])["composite_score"].idxmax()
    return df.loc[idx].sort_values("composite_score", ascending=False).reset_index(drop=True)
