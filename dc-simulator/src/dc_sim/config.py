"""All tuneable simulation parameters."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class FacilityConfig(BaseModel):
    """Facility layout configuration."""

    num_racks: int = 8
    servers_per_rack: int = 4
    gpus_per_server: int = 4


class ThermalConfig(BaseModel):
    """Thermal model parameters."""

    ambient_temp_c: float = 22.0
    crac_setpoint_c: float = 18.0
    crac_cooling_capacity_kw: float = 50.0
    thermal_mass_coefficient: float = 0.3
    max_safe_inlet_temp_c: float = 35.0
    critical_inlet_temp_c: float = 40.0
    crac_units: int = 2


class PowerConfig(BaseModel):
    """Power model parameters."""

    gpu_tdp_watts: int = 300
    server_base_power_watts: int = 200
    pdu_capacity_kw: float = 20.0
    facility_power_cap_kw: float = 120.0
    pue_overhead_factor: float = 1.4


class WorkloadConfig(BaseModel):
    """Workload generation parameters."""

    mean_job_arrival_interval_s: float = 300.0
    job_duration_range_s: tuple[int, int] = (600, 7200)
    gpu_requirement_range: tuple[int, int] = (1, 8)
    job_priority_range: tuple[int, int] = (1, 5)
    sla_deadline_range_s: tuple[float, float] = (600.0, 3600.0)


class ClockConfig(BaseModel):
    """Simulation clock parameters."""

    tick_interval_s: float = 60.0
    realtime_factor: float = 0.0


class SimConfig(BaseModel):
    """Complete simulation configuration."""

    facility: FacilityConfig = Field(default_factory=FacilityConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    power: PowerConfig = Field(default_factory=PowerConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    clock: ClockConfig = Field(default_factory=ClockConfig)
    rng_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        """Load config from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(_flatten_for_pydantic(data))

    @classmethod
    def from_env(cls) -> "SimConfig":
        """Load config from environment variables (simplified)."""
        import os

        # Check for config file path
        config_path = os.getenv("DC_SIM_CONFIG")
        if config_path:
            return cls.from_yaml(config_path)
        return cls()


def _flatten_for_pydantic(obj: Any) -> dict:
    """Flatten nested dict for Pydantic validation."""
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, dict) and k in ("facility", "thermal", "power", "workload", "clock"):
                result[k] = _flatten_for_pydantic(v)
            elif isinstance(v, list) and len(v) == 2:
                result[k] = tuple(v)  # YAML lists -> tuples for ranges
            else:
                result[k] = v
        return result
    return obj
