"""Thermal model: rack inlet/outlet temperatures from power and cooling.

Improvements over v1:
- Hot-aisle/cold-aisle layout with heat recirculation between adjacent racks
- Per-rack humidity tracking (affects cooling efficiency)
- Non-linear cooling efficiency (degrades at high temperatures)
- Ambient temperature that drifts with outside conditions (time-of-day)
"""

import math
from dataclasses import dataclass, field

from dc_sim.config import SimConfig


@dataclass
class RackThermalState:
    """Thermal state for a single rack."""

    rack_id: int
    inlet_temp_c: float
    outlet_temp_c: float
    heat_generated_kw: float
    throttled: bool = False
    humidity_pct: float = 45.0
    delta_t_c: float = 0.0  # Outlet - Inlet


@dataclass
class FacilityThermalState:
    """Thermal state for the facility."""

    racks: list[RackThermalState] = field(default_factory=list)
    ambient_temp_c: float = 22.0
    avg_humidity_pct: float = 45.0


class ThermalModel:
    """Simulates rack temperatures from power draw and cooling.

    Layout: racks are arranged in a row. Adjacent racks share some heat
    (hot-aisle recirculation). Racks at the ends of the row run cooler
    because they only have one neighbour.
    """

    # Heat recirculation coefficient: fraction of neighbour's exhaust
    # heat that bleeds into this rack's inlet
    RECIRCULATION_COEFF = 0.08

    # Humidity parameters
    HUMIDITY_BASE = 45.0  # % RH baseline
    HUMIDITY_HEAT_COEFF = 1.5  # % RH drop per kW above 3 kW (hot air holds more moisture -> lower RH)
    HUMIDITY_COOLING_COEFF = 0.8  # % RH rise per unit cooling factor

    def __init__(self, config: SimConfig, rng_seed: int = 42):
        self.config = config
        self.facility = config.facility
        self.thermal_cfg = config.thermal
        self._rng = __import__("numpy").random.default_rng(rng_seed + 200)
        self._inlet_temps: dict[int, float] = {}
        self._humidity: dict[int, float] = {}
        self._initialise_temps()

    def _initialise_temps(self) -> None:
        """Set initial inlet temps to ambient and humidity to baseline."""
        for rack_id in range(self.facility.num_racks):
            self._inlet_temps[rack_id] = self.thermal_cfg.ambient_temp_c
            self._humidity[rack_id] = self.HUMIDITY_BASE

    def _effective_ambient(self, sim_time: float) -> float:
        """Outside temperature varies with time of day.

        Sinusoidal daily cycle: cooler at night, warmer mid-afternoon.
        This affects CRAC efficiency (harder to reject heat when it's hot outside).
        """
        hour = ((sim_time / 3600.0) + 8.0) % 24.0  # start at 08:00
        base = self.thermal_cfg.ambient_temp_c
        # +/- 4°C daily swing: peak at 15:00, trough at 04:00
        variation = 4.0 * math.sin(2.0 * math.pi * (hour - 4.0) / 24.0)
        return base + variation

    def _cooling_efficiency(self, inlet_temp: float, humidity: float) -> float:
        """Cooling efficiency degrades at high temperatures and humidity.

        Returns a multiplier 0.7-1.0 applied to cooling capacity.
        """
        # Temperature penalty: efficiency drops above 30°C inlet
        temp_penalty = max(0.0, (inlet_temp - 30.0) * 0.02)
        # Humidity penalty: efficiency drops above 60% RH
        humid_penalty = max(0.0, (humidity - 60.0) * 0.005)
        return max(0.7, 1.0 - temp_penalty - humid_penalty)

    def step(
        self,
        rack_power_kw: dict[int, float],
        cooling_capacity_factor: dict[int, float],
        tick_interval_s: float,
        sim_time: float = 0.0,
    ) -> FacilityThermalState:
        """
        Advance thermal state by one tick.
        rack_power_kw: rack_id -> heat generated (kW)
        cooling_capacity_factor: rack_id -> 0.0-1.0 (1.0 = full, 0.5 = degraded, 0 = failed)
        sim_time: current simulation time in seconds (for ambient variation)
        """
        racks: list[RackThermalState] = []
        num_racks = self.facility.num_racks
        effective_ambient = self._effective_ambient(sim_time)

        # First pass: compute outlet temps from previous state (for recirculation)
        prev_outlets: dict[int, float] = {}
        for rack_id in range(num_racks):
            prev_inlet = self._inlet_temps.get(rack_id, effective_ambient)
            heat_kw = rack_power_kw.get(rack_id, 0.0)
            prev_outlets[rack_id] = prev_inlet + (heat_kw * 5.0)

        for rack_id in range(num_racks):
            heat_kw = rack_power_kw.get(rack_id, 0.0)
            cooling_factor = cooling_capacity_factor.get(rack_id, 1.0)
            prev_inlet = self._inlet_temps.get(rack_id, effective_ambient)
            humidity = self._humidity.get(rack_id, self.HUMIDITY_BASE)

            # Hot-aisle recirculation from adjacent racks
            recirculation_heat = 0.0
            for neighbour in (rack_id - 1, rack_id + 1):
                if 0 <= neighbour < num_racks:
                    neighbour_exhaust = prev_outlets.get(neighbour, effective_ambient)
                    # Heat leaking from neighbour's hot aisle
                    recirculation_heat += self.RECIRCULATION_COEFF * max(0, neighbour_exhaust - prev_inlet)

            # Cooling: base capacity, adjusted for efficiency and factor
            efficiency = self._cooling_efficiency(prev_inlet, humidity)
            cooling_per_rack = (
                self.thermal_cfg.crac_cooling_capacity_kw / num_racks
            ) * cooling_factor * efficiency

            # CRAC has to work harder when outside temp is high
            ambient_penalty = max(0.0, (effective_ambient - self.thermal_cfg.ambient_temp_c) * 0.02)
            cooling_per_rack *= max(0.8, 1.0 - ambient_penalty)

            heat_removed = cooling_per_rack
            net_heat = heat_kw + recirculation_heat - heat_removed

            # Temperature delta
            temp_delta = net_heat * self.thermal_cfg.thermal_mass_coefficient * (tick_interval_s / 60.0)
            new_inlet = prev_inlet + temp_delta
            new_inlet = max(effective_ambient, min(60.0, new_inlet))
            self._inlet_temps[rack_id] = new_inlet

            # Outlet temperature: inlet + delta_T proportional to heat and airflow
            delta_t = heat_kw * 5.0  # ~5°C rise per kW
            outlet = new_inlet + delta_t

            # Humidity update
            heat_effect = -self.HUMIDITY_HEAT_COEFF * max(0, heat_kw - 3.0)
            cooling_effect = self.HUMIDITY_COOLING_COEFF * cooling_factor
            noise = float(self._rng.normal(0, 0.3))
            new_humidity = self.HUMIDITY_BASE + heat_effect + cooling_effect + noise
            new_humidity = max(20.0, min(80.0, new_humidity))
            self._humidity[rack_id] = new_humidity

            throttled = new_inlet >= self.thermal_cfg.critical_inlet_temp_c

            racks.append(
                RackThermalState(
                    rack_id=rack_id,
                    inlet_temp_c=new_inlet,
                    outlet_temp_c=outlet,
                    heat_generated_kw=heat_kw,
                    throttled=throttled,
                    humidity_pct=new_humidity,
                    delta_t_c=delta_t,
                )
            )

        avg_humidity = sum(r.humidity_pct for r in racks) / max(1, len(racks))

        return FacilityThermalState(
            racks=racks,
            ambient_temp_c=effective_ambient,
            avg_humidity_pct=avg_humidity,
        )

    def reset(self) -> None:
        """Reset to initial state."""
        self._initialise_temps()
