"""Carbon intensity and electricity cost models with time-varying profiles."""

import math
from dataclasses import dataclass

from dc_sim.config import SimConfig


@dataclass
class CarbonState:
    """Carbon and cost snapshot for the facility."""

    carbon_intensity_gco2_kwh: float  # Current grid carbon intensity
    carbon_rate_gco2_s: float  # Instantaneous carbon emission rate (g CO2/s)
    cumulative_carbon_kg: float  # Total carbon emitted since start
    electricity_price_gbp_kwh: float  # Current spot price
    cost_rate_gbp_h: float  # Instantaneous cost rate (GBP/hour)
    cumulative_cost_gbp: float  # Total cost since start


class CarbonModel:
    """
    Time-varying carbon intensity and electricity cost.

    Carbon intensity follows a realistic UK grid profile:
      - Base ~200 g CO2/kWh
      - Lower at night (more wind/nuclear share), higher mid-afternoon (gas peakers)
      - Sinusoidal daily pattern with some noise

    Electricity price follows a similar but shifted pattern:
      - Base ~0.15 GBP/kWh
      - Peaks in morning (07-09) and evening (17-19) — demand peaks
      - Cheapest overnight (01-05)
    """

    def __init__(self, config: SimConfig, rng_seed: int = 42):
        self.config = config
        self._rng = __import__("numpy").random.default_rng(rng_seed + 100)
        self._cumulative_carbon_kg: float = 0.0
        self._cumulative_cost_gbp: float = 0.0

    def _hour_of_day(self, sim_time: float) -> float:
        """Convert simulation time (seconds from epoch) to hour of day (0-24)."""
        # Simulation starts at 08:00 by convention
        return ((sim_time / 3600.0) + 8.0) % 24.0

    def carbon_intensity(self, sim_time: float) -> float:
        """
        Grid carbon intensity in g CO2/kWh.

        UK-realistic profile:
          - Base: 200 g/kWh
          - Night trough (~03:00): ~140 g/kWh
          - Afternoon peak (~15:00): ~280 g/kWh
          - Small random noise +-10 g/kWh
        """
        hour = self._hour_of_day(sim_time)
        base = 200.0
        # Peak around hour 15, trough around hour 3
        daily_variation = 60.0 * math.sin(2.0 * math.pi * (hour - 3.0) / 24.0)
        noise = float(self._rng.normal(0, 5))
        return max(50.0, base + daily_variation + noise)

    def electricity_price(self, sim_time: float) -> float:
        """
        Electricity spot price in GBP/kWh.

        UK-realistic profile:
          - Base: 0.15 GBP/kWh
          - Morning peak (~08:00): +0.08
          - Evening peak (~18:00): +0.06
          - Night trough (~03:00): -0.05
        """
        hour = self._hour_of_day(sim_time)
        base = 0.15
        # Double-peak pattern: morning and evening
        morning_peak = 0.08 * math.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
        evening_peak = 0.06 * math.exp(-0.5 * ((hour - 18.0) / 2.0) ** 2)
        night_dip = -0.05 * math.exp(-0.5 * ((hour - 3.0) / 2.5) ** 2)
        noise = float(self._rng.normal(0, 0.005))
        return max(0.02, base + morning_peak + evening_peak + night_dip + noise)

    def step(self, sim_time: float, total_power_kw: float, tick_interval_s: float) -> CarbonState:
        """Compute carbon and cost for this tick."""
        ci = self.carbon_intensity(sim_time)
        price = self.electricity_price(sim_time)

        # Energy consumed this tick (kWh)
        energy_kwh = total_power_kw * (tick_interval_s / 3600.0)

        # Carbon emitted this tick (kg)
        carbon_kg = (ci * energy_kwh) / 1000.0
        self._cumulative_carbon_kg += carbon_kg

        # Cost this tick (GBP)
        cost_gbp = price * energy_kwh
        self._cumulative_cost_gbp += cost_gbp

        # Instantaneous rates
        carbon_rate = (ci * total_power_kw) / 1000.0  # g CO2/s equivalent: (g/kWh * kW) / 1000 -> g/s... actually let's keep per-second
        # carbon_rate in g CO2 per second = ci * total_power_kw / 3600
        carbon_rate_gco2_s = ci * total_power_kw / 3600.0
        cost_rate_gbp_h = price * total_power_kw

        return CarbonState(
            carbon_intensity_gco2_kwh=ci,
            carbon_rate_gco2_s=carbon_rate_gco2_s,
            cumulative_carbon_kg=self._cumulative_carbon_kg,
            electricity_price_gbp_kwh=price,
            cost_rate_gbp_h=cost_rate_gbp_h,
            cumulative_cost_gbp=self._cumulative_cost_gbp,
        )

    def reset(self) -> None:
        """Reset cumulative counters."""
        self._cumulative_carbon_kg = 0.0
        self._cumulative_cost_gbp = 0.0
