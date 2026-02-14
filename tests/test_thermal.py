"""Tests for the thermal model."""

import pytest

from dc_sim.config import SimConfig
from dc_sim.models.thermal import ThermalModel


def test_rack_with_high_load_rises_temp():
    """A rack with high GPU util and no cooling should see rising inlet temps."""
    config = SimConfig()
    config.thermal.crac_cooling_capacity_kw = 0
    config.thermal.thermal_mass_coefficient = 0.5
    model = ThermalModel(config)

    rack_power = {0: 10.0}
    cooling = {0: 1.0}
    tick = 60.0

    state0 = model.step(rack_power, cooling, tick)
    inlet0 = state0.racks[0].inlet_temp_c

    for _ in range(10):
        state = model.step(rack_power, cooling, tick)

    inlet_final = state.racks[0].inlet_temp_c
    assert inlet_final > inlet0
    assert inlet_final > 22.0


def test_rack_with_zero_load_converges_to_ambient():
    """A rack with zero load should converge toward ambient temperature.

    The effective ambient varies with time-of-day (+/-4C swing),
    so we pass sim_time=0 which maps to 08:00 and check against
    the effective ambient at that hour.
    """
    config = SimConfig()
    config.thermal.ambient_temp_c = 22.0
    model = ThermalModel(config)

    rack_power = {0: 0.0}
    cooling = {0: 1.0}
    tick = 60.0

    model._inlet_temps[0] = 30.0
    for i in range(50):
        state = model.step(rack_power, cooling, tick, sim_time=float(i * 60))

    inlet = state.racks[0].inlet_temp_c
    effective_ambient = state.ambient_temp_c
    # Inlet should converge close to effective ambient
    assert abs(inlet - effective_ambient) < 3.0


def test_thermal_throttling_activates():
    """Thermal throttling should activate when inlet temp exceeds critical threshold."""
    config = SimConfig()
    config.thermal.critical_inlet_temp_c = 40.0
    config.thermal.thermal_mass_coefficient = 1.0
    config.thermal.crac_cooling_capacity_kw = 0
    model = ThermalModel(config)

    rack_power = {0: 20.0}
    cooling = {0: 0.0}
    tick = 60.0

    throttled = False
    for _ in range(30):
        state = model.step(rack_power, cooling, tick)
        if state.racks[0].throttled:
            throttled = True
            break

    assert throttled
