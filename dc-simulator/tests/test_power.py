"""Tests for the power model."""

import pytest

from dc_sim.config import SimConfig
from dc_sim.models.power import PowerModel


def test_server_power_at_full_util():
    """Server power draw at 100% GPU util should equal base + gpu_tdp * num_gpus."""
    config = SimConfig()
    model = PowerModel(config)
    cfg = config.power
    fac = config.facility

    server_id = "rack-0-srv-0"
    util = {s: 0.05 for s in [f"rack-{r}-srv-{s}" for r in range(fac.num_racks) for s in range(fac.servers_per_rack)]}
    util[server_id] = 1.0

    state = model.compute(util, set(), {}, None, None)
    rack0 = state.racks[0]
    srv0 = next(s for s in rack0.servers if s.server_id == server_id)

    expected_gpu = cfg.gpu_tdp_watts * fac.gpus_per_server
    expected_total = cfg.server_base_power_watts + expected_gpu
    assert abs(srv0.total_power_draw_w - expected_total) < 1.0


def test_facility_power_pue():
    """Facility power should equal IT power × dynamic PUE."""
    config = SimConfig()
    model = PowerModel(config)
    fac = config.facility
    util = {f"rack-{r}-srv-{s}": 0.5 for r in range(fac.num_racks) for s in range(fac.servers_per_rack)}

    state = model.compute(util, set(), {}, None, None)
    # PUE is now dynamic — verify total = it_power * reported PUE
    expected_total = state.it_power_kw * state.pue
    assert abs(state.total_power_kw - expected_total) < 0.1
    # Dynamic PUE should be >= base PUE (load penalty is non-negative)
    assert state.pue >= config.power.pue_overhead_factor


def test_headroom_negative_when_cap_exceeded():
    """headroom_kw should be negative when power cap is exceeded."""
    config = SimConfig()
    config.power.facility_power_cap_kw = 10.0
    model = PowerModel(config)
    fac = config.facility
    util = {f"rack-{r}-srv-{s}": 1.0 for r in range(fac.num_racks) for s in range(fac.servers_per_rack)}

    state = model.compute(util, set(), {}, None, None)
    assert state.power_cap_exceeded
    assert state.headroom_kw < 0
