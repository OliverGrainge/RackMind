"""Cooling system model: CRAC units, chilled water loop, cooling tower metrics.

Models realistic data centre cooling infrastructure:
- Individual CRAC (Computer Room Air Conditioning) units with supply/return temps
- Chilled water loop temperatures and flow rates
- Cooling tower wet-bulb approach
- COP (Coefficient of Performance) varying with conditions
- Pump and fan power consumption
"""

import math
from dataclasses import dataclass, field

from dc_sim.config import SimConfig


@dataclass
class CracUnitState:
    """Telemetry for a single CRAC unit."""

    unit_id: int
    # Temperatures
    supply_air_temp_c: float = 15.0  # Cold air output
    return_air_temp_c: float = 30.0  # Hot air input
    # Air flow
    fan_speed_pct: float = 50.0  # 0-100
    airflow_cfm: float = 10000.0  # Cubic feet per minute
    # Water side
    chw_supply_temp_c: float = 7.0  # Chilled water supply
    chw_return_temp_c: float = 12.0  # Chilled water return
    chw_flow_rate_lps: float = 5.0  # Litres per second
    # Performance
    cooling_output_kw: float = 50.0  # Actual cooling delivered
    cooling_capacity_kw: float = 50.0  # Maximum capacity
    load_pct: float = 50.0  # % of capacity in use
    # Status
    operational: bool = True
    fault_code: int = 0  # 0 = no fault


@dataclass
class CoolingTowerState:
    """Telemetry for the cooling tower / condenser."""

    # Water loop
    condenser_supply_temp_c: float = 28.0  # Water to condenser
    condenser_return_temp_c: float = 33.0  # Water from condenser
    # Ambient
    wet_bulb_temp_c: float = 18.0  # Wet-bulb temperature
    approach_temp_c: float = 5.0  # Condenser supply - wet bulb
    # Fan
    fan_speed_pct: float = 40.0
    # Performance
    heat_rejection_kw: float = 100.0


@dataclass
class FacilityCoolingState:
    """Facility-wide cooling system state."""

    crac_units: list[CracUnitState] = field(default_factory=list)
    cooling_tower: CoolingTowerState = field(default_factory=CoolingTowerState)
    # Aggregates
    total_cooling_output_kw: float = 0.0
    total_cooling_capacity_kw: float = 0.0
    cooling_load_pct: float = 0.0
    # Efficiency
    cop: float = 4.0  # Coefficient of Performance (cooling kW / electrical kW)
    cooling_power_kw: float = 0.0  # Electrical power consumed by cooling
    # Chilled water plant
    chw_plant_supply_temp_c: float = 7.0
    chw_plant_return_temp_c: float = 12.0
    chw_plant_delta_t_c: float = 5.0
    # Pump
    pump_power_kw: float = 2.0
    pump_flow_rate_lps: float = 20.0


class CoolingModel:
    """Simulates CRAC units, chilled water loop, and cooling tower.

    Key behaviours:
    - CRAC fan speed adjusts to match heat load
    - Supply air temperature depends on chilled water temp and fan speed
    - Chilled water delta-T increases with heat load
    - COP degrades at higher ambient/wet-bulb temperatures
    - Cooling tower approach temperature varies with wet-bulb
    - CRAC failures reduce total cooling capacity
    """

    # CRAC unit parameters
    CRAC_MAX_COOLING_KW = 50.0  # Per unit
    CRAC_MAX_AIRFLOW_CFM = 20000.0
    CRAC_MIN_SUPPLY_AIR_C = 12.0
    CRAC_MAX_RETURN_AIR_C = 45.0

    # Chilled water plant
    CHW_DESIGN_SUPPLY_C = 7.0
    CHW_DESIGN_RETURN_C = 12.0
    CHW_DESIGN_FLOW_LPS = 5.0  # Per CRAC unit

    # Cooling tower
    TOWER_DESIGN_APPROACH_C = 5.0  # Approach to wet-bulb

    # COP reference values
    COP_DESIGN = 4.5  # At design conditions (7°C CHW, 18°C WB)
    COP_MIN = 2.0  # Minimum COP at extreme conditions

    def __init__(self, config: SimConfig, rng_seed: int = 42):
        self.config = config
        self._rng = __import__("numpy").random.default_rng(rng_seed + 600)
        self._crac_units = config.thermal.crac_units
        # Track CRAC operational status
        self._crac_faults: dict[int, int] = {}  # unit_id -> fault_code (0 = ok)

    def step(
        self,
        total_it_heat_kw: float,
        ambient_temp_c: float = 22.0,
        crac_setpoints: dict[int, float] | None = None,
        crac_failed_units: set[int] | None = None,
        sim_time: float = 0.0,
    ) -> FacilityCoolingState:
        """Compute cooling system state.

        Args:
            total_it_heat_kw: total IT heat load to reject (kW)
            ambient_temp_c: outside dry-bulb temperature
            crac_setpoints: unit_id -> supply air setpoint override
            crac_failed_units: set of CRAC unit IDs that are failed
            sim_time: current simulation time
        """
        crac_setpoints = crac_setpoints or {}
        failed_units = crac_failed_units or set()

        # ── Wet-bulb temperature (approximation from dry-bulb) ──
        # Wet-bulb is typically 3-8°C below dry-bulb depending on humidity
        # Use sinusoidal daily variation
        hour = (sim_time / 3600.0) % 24.0
        wb_depression = 5.0 + 2.0 * math.sin(2 * math.pi * (hour - 6) / 24)
        wet_bulb = ambient_temp_c - wb_depression
        noise = self._rng.normal(0, 0.3)
        wet_bulb += noise

        # ── Cooling tower ──
        approach = self.TOWER_DESIGN_APPROACH_C + max(0, (wet_bulb - 18) * 0.15)
        condenser_supply = wet_bulb + approach
        condenser_return = condenser_supply + 5.0  # Design delta-T

        tower_fan_pct = min(100, max(20, (total_it_heat_kw / (self.CRAC_MAX_COOLING_KW * self._crac_units)) * 100))
        heat_rejection = total_it_heat_kw * 1.1  # Heat rejected = IT heat + compressor heat

        tower_state = CoolingTowerState(
            condenser_supply_temp_c=round(condenser_supply, 1),
            condenser_return_temp_c=round(condenser_return, 1),
            wet_bulb_temp_c=round(wet_bulb, 1),
            approach_temp_c=round(approach, 1),
            fan_speed_pct=round(tower_fan_pct, 1),
            heat_rejection_kw=round(heat_rejection, 1),
        )

        # ── Chilled water plant ──
        # CHW supply temp varies with condenser conditions
        chw_supply = self.CHW_DESIGN_SUPPLY_C + max(0, (condenser_supply - 28) * 0.2)
        chw_supply += self._rng.normal(0, 0.1)

        # Heat load determines CHW delta-T
        total_capacity = self.CRAC_MAX_COOLING_KW * max(1, self._crac_units - len(failed_units))
        load_fraction = min(1.0, total_it_heat_kw / max(1, total_capacity))

        chw_delta_t = 3.0 + load_fraction * 4.0  # 3-7°C delta-T
        chw_return = chw_supply + chw_delta_t

        # ── COP (Coefficient of Performance) ──
        # COP degrades at higher condenser temps and lower CHW temps
        cop = self.COP_DESIGN
        cop -= max(0, (condenser_supply - 28) * 0.08)  # Warmer ambient → lower COP
        cop -= max(0, (self.CHW_DESIGN_SUPPLY_C - chw_supply) * 0.1)  # Colder CHW → lower COP
        cop += max(0, (28 - condenser_supply) * 0.05)  # Cooler ambient → higher COP
        cop = max(self.COP_MIN, min(6.0, cop))

        # ── CRAC units ──
        heat_per_crac = total_it_heat_kw / max(1, self._crac_units - len(failed_units))
        crac_states = []
        total_cooling = 0.0
        total_cap_sum = 0.0

        for unit_id in range(self._crac_units):
            is_failed = unit_id in failed_units

            if is_failed:
                crac_states.append(CracUnitState(
                    unit_id=unit_id,
                    supply_air_temp_c=ambient_temp_c,
                    return_air_temp_c=ambient_temp_c,
                    fan_speed_pct=0,
                    airflow_cfm=0,
                    chw_supply_temp_c=chw_supply,
                    chw_return_temp_c=chw_supply,
                    chw_flow_rate_lps=0,
                    cooling_output_kw=0,
                    cooling_capacity_kw=self.CRAC_MAX_COOLING_KW,
                    load_pct=0,
                    operational=False,
                    fault_code=1,
                ))
                total_cap_sum += self.CRAC_MAX_COOLING_KW
                continue

            # This CRAC handles its share of the heat
            unit_load = min(self.CRAC_MAX_COOLING_KW, heat_per_crac)
            unit_load_pct = (unit_load / self.CRAC_MAX_COOLING_KW) * 100.0

            # Fan speed scales with load
            fan_pct = max(30, min(100, 30 + 70 * (unit_load / self.CRAC_MAX_COOLING_KW)))

            # Airflow scales with fan speed
            airflow = self.CRAC_MAX_AIRFLOW_CFM * (fan_pct / 100.0)

            # Supply air temp: depends on CHW supply and heat exchange effectiveness
            # Better heat exchange at higher fan speed
            effectiveness = 0.7 + 0.2 * (fan_pct / 100.0)
            supply_air = chw_supply + (1 - effectiveness) * (ambient_temp_c - chw_supply)

            # Apply setpoint override
            setpoint = crac_setpoints.get(unit_id)
            if setpoint is not None:
                supply_air = max(self.CRAC_MIN_SUPPLY_AIR_C, min(25, setpoint))

            # Return air temp based on heat absorbed
            return_air = supply_air + (unit_load / max(0.1, airflow * 0.0012))  # Simplified Q = m*cp*dT

            # Chilled water flow for this unit
            chw_flow = self.CHW_DESIGN_FLOW_LPS * (fan_pct / 100.0) * 1.2
            chw_unit_return = chw_supply + (unit_load / max(0.1, chw_flow * 4.186))

            crac_states.append(CracUnitState(
                unit_id=unit_id,
                supply_air_temp_c=round(supply_air, 1),
                return_air_temp_c=round(return_air, 1),
                fan_speed_pct=round(fan_pct, 1),
                airflow_cfm=round(airflow, 0),
                chw_supply_temp_c=round(chw_supply, 1),
                chw_return_temp_c=round(chw_unit_return, 1),
                chw_flow_rate_lps=round(chw_flow, 2),
                cooling_output_kw=round(unit_load, 1),
                cooling_capacity_kw=self.CRAC_MAX_COOLING_KW,
                load_pct=round(unit_load_pct, 1),
                operational=True,
                fault_code=0,
            ))
            total_cooling += unit_load
            total_cap_sum += self.CRAC_MAX_COOLING_KW

        # ── Cooling power consumption ──
        cooling_power = total_cooling / cop  # Electrical power for cooling

        # Pump power (scales with flow)
        total_flow = sum(c.chw_flow_rate_lps for c in crac_states)
        pump_power = 1.0 + total_flow * 0.15  # Base + flow-proportional

        return FacilityCoolingState(
            crac_units=crac_states,
            cooling_tower=tower_state,
            total_cooling_output_kw=round(total_cooling, 1),
            total_cooling_capacity_kw=round(total_cap_sum, 1),
            cooling_load_pct=round((total_cooling / max(1, total_cap_sum)) * 100, 1),
            cop=round(cop, 2),
            cooling_power_kw=round(cooling_power, 1),
            chw_plant_supply_temp_c=round(chw_supply, 1),
            chw_plant_return_temp_c=round(chw_return, 1),
            chw_plant_delta_t_c=round(chw_delta_t, 1),
            pump_power_kw=round(pump_power, 1),
            pump_flow_rate_lps=round(total_flow, 1),
        )

    def set_crac_fault(self, unit_id: int, fault_code: int = 1) -> None:
        """Mark a CRAC unit as faulted."""
        self._crac_faults[unit_id] = fault_code

    def clear_crac_fault(self, unit_id: int) -> None:
        """Clear CRAC fault."""
        self._crac_faults.pop(unit_id, None)

    def get_failed_units(self) -> set[int]:
        """Return set of failed CRAC unit IDs."""
        return {uid for uid, code in self._crac_faults.items() if code > 0}

    def reset(self) -> None:
        """Clear all faults."""
        self._crac_faults.clear()
