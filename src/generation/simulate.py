"""WNTR simulation wrapper for DCN hydraulic modelling.

Provides a thin, config-driven interface around WNTR's EpanetSimulator and
WNTRSimulator. All simulation parameters (duration, timestep, solver choice)
come from a ScenarioConfig; nothing is hardcoded here.

DCN-specific adaptations:
- Cooling load demand patterns follow diurnal + seasonal HVAC schedules
  rather than the stochastic potable-water patterns in WNTR examples.
- After hydraulic simulation, a lightweight thermal post-processing step
  computes supply and return temperatures at each node, using the hydraulic
  results (flow rates) as input. This is separate from WNTR.
- Results are returned as a SimulationResult dataclass that bundles the
  WNTR SimulationResults object with the thermal results and fault metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import wntr

from generation.faults import FaultMetadata

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Bundle of hydraulic results, thermal results, and fault metadata.

    Attributes:
        hydraulic: WNTR SimulationResults object (pressure, flow, demand,
            head at every node/link for every timestep).
        temperature_supply: DataFrame of supply-side node temperatures
            [K or °C] indexed by time, columns by node name.
        temperature_return: DataFrame of return-side node temperatures
            indexed by time, columns by node name.
        fault_metadata: Description of the injected fault, or None for
            healthy baseline scenarios.
        scenario_id: Identifier of the scenario that produced these results.
    """

    hydraulic: Any  # wntr.sim.results.SimulationResults
    temperature_supply: pd.DataFrame
    temperature_return: pd.DataFrame
    fault_metadata: FaultMetadata | None
    scenario_id: str


def run_simulation(
    wn: wntr.network.WaterNetworkModel,
    duration: float,
    hydraulic_timestep: float,
    solver: str = "epanet",
    scenario_id: str = "unnamed",
    fault_metadata: FaultMetadata | None = None,
) -> SimulationResult:
    """Run an extended-period hydraulic simulation and thermal post-processing.

    Args:
        wn: Fully configured WNTR network model (topology + demands + faults).
        duration: Total simulation duration in seconds.
        hydraulic_timestep: Hydraulic solver timestep in seconds.
        solver: 'epanet' (EpanetSimulator) or 'wntr' (WNTRSimulator).
            EpanetSimulator is faster for large networks.
        scenario_id: Identifier carried through to the result bundle.
        fault_metadata: Pre-computed fault metadata to attach to results,
            or None for healthy baseline scenarios.

    Returns:
        SimulationResult containing hydraulic and thermal results.

    Raises:
        RuntimeError: If the WNTR solver fails to converge.
    """
    raise NotImplementedError


def compute_thermal_profile(
    hydraulic: Any,
    wn: wntr.network.WaterNetworkModel,
    t_supply_source: float = 6.0,
    t_return_design: float = 13.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute supply and return temperature profiles from hydraulic results.

    Uses a steady-state energy balance at each node for every timestep:
    heat absorbed at substation nodes is determined by the building cooling
    load (demand), and temperatures propagate along pipes using the flow
    direction from WNTR results.

    This is a custom thermal layer on top of WNTR — WNTR itself does not
    model temperature in closed-loop cooling systems.

    Args:
        hydraulic: WNTR SimulationResults object.
        wn: The WaterNetworkModel used for the simulation (needed for pipe
            geometry and node types).
        t_supply_source: Supply temperature at the chiller plant in °C.
            Typical DCN value: 5–7°C.
        t_return_design: Design return temperature at the chiller plant in °C.
            Typical DCN value: 12–15°C.

    Returns:
        Tuple of (temperature_supply, temperature_return) DataFrames,
        each indexed by simulation time and with node names as columns.
    """
    raise NotImplementedError


def apply_cooling_load_pattern(
    wn: wntr.network.WaterNetworkModel,
    pattern_type: str = "diurnal",
    seed: int = 42,
) -> wntr.network.WaterNetworkModel:
    """Attach a DCN-appropriate cooling load demand pattern to all substations.

    Unlike potable-water demand, DCN cooling loads follow HVAC schedules:
    - Diurnal: peak in working hours (08:00–18:00), low overnight.
    - Seasonal: higher in summer, near-zero in winter.
    - No stochastic per-household variation.

    Args:
        wn: WNTR network model. Junction nodes flagged as substations (via
            node tag or naming convention) receive the pattern.
        pattern_type: 'diurnal' for a daily profile, 'seasonal' for a
            365-day envelope (requires duration ≥ 1 year).
        seed: Random seed for minor load variation between substations.

    Returns:
        The modified WaterNetworkModel (modified in-place, also returned for
        chaining).
    """
    raise NotImplementedError
