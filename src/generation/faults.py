"""Fault injection module for DCN benchmark scenarios.

Implements the five fault types defined for the benchmark dataset. Each
injector takes a cloned WNTR WaterNetworkModel (or its simulation results)
and modifies it in-place, returning metadata that describes the injected
fault for use as ground-truth labels.

Fault types:
1. Leaks          — WNTR add_leak(); vary size, location, onset profile.
2. Sensor drift   — Post-simulation perturbation of pressure/flow/temperature.
3. Low Delta-T    — Reduced temperature differential at substations (altered
                    demand or bypass flows).
4. Valve faults   — Stuck open/closed or partial blockage via WNTR controls.
5. Pump degradation — Modified pump head curves simulating wear.

All parameters are expected to come from a YAML config (see configs/) so that
nothing is hardcoded here. Fault onset can be abrupt (step change) or
progressive (linear ramp).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import wntr

logger = logging.getLogger(__name__)

FaultType = Literal["leak", "sensor_drift", "low_delta_t", "valve_fault", "pump_degradation"]


@dataclass
class FaultMetadata:
    """Ground-truth description of a single injected fault.

    Attributes:
        fault_type: One of the five canonical fault type strings.
        location: Node or link name where the fault was injected.
        severity: Normalised severity in [0, 1] (0 = no fault, 1 = maximum).
        onset_time: Simulation time (seconds) at which the fault starts.
        onset_profile: 'abrupt' or 'progressive'.
        extra: Additional fault-specific metadata (e.g. drift slope).
    """

    fault_type: FaultType
    location: str
    severity: float
    onset_time: float
    onset_profile: Literal["abrupt", "progressive"] = "abrupt"
    extra: dict[str, Any] = field(default_factory=dict)


def inject_leak(
    wn: wntr.network.WaterNetworkModel,
    node_name: str,
    area: float,
    onset_time: float,
    onset_profile: Literal["abrupt", "progressive"] = "abrupt",
) -> FaultMetadata:
    """Inject a pipe leak at a junction node using WNTR's built-in mechanism.

    Calls wn.get_node(node_name).add_leak() and adds the appropriate WNTR
    controls for onset timing. Leak area ranges from ~0.5% to 10% of the
    equivalent pipe cross-section.

    Args:
        wn: WNTR network model to modify in-place.
        node_name: Name of the junction node at which the leak originates.
        area: Orifice area in m² (controls flow rate out of the system).
        onset_time: Simulation time in seconds when the leak begins.
        onset_profile: 'abrupt' for a step change, 'progressive' for a linear
            ramp over one hour.

    Returns:
        FaultMetadata describing the injected fault.
    """
    node = wn.get_node(node_name)

    if onset_profile == "abrupt":
        node.add_leak(wn, area=area, discharge_coeff=0.75,
                      start_time=int(onset_time), end_time=None)
    else:
        # Progressive: approximate a 1-hour linear ramp with two discrete steps.
        # Step 1 — half area from onset; Step 2 — full area one hour later.
        node.add_leak(wn, area=area * 0.5, discharge_coeff=0.75,
                      start_time=int(onset_time),
                      end_time=int(onset_time) + 3600)
        node.add_leak(wn, area=area, discharge_coeff=0.75,
                      start_time=int(onset_time) + 3600, end_time=None)

    # Severity normalised against 0.01 m² (≈ a 10 cm-diameter orifice,
    # roughly the upper bound of a credible pipe leak).
    severity = float(np.clip(area / 0.01, 0.0, 1.0))

    logger.info(
        "Injected %s leak at %s: area=%.5f m²  onset=%gs  severity=%.2f",
        onset_profile, node_name, area, onset_time, severity,
    )
    return FaultMetadata(
        fault_type="leak",
        location=node_name,
        severity=severity,
        onset_time=float(onset_time),
        onset_profile=onset_profile,
    )


def inject_sensor_drift(
    results: dict[str, Any],
    sensor_name: str,
    sensor_type: Literal["pressure", "flow", "temperature"],
    drift_rate: float,
    noise_std: float,
    onset_time: float,
    rng: np.random.Generator | None = None,
) -> FaultMetadata:
    """Apply linear drift and additive Gaussian noise to a sensor time-series.

    This is a post-simulation perturbation: *results* is modified in-place
    so that the named sensor's readings deviate from ground truth. The
    unperturbed values are preserved under a '_clean' suffix key for label
    generation.

    Args:
        results: Dictionary of WNTR simulation result DataFrames, keyed by
            quantity (e.g. 'pressure', 'flow').
        sensor_name: Column name (node or link) in the relevant DataFrame.
        sensor_type: Which result table to perturb.
        drift_rate: Drift slope in [units]/s (e.g. Pa/s for pressure).
        noise_std: Standard deviation of additive Gaussian noise.
        onset_time: Simulation time in seconds when drift begins.
        rng: Optional NumPy random generator for reproducibility.

    Returns:
        FaultMetadata describing the injected fault.
    """
    raise NotImplementedError


def inject_low_delta_t(
    wn: wntr.network.WaterNetworkModel,
    substation_node: str,
    bypass_fraction: float,
    onset_time: float,
) -> FaultMetadata:
    """Model Low Delta-T syndrome at a substation node.

    Low Delta-T arises when the temperature differential between supply and
    return drops below the design value, reducing chiller efficiency. It is
    modelled here as a bypass flow (a fraction of the substation demand flows
    directly from supply to return without heat exchange) or equivalently as
    a demand increase that dilutes the return temperature.

    Args:
        wn: WNTR network model to modify in-place.
        substation_node: Junction node representing the affected substation.
        bypass_fraction: Fraction of design flow that bypasses heat exchange
            (0 = normal, 1 = full bypass).
        onset_time: Simulation time in seconds when the fault begins.

    Returns:
        FaultMetadata describing the injected fault.
    """
    raise NotImplementedError


def inject_valve_fault(
    wn: wntr.network.WaterNetworkModel,
    valve_name: str,
    mode: Literal["stuck_open", "stuck_closed", "partial_blockage"],
    blockage_fraction: float = 1.0,
    onset_time: float = 0.0,
) -> FaultMetadata:
    """Simulate a valve fault by modifying WNTR valve controls.

    For 'stuck_open' and 'stuck_closed' modes, the valve setting is fixed
    regardless of control rules. For 'partial_blockage', the minor loss
    coefficient is increased proportionally.

    Args:
        wn: WNTR network model to modify in-place.
        valve_name: Name of the WNTR valve element.
        mode: Type of valve fault to inject.
        blockage_fraction: For partial_blockage only — fraction of full
            blockage (0 = open, 1 = fully blocked).
        onset_time: Simulation time in seconds when the fault begins.

    Returns:
        FaultMetadata describing the injected fault.
    """
    raise NotImplementedError


def inject_pump_degradation(
    wn: wntr.network.WaterNetworkModel,
    pump_name: str,
    degradation_factor: float,
    onset_time: float,
) -> FaultMetadata:
    """Degrade a pump by scaling down its head curve.

    Multiplies the head values in the pump's head curve by
    (1 - degradation_factor), simulating wear-induced efficiency loss.
    A degradation_factor of 0.0 means no change; 1.0 means the pump
    produces zero head.

    Args:
        wn: WNTR network model to modify in-place.
        pump_name: Name of the WNTR pump element.
        degradation_factor: Fractional reduction in pump head in [0, 1).
        onset_time: Simulation time in seconds when degradation begins.

    Returns:
        FaultMetadata describing the injected fault.
    """
    raise NotImplementedError
