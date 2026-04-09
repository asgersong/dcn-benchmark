"""Fault label generation for DCN benchmark graph datasets.

Translates FaultMetadata objects (produced during scenario generation) into
per-node and per-edge integer label arrays aligned with the graph structure
produced by graph_builder.py.

Three labelling schemes are supported:
1. Binary detection:      0 = healthy, 1 = faulty (any type).
2. Fault type classification: 0 = healthy, 1–5 = fault type index.
3. Fault localisation:   node/edge-level indicator (1 only at fault location).

Label arrays are stored as the `y` attribute on PyG Data objects, with the
labelling scheme recorded in Data.label_scheme for downstream use.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from generation.faults import FaultMetadata, FaultType

logger = logging.getLogger(__name__)

# Canonical integer codes for each fault type
FAULT_TYPE_CODES: dict[FaultType | None, int] = {
    None: 0,
    "leak": 1,
    "sensor_drift": 2,
    "low_delta_t": 3,
    "valve_fault": 4,
    "pump_degradation": 5,
}

LabelScheme = Literal["binary", "classification", "localization"]


def make_node_labels(
    node_names: list[str],
    fault_metadata: FaultMetadata | None,
    scheme: LabelScheme = "binary",
) -> np.ndarray:
    """Generate a per-node integer label array.

    For localization, only the node named in fault_metadata.location is
    labelled as faulty. For binary and classification schemes, all nodes in
    the network receive the same label (reflecting the scenario-level fault).

    Args:
        node_names: Ordered list of node names matching the graph node order.
        fault_metadata: Fault description for this scenario, or None for
            healthy baseline.
        scheme: Labelling scheme to apply.

    Returns:
        Integer numpy array of shape (n_nodes,).
    """
    raise NotImplementedError


def make_edge_labels(
    edge_names: list[str],
    fault_metadata: FaultMetadata | None,
    scheme: LabelScheme = "binary",
) -> np.ndarray:
    """Generate a per-edge integer label array.

    Analogous to make_node_labels() but for edge-level faults (valve faults,
    pump degradation, leaks modelled on links).

    Args:
        edge_names: Ordered list of edge identifiers (pipe/valve/pump names).
        fault_metadata: Fault description for this scenario, or None for
            healthy baseline.
        scheme: Labelling scheme to apply.

    Returns:
        Integer numpy array of shape (n_edges,).
    """
    raise NotImplementedError


def fault_is_active(
    fault_metadata: FaultMetadata,
    simulation_time: float,
) -> bool:
    """Determine whether a fault is active at a given simulation timestep.

    For abrupt faults the fault is active from onset_time onwards. For
    progressive faults it ramps linearly from 0 to full severity over one
    hour after onset_time.

    Args:
        fault_metadata: Fault description including onset_time and
            onset_profile.
        simulation_time: Current simulation time in seconds.

    Returns:
        True if the fault should be considered active at this timestep.
    """
    raise NotImplementedError
