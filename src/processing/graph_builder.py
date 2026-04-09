"""Convert raw WNTR simulation results into PyTorch Geometric Data objects.

This module bridges the gap between WNTR's tabular simulation output and the
graph-structured input required by GNN models. Each simulation timestep (or
a time-aggregated window) becomes one PyG Data object.

Graph structure (single-graph representation — see CLAUDE.md for dual-graph
alternative):
- Nodes  = junctions (+ virtual nodes for reservoirs/tanks)
- Edges  = pipes, pumps, valves (directed, supply direction)
- Node features: [pressure, demand, T_supply, T_return]
- Edge features: [flow_rate, diameter, length, roughness]

Temporal snapshots are stored as a list of Data objects or a PyG
TemporalData object depending on the downstream use case.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import wntr

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency when torch is not installed
try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Data = Any  # type: ignore[misc]


def build_graph_dataset(
    result_dir: Path,
    scenario_id: str,
    snapshot_interval: int = 1,
) -> list[Data]:
    """Build a list of PyG Data objects from a saved scenario result directory.

    Reads hydraulic results (Parquet) and fault metadata (JSON) from
    result_dir / scenario_id, then calls build_snapshot() for each selected
    timestep.

    Args:
        result_dir: Root directory containing per-scenario subdirectories
            (typically data/raw/).
        scenario_id: Subdirectory name matching the scenario that produced
            the results.
        snapshot_interval: Take every Nth timestep as a graph snapshot.
            1 = all timesteps, 10 = every 10th, etc.

    Returns:
        List of PyG Data objects, one per selected timestep.
    """
    raise NotImplementedError


def build_snapshot(
    wn: wntr.network.WaterNetworkModel,
    hydraulic_at_t: dict[str, pd.Series],
    thermal_at_t: dict[str, pd.Series],
    fault_labels: dict[str, int] | None = None,
) -> Data:
    """Build a single PyG Data object for one simulation timestep.

    Constructs node feature matrix X, edge index, edge feature matrix E, and
    optional fault labels from the hydraulic and thermal results at time t.

    Args:
        wn: WNTR network model (used for static edge attributes: diameter,
            length, roughness).
        hydraulic_at_t: Dict mapping quantity names ('pressure', 'demand',
            'flow') to Series indexed by node/link name.
        thermal_at_t: Dict mapping 'T_supply' and 'T_return' to Series
            indexed by node name.
        fault_labels: Optional dict mapping node/link name to integer fault
            label (0 = healthy, 1+ = fault type index). If None, all labels
            are set to 0 (healthy baseline).

    Returns:
        PyG Data object with attributes x, edge_index, edge_attr, y.
    """
    raise NotImplementedError


def wntr_to_networkx(wn: wntr.network.WaterNetworkModel) -> nx.DiGraph:
    """Extract the directed hydraulic graph from a WNTR network model.

    Pipes, pumps, and valves become directed edges (source → destination in
    the supply flow direction). Node attributes (elevation, base_demand) and
    edge attributes (diameter, length, roughness) are preserved as NetworkX
    graph attributes.

    Args:
        wn: WNTR network model.

    Returns:
        Directed NetworkX graph with node and edge attributes populated.
    """
    raise NotImplementedError
