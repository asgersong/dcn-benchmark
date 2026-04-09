"""Network topology builders for District Cooling Networks.

Generates synthetic DCN topologies of varying sizes and loopedness using
NetworkX, then converts them to WNTR WaterNetworkModel objects ready for
hydraulic simulation.

Three target scales (following DiTEC conventions):
- Small:  20-50 nodes, 1-2 loops  — unit tests, fast iteration
- Medium: 100-200 nodes, multiple loops — campus / small district
- Large:  500+ nodes, brownfield-style irregular loops — city district

DCN-specific topology notes:
- All topologies are closed-loop (highly looped, not tree-structured).
- Each topology must include at least one reservoir node (chiller plant),
  at least one variable-speed pump, and a mix of pipe diameters (DN200-DN800
  for mains, smaller for branch pipes).
- Base graphs are generated with NetworkX (grid or Watts-Strogatz for
  brownfield irregularity), then converted to WNTR objects.
- Only the supply network is modelled hydraulically; the return network is
  handled by the post-processing thermal layer (see processing/labeling.py).
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import wntr

logger = logging.getLogger(__name__)


def build_small_network(seed: int = 42) -> wntr.network.WaterNetworkModel:
    """Build a small DCN topology (20-50 nodes, 1-2 loops).

    Suitable for unit testing and rapid prototyping. Uses a simple grid
    graph with one added cross-link to create a single loop, then attaches
    a reservoir (chiller plant) and a pump.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A WNTR WaterNetworkModel representing the supply network.
    """
    raise NotImplementedError


def build_medium_network(seed: int = 42) -> wntr.network.WaterNetworkModel:
    """Build a medium DCN topology (100-200 nodes, multiple loops).

    Represents a campus or small district. Uses a 2-D grid graph with
    additional shortcut edges to introduce loop redundancy typical of
    brownfield DCNs.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A WNTR WaterNetworkModel representing the supply network.
    """
    raise NotImplementedError


def build_large_network(seed: int = 42) -> wntr.network.WaterNetworkModel:
    """Build a large brownfield DCN topology (500+ nodes).

    Represents a city-scale district. Uses a Watts-Strogatz small-world
    graph to capture the irregular loop structure of real brownfield
    networks, then scales pipe diameters and demands accordingly.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A WNTR WaterNetworkModel representing the supply network.
    """
    raise NotImplementedError


def networkx_to_wntr(
    graph: nx.Graph,
    reservoir_nodes: list[Any],
    pump_edges: list[tuple[Any, Any]],
    pipe_diameter_map: dict[tuple[Any, Any], float] | None = None,
) -> wntr.network.WaterNetworkModel:
    """Convert a NetworkX graph to a WNTR WaterNetworkModel.

    Assigns hydraulic attributes (pipe diameter, roughness, length) from
    the provided maps or sensible DCN defaults. Reservoir nodes become WNTR
    reservoirs; pump edges become WNTR pumps with a default head curve.

    Args:
        graph: Undirected NetworkX graph. Nodes may carry 'elevation' and
            'base_demand' attributes; edges may carry 'length' and 'diameter'.
        reservoir_nodes: Node IDs to register as WNTR reservoirs (chiller
            plants). Must be present in *graph*.
        pump_edges: Edge (u, v) pairs to register as WNTR pumps rather than
            plain pipes.
        pipe_diameter_map: Optional mapping from edge (u, v) to diameter in
            metres. Defaults to 0.3 m (DN300) for all pipes.

    Returns:
        Populated WNTR WaterNetworkModel ready for simulation.
    """
    raise NotImplementedError
