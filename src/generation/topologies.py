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
import numpy as np
import wntr

logger = logging.getLogger(__name__)


def build_small_network(seed: int = 42) -> wntr.network.WaterNetworkModel:
    """Build a small organic DCN topology (20–30 nodes, 3–5 loops).

    Generates a realistic supply-network topology using an MST-plus-loops
    approach that mirrors how real DCNs are built and expanded:

    1. Scatter substations with mild clustering (buildings cluster in blocks).
    2. Place the chiller plant to the west of the demand area.
    3. Build a minimum spanning tree from the chiller outward — the natural
       radial backbone of an initially-built DCN.
    4. Add back a small number of short non-MST edges as cross-connections,
       representing the redundancy loops added during network expansion.
    5. Assign pipe diameters hierarchically by BFS depth from the chiller:
       trunk (DN400), primary mains (DN300), service branches (DN200).
    6. Pipe lengths are Euclidean distances between node coordinates (m).

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A WNTR WaterNetworkModel representing the supply network.
    """
    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel()

    # ── Demand pattern ───────────────────────────────────────────────────────
    # Diurnal cooling load: low overnight, peak during working hours.
    # Unlike potable water, DCN loads follow HVAC schedules — no stochastic
    # per-household noise, just a smooth building-occupancy envelope.
    diurnal = [
        0.10, 0.10, 0.10, 0.10, 0.15, 0.30,   # 00–05 h: overnight low
        0.60, 0.85, 1.00, 1.00, 1.00, 1.00,   # 06–11 h: morning ramp + peak
        1.00, 0.95, 0.95, 0.90, 0.85, 0.75,   # 12–17 h: sustained peak
        0.55, 0.40, 0.30, 0.20, 0.15, 0.10,   # 18–23 h: evening ramp-down
    ]
    wn.add_pattern("DIURNAL", diurnal)

    # ── Node placement ───────────────────────────────────────────────────────
    # 20 substations scattered across 3 demand clusters (representing blocks
    # of buildings in a district). Coordinates are in metres.
    n_junctions = 20
    n_clusters = 3
    area_size = 1200.0  # m, extent of the district

    cluster_centres = rng.uniform(200, area_size - 200, (n_clusters, 2))
    junction_coords: list[tuple[float, float]] = []
    for i in range(n_junctions):
        centre = cluster_centres[i % n_clusters]
        # ±150 m scatter around each cluster centre
        xy = centre + rng.normal(0, 150, 2)
        xy = np.clip(xy, 0, area_size)
        junction_coords.append((float(xy[0]), float(xy[1])))

    # Chiller plant: placed 350 m west of the leftmost cluster centre,
    # mid-height — a typical peripheral location for a large chiller plant.
    leftmost = cluster_centres[np.argmin(cluster_centres[:, 0])]
    chiller_coord = (float(leftmost[0]) - 350.0, float(leftmost[1]))

    # ── Build MST backbone ───────────────────────────────────────────────────
    # Index 0 = chiller, indices 1..n = junctions (matching junction_coords).
    all_coords = [chiller_coord] + junction_coords
    n_total = len(all_coords)

    G_full = nx.Graph()
    for i, xy in enumerate(all_coords):
        G_full.add_node(i, coord=xy)
    for i in range(n_total):
        for j in range(i + 1, n_total):
            xi, yi = all_coords[i]
            xj, yj = all_coords[j]
            dist = float(np.hypot(xi - xj, yi - yj))
            G_full.add_edge(i, j, length=dist)

    mst = nx.minimum_spanning_tree(G_full, weight="length")

    # ── Add cross-connections (redundancy loops) ─────────────────────────────
    # Pick the shortest non-MST edges up to a distance cap. These represent
    # cross-connections added during network expansion for N-1 redundancy.
    # Targeting ~4 extra edges → 4 independent loops in a 20-node network.
    n_cross = 4
    max_cross_dist = 500.0  # m — only connect nodes that are reasonably close

    non_mst = [
        (u, v, G_full[u][v]["length"])
        for u, v in G_full.edges()
        if not mst.has_edge(u, v) and G_full[u][v]["length"] <= max_cross_dist
        # Skip any edge that touches the chiller node (node 0); the chiller
        # connects to the network through the pump only.
        and u != 0 and v != 0
    ]
    non_mst.sort(key=lambda t: t[2])
    for u, v, length in non_mst[:n_cross]:
        mst.add_edge(u, v, length=length)

    # ── Diameter assignment by BFS depth from pump entry ────────────────────
    # Find the junction directly connected to the chiller in the MST — this
    # is where the pump terminates, i.e., the network entry point.
    chiller_neighbours = [n for n in mst.neighbors(0) if n != 0]
    pump_entry = min(chiller_neighbours,
                     key=lambda n: G_full[0][n]["length"])

    # BFS depths from the pump entry node (junction-only subgraph).
    junction_subgraph = mst.subgraph([n for n in mst.nodes if n != 0])
    bfs_depth: dict[int, int] = {
        node: depth
        for depth, nodes in enumerate(nx.bfs_layers(junction_subgraph, pump_entry))
        for node in nodes
    }

    def _diameter_from_depth(u: int, v: int) -> float:
        """Hierarchical diameter: trunk → mains → service branches."""
        depth = min(bfs_depth.get(u, 99), bfs_depth.get(v, 99))
        if depth <= 1:
            return 0.40   # DN400 — trunk mains
        elif depth <= 3:
            return 0.30   # DN300 — primary distribution
        else:
            return 0.20   # DN200 — service branches

    # ── Populate WNTR model ──────────────────────────────────────────────────
    wn.add_reservoir("CHILLER", base_head=0.0, coordinates=chiller_coord)

    for i, (x, y) in enumerate(junction_coords):
        name = f"J-{i + 1}"
        demand = float(rng.uniform(0.003, 0.007))  # m³/s (3–7 L/s per substation)
        wn.add_junction(
            name,
            base_demand=demand,
            demand_pattern="DIURNAL",
            elevation=0.0,
            coordinates=(x, y),
        )

    # Pump: CHILLER → pump_entry junction.
    # Design point: Q ≈ 0.08 m³/s at H ≈ 55 m (total peak ~100 L/s).
    wn.add_curve(
        "PUMP-CURVE", "HEAD",
        [
            (0.00, 68.0),
            (0.04, 63.0),
            (0.08, 55.0),
            (0.12, 43.0),
            (0.16, 27.0),
            (0.20,  5.0),
        ],
    )
    wn.add_pump(
        "PUMP-1", "CHILLER", f"J-{pump_entry}",
        pump_type="HEAD", pump_parameter="PUMP-CURVE",
    )

    # Pipes from every MST edge that is not the chiller→pump_entry edge.
    pipe_idx = 1
    for u, v, data in mst.edges(data=True):
        if 0 in (u, v):
            continue  # chiller side handled by the pump above
        n1 = f"J-{u}"
        n2 = f"J-{v}"
        length = data["length"]
        diam = _diameter_from_depth(u, v)
        wn.add_pipe(
            f"P-{pipe_idx}", n1, n2,
            length=max(length, 10.0),   # enforce 10 m minimum (avoids near-zero)
            diameter=diam,
            roughness=130.0,            # Hazen-Williams C, steel in good condition
            minor_loss=0.0,
        )
        pipe_idx += 1

    logger.info(
        "Built small DCN network: %d junctions, %d pipes, 1 pump, 1 reservoir  "
        "(pump entry: J-%d, cross-connections added: %d)",
        wn.num_junctions, wn.num_pipes, pump_entry,
        mst.number_of_edges() - (n_total - 1),   # edges above MST = loops
    )
    return wn


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
