#!/usr/bin/env python3
"""Simulation verification: healthy DCN vs single abrupt leak fault.

Builds the small 20-node test network, runs a 24-hour hydraulic simulation
for both a healthy baseline and a leak scenario, then saves a comparison
plot to plots/verify_simulation.png.

Usage (from project root):
    python scripts/verify_simulation.py

Requirements: wntr, networkx, matplotlib, numpy
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make src/ importable when running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from generation.faults import inject_leak
from generation.simulate import run_simulation
from generation.topologies import build_small_network

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Scenario parameters ──────────────────────────────────────────────────────
DURATION        = 86_400   # 24 h in seconds
TIMESTEP        = 3_600    # 1 h reporting interval
LEAK_AREA       = 5e-4     # 5 cm²  →  ~10 L/s at 40 m pressure (~10 % of peak flow)
LEAK_ONSET_H    = 8        # fault begins at hour 8
LEAK_ONSET_S    = LEAK_ONSET_H * TIMESTEP
# ─────────────────────────────────────────────────────────────────────────────


def _pick_leak_node(wn: object) -> str:
    """Choose a leak node near the geometric centre of the network.

    Avoids the pump-entry node (which is atypically well-supplied) and
    selects the junction closest to the centroid of all junction coordinates.
    """
    import wntr as _wntr
    assert isinstance(wn, _wntr.network.WaterNetworkModel)
    coords = np.array([wn.get_node(n).coordinates for n in wn.junction_name_list])
    centroid = coords.mean(axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    idx = int(np.argmin(dists))
    return wn.junction_name_list[idx]


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "plots"
    out_dir.mkdir(exist_ok=True)

    # ── 1. Healthy baseline ──────────────────────────────────────────────────
    logger.info("Building small DCN network (seed=42)…")
    wn_h = build_small_network(seed=42)
    leak_node = _pick_leak_node(wn_h)
    logger.info("Leak node selected: %s (closest to network centroid)", leak_node)

    logger.info("Running HEALTHY simulation…")
    res_h = run_simulation(
        wn_h, duration=DURATION, hydraulic_timestep=TIMESTEP,
        solver="wntr", scenario_id="healthy",
    )

    # ── 2. Fault scenario (leak) ─────────────────────────────────────────────
    logger.info("Building fault network (leak at %s, onset t=%dh)…", leak_node, LEAK_ONSET_H)
    wn_f = build_small_network(seed=42)
    fault = inject_leak(
        wn_f,
        node_name=leak_node,
        area=LEAK_AREA,
        onset_time=LEAK_ONSET_S,
        onset_profile="abrupt",
    )

    logger.info("Running FAULT simulation…")
    res_f = run_simulation(
        wn_f, duration=DURATION, hydraulic_timestep=TIMESTEP,
        solver="wntr", scenario_id=f"leak_{leak_node}",
        fault_metadata=fault,
    )

    # ── 3. Extract DataFrames ────────────────────────────────────────────────
    p_h    = res_h.hydraulic.node["pressure"]     # shape: (timesteps, nodes)
    p_f    = res_f.hydraulic.node["pressure"]
    flow_h = res_h.hydraulic.link["flowrate"]     # shape: (timesteps, links)
    flow_f = res_f.hydraulic.link["flowrate"]

    times_h = p_h.index / 3600.0  # convert seconds → hours

    # Timestep index closest to 12 h (peak demand, well after leak onset)
    t12 = p_h.index[np.argmin(np.abs(p_h.index - 43_200))]

    # ── 4. Print summary ─────────────────────────────────────────────────────
    _print_summary(wn_h, p_h, p_f, t12, leak_node)

    # ── 5. Plot ──────────────────────────────────────────────────────────────
    fig = _make_figure(wn_h, p_h, p_f, flow_h, flow_f, times_h, t12, leak_node)
    out_path = out_dir / "verify_simulation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Plot saved → %s", out_path)
    print(f"\nPlot saved to: {out_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_summary(wn, p_h, p_f, t12, leak_node: str) -> None:
    print("\n" + "=" * 60)
    print("DCN Simulation Verification Summary")
    print("=" * 60)
    print(f"  Nodes  : {wn.num_junctions} junctions + 1 reservoir")
    print(f"  Links  : {wn.num_pipes} pipes + 1 pump")
    print(f"  Steps  : {len(p_h)} timesteps over 24 h")
    print()
    junctions = wn.junction_name_list
    print(f"  Mean pressure, all nodes (healthy) : {p_h[junctions].mean().mean():.2f} m")
    print(f"  Mean pressure, all nodes (fault)   : {p_f[junctions].mean().mean():.2f} m")
    print()
    print(f"  Pressure at {leak_node} at t=12 h:")
    print(f"    Healthy : {p_h.loc[t12, leak_node]:.3f} m")
    print(f"    Fault   : {p_f.loc[t12, leak_node]:.3f} m")
    drop = p_h.loc[t12, leak_node] - p_f.loc[t12, leak_node]
    print(f"    Drop    : {drop:.3f} m  ({100*drop/p_h.loc[t12, leak_node]:.1f} %)")
    print("=" * 60 + "\n")


def _make_figure(wn, p_h, p_f, flow_h, flow_f, times_h, t12, leak_node: str):
    junctions = wn.junction_name_list

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f"DCN Verification — Healthy vs Leak at {leak_node} (area={LEAK_AREA*1e4:.1f} cm², onset t={LEAK_ONSET_H}h)",
        fontsize=13, fontweight="bold",
    )
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

    ax_nh  = fig.add_subplot(gs[0, 0])   # network: healthy pressure at t12
    ax_nd  = fig.add_subplot(gs[0, 1])   # network: pressure drop at t12
    ax_ts  = fig.add_subplot(gs[0, 2])   # demand (flow from chiller) over time
    ax_p   = fig.add_subplot(gs[1, 0:2]) # pressure time series
    ax_f   = fig.add_subplot(gs[1, 2])   # flow in leak-adjacent pipes

    # ── Panel 1: healthy pressure map ────────────────────────────────────────
    vals_h = {n: p_h.loc[t12, n] for n in junctions}
    _draw_network(ax_nh, wn, vals_h, cmap="RdYlGn",
                  title=f"Pressure at t=12 h  (healthy) [m]",
                  highlight=None)

    # ── Panel 2: pressure drop map ───────────────────────────────────────────
    delta = {n: p_h.loc[t12, n] - p_f.loc[t12, n] for n in junctions}
    _draw_network(ax_nd, wn, delta, cmap="Reds",
                  title=f"Pressure drop fault−healthy at t=12 h [m]",
                  highlight=leak_node)

    # ── Panel 3: pump/chiller flow (total demand) ────────────────────────────
    pump_name = wn.pump_name_list[0]
    ax_ts.plot(times_h, np.abs(flow_h[pump_name]) * 1000, "steelblue", lw=2, label="Healthy")
    ax_ts.plot(times_h, np.abs(flow_f[pump_name]) * 1000, "tomato",    lw=2, label="Fault")
    ax_ts.axvline(LEAK_ONSET_H, color="black", ls=":", lw=1.5)
    ax_ts.set_xlabel("Time [h]")
    ax_ts.set_ylabel("Total system flow [L/s]")
    ax_ts.set_title("Total pump flow (healthy vs fault)")
    ax_ts.legend(fontsize=8)

    # ── Panel 4: pressure time series at leak node + 3 neighbours ────────────
    neighbours = _adjacent_junctions(wn, leak_node)[:3]
    for node in [leak_node] + neighbours:
        lw   = 2.5 if node == leak_node else 1.2
        alph = 1.0 if node == leak_node else 0.65
        ax_p.plot(times_h, p_h[node], color="steelblue", lw=lw, alpha=alph)
        ax_p.plot(times_h, p_f[node], color="tomato",    lw=lw, alpha=alph, ls="--")
        # Label the leak node lines
        if node == leak_node:
            ax_p.annotate(node, xy=(times_h[-1], float(p_h[node].iloc[-1])),
                          fontsize=7, color="steelblue", va="center")
    ax_p.axvline(LEAK_ONSET_H, color="black", ls=":", lw=1.5, label="Leak onset")
    ax_p.set_xlabel("Time [h]")
    ax_p.set_ylabel("Pressure [m]")
    ax_p.set_title(f"Pressure at {leak_node} (bold) and 3 neighbours\n(solid=healthy, dashed=fault)")
    ax_p.legend(
        handles=[
            mlines.Line2D([], [], color="steelblue", lw=2,  label="Healthy"),
            mlines.Line2D([], [], color="tomato",    lw=2, ls="--", label="Fault"),
            mlines.Line2D([], [], color="black",     lw=1.5, ls=":", label="Leak onset"),
        ],
        fontsize=8,
    )

    # ── Panel 5: flow in pipes connected to the leak node ────────────────────
    leak_pipes = _adjacent_pipes(wn, leak_node)
    colours = plt.cm.tab10(np.linspace(0, 0.5, len(leak_pipes)))
    for pipe, col in zip(leak_pipes, colours):
        ax_f.plot(times_h, np.abs(flow_h[pipe]) * 1000, color=col, lw=1.8,
                  label=pipe)
        ax_f.plot(times_h, np.abs(flow_f[pipe]) * 1000, color=col, lw=1.8,
                  ls="--", alpha=0.7)
    ax_f.axvline(LEAK_ONSET_H, color="black", ls=":", lw=1.5)
    ax_f.set_xlabel("Time [h]")
    ax_f.set_ylabel("Flow [L/s]")
    ax_f.set_title(f"Flow in pipes adjacent to {leak_node}\n(solid=healthy, dashed=fault)")
    ax_f.legend(fontsize=7)

    return fig


def _draw_network(ax, wn, node_values: dict, cmap: str, title: str,
                  highlight: str | None) -> None:
    """Draw the network as a scatter-plot coloured by node_values."""
    import networkx as nx

    # Node positions from WNTR coordinates
    pos = {n: wn.get_node(n).coordinates for n in wn.node_name_list}

    G = wn.to_graph().to_undirected()
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=1.0,
                           arrows=False)

    junctions = wn.junction_name_list
    xs = [pos[n][0] for n in junctions]
    ys = [pos[n][1] for n in junctions]
    vals = [node_values.get(n, 0.0) for n in junctions]
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1e-6

    sc = ax.scatter(xs, ys, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=55, zorder=4)
    plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)

    # Reservoir marker
    for rname in wn.reservoir_name_list:
        ax.scatter(*pos[rname], marker="s", s=120, color="#1f77b4",
                   zorder=5, label="Chiller")

    # Pump marker (midpoint of pump link)
    for pname in wn.pump_name_list:
        pump = wn.get_link(pname)
        mx = (pos[pump.start_node_name][0] + pos[pump.end_node_name][0]) / 2
        my = (pos[pump.start_node_name][1] + pos[pump.end_node_name][1]) / 2
        ax.scatter(mx, my, marker="^", s=100, color="green", zorder=5)

    # Leak / highlight node
    if highlight and highlight in pos:
        ax.scatter(*pos[highlight], marker="*", s=250, color="red",
                   zorder=6, label=f"Leak ({highlight})")
        ax.legend(fontsize=7, loc="upper right")

    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")


def _adjacent_junctions(wn, node_name: str) -> list[str]:
    """Return junction names directly connected to node_name by a pipe."""
    neighbours = []
    for _, pipe in wn.pipes():
        if pipe.start_node_name == node_name:
            n = pipe.end_node_name
        elif pipe.end_node_name == node_name:
            n = pipe.start_node_name
        else:
            continue
        if n in wn.junction_name_list and n not in neighbours:
            neighbours.append(n)
    return neighbours


def _adjacent_pipes(wn, node_name: str) -> list[str]:
    """Return names of pipes that start or end at node_name."""
    return [
        name for name, pipe in wn.pipes()
        if pipe.start_node_name == node_name or pipe.end_node_name == node_name
    ]


if __name__ == "__main__":
    main()
