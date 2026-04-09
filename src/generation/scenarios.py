"""Scenario orchestration and configuration for dataset generation.

A *scenario* is a single (topology, fault-config, random-seed) triple that
produces one labelled hydraulic time-series. This module is responsible for:

1. Loading scenario configs from YAML files (see configs/).
2. Expanding a config into a list of concrete Scenario objects (e.g.
   cartesian product of fault types × locations × severities).
3. Running a batch of scenarios, optionally in parallel via multiprocessing,
   and persisting results to data/raw/ as Parquet files.

Each scenario is fully described by its ScenarioConfig dataclass so that it
can be reproduced exactly from a config snapshot (random seed included).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from generation.faults import FaultMetadata, FaultType

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Complete specification for a single simulation scenario.

    Attributes:
        scenario_id: Unique string identifier (used as the output file stem).
        topology: One of 'small', 'medium', 'large'.
        topology_seed: Random seed for topology generation.
        simulation_duration: Total simulation time in seconds.
        hydraulic_timestep: Hydraulic solver timestep in seconds.
        fault_type: Fault type to inject, or None for a healthy baseline.
        fault_params: Fault-specific parameters passed to the injector.
        seed: Global random seed for this scenario.
        output_dir: Directory where raw results are written.
    """

    scenario_id: str
    topology: str
    topology_seed: int
    simulation_duration: float
    hydraulic_timestep: float
    fault_type: FaultType | None
    fault_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    output_dir: Path = Path("data/raw")


def load_configs(config_path: Path) -> list[ScenarioConfig]:
    """Load and expand scenario configurations from a YAML file.

    The YAML file may specify parameter sweeps using list values; this
    function performs a cartesian expansion to produce one ScenarioConfig
    per concrete parameter combination.

    Args:
        config_path: Path to the YAML scenario config file.

    Returns:
        List of fully-resolved ScenarioConfig objects.
    """
    raise NotImplementedError


def run_scenario(config: ScenarioConfig) -> Path:
    """Execute a single scenario end-to-end and persist results.

    Steps:
    1. Build the topology specified in config.topology.
    2. Clone the base network and inject the configured fault (if any).
    3. Run the WNTR extended-period simulation.
    4. Save raw results (Parquet) and fault metadata (JSON) to
       config.output_dir / config.scenario_id.

    Args:
        config: Fully-resolved scenario specification.

    Returns:
        Path to the directory containing saved results for this scenario.
    """
    raise NotImplementedError


def run_batch(
    configs: list[ScenarioConfig],
    n_workers: int = 1,
) -> list[Path]:
    """Run a batch of scenarios, optionally in parallel.

    Each scenario is independent, so they can safely be distributed across
    worker processes. Uses multiprocessing.Pool with n_workers processes.

    Args:
        configs: List of scenario configs to execute.
        n_workers: Number of parallel worker processes. Defaults to 1
            (serial execution) for reproducibility during development.

    Returns:
        List of output paths, one per scenario, in the same order as configs.
    """
    raise NotImplementedError
