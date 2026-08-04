"""Run the complete FSM-versus-SFSM million-agent benchmark.

This script is intentionally a visible, direct entry point. The numerical kernels live
in the installable ``sfsm_orchestration`` package; this file owns command-line parsing,
configuration reporting and result-directory selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sfsm_orchestration.benchmark import load_config, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the million-agent workload, tune the FSM threshold on an "
            "independent validation stream, evaluate FSM and SFSM over all test "
            "seeds, verify serial/MapReduce equality, and write raw plus aggregate results."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/full.json",
        help="JSON configuration file. Use quick.json for a reduced smoke run.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/generated/million_agent",
        help="Directory receiving CSV and JSON outputs.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration before execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.print_config:
        print(json.dumps({
            "accuracies": config.accuracies,
            "components": config.components,
            "agents_per_component": config.agents_per_component,
            "total_agents": config.total_agents,
            "test_seeds": config.test_seeds,
            "timing_repeats": config.timing_repeats,
            "workers": config.workers,
            "master_seed": config.master_seed,
        }, indent=2))
    run(config, Path(args.output_dir))


if __name__ == "__main__":
    main()
