"""One-command orchestration of every reproducibility experiment."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def execute(label: str, arguments: list[str]) -> None:
    command = [PYTHON, *arguments]
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def quick_profile() -> None:
    execute(
        "semantic and distributed tests",
        ["-m", "pytest", "-q"],
    )
    execute(
        "reduced FSM/SFSM benchmark",
        [
            "experiments/run_million_agent.py",
            "--config", "experiments/configs/quick.json",
            "--output-dir", "results/generated/quick/million_agent",
            "--print-config",
        ],
    )
    execute(
        "reduced worker scaling",
        [
            "experiments/run_worker_scaling.py",
            "--config", "experiments/configs/quick.json",
            "--workers", "1", "2",
            "--repeats", "2",
            "--output", "results/generated/quick/million_agent/worker_scaling.csv",
        ],
    )
    execute(
        "small direct-enumeration exactness check",
        [
            "experiments/benchmark_fbs.py",
            "--out-dir", "results/generated/quick/exactness",
            "--sizes", "12",
            "--loops", "2",
            "--strengths", ".4",
            "--seeds", "1",
            "--brute-n", "12",
        ],
    )
    execute(
        "reference figure regeneration",
        [
            "experiments/make_figures.py",
            "--results", "results/reference/million_agent",
            "--out", "results/generated/quick/figures",
        ],
    )


def full_profile() -> None:
    execute("semantic and distributed tests", ["-m", "pytest", "-q"])
    execute(
        "million-agent FSM/SFSM benchmark",
        [
            "experiments/run_million_agent.py",
            "--config", "experiments/configs/full.json",
            "--output-dir", "results/generated/million_agent",
            "--print-config",
        ],
    )
    execute(
        "worker scaling",
        [
            "experiments/run_worker_scaling.py",
            "--config", "experiments/configs/full.json",
            "--output", "results/generated/million_agent/worker_scaling.csv",
        ],
    )
    execute(
        "270-instance loopy exactness benchmark",
        [
            "experiments/benchmark_fbs.py",
            "--out-dir", "results/generated/exactness/main",
            "--sizes", "32", "64", "128",
            "--loops", "2", "4", "6",
            "--strengths", ".4", ".8", "1.2",
            "--seeds", "10",
        ],
    )
    execute(
        "direct-enumeration validation",
        [
            "experiments/benchmark_fbs.py",
            "--out-dir", "results/generated/exactness/bruteforce",
            "--sizes", "16",
            "--loops", "2", "4", "6",
            "--strengths", ".4", ".8", "1.2",
            "--seeds", "10",
            "--brute-n", "16",
        ],
    )
    execute("independent consistency validation", ["experiments/validate_trifecta.py"])
    execute(
        "figure generation",
        [
            "experiments/make_figures.py",
            "--results", "results/generated/million_agent",
            "--out", "results/generated/figures",
        ],
    )
    execute(
        "reference verification",
        [
            "-m", "sfsm_orchestration.verify",
            "--reference", "results/reference/million_agent",
            "--generated", "results/generated/million_agent",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete SFSM release experiments")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    args = parser.parse_args()
    quick_profile() if args.profile == "quick" else full_profile()


if __name__ == "__main__":
    main()
