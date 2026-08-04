from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate publication figures from result CSVs")
    parser.add_argument("--results", default="results/reference/million_agent")
    parser.add_argument("--out", default="results/generated/figures")
    args = parser.parse_args()

    results = Path(args.results)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    accuracy = pd.read_csv(results / "million_agent_accuracy.csv")
    figure = plt.figure(figsize=(6.2, 3.9))
    plt.errorbar(
        accuracy.agent_accuracy * 100,
        accuracy.fsm_mean * 100,
        yerr=accuracy.fsm_sd * 100,
        marker="o",
        capsize=3,
        label="FSM",
    )
    plt.errorbar(
        accuracy.agent_accuracy * 100,
        accuracy.sfsm_mean * 100,
        yerr=accuracy.sfsm_sd * 100,
        marker="s",
        capsize=3,
        label="SFSM",
    )
    plt.xlabel("Mean agent accuracy (%)")
    plt.ylabel("Final actual-output accuracy (%)")
    plt.ylim(0, 102)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output / "accuracy_vs_agent_accuracy.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    raw = pd.read_csv(results / "million_agent_raw.csv")
    rows = []
    for agent_accuracy, group in raw.groupby("agent_accuracy"):
        rows.append(
            {
                "agent_accuracy": agent_accuracy,
                "fsm": group.fsm_time_ms.median(),
                "serial": group.sfsm_serial_time_ms.median(),
                "mapreduce": group.sfsm_mapreduce_time_ms.median(),
            }
        )
    timing = pd.DataFrame(rows)
    figure = plt.figure(figsize=(6.2, 3.9))
    plt.plot(timing.agent_accuracy * 100, timing.fsm, marker="o", label="FSM")
    plt.plot(timing.agent_accuracy * 100, timing.serial, marker="s", label="SFSM serial")
    plt.plot(
        timing.agent_accuracy * 100,
        timing.mapreduce,
        marker="^",
        label="SFSM MapReduce",
    )
    plt.xlabel("Mean agent accuracy (%)")
    plt.ylabel("Orchestration time (ms)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output / "million_agent_time.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    worker_path = results / "worker_scaling.csv"
    if worker_path.exists():
        worker = pd.read_csv(worker_path)
        figure = plt.figure(figsize=(6.2, 3.9))
        plt.plot(worker.workers, worker.speedup, marker="o", label="Measured")
        plt.plot(worker.workers, worker.workers, linestyle="--", label="Ideal")
        plt.xlabel("Map workers")
        plt.ylabel("Speed-up over serial SFSM")
        plt.xticks(worker.workers)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        figure.savefig(output / "mapreduce_speedup.png", dpi=240, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
