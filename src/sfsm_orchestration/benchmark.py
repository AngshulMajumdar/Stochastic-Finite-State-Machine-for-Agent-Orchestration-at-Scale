from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .core import BenchmarkConfig, accuracy, fsm_select, generate_agent_graph, sfsm_select
from .distributed import mapreduce_checksum


def load_config(path: str | Path) -> BenchmarkConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["accuracies"] = tuple(data["accuracies"])
    data["correct_beta"] = tuple(data["correct_beta"])
    data["incorrect_beta"] = tuple(data["incorrect_beta"])
    return BenchmarkConfig(**data)


def tune_threshold(p: float, config: BenchmarkConfig) -> float:
    rng = np.random.default_rng(config.master_seed + int(round(10_000 * p)) + 90_000_000)
    correct, scores, _ = generate_agent_graph(p, rng, config)
    best_accuracy, best_threshold = -1.0, 0.5
    for threshold in np.linspace(0.30, 0.90, 121):
        current = accuracy(correct, fsm_select(scores, float(threshold)))
        if current > best_accuracy:
            best_accuracy, best_threshold = current, float(threshold)
    return best_threshold


def median_ms(function, repeats: int) -> float:
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns(); function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.median(samples))


def run(config: BenchmarkConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {p: tune_threshold(p, config) for p in config.accuracies}
    rows: list[dict[str, float | int]] = []

    for p_index, p in enumerate(config.accuracies):
        for seed_index in range(config.test_seeds):
            seed = config.master_seed + 100_000 * p_index + seed_index
            correct, scores, priors = generate_agent_graph(
                p, np.random.default_rng(seed), config
            )
            fsm_indices = fsm_select(scores, thresholds[p])
            sfsm_indices = sfsm_select(scores, priors)
            fsm_ms = median_ms(lambda: fsm_select(scores, thresholds[p]), config.timing_repeats)
            sfsm_ms = median_ms(lambda: sfsm_select(scores, priors), config.timing_repeats)
            serial_checksum = int(sfsm_indices.sum(dtype=np.int64))
            mr_checksum, mr_count, mr_ms = mapreduce_checksum(
                scores, priors, config.workers, config.timing_repeats
            )
            if mr_count != config.components or mr_checksum != serial_checksum:
                raise RuntimeError("MapReduce output does not match serial SFSM selection")
            rows.append({
                "agent_accuracy": p, "seed": seed,
                "fsm_threshold": thresholds[p],
                "fsm_accuracy": accuracy(correct, fsm_indices),
                "sfsm_accuracy": accuracy(correct, sfsm_indices),
                "fsm_time_ms": fsm_ms,
                "sfsm_serial_time_ms": sfsm_ms,
                "sfsm_mapreduce_time_ms": mr_ms,
            })

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "million_agent_raw.csv", index=False)
    accuracy_rows, time_rows = [], []
    for p in config.accuracies:
        block = raw[np.isclose(raw.agent_accuracy, p)]
        accuracy_rows.append({
            "agent_accuracy": p,
            "fsm_mean": block.fsm_accuracy.mean(),
            "fsm_sd": block.fsm_accuracy.std(ddof=1),
            "sfsm_mean": block.sfsm_accuracy.mean(),
            "sfsm_sd": block.sfsm_accuracy.std(ddof=1),
            "gain_points": 100 * (block.sfsm_accuracy.mean() - block.fsm_accuracy.mean()),
        })
        time_rows.append({
            "agent_accuracy": p,
            "fsm_ms": block.fsm_time_ms.mean(),
            "fsm_sd": block.fsm_time_ms.std(ddof=1),
            "sfsm_serial_ms": block.sfsm_serial_time_ms.mean(),
            "sfsm_serial_sd": block.sfsm_serial_time_ms.std(ddof=1),
            "sfsm_mapreduce_ms": block.sfsm_mapreduce_time_ms.mean(),
            "sfsm_mapreduce_sd": block.sfsm_mapreduce_time_ms.std(ddof=1),
            "mapreduce_speedup": block.sfsm_serial_time_ms.mean() / block.sfsm_mapreduce_time_ms.mean(),
        })
    pd.DataFrame(accuracy_rows).to_csv(output_dir / "million_agent_accuracy.csv", index=False)
    pd.DataFrame(time_rows).to_csv(output_dir / "million_agent_time.csv", index=False)
    metadata = {
        **asdict(config), "total_agents": config.total_agents,
        "graph_structure": (
            f"{config.components} independent output-selection components with "
            f"{config.agents_per_component} agent nodes each"
        ),
        "terminal_output_rule": "one index into the actual generated outputs; no voting or synthesis",
        "fsm_policy": "first actual output crossing validation-tuned score threshold; fallback to first",
        "sfsm_policy": "actual output with maximum posterior probability of correctness",
        "timing_scope": "orchestration selection only; agent/tool execution excluded",
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "thresholds": {str(k): v for k, v in thresholds.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(accuracy_rows).to_string(index=False))
    print(pd.DataFrame(time_rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the million-agent FSM/SFSM benchmark")
    parser.add_argument("--config", required=True, help="JSON benchmark configuration")
    parser.add_argument("--output-dir", default="results/generated/million_agent")
    args = parser.parse_args()
    run(load_config(args.config), Path(args.output_dir))


if __name__ == "__main__":
    main()
