from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sfsm_orchestration.benchmark import load_config
from sfsm_orchestration.core import generate_agent_graph, sfsm_select
from sfsm_orchestration.distributed import mapreduce_checksum


def median_ms(function, repeats: int) -> float:
    function()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure SFSM MapReduce worker scaling")
    parser.add_argument("--config", default="experiments/configs/full.json")
    parser.add_argument("--accuracy", type=float, default=0.4)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", default="results/generated/million_agent/worker_scaling.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(config.master_seed + 71)
    _, scores, priors = generate_agent_graph(args.accuracy, rng, config)
    serial_indices = sfsm_select(scores, priors)
    serial_checksum = int(serial_indices.sum(dtype=np.int64))
    serial_ms = median_ms(lambda: sfsm_select(scores, priors), args.repeats)

    rows = []
    for workers in args.workers:
        if workers == 1:
            elapsed = serial_ms
        else:
            checksum, count, elapsed = mapreduce_checksum(scores, priors, workers, args.repeats)
            if checksum != serial_checksum or count != config.components:
                raise RuntimeError(f"worker-count {workers} does not match serial output")
        rows.append({
            "workers": workers,
            "time_ms": elapsed,
            "speedup": serial_ms / elapsed,
            "parallel_efficiency": serial_ms / (workers * elapsed),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    metadata = {
        "config": args.config,
        "accuracy": args.accuracy,
        "workers": args.workers,
        "repeats": args.repeats,
        "total_agents": config.total_agents,
        "timing_scope": "orchestration kernel only; process-pool startup and memory-map creation excluded",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
