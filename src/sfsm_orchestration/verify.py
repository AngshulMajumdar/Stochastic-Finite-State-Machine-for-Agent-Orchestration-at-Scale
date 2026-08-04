from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare_csv(reference: Path, generated: Path, atol: float, rtol: float) -> None:
    a, b = pd.read_csv(reference), pd.read_csv(generated)
    if list(a.columns) != list(b.columns):
        raise AssertionError(f"column mismatch: {reference.name}")
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {reference.name}: {a.shape} != {b.shape}")
    for column in a.columns:
        if pd.api.types.is_numeric_dtype(a[column]):
            if not np.allclose(a[column], b[column], atol=atol, rtol=rtol, equal_nan=True):
                delta = np.nanmax(np.abs(a[column].to_numpy() - b[column].to_numpy()))
                raise AssertionError(f"{reference.name}:{column} maximum difference {delta}")
        elif not a[column].equals(b[column]):
            raise AssertionError(f"text mismatch: {reference.name}:{column}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated paper results with frozen references")
    parser.add_argument("--reference", default="results/reference/million_agent")
    parser.add_argument("--generated", default="results/generated/million_agent")
    parser.add_argument("--accuracy-atol", type=float, default=0.004)
    args = parser.parse_args()
    ref, gen = Path(args.reference), Path(args.generated)
    compare_csv(ref / "million_agent_accuracy.csv", gen / "million_agent_accuracy.csv", args.accuracy_atol, 0.0)
    # Timing is machine-dependent; verify schema and finite positive measurements only.
    timing = pd.read_csv(gen / "million_agent_time.csv")
    required = {"fsm_ms", "sfsm_serial_ms", "sfsm_mapreduce_ms"}
    if not required.issubset(timing.columns):
        raise AssertionError("generated timing table has the wrong schema")
    if not np.isfinite(timing[list(required)].to_numpy()).all() or (timing[list(required)] <= 0).any().any():
        raise AssertionError("generated timing values must be finite and positive")
    print("Reference verification passed.")


if __name__ == "__main__":
    main()
