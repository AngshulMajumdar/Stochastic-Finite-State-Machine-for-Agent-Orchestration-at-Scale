from __future__ import annotations

import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .core import sfsm_select


def chunk_ranges(length: int, workers: int) -> list[tuple[int, int]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    return [(length * w // workers, length * (w + 1) // workers) for w in range(workers)]


def _memmap_worker(args: tuple[str, str, tuple[int, int], str, int, int]) -> tuple[int, int]:
    score_path, prior_path, shape, dtype, start, stop = args
    scores = np.memmap(score_path, mode="r", dtype=np.dtype(dtype), shape=shape)
    priors = np.memmap(prior_path, mode="r", dtype=np.dtype(dtype), shape=shape)
    selected = sfsm_select(scores[start:stop], priors[start:stop])
    return int(selected.sum(dtype=np.int64)), int(selected.size)


def mapreduce_checksum(
    scores: np.ndarray,
    priors: np.ndarray,
    workers: int,
    repeats: int = 1,
) -> tuple[int, int, float]:
    """Map SFSM selection over row partitions and reduce compact checksums.

    Read-only NumPy memory maps make the implementation portable across Linux,
    macOS and Windows without copying the complete million-agent arrays into
    every worker process.
    """
    with tempfile.TemporaryDirectory(prefix="sfsm-mapreduce-") as temp_dir:
        temp = Path(temp_dir)
        score_path = temp / "scores.dat"
        prior_path = temp / "priors.dat"
        score_map = np.memmap(score_path, mode="w+", dtype=scores.dtype, shape=scores.shape)
        prior_map = np.memmap(prior_path, mode="w+", dtype=priors.dtype, shape=priors.shape)
        score_map[:] = scores
        prior_map[:] = priors
        score_map.flush(); prior_map.flush()
        del score_map, prior_map

        jobs = [
            (str(score_path), str(prior_path), scores.shape, str(scores.dtype), start, stop)
            for start, stop in chunk_ranges(scores.shape[0], workers)
        ]
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            list(pool.map(_memmap_worker, jobs))  # warm-up
            timings = []
            checksum = count = 0
            for _ in range(repeats):
                start_time = time.perf_counter_ns()
                mapped = list(pool.map(_memmap_worker, jobs))
                checksum = sum(item[0] for item in mapped)
                count = sum(item[1] for item in mapped)
                timings.append((time.perf_counter_ns() - start_time) / 1e6)
        return checksum, count, float(np.median(timings))
