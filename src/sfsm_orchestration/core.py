from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    accuracies: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    components: int = 125_000
    agents_per_component: int = 8
    test_seeds: int = 10
    timing_repeats: int = 7
    workers: int = 4
    master_seed: int = 20_260_730
    correct_beta: tuple[float, float] = (4.0, 2.0)
    incorrect_beta: tuple[float, float] = (2.0, 4.0)
    reliability_span: float = 0.12

    @property
    def total_agents(self) -> int:
        return self.components * self.agents_per_component


def calibrated_reliabilities(mean_accuracy: float, agents: int, span: float) -> np.ndarray:
    if not 0 < mean_accuracy <= 1:
        raise ValueError("mean_accuracy must lie in (0, 1]")
    if agents < 1:
        raise ValueError("agents must be positive")
    if mean_accuracy == 1.0:
        return np.ones(agents, dtype=np.float64)

    offsets = np.linspace(-span, span, agents)
    base = math.log(mean_accuracy / (1.0 - mean_accuracy))
    lo, hi = -10.0, 10.0
    for _ in range(80):
        shift = (lo + hi) / 2.0
        q = 1.0 / (1.0 + np.exp(-(base + offsets + shift)))
        if q.mean() > mean_accuracy:
            hi = shift
        else:
            lo = shift
    return 1.0 / (1.0 + np.exp(-(base + offsets + (lo + hi) / 2.0)))


def generate_agent_graph(
    mean_accuracy: float,
    rng: np.random.Generator,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a block-sparse graph with one actual output per agent node.

    Rows are independent output-selection components. Columns are agent nodes.
    ``correct`` is used only by the evaluator; the orchestrators observe verifier
    scores and reliability priors, not correctness.
    """
    reliabilities = calibrated_reliabilities(
        mean_accuracy, config.agents_per_component, config.reliability_span
    )
    shape = (config.components, config.agents_per_component)
    correct = rng.random(shape) < reliabilities[None, :]
    scores = np.empty(shape, dtype=np.float64)
    n_correct = int(correct.sum())
    n_incorrect = correct.size - n_correct
    if n_correct:
        scores[correct] = rng.beta(*config.correct_beta, size=n_correct)
    if n_incorrect:
        scores[~correct] = rng.beta(*config.incorrect_beta, size=n_incorrect)
    priors = np.broadcast_to(reliabilities, shape).copy()
    return correct, scores, priors


def fsm_select(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Hard FSM: return the first actual output crossing a fixed edge threshold."""
    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional array")
    accepted = scores >= threshold
    has_accepted = accepted.any(axis=1)
    first_accepted = accepted.argmax(axis=1)
    return np.where(has_accepted, first_accepted, 0).astype(np.int64, copy=False)


def sfsm_log_posterior(scores: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Posterior log odds under the calibrated Beta(4,2)/Beta(2,4) model."""
    if scores.shape != priors.shape:
        raise ValueError("scores and priors must have identical shapes")
    eps = np.finfo(np.float64).eps
    clipped_scores = np.clip(scores, eps, 1.0 - eps)
    clipped_priors = np.clip(priors, eps, 1.0 - eps)
    return (
        np.log(clipped_priors / (1.0 - clipped_priors))
        + 2.0 * np.log(clipped_scores / (1.0 - clipped_scores))
    )


def sfsm_select(scores: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Select one actual produced output by maximum posterior correctness."""
    if np.all(priors == 1.0):
        return np.zeros(scores.shape[0], dtype=np.int64)
    return sfsm_log_posterior(scores, priors).argmax(axis=1).astype(np.int64)


def accuracy(correct: np.ndarray, selected: np.ndarray) -> float:
    if correct.ndim != 2 or selected.shape != (correct.shape[0],):
        raise ValueError("selected must contain one output index per component")
    if np.any(selected < 0) or np.any(selected >= correct.shape[1]):
        raise ValueError("selected contains an index that is not an actual produced output")
    return float(correct[np.arange(correct.shape[0]), selected].mean())
