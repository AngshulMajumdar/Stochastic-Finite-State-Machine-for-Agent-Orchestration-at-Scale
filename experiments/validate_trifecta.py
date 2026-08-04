"""Small exact checks for the Feynman--Bethe--Schwinger--Dyson manuscript.

The script uses only NumPy and enumerates small finite models.  It validates:
1. the deterministic limit of a stochastic transition;
2. equality of trajectory enumeration and transfer-operator evaluation;
3. exactness of belief propagation on a tree;
4. the discrete Schwinger--Dyson identity under local spin flips.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np


def deterministic_limit() -> None:
    deterministic = np.array([1.0, 0.0])
    print("E1 deterministic limit")
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        stochastic = np.array([1.0 - eps, eps])
        tv = 0.5 * np.abs(stochastic - deterministic).sum()
        print(f"  epsilon={eps:8.1e}  TV={tv:8.1e}")


def path_sum_check(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    n_state, n_agent, horizon = 4, 3, 5
    mu = rng.random(n_state)
    mu /= mu.sum()
    terminal = rng.random(n_state) + 0.1

    # K[t, x, a, x'] is a positive trajectory factor, not necessarily a
    # transition probability; summing over a gives the transfer matrix.
    factors = rng.random((horizon, n_state, n_agent, n_state)) + 0.05
    transfer = factors.sum(axis=2)

    propagated = mu.copy()
    for t in range(horizon):
        propagated = propagated @ transfer[t]
    z_transfer = float(propagated @ terminal)

    z_enum = 0.0
    for states in product(range(n_state), repeat=horizon + 1):
        x0 = states[0]
        for agents in product(range(n_agent), repeat=horizon):
            weight = mu[x0] * terminal[states[-1]]
            for t, a in enumerate(agents):
                weight *= factors[t, states[t], a, states[t + 1]]
            z_enum += weight

    print("\nE2 path sum = transfer operator")
    print(f"  enumeration: {z_enum:.12f}")
    print(f"  transfer:    {z_transfer:.12f}")
    print(f"  abs error:   {abs(z_enum - z_transfer):.3e}")


def all_spins(n: int) -> np.ndarray:
    return np.asarray(list(product((-1.0, 1.0), repeat=n)), dtype=float)


def exact_ising(
    x: np.ndarray,
    h: np.ndarray,
    edges: list[tuple[int, int]],
    coupling: np.ndarray,
) -> np.ndarray:
    score = x @ h
    for k, (i, j) in enumerate(edges):
        score += coupling[k] * x[:, i] * x[:, j]
    score -= score.max()
    p = np.exp(score)
    return p / p.sum()


def bp_ising(
    h: np.ndarray,
    edges: list[tuple[int, int]],
    coupling: np.ndarray,
    max_iter: int = 10_000,
    tol: float = 1e-13,
    damping: float = 0.5,
) -> np.ndarray:
    n = h.size
    neighbours: list[list[int]] = [[] for _ in range(n)]
    jmap: dict[tuple[int, int], float] = {}
    for k, (i, j) in enumerate(edges):
        neighbours[i].append(j)
        neighbours[j].append(i)
        jmap[(i, j)] = coupling[k]
        jmap[(j, i)] = coupling[k]

    msg = {(i, j): 0.0 for i in range(n) for j in neighbours[i]}
    for _ in range(max_iter):
        updated: dict[tuple[int, int], float] = {}
        max_change = 0.0
        for i in range(n):
            for j in neighbours[i]:
                field = h[i]
                for k in neighbours[i]:
                    if k == j:
                        continue
                    argument = np.tanh(jmap[(i, k)]) * msg[(k, i)]
                    argument = np.clip(argument, -1 + 1e-15, 1 - 1e-15)
                    field += np.arctanh(argument)
                value = np.tanh(field)
                value = (1.0 - damping) * msg[(i, j)] + damping * value
                updated[(i, j)] = value
                max_change = max(max_change, abs(value - msg[(i, j)]))
        msg = updated
        if max_change < tol:
            break

    means = np.empty(n)
    for i in range(n):
        field = h[i]
        for k in neighbours[i]:
            argument = np.tanh(jmap[(i, k)]) * msg[(k, i)]
            argument = np.clip(argument, -1 + 1e-15, 1 - 1e-15)
            field += np.arctanh(argument)
        means[i] = np.tanh(field)
    return means


def sd_residuals(
    x: np.ndarray,
    p: np.ndarray,
    h: np.ndarray,
    edges: list[tuple[int, int]],
    coupling: np.ndarray,
) -> np.ndarray:
    n = h.size
    neighbours: list[list[int]] = [[] for _ in range(n)]
    jmap: dict[tuple[int, int], float] = {}
    for k, (i, j) in enumerate(edges):
        neighbours[i].append(j)
        neighbours[j].append(i)
        jmap[(i, j)] = coupling[k]
        jmap[(j, i)] = coupling[k]

    residuals: list[float] = []
    for i in range(n):
        local_field = h[i].copy() if np.ndim(h[i]) else h[i]
        local_field = np.full(x.shape[0], float(local_field))
        for j in neighbours[i]:
            local_field += jmap[(i, j)] * x[:, j]
        exponential = np.exp(-2.0 * x[:, i] * local_field)

        # f=1
        residuals.append(float(p @ (exponential - 1.0)))

        # f=x_j for each neighbour; T_i leaves x_j unchanged.
        for j in neighbours[i]:
            residuals.append(float(p @ (x[:, j] * (exponential - 1.0))))
    return np.asarray(residuals)


def bethe_and_sd_check(seed: int = 19) -> None:
    rng = np.random.default_rng(seed)
    n = 8
    x = all_spins(n)
    h = rng.normal(0.0, 0.25, size=n)

    tree_edges = [(int(rng.integers(0, i)), i) for i in range(1, n)]
    tree_j = rng.normal(0.0, 0.35, size=len(tree_edges))
    p_tree = exact_ising(x, h, tree_edges, tree_j)
    exact_tree_means = p_tree @ x
    bp_tree_means = bp_ising(h, tree_edges, tree_j)
    tree_error = np.max(np.abs(exact_tree_means - bp_tree_means))
    tree_sd = sd_residuals(x, p_tree, h, tree_edges, tree_j)

    edge_set = {tuple(sorted(edge)) for edge in tree_edges}
    chords: list[tuple[int, int]] = []
    for candidate in ((0, n - 1), (1, n - 2), (0, n - 2), (2, n - 1)):
        key = tuple(sorted(candidate))
        if key not in edge_set:
            chords.append(candidate)
            edge_set.add(key)
        if len(chords) == 2:
            break
    loopy_edges = tree_edges + chords
    loopy_j = np.concatenate([tree_j, rng.normal(0.0, 0.6, size=len(chords))])
    p_loop = exact_ising(x, h, loopy_edges, loopy_j)
    exact_loop_means = p_loop @ x
    bp_loop_means = bp_ising(h, loopy_edges, loopy_j)
    loop_error = np.max(np.abs(exact_loop_means - bp_loop_means))
    loop_sd = sd_residuals(x, p_loop, h, loopy_edges, loopy_j)

    print("\nE3 Bethe and Schwinger--Dyson checks")
    print(f"  tree BP max marginal error:  {tree_error:.3e}")
    print(f"  loopy BP max marginal error: {loop_error:.3e}")
    print(f"  tree exact SD max residual:  {np.max(np.abs(tree_sd)):.3e}")
    print(f"  loop exact SD max residual:  {np.max(np.abs(loop_sd)):.3e}")


def main() -> None:
    deterministic_limit()
    path_sum_check()
    bethe_and_sd_check()


if __name__ == "__main__":
    main()
