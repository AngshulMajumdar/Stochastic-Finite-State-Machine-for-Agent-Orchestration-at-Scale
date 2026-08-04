"""Reproducible benchmarks for Feynman--Bethe--Schwinger--Dyson orchestration.

The benchmark models binary orchestration decisions with a sparse Ising factor graph

    p(s) proportional exp(h^T s + sum_(i,j) J_ij s_i s_j).

Variable 0 is the terminal action. A random tree is augmented with a bounded number
of verification/dependency chords. The proposed solver:

1. runs loopy Bethe inference;
2. evaluates discrete Schwinger--Dyson star residuals;
3. chooses a feedback set from the cyclic core using the residuals;
4. conditions on that set and evaluates every remaining tree exactly;
5. sums the conditioned contributions (the finite Feynman path sum).

For this bounded-loop benchmark family the result is exact. Small cases are also
checked by full 2^n enumeration. Baselines are a hard greedy FSM, independent mean
field, finite-budget Gibbs Monte Carlo, and loopy Bethe.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from numba import njit
from scipy.special import expit, logsumexp

SPINS = np.asarray([-1.0, 1.0])
EPS = 1e-9


@dataclass
class Instance:
    h: np.ndarray
    edges: list[tuple[int, int]]
    coupling: np.ndarray
    n_loops: int
    strength: float

    @property
    def n(self) -> int:
        return int(self.h.size)

    def matrix(self) -> np.ndarray:
        mat = np.zeros((self.n, self.n), dtype=float)
        for (i, j), value in zip(self.edges, self.coupling):
            mat[i, j] = value
            mat[j, i] = value
        return mat


def make_instance(n: int, n_loops: int, strength: float, rng: np.random.Generator) -> Instance:
    """Generate a sparse orchestration graph with an ambiguous terminal variable."""
    edges: list[tuple[int, int]] = []
    edge_set: set[tuple[int, int]] = set()

    # A rooted execution tree. The terminal/decision variable 0 receives several
    # direct agent reports, while deeper nodes represent delegated subwork.
    for i in range(1, n):
        if i <= 4 or rng.random() < 0.12:
            parent = 0
        else:
            parent = int(rng.integers(0, i))
        e = (min(parent, i), max(parent, i))
        edges.append(e)
        edge_set.add(e)

    # Verification/shared-source factors create loops. Some touch the terminal
    # neighbourhood because that is where cross-agent consistency matters.
    attempts = 0
    while len(edges) < (n - 1 + n_loops):
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("Could not generate requested chords")
        if rng.random() < 0.45:
            i = 0
            j = int(rng.integers(5, n))
        else:
            i, j = rng.choice(n, size=2, replace=False)
            i, j = int(i), int(j)
        if i == j:
            continue
        e = (min(i, j), max(i, j))
        if e in edge_set:
            continue
        edges.append(e)
        edge_set.add(e)

    h = rng.normal(0.0, 0.28, size=n)
    h[0] = rng.normal(0.0, 0.06)  # terminal action is intentionally evidence-driven

    coupling = np.empty(len(edges), dtype=float)
    for k in range(len(edges)):
        scale = strength * (1.20 if k >= n - 1 else 0.85)
        magnitude = rng.uniform(0.35 * scale, 1.35 * scale)
        coupling[k] = magnitude if rng.random() < 0.5 else -magnitude

    return Instance(h=h, edges=edges, coupling=coupling, n_loops=n_loops, strength=strength)


def graph_from(inst: Instance, excluded: set[int] | None = None) -> nx.Graph:
    excluded = excluded or set()
    g = nx.Graph()
    g.add_nodes_from(i for i in range(inst.n) if i not in excluded)
    g.add_edges_from((i, j) for i, j in inst.edges if i not in excluded and j not in excluded)
    return g


def energy_score(spins: np.ndarray, inst: Instance) -> float:
    value = float(inst.h @ spins)
    for (i, j), coupling in zip(inst.edges, inst.coupling):
        value += float(coupling * spins[i] * spins[j])
    return value


def brute_force_means(inst: Instance) -> tuple[np.ndarray, float]:
    if inst.n > 24:
        raise ValueError("Brute force restricted to n <= 24")
    states = np.asarray(list(product((-1.0, 1.0), repeat=inst.n)), dtype=float)
    scores = states @ inst.h
    for (i, j), coupling in zip(inst.edges, inst.coupling):
        scores += coupling * states[:, i] * states[:, j]
    log_z = float(logsumexp(scores))
    weights = np.exp(scores - log_z)
    return weights @ states, log_z


def forest_exact(h: np.ndarray, edges: list[tuple[int, int]], coupling: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact means and log partition function for a forest."""
    n = int(h.size)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    active = np.ones(n, dtype=bool)
    for (i, j), value in zip(edges, coupling):
        neighbours[i].append((j, float(value)))
        neighbours[j].append((i, float(value)))

    # Memoized directed log-messages. Each array is indexed by receiver spin.
    memo: dict[tuple[int, int], np.ndarray] = {}

    def message(i: int, j: int) -> np.ndarray:
        key = (i, j)
        if key in memo:
            return memo[key]
        out = np.empty(2, dtype=float)
        coupling_ij = next(value for node, value in neighbours[i] if node == j)
        for sj_idx, sj in enumerate(SPINS):
            terms = np.empty(2, dtype=float)
            for si_idx, si in enumerate(SPINS):
                val = h[i] * si + coupling_ij * si * sj
                for k, _ in neighbours[i]:
                    if k != j:
                        val += message(k, i)[si_idx]
                terms[si_idx] = val
            out[sj_idx] = logsumexp(terms)
        memo[key] = out
        return out

    means = np.zeros(n, dtype=float)
    log_z = 0.0
    seen: set[int] = set()
    for root in range(n):
        if root in seen:
            continue
        component = nx.node_connected_component(nx.Graph([(i, j) for i, j in edges]), root) if edges else {root}
        # nx.Graph with no edge omits isolated nodes; correct that case.
        if not component:
            component = {root}
        seen.update(component)
        root_log = np.empty(2, dtype=float)
        for r_idx, sr in enumerate(SPINS):
            val = h[root] * sr
            for k, _ in neighbours[root]:
                val += message(k, root)[r_idx]
            root_log[r_idx] = val
        log_z += float(logsumexp(root_log))

        for i in component:
            log_b = np.empty(2, dtype=float)
            for si_idx, si in enumerate(SPINS):
                val = h[i] * si
                for k, _ in neighbours[i]:
                    val += message(k, i)[si_idx]
                log_b[si_idx] = val
            prob = np.exp(log_b - logsumexp(log_b))
            means[i] = float(prob @ SPINS)

    # The graph construction above misses isolated vertices when edges is nonempty.
    connected_nodes = {u for edge in edges for u in edge}
    for i in range(n):
        if i not in connected_nodes:
            local = np.asarray([-h[i], h[i]], dtype=float)
            log_z += float(logsumexp(local)) if i not in seen else 0.0
            prob = np.exp(local - logsumexp(local))
            means[i] = float(prob @ SPINS)
    return means, log_z


def forest_exact_safe(h: np.ndarray, edges: list[tuple[int, int]], coupling: np.ndarray) -> tuple[np.ndarray, float]:
    """Forest exact inference with explicit components, including isolated nodes."""
    n = int(h.size)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    if not nx.is_forest(g):
        raise ValueError("Conditioned graph is not a forest")

    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), value in zip(edges, coupling):
        neighbours[i].append((j, float(value)))
        neighbours[j].append((i, float(value)))
    memo: dict[tuple[int, int], np.ndarray] = {}

    def message(i: int, j: int) -> np.ndarray:
        key = (i, j)
        if key in memo:
            return memo[key]
        jij = next(value for node, value in neighbours[i] if node == j)
        out = np.empty(2, dtype=float)
        for sj_idx, sj in enumerate(SPINS):
            vals = []
            for si_idx, si in enumerate(SPINS):
                val = h[i] * si + jij * si * sj
                for k, _ in neighbours[i]:
                    if k != j:
                        val += message(k, i)[si_idx]
                vals.append(val)
            out[sj_idx] = logsumexp(vals)
        memo[key] = out
        return out

    means = np.empty(n, dtype=float)
    total_log_z = 0.0
    for component in nx.connected_components(g):
        root = next(iter(component))
        root_log = np.empty(2, dtype=float)
        for idx, spin in enumerate(SPINS):
            val = h[root] * spin
            for k, _ in neighbours[root]:
                val += message(k, root)[idx]
            root_log[idx] = val
        total_log_z += float(logsumexp(root_log))
        for i in component:
            log_b = np.empty(2, dtype=float)
            for idx, spin in enumerate(SPINS):
                val = h[i] * spin
                for k, _ in neighbours[i]:
                    val += message(k, i)[idx]
                log_b[idx] = val
            prob = np.exp(log_b - logsumexp(log_b))
            means[i] = float(prob @ SPINS)
    return means, total_log_z


def bp_full(inst: Instance, seed: int = 0, restarts: int = 3) -> dict:
    """Damped loopy sum-product for a pairwise Ising model.

    Messages are cavity magnetizations indexed by directed edges. The update is
    vectorized over all directed edges, which keeps the benchmark reproducible
    without making Python dictionary operations the dominant cost.
    """
    n = inst.n
    m = len(inst.edges)
    src = np.empty(2 * m, dtype=np.int64)
    dst = np.empty(2 * m, dtype=np.int64)
    rev = np.empty(2 * m, dtype=np.int64)
    jdir = np.empty(2 * m, dtype=float)
    neighbours: list[list[int]] = [[] for _ in range(n)]
    directed_index: dict[tuple[int, int], int] = {}
    for k, ((i, j), value) in enumerate(zip(inst.edges, inst.coupling)):
        e, r = 2 * k, 2 * k + 1
        src[e], dst[e], rev[e], jdir[e] = i, j, r, value
        src[r], dst[r], rev[r], jdir[r] = j, i, e, value
        directed_index[(i, j)] = e
        directed_index[(j, i)] = r
        neighbours[i].append(j)
        neighbours[j].append(i)

    tanh_j = np.tanh(jdir)
    rng = np.random.default_rng(seed)
    best: dict | None = None
    for restart in range(restarts):
        msg = np.zeros(2 * m, dtype=float) if restart == 0 else rng.uniform(-0.2, 0.2, size=2 * m)
        converged = False
        damping = 0.35
        for iteration in range(1200):
            argument = np.clip(tanh_j * msg, -1 + 1e-14, 1 - 1e-14)
            u = np.arctanh(argument)
            incoming = np.zeros(n, dtype=float)
            np.add.at(incoming, dst, u)
            cavity_field = inst.h[src] + incoming[src] - u[rev]
            target = np.tanh(cavity_field)
            updated = (1.0 - damping) * msg + damping * target
            max_change = float(np.max(np.abs(updated - msg))) if updated.size else 0.0
            msg = updated
            if max_change < 1e-9:
                converged = True
                break

        argument = np.clip(tanh_j * msg, -1 + 1e-14, 1 - 1e-14)
        u = np.arctanh(argument)
        incoming = np.zeros(n, dtype=float)
        np.add.at(incoming, dst, u)
        full_field = inst.h + incoming
        means = np.tanh(full_field)
        node_prob = [np.asarray([(1.0 - mean) / 2.0, (1.0 + mean) / 2.0]) for mean in means]

        pair_prob: dict[tuple[int, int], np.ndarray] = {}
        pair_entropy = 0.0
        pair_energy = 0.0
        for k, ((i, j), jij) in enumerate(zip(inst.edges, inst.coupling)):
            e, r = 2 * k, 2 * k + 1
            cavity_i = float(np.arctanh(np.clip(msg[e], -1 + 1e-14, 1 - 1e-14)))
            cavity_j = float(np.arctanh(np.clip(msg[r], -1 + 1e-14, 1 - 1e-14)))
            log_b = np.empty((2, 2), dtype=float)
            for ii, si in enumerate(SPINS):
                for jj, sj in enumerate(SPINS):
                    log_b[ii, jj] = jij * si * sj + cavity_i * si + cavity_j * sj
            b = np.exp(log_b - logsumexp(log_b))
            pair_prob[(i, j)] = b
            pair_entropy -= float(np.sum(b * np.log(np.clip(b, EPS, 1.0))))
            corr = float(sum(b[ii, jj] * SPINS[ii] * SPINS[jj] for ii in range(2) for jj in range(2)))
            pair_energy -= float(jij * corr)

        degrees = np.asarray([len(row) for row in neighbours], dtype=int)
        node_entropies = np.asarray([
            -float(np.sum(p * np.log(np.clip(p, EPS, 1.0)))) for p in node_prob
        ])
        energy = -float(inst.h @ means) + pair_energy
        entropy_bethe = pair_entropy + float(np.sum((1 - degrees) * node_entropies))
        free_energy = energy - entropy_bethe

        candidate = {
            "means": means,
            "node_prob": node_prob,
            "pair_prob": pair_prob,
            "messages": msg,
            "free_energy": float(free_energy),
            "converged": converged,
            "iterations": iteration + 1,
            "neighbours": neighbours,
            "directed_index": directed_index,
        }
        if best is None or candidate["free_energy"] < best["free_energy"]:
            best = candidate
    assert best is not None
    return best

def sd_star_residuals(inst: Instance, bp: dict) -> np.ndarray:
    """Low-order Schwinger--Dyson residual under a pairwise closure.

    For each directed edge i--j, retain the Bethe pair belief b_ij exactly and
    close all other neighbours of i by their Bethe means. The flip identity

        E[exp(-2 s_i H_i)-1] = 0

    is then evaluated under this pairwise closure. The largest incident residual
    is used as the score of variable i. This costs O(|E|), unlike explicit star
    enumeration at high-degree aggregation nodes.
    """
    residual = np.zeros(inst.n, dtype=float)
    means = np.asarray(bp["means"], dtype=float)
    neighbours = bp["neighbours"]
    jmap: dict[tuple[int, int], float] = {}
    for (i, j), value in zip(inst.edges, inst.coupling):
        jmap[(i, j)] = float(value)
        jmap[(j, i)] = float(value)

    for i in range(inst.n):
        best = 0.0
        total_mean_field = float(inst.h[i]) + sum(jmap[(i, k)] * means[k] for k in neighbours[i])
        for j in neighbours[i]:
            edge = (min(i, j), max(i, j))
            b = bp["pair_prob"][edge]
            value = 0.0
            for si_idx, si in enumerate(SPINS):
                for sj_idx, sj in enumerate(SPINS):
                    probability = b[si_idx, sj_idx] if i < j else b[sj_idx, si_idx]
                    local_field = total_mean_field - jmap[(i, j)] * means[j] + jmap[(i, j)] * sj
                    exponent = float(np.clip(-2.0 * si * local_field, -60.0, 60.0))
                    value += float(probability) * (math.exp(exponent) - 1.0)
            best = max(best, abs(value))
        residual[i] = best
    return residual

def sd_feedback_set(inst: Instance, residual: np.ndarray) -> list[int]:
    g = graph_from(inst)
    selected: list[int] = []
    while not nx.is_forest(g):
        core = nx.k_core(g, k=2)
        if core.number_of_nodes() == 0:
            break
        cycle_counts = {node: 0 for node in core.nodes}
        for cycle in nx.cycle_basis(core):
            for node in cycle:
                cycle_counts[node] += 1
        def score(node: int) -> tuple[float, int, int]:
            # Residual is the primary signal; cycle participation and degree break ties.
            return (
                abs(float(residual[node])) * (1.0 + cycle_counts[node]),
                cycle_counts[node],
                core.degree[node],
            )
        node = max(core.nodes, key=score)
        selected.append(int(node))
        g.remove_node(node)
    return selected


def exact_cutset(inst: Instance, cutset: Sequence[int]) -> tuple[np.ndarray, float]:
    cutset = list(cutset)
    cset = set(cutset)
    remaining = [i for i in range(inst.n) if i not in cset]
    old_to_new = {old: new for new, old in enumerate(remaining)}

    internal_edges: list[tuple[int, int]] = []
    internal_j: list[float] = []
    cross: list[tuple[int, int, float]] = []
    cut_edges: list[tuple[int, int, float]] = []
    for (i, j), value in zip(inst.edges, inst.coupling):
        if i in cset and j in cset:
            cut_edges.append((i, j, float(value)))
        elif i in cset or j in cset:
            c, r = (i, j) if i in cset else (j, i)
            cross.append((c, r, float(value)))
        else:
            internal_edges.append((old_to_new[i], old_to_new[j]))
            internal_j.append(float(value))

    g_rem = nx.Graph()
    g_rem.add_nodes_from(range(len(remaining)))
    g_rem.add_edges_from(internal_edges)
    if not nx.is_forest(g_rem):
        raise ValueError("Cutset does not break all cycles")

    log_weights: list[float] = []
    conditional_means: list[np.ndarray] = []
    assignments: list[np.ndarray] = []
    for values in product((-1.0, 1.0), repeat=len(cutset)):
        cmap = {node: float(value) for node, value in zip(cutset, values)}
        constant = sum(float(inst.h[node] * cmap[node]) for node in cutset)
        constant += sum(value * cmap[i] * cmap[j] for i, j, value in cut_edges)
        h_rem = inst.h[remaining].copy()
        for c, r, value in cross:
            h_rem[old_to_new[r]] += value * cmap[c]
        if remaining:
            m_rem, log_z_rem = forest_exact_safe(h_rem, internal_edges, np.asarray(internal_j))
        else:
            m_rem = np.asarray([], dtype=float)
            log_z_rem = 0.0
        full_mean = np.empty(inst.n, dtype=float)
        for node, value in cmap.items():
            full_mean[node] = value
        for idx, node in enumerate(remaining):
            full_mean[node] = m_rem[idx]
        log_weights.append(constant + log_z_rem)
        conditional_means.append(full_mean)
        assignments.append(np.asarray(values, dtype=float))

    log_weights_arr = np.asarray(log_weights, dtype=float)
    log_z = float(logsumexp(log_weights_arr))
    weights = np.exp(log_weights_arr - log_z)
    means = np.sum(weights[:, None] * np.asarray(conditional_means), axis=0)
    return means, log_z


def fbs_sd(inst: Instance, seed: int = 0) -> dict:
    start = time.perf_counter()
    bp = bp_full(inst, seed=seed)
    bp_time_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    residual = sd_star_residuals(inst, bp)
    cutset = sd_feedback_set(inst, residual)
    means, log_z = exact_cutset(inst, cutset)
    correction_time_ms = (time.perf_counter() - start) * 1000
    return {
        "means": means,
        "log_z": log_z,
        "cutset": cutset,
        "max_sd_before": float(np.max(np.abs(residual))),
        "bp": bp,
        "bp_time_ms": bp_time_ms,
        "correction_time_ms": correction_time_ms,
    }


def mean_field(inst: Instance, seed: int = 0, restarts: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    jmat = inst.matrix()
    best_m = None
    best_f = math.inf
    for restart in range(restarts):
        m = np.zeros(inst.n) if restart == 0 else rng.uniform(-0.25, 0.25, size=inst.n)
        for _ in range(5000):
            target = np.tanh(inst.h + jmat @ m)
            new = 0.65 * m + 0.35 * target
            if np.max(np.abs(new - m)) < 1e-11:
                m = new
                break
            m = new
        p_plus = np.clip((1.0 + m) / 2.0, EPS, 1.0 - EPS)
        entropy_term = np.sum(p_plus * np.log(p_plus) + (1 - p_plus) * np.log(1 - p_plus))
        free_energy = -float(inst.h @ m) - 0.5 * float(m @ jmat @ m) + float(entropy_term)
        if free_energy < best_f:
            best_f = free_energy
            best_m = m.copy()
    assert best_m is not None
    return best_m


def hard_fsm(inst: Instance, seed: int = 0, restarts: int = 12) -> np.ndarray:
    """Deterministic greedy route: coordinate-ascent MAP with several starts."""
    rng = np.random.default_rng(seed)
    jmat = inst.matrix()
    best = None
    best_score = -math.inf
    for restart in range(restarts):
        if restart == 0:
            spins = np.where(inst.h >= 0, 1.0, -1.0)
        else:
            spins = rng.choice(np.asarray([-1.0, 1.0]), size=inst.n)
        order = np.arange(inst.n)
        for _ in range(200):
            changed = False
            rng.shuffle(order)
            for i in order:
                field = inst.h[i] + jmat[i] @ spins
                value = 1.0 if field >= 0 else -1.0
                if value != spins[i]:
                    spins[i] = value
                    changed = True
            if not changed:
                break
        score = energy_score(spins, inst)
        if score > best_score:
            best_score = score
            best = spins.copy()
    assert best is not None
    return best


def neighbour_arrays(inst: Instance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(inst.n)]
    for (i, j), value in zip(inst.edges, inst.coupling):
        neighbours[i].append((j, float(value)))
        neighbours[j].append((i, float(value)))
    max_degree = max(len(row) for row in neighbours)
    idx = -np.ones((inst.n, max_degree), dtype=np.int64)
    val = np.zeros((inst.n, max_degree), dtype=np.float64)
    deg = np.zeros(inst.n, dtype=np.int64)
    for i, row in enumerate(neighbours):
        deg[i] = len(row)
        for k, (j, coupling) in enumerate(row):
            idx[i, k] = j
            val[i, k] = coupling
    return idx, val, deg


@njit(cache=True)
def gibbs_terminal_numba(h, neighbour_idx, neighbour_j, degree, burn, samples, thin, chains, seed):
    np.random.seed(seed)
    n = h.shape[0]
    spins = np.empty((chains, n), dtype=np.float64)
    for c in range(chains):
        for i in range(n):
            spins[c, i] = 1.0 if np.random.random() < 0.5 else -1.0
    total = 0.0
    count = 0
    sweeps = burn + samples * thin
    for sweep in range(sweeps):
        for i in range(n):
            for c in range(chains):
                field = h[i]
                for k in range(degree[i]):
                    j = neighbour_idx[i, k]
                    field += neighbour_j[i, k] * spins[c, j]
                if field >= 0:
                    p_plus = 1.0 / (1.0 + math.exp(-2.0 * field))
                else:
                    ef = math.exp(2.0 * field)
                    p_plus = ef / (1.0 + ef)
                spins[c, i] = 1.0 if np.random.random() < p_plus else -1.0
        if sweep >= burn and ((sweep - burn) % thin == 0):
            for c in range(chains):
                total += 1.0 if spins[c, 0] > 0 else 0.0
                count += 1
    return total / count


def gibbs_terminal(inst: Instance, seed: int) -> float:
    idx, val, deg = neighbour_arrays(inst)
    return float(gibbs_terminal_numba(inst.h, idx, val, deg, 250, 450, 1, 6, seed))


def clip_prob(q: float) -> float:
    return float(np.clip(q, 1e-6, 1.0 - 1e-6))


def metrics(p: float, q: float) -> dict[str, float]:
    q = clip_prob(q)
    p = float(np.clip(p, EPS, 1.0 - EPS))
    kl = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    brier_excess = (p - q) ** 2
    bayes_action = p >= 0.5
    action = q >= 0.5
    risk = (1 - p) if action else p
    bayes_risk = min(p, 1 - p)
    excess_risk = risk - bayes_risk
    return {
        "kl_excess": float(max(0.0, kl)),
        "brier_excess": float(brier_excess),
        "terminal_excess_risk": float(max(0.0, excess_risk)),
        "abs_prob_error": float(abs(p - q)),
    }


def expected_calibration_error(records: list[tuple[float, float]], bins: int = 10) -> float:
    if not records:
        return math.nan
    p = np.asarray([r[0] for r in records])
    q = np.asarray([r[1] for r in records])
    boundaries = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (q >= lo) & (q < hi if hi < 1 else q <= hi)
        if np.any(mask):
            ece += mask.mean() * abs(float(q[mask].mean() - p[mask].mean()))
    return float(ece)


def run(args: argparse.Namespace) -> tuple[list[dict], dict]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compile numba before timing.
    dummy = make_instance(8, 1, 0.4, np.random.default_rng(123))
    gibbs_terminal(dummy, 123)

    rows: list[dict] = []
    brute_max_mean = 0.0
    brute_max_logz = 0.0
    cutset_sizes: list[int] = []
    sd_before: list[float] = []

    methods = ["Hard FSM", "Mean field", "Gibbs MC", "Loopy Bethe", "Proposed FBS-SD"]
    total_jobs = len(args.sizes) * len(args.loops) * len(args.strengths) * args.seeds
    job = 0
    for n in args.sizes:
        for loops in args.loops:
            for strength in args.strengths:
                for seed_idx in range(args.seeds):
                    job += 1
                    seed = args.base_seed + 100000 * n + 1000 * loops + 100 * int(round(10 * strength)) + seed_idx
                    rng = np.random.default_rng(seed)
                    inst = make_instance(n, loops, strength, rng)

                    t0 = time.perf_counter()
                    proposed = fbs_sd(inst, seed=seed + 1)
                    proposed_time = (time.perf_counter() - t0) * 1000
                    p_true = float((1.0 + proposed["means"][0]) / 2.0)
                    cutset_sizes.append(len(proposed["cutset"]))
                    sd_before.append(proposed["max_sd_before"])

                    if n <= args.brute_n:
                        exact_means, exact_logz = brute_force_means(inst)
                        brute_max_mean = max(brute_max_mean, float(np.max(np.abs(exact_means - proposed["means"]))))
                        brute_max_logz = max(brute_max_logz, abs(exact_logz - proposed["log_z"]))

                    method_outputs: list[tuple[str, float, float, dict]] = []

                    t0 = time.perf_counter()
                    hard = hard_fsm(inst, seed=seed + 2)
                    hard_time = (time.perf_counter() - t0) * 1000
                    hard_q = 1.0 - 1e-6 if hard[0] > 0 else 1e-6
                    method_outputs.append(("Hard FSM", hard_q, hard_time, {}))

                    t0 = time.perf_counter()
                    mf = mean_field(inst, seed=seed + 3)
                    mf_time = (time.perf_counter() - t0) * 1000
                    method_outputs.append(("Mean field", float((1 + mf[0]) / 2), mf_time, {}))

                    t0 = time.perf_counter()
                    mc_q = gibbs_terminal(inst, seed + 4)
                    mc_time = (time.perf_counter() - t0) * 1000
                    method_outputs.append(("Gibbs MC", mc_q, mc_time, {}))

                    # The Bethe baseline is exactly the first stage of the proposed
                    # method, so reuse it and report only that stage's measured time.
                    bp = proposed["bp"]
                    bp_time = proposed["bp_time_ms"]
                    bp_q = float((1 + bp["means"][0]) / 2)
                    method_outputs.append(("Loopy Bethe", bp_q, bp_time, {"converged": int(bp["converged"])}))

                    method_outputs.append(("Proposed FBS-SD", p_true, proposed_time, {
                        "cutset_size": len(proposed["cutset"]),
                        "max_sd_before": proposed["max_sd_before"],
                    }))

                    for method, q, elapsed, extras in method_outputs:
                        row = {
                            "n": n,
                            "loops": loops,
                            "strength": strength,
                            "seed": seed_idx,
                            "method": method,
                            "p_true": p_true,
                            "q_pred": q,
                            "runtime_ms": elapsed,
                        }
                        row.update(metrics(p_true, q))
                        row.update(extras)
                        rows.append(row)

                    if job % max(1, total_jobs // 10) == 0:
                        print(f"Completed {job}/{total_jobs} instances")

    csv_path = out_dir / "benchmark_results.csv"
    fields = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict] = {}
    for method in methods:
        mr = [r for r in rows if r["method"] == method]
        calibration_pairs = [(float(r["p_true"]), float(r["q_pred"])) for r in mr]
        summary[method] = {
            "instances": len(mr),
            "kl_excess_mean": float(np.mean([r["kl_excess"] for r in mr])),
            "brier_excess_mean": float(np.mean([r["brier_excess"] for r in mr])),
            "terminal_excess_risk_mean": float(np.mean([r["terminal_excess_risk"] for r in mr])),
            "abs_prob_error_mean": float(np.mean([r["abs_prob_error"] for r in mr])),
            "abs_prob_error_p95": float(np.quantile([r["abs_prob_error"] for r in mr], 0.95)),
            "ece": expected_calibration_error(calibration_pairs),
            "runtime_ms_median": float(np.median([r["runtime_ms"] for r in mr])),
            "runtime_ms_mean": float(np.mean([r["runtime_ms"] for r in mr])),
        }

    by_loops: dict[str, dict] = {}
    for loops in args.loops:
        by_loops[str(loops)] = {}
        for method in methods:
            mr = [r for r in rows if r["method"] == method and r["loops"] == loops]
            by_loops[str(loops)][method] = {
                "kl_excess_mean": float(np.mean([r["kl_excess"] for r in mr])),
                "abs_prob_error_mean": float(np.mean([r["abs_prob_error"] for r in mr])),
                "terminal_excess_risk_mean": float(np.mean([r["terminal_excess_risk"] for r in mr])),
            }

    by_strength: dict[str, dict] = {}
    for strength in args.strengths:
        by_strength[str(strength)] = {}
        for method in methods:
            mr = [r for r in rows if r["method"] == method and r["strength"] == strength]
            by_strength[str(strength)][method] = {
                "kl_excess_mean": float(np.mean([r["kl_excess"] for r in mr])),
                "abs_prob_error_mean": float(np.mean([r["abs_prob_error"] for r in mr])),
                "terminal_excess_risk_mean": float(np.mean([r["terminal_excess_risk"] for r in mr])),
            }

    by_size: dict[str, dict] = {}
    for n in args.sizes:
        by_size[str(n)] = {}
        for method in methods:
            mr = [r for r in rows if r["method"] == method and r["n"] == n]
            by_size[str(n)][method] = {
                "runtime_ms_median": float(np.median([r["runtime_ms"] for r in mr])),
                "abs_prob_error_mean": float(np.mean([r["abs_prob_error"] for r in mr])),
            }

    metadata = {
        "config": vars(args),
        "summary": summary,
        "by_loops": by_loops,
        "by_strength": by_strength,
        "by_size": by_size,
        "validation": {
            "brute_force_max_mean_error": brute_max_mean,
            "brute_force_max_logz_error": brute_max_logz,
            "mean_cutset_size": float(np.mean(cutset_sizes)),
            "max_cutset_size": int(np.max(cutset_sizes)),
            "mean_initial_sd_residual": float(np.mean(sd_before)),
            "max_initial_sd_residual": float(np.max(sd_before)),
        },
    }
    with (out_dir / "benchmark_summary.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))
    return rows, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="benchmark_output")
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--loops", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.4, 0.8, 1.2])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260729)
    parser.add_argument("--brute-n", type=int, default=0, help="Brute-force validation for n <= this value")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
