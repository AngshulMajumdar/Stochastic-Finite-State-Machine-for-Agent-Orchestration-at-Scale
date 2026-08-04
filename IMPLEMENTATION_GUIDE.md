# Implementation Guide

## 1. Install and import

```bash
python -m pip install -e .
```

```python
from sfsm_orchestration.core import (
    BenchmarkConfig,
    generate_agent_graph,
    fsm_select,
    sfsm_select,
    accuracy,
)
```

## 2. Minimal serial example

```python
import numpy as np
from sfsm_orchestration.core import BenchmarkConfig, generate_agent_graph, fsm_select, sfsm_select

config = BenchmarkConfig(components=10_000, agents_per_component=8)
rng = np.random.default_rng(config.master_seed)
correct, scores, priors = generate_agent_graph(0.4, rng, config)

fsm_indices = fsm_select(scores, threshold=0.665)
sfsm_indices = sfsm_select(scores, priors)
```

Both arrays contain one actual output index per component.

## 3. Distributed example

```python
from sfsm_orchestration.distributed import mapreduce_checksum

checksum, count, elapsed_ms = mapreduce_checksum(
    scores=scores,
    priors=priors,
    workers=4,
    repeats=7,
)
```

The distributed implementation opens shared read-only memory maps rather than copying the complete million-agent arrays into each process.

## 4. Replace the benchmark verifier

The reference posterior assumes different beta score distributions for correct and incorrect outputs. A production system should replace this with a calibrated model trained on execution logs.

The production interface should expose either:

- posterior correctness for each actual output; or
- a likelihood ratio that can be combined with the reliability prior.

Calibration must be measured on held-out data and segmented by task family.

## 5. Add explicit actions

Production orchestration usually chooses among more than candidate outputs. Represent actions such as:

```text
accept output i
verify output i
retry agent j
escalate to agent k
abstain
request human review
```

Assign each action a cost and terminal loss. Select the action with minimum posterior expected loss rather than maximum raw score.

## 6. Compile a sparse workflow

For each active variable, store a compact state ID. For each factor, store:

```text
factor type
incident variable IDs
parameter block offset
evidence version
```

Group factors by type so that workers can evaluate them in vectorized kernels. Do not store full prompts or transcripts in the factor graph; store object references and compact features.

## 7. Active-frontier update

After a tool returns:

1. append the result to the evidence log;
2. update state and eligibility;
3. activate new variables and factors;
4. invalidate affected messages;
5. run residual-scheduled updates;
6. correct the cyclic core if the residual budget is exceeded;
7. return the next action.

## 8. MapReduce on a cluster

The workstation implementation maps row partitions. A cluster implementation should use immutable, versioned task records.

Mapper input:

```text
run_id
partition_id
graph_version
evidence_version
parameter_version
random_seed
```

Mapper output:

```text
partition_id
selected indices or conditional marginals
log partition contribution
residual summary
processed count
checksum
```

The reducer rejects mixed versions, duplicate partition IDs with conflicting checksums, and incomplete counts.

## 9. Observability

Log at least:

- model and calibration version;
- evidence available at decision time;
- selected action and fallback;
- posterior success or expected loss;
- latency and cost;
- residual and approximation status;
- eventual outcome when known;
- human override.

These records support calibration, drift detection and incident review.

## 10. Production rollout

Start in shadow mode. Compare SFSM recommendations with the current FSM without changing execution. Enable actions gradually: read-only selection, verification and retry, then higher-impact routing. Keep hard policy constraints external to learned probabilities.
