# Reproducibility Protocol

## 1. What is frozen

`results/reference/` contains the exact CSV and JSON files used by the release. Treat these files as immutable evidence. New runs are written under `results/generated/`, which is created only when an experiment is executed.

## 2. Environment

Recommended environment:

- Python 3.11 or newer;
- 8-16 GB RAM for the million-node benchmark;
- a local filesystem that supports memory mapping;
- multiple CPU cores for the MapReduce experiment.

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-build-isolation --no-deps
```

Alternative Conda setup:

```bash
conda env create -f environment.yml
conda activate sfsm-orchestration
python -m pip install -e .
```

## 3. Profiles

### Quick profile

```bash
python experiments/run_all.py --profile quick
```

This checks installation, multiprocessing, output generation, reduced exactness validation and figure generation. It follows the full code path with fewer components, seeds and repetitions.

### Full profile

```bash
python experiments/run_all.py --profile full
```

This runs:

1. one-million-node FSM/SFSM benchmark;
2. worker scaling;
3. 270-instance loopy exactness benchmark;
4. direct-enumeration validation on small instances;
5. independent trifecta consistency checks;
6. figure generation.

## 4. Individual experiments

Million-node benchmark:

```bash
python experiments/run_million_agent.py \
  --config experiments/configs/full.json \
  --output-dir results/generated/million_agent
```

Worker scaling:

```bash
python experiments/run_worker_scaling.py \
  --config experiments/configs/full.json \
  --output results/generated/million_agent/worker_scaling.csv
```

Exactness benchmark:

```bash
python experiments/benchmark_fbs.py \
  --out-dir results/generated/exactness/main \
  --sizes 32 64 128 \
  --loops 2 4 6 \
  --strengths .4 .8 1.2 \
  --seeds 10
```

Brute-force validation:

```bash
python experiments/benchmark_fbs.py \
  --out-dir results/generated/exactness/bruteforce \
  --sizes 16 \
  --loops 2 4 6 \
  --strengths .4 .8 1.2 \
  --seeds 10 \
  --brute-n 16
```

Independent consistency validation:

```bash
python experiments/validate_trifecta.py
```

## 5. Verification rules

```bash
python -m sfsm_orchestration.verify \
  --reference results/reference/million_agent \
  --generated results/generated/million_agent
```

The verifier applies different standards to different outputs.

- Output-admissibility and serial/distributed equality are exact checks.
- Accuracy summaries are checked within Monte Carlo tolerance.
- Timing tables are checked for schema, finiteness and positivity because timings depend on hardware and scheduling.

## 6. Randomness

All benchmark seeds are explicit. The master seed is stored in each JSON configuration and copied into `metadata.json`. FSM thresholds are tuned on a disjoint validation seed range before test seeds are generated.

## 7. Terminal-output invariant

Every selected result is an integer index into a row of actually generated outputs. Correctness is hidden from the policies and used only by the evaluator. Tests reject an index outside the generated row.

## 8. Expected output files

```text
results/generated/million_agent/million_agent_raw.csv
results/generated/million_agent/million_agent_accuracy.csv
results/generated/million_agent/million_agent_time.csv
results/generated/million_agent/worker_scaling.csv
results/generated/million_agent/metadata.json
results/generated/exactness/main/benchmark_results.csv
results/generated/exactness/main/benchmark_summary.json
results/generated/exactness/bruteforce/bruteforce_validation_results.csv
results/generated/exactness/bruteforce/bruteforce_validation_summary.json
results/generated/figures/*.png
```

## 9. Timing scope

Reported benchmark time covers orchestration selection only. Simulated agent generation, external model latency, network transport and production persistence are deliberately excluded. Worker-scaling timing excludes process-pool startup and memory-map creation after warm-up.

## 10. Integrity

`SHA256SUMS` records the checksum of every released file except itself. Verify with:

```bash
sha256sum -c SHA256SUMS
```
