PYTHON ?= python

.PHONY: install test quick reproduce verify figures clean audit

install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e . --no-build-isolation --no-deps

test:
	$(PYTHON) -m pytest -q

quick:
	$(PYTHON) experiments/run_all.py --profile quick

reproduce:
	$(PYTHON) experiments/run_all.py --profile full

verify:
	$(PYTHON) -m sfsm_orchestration.verify \
		--reference results/reference/million_agent \
		--generated results/generated/million_agent

figures:
	$(PYTHON) experiments/make_figures.py \
		--results results/reference/million_agent \
		--out results/generated/figures

audit:
	$(PYTHON) scripts/audit_release.py

clean:
	rm -rf results/generated .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
