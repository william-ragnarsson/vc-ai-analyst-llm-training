.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup data lint format typecheck test test-core test-parity gradcheck leakage check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install pinned dependencies
	@test -d .venv || python3 -m venv .venv
	$(PIP) install -q -r requirements.lock
	$(PIP) install -q -e ".[dev,paper]"

data:  ## Rebuild data/processed from the raw export and print the drop ledger
	$(PY) -m vcml.core.clean

lint:  ## Check style
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

format:  ## Apply formatting
	.venv/bin/ruff format src tests
	.venv/bin/ruff check --fix src tests

typecheck:  ## Strict type checking on the frozen core
	.venv/bin/mypy

test:  ## Run the full suite (excluding slow tests)
	$(PY) -m pytest -m "not slow" -q

test-core:  ## The gate every PR must pass
	$(PY) -m pytest tests/core tests/models tests/leakage -q

test-parity:  ## Verify hand-written implementations against scikit-learn
	$(PY) -m pytest -m parity -q

gradcheck:  ## Finite-difference gradient checks
	$(PY) -m pytest -m gradcheck -q

leakage:  ## Leakage guards, including the permutation canary
	$(PY) -m pytest tests/leakage -q

check: lint typecheck test-core  ## What CI runs

clean:  ## Remove derived artefacts
	rm -rf data/processed paper/generated .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} +
