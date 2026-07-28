# Common tasks. `make help` lists them.
# Works on Unix; on Windows use the equivalent commands under `bwt --help`.

PYTHON ?= python
VENV   ?= .venv
BIN    := $(VENV)/bin
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
endif

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev install-deep test test-fast lint format \
        typecheck coverage prepare train benchmark calibrate explain serve \
        docker clean clean-all

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtual environment
	$(PYTHON) -m venv $(VENV)

install: venv  ## Install runtime dependencies
	$(BIN)/pip install -r requirements.txt && $(BIN)/pip install -e .

install-dev: venv  ## Install runtime + test dependencies
	$(BIN)/pip install -r requirements-dev.txt && $(BIN)/pip install -e ".[dev]"

install-deep:  ## Install PyTorch with CUDA 12.8 (neural pipelines)
	$(BIN)/pip install torch --index-url https://download.pytorch.org/whl/cu128

test:  ## Run the full test suite
	$(BIN)/pytest

test-fast:  ## Run only tests that need no dataset
	$(BIN)/pytest -m "not slow"

coverage:  ## Run tests with a coverage report
	$(BIN)/pytest --cov=bwt --cov-report=term-missing --cov-report=html

lint:  ## Check style and common errors
	$(BIN)/ruff check src tests

format:  ## Auto-fix what ruff can fix
	$(BIN)/ruff check --fix src tests

prepare:  ## Epoch the default task and populate the cache
	$(BIN)/bwt prepare

train:  ## Train the default model
	$(BIN)/bwt train

benchmark:  ## Run the resumable benchmark for ten minutes
	$(BIN)/bwt benchmark --time-budget 600

calibrate:  ## Measure the calibration curve
	$(BIN)/bwt calibrate --pipeline eegnet

explain:  ## Generate ERD curves and CSP topographies
	$(BIN)/bwt explain

serve:  ## Run the web service
	$(BIN)/bwt serve

docker:  ## Build the serving image
	docker build -t bwt:latest .

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-all: clean  ## Also remove epoched data, models and reports
	rm -rf cache artifacts reports uploads
