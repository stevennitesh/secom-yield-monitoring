PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff

.PHONY: help install lint format format-check test coverage check

help:
	@printf '%s\n' \
		'Targets:' \
		'  install       Install project and dev tooling from requirements.txt' \
		'  lint          Run Ruff lint checks' \
		'  format        Format active Python files with Ruff' \
		'  format-check  Check Ruff formatting without modifying files' \
		'  test          Run pytest' \
		'  coverage      Run pytest with coverage report' \
		'  check         Run lint, format-check, and test'

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-build-isolation

lint:
	$(RUFF) check .

format:
	$(RUFF) format src tests scripts

format-check:
	$(RUFF) format --check src tests scripts

test:
	$(PYTEST) -q

coverage:
	$(PYTEST) --cov=secom --cov-report=term-missing

check: lint format-check test
