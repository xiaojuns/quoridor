SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3.10
ENV_PYTHON := training/.venv/bin/python
UV := .tools/uv/bin/uv
export PYTHONPATH := training
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export PYTHONDONTWRITEBYTECODE := 1
export UV_CACHE_DIR := $(CURDIR)/.tools/uv-cache
export UV_PYTHON_DOWNLOADS := never

.PHONY: help install check smoke arena experiment environment

help:
	@printf '%s\n' \
	  'make install     Install locked Linux x86_64 Python 3.10 CPU environment' \
	  'make check       Validate environment, lock, YAML configs, and M0/reference tests' \
	  'make smoke       Reserved smoke interface; M0 reports NOT_IMPLEMENTED' \
	  'make arena       Reserved frozen-arena interface; M0 reports NOT_IMPLEMENTED' \
	  'make experiment  Reserved budgeted-training interface; M0 reports NOT_IMPLEMENTED' \
	  '                 Set CONFIG=path/to/config.yaml to select a config'

install:
	@PYTHON="$(PYTHON)" bash scripts/bootstrap-env.sh

environment:
	@test -x "$(ENV_PYTHON)" || { printf '%s\n' 'Missing locked environment. Run make install first.' >&2; exit 1; }
	@test -x "$(UV)" || { printf '%s\n' 'Missing pinned uv. Run make install first.' >&2; exit 1; }
	@test "$$($(UV) --version)" = 'uv 0.9.5' || { printf '%s\n' 'Wrong uv version. Run make install first.' >&2; exit 1; }
	@$(ENV_PYTHON) scripts/check-environment.py

check: environment
	@$(UV) lock --project training --check --offline
	@$(UV) sync --project training --locked --offline --group dev --check
	@$(ENV_PYTHON) -m quoridor.project_cli check-config --config configs/smoke.yaml
	@$(ENV_PYTHON) -m quoridor.project_cli check-config --config configs/arena.yaml
	@$(ENV_PYTHON) -m quoridor.project_cli check-config --config configs/experiment.yaml
	@$(ENV_PYTHON) -m pytest -q tests/m0 training/tests/test_codec_game.py training/tests/test_mcts.py training/tests/test_network.py training/tests/test_numerics.py

smoke: environment
	@$(ENV_PYTHON) -m quoridor.project_cli smoke --config "$(or $(CONFIG),configs/smoke.yaml)"

arena: environment
	@$(ENV_PYTHON) -m quoridor.project_cli arena --config "$(or $(CONFIG),configs/arena.yaml)"

experiment: environment
	@$(ENV_PYTHON) -m quoridor.project_cli experiment --config "$(or $(CONFIG),configs/experiment.yaml)"
