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

.PHONY: help install check check-m1 check-m2 check-m3 check-m4 check-m5 smoke arena experiment environment

help:
	@printf '%s\n' \
	  'make install     Install locked Linux x86_64 Python 3.10 CPU environment' \
	  'make check       Validate environment, lock, YAML configs, and M0-M5/reference tests' \
	  'make check-m1    Run new rules, episode, and compatibility tests' \
	  'make check-m2    Run encoder, tensor network, and model checkpoint tests' \
	  'make check-m3    Run PUCT, exact outcomes, neural inference, and deadline tests' \
	  'make check-m4    Run self-play, replay, trainer, and resume tests' \
	  'make check-m5    Run frozen opening, paired arena, and statistical gate tests' \
	  'make smoke       Run the bounded M4 self-play/update/checkpoint smoke' \
	  'make arena       Run frozen M5 pairs; requires CANDIDATE=... CHAMPION=...' \
	  'make experiment  NOT_IMPLEMENTED; budgeted training is scheduled for M6' \
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
	@$(ENV_PYTHON) -m pytest -q tests/m0 tests/m1 tests/m2 tests/m3 tests/m4 tests/m5 training/tests/test_codec_game.py training/tests/test_mcts.py training/tests/test_network.py training/tests/test_numerics.py

check-m1: environment
	@$(ENV_PYTHON) -m pytest -q tests/m1

check-m2: environment
	@$(ENV_PYTHON) -m pytest -q tests/m2

check-m3: environment
	@$(ENV_PYTHON) -m pytest -q tests/m3

check-m4: environment
	@$(ENV_PYTHON) -m pytest -q tests/m4

check-m5: environment
	@$(ENV_PYTHON) -m pytest -q tests/m5

smoke: environment
	@$(ENV_PYTHON) -m quoridor.project_cli smoke-run --config "$(or $(CONFIG),configs/smoke.yaml)"

arena: environment
	@$(ENV_PYTHON) -m quoridor.project_cli arena --config "$(or $(CONFIG),configs/arena.yaml)" --candidate "$(CANDIDATE)" --champion "$(CHAMPION)" $(if $(CANDIDATE_SHA256),--candidate-sha256 "$(CANDIDATE_SHA256)") $(if $(CHAMPION_SHA256),--champion-sha256 "$(CHAMPION_SHA256)")

experiment: environment
	@$(ENV_PYTHON) -m quoridor.project_cli experiment --config "$(or $(CONFIG),configs/experiment.yaml)"
