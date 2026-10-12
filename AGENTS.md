# Project Entry

Read `progress.md`, `tasks.yaml`, then `docs/spec.md`, `docs/design.md`, and
`docs/eval.md`. M0-M4 are complete. Read `docs/model-v1.md`, `docs/search-v1.md`,
`docs/training-v1.md`, `docs/arena-v1.md` and `docs/verification-m5.md` for APIs
and actual evidence. M5 is in progress; stop before M6. A passing engineering
suite is not completion of the required real 200-pair arena run.
Existing Phase 4 code and ignored local assets are
legacy evidence, not proof of compliance with the new contract.

# Workflow

- Use Python/PyTorch, YAML configs, JSONL runtime events, pytest/Hypothesis, Git.
- Install the isolated CPU environment with `make install`; dependencies are
  fixed by `training/uv.lock`. Never update the lock implicitly during checks.
- Keep rules independent of networks. Networks accept tensors; encoders call
  rules. Treat values as the current player's perspective and flip on backup.
- New rules use `quoridor.engine` / `quoridor.episode`. Root package exports,
  `game` / `codec`, browser and native are legacy; no automatic backend fallback.
- New tensor models use `policy_value`, with external `encoder` and `policy`.
  The input is explicitly partially observed; exact search outcomes/cache keys
  must retain full episode history. Legacy 16-plane weights are incompatible.
- New search uses `puct` with a shared `Deadline` and `search_evaluator` for
  inference. Check its status before using an action; partial visits are
  diagnostics. Search-local cycles and truncated simulations are not game labels.
- Every training job needs a finite wall-clock budget and a unique traceable
  `runs/` directory. Preserve legacy `training/runs/` and local checkpoints.
- M4 smoke is a real bounded selfplay/update/checkpoint/reload run. Its small
  workload may differ from long-run starting parameters, but every effective
  setting and elapsed budget must be recorded.
- Arena settings and opening assets are frozen by hashes. Change the protocol
  through review and a new version, never through runtime tuning overrides.

# Verification

Run `make check` before reporting a milestone complete. Run relevant focused
checks while editing. `make check-m5` covers frozen arena integrity and scoring.
`make arena` requires explicit CANDIDATE/CHAMPION model-only checkpoints; use
CANDIDATE_SHA256/CHAMPION_SHA256 to pin external identities. `make experiment`
still returns NOT IMPLEMENTED with nonzero status until M6. Never
report stub execution, skipped native/GPU tests, historical reports, or loss
improvement as a passed pipeline or strength evaluation. Record actual commands,
outcomes, limitations, and the next task in `progress.md`.

Fresh final openings are frozen by `configs/reserved-final-v1.yaml` and its
digest. Inspect their integrity, never their performance for tuning. Historical
final openings were exposed. The fresh suite needs its own frozen execution
protocol before a later independent final evaluation; do not override dev YAML.
