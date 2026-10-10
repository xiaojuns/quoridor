# Project Entry

Read `progress.md`, `tasks.yaml`, then `docs/spec.md`, `docs/design.md`, and
`docs/eval.md`. M0-M2 are complete. Read `docs/model-v1.md` and
`docs/verification-m2.md` for the new encoder/model APIs and actual evidence.
Stop after M2; M3-M6 require a later task.
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
- Every training job needs a finite wall-clock budget and a unique traceable
  `runs/` directory. Preserve legacy `training/runs/` and local checkpoints.
- Arena settings and opening assets are frozen by hashes. Change the protocol
  through review and a new version, never through runtime tuning overrides.

# Verification

Run `make check` before reporting a milestone complete. Run relevant focused
checks while editing. `make smoke`, `make arena`, and `make experiment` currently
return NOT IMPLEMENTED with a nonzero status until M4/M5/M6, respectively. Never
report stub execution, skipped native/GPU tests, historical reports, or loss
improvement as a passed pipeline or strength evaluation. Record actual commands,
outcomes, limitations, and the next task in `progress.md`.
