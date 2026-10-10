# Project Entry

Read `progress.md`, `tasks.yaml`, then `docs/spec.md`, `docs/design.md`, and
`docs/eval.md`. The current milestone is M0. Stop after its acceptance checks;
M1-M6 require a later task. Existing Phase 4 code and ignored local assets are
legacy evidence, not proof of compliance with the new contract.

# Workflow

- Use Python/PyTorch, YAML configs, JSONL runtime events, pytest/Hypothesis, Git.
- Install the isolated CPU environment with `make install`; dependencies are
  fixed by `training/uv.lock`. Never update the lock implicitly during checks.
- Keep rules independent of networks. Networks accept tensors; encoders call
  rules. Treat values as the current player's perspective and flip on backup.
- Every training job needs a finite wall-clock budget and a unique traceable
  `runs/` directory. Preserve legacy `training/runs/` and local checkpoints.
- Arena settings and opening assets are frozen by hashes. Change the protocol
  through review and a new version, never through runtime tuning overrides.

# Verification

Run `make check` before reporting a milestone complete. Run relevant focused
checks while editing. `make smoke`, `make arena`, and `make experiment` currently
return NOT IMPLEMENTED with a nonzero status; this is expected in M0. Never
report stub execution, skipped native/GPU tests, historical reports, or loss
improvement as a passed pipeline or strength evaluation. Record actual commands,
outcomes, limitations, and the next task in `progress.md`.
