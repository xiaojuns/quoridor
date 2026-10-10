# M4 Verification Record

Date: 2026-10-10. M4 started from the verified M3 worktree at
`1d45df03b0952ad0e1da49eda6832ae7c9d4bc5b`; the M3 changes were present and
were not reverted. No long-run strength experiment was run.

## Actual checks

The locked Linux CPU environment reported Python 3.10.12, torch 2.7.1+cpu,
NumPy 2.2.6, PyYAML 6.0.3, pytest 8.4.2 and Hypothesis 6.142.4.

| Command | Actual result |
| --- | --- |
| Baseline `make check` before M4 edits | Exit 0; 586 passed in 85.73s; no skips |
| `make check-m4` | Exit 0; **17 passed in 6.50s**; no skips |
| Final `make check` | Exit 0; **603 passed in 90.74s**; no skips |
| `git diff --check` | Exit 0 |
| `make smoke CONFIG=configs/smoke.yaml` | Exit 0; one complete game, one replay position, one optimizer step, checkpoint reload and model artifact save |

The cumulative command also passed the locked offline `uv` lock/sync checks,
all three YAML validations and the frozen arena/opening digest check. The
focused command is reproducible with `make check-m4`; the complete gate with
`make check`.

## Smoke evidence

The final smoke run is retained at
`runs/smoke/20261010T130800Z-dc36bd89ff77403a8c7a6e4526004ec1` (ignored local
runtime output). Its `run.json`, `events.jsonl` and `summary.json` show:

- status `completed`, elapsed 1.166 seconds, one game and one ply;
- the tactical fixture `tactical-immediate-goal-v1` reached a goal and was
  admitted atomically to replay;
- model architecture 64 channels / 4 residual blocks, 8 simulations, root
  noise enabled, temperature 1.0, replay capacity 16, batch size 1 and one
  optimizer step;
- training checkpoint SHA-256
  `51783afe5dfa29093cbb795abb7eebcb3413ac52e8d46375190969b227512dde`;
- final inference-weights SHA-256
  `57368edc411e5fc2e50dc00177c70a1b91b21cc4fcf946aa7b15e78bef3ce278`;
- loss `1.000000238418579`, zero truncated games and zero failed games.

The smoke is an engineering path check. Its near-goal fixture and zero-weight,
winning-policy initialization do not measure opening quality, convergence or
playing strength. The long-run starting settings remain documented separately:
128 simulations, 100,000 replay positions, batch size 128 and 50–100 games per
round.

## Acceptance evidence

The M4 tests exercise complete goal and repetition-draw labels from each
recorded current-player perspective, exclusion and counters for truncated or
failed games, exact history-prefix admission, canonical policy/action transforms,
oldest-first replay eviction, masked finite policy/value updates, optimizer
state, scheduler/scaler and RNG restoration, replay/hash/lineage compatibility,
and deterministic post-resume sampling/update behavior. A mismatch fails before
the destination trainer or replay is changed.

Every self-play move uses a shared deadline and requires a completed search with
a legal action. Partial search visits and root estimates are never used as
moves or labels. The actor advances seeds per move, records original-coordinate
root targets, and marks time/ply/interrupt/failure stops explicitly. The replay
encoder transforms those targets into canonical network coordinates before
checking legal support.

M2 inference weights and M4 resume checkpoints are separate artifact kinds.
Resume metadata binds the effective config and lock hashes, rules/codec/value,
encoder/model/search versions, architecture, code identity, RNG streams, replay
hash and counters. Legacy assets are rejected; no migration or historical data
conversion was performed.

## Scope and next task

`make smoke` now invokes the real bounded `smoke-run` path. The direct historical
`project_cli smoke` placeholder remains for the retained M0 contract test; it is
not the Makefile workload. `make arena` and `make experiment` still return
explicit nonzero NOT_IMPLEMENTED errors for M5 and M6. No GPU/native execution,
long-run training, arena game or strength claim was made. M5 is the next eligible
task.
