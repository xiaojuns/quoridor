# Project Progress

Date: 2026-10-10. Workstream: `quoridor-alphazero-v1`.
Current milestone: M0 completed and verified. This round stops after M0.
Next implementation milestone: M1; do not begin it in this round.

## Baseline Audit

The starting worktree was clean at commit `8a3582f` (2026-09-22,
"Validate F21 value scaling against classic-hard"). The repository contains a
browser/classic AI, Python reference rules, a 209-action codec, PyTorch SE-ResNet
policy/value model, Python PUCT, C++ batched MCTS, self-play, replay, trainer,
checkpoint, and arena implementations. This is an engineering rebuild over
substantial existing work, not a new empty player.

Source inspection confirms the action offsets 0/81/145 and current-player value
convention. The network defaults to 192 channels and 16 residual blocks, with
legacy 14/16 input planes. Engine/codec imports are independent of neural code;
the network module still mixes rule-derived encoding with inference.

The existing Python/C++ rules make the acting player lose at third repetition
or ply 256. Self-play/arena assume a winner and +/-1 labels. They do not implement
the new goal/repetition-draw/runtime-truncation contract. M1-M4 must migrate this
behavior and isolate old replay/checkpoints. Existing features do not expose
complete repetition history to the network; M2 has an explicit decision gate.

## Local Assets and Historical Reports

Read-only inventory found `training/runs/` contains 12,648,686,985 bytes
(approximately 11.78 GiB), 14,766 files: 90 `.pt` models/checkpoints, 330 `.npy`
tensor files, 10,880 `.qrp` replay files, 3,333 JSON files, and one JSONL file.
These assets are ignored by Git and absent from a fresh clone.

The saved champion is generation 14, 48,245,903 bytes; `latest` exists while
`best` and `product` are null. The F21 baseline is a 192-channel/16-block/16-plane
model at generation 1, step 192 (48,266,271 bytes).

This round recomputed raw SHA-256 for two existing model files and matched their
saved records:

- Champion: `67b580c9087f9f4c73f4a3ca3887b05bfdadd57baa20df2406bea542bed45696`.
- F21: `1ab9a6350178436c6cbf39fe76bb65ff31dace15bbd0a2ef670624d79a984b01`.

Saved manifests report the following data; contents were not deeply revalidated:

| Dataset | Training | Holdout / development |
| --- | --- | --- |
| Classic teacher | 45,059 positions / 1,352 games | 5,033 positions / 150 games |
| Recovery challenge v2 | 27,025 positions / 538 games | 2,356 positions / 48 games |
| F54 policy teacher | 1,024 positions | 512 positions |
| Bootstrap self-play replay | 960 games / 38,621 plies | None |
| F21 block-01 replay | 256 games / 11,881 plies | None |

| Saved evaluation | Wins / losses | Saved conclusion |
| --- | --- | --- |
| Old champion vs classic-hard | 71 / 329 | Product gate failed |
| F61, F21 value scale 0.5 vs 1.0 | 45 / 19 | Development screen passed |
| F62, same comparison confirmation | 231 / 169 | Confirmation passed |
| F64, scaled F21 vs classic-hard | 175 / 225 | Confirmation failed |

F64 used 200 paired openings / 400 games, candidate native MCTS 800 simulations
with value scale 0.5, and classic `nodes=587`, `maxDepth=5`. Its saved win rate is
43.75%, 90% paired-bootstrap interval 39.75%-48.00%, Elo approximately -43.66.
The saved integrity gate passed but confirmation and runtime promotion did not.
The old GPU ledger records 90.97628775409638 of 120 GPU-hours across 105
invocations, last updated 2026-09-21. These are historical records, not new
experiments or an approval to continue that lineage.

Local provenance anchors (ignored assets, not portable deliverables):

- [Champion manifest](training/runs/phase4/champions/manifest.json).
- [F21 checkpoint metadata](training/runs/phase4/phase4f/recovery-v3-stage-a-block-01-v1/checkpoints/checkpoint-g00000001-s000000000192.json).
- [Classic teacher manifest](training/runs/phase4/teacher-data/teacher-manifest.json).
- [F54 dataset manifest](training/runs/phase4/phase4f/f54-learner-teacher-policy-dataset-v1/dataset-manifest.json).
- [F64 confirmation gate](training/runs/phase4/phase4f/f64-scaled-f21-classic-confirmation-v1/confirmation-gate.json)
  and [arena result](training/runs/phase4/phase4f/f64-scaled-f21-classic-confirmation-v1/confirmation-arena/result.json).
- [GPU ledger](training/runs/phase4/gpu-budget-ledger.json).

README/legacy planning prose lags the latest F64 artifacts. Historical strength
results use old rules and cannot be compared directly to the new protocol.
No local run, dataset, replay, checkpoint, champion, or GPU ledger was modified.

## M0 Implementation

Added the new project entry and spec/design/eval contracts, M0-M6 dependency and
acceptance roadmap, locked CPU environment, strict YAML configs, raw arena and
opening digests, project CLI, and run/deadline/JSONL infrastructure. The new
commands are distinct from legacy `quoridor-az`/phase drivers.

Configuration checks enforce 9x9, 10 walls, 209 actions, current-player values,
finite budgets, isolated `runs/` output, and frozen arena settings. Runtime unit
tests cover source/config/lock provenance, unique directories, JSONL, budget
expiration, interruptions, and explicit failure statuses. These helpers are not
yet connected to actors, optimizers, or real evaluation.

## Executed Verification

The following commands were actually run on Linux x86_64, Python 3.10.12:

| Command / check | Observed result |
| --- | --- |
| `make install` | Exit 0; 22 locked packages installed in `training/.venv` using pinned uv 0.9.5 |
| Environment validation | torch 2.7.1+cpu, numpy 2.2.6, PyYAML 6.0.3, pytest 8.4.2, Hypothesis 6.142.4 matched exact pins |
| `make check` | Exit 0; **82 passed in 11.01s**, including real Hypothesis properties and selected legacy CPU regressions |
| Offline lock/environment checks within `make check` | Lock consistent; all 22 installed packages audited; `Would make no changes` |
| Three YAML checks within `make check` | Validated configs; arena raw YAML and opening hashes matched |
| `make smoke` | Explicit JSONL `not_implemented`, owning milestone M4; CLI exit 3, make exit 2 |
| `make arena` | Explicit JSONL `not_implemented`, owning milestone M5; CLI exit 3, make exit 2 |
| `make experiment` | Explicit JSONL `not_implemented`, owning milestone M6; CLI exit 3, make exit 2 |
| `make help`, shell syntax, `git diff --check` | Exit 0 |
| Roadmap structure | YAML parsed; M0-M6 IDs, dependencies, and acceptance conditions inspected |
| Documentation consistency review | No blocking mismatch; prior exposure of legacy final openings explicitly documented |

M0 tests exercised configuration errors, frozen config/opening rejection,
unimplemented entry failures, codec properties, exact deadline boundaries,
run provenance/JSONL/statuses in temporary directories, and engine import with
ML modules blocked. These are engineering checks, not completed game pipelines.
No new project `runs/` directory was created by any reserved entry.

No self-play/training job, native conformance workload, CUDA computation,
new arena, replay migration, or strength experiment has run in this round.
The existing CPU reference tests selected for M0 protect legacy behavior;
passing them does not certify the new outcome contract.
Synthetic forward/backward passes and optimizer updates inside reference unit
tests are engineering validation, not a training run or strength experiment.

## Next Task

M0 acceptance is complete; work stops here. The next planned implementation task,
requiring a later request, is M1: board-only engine and episode-history boundary, proper outcome
types, goal/threefold/truncation tests, hardened rule invariants, and explicit
legacy compatibility. M2-M6 remain pending. The reserved `make smoke`,
`make arena`, and `make experiment` currently fail explicitly; no successful
end-to-end training or arena interface is claimed.
