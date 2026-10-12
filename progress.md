# Project Progress

Date: 2026-10-10. Workstream: `quoridor-alphazero-v1`.
Current milestone: M5 implementation and engineering verification; formal arena pending.
M0-M4 are completed. This follow-up selects M5 and stops before M6.

## M0 Baseline Audit

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

## M0 Local Assets and Historical Reports

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

## M0 Executed Verification

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

## M1 Work and Baseline

Selected M1 as the highest-priority unfinished milestone: its sole dependency M0
is complete; M2-M6 still depend on unfinished work. This follow-up request
supersedes the previous round's M0 stopping point. The starting revision is
`5494513f47c810f499e9691e71fdd64ee7507681`, with a clean worktree.

Before implementation, `make check` was rerun: exit 0, **82 passed in 10.03s**.
All 22 locked packages matched; offline sync reported `Would make no changes`;
the three YAML configs and frozen arena/opening hashes passed validation.

Implemented `quoridor.engine`: immutable validated board-only state, 209-action
codec, all legal actions and masks, wall/path/jump rules, immutable transitions,
and canonical versioned board keys. `quoridor.episode` owns complete ordered
history, validates its keys and transitions, and distinguishes goal wins,
threefold draws, unlabelled truncations, and explicit failures. Terminal values
and complete-episode targets use each recorded player's perspective. Search
cache identity includes history, not just the current board.

`rules_contract` rejects legacy/unversioned artifact metadata and unsupported
new-rule backends. Native and browser remain historical implementations;
only Python is enabled for `quoridor-v1`. No native conformance is claimed or
skipped as a substitute for passing it. The unchanged native acceptance
condition applies before any new-version native backend can be enabled.

Independent review found three initial episode-constructor defects: premature
ply-limit outcomes, garbage history keys, and continuing after an already
completed draw. All were fixed with strict validation and regression tests;
five reviewer probes now reject correctly. Evidence and copyable reproduction
commands are saved in [docs/verification-m1.md](docs/verification-m1.md).

## M1 Executed Verification

| Command / check | Observed result |
| --- | --- |
| Baseline `make check` | Exit 0; 82 passed in 10.03s |
| Engine focused pytest (command in report) | Exit 0; 66 passed in 7.74s |
| Episode focused pytest (final) | Exit 0; 45 passed in 0.80s |
| Final `make check` | Exit 0; **226 passed in 18.15s**, no skips: 144 M1 + 82 existing tests |
| Lock/environment/YAML/digests within `make check` | All passed; 22 packages match, offline sync reports no changes |
| Comparison against starting commit | 10 config/fixture files byte-identical; all M0-M6 acceptance criteria and lock unchanged |
| Independent review regressions | All 5 invalid-history/outcome probes rejected |
| `git diff --check` | Exit 0 |

`make check-m1` is the focused M1 reproduction entry. The full gate includes
tactical cases, actual Hypothesis reachable-state properties, full action-mask
checks, real legal 256-ply boundary traces, and legacy boundary tests. No test
was skipped or weakened to obtain completion, and no frozen input was changed.

No training/strength experiment was required or run. Synthetic unit tests are
not self-play jobs or strength evidence. A fake-clock integration verifies a
goal at an overrun keeps its game label while temporary RunSession JSONL/summary
records elapsed time and `budget_exhausted`. Search/actor/optimizer scheduling
enforcement remains M3/M4/M6; checkpoint/replay migration remains M2/M4.
No local historical model, replay, dataset, or GPU ledger was modified.

## M2 Work and Baseline

M2 is the highest-priority incomplete milestone with all dependencies met (M1).
The current request authorizes this follow-up beyond the prior M1 stop point.
Starting revision: `02a3bc1`, clean worktree. Before implementation, `make check`
passed **226 tests in 17.97s**, with no skips; environment/lock/YAML checks passed.

This is engineering implementation and deterministic verification, not a training
or strength experiment. No historical model is being adapted or trained. The
first encoder explicitly uses partial history observation, with a deterministic
same-input/different-history test demonstrating different future draw outcomes.
The exact-history responsibilities and limits are documented in
[docs/model-v1.md](docs/model-v1.md). Frozen inputs and acceptance criteria remain
unchanged.

Implemented `encoder` and stdlib-only `symmetry` for versioned 16-plane features,
full-history/action/policy transforms and current-player value semantics. Legality
and distances derive only from the engine; stopped episodes have zero action
support. `policy_value` is a pure tensor residual network with external `policy`
masking and JSON-safe `value_diagnostics`. `configs/model.yaml` / `model_config`
declare and validate the untuned 64-channel, 4-block architecture. Root legacy
exports now load lazily, preserving their behavior without importing rules when
the tensor-only model is imported.

`model_checkpoint` validates complete rule/codec/value/encoder/architecture
metadata, state-dict shapes/dtypes/finite values and optional expected hashes.
Temporary save/reload tests reproduce inference outputs, retain caller RNG, and
reject old 14/16-plane assets. It saves model weights only; optimizer/replay/RNG
training continuation is still M4. No historic checkpoint has been converted.

## M2 Executed Verification

| Command / check | Observed result |
| --- | --- |
| Baseline `make check` | Exit 0; 226 passed in 17.97s |
| Model + YAML focused pytest | Exit 0; 102 passed in 2.66s |
| Model checkpoint focused pytest | Exit 0; 66 passed in 4.23s |
| Encoder/symmetry focused after CPU fix | Exit 0; 59 passed in 50.17s; final added deadline groups separately passed 10 cases |
| Final cumulative `make check` | Exit 0; **471 passed in 74.37s**, no skips (245 M2 + 226 existing) |
| Lock/environment/config gates | All passed; 22 packages unchanged, offline sync makes no changes |
| Frozen-input/criteria comparison | 10 prior config/fixture files, all acceptance criteria, lock and legacy network/checkpoint/game byte-identical |
| `make help`, `git diff --check` | Exit 0 |

Full commands, acceptance mapping, partial-observation evidence and limitations
are saved in [docs/verification-m2.md](docs/verification-m2.md).
`make check-m2` reproduces the complete focused M2 suite. The existing M1 import
test now guards the very first package import; this strengthens isolation after
lazy exports and does not change a frozen fixture or reduce its assertions.
Review also led to explicit CPU tensor factories and clear sparse-input errors.

No self-play/training job, GPU/native workload, new arena, strength experiment,
or historical asset modification occurred. Synthetic gradients and checkpoint
roundtrips are unit tests. Actual gameplay calibration and the strength cost of
partial history observation remain unmeasured. Any later experiment must first
register its hypothesis, YAML configuration and finite time budget.

## M3 Work and Baseline

Selected M3 as the highest-priority unfinished task with all dependencies met
(M1 and M2). The follow-up request authorizes work beyond the previous M2 stop.
Starting revision: `1d45df03b0952ad0e1da49eda6832ae7c9d4bc5b`, clean worktree.
Before any implementation, `make check` passed **471 tests in 75.01s**, exit 0,
no skips. All 22 locked packages, offline lock/sync, three YAML configs and
frozen arena/opening checks passed.

M3 implements new-rule PUCT and model inference independently of the legacy
search. Validation uses deterministic oracles and bounded synthetic inference;
no training or strength experiment is required. Existing acceptance conditions,
frozen configurations/fixtures and the dependency lock remain unchanged.

Implemented `puct-v1` with complete legal expansion, parent-perspective PUCT,
exact outcome leaves, full-history tree keys, and `root-visits-v1` raw visit
targets. Root expansion does not count as a simulation but consumes the shared
time budget; each complete backup contributes one root visit.
The evaluator bridges M2 canonical features/model outputs back to original
coordinates without changing the current-player value sign. Strict eval mode,
finite outputs and validated YAML settings are enforced. Arena inference remains
deterministic; selfplay noise uses a local seeded RNG at the root only.

Search-local cycles are approximate zero-valued leaves, not game draws. True
third repetitions use the exact episode outcome first. Simulated ply-limit
truncation stops the search without backing up an unknown value. Deadline
exhaustion retains complete prior visits but returns no selected action; a
partial search cannot masquerade as a completed move or provide game labels.
The new APIs and versioned conventions are in [docs/search-v1.md](docs/search-v1.md).

Review found an internal gamma rejection loop could bypass outer deadline
checks for finite alpha `1e308`: a 0.1-second budget exceeded an external
3-second timeout. Deadline checks now run inside each random draw; the same
probe returned `budget_exhausted` in 0.1007 seconds with no action or visits.
Independent fake-clock regression and reference seeded priors verify the fix
without changing acceptance conditions or banning the test input. There are
no remaining known blockers for M3.

## M3 Executed Verification

| Command / check | Observed result |
| --- | --- |
| Baseline `make check` | Exit 0; 471 passed in 75.01s, no skips |
| Final `make check-m3` | Exit 0; **115 passed in 11.95s**, no skips |
| Final cumulative `make check` | Exit 0; **586 passed in 85.70s**, no skips (115 M3 + 471 existing) |
| Environment, offline lock/sync, YAML and frozen hashes | Passed; all 22 packages unchanged; offline sync reports no changes |
| Frozen input / criterion comparison | 11 prior config/fixture files, all M0-M6 acceptance criteria, lock, existing rules/legacy search and 10 prior test files unchanged |
| `make help`, `git diff --check`, roadmap/result consistency | Exit 0 |

Actual output, focused checks, defect evidence, acceptance mapping and copyable
reproduction commands are saved in [docs/verification-m3.md](docs/verification-m3.md)
and [docs/verification-m3-results.jsonl](docs/verification-m3-results.jsonl).
The tests include both players' immediate wins, odd/even terminal and nonterminal
backup, formula/tie/target checks, Hypothesis legal-support invariants,
same-board/different-history adjudication, actual M2 model search, and a
temporary RunSession JSONL integration that truthfully records partial budget
exhaustion. Existing M0 tests still verify all three execution stubs fail nonzero.

No selfplay/training job, GPU/native workload, new arena, strength experiment,
or historical asset modification occurred. No playing-strength improvement is
claimed. Timing is cooperative and cannot preempt an in-flight rule/model call.
The reference implementation uses fresh trees, one-position inference and full
episode validation; performance optimization and training scheduling are later
work. Any later experiment must first register hypothesis, YAML configuration
and finite time budget.

## Historical M4 Selection

M3 is complete; M4 is now in progress with its dependency satisfied. This round
implements bounded selfplay, fresh compatible replay, policy/value training,
complete checkpoint/resume and actual `make smoke`; it stops before M5.

## M4 Baseline and Registration

The highest-priority eligible task is M4. Before implementation, the cumulative
`make check` passed **586 tests in 85.73s**, exit 0, with no skips. The locked
environment, offline sync, YAML checks and frozen arena/opening hashes passed.
The starting worktree includes the uncommitted, verified M3 changes; no existing
M3 or frozen input is being reverted.

The requested long-run starting settings are recorded as design defaults:
NetworkConfig width 64 with 4 residual blocks, 128 simulations per selfplay
move, replay capacity 100,000, batch size 128, and 50–100 new games per round.
M4 smoke uses a separately declared tiny workload and a finite wall-clock budget
so CPU verification is bounded; it records effective settings and does not claim
long-run performance. No strength experiment is registered or run.

The M4 hypothesis is operational: a complete new-rule selfplay game can produce
versioned root-visit policy targets and current-player terminal labels; one finite
CPU optimizer update can consume them; a saved artifact can reload with identical
model, optimizer, RNG, replay lineage and configuration identity. The smoke
configuration, seed and time budget are registered before execution; truncation
is retained as a count and excluded from replay.

`make smoke` is now implemented as the real `smoke-run` workload; the historical
direct `project_cli smoke` compatibility stub remains covered by the M0 test.
`make arena` and `make experiment` remain explicitly unimplemented for M5/M6.

## M4 Executed Verification

M4 is complete. The final focused gate `make check-m4` passed 17 tests in 6.50s,
with no skips; the final cumulative `make check` passed 603 tests in 90.74s,
with no skips. Environment, locked offline sync, YAML and frozen arena/opening
checks passed, and `git diff --check` passed.

The actual final `make smoke CONFIG=configs/smoke.yaml` command exited 0 and
created `runs/smoke/20261010T130800Z-dc36bd89ff77403a8c7a6e4526004ec1`. It recorded one
completed tactical immediate-goal game (one ply), one admitted replay position,
one optimizer update, and a successful training-resume reload. The effective
model was the requested 64-channel/4-block architecture; smoke used 8
simulations, root noise and temperature 1.0 from `configs/smoke.yaml`, replay
capacity 16, batch size 1 and one optimizer step. Its training checkpoint SHA is
`51783afe5dfa29093cbb795abb7eebcb3413ac52e8d46375190969b227512dde`; the saved
inference model SHA is `57368edc411e5fc2e50dc00177c70a1b91b21cc4fcf946aa7b15e78bef3ce278`.
The loss was `1.000000238418579`, elapsed time 1.166s, and run status was
`completed` with zero truncated/failed games.

The run's JSONL, summaries and artifacts remain under the ignored run directory.
Exact commands, output, hashes and acceptance mapping are in
`docs/verification-m4.md` and `docs/verification-m4-results.jsonl`. New tests
cover repetition-draw zero labels, whole-game stop exclusion, canonical policy
transforms, finite masked updates, deterministic eviction, and fail-closed
lineage/RNG/replay checkpoint restoration. No long-run training or strength
experiment was run; the tactical smoke is an engineering path check only.

M4 is complete and this round stops before M5. `make arena` and `make experiment`
remain explicitly unimplemented.

## M5 Selection and Implementation

This follow-up selected M5, the highest-priority unfinished milestone with all
dependencies satisfied (M4 complete), and supersedes the historical M4 stop
above. Starting revision was `eaa2ca37ccf7054829cfcab4fc2d6723e9ce6f55`, with a
clean worktree synchronized with origin. Before implementation, `make check`
passed **603 tests in 89.87s**, no skips; lock/environment/config gates passed.

Implemented `arena_v1` and `make arena` with explicit compatible model-only
candidate/champion snapshots, byte hashes and metadata validation, all 200
development prefixes replayed under M1 with complete history, side-swapped
games, real PUCT and per-search/game JSONL. The unchanged protocol uses CPU,
256 simulations, no noise, temperature zero and one shared 3600-second budget.
It includes model loading, prefix validation, search, statistics and finalization.

Independent statistics score draws as 0.5, retain every completed game's
diagnostic score, and bootstrap whole complete pairs using the frozen central
95% interval. All 400 games and error-free execution are required before the
unchanged score >= 0.55 and lower bound > 0.50 gates can pass. Interrupted or
truncated runs retain evidence and exit nonzero. Review found and fixed a
finalization overrun that could leave a stale promotion flag; regressions now
cover deadline expiry and exceptions after actual result publication. Consumers
must require both completed run and result artifacts; JSONL summary events are
explicitly provisional. No champion store was updated.

Fixed random-legal, move-only shortest-path and classic diagnostic opponent APIs
are versioned separately. Classic is an optional Node adapter with fixed
587-node/depth-5 settings and explicitly documented legacy internal heuristics;
it is not a compatible neural champion. A real 15-second-budget bridge check
finished in 5.5758s with repeatable legal actions for both players.

Fresh final-suite generation was preregistered in
`configs/reserved-final-v1-generation.yaml`: 200 legal 16-ply prefixes, seed
2026101005, 180-second budget, no models or performance-based selection.
Generation completed in 100.3968s; independent regeneration completed in
100.0880s and reproduced identical bytes. The SHA-256 is
`2543fb5935d06a3de3afb67fae290d014c5a4a40d48bde0af97d3d29113f55dd`.
All final board symmetry orbits are distinct and excluded against the four
historical suites. This freezes new assets before M6; it is not a strength
evaluation. Formal final use needs a separately frozen execution protocol.

## M5 Executed Verification

| Command / check | Actual result |
| --- | --- |
| Baseline `make check` | Exit 0; 603 passed in 89.87s, no skips |
| Final `make check-m5` | Exit 0; **87 passed in 36.17s**, no skips |
| Final cumulative `make check` | Exit 0; **690 passed in 123.08s**, no skips |
| Locked environment, offline lock/sync, YAML and frozen digests | Passed; all 22 packages unchanged |
| Frozen-input/acceptance comparison | 31 prior config/fixture/dependency/test files byte-identical; all M0-M6 acceptance arrays unchanged |
| `make help` | Exit 0 |
| `git diff --check`, new-file whitespace and YAML/JSONL status checks | Passed |
| `make arena` with no models | Make/CLI exit 2; explicit missing-input error |
| `make experiment` | Make exit 2, CLI exit 3; explicit NOT IMPLEMENTED for M6 |
| Legacy champion compatibility attempt | Make exit 2, CLI exit 1; incompatible checkpoint rejected before any game |

Full commands, actual output, acceptance scope and failure reproduction are in
[docs/verification-m5.md](docs/verification-m5.md) and
[docs/verification-m5-results.jsonl](docs/verification-m5-results.jsonl).
[docs/reserved-openings-v1.md](docs/reserved-openings-v1.md) records generation
and independent regeneration commands/results. Real neural tactical goal/draw
tests and a synthetic 400-game schedule are engineering verification only;
neither is a completed frozen neural comparison. No old test, frozen fixture,
evaluation setting or acceptance condition was weakened. Historical models,
replay, datasets and GPU ledger were not modified.

## M5 Blocker and Next Task

M5 remains **in progress**; M6 is not started. The outstanding acceptance is an
actual complete 200-pair/400-game comparison under the frozen arena protocol.
The existing legacy champion is incompatible with M2 model-only checkpoints.
The actual attempt is retained at
`runs/arena/20261010T135746Z-58708042fd9b42cebe4cd106fdbc8ec0`, status `failed`,
elapsed 0.6606s, zero games, with `checkpoint fields mismatch; legacy checkpoints
require explicit migration`. Compatible M4 smoke weights exist, but were
deliberately initialized for a tactical one-ply engineering workload; no
compatible candidate/champion pair has been designated for this comparison.
No full arena, promotion or playing-strength improvement is claimed.

Next task remains M5: designate compatible immutable model snapshots, register
their exact hashes, comparison hypothesis and the frozen 3600-second budget,
then run `make arena CANDIDATE=... CHAMPION=...` as documented. If it cannot
complete within that budget, preserve the incomplete result and measured
bottleneck; do not reduce pair count, simulations or confidence requirements.
The blocker and pending acceptance are also recorded in `tasks.yaml`.
