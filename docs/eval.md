# Validation and Strength Evaluation

Status: M0 evaluation contract, 2026-10-10. Passing infrastructure checks is not
evidence of playing strength. `progress.md` records actual commands and results;
this document defines what later milestones must establish.

## Engineering Acceptance

`make check` is the cumulative CPU validation gate. It validates the new YAML contracts,
the frozen arena configuration/opening hashes, run-directory and JSONL helpers,
deadline behavior, and explicit failure of unfinished execution entry points.
pytest covers boundary/error behavior; Hypothesis covers input variation and
invariants. Selected legacy CPU reference tests protect retained functionality.
`make check-m1` isolates the new rules, episode outcomes and compatibility tests;
the cumulative check includes these tests. The full browser/native/historical
driver suites are separate checks and must not be implied by M0/M1 results.
No new network training or arena games are required for M0/M1 acceptance.

Later acceptance must cover:

| Milestone | Evidence required |
| --- | --- |
| M1 | Rules, reversible 209 codec, complete masks, path preservation, jumps, illegal-action isolation, goal/repetition/truncation distinction |
| M2 | Finite logits/value, exact tensor shapes, perspective/action transforms, value-sign tests, encoder history decision, checkpoint compatibility |
| M3 | Deterministic PUCT fixtures, odd/even backup signs, goal/draw leaves, legal visit distribution, history/cycle handling, deadline propagation |
| M4 | Bounded self-play/train/save/reload smoke, visit-based targets, correct final labels, excluded truncated games, JSONL/provenance and resume gates |
| M5 | Frozen hash verification, color-swapped pairs, deterministic evaluation, incomplete-match rejection, scoring and confidence fixtures |
| M6 | Budgeted reproducible experiment, complete artifact lineage, frozen arena results against declared baselines, measured resource use |

Property tests use reproducible seeds/examples and shrink failures into regression
cases. Differential tests compare the Python reference with any retained native
backend only after both implement the same new rules version; comparing two
legacy implementations does not certify the new repetition policy.

## Frozen Arena Protocol

`configs/arena.yaml` and `configs/arena.sha256` define the versioned M5 protocol.
Validate its raw-file SHA-256 and the opening-suite SHA-256 before creating games.
The initial protocol uses CPU, seed `20261010`, 256 simulations per move,
`root_noise: false`, `temperature: 0`, and the declared fixed PUCT constant.
Both players use the same search/device budget. No hidden heuristic pruning,
teacher guidance, post-hoc value scaling, or candidate-specific search settings
are allowed.

Use 200 development opening pairs (400 games). For each opening, play the
candidate once as each player against the same champion. Keep the paired opening
and seed schedule fixed, and record any deterministic tie-break. The development
opening suite is existing `training/fixtures/openings/dev.json`; its legacy hash
format does not prove its states satisfy the new episode contract. M5 must load
and validate each opening under M1 and define the inherited repetition history
explicitly. If the fixture lacks a full path/history, use a declared fresh episode
from that position and version that opening interpretation in the protocol.

`training/fixtures/openings/final.json` was used in historical Phase 4F evaluation;
it is not an untouched holdout. Keep it for legacy comparisons and declare that
exposure in any reused result. M5 must freeze a fresh reserved final suite before
M6 tuning if the final comparison is to support an independent holdout claim.
Tuning uses development evaluation only. Do not inspect final performance
repeatedly to choose models, temperatures, value scales, or architecture settings.
Reference opponents include random-legal, a shortest-path/simple wall baseline,
the existing classic AI, and the current champion where compatible; each gets
a declared fixed configuration and version. No incompatible legacy champion can
be used without a documented adapter and migration assessment.

Candidate/champion paths are explicit future arena inputs. Record SHA-256 of
each model and its architecture/encoder/rules metadata. Artifact names such as
`latest` are not immutable opponent identities. Promotion creates a new versioned
champion identity only after all gates pass; M0 creates no champion.

## Scores and Confidence

A completed win scores 1, completed loss 0, and repetition draw 0.5. Report wins,
losses, draws, total scheduled/completed games, complete/incomplete pairs,
truncations by reason, illegal actions, numerical failures, and runtime.
The candidate score is the mean of all completed game scores only when all
scheduled games and pairs finished without eligibility errors. A partial score
may be shown as a diagnostic and must carry `decision_eligible: false`.

The unit of resampling is an opening pair, not a single game. Let a pair's score
be the mean of its two candidate game scores. Use 10,000 bootstrap resamples of
the 200 pair scores, with the frozen seed, and report a percentile 95% confidence
interval. Promotion requires mean score at least 0.55 and confidence lower bound
strictly above 0.50, in addition to complete pairs, no illegal actions, no
numerical/protocol errors, and exact identity/configuration binding. M5 tests the
bootstrap and thresholds with independent synthetic fixtures.

Any truncated game makes its pair incomplete and the entire comparison
ineligible. Budget exhaustion, early stopping, favorable subsets, and failed
games cannot become wins or losses. Do not discard them and then promote based
on a reduced denominator. A rerun follows an explicit fresh protocol/run identity
or validated identical-config resume; changing the budget is a new protocol.

An Elo estimate may be reported as a secondary descriptive statistic with its
assumptions and uncertainty. The frozen paired score/confidence gates control
promotion. A low training loss, teacher imitation accuracy, self-play win rate,
or successful checkpoint reload does not substitute for those gates.

## Experiment Discipline

Every M6 experiment declares a hypothesis, one controlled change, baseline,
seed(s), maximum wall time, hardware/device, search budget, stopping conditions,
and result-selection rule before execution. Keep complete config, lockfile/code
hashes, effective device, replay and checkpoint lineage, JSONL events, and final
summary under its run directory. Report elapsed time, games, truncated fraction,
samples, optimizer steps, throughput, and arena identity alongside loss metrics.

Reproduce the selected result from an independently identified run when the
budget permits. Numerical failures, reruns, interruptions, and exhausted budgets
remain visible in the record. A failed or unfinished experiment is a valid
reported outcome, not successful training. The M0/M1 engineering tasks run none
of these experiments.

## Historical Evidence and Current Claims

The repository contains substantial prior self-play, replay, checkpoints, and
phase-specific evaluation reports. The most recent inspected F64 result is
175 wins, 225 losses, zero draws, 400 games: 43.75%, without promotion. Prior
reports used legacy adjudication and protocols; their scores are not directly
comparable with this contract. M0 reads those artifacts to understand progress
and failure modes but does not rerun them or validate their complete provenance.

Only commands listed as executed in `progress.md` count as verification for this
round. Planned tests, existing reports, configuration parsing, and unimplemented
entry-point failures must be labeled separately from new self-play/training or
arena performance.
