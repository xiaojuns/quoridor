# Quoridor AlphaZero Training

Phase 2 provides a correctness-first Python reference pipeline:

- Phase 0-compatible rules and 209-action codec;
- canonical state encoding and left-right augmentation;
- SE-ResNet policy/value network;
- complete-action PUCT MCTS;
- self-play, replay, training, checkpoint resume, and arena evaluation.

The browser game remains independent of these Python dependencies.

Phase 3 adds a C++17 rule engine and multi-root MCTS. The native scheduler
batches one leaf from every active self-play actor into a single policy/value
network call while preserving the Phase 2 Python implementation as the
correctness reference.

## Environment

The checked local environment is Python 3.10, PyTorch 2.7.1 with CUDA 12.6,
and an NVIDIA Quadro RTX 6000. Run modules from the repository root with:

```bash
PYTHONPATH=training python3 -m quoridor.cli smoke --device cpu
```

CUDA validation uses:

```bash
PYTHONPATH=training python3 -m quoridor.cli smoke --device cuda
```

Resume model, optimizer, RNG, step, and replay state from the latest checkpoint:

```bash
PYTHONPATH=training python3 -m quoridor.cli smoke --device cuda --resume
```

Add `--full-game` to start smoke self-play and arena games from the standard
initial position. Without it, smoke uses a legal near-terminal position so CI
checks the complete data path quickly.

Run the configured reference iteration from the standard initial position:

```bash
PYTHONPATH=training python3 -m quoridor.cli train --device cuda
```

The reference command is intentionally much more expensive than smoke. Add
`--resume` to continue its latest checkpoint and replay buffer.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=training \
  python3 -m unittest discover -s training/tests -v
```

## Native Engine

Build the C++ core, Python extension, and native unit tests without downloading
another toolchain:

```bash
./scripts/build-native-engine.sh
```

The build uses GCC/CMake/Ninja and the pybind11 headers already bundled with the
installed PyTorch. Build output stays below `native/build/` and is ignored.

Run the complete Phase 3 acceptance workload:

```bash
./scripts/run-phase3-validation.sh
```

It checks 48 committed fixtures, 10,000 deterministic reachable states and
transitions, 100 complete random games, zero illegal actions, at least 20x
legal-action, prevalidated-transition, and combined rule throughput, and a
batched native-MCTS train/resume smoke. The acceptance script takes no workload
override arguments, so a reduced development run cannot be reported as a full
pass. Timestamped JSON reports are written below
`training/runs/native-validation/`. Keep the complete terminal output and both
report paths for review.

For a shorter build/integration check:

```bash
npm run build:native
npm run smoke:native
```

## Native Training

Run one reference training iteration on CUDA:

```bash
./scripts/train-native.sh
```

Run several checkpoint iterations or select another config/device:

```bash
./scripts/train-native.sh --iterations 3
./scripts/train-native.sh --config training/configs/smoke.json --device cpu
```

The script builds the native extension first and automatically adds its output
to `PYTHONPATH`. If the configured run directory contains checkpoint files, it
adds `--resume` and restores model, optimizer, RNG, step, replay, and the
accepted champion. The files are scanned directly even if `latest.json` was not
written before an interruption. Each completed iteration writes a
checkpoint; terminal output omits the long per-step metric list, while the full
iteration summary and logs remain under the configured run directory.

Stop a foreground run with `Ctrl+C`. A checkpoint is published only after its
arena finishes, so the interrupted iteration is discarded and the last
completed checkpoint remains valid. Run the same command again to resume from
it. Rejected candidates remain trainable but do not replace the accepted
champion used for self-play. Each resumed generation gets a new persisted actor
seed. Both supplied configs retain the 8 GiB artifact cap, and the driver
refuses to start when the run directory filesystem has less than 12 GiB free.

## Phase 4 Throughput Calibration

The paired arena opening assets are frozen at
`training/fixtures/openings/dev.json` and
`training/fixtures/openings/final.json`. Each suite contains 200 fully replayed,
stratified legal prefixes; the suites are content-disjoint. Verify that the
committed files still match deterministic generation with:

```bash
./scripts/generate_phase4_openings.py --check
```

The frozen schema `contentHash` values are shown below. They bind canonical
opening content for arena policy checks; they are not raw JSON file digests.

```text
dev:   sha256:9cde8e3302916d023957b60d4a433f31f0dfc995881389aca95daee4730b11c4
final: sha256:65f1bb5223e8e7c9da469b69b891443458d631e8058cfb4a14f57f3d1243833d
```

The fast screen uses the first 32 openings in canonical suite order; formal
promotion uses all 200 `dev` openings. The locked `final` suite is reserved for
the final comparison with the classic hard AI and must not be used for tuning.

Run the short wiring smoke before committing GPU time:

```bash
./scripts/calibrate-phase4.sh
```

This default `quick` profile uses native MCTS and CUDA inference, but its report
always has `decisionEligible: false`. It does not train, encode tensor samples,
or write checkpoints. It does capture raw states, actions, visit counts, and
root values through the compact `ReplayStore`, so the measured games/hour
includes the production replay write path.

After the quick report has been reviewed, run the fixed decision matrix with:

```bash
./scripts/calibrate-phase4.sh --profile full
```

The full profile covers both `128x10` and `192x16` networks, actor counts
`8/16/32/64`, and `128/256` simulations. The default two games per actor runs
16 scenarios and 960 complete games; this minimum is required so every actor
configuration exercises dynamic slot refill. The two network sizes share the
same workload seeds and run in an interleaved order. Values below
`--full-games-per-actor 2` can run for diagnostics but cannot produce a
decision-eligible report.

The completed full matrix selected `192x16` under the frozen 60% throughput
rule. `phase4-bootstrap.json` is the selected 64-game pilot workload and uses
32 actors, matching the measured `a32/s256/g64` scenario. Workloads with at
least 128 games may use 64 actors, matching the measured refill workload.
`phase4-small-pilot.json` remains only as a diagnostic counterfactual.

The launcher builds the native extension, requires PyTorch CUDA, and enforces
the confirmed local-only limits across `training/runs/phase4`: 8 GiB of
artifacts, at least 20 GiB free disk, and 72 total GPU-hours. A single-writer
lock prevents concurrent Phase 4 calibration, while a self-hashed GPU ledger
automatically carries calibration time across restarts. Use
`--gpu-hours-used HOURS` only for Phase 4 GPU work performed outside that
ledger; its recorded floor cannot be reduced by later commands.

Every run gets an immutable `run-manifest.json`, compact replay shards, and an
atomically replaced `report.json` under
`training/runs/phase4/calibration/<run-id>/`. The report includes games/hour,
scheduled simulations/second, evaluator batch-size p50/p95, `nvidia-smi` GPU
utilization, termination distribution, replay verification, and peak CUDA
memory. A full report is not decision-eligible if telemetry or any required
metric is missing. Stop with `Ctrl+C` or `SIGTERM`; completed scenarios remain
in an `interrupted` report, which is never decision-eligible.

## Phase 4D Training Gates

Phase 4D completed its sequential gates. The gate-v2 current-source `192x16`,
batch-512 CUDA AMP memory smoke passed:

```bash
./scripts/run-phase4d-training-memory-smoke.sh
```

The smoke performs four real training updates while forbidding self-play, replay,
checkpoint, and model writes. A pass requires actual CUDA GradScaler and FP16
autocast observations, Torch peak-memory data, at least two valid `nvidia-smi`
samples, and all shared resource limits.

The CPU-only classic-hard teacher set has also completed:

```bash
./scripts/generate-phase4-teacher-data.py
```

Stop with `Ctrl+C`; complete games and the cursor remain durable. Run the same
command again to continue. `--verify-only` deeply validates an existing set
without starting the Node bridge. The default target is 50,000 positions with
whole-game 90/10 train/holdout isolation and horizontal-mirror-orbit dedup.

The current-source warm model was generated in a separate v4 lineage with:

```bash
./scripts/pretrain-phase4d-teacher.sh --run-root training/runs/phase4/teacher-pretraining-v4 --new
```

The verified teacher set is fixed at
`training/runs/phase4/teacher-data`; noncanonical teacher roots are rejected so
the later pilot can reproduce the same deep dataset verification. This is a
resumable CUDA AMP job. It writes the exact model-only warm artifact,
the self-hashed report, completion manifest, and summary. The verified current
bundle is
`training/runs/phase4/teacher-pretraining-v4/summary.json`; v2 and v3 remain
historical evidence for earlier source lineages.

The old default `training/runs/phase4/phase4d-pilot` already contains a partial
pilot bound to pre-gate-v2 source and must not be reused. Stage confirmation was
given on 2026-07-26; start the current warm/cold pilot with the v4 summary and a
fresh output root using this complete one-line command:

```bash
./scripts/run-phase4d-warm-cold-pilot.sh --pretraining-summary training/runs/phase4/teacher-pretraining-v4/summary.json --output-root training/runs/phase4/phase4d-pilot-v2
```

Use that same complete command to resume `phase4d-pilot-v2`; omitting
`--output-root` would select the stale default root.

The completed pilot selected warm after a 62-2 paired screen: raw win rate was
`0.96875` and the opening-pair bootstrap one-sided 95% lower bound was
`0.921875`. Both lanes completed five generations and 320 replay games with no
teacher references. This is explicitly a bootstrap screen, not a formal
champion declaration; the selected continuation artifact is the warm
generation-4 checkpoint.

The pilot alternates `cold gN` then `warm gN`, with the same seed and five
generations per lane. Every trainer call uses an absolute cumulative target, so
re-running the command safely reconciles a checkpoint whose report was
interrupted without advancing an extra generation. After both lanes reach
generation 4, the script runs 64 paired games on the frozen first 32 dev
openings at 256 native-MCTS simulations per side. Warm is selected only at a
raw win rate of at least 55% and a paired one-sided 95% lower bound above 50%;
this screen never declares a formal champion. An interrupted arena restarts,
and its partial JSONL is not a result.

Pilot preparation accepts only the fully verified production pretraining
bundle: teacher stores, current memory-smoke evidence, eight fixed-holdout
evaluations through step 512, run/completion manifests, report, checkpoint,
and warm artifact must share one lineage. The base protocol is the exact
checked-in `training/configs/phase4-bootstrap.json`; `--base-config` may point
to an identical copy but cannot change hyperparameters. Preparation freezes
the current default source fingerprint and loaded native extension hash, and
every resume/validation checks both before another lane can start.

The formal raw-replay driver is available through:

```bash
./scripts/train-phase4.sh --config training/configs/phase4-bootstrap.json \
  --generations 1
```

Its default mode creates a lineage once and strictly resumes it thereafter.
`Ctrl+C` preserves complete replay games and the last checksummed checkpoint;
rerun the same command to continue. Phase 4D runs are capped at five committed
generations and share the 10 GPU-hour stage budget. The reference CLI still
rejects `phase4-*.json`, preventing accidental use of the legacy in-memory
replay or old arena path.

For orchestration and crash recovery, prefer the idempotent cumulative form:

```bash
./scripts/train-phase4.sh --config path/to/effective-config.json \
  --target-generations 3
```

Targets are counts in `[0, 5]`; target `5` means generations `0` through `4`
are committed. `--target-generations` and `--generations` are mutually
exclusive.

## Phase 4E0 Preparation

Phase 4E0 is CPU-only. Its versioned preparation command extracts and verifies
the selected warm generation-4 model as a model-only baseline and makes the
14,175-ply warm replay an independently audited migration candidate. It also
builds/tests the native engine and commits the current-source CPU replay-input
gate required by later Phase 4E training:

```bash
./scripts/prepare-phase4e.sh
./scripts/prepare-phase4e.sh --verify-only
```

Stop either command with `Ctrl+C` and rerun the exact command. Complete
immutable artifacts are verified and reused; an incomplete replay transfer is
rolled back. Replay preparation intentionally performs repeated CPU deep
verification and can take several minutes; the launcher prints a stage message
before the quiet validation pass. The migration is evidence for the future
Bootstrap lineage and is not silently inserted into an existing run.

The checked-in
`training/configs/phase4e-execution-prepared-v1.json` intentionally has
`stageBudget.status=pending`. Consequently this command fails before native
build or CUDA access:

```bash
./scripts/train-phase4e.sh
```

The first Bootstrap initialization attempt exposed a driver scope bug:
`initialize_new_run()` referenced a bootstrap verifier that had only been
imported in another method. It raised `NameError` before any generation,
checkpoint, replay game, replay ply, or optimizer step was committed. The
failed invocation's measured `0.0003736264895026882 GPU-hours` remains charged
in the append-only ledger; the audited global total is
`10.041741419628574 / 72 GPU-hours`.

The CPU-only Bootstrap initialization repair and E1 rebaseline preparation are
complete. They do not authorize CUDA, E1 calibration, or Bootstrap training.
The final source fingerprint is
`a011bbc22c5c07627bcd25c2eae9b8d8f623b6eb970ea8b3d33839938fa57381`.
The failed Bootstrap lineage was preserved with incident self-hash
`a3165c13a606a94c42c6f59775e2244f760ebddcb35cede65aabb7a77c69295d`;
the old completed E1 evidence was superseded with incident self-hash
`70089388d5e4c2198de68ceea40868402b8f65316fd28bfe2ea9f3aade176d43`.
Do not rerun the old `phase4e-execution-frozen-v1.json` five-generation
command.

The current-source replay input gate passed and is sealed at
`training/runs/phase4/replay-input-gate/report-a011bbc22c5c07627bcd25c2eae9b8d8f623b6eb970ea8b3d33839938fa57381-ab09e513c2bfb953eac60bfa2f07677a451b8a08ec670560fd68bb705bc7e8b3-8e70f78c7d29e21eef999124206bcb8c83850d849b869ff866a7954ece14f6c5.json`
with SHA-256
`2643d1fc1e67cec4756f66525daab30a3af1e69902f94fa3108e9da7c8701959`.
Native tests passed 3/3 and the complete CPU Python suite passed 381/381.

The Phase 4 aggregate artifact ceiling remains 8 GiB and the free-disk stop
line remains 20 GiB. The user explicitly confirmed the new E1 calibration on
2026-07-28. The report review, frozen-v2 creation, and Phase 4E stage-ledger
rebase are now complete. Bootstrap reached generation 14, which passed health,
the 64-game screen, and the 400-game confirmation before becoming the first
formal champion.

The training launcher does not run promotion gates: each generation report
records `promotion.status=not-run`. After generation 4 is reviewed, health,
screen, confirmation, history, and champion publication remain a separate
explicitly confirmed stage. The completed block reached step `320` with `320`
replay games and `12,813` plies. The final full checkpoint SHA-256 is
`9af626e93f61f704df3df31017c93b8f606c6bb4db977d2dbd16e68df0007cae`, and the
candidate-model SHA-256 is
`5a410ca04ab244863ef7540812ac6eb3635a8b4a51061a7de3f71ce55effc639`.

Every new Phase 4 checkpoint uses the v2 commit contract: the full resume
checkpoint is paired with an immutable, lineage-bound `candidate-model` artifact
that contains tensors only. Formal promotion and arena identity bind the safe
model artifact SHA-256; the full checkpoint remains a resume/lineage reference.
Legacy v1 checkpoints remain resumable but cannot enter formal promotion.

Formal promotion uses the first 32 frozen `dev` openings for the 64-game screen
and all 200 `dev` openings for confirmation and history gates. The `final`
opening suite remains untouched until the final classic-hard-AI comparison.
Arena results are accepted only after every stored ply is replayed, and champion
publication is serialized, history-complete, constrained to the Phase 4 artifact
root, and guarded by the global 8 GiB / 20 GiB resource contract. The first
formal champion is `g00000014-s00000000-67b580c9087f`; best and product pointers
remain unset.

## Phase 4F Final Evaluation

Phase 4F0 is CPU-only and is complete. Verify its frozen evidence with:

```bash
./scripts/prepare-phase4f.sh --verify-only
```

The contract is `training/configs/phase4f-final-frozen-v1.json` with file
SHA-256 `bfd4abc327518b6215b27c16bd673ebb5247fe950fcc99d8d0a46fb73c82f6b2`.
The fixed classic-hard workload is 587 nodes per move. The wall-clock workload
requests 800 ms so the measured move remains below the strict 850 ms gate.

The fixed track completed at 71-329 (17.75%) and failed the product gate. The
wall-clock track is stopped because it cannot repair the mandatory fixed-track
failure. The following commands are retained only as historical runner syntax:

```bash
./scripts/run-phase4f-final.sh --track fixed
./scripts/run-phase4f-final.sh --track wall-clock
```

The CPU-only failure diagnosis can be reproduced with:

```bash
./scripts/analyze-phase4f-failure.sh
```

It deeply replays all 402 games and writes an immutable self-hashed report to
`training/runs/phase4/phase4f/failure-analysis/report.json`. Exact final-suite
positions and trajectories must not enter recovery training. A future product
gate requires a newly frozen unseen final suite. See
`docs/PHASE4F_FAILURE_RECOVERY.md` for the recovery boundary and GPU reserve.

## Phase 4E1 Calibration

The calibration contract is
`training/configs/phase4e1-calibration-v1.json` with canonical SHA-256
`47fbe72edd9c75c10b97b0b3f1475205bf6d1630234314a5d11d1ceb43899955`.
The CPU-only repair and preservation quarantine are validated, and the user
has explicitly authorized this rebaseline calibration. Start and resume it
with the identical command:

```bash
./scripts/calibrate-phase4e1.sh \
  --contract training/configs/phase4e1-calibration-v1.json \
  --device-index 0 \
  --gpu-hours-used 0
```

Do not edit fingerprinted source under `training/quoridor/`, `native/`, `src/`,
or `scripts/` while the run is active. `--gpu-hours-used` counts only usage
outside the shared ledger; all current usage is already recorded, so it must
remain `0`.

The launcher builds/tests the native module and verifies the current-source
replay-input gate before CUDA work. It then runs a `192x16`, 128-game,
64-actor, 512-simulation self-play calibration with 24 temperature moves,
followed by an 80-game, 64-actor, 800-simulation batched arena smoke over
`dev[:40]`. Both arena participants are the same audited warm Bootstrap model;
the run measures the production-shaped arena scheduler and is not a promotion
decision.

The exact arena search payload is independently bound to both selector
identities, the pair-store contract, and the report. Its canonical SHA-256 is
`89545b401d92a26c2d659ea4736ea6ffc76a56ab7733f8fe1c1f6636c0f6428a`;
it includes the Standard config hashes, PUCT parameters, zero temperature,
disabled root noise, and seed `20260714`, not only the 800-simulation count.

Press `Ctrl+C` to stop and rerun the identical command to resume. A committed
self-play report/replay or an atomically persisted complete candidate, plus
complete opening-pair artifacts, are deep-verified before reuse. Arena batch,
GPU telemetry, memory, and elapsed-time aggregates are persisted during the
run, so resumed evidence is not based only on the last attempt. Cached
components and partial pair stores are deep-verified before a new GPU-ledger
invocation. The final report embeds a self-hashed ledger snapshot; later runs
may be appended, but the snapshot's accounting records cannot be rolled back.

Artifact reuse applies only when the source fingerprint is unchanged. If an
incomplete run belongs to another source fingerprint, the launcher acquires
the global Phase 4 lock and atomically moves the whole run under
`training/runs/phase4/phase4e1/quarantine/` before querying CUDA or opening a
new ledger invocation. A self-hashed incident records the old manifest, full
file inventory, both source hashes, and the ledger snapshot; the replacement
run manifest binds that incident. Previous GPU time remains charged, and the
complete E1 workload starts again under the new source provenance.

E1 has a 6 GPU-hour cap and retains the global 72 GPU-hour, 8 GiB artifact, and
20 GiB free-disk stop lines. Its deterministic decision report is:

```text
training/runs/phase4/phase4e1/calibration/
phase4e1-calibration-47fbe72edd9c75c1/report.json
```

The command never trains, writes a training checkpoint, performs formal
promotion, or freezes the Phase 4E execution budget. The old report was
reviewed and the old CPU-only freeze was committed on 2026-07-28 as
`training/configs/phase4e-execution-frozen-v1.json`, with file SHA-256
`0ae33064ebd4ded3d9f9f0b96e9aaa89a736dcf3b5b05bcaa44d418afc9743ad`.
It freezes a `59.96104 GPU-hour` hard stage limit and a 64-generation ceiling.
Because source-attested training code is being repaired, that report and
frozen-v1 contract are historical provenance and must be superseded before
training can resume. The historical report and frozen-v1 contract were
superseded by frozen-v2 and the audited stage-ledger rebase. Global GPU usage,
including failed invocations, is never rolled back.

The replacement E1 calibration completed and passed the official verifier. Its
report is
`training/runs/phase4/phase4e1/calibration/phase4e1-calibration-47fbe72edd9c75c1/report.json`
with SHA-256
`c162921ba56ed0d4d6b86bc430a74963bddb54fe2029ec02478d34ac4c587039` and
`decisionEligible=true`. It used `2.656247726123418 GPU-hours`, bringing the
global ledger to `11.115946631046116 / 72 GPU-hours`. The 128-game self-play
and 80-game arena smoke completed with no training, checkpoint, or promotion.
The report is reviewed. Frozen-v2 SHA-256 is
`453c60235859f417d5f7b295d0d764ac5778a7b9927bdfcedcb4dade0b304264`, the
completed rebase receipt SHA-256 is
`3c7cdd651b107fb4204a020af2813145f1c693e76d164a8a0ac2f045492c6849`, and the
stage ledger now binds the new report. Bootstrap training still requires a
separate explicit confirmation.

## Artifacts

Experiment outputs are stored below `training/runs/` and ignored by Git. Both
provided configs enforce an 8 GiB artifact limit. The reference config is not a
claim that its default budget is sufficient for final-strength training; native
engine throughput and disk planning are Phase 3 and Phase 4 concerns.
