# Engineering Design and Migration Boundary

Status: M0 design, 2026-10-10. `docs/spec.md` is the behavioral contract;
`tasks.yaml` schedules its implementation. Existing training code remains a
legacy implementation until the corresponding milestone has been accepted.

## Existing Repository

The repository already contains a browser game/classic AI (`src/`), a Python
reference engine and AlphaZero pipeline (`training/quoridor/`), a C++ backend
(`native/`), cross-language fixtures, many legacy phase-specific drivers, and
historical training artifacts. This is a migration over an existing system.

Reusable foundations include the 209-action codec, full legality/path checks,
current-player MCTS value convention, residual policy/value network, replay and
checkpoint helpers, paired arena infrastructure, and opening fixtures. The
current large model is approximately 192 channels by 16 residual blocks with
14/16-channel inputs. The legacy network module also performs game feature
derivation; M2 separates that responsibility from tensor inference.

Legacy rules declare repetition/ply-cap losses, legacy drivers mostly consume
JSON configuration, and old replay assumes decisive +/-1 results. Those paths
remain historical evidence; new entry points do not delegate to them and claim
that the new contract has run. The latest inspected F64 report records 175 wins
and 225 losses in 400 games (43.75% score), without promotion. This is a historical
read, not a result reproduced in M0. Source and artifact inventory belongs in
`progress.md`; no old run is deleted or mutated to fit the new specification.

## Layer Responsibilities

| Layer | Owns | Must not own |
| --- | --- | --- |
| Engine / action codec | Board state, legal moves, paths, walls, goal, reversible actions | PyTorch, inference, optimizer, clocks |
| Episode | Exact position history, repetition draw, outcome records, ply count | Network architecture |
| Encoder | Versioned features and perspective/action transforms derived through engine APIs | A second independent rule implementation |
| Network | PyTorch tensor forward pass, architecture metadata | Legality, wall rules, repetition adjudication |
| Search | PUCT edge statistics, evaluator adapter, simulation history, deadline checks | Training optimization or arena promotion |
| Self-play / replay | Game records, visit targets, complete-game value labels, sampling lineage | Invented outcomes for runtime truncation |
| Trainer | Optimization, checkpoint/resume compatibility, seed control | Arena configuration changes |
| Arena | Frozen matched protocol, paired scoring, confidence and eligibility | Training noise, on-the-fly protocol tuning |
| Run infrastructure | YAML validation, hashes, run metadata, JSONL, elapsed time | Declaring an unimplemented pipeline successful |

The target dependency direction is orchestration -> episode/search/encoder ->
engine, and evaluator -> encoder + network. The engine is independently importable
and testable without neural dependencies. The model can consume synthetic tensors
without a game object. Native acceleration is optional after reference contracts
pass; it never supplies a different rule or value convention.

## Planned Public Contracts

The exact classes are finalized in their owning milestones. These responsibilities
are stable:

```text
engine.initial_state() -> BoardState
engine.legal_actions(state) -> 209-bit mask
engine.apply(state, action) -> BoardState
engine.goal_winner(state) -> player | None
episode.advance(state, history, action) -> state, history, outcome
encoder.encode(state, history) -> tensor, reversible action transform
network(tensor) -> logits[batch,209], values[batch]
search.run(state, history, evaluator, limits) -> action, visits, diagnostics
selfplay.play(...) -> complete episode | truncated episode | failure
arena.compare(candidate, champion, frozen_protocol) -> eligible result | incomplete
```

State must remain immutable or provide proven copy/isolation semantics so search
cannot corrupt another actor's history. M1 tests illegal-action rejection and
unchanged input. Transposition and evaluation caches distinguish the information
their results depend on: exact episode adjudication cannot use a board-only key.
Encoder/network cache compatibility includes the feature version and model hash.

## Configuration and Interfaces

Root `configs/smoke.yaml`, `configs/arena.yaml`, and `configs/experiment.yaml`
are new entry configurations. Their shared schema records version, mode, seed,
device, rules, finite time budget, maximum game plies, output root, and search
parameters. The arena adds a frozen protocol identifier, opening-suite hash,
paired-match count, confidence procedure, and promotion thresholds.

`configs/arena.sha256` binds the exact YAML bytes. Arena validation also verifies
the opening-suite bytes against their declared SHA-256. A changed protocol is a
new reviewed version with a new digest, not an override of the existing version.
CLI checkpoint inputs are future runtime inputs; their hashes enter each run's
metadata and cannot alter the frozen search or scoring configuration.

The planned M5 invocation is `make arena CANDIDATE=path/to/candidate.pt
CHAMPION=path/to/champion.pt CONFIG=configs/arena.yaml`, forwarded to
`quoridor-project arena --candidate ... --champion ... --config ...`.
M0 accepts only the config selection and reports unimplemented; it does not
load either model. M4 smoke uses `make smoke CONFIG=configs/smoke.yaml`;
M6 uses `make experiment CONFIG=path/to/experiment.yaml`. Their future YAML
schemas add explicit self-play, optimizer, replay, and checkpoint workloads.

| Entry | M0 behavior | Owning implementation milestone |
| --- | --- | --- |
| `make install` | Install from `training/uv.lock`, including development tools | M0 |
| `make check` | Validate M0 infrastructure and selected CPU reference regressions | M0 |
| `make smoke` | Explicit unimplemented error, nonzero exit | M4 |
| `make arena` | Explicit unimplemented error, nonzero exit | M5 |
| `make experiment` | Explicit unimplemented error, nonzero exit | M6 |
| `python -m quoridor.project_cli check-config --config ...` | Validate only, do not execute a run | M0 |

The `quoridor-project` CLI is the new workstream entry; `quoridor-az` and old
phase scripts remain legacy. Future smoke performs a small bounded self-play,
training, checkpoint, and reload cycle. Future arena takes explicit candidate and
champion identities and uses its frozen protocol. Future experiment takes an
explicit YAML plan and records its total budget and lineage. Runtime overrides
that invalidate the frozen arena are rejected. Current placeholder success is
not permitted, including success after only parsing configuration.

## Runs, Deadlines, and Logs

All future smoke/training/arena/experiment runs allocate unique directories
under the configured `run_root` (default `runs/`). Allocation never overwrites an
existing run. M0 infrastructure writes a validated effective `config.yaml`,
`run.json` provenance, `events.jsonl`, and a terminal `summary.json`; later
milestones add replay, checkpoints, per-game records, and aggregate game/training
metrics.

Required provenance includes a run identifier and mode, UTC start time, budget,
seed, code revision and dirty status, rules/codec/value versions, configuration
hash, dependency lock hash, and device. Later model-consuming runs also record
input checkpoint hashes and encoder/model architecture versions. Event records
are independent JSON objects with UTC timestamp, event type, and structured
fields; nonfinite JSON numbers are forbidden. Report explicit start, finish,
interruption, truncation, and failure states. Artifacts have stable hashes and
cannot be silently reused when their configuration or source identity differs.

Use a monotonic clock for elapsed runtime and a UTC clock for provenance.
Each real execution establishes its deadline before actors, search, or training
begin and passes remaining time to those components. Check before new games,
simulations, batches, and checkpoint work. Bound long operations and preserve
partial-run diagnostics on interruption. Record actual elapsed time and any
overrun; exhausting a run budget stops work and never fabricates game labels.
Checkpoint data can be resumed only after compatibility and lineage checks.
M0 tests deadline calculation and run artifacts; scheduling and actor/optimizer
enforcement are future M3/M4/M6 work.

## Milestone Order

M0 establishes contracts, lockfile, configuration, run helpers, and checks. M1
migrates engine/episode outcomes and establishes property/differential tests. M2
separates encoder/model and resolves history observability with versioned inputs.
M3 validates PUCT and deadline behavior with deterministic evaluators. M4 creates
fresh-version self-play/replay/training and the real smoke loop. M5 enables frozen
paired arena and promotion decisions. M6 enables bounded, reproducible experiments
and measured strength improvement. Existing code is reused only behind each
milestone's tests and explicit compatibility boundary.

No accelerator run, fresh self-play, replay migration, architecture tuning,
promotion, or strength experiment is performed as part of M0 or M1.

## M1 Implementation Boundary

`training/quoridor/engine.py` is the new Python reference. `BoardState` contains
only pawn positions, immutable wall sets, inventories, and `to_play`.
Construction validates coordinates, inventory conservation, wall structure,
both goal paths, and goal/side consistency. `initial_state`, `legal_actions`,
`legal_action_mask`, `apply_action`, and `goal_winner` are public entry points;
`position_key` is a stable versioned string containing all board fields.
Action encoding retains the 0/81/145 offsets and 209 total IDs.

`training/quoridor/episode.py` owns ordered full-position history, outcomes,
ply counts and runtime-stop handling. The board can continue through a repeated
position; the episode adjudicates the third occurrence as a draw. Outcome labels
are available only for completed goal or repetition results. `start_episode`
counts the starting position once; `advance_episode` appends committed actions.
`Episode` construction validates every canonical board key, each legal adjacent
transition, and the absence of earlier goal/repetition/ply-limit endings.
`adjudication_key` includes the full history and ply cap, preventing board-only
cache aliasing. `value_targets(recorded_players)` returns all labels together
or `None` for an unfinished, truncated, or failed episode.

`enforce_deadline`, `interrupt_episode`, and `fail_episode` give callers explicit
stop boundaries. If a goal finishes during an operation that overruns its budget,
its episode result remains a goal while `RunSession` records observed elapsed
time and `budget_exhausted` run status. A tested synthetic integration exercises
this distinction; real actor/search/trainer scheduling remains M3/M4/M6 work.

New imports must use these modules explicitly. Existing `game`, `codec`, root
package exports, classic/browser adapters, and native bindings remain legacy.
This separation avoids relabeling historical data or letting old native code
silently adjudicate new games. `rules_contract.load_engine` exposes only Python
and rejects every unsupported backend before import. The native conformance
condition in `tasks.yaml` remains mandatory before enabling native support.
`require_compatible_artifact` checks rules/codec/value identity only; actual
checkpoint and replay loading remains M2/M4 work.

`make check-m1` runs new rule/outcome/property/boundary tests; `make check` runs
them alongside M0 and retained legacy CPU regressions. Existing conformance
fixtures and frozen arena/opening bytes are unchanged.
