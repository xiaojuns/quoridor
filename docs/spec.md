# Two-player Quoridor AlphaZero Contract

Status: contract established in M0, 2026-10-10. This document specifies the new M0-M6
workstream. It does not certify that the existing browser, Python, or C++
implementations already satisfy these requirements. M0 delivers engineering
entry points and contracts; M1 implements separate Python board/episode APIs.

## Scope and Versions

The player uses a policy/value network, PUCT MCTS, and self-play training.
Implementation uses Python, PyTorch, pytest, Hypothesis, Git, locked dependencies,
YAML configuration, and JSONL event logs. Initial engineering support is CPU on
Linux x86_64; accelerator training requires later validation and an explicit run
configuration.

Every new replay, checkpoint, and run must identify `rules_version: quoridor-v1`,
`codec_version: az209-v1`, and `value_perspective: current-player-v1`.
These names belong to this workstream, not the legacy numeric codec version.
Action numbering can be reused; changed result semantics make old data
incompatible without explicit migration and provenance.

## Board and Actions

The board is 9 by 9. Both players start with 10 walls. Absolute coordinates are
zero-based `(row, col)` from the upper left. Player 0 starts at `(8, 4)` and wins
by reaching row 0; player 1 starts at `(0, 4)` and wins by reaching row 8. Player 0
moves first unless an opening explicitly specifies the side to move.

Each turn moves the pawn or places one wall. Pawn moves include cardinal steps,
straight jumps over an adjacent opponent, and the permitted diagonal alternatives
when the straight jump is blocked by a wall or board boundary. Pawns cannot
occupy the same square. Walls cannot overlap or cross and cannot remove either
player's path to its goal row. Path existence ignores pawn occupancy; movement
legality does not. Wall anchors occupy an 8 by 8 lattice.

The complete action space has exactly 209 entries:

| Indices | Action | Absolute encoding |
| --- | --- | --- |
| 0-80 | Pawn destination | `9 * row + col` |
| 81-144 | Horizontal wall anchor | `81 + 8 * row + col` |
| 145-208 | Vertical wall anchor | `145 + 8 * row + col` |

The codec maps all 209 actions reversibly, including currently illegal actions.
The engine supplies the legality mask separately. No heuristic wall pruning is
allowed in the training policy or baseline PUCT action set. Illegal actions are
rejected without mutation; they are implementation errors, not game losses.

## State and Ownership

The engine owns immutable board state, legal actions, transitions, and goal
detection. It does not import PyTorch, neural inference, replay/training code, or
wall-clock services. The episode layer owns repetition history, adjudication,
ply counts, and runtime limits.

A full-position key contains both absolute pawn positions, both wall inventories,
all horizontal and vertical wall anchors in stable order, and the side to move.
It excludes ply count, wall time, result metadata, and occurrence counters. The
episode starts with the initial position counted once and counts every committed
post-action position. A position with a different side to move is a different
position.

Search receives both board state and episode history. An identical board with
different history must not share history-sensitive terminal decisions through a
board-only cache or transposition key.

## Outcomes: Goal, Repetition, and Truncation

These are distinct events with distinct data semantics:

| Event | Episode status | Winner | Value target | Replay eligibility |
| --- | --- | --- | --- | --- |
| Pawn reaches its goal | `terminated`, reason `goal` | Player 0 or 1 | +1/-1 from each recorded player's view | Eligible |
| Third occurrence of a full position | `terminated`, reason `repetition` | None, draw | 0 | Eligible |
| Unfinished game reaches `max_game_plies` | `truncated`, reason `ply_limit` | None | Missing | Entire episode excluded |
| Run deadline expires or execution is interrupted | `truncated`, reason `time_budget` or `interrupted` | None | Missing | Entire episode excluded |
| Illegal action or numerical/infrastructure failure | `failed` with explicit reason | None | Missing | Entire episode excluded |

Threefold repetition is a project adjudication extension, not an assertion about
official Quoridor tournament rules. The threshold is 3. The baseline per-game
runtime cap is 256 plies, where one ply is one player's committed action. The cap
is operational and does not define who won or whether the position is a draw.

After a committed legal action, inspect goal first, then repetition, then the ply
cap. Thus a genuine goal or repetition result reached on ply 256 completes the
game. Check the run deadline before beginning further work; an unfinished game
stopped at that boundary is truncated. If an operation finishes after its deadline,
record the observed duration and classify its unfinished episode as truncated;
never start another operation to conceal the overrun. M4/M6 must bound such
operations and propagate the deadline into actors and search.

`terminated` and `truncated` cannot both be true. `winner: null` alone does not
distinguish a draw from missing labels; consumers must inspect status and reason.
Arena truncation makes a match incomplete and the comparison ineligible for
promotion, even if some games finished. A search-local repeated path may return
0 to bound a simulation; that is a search approximation, not a completed draw
and never a source of replay outcome labels.

The legacy implementation instead awards a loss to the acting player at its
third repetition or the 256th ply. Existing +/-1 replay labels and checkpoint
value heads must not be silently relabeled, resumed, or declared compatible.
M1 defines a tested migration boundary; M4 starts new-version replay unless a
separately validated migration records its source hashes and conversion rules.

## Perspective and Network Contract

The model predicts value from the current player to move: +1 for that player's
eventual win, -1 for its loss, and 0 for a completed repetition draw. After each
legal action, side to move changes, including terminal positions. A goal leaf
therefore has value -1 for the next player. Each stored training sample remembers
its own `to_play` so final labels never depend on index parity alone.

The encoder is outside the network. It canonicalizes the current player's goal
to the upper edge; when needed, rotate coordinates by 180 degrees and swap
player roles. Square coordinates map to `(8-row, 8-col)`; wall anchors map to
`(7-row, 7-col)`. Horizontal and vertical orientations do not change. Policy
targets and selected actions use the same reversible mapping. The canonical
current-player value keeps its sign.

The network accepts a versioned floating tensor `[batch, channels, 9, 9]` and
returns unmasked policy logits `[batch, 209]` and bounded values `[batch]`.
Feature extraction, rules, and legal masking do not live inside the model.
Only left/right mirroring and 180-degree rotation with player swap are supported
symmetries. All state, history, policy, action, and player identities must transform
consistently. A current-player value is invariant under these player-preserving
perspective mappings; any absolute-player value representation must explicitly
account for the player swap. A 90-degree rotation is not a rule symmetry.

Input feature layout is an M2 decision gate. Board planes and the current
repetition count alone do not expose the full history needed for threefold
adjudication. M2 must either specify a sufficient history representation or
document the network's partial observation and validate its consequences. It
must not claim legacy 14/16 planes make the repetition game Markov. The engine
and search must still have the exact episode history.

## Search and Learning Contract

Each edge stores `P`, `N`, `W`, and `Q = W/N`, with `Q` expressed from the parent
player's viewpoint. PUCT selects the largest
`Q + c_puct * P * sqrt(max(1, sum(N))) / (1 + N)`. The floor ensures priors
guide the first selection when the node has no visits. Break equal scores
deterministically and test this case explicitly. Negate the leaf value once per
edge during backup. Terminal leaves use the exact episode outcome without inference.
M3 tests goal, draw, odd/even-depth backup, complete legal expansion, and
history-sensitive cycles using deterministic evaluators before model strength
claims.

Mask illegal logits and normalize only over legal actions. Numerical failures are
reported explicitly; a fallback must not silently make an arena result eligible.
Root noise and training temperature are configurable for self-play. Arena uses
neither root noise nor stochastic temperature. Tie-breaking and random seeds
are recorded. Search always returns legal actions and 209-entry visit data.

Self-play policy targets are normalized root visit counts, not network priors or
heuristic selections. At temperature zero, action selection takes the maximal
visit count with a deterministic tie-break; target construction remains explicit
and versioned. Baseline learning minimizes policy cross-entropy plus value MSE
and declared weight decay. No shaped rewards, truncated bootstrapped labels,
teacher targets, or auxiliary losses enter the baseline without a versioned
experiment. Preserve seeds, full effective configuration, replay lineage,
checkpoint hashes, code revision/dirty status, time budget, and outcome counts.

## M0 Executable Boundary

M0 provides dependency installation, `make check`, YAML validation, frozen arena
configuration verification, and tested run-directory/JSONL/deadline helpers.
These helpers are infrastructure; they do not yet enforce a training scheduler.
`make smoke`, `make arena`, and `make experiment` are reserved interfaces and
must fail explicitly as unimplemented with nonzero status. Configuration
validation does not imply that training or evaluation ran. M1-M6 acceptance is
tracked in `tasks.yaml`. The M0 round stopped at infrastructure; the authorized
M1 follow-up implements rules only and leaves these execution entries unavailable.

## M1 Compatibility Boundary

New rule code imports `quoridor.engine` and `quoridor.episode` explicitly.
`quoridor.game`, `quoridor.codec`, and root-package exports remain legacy APIs;
the old browser and C++ backend keep their historical outcome semantics.
`rules_contract.load_engine()` exposes only the new Python engine. Requests for
native, browser, automatic selection, or a different rules version fail before
backend import; there is no fallback to a legacy adjudicator. No native backend
is enabled for `quoridor-v1`. Enabling one later still requires the unchanged
M1 acceptance condition: passing new-version differential tests without skips.

`require_compatible_artifact()` rejects missing or mismatched rule, codec, and
value-perspective metadata. A legacy numeric codec version and matching action
numbers alone cannot establish compatibility. This is an identity gate, not a
checkpoint/replay loader: M2/M4 must additionally verify encoder, architecture,
content hashes, lineage, and complete episode history before loading data.
