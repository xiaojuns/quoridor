# M2 Encoder and Model Contract

Status: implemented M2 interfaces; engineering validation is recorded in
`verification-m2.md`. No training or playing-strength claim is made.

## Versions and Ownership

The new path uses `engine` / `episode` -> `encoder` -> `policy_value`, with
`policy` applying the external legal mask. `symmetry` transforms new-rule
boards, full histories, actions and policies without importing tensor libraries.
`policy_value` imports only PyTorch and pure metadata; it defines no game rules.

`feature_spec.py` declares `az-encoder-v1-partial-16` and `az-resnet-v1`.
Rule, codec and value identities remain `quoridor-v1`, `az209-v1` and
`current-player-v1`. Legacy `network.py` and checkpoints remain separate, even
when they also use 16 planes. Root legacy exports are lazy so importing the new
tensor model does not import old rules. Existing root API names still resolve
to their original legacy implementations.

## Input Layout

`encode_episode(episode, mirror=False)` returns `EncodedPosition` with an
independent CPU float32 tensor `[16,9,9]`, a CPU bool legal mask `[209]`, and a
reversible original-to-input transform. The current player becomes player 0,
whose goal is the upper row. All features lie in `[0,1]`.

| Plane | Meaning | Encoding |
| --- | --- | --- |
| 0 | Current player's pawn | One occupied square |
| 1 | Opponent's pawn | One occupied square |
| 2 | Horizontal walls | Anchors in the upper-left 8×8 subgrid |
| 3 | Vertical walls | Anchors in the upper-left 8×8 subgrid |
| 4 | Current player's wall inventory | Constant, inventory / 10 |
| 5 | Opponent's wall inventory | Constant, inventory / 10 |
| 6 | Legal pawn destinations | Engine mask entries 0–80 reshaped 9×9 |
| 7 | Legal horizontal wall anchors | Engine mask entries 81–144 in 8×8 |
| 8 | Legal vertical wall anchors | Engine mask entries 145–208 in 8×8 |
| 9 | Current player's goal row | Ones on row 0 |
| 10 | Opponent's goal row | Ones on row 8 |
| 11 | Operational ply progress | Constant, plies / episode.max_game_plies |
| 12 | Current full-position occurrence count | Constant, count / 3 |
| 13 | Current pawn's wall-only distance to goal | Constant, engine distance / 80 |
| 14 | Opponent pawn's wall-only distance to goal | Constant, engine distance / 80 |
| 15 | Bias plane | Ones |

The last row and column of wall-anchor planes are zero. Distances and legality
come from public engine APIs; the encoder has no BFS or independent wall/move
implementation. Finished goal/draw/truncation/failure episodes have zero policy
support (planes 6–8 and legal mask). Other descriptive features remain available;
this does not invent a label for a runtime stop.

## History Decision and Its Limits

This version deliberately uses partial observation. It exposes the current
position's occurrence count and operational progress, but not all historical
position counts. Its fixed-size input and compact convolutional baseline permit
engineering the new interfaces without choosing a sequence or set architecture
before a reproducible training baseline exists. This is a representation choice,
not evidence that history can be discarded without a strength cost.

The deterministic collision test constructs a left four-ply cycle and a right
four-ply cycle. Both return to the same board, at ply 4 with current count 2;
their encoded tensors are identical. A common five-ply continuation reaches a
third intermediate left position in only one history: one episode draws while
the other remains live. This verifies a concrete consequence of partial
observation. A network receiving only these tensors cannot distinguish the
two hidden histories at the original observation.

Exact terminal checks, labels and search adjudication must therefore keep
`Episode.history` and use its `adjudication_key`. Features are not a Markov
state or a key for outcome/PUCT-tree caching. A raw tensor-forward cache may
share identical inputs for the same model; that does not permit sharing
history-sensitive terminal results. M3 checks exact outcomes before inference.
Actual value-fit or strength costs of this choice remain unmeasured and belong
to a preregistered, budgeted later experiment. A fuller representation requires
a new encoder version and fresh compatibility checks.

## Reversible Transforms and Values

`Symmetry(mirror=..., rotate_swap=...)` supports four combinations. Rotation
maps square coordinates to `(8-r,8-c)`, wall anchors to `(7-r,7-c)`, and swaps
player IDs and inventories. Horizontal reflection maps square columns with 8
and wall-anchor columns with 7. Orientations remain horizontal/vertical.
All combinations are involutions and commute.

The same global transform applies to every full-position key in a history;
historical positions are never individually canonicalized by their own player.
`transform.policy(values)` permutes the 209 entries so
`output[transform.action(i)] == input[i]`. The inverse maps model coordinates
back to actual game coordinates. A current-player value keeps its sign under
these transforms; goal winners' identities swap as needed, goal leaf value
remains -1 for the next player, draws remain 0, and truncations remain unlabelled.

These are label and coordinate invariants. An untrained convolutional network
is not claimed to produce identical scores under horizontal reflection.

## Tensor Network, Policy and Diagnostics

`PolicyValueNetwork(NetworkConfig(...))` accepts dense floating tensors
`[batch,16,9,9]` and returns raw logits `[batch,209]` and tanh values `[batch]`.
It validates shape, dtype/device, and finite inputs/outputs. It consumes no game
objects and applies no legal mask. Evaluation callers use `.eval()` with
inference/no-grad mode; training callers own optimization.

`configs/model.yaml` records the complete architecture: residual width 64,
4 blocks, 8 policy/value-head channels, and a 64-unit value hidden layer.
`model_config.load_model_config` validates exact YAML keys, encoder version,
architecture types and shape. This is an untuned engineering baseline. The
8-channel / 1-block `NetworkConfig.smoke()` is only a unit-test workload.

`masked_policy(logits, legal_mask)` requires a bool mask of identical shape,
preserves legal support, normalizes active rows, and returns zeros for all-false
rows. Nonfinite logits, including illegal entries, fail explicitly. This mask
operation lives outside the model and never supplies replacement priors after
a numerical error.

`value_diagnostics(predictions, targets)` accepts completed-game targets only
(-1, 0, +1) and returns JSON-safe MSE, MAE, bias, decisive sign accuracy,
draw absolute error, and fixed equal-width value-bin means. Calibration MAE is
the sample-weighted absolute difference between mean prediction and mean target
inside each bin; empty-bin means and absent subsets are `null`. Bins cover
`[-1,1]`, include the lower endpoint and exclude the upper except in the last bin.
Truncated targets must never be filled with zero to enter this function. Metrics
are diagnostic, not a strength test; current validation uses synthetic fixtures.

## Model Artifact Boundary

`save_model_checkpoint(path, model)` writes a new artifact without overwriting
an existing path and returns its path and SHA-256. The parent directory must
already exist. `load_model_checkpoint(path, expected_sha256=...,
expected_config=...)` returns a new CPU/eval model, artifact identity, and copied
metadata. Files use the `quoridor-model-weights` schema with complete encoder
layout/normalization, rules/codec/value identities, architecture and state dict.

Loading hashes the same bytes it deserializes using `weights_only=True`.
Metadata, every parameter/buffer key, shape, dtype, and finite values are
validated before CPU model allocation. Negative BatchNorm variances/counters
are rejected. M2 artifacts are float32; conversion is not implicit. Loading
preserves the caller's RNG and never partly mutates a caller-owned model.
Optional expected hashes bind a previously recorded identity; merely computing
a new hash does not authenticate provenance.

This API saves model weights only. Optimizer/replay/RNG continuation, run lineage,
budgeted training schedules and resuming a training job remain M4. Legacy
14/16-plane metadata, missing rules, changed plane semantics, or different value
perspectives are rejected rather than inferred compatible from tensor shape.
