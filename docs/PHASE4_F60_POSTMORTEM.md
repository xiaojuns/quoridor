# Phase 4 F60 Training Postmortem

Date: 2026-09-17

This report updates `PHASE4_ROOT_CAUSE_ABLATION.md` with the evidence from
F40-F60. It is a no-training audit. No replay, checkpoint, or GPU-ledger state
was changed while producing it.

## Executive verdict

The project does not have one hidden CUDA, sign, legality-mask, or search
budget bug that explains the failed training. The main failure is that the
optimization and acceptance criteria ceased to measure the behavior that must
improve: playing strength under the deployed 800-simulation MCTS.

F58 is the clearest causal example. It changed only the policy head, preserved
the value output exactly, improved teacher cross-entropy by 5.00%, and passed
the configured mean-KL gate. The resulting model then lost 27-37 to its F21
parent in the F60 paired arena. This proves that the selected teacher objective
can improve while actual search strength becomes worse.

There is a second, independent model problem. The earlier frozen F21 ablation
showed that early-game value estimates are frequently confident and wrong.
At 800 simulations, replacing the learned value with zero improved early
classic-action agreement from 25% to 45%. Thus the baseline has a harmful
policy/value interaction, while the attempted policy-only correction also
moved in the wrong direction.

The practical conclusion is to stop the current classic-rank distillation
lineage. More steps, a larger artifact cap, or another soft-loss gate will not
repair the objective mismatch.

## Evidence

### 1. F21 has an early-game value calibration problem

The frozen root-cause ablation used 60 held-out positions:

| Diagnostic | Result |
| --- | ---: |
| Early policy top-1 agreement with classic | 45% |
| Early value outcome-sign accuracy | 40% |
| Early value MSE to final outcome | 2.023 |
| F21 policy + F21 value, early agreement at 800 simulations | 25% |
| F21 policy + zero value, early agreement at 800 simulations | 45% |
| Search agreement, 64 to 2048 simulations | 68.3% to 65.0% |

Increasing simulations did not repair the bias. The search became more
confident without becoming more consistent with the reference. This rules out
"just search more" as a solution for the current network.

Source: `docs/PHASE4_ROOT_CAUSE_ABLATION.md` and
`training/runs/phase4/root-cause-ablation-v1/report.json`.

### 2. The corrected teacher search itself was stable

F52 compared 128 positions at its primary and reference classic-search
budgets. The original gate checked selected-action agreement and the Top-12
set. This audit additionally reconstructed the exact rank targets used for
training and compared their full distributions.

| Reconstructed F52 metric | All | Early | Middle | Late |
| --- | ---: | ---: | ---: | ---: |
| Positions | 128 | 64 | 48 | 16 |
| Target JS divergence | 0 | 0 | 0 | 0 |
| Target total variation | 0 | 0 | 0 | 0 |
| Ordered Top-12 exact match | 100% | 100% | 100% | 100% |

Therefore teacher-budget instability is not the explanation for F58/F60.
The target construction can still be semantically wrong for learning, but it
was deterministic and budget-stable on the audited positions.

Sources:

- `training/runs/phase4/phase4f/f52-teacher-budget-correction-v1/classic-root/manifest.json`
- `training/quoridor/phase4f51_learner_teacher_probe.py`

### 3. F58 optimized the proxy but regressed in the arena

F58 froze the trunk and value head, trained only the policy head, and selected
step 32.

| F58 metric | Result |
| --- | ---: |
| Accepted optimizer updates | 28 |
| Teacher development cross-entropy improvement | 5.00% |
| Early cross-entropy improvement | 5.45% |
| Anchor mean forward KL | 0.0180 |
| Value-output maximum absolute change | 0 |
| F60 result against F21 | 27-37 |
| F60 win rate | 42.19% |
| F60 Elo estimate | -54.7 |

The F60 pair results were 21 split pairs, eight 0-2 losses, and three 2-0
wins. The deficit was five wins on each player side, so the result is not
explained by first-player bias.

Sources:

- `training/runs/phase4/phase4f/f58-kl-trust-region-policy-head-pilot-v1/pilot-report.json`
- `training/runs/phase4/phase4f/f60-kl-lineage-screen-v1/screen-gate.json`
- `training/runs/phase4/phase4f/f60-kl-lineage-screen-v1/screen-arena/result.json`

### 4. The KL gate was an average, not a local trust region

The implementation named the threshold a hard maximum, but
`full_anchor_forward_kl` averages KL over 512 anchor positions and their
mirrors. It does not cap per-position drift and it says nothing about new
states reached after the policy changes.

An offline replay of all 2,910 F60 arena positions found:

| F21 to F58 policy drift | All | Early | Middle | Late |
| --- | ---: | ---: | ---: | ---: |
| Mean forward KL | 0.0144 | 0.0214 | 0.0208 | 0.0074 |
| 95th percentile KL | 0.0414 | 0.0505 | 0.0464 | 0.0282 |
| Positions above 0.02 | 31.1% | 43.8% | 48.9% | 12.5% |
| Raw policy top-1 agreement | 94.0% | 93.1% | 90.6% | 97.4% |

On the 64 fresh initial arena positions, mean KL was already 0.0205, 53.1%
of positions exceeded 0.02, and the maximum was 0.0476. Across the full arena
the maximum was 0.0893.

F58 also increased policy entropy by 0.123 nats on average and reduced the
raw probability of the action actually selected by MCTS by 0.023. Small raw
top-1 changes are enough to alter visit allocation in finite PUCT search.

### 5. Teacher loss did not measure decision improvement

On the 512-position F54 development set:

| Metric | F21 | F58 |
| --- | ---: | ---: |
| Teacher cross-entropy | 3.8983 | 3.7032 |
| Teacher top-1 agreement | 33.40% | 33.79% |
| Early teacher top-1 agreement | 26.95% | 29.30% |
| Middle teacher top-1 agreement | 32.81% | 31.77% |
| Late teacher top-1 agreement | 60.94% | 57.81% |

The 5% cross-entropy gain produced only a 0.39 percentage-point overall
top-1 gain and made middle and late top-1 agreement worse. The rank target
rewards probability mass over a stable Top-12 list, but it does not encode the
score margin or demonstrate that matching that distribution improves the
learner's finite-budget MCTS.

## Root causes

### Proven primary cause: objective and promotion misalignment

Model selection used soft teacher cross-entropy and mean anchor KL as primary
evidence. Neither metric is monotonic with arena strength. F58/F60 directly
demonstrates this failure. This is the main reason the process could repeatedly
approve changes that did not improve the product objective.

### Proven model defect: early value overconfidence

The F21 component ablation isolates a harmful early-game interaction between
the learned value and policy. More simulations do not remove it. This must be
tested with an actual-strength intervention before further value training.

### Proven protection defect: distribution-local mean KL

The F58 gate controlled an average on one fixed occupancy sample. It allowed
large tail drift and did not protect states induced by the modified policy.
Calling it a hard trust region overstated what it guaranteed.

### Strong contributing factor: fragmented, evaluation-heavy learning

The 80 GPU-hour ledger is not 80 hours of continuous model optimization.
Many stages generated teacher searches, ran arenas, rebuilt gates, or restarted
short lineages. F58 itself stopped after only 28 accepted updates. AlphaZero's
closed loop depends on sustained self-play, visit targets generated by the
same search stack, replay diversity, and repeated optimization. The current
workflow substituted many small proxy-gated experiments for that loop.

### Structural difficulty: Quoridor is not a small search problem

The board is small, but wall placement creates a broad legal-action set,
long-horizon path constraints, and tactical interactions that are not local.
The early ablation averaged 68.5 legal moves, while 2048 simulations visited
only 40.4 actions. A smaller board does not imply a materially easier game tree.

## Falsified explanations

- **Global value-sign inversion:** false. The learned value is useful on
  average and improves uniform-policy search, although it is miscalibrated in
  important early states.
- **Incomplete legal-action labels:** false. All 512 F54 development samples
  contain the full legal-action inventory.
- **F52 search-budget instability:** false on the audited 128 positions. The
  exact training distributions are identical between primary and reference
  budgets.
- **Value-head damage caused F60:** false. F58's value outputs are bitwise
  unchanged at the evaluated positions; the regression came from policy/search
  interaction.
- **First-player imbalance caused F60:** false. The candidate lost the same
  five-win margin on both player sides.

## What remains unknown

- Whether reducing or recalibrating F21 value influence improves actual paired
  game strength, rather than classic-action agreement alone.
- Whether a much smaller policy update could improve strength before the
  harmful drift appears. F58 did not retain every intermediate checkpoint and
  selected only by teacher loss.
- Whether classic-hard is a useful training teacher at all. Stable rankings do
  not prove that its soft rank distribution is a better prior for neural MCTS.

## Required next experiment

Do not start another training phase first. Run one inference-only causal test:

1. Add a frozen evaluator option that multiplies the F21 value output by a
   constant only inside MCTS. It must not change policy logits or model weights.
2. Compare standard F21 (`value_scale=1.0`) with
   `value_scale=0.5` on 32 fresh paired openings at the deployed
   800-simulation runtime.
3. Use paired-arena win rate as the primary result. Also record early root
   action changes and value magnitude, but do not promote on those proxies.
4. Stop if the scaled-value candidate does not score above 50%. Do not train a
   new value target in that case.
5. If it wins, repeat once on a fresh 200-pair confirmation suite before
   designing value-only calibration training.

This experiment tests the strongest remaining causal hypothesis using the
existing model and no optimizer step. It changes one variable, has a clear
failure condition, and prevents another long sequence from being justified by
surrogate losses.

## Process constraints for any later training

- Arena strength is the primary acceptance criterion; losses and teacher
  agreement are diagnostics only.
- Keep a single frozen baseline and retain every evaluated checkpoint.
- Require per-position KL tail metrics (`p95`, `p99`, and maximum), not only a
  mean, on both anchor and fresh occupancy states.
- Do not authorize a large run from a 64-game positive screen. Use the screen
  only to reject obvious regressions; confirmation must be separately powered.
- Set a stop-loss before execution: one hypothesis, one intervention, one
  confirmation, then stop or redesign.

## F61 causal-screen result

F61 executed the prescribed inference-only intervention on 2026-09-17. Both
participants used the identical F21 model artifact, identical policy logits,
the same native backend, and 800 simulations. The candidate multiplied only
the network value returned to MCTS by `0.5`; the baseline used `1.0`.

| Result | Value |
| --- | ---: |
| Opening pairs / games | 32 / 64 |
| Scaled-value wins / losses | 45 / 19 |
| Scaled-value win rate | 70.3125% |
| 90% paired-bootstrap win-rate CI | 64.0625% to 78.1250% |
| Elo estimate | +149.8 |
| 90% Elo CI | +100.4 to +221.1 |

The result passed the precommitted `winRate > 0.50` screen threshold. Because
the model weights and policy were identical, this is direct game-strength
evidence that the original F21 value magnitude harms its finite-budget MCTS.
It does not yet identify `0.5` as the optimal scale and does not authorize
value training. The next allowed action is the precommitted 200-pair fresh
confirmation of `0.5` versus `1.0`; it still requires separate confirmation.

Sources:

- `training/runs/phase4/phase4f/f61-f21-value-scale-screen-v1/execution-contract.json`
- `training/runs/phase4/phase4f/f61-f21-value-scale-screen-v1/screen-gate.json`
- `training/runs/phase4/phase4f/f61-f21-value-scale-screen-v1/screen-arena/result.json`
