# Phase 4 Root-Cause Ablation Report

Date: 2026-08-24  
Final raw report: `training/runs/phase4/root-cause-ablation-v1/report.json`  
Raw report core SHA-256: `b96a4e21af161b63dadd8ec2bb8901b33b9805eddefb1bd153f1b0ca0e1a3eca`

## Executive conclusion

The main failure is not a global MCTS sign error and is not solved by adding
more simulations to the current network. The frozen F21 model has learned a
useful policy and a partially useful value function, but both are weakest in
the high-branching early game. The early-game value is frequently confident
and wrong. PUCT therefore reinforces a biased policy/value combination instead
of correcting it as the simulation budget grows.

The system is also severely undertrained relative to its 12.0 million
parameters. Repeated short lineages, low self-play throughput, near-one-hot
teacher targets, and small noisy promotion screens turned most of the 77 GPU
hours into search and evaluation rather than weight updates.

## Method

- Frozen model: F21 Stage A generation 1, 192 optimizer steps.
- Architecture: 192 channels, 16 residual blocks, 12,026,482 parameters.
- Positions: 60 deterministic holdout positions from 35 games.
- Balance: 20 positions each for early (`ply < 16`), middle (`16-39`), and
  late (`>= 40`), balanced by side to move.
- Reference: deterministic classic-hard Negamax, maximum depth 5 and 587-node
  budget.
- Full-model search scaling: 64, 256, 800, and 2048 simulations.
- Component ablation at 800 simulations:
  - learned policy + learned value
  - learned policy + zero value
  - uniform policy + learned value
  - uniform policy + zero value
- Root noise was disabled and temperature was zero.
- No training, replay generation, checkpoint write, or GPU-ledger update was
  performed.

## Direct network quality

| Metric | Overall | Early | Middle | Late |
| --- | ---: | ---: | ---: | ---: |
| Policy top-1 agreement with classic | 70.0% | 45.0% | 85.0% | 80.0% |
| Policy top-5 coverage | 91.7% | 80.0% | 95.0% | 100.0% |
| Mean probability on classic action | 63.6% | 37.8% | 75.8% | 77.1% |
| Value outcome-sign accuracy | 68.3% | 40.0% | 85.0% | 80.0% |
| Value MSE to final outcome | 1.175 | 2.023 | 0.657 | 0.846 |

The value head emitted `abs(value) > 0.9` on 36 of 60 positions. Eleven of
those high-confidence predictions had the wrong outcome sign. Five of the
eleven occurred in the 20-position early band.

This rules out a simple global sign inversion: learned value improves an
otherwise uniform search from 25.0% to 50.0% classic agreement. It is useful on
average, but badly calibrated and actively misleading in important early
positions.

## Search scaling

| Simulations | Classic agreement | Early | Middle | Late | Mean visited actions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 68.3% | 35.0% | 85.0% | 85.0% | 11.0 |
| 256 | 66.7% | 25.0% | 90.0% | 85.0% | 16.1 |
| 800 | 65.0% | 25.0% | 85.0% | 85.0% | 18.3 |
| 2048 | 65.0% | 25.0% | 85.0% | 85.0% | 19.9 |

Increasing search from 64 to 2048 simulations does not improve agreement with
the reference. At 800 to 2048 simulations, 95% of selected actions are
unchanged. Search has saturated around the network's bias.

Early positions average 68.5 legal actions. Even at 2048 simulations, search
visits only 40.4 of them on average, while the most visited action receives
73.3% of root visits. More simulations make the root more confident without
making its decisions better against the reference.

## Component ablation at 800 simulations

| Policy | Value | Overall | Early | Middle | Late |
| --- | --- | ---: | ---: | ---: | ---: |
| Learned | Learned | 65.0% | 25.0% | 85.0% | 85.0% |
| Learned | Zero | 68.3% | 45.0% | 80.0% | 80.0% |
| Uniform | Learned | 50.0% | 15.0% | 65.0% | 70.0% |
| Uniform | Zero | 25.0% | 10.0% | 25.0% | 40.0% |

The policy head is the larger source of useful domain knowledge. The value
head is also useful in isolation, but its interaction with the learned policy
costs 20 percentage points in early-game agreement. This is consistent with a
miscalibrated value function overriding reasonable policy priors in broad wall
positions.

## Runtime evidence

- Hardware: Quadro RTX 6000, 24 GiB, CUDA 12.6.
- Full model at 2048 simulations: 3,505 simulations/second over 60 concurrent
  positions.
- Peak CUDA allocation during the matrix: about 91 MB.
- Mean native evaluator batch size at 2048 simulations: 35 states.

The diagnostic did not sample GPU utilization, so allocation alone is not an
occupancy measurement. It does show that memory capacity is not the limiting
resource and is consistent with the earlier low-utilization calibration.

## Root causes, ordered by impact

1. **Early-game value failure.** Outcome-only value targets are too sparse and
   noisy for wall-heavy positions. The head is overconfident rather than merely
   uncertain, which makes PUCT amplify errors.
2. **Early-game policy coverage failure.** The policy imitates classic well in
   middle and late positions but only reaches 45% top-1 agreement in the early
   band, where branching is largest.
3. **Search is prior-bound.** Additional simulations concentrate visits but do
   not correct the network. The current 512/800-simulation training and arena
   settings cannot compensate for biased heads.
4. **Insufficient continuous learning.** A 12M-parameter model was trained in
   short, repeatedly reset lineages with only hundreds to low thousands of
   optimizer steps and games per lineage.
5. **Teacher target mismatch.** The teacher policy is effectively a near-one-hot
   classic action, not a calibrated distribution over good alternatives. It
   supplies imitation but little uncertainty or counterfactual value signal.
6. **Evaluation-driven churn.** Repeated 64-game screens and rollbacks consumed
   substantial compute while adding no training signal.

## What this report does not prove

Classic action agreement is a diagnostic proxy, not ground-truth game strength.
The holdout was not used for gradient updates, but it was used by prior model
selection gates, so this is not an untouched final evaluation set. Final model
decisions still require isolated paired arenas after the training system is
changed.

## Required corrective sequence

1. Stop the current F38/F39 teacher variants and preserve the F21 model as the
   diagnostic baseline.
2. Replace the value objective with a calibrated multi-signal target for early
   positions: terminal outcome plus bounded path/tempo targets or reanalyzed
   search value. Track calibration error and confident-wrong rate, not only MSE.
3. Replace near-one-hot teacher policy with visit/rank distributions over
   multiple good actions. Oversample diverse early wall positions.
4. Train a smaller 64-96 channel, 6-10 block model in one persistent lineage;
   retain optimizer and replay instead of restarting every micro-block.
5. Improve search coverage with domain-safe wall candidate proposals or
   progressive widening. Do not simply raise simulations on the present PUCT.
6. Promote only after head ablations improve on a fresh diagnostic suite, then
   use a sufficiently powered paired arena rather than repeated 64-game gates.

The practical single-machine path is a hybrid search: use classic wall
proposals and a bounded classic heuristic as an early search scaffold, train
the network to predict residual policy/value improvements, and anneal the
scaffold only after value calibration and fresh-arena strength both hold.
