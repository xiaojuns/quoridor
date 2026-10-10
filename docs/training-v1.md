# M4 Training Contract

M4 connects the new `quoridor-v1` engine, `puct-v1` search and M2 tensor model
into a bounded CPU self-play and update loop. Legacy rule and training modules
remain available to historical callers; M4 uses the `NewSelfPlay*` and
`Replay*` APIs.

The long-run starting settings are width 64 with four residual blocks, 128
search simulations per move, a 100,000-position oldest-first replay limit,
batch size 128, and 50–100 new games per round. They are starting settings for
future runs, not measured strength results. The M4 smoke uses one tactical game,
eight simulations from `configs/smoke.yaml`, replay capacity 16, batch size 1
and one optimizer step so the locked CPU check remains bounded.

Only a `completed` search with a legal action can commit a self-play move.
Partial or budget-exhausted searches end the actor as an explicit stop; their
visits and root estimate never become a move or label. The actor stores exact
pre-action episodes and raw original-coordinate root visits. Replay admits only
terminated goal/repetition episodes, verifies exact history prefixes, and
atomically excludes truncated/failed games while retaining reason counters.
Terminal labels use each recorded `to_play`, so repetition is 0 and goals are
±1. Encoder canonicalization transforms raw policy/action IDs before training;
illegal target mass is rejected.

`AlphaZeroTrainer(require_legal_masks=True)` computes masked policy
cross-entropy plus value MSE, checks finite losses/gradients/parameters, clips
gradients and advances an optional cosine schedule. `training_checkpoint` is a
separate resume artifact from M2 inference weights: it atomically stores model,
optimizer, scheduler/scaler, Python/NumPy/torch RNG, replay state/hash, counters,
architecture and config/lock/rules/encoder/model/search/code lineage. Loading
validates identity and restores as one fail-closed transaction; mismatches leave
the destination unchanged. Legacy artifacts are rejected.

`make smoke` invokes the real `smoke-run` path. It creates a unique RunSession,
logs effective settings and phases in JSONL, plays one near-goal tactical game,
admits its terminal sample, performs one CPU update, saves/reloads a training
checkpoint and publishes compatible M2 weights. The fixture is recorded as
`tactical-immediate-goal-v1`; this proves the engineering path and says nothing
about strength or convergence. Arena and experiment remain M5/M6 stubs.
