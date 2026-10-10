# M3 Search Contract

`quoridor.puct` implements `puct-v1` over immutable new-rule `Episode` objects.
It does not import a network, encoder, tensor library or legacy engine/search.
`search_evaluator.NeuralEvaluator` connects the M2 encoder and tensor model;
`search_config.search_config_from_project` reads the existing validated YAML
snapshot. The legacy `mcts.py` and its weights/adjudication remain separate.

## Public Interface

```python
from quoridor.episode import start_episode
from quoridor.project_config import load_project_config
from quoridor.project_runtime import Deadline
from quoridor.model_checkpoint import load_model_checkpoint
from quoridor.puct import run_search
from quoridor.search_config import search_config_from_project
from quoridor.search_evaluator import NeuralEvaluator

project = load_project_config("configs/arena.yaml")
search = search_config_from_project(project)
model = load_model_checkpoint("candidate.pt").model  # compatible M2 artifact
deadline = Deadline(project.document["budget"]["wall_time_seconds"])
result = run_search(
    start_episode(max_game_plies=project.document["budget"]["max_game_plies"]),
    NeuralEvaluator(model), search, deadline=deadline,
)
```

This is a library usage example, not an executed arena command. In real runs
the caller passes its already-running `RunSession.deadline` through every search;
it must not reset the total budget for every move. Model loading, actors and
training also need budget enforcement in their owning milestones.

`SearchConfig` holds simulations, `c_puct`, mode, root noise, temperature, seed,
Dirichlet alpha and fraction. Evaluation mode requires no noise and temperature
zero. The YAML bridge maps arena to evaluation and smoke/experiment to selfplay;
it uses their exact simulation count, constant, noise, temperature and seed.
For enabled selfplay noise, v1 fixes alpha at 0.3 and fraction at 0.25. These
defaults are engineering choices, not tuned strength results. No frozen YAML
file is modified or overridden. Future run records must record the full effective
search config and version, including these defaults.

An evaluator receives `(episode, *, deadline)` and returns `Evaluation(logits,
value)`: 209 **original-board-coordinate** logits and a current-player value.
All logits, including illegal ones, must be finite; value must be finite and in
[-1, 1]. Invalid evaluations fail explicitly without clipping or uniform fallback.
The neural adapter requires an already-eval model and never toggles caller state;
inference uses no autograd and converts canonical logits back through the inverse
action transform. Current-player values keep their sign. The caller must hold
the same model weights fixed for a search; concurrent training is not supported.

## Expansion, Selection and Backup

Every legal engine action has an edge. Illegal actions have zero priors, visits,
value sums and target probability. Legal logits are normalized with stable
softmax. Each edge stores prior P, visit count N and parent-perspective W.
Unvisited Q is zero; otherwise Q = W/N. Selection maximizes exactly:

```text
Q + c_puct * P * sqrt(max(1, sum(N))) / (1 + N)
```

Equal scores choose the lowest action ID. The exploration floor makes the
first selection use the prior. Every backed-up edge negates the child-perspective
value exactly once. Exact goal and repetition outcomes bypass neural inference.
Root value is the initial evaluator estimate (or exact terminal value), not a
visit-weighted claim of calibrated strength. Edge W/N carries search estimates.

Root expansion occurs before counted simulations. Each complete simulation
traverses at least one root edge and backs up once; the sum of root visits equals
`completed_simulations`. An interrupted simulation adds no visits at any depth.
Expansions or evaluator calls made before interruption may remain in diagnostic
snapshots, with zero added visits. There are no virtual visits or prior pseudocounts.

The v1 policy target is always `N / sum(N)`, independently of temperature.
With zero visits it is an all-zero vector, never the priors. Temperature zero
selects the maximal visit count, breaking ties by lowest action ID. Positive
selfplay temperature samples from visit counts raised to `1 / temperature`;
it changes action selection only. A local seeded RNG supplies root-only
Dirichlet noise and action sampling without altering global RNG state.

## History, Cycles and Stops

Every call builds a fresh tree, with nodes identified by the complete
`Episode.adjudication_key` (ordered history, ply cap and outcome). No network
tensor or board-only key caches adjudication. A feature collision from M2's
partial observation therefore cannot share an incorrect terminal decision.
No tree/evaluation cache survives between calls or across model identities.

A simulation tracks only its own ancestor full-position keys. Encountering an
ancestor again may back up zero and increment `cycle_cutoffs`; this leaves the
real episode unfinished. Positions that occurred only before the search root
are not automatic search cycles. A genuine third occurrence is adjudicated
first and counted as an exact terminal leaf. These two uses of zero are distinct;
only actual complete episodes can produce M4 replay outcome labels.

| Search status | Meaning | Action / labels |
| --- | --- | --- |
| `completed` | Requested simulations all backed up within the observed budget | Legal selected action; policy is a search target, not an outcome label |
| `terminal` | Root already has a goal or repetition result | No action or visits; exact current-player root value |
| `root_stopped` | Root already truncated or failed | No action, root value or outcome labels |
| `simulation_truncated` | A simulated continuation reaches its ply limit | No action; prior complete visits remain diagnostics; unknown leaf value is not backed up |
| `budget_exhausted` | Cooperative deadline check fails | No action; prior complete visits remain diagnostics; zero-visit policy remains zero |

Existing stopped-root adjudication is returned even if the supplied deadline
has expired, because no search work is required. Live roots check the deadline
before work. Numeric/evaluator errors raise explicit exceptions; infrastructure
errors and interruptions propagate to the caller/`RunSession`. They never turn
into eligible completed searches. M4 must handle these statuses explicitly and
exclude incomplete games from replay.

The shared deadline brackets expensive search operations and is passed into the
evaluator. Only complete simulations are committed. Time accounting reports
the shared deadline's elapsed seconds, total budget and observed overrun.
Timing is cooperative: an atomic rule operation or model forward cannot be
preempted; after it returns, expiration prevents subsequent search work. This
is not a hard real-time guarantee. The current implementation retains the public
Episode constructor's full-history validation and single-position inference;
throughput optimization, batching, tree reuse and native search are not claimed.

## Verification Boundary

`make check-m3` runs independent tactical/sign oracles, legal-support/visit
properties, history/cycle/outcome cases, strict configuration/numerics,
deterministic evaluation and seeded selfplay, neural coordinate/inference checks,
and injected-clock deadline/JSONL integration. `make check` includes these tests
with M0-M2 and selected legacy regressions. Actual results and commands belong
in `verification-m3.md`; they do not establish playing strength.

M3 does not implement actors, replay, optimization, training resume, real smoke,
arena scheduling or experiments. `make smoke`, `make arena` and `make experiment`
remain explicit nonzero NOT IMPLEMENTED errors until M4/M5/M6.
