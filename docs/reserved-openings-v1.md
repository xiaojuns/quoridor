# Reserved final opening asset

M5 freezes `az-reserved-final-v1` before M6 tuning. This is engineering asset
generation, not a strength experiment. No candidate, champion, model output,
or game score participates in generation or selection. Historical
`training/fixtures/openings/final.json` was exposed in Phase 4F evaluation and
does not qualify as an untouched holdout.

The hypothesis, exact selection rule, seed, algorithm, source identities and
180-second maximum generation time were saved in
`configs/reserved-final-v1-generation.yaml` before execution. The algorithm
uses Python `random.Random(2026101005)` and samples one index uniformly from
the engine's complete ascending legal-action list at each of 16 plies. It
starts every attempt from the standard player-zero initial board, rejects any
prefix that ends an episode, and accepts the first 200 distinct final boards
that survive the exclusions. This is a declared opening distribution, not a
claim that openings are balanced or representative of skilled play.

Each entry retains all actions and all 17 ordered full-position keys. Loading
replays every prefix through `quoridor.engine` and `quoridor.episode`, rejects
illegal or terminal prefixes, and compares the derived history and final board
exactly. Arena use must continue the inherited full history, not start a fresh
episode at the final board. Prefix plies count towards the unchanged 256-ply
episode limit.

Uniqueness and disjointness use the minimum full-position key under identity,
left/right reflection, 180-degree rotation with player swap, and their
composition. Keys include side to move, player inventories, pawns and walls.
The exclusion sources are the existing dev/final and recovery-dev/recovery-final
assets. Their raw bytes are hash-bound. Historical boards are converted only
for overlap exclusion; no legacy result or repetition label is imported.

The frozen manifest is `configs/reserved-final-v1.yaml`, with its raw SHA-256
in `configs/reserved-final-v1.sha256`. It binds the generation configuration
and `training/fixtures/openings/az-reserved-v1/final.json`. The asset SHA-256 is
`2543fb5935d06a3de3afb67fae290d014c5a4a40d48bde0af97d3d29113f55dd`.
Existing frozen configs and opening files are unchanged. `arena-dev-v1` still
uses its original development suite. Using the new reserved suite formally
requires a separately reviewed, frozen evaluation protocol; no run-time arena
override selects it. M6 must not use repeated reserved evaluations for tuning.

Run these commands from the repository root with the locked environment:

```sh
PYTHONPATH=training training/.venv/bin/python -m quoridor.reserved_openings validate
PYTHONPATH=training training/.venv/bin/python -m quoridor.reserved_openings generate --output /tmp/az-reserved-final-v1-reproduced.json
cmp training/fixtures/openings/az-reserved-v1/final.json /tmp/az-reserved-final-v1-reproduced.json
PYTHONPATH=training training/.venv/bin/python -m pytest -q tests/m5/test_reserved_openings.py
```

Generation refuses to overwrite an existing output. Choose a fresh output path
when repeating the reproduction command. Each generation receives its own
`runs/reserved-generation/<uuid>/` directory with the preregistered YAML,
code/dependency provenance and JSONL events. Partial or expired generation
raises an error rather than returning fewer openings.

Actual first generation completed in 100.3968 seconds, below 180 seconds,
and produced 200 openings (482,105 bytes). Its run directory is
`runs/reserved-generation/cecc63f6809741e3a8f326dc995f05d8`.
The focused tests completed with **10 passed in 9.37s**, including full replay
of all 200 openings, deterministic small regeneration, all supported symmetry
identities, hash rejection, duplicate/overlap rejection, invalid history and
expired-budget handling. These results make no playing-strength claim.

Independent full regeneration completed in 100.0880 seconds in
`runs/reserved-generation/d2cd8610b13c4d05a8c1a286c20e6d24`. It produced the
same 200-opening SHA-256; the `cmp` command above exited 0, confirming exact raw
byte identity. Actual commands and results are also saved in
`docs/reserved-openings-v1-results.jsonl`.
