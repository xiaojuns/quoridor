# M5 verification and remaining acceptance

Date: 2026-10-10. Baseline revision:
`eaa2ca37ccf7054829cfcab4fc2d6723e9ce6f55`, clean and synchronized with origin.
M5 was the highest-priority unfinished milestone with all dependencies met
(M4 completed). This follow-up supersedes the prior M4 stopping point; M6 is
not started. M5 remains **in progress** until an actual frozen 200-pair neural
comparison completes. Passing the engineering tests below does not fulfill
that outstanding acceptance condition.

## Checks actually run

Before edits, `make check` passed **603 tests in 89.87s**, with no skips.
All 22 locked packages, offline lock/sync, YAML and frozen digests passed.
Final results and captured command output are saved in
`verification-m5-results.jsonl`.

| Command / check | Actual result |
| --- | --- |
| Baseline `make check` | Exit 0; 603 passed in 89.87s, no skips |
| Final `make check-m5` | Exit 0; **87 passed in 36.17s**, no skips |
| Final cumulative `make check` | Exit 0; **690 passed in 123.08s**, no skips |
| Locked environment, offline lock/sync, YAML and frozen digests | Passed; all 22 packages unchanged; offline sync would make no changes |
| `make help` | Exit 0; documents implemented arena and unimplemented experiment |
| `make arena` without inputs | Make/CLI exit 2; explicit nonempty candidate/champion input error |
| `make experiment` | Make exit 2, CLI exit 3; explicit `not_implemented`, M6 |
| Baseline preservation | 31 existing files byte-identical; all acceptance arrays unchanged |
| `git diff --check`, new-file whitespace and YAML/JSONL status checks | Passed |

Reproduce the final focused and cumulative gates from the repository root:

```sh
make check-m5
make check
```

The checks cover all 200 dev opening prefixes under M1 with inherited history;
real 256-simulation neural PUCT on both tactical goal legs and a repetition
draw; actual artifact loading/hash and compatibility checks; synthetic full
400-game scheduling with side swaps; independent W/L/D, denominator, central
95% bootstrap and threshold fixtures; interruption/deadline/illegal/numerical
failure paths; and the fresh reserved suite. The full-schedule test substitutes
synthetic game results: it is **not** 400 neural games or a strength result.

An independent review found that expiration after statistics could return
`promotionEligible=true` while the run summary said `budget_exhausted`. The
fake-clock reproducer used 400 synthetic wins. Regression coverage now includes
expiry during bootstrap, after statistics and during finalization, as well as
interruption between games and exceptions after actual result publication.
Finalization failures invalidate both top-level and nested promotion flags;
summary events are provisional until the run and result both complete.
Cancelled results retain completed counts and
diagnostic scores but cannot promote. Statistics use the original protocol's
central 95% interval, not a less strict one-sided 95% bound.

## Concrete unresolved input and full-run evidence

This command was actually executed to check the existing champion's compatibility:

```sh
make arena \
  CANDIDATE=runs/smoke/20261010T130800Z-dc36bd89ff77403a8c7a6e4526004ec1/model-weights.pt \
  CHAMPION=training/runs/phase4/champions/champion-g00000014-s00000000-67b580c9087f.pt
```

Observed: make exit **2**, CLI exit **1**, `arena_failed` with
`incompatible champion model artifact: checkpoint fields mismatch; legacy checkpoints require explicit migration`.
Run `runs/arena/20261010T135746Z-58708042fd9b42cebe4cd106fdbc8ec0` records status
`failed`, elapsed 0.6606s, and **zero games**. It is an expected compatibility
rejection, not a completed arena. Subsequent implementation also logs requested
model paths and per-artifact hashes before deserialization for failed preflight
audits. The original failed run is retained without modification.

The only identified compatible local model family comes from M4's tactical
smoke; its weights were deliberately initialized for a one-ply goal. No
candidate/champion pair for a meaningful new-rule comparison has been supplied,
and no historical champion was migrated. No complete frozen 400-game neural
comparison or promotion occurred. Do not mark M5 complete or start M6 based on
mock schedules, tactical fixtures, this rejection, or synthetic score gates.

Next: designate compatible model-only candidate and champion snapshots and
record their exact hashes and evaluation hypothesis before running the documented
`make arena` invocation. Keep the unchanged 3600-second/256-simulation/200-pair
protocol. If its budget expires, retain the incomplete result and address the
measured bottleneck; do not reduce pairs/simulations or extend this frozen budget.

## Reserved suite and fixed opponents

Fresh reserved final generation was registered in
`configs/reserved-final-v1-generation.yaml` before execution: 200 legal 16-ply
prefixes, seed 2026101005, 180 seconds, no model/performance selection. Generation
took 100.3968s. An independent complete regeneration took 100.0880s and `cmp`
confirmed identical bytes. Frozen SHA-256:
`2543fb5935d06a3de3afb67fae290d014c5a4a40d48bde0af97d3d29113f55dd`.
See `reserved-openings-v1.md` and its JSONL for exact commands and run directories.
Every final position is distinct under mirror/player-swap symmetry and excluded
from all four historical suites. Historical final exposure is declared; neither
suite has been used for model selection in this task.

Seven opponent unit checks passed in 2.57s. An additional real Node v22.17.1
bridge check used a declared 15-second deadline: both starting players returned
legal repeatable actions (player 0: 67; player 1: 13), 587 nodes and depth 3,
elapsed 5.5758s. This is action-interface verification, not an arena match.
Node is an optional legacy-opponent dependency, not part of the locked Python
CPU gate; tests use an explicit bridge test double and do not skip absent Node.
The actual optional integration command is:

```sh
PYTHONPATH=training training/.venv/bin/python - <<'PY'
import json
from quoridor import engine
from quoridor.arena_opponents import ClassicOpponent
from quoridor.episode import start_episode
from quoridor.project_runtime import Deadline
budget = Deadline(15)
opponent = ClassicOpponent(deadline=budget)
try:
    results = []
    for side in (0, 1):
        episode = start_episode(engine.initial_state(side))
        first = opponent.choose(episode, deadline=budget)
        repeat = opponent.choose(episode, deadline=budget)
        assert first == repeat
        assert first.action in engine.legal_actions(episode.state)
        results.append({'side': side, 'action': first.action, 'nodes': first.nodes, 'depth': first.depth})
    print(json.dumps({'status': 'passed', 'kind': 'real-node-bridge-diagnostic', 'budget_seconds': 15, 'elapsed_seconds': budget.elapsed_seconds, 'results': results}))
finally:
    opponent.close()
PY
```

## Frozen-input preservation

Compared with the baseline using `git ls-tree`/`git show`: **31** existing
config/fixture/dependency manifest/lock/M0-M4 test files are byte-identical; all M0-M6 acceptance
arrays are unchanged. Only new reserved/opponent assets were added. Legacy
training data, models, GPU ledger, old rules and evaluation assets were not
modified. `make experiment` remains explicitly unimplemented.

Reproduce the preservation check:

```sh
training/.venv/bin/python - <<'PY'
import subprocess
from pathlib import Path
import yaml
baseline = 'eaa2ca37ccf7054829cfcab4fc2d6723e9ce6f55'
paths = subprocess.check_output([
    'git', 'ls-tree', '-r', '--name-only', baseline, '--',
    'configs', 'training/fixtures', 'training/uv.lock', 'training/pyproject.toml',
    'tests/m0', 'tests/m1', 'tests/m2', 'tests/m3', 'tests/m4',
], text=True).splitlines()
for name in paths:
    assert Path(name).read_bytes() == subprocess.check_output([
        'git', 'show', f'{baseline}:{name}'
    ]), name
old = yaml.safe_load(subprocess.check_output([
    'git', 'show', f'{baseline}:tasks.yaml'
], text=True))
current = yaml.safe_load(Path('tasks.yaml').read_text())
assert {m['id']: m['acceptance'] for m in old['milestones']} == {
    m['id']: m['acceptance'] for m in current['milestones']
}
print(f'{len(paths)} files byte-identical; all acceptance arrays unchanged')
PY
```
