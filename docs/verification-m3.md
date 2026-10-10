# M3 Verification Record

Date: 2026-10-10. Starting revision:
`1d45df03b0952ad0e1da49eda6832ae7c9d4bc5b` (clean worktree).
M3 was the highest-priority unfinished task with both M1 and M2 complete.
No training, tuning, selfplay job or strength experiment was required or run.

## Actual Checks

Environment: Linux x86_64, Python 3.10.12, locked CPU PyTorch 2.7.1+cpu,
NumPy 2.2.6, PyYAML 6.0.3, pytest 8.4.2 and Hypothesis 6.142.4.
Machine-readable observations and final command output are saved in
`verification-m3-results.jsonl`. Run reproduction commands from the repo root.

| Command / stage | Observed result |
| --- | --- |
| Baseline `make check`, before edits | Exit 0; **471 passed in 75.01s**, no skips; environment/lock/YAML gates passed |
| Adapter/config focused pytest | Exit 0; 38 passed in 2.82s |
| Initial independent core/deadline pytest | Exit 0; 64 passed in 8.15s |
| Initial import/JSONL integration pytest | Exit 0; 2 passed in 0.48s |
| Final `make check-m3`, including review regressions | Exit 0; **115 passed in 11.95s**, no skips |
| Final cumulative `make check` | Exit 0; **586 passed in 85.70s**, no skips (115 M3 + 471 existing); all environment/lock/YAML gates passed |
| Frozen-input / acceptance comparison below | Exit 0; 11 prior config/fixture files and all M0-M6 acceptance criteria unchanged; lock, old rules/search and 10 prior test files unchanged |
| `make help`, `git diff --check`, roadmap/result consistency | Exit 0; M3 entry documented, unchanged acceptance, M0-M3 completed / M4-M6 pending, JSONL contains successful final checks |

Focused command used for the adapter/config checks:

```bash
env PYTHONPATH=training OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 training/.venv/bin/python -m pytest -q tests/m3/test_search_evaluator.py tests/m3/test_search_config.py
```

The initial core/deadline command was
`training/.venv/bin/python -m pytest -q tests/m3/test_puct.py tests/m3/test_puct_deadlines.py`.
The initial runtime command used the environment prefix above and
`python -m pytest -q tests/m3/test_search_runtime.py`. Use the explicit environment
prefix when reproducing individual test selections. The final complete
reproduction entries are:

```bash
make check-m3
make check
```

## Acceptance Evidence

| Unchanged M3 acceptance | Actual test evidence |
| --- | --- |
| Immediate wins and odd/even-depth signs | Independent one-ply win oracle for both players; real two/three-ply terminal paths with hand-computed W; positive/negative nonterminal leaves at depths 1-4 |
| Complete legal expansion, no illegal visits/probabilities | Initial 131 legal actions including all 128 walls; illegal high logits masked; full 209 entries; reachable-state Hypothesis invariants; legal engine transitions |
| Versioned visit/target convention | Root expansion excluded from simulations, root N sum equals complete backups, raw `N/sum(N)` even at temperature zero, deterministic ties, discriminating check against legacy `sqrt(N+1)` |
| History-sensitive exact decisions | Equal current boards and current occurrence counts with different complete histories; same continuation is an exact third repetition in one history and only a local cycle in the other; node keys contain full history |
| Explicit deadlines and no invented cycle outcome | Preexpired roots, evaluator/transition overruns, partial and exact-boundary expiration, stopped roots, gamma rejection deadline, real goal precedence, unlabelled ply-limit leaves, unchanged real episode and no fabricated labels |

Additional checks cover invalid logits/values and shapes without silent repair,
strict evaluation config, seed-independent evaluation, seeded root-only selfplay
noise, unchanged global RNG, finished leaves bypassing inference, all 209 inverse
canonical/mirror mappings, eval-only model/BatchNorm immutability, and actual
M2 model → new-rule PUCT integration. A fresh subprocess blocks torch, numpy,
encoder, model and legacy imports while core search plays a legal simulation.

The temporary `RunSession` integration uses a fake clock to stop during the
third evaluator call: exactly one prior simulation remains, action is absent,
overrun is 0.5 seconds, JSONL and summary both report `budget_exhausted`, and the
real episode remains unfinished without outcome targets. This tests the logging
boundary; it does not implement selfplay scheduling.

## Review Finding and Fix

Independent review reproduced a deadline defect before completion: with enabled
root noise and finite `dirichlet_alpha=1e308`, Python's gamma rejection sampler
could loop internally forever. The pre-fix search with `Deadline(0.1)` exceeded
an external subprocess timeout of **3 seconds**. Checks only before/after the
sampler could not enforce the budget. A separate implementation review found
the same defect.

The fix checks the shared deadline on every RNG draw inside rejection sampling;
overflow/zero noise normalization raises `SearchError` explicitly. It does not
weaken the acceptance condition or make the test pass by banning that parameter.
The same independent probe after the fix returned
`budget_exhausted None 0 0.1007` (status, action, visits, elapsed seconds).
For seed 17, all 131 ordinary root priors remained exactly equal to the standard
`random.Random(17)` gamma reference sequence.

The permanent regression uses a fake clock and an independent draw-count guard
so a future defect fails the test instead of hanging the test process. Four
draws reach the declared deadline with zero committed simulations. Reproduce:

```bash
env PYTHONPATH=training OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 training/.venv/bin/python -m pytest -q tests/m3/test_puct_deadlines.py::test_deadline_is_checked_inside_gamma_rejection_sampling
```

## Frozen Inputs and Criteria Reproduction

The following comparison was actually run and passed:

```bash
training/.venv/bin/python - <<'PY'
from pathlib import Path
import subprocess
import yaml
base = '1d45df03b0952ad0e1da49eda6832ae7c9d4bc5b'
paths = subprocess.check_output([
    'git', 'ls-tree', '-r', '--name-only', base,
    'configs', 'training/fixtures', 'tests/fixtures',
]).decode().splitlines()
for name in paths:
    assert Path(name).read_bytes() == subprocess.check_output(['git', 'show', f'{base}:{name}']), name
old = yaml.safe_load(subprocess.check_output(['git', 'show', f'{base}:tasks.yaml']))
new = yaml.safe_load(Path('tasks.yaml').read_text())
assert {m['id']: m['acceptance'] for m in old['milestones']} == {m['id']: m['acceptance'] for m in new['milestones']}
protected = ['training/uv.lock', 'training/quoridor/engine.py', 'training/quoridor/episode.py', 'training/quoridor/mcts.py', 'training/quoridor/evaluator.py', 'training/quoridor/network.py', 'training/quoridor/game.py']
protected += subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', base, 'tests/m0', 'tests/m1', 'tests/m2']).decode().splitlines()
for name in protected:
    assert Path(name).read_bytes() == subprocess.check_output(['git', 'show', f'{base}:{name}']), name
print(len(paths), 'unchanged config/fixture files; acceptance, lock, existing rules/search and prior tests unchanged')
PY
```

Arena YAML SHA-256 remains
`cc97538bdc08aaa81ae1168fb48269da837246871544d4f38b08041d1f7aecb1`.
No historical replay, checkpoint, dataset, run or GPU ledger was modified.

## Limits and Next Task

M3 provides single-process reference PUCT with fresh trees and single-position
inference. Public episode transitions retain full-history validation. Batching,
tree reuse, native/GPU search, measured throughput and playing strength are not
claimed. Deadline enforcement is cooperative: in-flight rule/model operations
cannot be preempted, and any observed overrun is recorded.

M3 is complete with no known remaining blocker. This round stops before M4.

M4 remains next: actual bounded selfplay/replay/train/checkpoint smoke. Actors,
optimizer/replay lineage and complete training resume are not implemented here.
M5 frozen arena execution and M6 experiments remain pending. `make smoke`,
`make arena` and `make experiment` continue to fail explicitly as unimplemented;
the retained M0 tests verify their CLI nonzero behavior. The make wrappers are
unchanged; their direct stub invocations were recorded in M0. Temporary
synthetic model/search checks are not training jobs or strength evaluations.
