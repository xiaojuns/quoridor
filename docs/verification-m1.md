# M1 Verification Record

Date: 2026-10-10. Base revision:
`5494513f47c810f499e9691e71fdd64ee7507681` (clean before this task).
Selected task: M1, whose sole dependency M0 was completed.
No training/strength experiment was required or run.

## Actual Commands and Results

Commands run from the repository root on Linux x86_64 / Python 3.10.12:

| Stage | Command | Actual result |
| --- | --- | --- |
| Before implementation | `make check` | Exit 0; 82 passed in 10.03s |
| Engine focused | `PYTHONPATH=training training/.venv/bin/python -m pytest -q tests/m1/test_engine.py` | Exit 0; 66 passed in 7.74s |
| Episode focused, final | `PYTHONPATH=training training/.venv/bin/python -m pytest -q tests/m1/test_episode.py` | Exit 0; 45 passed in 0.80s |
| Final cumulative gate | `make check` | Exit 0; **226 passed in 18.15s**, no skips |
| Formatting | `git diff --check` | Exit 0 |

The final cumulative gate includes 144 M1 tests (66 engine, 45 episode, 33
compatibility) and the existing 82 M0/legacy CPU tests. It also passed offline
lock consistency, exact installed-environment checking, all three YAML checks,
and arena/opening digest validation. uv reported `Would make no changes` for
22 packages; exact direct pins remain torch 2.7.1+cpu, numpy 2.2.6, PyYAML 6.0.3,
pytest 8.4.2, Hypothesis 6.142.4. Use `make check-m1` to reproduce only M1 tests.
New Hypothesis tests use `derandomize=True`; their shrinking and example
generation run under the existing locked Hypothesis version.

## Acceptance Evidence

| Unchanged acceptance criterion | Evidence |
| --- | --- |
| Independent engine | Subprocess import guards prohibit neural/clock imports; new modules operate with legacy rule/native imports blocked |
| 209 codec, masks, isolation | Exhaustive roundtrip and mask/apply checks; invalid actions preserve input; immutable containers enforced |
| Tactical and property invariants | Four jump directions, boundary/blocked diagonals, wall overlaps/crossing, both players' path preservation; independent edge-graph flood-fill and reachable-state properties |
| Distinct outcomes and labels | Real legal cycles produce third-occurrence draws; real legal 256-ply traces test goal priority and unlabelled cap; deadline/interruption/failure and current-player targets tested |
| Native condition | No new-version native backend is enabled; loader explicitly rejects native/browser/automatic fallback. New-version differential tests remain required before enabling one |

The metadata gate rejects legacy or unversioned rule/codec/value identities.
It does not claim checkpoint/encoder/replay compatibility, which remains M2/M4.
New episode constructors verify every board key and legal history transition,
including rejecting continuations past an earlier ending.

## Defects Found During Review and Their Reproduction

The first integration run passed 119 tests in 22.72s. Independent review then
found that the first episode implementation accepted a premature ply-limit
outcome, garbage history, and continuation after a completed repetition draw.
The initial probes returned these incorrect states:

```text
premature_cap: plies=0, outcome=truncated/ply_limit
garbage_history: plies=1, outcome=None
continued_draw: terminated/repetition -> plies=9, outcome=None
```

These were implementation defects; acceptance criteria and frozen fixtures were
not changed. Strict canonical-key, adjacent-transition, prior-ending and outcome
validation now reject all three, plus noncanonical keys and disconnected moves.
The independent reviewer reran all five probes: 5/5 rejected as expected.
The following original defect probes now pass as rejection checks:

```bash
PYTHONPATH=training training/.venv/bin/python - <<'PY'
from dataclasses import replace
from quoridor import engine
from quoridor.episode import Episode, Outcome, start_episode, advance_episode

initial = start_episode()
drawn = initial
for action in (75, 3, 76, 4) * 2:
    drawn = advance_episode(drawn, action)
child = engine.apply_action(drawn.state, 67)
probes = {
    'premature_cap': lambda: replace(initial, outcome=Outcome('truncated', 'ply_limit')),
    'garbage_history': lambda: replace(initial, history=('not a position key', initial.history[-1])),
    'continued_draw': lambda: Episode(child, (*drawn.history, engine.position_key(child))),
}
for name, probe in probes.items():
    try:
        probe()
    except ValueError:
        print(name, 'rejected')
    else:
        raise AssertionError(name)
PY
```

Regression tests are named
`test_history_rejects_fake_ply_cap_garbage_legacy_and_disconnected_keys` and
`test_history_rejects_continuing_after_a_real_draw_or_ply_limit` in
`tests/m1/test_episode.py`.

## Frozen Inputs and Criteria

This comparison was actually run and passed: **10 config/fixture files
byte-identical, every M0-M6 acceptance criterion unchanged, dependency lock
unchanged**. It is reproducible with:

```bash
training/.venv/bin/python - <<'PY'
from pathlib import Path
import subprocess
import yaml
base = '5494513f47c810f499e9691e71fdd64ee7507681'
paths = subprocess.check_output([
    'git', 'ls-tree', '-r', '--name-only', base,
    'configs', 'training/fixtures', 'tests/fixtures',
]).decode().splitlines()
for name in paths:
    assert Path(name).read_bytes() == subprocess.check_output(['git', 'show', f'{base}:{name}']), name
old = yaml.safe_load(subprocess.check_output(['git', 'show', f'{base}:tasks.yaml']))
new = yaml.safe_load(Path('tasks.yaml').read_text())
assert {m['id']: m['acceptance'] for m in old['milestones']} == {m['id']: m['acceptance'] for m in new['milestones']}
assert Path('training/uv.lock').read_bytes() == subprocess.check_output(['git', 'show', f'{base}:training/uv.lock'])
print(len(paths), 'unchanged config/fixture files; criteria and lock unchanged')
PY
```

The arena YAML SHA-256 remains
`cc97538bdc08aaa81ae1168fb48269da837246871544d4f38b08041d1f7aecb1`.

## Limits and Next Task

No new self-play/training job, GPU computation, native/browser conformance run,
replay migration, checkpoint adaptation, arena match, or strength experiment ran.
Tests use synthetic legal game traces and temporary run directories. A fake-clock
integration proves a goal completed after its budget retains the goal label while
`RunSession` records elapsed time and `budget_exhausted`; real scheduler enforcement
remains M3/M4/M6. Legacy test success certifies only retained legacy behavior.

M1 is complete with no known blocking issue. Stop here. M2 is next: separate the
encoder from the network, resolve history observability, and version transforms
and checkpoint identity. M2-M6 remain pending. Execution stubs still report
`NOT IMPLEMENTED` until their owning milestones.
