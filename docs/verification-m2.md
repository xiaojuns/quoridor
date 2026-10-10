# M2 Verification Record

Date: 2026-10-10. Starting revision:
`02a3bc196603a6f6961e8b9a37325a0176278396` (clean worktree).
M2 was the highest-priority unfinished task with its M1 dependency complete.
No training, tuning or strength experiment was needed or executed.

## Actual Checks

Linux x86_64, Python 3.10.12, pinned CPU PyTorch 2.7.1+cpu; dependency lock and
all 22 installed packages unchanged. Run from the repository root:

| Command / stage | Actual result |
| --- | --- |
| Baseline `make check`, before edits | Exit 0; 226 passed in 17.97s |
| `env PYTHONPATH=training OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 training/.venv/bin/python -m pytest -q tests/m2/test_model_config.py tests/m2/test_policy_value.py` | Exit 0; 102 passed in 2.66s |
| `training/.venv/bin/python -m pytest -q tests/m2/test_model_checkpoint.py` | Exit 0; 66 passed in 4.23s |
| Encoder/symmetry focused check after explicit CPU fix | Exit 0; 59 passed in 50.17s; one further deadline case subsequently added |
| Focused final deadline/stop groups | Exit 0; 10 passed in 1.37s |
| Final cumulative `make check` | Exit 0; **471 passed in 74.37s**, no skips |
| `make help` and `git diff --check` | Exit 0 |

The final gate covers all 245 M2 cases: 60 encoder/symmetry, 71 tensor network,
31 model YAML, 66 model checkpoints, and 17 policy/diagnostic cases. It also
passes the 226 existing M0/M1/legacy CPU tests and offline lock/environment,
run YAML and frozen arena/opening checks. New Hypothesis properties use
`derandomize=True` under the existing locked dependency version.

The focused reproduction entry for the complete final M2 suite is:

```bash
make check-m2
```

The exact encoder/symmetry focused commands above were:

```bash
env PYTHONPATH=training OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 training/.venv/bin/python -m pytest -q tests/m2/test_encoder.py tests/m2/test_symmetry.py
env PYTHONPATH=training OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 training/.venv/bin/python -m pytest -q tests/m2/test_encoder.py::test_all_completed_and_unlabelled_stops_have_no_legal_policy_support tests/m2/test_symmetry.py::test_repetition_draw_and_unlabelled_stops_survive_transformation
```

## Unchanged Acceptance and Evidence

| M2 acceptance | Evidence |
| --- | --- |
| Tensor-only model, finite 209 logits / bounded values | Fresh subprocess guards prohibit new/legacy rule imports; batch 1/multiple, forward/backward, input/parameter gradients, shape/dtype/numerical validation |
| Current-player value consistency | Goal next-player -1, draws 0, truncated labels absent; both players, rotations/swaps and recorded-player target invariance |
| Encoder derives rules through engine | Delegation tests observe public legal-mask and shortest-path calls; no encoder BFS, wall or move-rule reimplementation |
| Exact state/history/action/policy transforms | All 209 indices, asymmetric wall boards, every legal transition, complete-history roundtrip, support preservation, reachable-state Hypothesis cases |
| Checkpoint incompatibility rejection | Actual temporary save/reload reproduces outputs; old 14/16-plane metadata, rule/value/encoder mismatches, shapes, dtypes, nonfinite/invalid BN buffers and hash mismatches rejected |

YAML architecture, encoder layout, masking and value diagnostic formulas have
independent boundary/expected-value fixtures. Checkpoints are M2 inference
weight artifacts only; optimizer/replay/RNG training continuation remains M4.

## History Observability Evidence

`test_history_observation_collision_has_distinct_future_draw_adjudication`
constructs equal current boards at ply 4 with occurrence count 2 from different
left/right cycles. Their tensors and masks are identical. A shared five-ply
continuation produces a repetition draw only from the left history; the other
episode is still live. This is an intentional, tested partial-observation limit,
not an input claimed to represent the full Markov state.

Reproduce the decisive representation and roundtrip checks with:

```bash
env PYTHONPATH=training OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 training/.venv/bin/python -m pytest -q tests/m2/test_encoder.py::test_history_observation_collision_has_distinct_future_draw_adjudication tests/m2/test_model_checkpoint.py::test_roundtrip_exact_outputs_and_weights_including_batchnorm
```

The full history still belongs to `Episode` and future M3 search. Its exact
adjudication/cache identity cannot be replaced by the feature tensor. No actual
strength or gameplay calibration impact of partial observation was measured.

## Review Corrections and Limits

Review found that root-package eager legacy imports would violate a genuinely
independent tensor-model import. Root exports now resolve lazily. The existing
M1 import-isolation test was strengthened: it installs the import guard before
the very first package import, rather than first loading/deleting legacy modules.
No guard or assertion was removed. This is a test setup improvement, not a
changed frozen fixture or weakened acceptance standard.

Review also identified that encoder tensor factories could follow an ambient
default device despite promising CPU outputs. They now specify CPU explicitly;
a meta-default-device regression passes. Sparse policy/diagnostic inputs now
fail with explicit validation errors. Independent encoder/symmetry/layout
review found no blocking mismatch; deadline-stop transformations also preserve
timing evidence and missing value labels.

No historical asset was migrated, no optimizer/self-play/training job was run,
and no new arena or native/GPU workload was executed. Forward/backward operations
and temporary checkpoint roundtrips are engineering unit tests. No production
value calibration or playing-strength result is claimed. All three execution
stubs remain unavailable, as checked by the retained M0 tests.

## Frozen Inputs and Criteria Reproduction

Actually verified: 10 existing config/fixture files are byte-identical, every
M0-M6 acceptance criterion is unchanged, and the dependency lock and legacy
network/checkpoint/game files are unchanged. `configs/model.yaml` is a new
standalone architecture configuration; the arena protocol is not modified.

```bash
training/.venv/bin/python - <<'PY'
from pathlib import Path
import subprocess
import yaml
base = '02a3bc196603a6f6961e8b9a37325a0176278396'
paths = subprocess.check_output([
    'git', 'ls-tree', '-r', '--name-only', base,
    'configs', 'training/fixtures', 'tests/fixtures',
]).decode().splitlines()
for name in paths:
    assert Path(name).read_bytes() == subprocess.check_output(['git', 'show', f'{base}:{name}']), name
old = yaml.safe_load(subprocess.check_output(['git', 'show', f'{base}:tasks.yaml']))
new = yaml.safe_load(Path('tasks.yaml').read_text())
assert {m['id']: m['acceptance'] for m in old['milestones']} == {m['id']: m['acceptance'] for m in new['milestones']}
for name in ['training/uv.lock', 'training/quoridor/network.py', 'training/quoridor/checkpoint.py', 'training/quoridor/game.py']:
    assert Path(name).read_bytes() == subprocess.check_output(['git', 'show', f'{base}:{name}']), name
print(len(paths), 'unchanged config/fixture files; acceptance, lock and legacy implementations unchanged')
PY
```

Arena YAML SHA-256 remains
`cc97538bdc08aaa81ae1168fb48269da837246871544d4f38b08041d1f7aecb1`.

M2 is complete with no known blocking issue. Stop here; M3 (correct, bounded
PUCT using the new encoder/model and exact episode history) is next.
