"""Search stops remain truthful at the existing run/JSONL boundary."""

import json
import subprocess
import sys

from quoridor.episode import start_episode
from quoridor.project_config import PROJECT_ROOT, load_project_config
from quoridor.project_runtime import RunSession
from quoridor.puct import Evaluation, SearchConfig, run_search


def test_actual_m2_model_completes_bounded_new_rule_search():
    import torch

    from quoridor.engine import initial_state, legal_actions
    from quoridor.policy_value import NetworkConfig, PolicyValueNetwork
    from quoridor.project_runtime import Deadline
    from quoridor.search_evaluator import NeuralEvaluator

    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = PolicyValueNetwork(NetworkConfig.smoke()).eval()
        episode = start_episode(initial_state(to_play=1))
        result = run_search(
            episode, NeuralEvaluator(model), SearchConfig(simulations=3),
            deadline=Deadline(30),
        )
    assert result.status == "completed"
    assert result.action in legal_actions(episode.state)
    assert result.completed_simulations == sum(result.visits) == 3
    assert -1 <= result.root_value <= 1
    assert result.evaluations >= 2
    assert episode.plies == 0 and episode.outcome is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_search_import_and_play_do_not_load_neural_or_legacy_modules():
    program = r'''
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy'} or fullname in {
            'quoridor.game', 'quoridor.codec', 'quoridor.network',
            'quoridor.mcts', 'quoridor.encoder', 'quoridor.policy_value',
        }:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Guard())
from quoridor.episode import start_episode
from quoridor.project_runtime import Deadline
from quoridor.puct import Evaluation, SearchConfig, run_search
def oracle(episode, *, deadline):
    return Evaluation((0.0,) * 209, 0.25)
result = run_search(start_episode(), oracle, SearchConfig(simulations=1),
                    deadline=Deadline(20))
assert result.status == 'completed'
assert sum(result.visits) == 1
'''
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_partial_search_budget_is_logged_without_fabricating_game_result(tmp_path):
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    config = load_project_config(PROJECT_ROOT / "configs/smoke.yaml")
    episode = start_episode()
    calls = 0

    def slow_oracle(leaf, *, deadline):
        nonlocal calls
        assert not leaf.finished
        calls += 1
        if calls == 3:
            clock.now = deadline.seconds + 0.5
        return Evaluation((0.0,) * 209, 0.25)

    with RunSession(config, run_root=tmp_path, clock=clock) as run:
        result = run_search(
            episode, slow_oracle, SearchConfig(simulations=8), deadline=run.deadline,
        )
        run.log(
            "search_finished", search_status=result.status,
            completed_simulations=result.completed_simulations,
            visits=list(result.visits), action=result.action,
            overrun_seconds=result.overrun_seconds,
        )

    assert result.status == "budget_exhausted"
    assert calls == 3
    assert result.completed_simulations == sum(result.visits) == 1
    assert result.action is None
    assert result.overrun_seconds == 0.5
    assert episode.outcome is None
    assert episode.value_targets((0, 1)) is None
    events = [json.loads(line) for line in (run.path / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "run_started", "search_finished", "run_finished",
    ]
    assert events[1]["search_status"] == "budget_exhausted"
    assert events[1]["completed_simulations"] == 1
    assert events[2]["status"] == "budget_exhausted"
    summary = json.loads((run.path / "summary.json").read_text())
    assert summary["status"] == "budget_exhausted"
    assert summary["elapsed_seconds"] == 60.5
