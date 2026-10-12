"""M5 integration contracts; synthetic schedules are never strength results."""

from dataclasses import replace
import hashlib
import json

import pytest
import torch

from quoridor import arena_v1, engine
from quoridor.arena_v1 import (
    ArenaArtifact, ArenaExecutionError, ArenaGame, ArenaInputError,
    _opening_engine_episode, load_arena_artifacts, load_frozen_openings,
    play_arena_game, run_arena,
)
from quoridor.model_checkpoint import save_model_checkpoint
from quoridor.openings import generate_opening
from quoridor.policy_value import NetworkConfig, PolicyValueNetwork
from quoridor.project_cli import main
from quoridor.project_config import PROJECT_ROOT, load_project_config
from quoridor.project_runtime import BudgetExpired, Deadline, RunSession
from quoridor.puct import Evaluation, InvalidEvaluationError, SearchConfig, run_search


@pytest.fixture(scope="module")
def config():
    return load_project_config(PROJECT_ROOT / "configs/arena.yaml", expected_mode="arena")


@pytest.fixture(scope="module")
def openings(config):
    return load_frozen_openings(config)


def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        value = PolicyValueNetwork(NetworkConfig.smoke())
    with torch.no_grad():
        for parameter in value.parameters():
            parameter.zero_()
        value.policy_head[-1].bias[:9] = 100
    return value.eval()


def tactical_opening():
    actions = [engine.encode_move(8, 3), engine.encode_move(0, 5)]
    for row in range(7, 0, -1):
        actions.extend((engine.encode_move(row, 3), engine.encode_move(8 - row, 5)))
    return generate_opening(actions, suite="dev")


def test_all_two_hundred_frozen_prefixes_replay_with_inherited_history(config, openings):
    assert len(openings) == 200
    for opening in openings:
        episode = _opening_engine_episode(opening, max_game_plies=256)
        assert len(episode.history) == len(opening.actions) + 1
        assert episode.plies == opening.state.move_number
        assert not episode.finished
        assert episode.state.to_play == opening.state.current_player
    # A prefix that has already returned to its initial board must keep that
    # first occurrence; starting only at the final board would lose it.
    cycle = generate_opening((75, 3, 76, 4), suite="dev")
    episode = _opening_engine_episode(cycle, max_game_plies=256)
    assert episode.repetition_count == 2
    assert len(episode.history) == 5


def test_opening_digest_is_checked_again_at_actual_load(config, monkeypatch):
    from pathlib import Path

    real_read = Path.read_bytes
    frozen_path = (PROJECT_ROOT / config.document["arena"]["opening_suite"]).resolve()

    def changed_read(path):
        raw = real_read(path)
        return raw + b"\n" if path.resolve() == frozen_path else raw

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    with pytest.raises(ArenaInputError, match="SHA-256 mismatch at load"):
        load_frozen_openings(config)


def test_model_input_hash_metadata_and_architecture_are_required(tmp_path):
    candidate = save_model_checkpoint(tmp_path / "candidate.pt", model())
    champion = save_model_checkpoint(tmp_path / "champion.pt", model())
    loaded = load_arena_artifacts(
        candidate.path, champion.path,
        candidate_sha256=candidate.sha256, champion_sha256=champion.sha256,
    )
    assert loaded[0].sha256 == candidate.sha256
    assert loaded[1].sha256 == champion.sha256
    assert loaded[0].to_dict()["metadata"]["rules_contract"]["rules_version"] == "quoridor-v1"
    assert all(not module.training for module in loaded[0].model.modules())
    with pytest.raises(ArenaInputError, match="SHA-256 mismatch"):
        load_arena_artifacts(candidate.path, champion.path, candidate_sha256="0" * 64)
    bad = tmp_path / "legacy.pt"
    torch.save({"model": {}}, bad)
    with pytest.raises(ArenaInputError, match="incompatible candidate"):
        load_arena_artifacts(bad, champion.path)
    different = save_model_checkpoint(tmp_path / "different.pt", PolicyValueNetwork(NetworkConfig(channels=4, blocks=1)))
    with pytest.raises(ArenaInputError, match="architectures must match"):
        load_arena_artifacts(candidate.path, different.path)


def test_real_puct_uses_frozen_256_simulations_and_swapped_models(config):
    opening = tactical_opening()
    events = []
    for leg, candidate in ((0, 0), (1, 1)):
        result = play_arena_game(
            config, opening, leg=leg, candidate_player=candidate,
            candidate_model=model(), champion_model=model(), deadline=Deadline(20),
            event_callback=lambda event, **fields: events.append((event, fields)),
        )
        assert result.status == "completed"
        assert result.reason == "goal"
        assert result.winner == 0
        assert result.search_simulations == 256
        assert result.candidate_outcome == ("win" if leg == 0 else "loss")
    assert len(events) == 2
    for name, event in events:
        assert name == "arena_search"
        assert event["completed_simulations"] == event["expected_simulations"] == 256
        assert sum(event["visits"]) == 256
        assert event["action"] == engine.encode_move(0, 3)
        assert len(event["history_sha256"]) == 64


def test_real_repetition_draw_uses_opening_history_and_is_not_truncation(config, monkeypatch):
    # The initial board has occurred twice already in this reachable prefix.
    # Four more actual moves must draw; a board-only arena would miss it.
    opening = generate_opening((75, 3, 76, 4), suite="dev")

    class CycleEvaluator:
        def __init__(self, unused):
            pass

        def __call__(self, episode, *, deadline):
            action = (75, 3, 76, 4)[episode.plies % 4]
            logits = tuple(1000.0 if choice == action else -1000.0 for choice in range(209))
            return Evaluation(logits, 0.0)

    monkeypatch.setattr(arena_v1, "NeuralEvaluator", CycleEvaluator)
    result = play_arena_game(
        config, opening, leg=0, candidate_player=0,
        candidate_model=object(), champion_model=object(), deadline=Deadline(20),
    )
    assert result.status == "completed"
    assert result.reason == "repetition"
    assert result.candidate_outcome == "draw"
    assert result.winner is None
    assert result.plies == 8


@pytest.mark.parametrize("failure,reason", [(InvalidEvaluationError("NaN fixture"), "numerical_error"), (OSError("disk failure fixture"), "infrastructure_error"), (KeyboardInterrupt(), "interrupted")])
def test_execution_errors_and_interruption_have_no_winner(config, monkeypatch, failure, reason):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(arena_v1, "run_search", fail)
    result = play_arena_game(
        config, tactical_opening(), leg=0, candidate_player=0,
        candidate_model=model(), champion_model=model(), deadline=Deadline(20),
    )
    assert result.reason == reason
    assert result.winner is None
    assert result.candidate_outcome == "incomplete"
    if reason != "interrupted":
        assert result.detail == str(failure)


def _mock_inputs(monkeypatch, tmp_path, openings):
    participant = model()
    artifact = ArenaArtifact(tmp_path / "synthetic.pt", "a" * 64, participant, "{}")
    monkeypatch.setattr(arena_v1, "load_arena_artifacts", lambda *args, **kwargs: (artifact, artifact))
    monkeypatch.setattr(arena_v1, "load_frozen_openings", lambda *args, **kwargs: openings)


def test_all_400_games_are_scheduled_as_fixed_side_swapped_pairs(config, openings, monkeypatch, tmp_path):
    _mock_inputs(monkeypatch, tmp_path, openings)
    calls = []

    def complete(cfg, opening, **kwargs):
        calls.append((opening.id, kwargs["leg"], kwargs["candidate_player"], kwargs["deadline"]))
        assert cfg is config
        assert cfg.document["search"] == {"simulations": 256, "c_puct": 1.5, "root_noise": False, "temperature": 0.0}
        return ArenaGame(opening.id, kwargs["leg"], kwargs["candidate_player"], "completed", "repetition", None, 20, 256)

    monkeypatch.setattr(arena_v1, "play_arena_game", complete)
    result = run_arena(config, "candidate", "champion", run_root=tmp_path)
    assert result["games"] == result["scheduledGames"] == 400
    assert result["completePairs"] == 200
    assert result["draws"] == 400
    assert result["score"] == 0.5
    assert result["promotionEligible"] is False
    assert result["status"] == "completed"
    assert len({id(call[3]) for call in calls}) == 1
    for index, opening in enumerate(openings):
        assert calls[2 * index][:3] == (opening.id, 0, opening.state.current_player)
        assert calls[2 * index + 1][:3] == (opening.id, 1, 1 - opening.state.current_player)
    run_dir = __import__("pathlib").Path(result["run_dir"])
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "arena_game" for event in events) == 400
    assert json.loads((run_dir / "arena-result.json").read_text())["draws"] == 400
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("reason,exception,run_status", [("ply_limit", ArenaExecutionError, "failed"), ("interrupted", KeyboardInterrupt, "interrupted"), ("time_budget", BudgetExpired, "budget_exhausted")])
def test_incomplete_run_never_reports_success(config, openings, monkeypatch, tmp_path, reason, exception, run_status):
    _mock_inputs(monkeypatch, tmp_path, openings)
    now = [0.0]
    monkeypatch.setattr(arena_v1, "RunSession", lambda cfg, **kw: RunSession(cfg, clock=lambda: now[0], **kw))

    def incomplete(cfg, opening, **kwargs):
        if reason == "time_budget":
            now[0] = 3601.0
        return ArenaGame(opening.id, kwargs["leg"], kwargs["candidate_player"], "truncated", reason, None, 20, 256)

    monkeypatch.setattr(arena_v1, "play_arena_game", incomplete)
    with pytest.raises(exception):
        run_arena(config, "candidate", "champion", run_root=tmp_path)
    directories = list((tmp_path / "arena").iterdir())
    assert len(directories) == 1
    result = json.loads((directories[0] / "arena-result.json").read_text())
    assert result["status"] == "incomplete"
    assert not result["decisionEligible"]
    assert not result["promotionEligible"]
    assert json.loads((directories[0] / "summary.json").read_text())["status"] == run_status


def test_forged_config_snapshot_is_rejected_before_creating_a_run(config, tmp_path):
    document = config.document
    document["search"]["simulations"] = 1
    forged = replace(config, canonical_json=json.dumps(document))
    with pytest.raises(ArenaInputError, match="snapshot changed"):
        run_arena(forged, "candidate", "champion", run_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("expire_at", ["statistics", "finalization", "run_finished"])
def test_deadline_expiring_after_all_games_cannot_promote(config, openings, monkeypatch, tmp_path, expire_at):
    _mock_inputs(monkeypatch, tmp_path, openings)
    now = [0.0]
    monkeypatch.setattr(arena_v1, "RunSession", lambda cfg, **kw: RunSession(cfg, clock=lambda: now[0], **kw))

    def win(cfg, opening, **kwargs):
        return ArenaGame(opening.id, kwargs["leg"], kwargs["candidate_player"], "completed", "goal", kwargs["candidate_player"], 20, 256)

    monkeypatch.setattr(arena_v1, "play_arena_game", win)
    if expire_at == "statistics":
        original = arena_v1._statistics

        def expire(*args, **kwargs):
            result = original(*args, **kwargs)
            now[0] = 3601.0
            return result

        monkeypatch.setattr(arena_v1, "_statistics", expire)
    else:
        original = RunSession.log

        def expire(self, event, **fields):
            result = original(self, event, **fields)
            trigger = "run_finished" if expire_at == "run_finished" else "m5_arena_summary"
            if event == trigger:
                now[0] = 3601.0
            return result

        monkeypatch.setattr(RunSession, "log", expire)
    with pytest.raises(BudgetExpired):
        run_arena(config, "candidate", "champion", run_root=tmp_path)
    directory = next((tmp_path / "arena").iterdir())
    result = json.loads((directory / "arena-result.json").read_text())
    assert result["completedGames"] == 400
    assert result["wins"] == 400
    assert result["score"] == 1.0
    assert result["status"] == "incomplete"
    assert not result["decisionEligible"]
    assert not result["promotionEligible"]
    assert not result["promotionGate"]["decisionEligible"]
    assert not result["promotionGate"]["passed"]
    assert json.loads((directory / "summary.json").read_text())["status"] == "budget_exhausted"
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(len(events)))
    provisional = [event for event in events if event["event"] == "m5_arena_summary"]
    assert len(provisional) == 1 and provisional[0]["provisional"]
    if expire_at in {"finalization", "run_finished"}:
        assert events[-1]["event"] == "arena_finalization_invalidated"
        assert not events[-1]["promotionEligible"]
        assert not events[-1]["promotionGate"]["passed"]
    assert not list(directory.glob("*.tmp"))


def test_interruption_between_games_retains_true_run_status(config, openings, monkeypatch, tmp_path):
    _mock_inputs(monkeypatch, tmp_path, openings)

    def complete(cfg, opening, **kwargs):
        return ArenaGame(opening.id, kwargs["leg"], kwargs["candidate_player"], "completed", "repetition", None, 20, 256)

    monkeypatch.setattr(arena_v1, "play_arena_game", complete)
    original = RunSession.log

    def stop_after_game(self, event, **fields):
        result = original(self, event, **fields)
        if event == "arena_game":
            raise KeyboardInterrupt("between-game fixture")
        return result

    monkeypatch.setattr(RunSession, "log", stop_after_game)
    with pytest.raises(KeyboardInterrupt):
        run_arena(config, "candidate", "champion", run_root=tmp_path)
    directory = next((tmp_path / "arena").iterdir())
    result = json.loads((directory / "arena-result.json").read_text())
    assert result["games"] == result["completedGames"] == 1
    assert not result["promotionEligible"]
    assert json.loads((directory / "summary.json").read_text())["status"] == "interrupted"


@pytest.mark.parametrize("error,run_status", [(KeyboardInterrupt("after publish"), "interrupted"), (OSError("after publish"), "failed")])
def test_exception_after_real_result_publication_invalidates_gate(config, openings, monkeypatch, tmp_path, error, run_status):
    _mock_inputs(monkeypatch, tmp_path, openings)

    def win(cfg, opening, **kwargs):
        return ArenaGame(opening.id, kwargs["leg"], kwargs["candidate_player"], "completed", "goal", kwargs["candidate_player"], 20, 256)

    monkeypatch.setattr(arena_v1, "play_arena_game", win)
    original = arena_v1._save_result
    publications = []

    def publish_then_raise_once(path, summary):
        original(path, summary)
        publications.append(json.loads(path.read_text()))
        if len(publications) == 1:
            assert publications[0]["promotionEligible"]
            raise error

    monkeypatch.setattr(arena_v1, "_save_result", publish_then_raise_once)
    with pytest.raises(type(error)) as observed:
        run_arena(config, "candidate", "champion", run_root=tmp_path)
    assert observed.value is error
    assert len(publications) == 2
    directory = next((tmp_path / "arena").iterdir())
    result = json.loads((directory / "arena-result.json").read_text())
    assert result["status"] == "incomplete"
    assert not result["decisionEligible"]
    assert not result["promotionEligible"]
    assert not result["promotionGate"]["decisionEligible"]
    assert not result["promotionGate"]["passed"]
    assert json.loads((directory / "summary.json").read_text())["status"] == run_status
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "arena_finalization_invalidated"
    assert not events[-1]["promotionGate"]["passed"]


@pytest.mark.parametrize("extra", [["--candidate", "candidate.pt"], ["--champion", "champion.pt"], ["--candidate", "", "--champion", ""]])
def test_cli_requires_both_explicit_model_inputs(extra, capsys):
    assert main(["arena", "--config", "configs/arena.yaml", *extra]) == 2
    assert json.loads(capsys.readouterr().err)["event"] == "input_error"


def test_cli_rejects_search_overrides():
    with pytest.raises(SystemExit) as error:
        main(["arena", "--config", "configs/arena.yaml", "--candidate", "a", "--champion", "b", "--simulations", "1"])
    assert error.value.code == 2
