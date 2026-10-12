"""Fresh reserved openings, generated without consulting models or scores.

The reserved asset does not replace any historical opening suite. Its legal
prefixes retain exact M1 history; every final board is distinct under all four
supported symmetries, both within the suite and against historical assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import uuid

import yaml

from . import engine
from .episode import Episode, advance_episode, start_episode
from .project_config import PROJECT_ROOT
from .project_runtime import Deadline, _provenance
from .symmetry import Symmetry


ALGORITHM = "uniform-legal-prefix-orbit-exclusion-v1"
SUITE_ID = "az-reserved-final-v1"
HISTORY_SEMANTICS = "replay-from-standard-initial-position-v1"
GENERATION_CONFIG = PROJECT_ROOT / "configs/reserved-final-v1-generation.yaml"
MANIFEST = PROJECT_ROOT / "configs/reserved-final-v1.yaml"
_SYMMETRIES = tuple(Symmetry(mirror=m, rotate_swap=r) for m in (False, True) for r in (False, True))


def _bytes(document: dict) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def board_orbit_key(state: engine.BoardState) -> str:
    """Canonical identity including side to move and player inventories."""

    return min(engine.position_key(symmetry.state(state)) for symmetry in _SYMMETRIES)


def _legacy_board(payload: dict) -> engine.BoardState:
    def walls(name: str) -> frozenset[tuple[int, int]]:
        return frozenset(tuple(int(part) for part in item.split(",")) for item in payload[name])

    return engine.BoardState(
        pawns=tuple((pawn["row"], pawn["col"]) for pawn in payload["pawns"]),
        walls_remaining=tuple(payload["wallsRemaining"]),
        horizontal_walls=walls("horizontalWalls"),
        vertical_walls=walls("verticalWalls"),
        to_play=payload["currentPlayer"],
    )


def excluded_orbits(sources: list[dict], *, deadline: Deadline | None = None) -> set[str]:
    """Hash-bind historical source bytes before extracting only board identity.

    This conversion is solely for conservative overlap exclusion. It never
    imports legacy outcomes, histories, labels, or opening performance.
    """

    result: set[str] = set()
    for source in sources:
        if deadline is not None:
            deadline.check()
        raw = (PROJECT_ROOT / source["path"]).read_bytes()
        if _sha(raw) != source["sha256"]:
            raise ValueError("historical exclusion source SHA-256 mismatch")
        for opening in json.loads(raw)["openings"]:
            if deadline is not None:
                deadline.check()
            result.add(board_orbit_key(_legacy_board(opening["state"])))
    return result


def generate_suite(
    *, seed: int, count: int, prefix_plies: int, deadline: Deadline,
    excluded: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """Generate all-or-nothing deterministic content within a caller budget."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if type(prefix_plies) is not int or not 1 <= prefix_plies < 256:
        raise ValueError("prefix_plies must be in [1, 255]")
    rng = random.Random(seed)
    seen = set(excluded)
    openings = []
    while len(openings) < count:
        deadline.check()
        episode = start_episode()
        actions: list[int] = []
        for _ in range(prefix_plies):
            deadline.check()
            legal = engine.legal_actions(episode.state)
            deadline.check()
            action = legal[rng.randrange(len(legal))]
            episode = advance_episode(episode, action, deadline=deadline)
            actions.append(action)
            if episode.finished:
                break
        if episode.finished:
            deadline.check()
            continue
        orbit = board_orbit_key(episode.state)
        if orbit in seen:
            continue
        seen.add(orbit)
        key = engine.position_key(episode.state)
        openings.append({
            "id": "reserved-" + _sha(key.encode())[:24],
            "starting_player": 0,
            "actions": actions,
            "history": list(episode.history),
            "position_key": key,
            "orbit_key": orbit,
        })
    deadline.check()
    return {
        "schema_version": 1,
        "suite_id": SUITE_ID,
        "rules_version": engine.RULES_VERSION,
        "codec_version": engine.CODEC_VERSION,
        "history_semantics": HISTORY_SEMANTICS,
        "generation": {"algorithm": ALGORITHM, "seed": seed, "count": count, "prefix_plies": prefix_plies},
        "openings": openings,
    }


def validate_suite(
    payload: dict, *, count: int, seed: int, prefix_plies: int,
    excluded: set[str] | frozenset[str] = frozenset(), deadline: Deadline | None = None,
) -> tuple[Episode, ...]:
    """Replay every action; reject stale history, duplicate and overlapping boards."""

    expected = {
        "schema_version": 1, "suite_id": SUITE_ID,
        "rules_version": engine.RULES_VERSION, "codec_version": engine.CODEC_VERSION,
        "history_semantics": HISTORY_SEMANTICS,
        "generation": {"algorithm": ALGORITHM, "seed": seed, "count": count, "prefix_plies": prefix_plies},
    }
    if set(payload) != {*expected, "openings"} or any(payload[key] != value for key, value in expected.items()):
        raise ValueError("reserved opening suite metadata mismatch")
    if type(payload["openings"]) is not list or len(payload["openings"]) != count:
        raise ValueError("reserved opening count mismatch")
    seen = set(excluded)
    ids: set[str] = set()
    episodes = []
    for opening in payload["openings"]:
        if deadline is not None:
            deadline.check()
        if set(opening) != {"id", "starting_player", "actions", "history", "position_key", "orbit_key"}:
            raise ValueError("reserved opening fields mismatch")
        if type(opening["starting_player"]) is not int or opening["starting_player"] != 0:
            raise ValueError("reserved openings must start from player zero")
        actions = opening["actions"]
        if type(actions) is not list or len(actions) != prefix_plies:
            raise ValueError("reserved opening prefix length mismatch")
        episode = start_episode()
        for action in actions:
            if deadline is not None:
                deadline.check()
            episode = advance_episode(episode, action)
            if episode.finished:
                raise ValueError("reserved opening prefix is illegal or terminal")
        key = engine.position_key(episode.state)
        orbit = board_orbit_key(episode.state)
        if opening["history"] != list(episode.history) or opening["position_key"] != key:
            raise ValueError("reserved opening history/board disagrees with replay")
        if opening["orbit_key"] != orbit:
            raise ValueError("reserved opening orbit identity mismatch")
        if opening["id"] != "reserved-" + _sha(key.encode())[:24] or opening["id"] in ids:
            raise ValueError("reserved opening ID mismatch or duplicate")
        if orbit in seen:
            raise ValueError("reserved opening duplicates or overlaps a board symmetry orbit")
        ids.add(opening["id"])
        seen.add(orbit)
        episodes.append(episode)
    if deadline is not None:
        deadline.check()
    return tuple(episodes)


def load_reserved_suite(*, manifest: Path = MANIFEST, deadline: Deadline | None = None) -> tuple[Episode, ...]:
    """Validate the frozen manifest, raw suite, exclusions, and exact histories."""

    raw = manifest.read_bytes()
    expected = manifest.with_suffix(".sha256").read_text().strip()
    if _sha(raw) != expected:
        raise ValueError("reserved manifest SHA-256 mismatch")
    document = yaml.safe_load(raw)
    if document["schema_version"] != 1 or document["suite_id"] != SUITE_ID:
        raise ValueError("reserved manifest version mismatch")
    config_raw = (PROJECT_ROOT / document["generation_config"]).read_bytes()
    if _sha(config_raw) != document["generation_config_sha256"]:
        raise ValueError("reserved generation config SHA-256 mismatch")
    config = yaml.safe_load(config_raw)
    suite_raw = (PROJECT_ROOT / document["opening_suite"]).read_bytes()
    if _sha(suite_raw) != document["opening_suite_sha256"]:
        raise ValueError("reserved opening suite SHA-256 mismatch")
    excluded = excluded_orbits(config["exclusion_sources"], deadline=deadline)
    return validate_suite(
        json.loads(suite_raw), count=config["count"], seed=config["seed"],
        prefix_plies=config["prefix_plies"], excluded=excluded, deadline=deadline,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "validate"))
    parser.add_argument("--output", type=Path, help="New output path; existing files are never overwritten")
    args = parser.parse_args()
    if args.command == "validate":
        episodes = load_reserved_suite(deadline=Deadline(180))
        print(json.dumps({"status": "validated", "openings": len(episodes)}))
        return 0
    if args.output is None:
        parser.error("generate requires --output")
    config_raw = GENERATION_CONFIG.read_bytes()
    config = yaml.safe_load(config_raw)
    deadline = Deadline(config["budget_seconds"])
    run_dir = PROJECT_ROOT / "runs/reserved-generation" / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_bytes(config_raw)
    (run_dir / "run.json").write_bytes(_bytes({
        **_provenance(), "purpose": "engineering_asset_generation_no_strength_measurement",
        "config_sha256": _sha(config_raw), "output": str(args.output.resolve()),
    }))
    def log(event: str, **fields) -> None:
        with (run_dir / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": event, "elapsed_seconds": deadline.elapsed_seconds, **fields}, sort_keys=True) + "\n")
    log("generation_started", budget_seconds=deadline.seconds)
    try:
        excluded = excluded_orbits(config["exclusion_sources"], deadline=deadline)
        payload = generate_suite(
            seed=config["seed"], count=config["count"], prefix_plies=config["prefix_plies"],
            deadline=deadline, excluded=excluded,
        )
        validate_suite(payload, count=config["count"], seed=config["seed"], prefix_plies=config["prefix_plies"], excluded=excluded, deadline=deadline)
        raw = _bytes(payload)
        deadline.check()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(raw)
        log("generation_completed", count=config["count"], opening_suite_sha256=_sha(raw))
        print(json.dumps({"status": "completed", "count": config["count"], "sha256": _sha(raw), "run_dir": str(run_dir), "elapsed_seconds": deadline.elapsed_seconds}))
        return 0
    except BaseException as error:
        log("generation_failed", error=type(error).__name__, detail=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
