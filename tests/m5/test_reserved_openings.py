"""Reserved asset integrity and exact new-rule history, not playing strength."""

from copy import deepcopy
import hashlib
import json

import pytest
import yaml

from quoridor import engine
from quoridor.project_runtime import BudgetExpired, Deadline
from quoridor.reserved_openings import (
    GENERATION_CONFIG, MANIFEST, board_orbit_key, excluded_orbits,
    generate_suite, load_reserved_suite, validate_suite,
)
from quoridor.symmetry import Symmetry


def test_full_frozen_reserved_suite_replays_200_exact_histories():
    episodes = load_reserved_suite(deadline=Deadline(180))
    assert len(episodes) == 200
    assert all(not episode.finished and episode.plies == 16 for episode in episodes)
    assert all(len(episode.history) == 17 for episode in episodes)
    assert all(episode.history[0] == engine.position_key(engine.initial_state()) for episode in episodes)
    assert len({board_orbit_key(episode.state) for episode in episodes}) == 200
    excluded = excluded_orbits(yaml.safe_load(GENERATION_CONFIG.read_bytes())["exclusion_sources"])
    assert not {board_orbit_key(episode.state) for episode in episodes} & excluded


def test_seeded_generation_is_exactly_reproducible_and_symmetry_identity_is_invariant():
    kwargs = {"seed": 12345, "count": 3, "prefix_plies": 2}
    first = generate_suite(**kwargs, deadline=Deadline(20))
    second = generate_suite(**kwargs, deadline=Deadline(20))
    assert first == second
    episodes = validate_suite(first, **kwargs)
    for episode in episodes:
        for mirror in (False, True):
            for rotate in (False, True):
                transformed = Symmetry(mirror=mirror, rotate_swap=rotate).state(episode.state)
                assert board_orbit_key(transformed) == board_orbit_key(episode.state)


@pytest.mark.parametrize("mutation", ["duplicate", "overlap", "history", "action"])
def test_reserved_content_rejects_duplicate_overlap_and_invalid_history(mutation):
    kwargs = {"seed": 9, "count": 2, "prefix_plies": 2}
    payload = generate_suite(**kwargs, deadline=Deadline(20))
    excluded = set()
    if mutation == "duplicate":
        payload["openings"][1] = deepcopy(payload["openings"][0])
    elif mutation == "overlap":
        excluded.add(payload["openings"][0]["orbit_key"])
    elif mutation == "history":
        payload["openings"][0]["history"] = payload["openings"][0]["history"][-1:]
    else:
        payload["openings"][0]["actions"][0] = -1
    with pytest.raises(ValueError):
        validate_suite(payload, **kwargs, excluded=excluded)


def test_reserved_manifest_byte_change_is_rejected(tmp_path):
    changed = tmp_path / "reserved.yaml"
    changed.write_bytes(MANIFEST.read_bytes() + b"\n")
    changed.with_suffix(".sha256").write_bytes(MANIFEST.with_suffix(".sha256").read_bytes())
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_reserved_suite(manifest=changed)


def test_reserved_suite_byte_change_is_rejected(tmp_path):
    document = yaml.safe_load(MANIFEST.read_bytes())
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"openings": []}))
    document["opening_suite"] = str(broken)
    changed = tmp_path / "reserved.yaml"
    raw = yaml.safe_dump(document).encode()
    changed.write_bytes(raw)
    changed.with_suffix(".sha256").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="opening suite SHA-256"):
        load_reserved_suite(manifest=changed)


def test_old_source_digest_changes_are_rejected(tmp_path):
    broken = tmp_path / "old.json"
    broken.write_text('{"openings": []}')
    with pytest.raises(ValueError, match="exclusion source SHA-256"):
        excluded_orbits([{"path": str(broken), "sha256": "0" * 64}])


def test_expired_generation_never_returns_a_partial_suite():
    ticks = iter((0.0, 1.0))
    deadline = Deadline(0.1, clock=lambda: next(ticks))
    with pytest.raises(BudgetExpired):
        generate_suite(seed=1, count=200, prefix_plies=16, deadline=deadline)
