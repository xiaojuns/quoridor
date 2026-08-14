#!/usr/bin/env python3
"""Generate the frozen Phase 4 paired-arena opening assets.

The generator intentionally uses only the Python reference rule engine. Every
prefix is replayed again by ``load_opening_suite`` after the files are written,
so a generated state can never bypass the opening provenance contract.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from quoridor.codec import (
    HORIZONTAL_WALL_ACTION_OFFSET,
    current_repetition_count,
    encode_move,
    encode_wall,
)
from quoridor.game import (
    WALL_GRID_SIZE,
    State,
    apply_action_index,
    create_initial_state,
    is_legal_wall,
    is_terminal,
    legal_pawn_moves,
)
from quoridor.openings import (
    Opening,
    OpeningSuite,
    build_opening_suite,
    generate_opening,
    load_opening_suite,
    opening_suite_to_payload,
    opening_position_orbit_digest,
    save_opening_suite,
    validate_suite_partition,
    validate_suite_collection,
)


SUITE_SIZE = 200
DEPTHS = (8, 12, 16, 20, 24)
STARTING_PLAYERS = (0, 1)
ACTION_STRATA = ("pawn-heavy", "wall-heavy")
OPENINGS_PER_CELL = 10
SUITE_SEEDS = {
    "dev": 0x4D45565F504834,
    "final": 0x46494E5F504834,
    "recovery-dev-v1": 0x5245434F56444556,
    "recovery-final-v1": 0x5245434F5646494E,
    "recovery-fresh-dev-v1": 0x4652455348444556,
}
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "training" / "fixtures" / "openings"
MAX_PREFIX_ATTEMPTS = 2_000


@dataclass(frozen=True, slots=True)
class OpeningSpec:
    depth: int
    starting_player: int
    action_stratum: str
    ordinal: int

    @property
    def wall_actions(self) -> int:
        if self.action_stratum == "pawn-heavy":
            return self.depth // 4
        if self.action_stratum == "wall-heavy":
            return (self.depth * 2) // 3
        raise ValueError(f"unsupported action stratum: {self.action_stratum}")


def opening_specs() -> tuple[OpeningSpec, ...]:
    specs = tuple(
        OpeningSpec(depth, starting_player, action_stratum, ordinal)
        for depth in DEPTHS
        for starting_player in STARTING_PLAYERS
        for action_stratum in ACTION_STRATA
        for ordinal in range(OPENINGS_PER_CELL)
    )
    if len(specs) != SUITE_SIZE:
        raise AssertionError("opening stratum matrix does not contain 200 entries")
    return specs


def _action_schedule(spec: OpeningSpec, rng: random.Random) -> list[bool]:
    """Return a shuffled schedule where True means wall and False means pawn."""

    schedule = [True] * spec.wall_actions
    schedule.extend([False] * (spec.depth - spec.wall_actions))
    rng.shuffle(schedule)
    return schedule


def _choose_successor(
    state: State,
    *,
    wall_action: bool,
    rng: random.Random,
) -> tuple[int, State] | None:
    if wall_action:
        candidates = [
            (encode_wall(orientation, row, col), orientation, row, col)
            for orientation in ("h", "v")
            for row in range(WALL_GRID_SIZE)
            for col in range(WALL_GRID_SIZE)
        ]
        rng.shuffle(candidates)
        actions = []
        for action, orientation, row, col in candidates:
            if is_legal_wall(state, orientation, row, col):
                actions.append(action)
                break
    else:
        actions = [encode_move(row, col) for row, col in legal_pawn_moves(state)]
        rng.shuffle(actions)
    for action in actions:
        next_state = apply_action_index(state, action)
        if is_terminal(next_state):
            continue
        # Evaluation openings contain no repeated position, including a benign
        # second occurrence. This also rules out repetition termination.
        if current_repetition_count(next_state) != 1:
            continue
        return action, next_state
    return None


def _generate_prefix(spec: OpeningSpec, rng: random.Random) -> tuple[int, ...]:
    for _ in range(MAX_PREFIX_ATTEMPTS):
        state = create_initial_state(spec.starting_player)
        actions: list[int] = []
        failed = False
        for wall_action in _action_schedule(spec, rng):
            successor = _choose_successor(
                state,
                wall_action=wall_action,
                rng=rng,
            )
            if successor is None:
                failed = True
                break
            action, state = successor
            actions.append(action)
        if not failed:
            return tuple(actions)
    raise RuntimeError(f"could not generate legal opening for stratum {spec}")


def _generate_suite(
    name: str,
    *,
    forbidden_hashes: set[str],
    forbidden_orbits: set[str] | None = None,
) -> OpeningSuite:
    seed = SUITE_SEEDS[name]
    openings: list[Opening] = []
    seen_hashes = set(forbidden_hashes)
    seen_orbits = set(forbidden_orbits or ())
    for index, spec in enumerate(opening_specs()):
        # Per-opening streams keep later assets stable if generation retries for
        # an earlier stratum change.
        rng = random.Random(seed + index * 1_000_003)
        for _ in range(MAX_PREFIX_ATTEMPTS):
            actions = _generate_prefix(spec, rng)
            opening = generate_opening(
                actions,
                suite=name,
                starting_player=spec.starting_player,
            )
            orbit = opening_position_orbit_digest(opening)
            if opening.content_hash in seen_hashes or orbit in seen_orbits:
                continue
            openings.append(opening)
            seen_hashes.add(opening.content_hash)
            seen_orbits.add(orbit)
            break
        else:
            raise RuntimeError(f"could not generate unique opening for stratum {spec}")
    return build_opening_suite(openings, suite=name)


def generate_suites() -> tuple[OpeningSuite, OpeningSuite]:
    dev = _generate_suite("dev", forbidden_hashes=set())
    final = _generate_suite(
        "final",
        forbidden_hashes={opening.content_hash for opening in dev.openings},
    )
    validate_suite_partition(dev, final)
    return dev, final


def generate_recovery_suites(
    existing: Sequence[OpeningSuite],
) -> tuple[OpeningSuite, OpeningSuite]:
    forbidden_hashes = {
        opening.content_hash for suite in existing for opening in suite.openings
    }
    forbidden_orbits = {
        opening_position_orbit_digest(opening)
        for suite in existing
        for opening in suite.openings
    }
    recovery_dev = _generate_suite(
        "recovery-dev-v1",
        forbidden_hashes=forbidden_hashes,
        forbidden_orbits=forbidden_orbits,
    )
    all_dev = (*existing, recovery_dev)
    recovery_final = _generate_suite(
        "recovery-final-v1",
        forbidden_hashes={
            opening.content_hash for suite in all_dev for opening in suite.openings
        },
        forbidden_orbits={
            opening_position_orbit_digest(opening)
            for suite in all_dev
            for opening in suite.openings
        },
    )
    validate_suite_collection(*existing, recovery_dev, recovery_final)
    return recovery_dev, recovery_final


def generate_fresh_development_suite(
    existing: Sequence[OpeningSuite],
    *,
    forbidden_orbits: set[str] | None = None,
) -> OpeningSuite:
    """Generate a deterministic development suite outside frozen evaluation data."""

    hashes = {
        opening.content_hash for suite in existing for opening in suite.openings
    }
    orbits = {
        opening_position_orbit_digest(opening)
        for suite in existing
        for opening in suite.openings
    }
    orbits.update(forbidden_orbits or ())
    fresh = _generate_suite(
        "recovery-fresh-dev-v1",
        forbidden_hashes=hashes,
        forbidden_orbits=orbits,
    )
    validate_suite_collection(*existing, fresh)
    return fresh


def _distribution(suite: OpeningSuite) -> dict[str, object]:
    depths = Counter(len(opening.actions) for opening in suite.openings)
    starters = Counter(opening.starting_player for opening in suite.openings)
    action_cells = Counter(
        (
            len(opening.actions),
            sum(
                action >= HORIZONTAL_WALL_ACTION_OFFSET
                for action in opening.actions
            ),
        )
        for opening in suite.openings
    )
    return {
        "depths": dict(sorted(depths.items())),
        "startingPlayers": dict(sorted(starters.items())),
        "depthWallCells": {
            f"{depth}:{walls}": count
            for (depth, walls), count in sorted(action_cells.items())
        },
    }


def _write_and_reload(
    dev: OpeningSuite,
    final: OpeningSuite,
    output_directory: Path,
) -> tuple[OpeningSuite, OpeningSuite]:
    output_directory.mkdir(parents=True, exist_ok=True)
    dev_path = save_opening_suite(dev, output_directory / "dev.json")
    final_path = save_opening_suite(final, output_directory / "final.json")
    dev_path.chmod(0o644)
    final_path.chmod(0o644)

    # Loading replays every action and verifies every stored state and hash.
    loaded_dev = load_opening_suite(
        output_directory / "dev.json", expected_suite="dev"
    )
    loaded_final = load_opening_suite(
        output_directory / "final.json", expected_suite="final"
    )
    validate_suite_partition(loaded_dev, loaded_final)
    if loaded_dev != dev or loaded_final != final:
        raise AssertionError("written opening assets do not round-trip exactly")
    return loaded_dev, loaded_final


def _check_frozen_assets(dev: OpeningSuite, final: OpeningSuite, directory: Path) -> None:
    frozen_dev = load_opening_suite(directory / "dev.json", expected_suite="dev")
    frozen_final = load_opening_suite(directory / "final.json", expected_suite="final")
    validate_suite_partition(frozen_dev, frozen_final)
    if opening_suite_to_payload(frozen_dev) != opening_suite_to_payload(dev):
        raise RuntimeError("frozen dev.json differs from deterministic generation")
    if opening_suite_to_payload(frozen_final) != opening_suite_to_payload(final):
        raise RuntimeError("frozen final.json differs from deterministic generation")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="asset directory (default: training/fixtures/openings)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed assets instead of replacing them",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    dev, final = generate_suites()
    if arguments.check:
        _check_frozen_assets(dev, final, arguments.output_dir)
        loaded_dev, loaded_final = dev, final
    else:
        loaded_dev, loaded_final = _write_and_reload(
            dev, final, arguments.output_dir
        )

    for suite in (loaded_dev, loaded_final):
        path = arguments.output_dir / f"{suite.name}.json"
        size = path.stat().st_size if path.exists() else 0
        print(
            f"{suite.name}: count={len(suite.openings)} "
            f"hash={suite.content_hash} bytes={size} "
            f"distribution={_distribution(suite)}"
        )
    print("dev/final: replay validation and content-disjoint partition PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
