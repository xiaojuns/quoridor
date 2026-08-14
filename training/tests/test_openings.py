from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from quoridor.codec import (
    HORIZONTAL_WALL_ACTION_OFFSET,
    current_repetition_count,
    encode_move,
    position_hash,
)
from quoridor.openings import (
    PAIRING_CONTRACT,
    OpeningValidationError,
    build_opening_suite,
    generate_opening,
    load_opening_suite,
    opening_suite_to_payload,
    save_opening_suite,
    state_from_opening_payload,
    state_to_opening_payload,
    validate_opening,
    validate_suite_partition,
    validate_suite_collection,
)


def cycle_to_initial() -> tuple[int, ...]:
    return (
        encode_move(8, 3),
        encode_move(0, 3),
        encode_move(8, 4),
        encode_move(0, 4),
    )


FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "training" / "fixtures" / "openings"
)
FROZEN_SUITE_HASHES = {
    "dev": "sha256:9cde8e3302916d023957b60d4a433f31f0dfc995881389aca95daee4730b11c4",
    "final": "sha256:65f1bb5223e8e7c9da469b69b891443458d631e8058cfb4a14f57f3d1243833d",
    "recovery-dev-v1": "sha256:03c5170119b4bacbc696e2afe9767e51d4d3adde8f7c98682877ecfd2662f028",
    "recovery-final-v1": "sha256:1c53af85b154ef84539bc1305f77a503a0b0090db47accd3a72b8f4a80ae1d9a",
}


class OpeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dev = load_opening_suite(FIXTURE_DIRECTORY / "dev.json", expected_suite="dev")
        cls.final = load_opening_suite(FIXTURE_DIRECTORY / "final.json", expected_suite="final")

    def test_generation_preserves_full_history_and_defines_swapped_legs(self) -> None:
        opening = generate_opening(cycle_to_initial(), suite="dev")

        self.assertEqual(opening.state.move_number, 4)
        self.assertEqual(sum(opening.state.position_counts.values()), 5)
        self.assertEqual(current_repetition_count(opening.state), 2)
        self.assertEqual(opening.state.position_counts[position_hash(opening.state)], 2)

        payload = state_to_opening_payload(opening.state)
        self.assertEqual(len(payload["positionCounts"]), 4)
        restored = state_from_opening_payload(payload)
        self.assertEqual(state_to_opening_payload(restored), payload)

        first, second = opening.paired_legs()
        self.assertEqual(first.opening_id, opening.id)
        self.assertEqual(second.opening_id, opening.id)
        self.assertEqual((first.leg, second.leg), (0, 1))
        self.assertEqual(
            {first.candidate_player, second.candidate_player},
            {0, 1},
        )
        self.assertTrue(first.candidate_to_move)
        self.assertFalse(second.candidate_to_move)
        self.assertEqual(first.candidate_player, opening.state.current_player)
        self.assertEqual(second.opponent_player, opening.state.current_player)

    def test_ids_and_suite_hashes_are_content_stable(self) -> None:
        first = generate_opening(cycle_to_initial(), suite="dev")
        repeated = generate_opening(cycle_to_initial(), suite="dev")
        second = generate_opening(
            (encode_move(7, 4), encode_move(1, 4)),
            suite="dev",
        )

        self.assertEqual(first.id, repeated.id)
        self.assertEqual(first.content_hash, repeated.content_hash)
        forward = build_opening_suite((first, second), suite="dev")
        reverse = build_opening_suite((second, first), suite="dev")
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.pairing_contract, PAIRING_CONTRACT)

    def test_suite_save_load_replays_every_action_and_checks_hashes(self) -> None:
        suite = build_opening_suite(
            (generate_opening(cycle_to_initial(), suite="dev"),),
            suite="dev",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev-openings.json"
            save_opening_suite(suite, path)
            loaded = load_opening_suite(path, expected_suite="dev")
            self.assertEqual(loaded, suite)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["suiteHash"] = "sha256:" + "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                OpeningValidationError, "suite hash does not match"
            ):
                load_opening_suite(path)

    def test_validation_rejects_truncated_history_and_illegal_prefixes(self) -> None:
        opening = generate_opening(cycle_to_initial(), suite="dev")
        truncated = replace(
            opening.state,
            position_counts={position_hash(opening.state): 2},
        )
        with self.assertRaises(OpeningValidationError):
            validate_opening(replace(opening, state=truncated))

        with self.assertRaisesRegex(OpeningValidationError, "ply 0"):
            generate_opening((encode_move(0, 0),), suite="dev")

        payload = opening_suite_to_payload(
            build_opening_suite((opening,), suite="dev")
        )
        self.assertEqual(payload["suite"], "dev")
        self.assertIn("positionCounts", payload["openings"][0]["state"])

        with self.assertRaises(OpeningValidationError):
            generate_opening((), suite=[])  # type: ignore[arg-type]

    def test_dev_and_final_must_be_content_disjoint(self) -> None:
        shared_dev = generate_opening(cycle_to_initial(), suite="dev")
        shared_final = generate_opening(cycle_to_initial(), suite="final")
        dev = build_opening_suite((shared_dev,), suite="dev")
        final_with_leak = build_opening_suite((shared_final,), suite="final")
        with self.assertRaisesRegex(OpeningValidationError, "content-disjoint"):
            validate_suite_partition(dev, final_with_leak)

        distinct_final = build_opening_suite(
            (
                generate_opening(
                    (encode_move(7, 4), encode_move(1, 4)),
                    suite="final",
                ),
            ),
            suite="final",
        )
        validate_suite_partition(dev, distinct_final)


class FrozenOpeningAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dev = load_opening_suite(
            FIXTURE_DIRECTORY / "dev.json", expected_suite="dev"
        )
        cls.final = load_opening_suite(
            FIXTURE_DIRECTORY / "final.json", expected_suite="final"
        )

    def test_frozen_hashes_counts_and_partition(self) -> None:
        self.assertEqual(self.dev.content_hash, FROZEN_SUITE_HASHES["dev"])
        self.assertEqual(self.final.content_hash, FROZEN_SUITE_HASHES["final"])
        self.assertEqual(len(self.dev.openings), 200)
        self.assertEqual(len(self.final.openings), 200)
        validate_suite_partition(self.dev, self.final)

    def test_recovery_suites_are_content_and_mirror_orbit_disjoint(self) -> None:
        recovery_root = FIXTURE_DIRECTORY / "recovery-v1"
        recovery_dev = load_opening_suite(
            recovery_root / "recovery-dev-v1.json",
            expected_suite="recovery-dev-v1",
        )
        recovery_final = load_opening_suite(
            recovery_root / "recovery-final-v1.json",
            expected_suite="recovery-final-v1",
        )
        self.assertEqual(
            recovery_dev.content_hash,
            FROZEN_SUITE_HASHES["recovery-dev-v1"],
        )
        self.assertEqual(
            recovery_final.content_hash,
            FROZEN_SUITE_HASHES["recovery-final-v1"],
        )
        validate_suite_collection(self.dev, self.final, recovery_dev, recovery_final)

    def test_depth_and_starting_player_strata_are_balanced(self) -> None:
        expected_depths = {8: 40, 12: 40, 16: 40, 20: 40, 24: 40}
        expected_starters = {0: 100, 1: 100}
        for suite in (self.dev, self.final):
            with self.subTest(suite=suite.name):
                self.assertEqual(
                    Counter(len(opening.actions) for opening in suite.openings),
                    expected_depths,
                )
                self.assertEqual(
                    Counter(opening.starting_player for opening in suite.openings),
                    expected_starters,
                )
                self.assertEqual(
                    Counter(
                        (len(opening.actions), opening.starting_player)
                        for opening in suite.openings
                    ),
                    Counter(
                        {
                            (depth, starting_player): 20
                            for depth in expected_depths
                            for starting_player in (0, 1)
                        }
                    ),
                )

    def test_pawn_and_wall_action_strata_have_complete_clean_histories(self) -> None:
        expected_wall_counts = {
            (8, 2): 20,
            (8, 5): 20,
            (12, 3): 20,
            (12, 8): 20,
            (16, 4): 20,
            (16, 10): 20,
            (20, 5): 20,
            (20, 13): 20,
            (24, 6): 20,
            (24, 16): 20,
        }
        for suite in (self.dev, self.final):
            cells = Counter()
            stratified_cells = Counter()
            for opening in suite.openings:
                depth = len(opening.actions)
                wall_count = sum(
                    action >= HORIZONTAL_WALL_ACTION_OFFSET
                    for action in opening.actions
                )
                cells[(depth, wall_count)] += 1
                stratified_cells[(depth, opening.starting_player, wall_count)] += 1
                self.assertEqual(opening.state.move_number, depth)
                self.assertEqual(sum(opening.state.position_counts.values()), depth + 1)
                self.assertTrue(
                    all(count == 1 for count in opening.state.position_counts.values())
                )
                self.assertIsNone(opening.state.winner)
                self.assertIsNone(opening.state.termination_reason)
            with self.subTest(suite=suite.name):
                self.assertEqual(cells, expected_wall_counts)
                self.assertEqual(
                    stratified_cells,
                    Counter(
                        {
                            (depth, starting_player, wall_count): count // 2
                            for (depth, wall_count), count in expected_wall_counts.items()
                            for starting_player in (0, 1)
                        }
                    ),
                )


if __name__ == "__main__":
    unittest.main()
