from __future__ import annotations

import unittest

from quoridor.codec import position_hash, state_to_serializable
from quoridor.game import create_initial_state, legal_action_indices
from quoridor.phase4f29_fresh_screen import _self_hashed
from quoridor.phase4f42_curriculum_dataset import _validate_sample
from quoridor.run_manifest import sha256_bytes


class Phase4F42CurriculumDatasetTests(unittest.TestCase):
    def _sample(self) -> dict[str, object]:
        state = create_initial_state()
        actions = legal_action_indices(state)[:2]
        core = {
            "schemaVersion": 1,
            "kind": "phase4f42-curriculum-target-sample",
            "contractSha256": "a" * 64,
            "ordinal": 0,
            "split": "train",
            "band": "early",
            "source": {
                "gameId": "test-game",
                "plyIndex": 0,
                "teacherActionIndex": actions[0],
                "teacherOrbitSha256": "b" * 64,
                "positionSha256": sha256_bytes(position_hash(state).encode("ascii")),
            },
            "state": state_to_serializable(state),
            "policy": [[actions[0], 0.75], [actions[1], 0.25]],
            "value": 0.2,
            "valueComponents": {
                "terminalOutcome": 1.0,
                "regularizedSearch": -0.2,
                "pathTempo": 0.0,
            },
        }
        return _self_hashed(core, "sampleSha256")

    def test_valid_soft_target_sample_passes(self) -> None:
        _validate_sample(self._sample(), "a" * 64)

    def test_policy_must_be_normalized(self) -> None:
        sample = self._sample()
        core = dict(sample)
        core.pop("sampleSha256")
        core["policy"] = [[core["policy"][0][0], 0.5], [core["policy"][1][0], 0.25]]
        invalid = _self_hashed(core, "sampleSha256")
        with self.assertRaisesRegex(Exception, "not normalized"):
            _validate_sample(invalid, "a" * 64)

    def test_policy_action_must_be_legal(self) -> None:
        sample = self._sample()
        core = dict(sample)
        core.pop("sampleSha256")
        legal = set(legal_action_indices(create_initial_state()))
        illegal = next(action for action in range(209) if action not in legal)
        core["policy"] = [[illegal, 1.0]]
        invalid = _self_hashed(core, "sampleSha256")
        with self.assertRaisesRegex(Exception, "invalid action"):
            _validate_sample(invalid, "a" * 64)


if __name__ == "__main__":
    unittest.main()
