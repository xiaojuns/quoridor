from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace
from unittest import mock

import torch

from quoridor.codec import position_hash
from quoridor.evaluator import TorchEvaluator
from quoridor.game import apply_action_index, create_initial_state
from quoridor.mcts import MCTSConfig
from quoridor.native_mcts import NativeMCTS
from quoridor.native_validation import generate_reachable_samples
from quoridor.network import (
    LEGACY_INPUT_CHANNELS,
    PHASE4_INPUT_CHANNELS,
    NetworkConfig,
    PolicyValueNetwork,
    encode_state,
)


NATIVE_AVAILABLE = importlib.util.find_spec("quoridor_native") is not None


def _phase4_smoke_config() -> NetworkConfig:
    values = NetworkConfig.smoke().to_dict()
    values["input_channels"] = PHASE4_INPUT_CHANNELS
    return NetworkConfig.from_dict(values)


@unittest.skipUnless(NATIVE_AVAILABLE, "native extension has not been built")
class NativeEncoderTests(unittest.TestCase):
    def test_random_reachable_states_match_python_bit_for_bit(self) -> None:
        import quoridor_native

        states = [
            sample.state
            for sample in generate_reachable_samples(256, seed=0xE11C0DE)
        ]
        for input_channels in (LEGACY_INPUT_CHANNELS, PHASE4_INPUT_CHANNELS):
            with self.subTest(input_channels=input_channels):
                expected = torch.stack(
                    [
                        encode_state(state, input_channels=input_channels)
                        for state in states
                    ]
                )
                actual = torch.from_numpy(
                    quoridor_native.encode_states(
                        states,
                        input_channels=input_channels,
                    )
                )
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.equal(actual, expected))

    def test_phase4_targets_use_complete_history_and_canonical_coordinates(
        self,
    ) -> None:
        import quoridor_native

        states = []
        for starting_player, action in ((0, 8 * 9 + 3), (1, 1 * 9 + 4)):
            initial = create_initial_state(starting_player=starting_player)
            child = apply_action_index(initial, action)
            for occurrence in (1, 2):
                states.append(
                    replace(
                        initial,
                        move_number=occurrence,
                        position_counts={
                            position_hash(initial): 1,
                            position_hash(child): occurrence,
                        },
                    )
                )

        expected = torch.stack(
            [
                encode_state(state, input_channels=PHASE4_INPUT_CHANNELS)
                for state in states
            ]
        )
        actual = torch.from_numpy(
            quoridor_native.encode_states(
                states,
                input_channels=PHASE4_INPUT_CHANNELS,
            )
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            tuple(actual[:, 14:16].sum(dim=(2, 3)).flatten()),
            (1, 0, 0, 1, 1, 0, 0, 1),
        )

    def test_empty_batch_and_invalid_contract_are_checked(self) -> None:
        import quoridor_native

        encoded = quoridor_native.encode_states(
            [],
            input_channels=PHASE4_INPUT_CHANNELS,
        )
        self.assertEqual(encoded.shape, (0, PHASE4_INPUT_CHANNELS, 9, 9))
        self.assertEqual(str(encoded.dtype), "float32")
        with self.assertRaisesRegex(ValueError, "14 or 16"):
            quoridor_native.encode_states([], input_channels=15)

    def test_native_mcts_enables_native_torch_encoding(self) -> None:
        evaluator = TorchEvaluator(
            PolicyValueNetwork(_phase4_smoke_config()).eval(),
            "cpu",
            use_amp=False,
        )
        NativeMCTS(
            evaluator,
            MCTSConfig(
                simulations=1,
                add_root_noise=False,
                seed=7,
            ),
        )

        self.assertIsNotNone(evaluator._native_encoder)
        with mock.patch(
            "quoridor.evaluator.encode_state",
            side_effect=AssertionError("Python encoder should not run"),
        ):
            policy, value = evaluator.evaluate(create_initial_state())
        self.assertEqual(len(policy), 209)
        self.assertTrue(-1.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()
