from __future__ import annotations

import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from quoridor.codec import canonicalize_state, decode_action, position_hash
from quoridor.game import State, create_initial_state, legal_action_indices
from quoridor.evaluator import TorchEvaluator
from quoridor.network import (
    ACTION_SIZE,
    INPUT_CHANNELS,
    PHASE4_INPUT_CHANNELS,
    NetworkConfig,
    PolicyValueNetwork,
    encode_state,
    legal_action_mask,
    mask_policy_logits,
    masked_policy,
)


class NetworkTests(unittest.TestCase):
    def test_torch_evaluator_batches_states_in_one_forward_contract(self) -> None:
        evaluator = TorchEvaluator(
            PolicyValueNetwork(NetworkConfig.smoke()).eval(),
            "cpu",
        )
        results = evaluator.evaluate_batch(
            (create_initial_state(), create_initial_state(1))
        )

        self.assertEqual(len(results), 2)
        for policy, value in results:
            self.assertEqual(len(policy), ACTION_SIZE)
            self.assertTrue(-1.0 <= value <= 1.0)

    def test_config_round_trip_and_smoke_dimensions(self) -> None:
        reference = NetworkConfig()
        self.assertEqual(reference.channels, 192)
        self.assertEqual(reference.blocks, 16)
        self.assertEqual(NetworkConfig.from_dict(reference.to_dict()), reference)

        smoke = NetworkConfig.smoke()
        self.assertEqual(smoke.channels, 32)
        self.assertEqual(smoke.blocks, 2)

    def test_policy_value_shapes_and_value_range(self) -> None:
        model = PolicyValueNetwork(NetworkConfig.smoke()).eval()
        inputs = torch.randn(3, INPUT_CHANNELS, 9, 9)

        with torch.no_grad():
            policy_logits, value = model(inputs)

        self.assertEqual(tuple(policy_logits.shape), (3, ACTION_SIZE))
        self.assertEqual(tuple(value.shape), (3,))
        self.assertTrue(torch.isfinite(policy_logits).all())
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.all(value >= -1))
        self.assertTrue(torch.all(value <= 1))

    def test_batch_loss_backpropagates_through_both_heads(self) -> None:
        torch.manual_seed(7)
        model = PolicyValueNetwork(NetworkConfig.smoke()).train()
        inputs = torch.randn(4, INPUT_CHANNELS, 9, 9)
        policy_targets = torch.tensor([0, 80, 81, 208])
        value_targets = torch.tensor([1.0, -1.0, 0.5, -0.5])

        policy_logits, value = model(inputs)
        loss = F.cross_entropy(policy_logits, policy_targets) + F.mse_loss(
            value, value_targets
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.input_conv.weight.grad)
        self.assertGreater(torch.count_nonzero(model.input_conv.weight.grad), 0)
        self.assertIsNotNone(model.policy_head[-1].weight.grad)
        self.assertGreater(torch.count_nonzero(model.policy_head[-1].weight.grad), 0)
        self.assertIsNotNone(model.value_head[-2].weight.grad)
        self.assertGreater(torch.count_nonzero(model.value_head[-2].weight.grad), 0)

    def test_illegal_actions_are_masked_and_receive_zero_probability(self) -> None:
        logits = torch.arange(ACTION_SIZE, dtype=torch.float32).repeat(2, 1)
        legal_mask = torch.zeros(2, ACTION_SIZE, dtype=torch.bool)
        legal_mask[0, [0, 80, 208]] = True

        masked_logits = mask_policy_logits(logits, legal_mask)
        probabilities = masked_policy(logits, legal_mask)

        self.assertTrue(
            torch.equal(masked_logits[0, legal_mask[0]], logits[0, legal_mask[0]])
        )
        self.assertTrue(torch.isneginf(masked_logits[0, ~legal_mask[0]]).all())
        self.assertEqual(torch.count_nonzero(probabilities[0, ~legal_mask[0]]), 0)
        self.assertAlmostEqual(probabilities[0].sum().item(), 1.0)
        self.assertEqual(probabilities[0].argmax().item(), 208)
        self.assertEqual(torch.count_nonzero(probabilities[1]), 0)

    def test_network_rejects_non_batch_or_wrong_channel_input(self) -> None:
        model = PolicyValueNetwork(NetworkConfig.smoke())
        with self.assertRaisesRegex(ValueError, "inputs must have shape"):
            model(torch.randn(INPUT_CHANNELS, 9, 9))
        with self.assertRaisesRegex(ValueError, "inputs must have shape"):
            model(torch.randn(2, INPUT_CHANNELS - 1, 9, 9))

    def test_encode_initial_state_has_fixed_plane_semantics(self) -> None:
        state = create_initial_state()
        encoded = encode_state(state)

        self.assertEqual(tuple(encoded.shape), (INPUT_CHANNELS, 9, 9))
        self.assertEqual(encoded.dtype, torch.float32)
        self.assertEqual(encoded[0, 8, 4], 1)
        self.assertEqual(encoded[1, 0, 4], 1)
        self.assertEqual(encoded[0].sum(), 1)
        self.assertEqual(encoded[1].sum(), 1)
        self.assertEqual(encoded[2].sum(), 0)
        self.assertEqual(encoded[3].sum(), 0)
        self.assertTrue(torch.all(encoded[4] == 1))
        self.assertTrue(torch.all(encoded[5] == 1))
        self.assertEqual(encoded[6].sum(), 3)
        self.assertEqual(encoded[6, 7, 4], 1)
        self.assertEqual(encoded[6, 8, 3], 1)
        self.assertEqual(encoded[6, 8, 5], 1)
        self.assertTrue(torch.all(encoded[7, 0] == 0))
        self.assertTrue(torch.all(encoded[8, 8] == 0))
        self.assertTrue(torch.all(encoded[9, 0] == 1))
        self.assertTrue(torch.all(encoded[10, 8] == 1))
        self.assertEqual(encoded[11].sum(), 0)
        self.assertTrue(torch.all(encoded[12] == 1 / 3))
        self.assertTrue(torch.all(encoded[13] == 1))

    def test_encode_state_canonicalizes_player_and_history_features(self) -> None:
        state = State(
            pawns=((6, 2), (1, 5)),
            walls_remaining=(7, 4),
            horizontal_walls=frozenset({(2, 3)}),
            vertical_walls=frozenset({(5, 1)}),
            current_player=1,
            move_number=128,
        )
        state = replace(state, position_counts={position_hash(state): 2})
        canonical = canonicalize_state(state)
        encoded = encode_state(state)

        self.assertEqual(encoded[0, 7, 3], 1)
        self.assertEqual(encoded[1, 2, 6], 1)
        self.assertEqual(encoded[2, 5, 4], 1)
        self.assertEqual(encoded[3, 2, 6], 1)
        self.assertTrue(torch.all(encoded[4] == 0.4))
        self.assertTrue(torch.all(encoded[5] == 0.7))
        self.assertTrue(torch.all(encoded[11] == 0.5))
        self.assertTrue(torch.all(encoded[12] == 2 / 3))

        expected_targets = {
            (action["row"], action["col"])
            for index in legal_action_indices(canonical)
            if (action := decode_action(index))["type"] == "move"
        }
        actual_targets = {
            tuple(position)
            for position in torch.nonzero(encoded[6], as_tuple=False).tolist()
        }
        self.assertEqual(actual_targets, expected_targets)

        expected_mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
        expected_mask[list(legal_action_indices(canonical))] = True
        self.assertTrue(torch.equal(legal_action_mask(state), expected_mask))

    def test_phase4_encoding_marks_second_and_losing_third_occurrences(self) -> None:
        initial = create_initial_state()
        action = 8 * 9 + 3
        child = replace(initial, position_counts=dict(initial.position_counts))
        from quoridor.game import apply_action_index

        child = apply_action_index(child, action)
        child_hash = position_hash(child)

        second_state = replace(
            initial,
            move_number=1,
            position_counts={
                position_hash(initial): 1,
                child_hash: 1,
            },
        )
        second = encode_state(
            second_state,
            input_channels=PHASE4_INPUT_CHANNELS,
        )
        self.assertEqual(tuple(second.shape), (PHASE4_INPUT_CHANNELS, 9, 9))
        self.assertEqual(second[14, 8, 3], 1)
        self.assertEqual(second[15].sum(), 0)

        third_state = replace(
            second_state,
            move_number=2,
            position_counts={
                position_hash(initial): 1,
                child_hash: 2,
            },
        )
        third = encode_state(
            third_state,
            input_channels=PHASE4_INPUT_CHANNELS,
        )
        self.assertEqual(third[14].sum(), 0)
        self.assertEqual(third[15, 8, 3], 1)

    def test_phase4_repetition_targets_use_canonical_coordinates(self) -> None:
        initial = create_initial_state(starting_player=1)
        action = 1 * 9 + 4
        from quoridor.game import apply_action_index

        child = apply_action_index(initial, action)
        state = replace(
            initial,
            move_number=1,
            position_counts={
                position_hash(initial): 1,
                position_hash(child): 2,
            },
        )
        encoded = encode_state(state, input_channels=PHASE4_INPUT_CHANNELS)
        canonical_action = 7 * 9 + 4
        row, col = divmod(canonical_action, 9)
        self.assertEqual(encoded[15, row, col], 1)


if __name__ == "__main__":
    unittest.main()
