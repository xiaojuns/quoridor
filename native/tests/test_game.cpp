#include "quoridor/game.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using quoridor::Action;
using quoridor::Orientation;
using quoridor::Position;
using quoridor::State;
using quoridor::TerminationReason;

void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void check_invalid(Function function, const std::string &message) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(message);
}

bool has_position(const std::vector<Position> &positions, Position expected) {
  return std::find(positions.begin(), positions.end(), expected) !=
         positions.end();
}

bool has_action(const std::vector<int> &actions, int expected) {
  return std::find(actions.begin(), actions.end(), expected) != actions.end();
}

State with_current_repetition(State state, int count) {
  state.position_counts = {{quoridor::position_hash(state), count}};
  return state;
}

void test_action_codec() {
  check(quoridor::kActionSpaceSize == 209,
        "action space must contain 209 actions");
  check(quoridor::encode_move(0, 0) == 0, "first move action mismatch");
  check(quoridor::encode_move(8, 8) == 80, "last move action mismatch");
  check(quoridor::encode_wall('h', 0, 0) == 81,
        "first horizontal wall mismatch");
  check(quoridor::encode_wall('h', 7, 7) == 144,
        "last horizontal wall mismatch");
  check(quoridor::encode_wall('v', 0, 0) == 145,
        "first vertical wall mismatch");
  check(quoridor::encode_wall('v', 7, 7) == 208, "last vertical wall mismatch");

  for (int index = 0; index < quoridor::kActionSpaceSize; ++index) {
    check(quoridor::encode_action(quoridor::decode_action(index)) == index,
          "action round-trip mismatch");
    check(quoridor::mirror_action_index(quoridor::mirror_action_index(index)) ==
              index,
          "mirror action must be an involution");
    const int canonical = quoridor::to_canonical_action_index(index, 1);
    check(quoridor::from_canonical_action_index(canonical, 1) == index,
          "canonical action must round-trip");
    check(quoridor::to_canonical_action_index(index, 0) == index,
          "player zero action must already be canonical");
  }

  check_invalid([] { quoridor::decode_action(-1); },
                "negative action must be rejected");
  check_invalid([] { quoridor::decode_action(209); },
                "out-of-range action must be rejected");
  check_invalid([] { quoridor::encode_move(9, 0); },
                "out-of-range move must be rejected");
  check_invalid([] { quoridor::encode_wall('x', 0, 0); },
                "unknown wall orientation must be rejected");
}

void test_initial_state_and_fixture_hashes() {
  const State initial = quoridor::create_initial_state();
  check(quoridor::position_hash(initial) == "[1,0,8,4,0,4,10,10,[],[]]",
        "initial position hash mismatch");
  check(quoridor::current_repetition_count(initial) == 1,
        "initial position must be counted once");
  const std::vector<int> legal = quoridor::legal_action_indices(initial);
  check(legal.size() == 131, "initial legal action count mismatch");
  check(legal[0] == quoridor::encode_move(7, 4) &&
            legal[1] == quoridor::encode_move(8, 3) &&
            legal[2] == quoridor::encode_move(8, 5),
        "initial pawn actions mismatch");
  check(!quoridor::is_terminal(initial), "initial state must not be terminal");
  check(!quoridor::winner_value(initial).has_value(),
        "initial state must not have a winner value");

  State fixture;
  fixture.walls_remaining = {{9, 9}};
  fixture.vertical_walls = {{Position{4, 2}, Position{4, 5}}};
  fixture.current_player = 0;
  fixture.move_number = 2;
  fixture = with_current_repetition(fixture, 1);
  check(quoridor::position_hash(fixture) ==
            "[1,0,8,4,0,4,9,9,[],[\"4,2\",\"4,5\"]]",
        "reachable-002 position hash mismatch");
  check(quoridor::legal_action_indices(fixture).size() == 123,
        "reachable-002 legal action count mismatch");
  check(has_action(quoridor::legal_action_indices(fixture), 142),
        "reachable-002 sample action must be legal");

  const State next = quoridor::apply_action_index(fixture, 142);
  check(quoridor::position_hash(next) ==
            "[1,1,8,4,0,4,8,9,[\"7,5\"],[\"4,2\",\"4,5\"]]",
        "reachable-002 next hash mismatch");
  check(quoridor::position_hash(quoridor::canonicalize_state(next)) ==
            "[1,0,8,4,0,4,9,8,[\"0,2\"],[\"3,2\",\"3,5\"]]",
        "reachable-002 canonical next hash mismatch");
}

void test_pawn_movement_and_walls() {
  State open_jump;
  open_jump.pawns = {{Position{4, 4}, Position{3, 4}}};
  const std::vector<Position> open_moves =
      quoridor::legal_pawn_moves(open_jump);
  check(has_position(open_moves, Position{2, 4}),
        "pawn must jump over an adjacent opponent");
  check(!has_position(open_moves, Position{3, 4}),
        "pawn cannot land on the opponent");

  State blocked_jump = open_jump;
  blocked_jump.horizontal_walls = {{Position{2, 4}}};
  const std::vector<Position> blocked_moves =
      quoridor::legal_pawn_moves(blocked_jump);
  check(!has_position(blocked_moves, Position{2, 4}),
        "blocked jump must not be legal");
  check(has_position(blocked_moves, Position{3, 3}) &&
            has_position(blocked_moves, Position{3, 5}),
        "blocked jump must allow both diagonals");

  State wall_state;
  wall_state.horizontal_walls = {{Position{3, 3}}};
  check(quoridor::is_passage_blocked(wall_state, 3, 3, 4, 3),
        "horizontal wall must block its left segment");
  check(quoridor::is_passage_blocked(wall_state, 3, 4, 4, 4),
        "horizontal wall must block its right segment");
  check(!quoridor::is_passage_blocked(wall_state, 3, 5, 4, 5),
        "horizontal wall must not block a third segment");
  check(!quoridor::is_wall_structurally_valid(wall_state, 'h', 3, 4),
        "overlapping horizontal wall must be invalid");
  check(!quoridor::is_wall_structurally_valid(wall_state, 'v', 3, 3),
        "crossing wall must be invalid");
  check(quoridor::is_wall_structurally_valid(wall_state, 'v', 3, 5),
        "separate vertical wall must be structurally valid");
  check(quoridor::shortest_path_distance(quoridor::create_initial_state(), 0) ==
            8.0,
        "player zero initial distance mismatch");
  check(quoridor::shortest_path_distance(quoridor::create_initial_state(), 1) ==
            8.0,
        "player one initial distance mismatch");
}

void test_apply_and_terminal_rules() {
  const State initial = quoridor::create_initial_state();
  const State moved =
      quoridor::apply_action_index(initial, quoridor::encode_move(7, 4));
  check(quoridor::apply_legal_action_index(
            initial, quoridor::encode_move(7, 4)) == moved,
        "prevalidated transition must match checked apply");
  check(initial.pawns[0] == Position{8, 4},
        "apply must not mutate its source state");
  check(moved.pawns[0] == Position{7, 4} && moved.current_player == 1 &&
            moved.move_number == 1,
        "pawn transition fields mismatch");
  check_invalid(
      [&initial] {
        quoridor::apply_action_index(initial, quoridor::encode_move(0, 0));
      },
      "illegal pawn move must be rejected");

  State repetition = initial;
  const std::vector<int> cycle{
      quoridor::encode_move(8, 3),
      quoridor::encode_move(0, 3),
      quoridor::encode_move(8, 4),
      quoridor::encode_move(0, 4),
  };
  for (int action : cycle) {
    repetition = quoridor::apply_action_index(repetition, action);
  }
  check(!quoridor::is_terminal(repetition),
        "second occurrence must not be terminal");
  check(quoridor::current_repetition_count(repetition) == 2,
        "cycle must produce a second occurrence");
  for (int action : cycle) {
    repetition = quoridor::apply_action_index(repetition, action);
  }
  check(repetition.winner == 0 &&
            repetition.termination_reason == TerminationReason::Repetition,
        "mover must lose on the third occurrence");
  check(quoridor::winner_value(repetition).value() == 1.0,
        "side to move must own a repetition win");
  check(quoridor::winner_value(repetition, 1).value() == -1.0,
        "repetition mover must receive negative value");
  check(quoridor::legal_action_indices(repetition).empty(),
        "terminal state must have no legal action");

  State move_limit = initial;
  move_limit.move_number = quoridor::kMaxHalfMoves - 1;
  const State limit_result =
      quoridor::apply_action_index(move_limit, quoridor::encode_move(8, 3));
  check(limit_result.winner == 1 &&
            limit_result.termination_reason == TerminationReason::MoveLimit,
        "256th half-move must be a loss for the mover");

  State goal;
  goal.pawns = {{Position{1, 3}, Position{0, 4}}};
  goal.move_number = quoridor::kMaxHalfMoves - 1;
  goal = with_current_repetition(goal, 1);
  const State goal_result =
      quoridor::apply_action_index(goal, quoridor::encode_move(0, 3));
  check(goal_result.winner == 0 &&
            goal_result.termination_reason == TerminationReason::Goal,
        "goal must take priority over the move limit");
  check(quoridor::winner_value(goal_result).value() == -1.0,
        "post-goal side to move must receive negative value");

  const State probe =
      quoridor::apply_action_index(initial, quoridor::encode_move(8, 3));
  State repetition_at_limit = initial;
  repetition_at_limit.move_number = quoridor::kMaxHalfMoves - 1;
  repetition_at_limit.position_counts.set(quoridor::position_hash(probe), 2);
  const State priority_result = quoridor::apply_action_index(
      repetition_at_limit, quoridor::encode_move(8, 3));
  check(priority_result.winner == 1 &&
            priority_result.termination_reason == TerminationReason::Repetition,
        "repetition must take priority over the move limit");
}

void test_symmetries() {
  State state;
  state.current_player = 1;
  state.walls_remaining = {{9, 10}};
  state.vertical_walls = {{Position{4, 5}}};
  state.move_number = 1;
  state = with_current_repetition(state, 2);

  const State canonical = quoridor::canonicalize_state(state);
  check(canonical.current_player == 0,
        "canonical state must have player zero to move");
  check(quoridor::position_hash(canonical) == "[1,0,8,4,0,4,10,9,[],[\"3,2\"]]",
        "canonical fixture hash mismatch");
  check(quoridor::current_repetition_count(canonical) == 2,
        "canonical state must preserve current repetition count");
  check(quoridor::to_canonical_action_index(86, 1) == 139,
        "fixture canonical action mismatch");
  check(quoridor::mirror_action_index(86) == 83,
        "fixture mirrored action mismatch");

  const State mirrored = quoridor::mirror_state(state);
  check(quoridor::position_hash(mirrored) == "[1,1,8,4,0,4,9,10,[],[\"4,2\"]]",
        "mirrored fixture hash mismatch");
  check(quoridor::mirror_state(mirrored) == state,
        "mirror state must be an involution for serialized history");

  State history = quoridor::create_initial_state(1);
  history = quoridor::apply_action_index(
      history, quoridor::encode_wall('v', 4, 5));
  history = quoridor::apply_action_index(
      history, quoridor::encode_wall('h', 2, 1));
  const std::vector<int> cycle{
      quoridor::encode_move(1, 4),
      quoridor::encode_move(7, 4),
      quoridor::encode_move(0, 4),
      quoridor::encode_move(8, 4),
  };
  for (int action : cycle) {
    history = quoridor::apply_action_index(history, action);
  }
  const auto raw_counts = history.position_counts.materialize();
  check(history.current_player == 1 && raw_counts.size() == 6,
        "history fixture must contain multiple player-one positions");

  const State canonical_history = quoridor::canonicalize_state(history);
  const auto canonical_counts = canonical_history.position_counts.materialize();
  check(canonical_counts.size() == raw_counts.size(),
        "canonicalization must preserve every history entry");
  check(canonical_counts.at("[1,0,8,4,0,4,10,10,[],[]]") == 1,
        "player-one history must rotate and swap the initial position");
  check(canonical_counts.at(
            "[1,1,8,4,0,4,9,10,[],[\"3,2\"]]") == 1,
        "player-one history must transform historical wall positions");
  check(canonical_counts.at(
            "[1,0,8,4,0,4,9,9,[\"5,6\"],[\"3,2\"]]") == 2,
        "player-one history must preserve transformed occurrence counts");

  const State role_swapped = quoridor::rotate_and_swap_state(history);
  check(quoridor::rotate_and_swap_state(role_swapped) == history,
        "player-role transform must be an involution");
  check(quoridor::canonicalize_state(role_swapped) == canonical_history,
        "player-role transform must collapse to the same canonical state");
  const auto raw_legal = quoridor::legal_action_indices(history);
  const auto swapped_legal = quoridor::legal_action_indices(role_swapped);
  std::vector<int> mapped_legal;
  mapped_legal.reserve(raw_legal.size());
  for (const int action : raw_legal) {
    mapped_legal.push_back(quoridor::to_canonical_action_index(action, 1));
  }
  std::sort(mapped_legal.begin(), mapped_legal.end());
  auto sorted_swapped = swapped_legal;
  std::sort(sorted_swapped.begin(), sorted_swapped.end());
  check(mapped_legal == sorted_swapped,
        "player-role transform must preserve the legal action set");
  const int probe_action = raw_legal.front();
  check(quoridor::rotate_and_swap_state(
            quoridor::apply_action_index(history, probe_action)) ==
            quoridor::apply_action_index(
                role_swapped,
                quoridor::to_canonical_action_index(probe_action, 1)),
        "player-role transform must commute with legal transitions");

  const State mirrored_history = quoridor::mirror_state(history);
  check(mirrored_history.position_counts.materialize().size() ==
            raw_counts.size(),
        "mirror must preserve every history entry");
  check(quoridor::mirror_state(mirrored_history) == history,
        "mirror must be an involution for complete repetition history");
}

} // namespace

int main() {
  try {
    test_action_codec();
    test_initial_state_and_fixture_hashes();
    test_pawn_movement_and_walls();
    test_apply_and_terminal_rules();
    test_symmetries();
  } catch (const std::exception &error) {
    std::cerr << "native game test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "native game tests passed\n";
  return 0;
}
