#include "quoridor/encoder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>

namespace quoridor {
namespace {

inline constexpr int kBoardArea = kBoardSize * kBoardSize;
inline constexpr int kDistanceScale = kBoardArea - 1;

std::size_t offset(int plane, int row, int col) {
  return static_cast<std::size_t>(plane * kBoardArea + row * kBoardSize + col);
}

std::array<int, kBoardArea> goal_distance_map(const State &state,
                                               int goal_row) {
  std::array<int, kBoardArea> distances;
  distances.fill(kBoardArea);
  std::array<Position, kBoardArea> queue;
  std::size_t head = 0;
  std::size_t tail = 0;

  for (int col = 0; col < kBoardSize; ++col) {
    distances[goal_row * kBoardSize + col] = 0;
    queue[tail++] = Position{goal_row, col};
  }

  constexpr std::array<Position, 4> directions{{
      Position{-1, 0},
      Position{1, 0},
      Position{0, -1},
      Position{0, 1},
  }};
  while (head < tail) {
    const Position current = queue[head++];
    const int current_index = current.row * kBoardSize + current.col;
    const int next_distance = distances[current_index] + 1;
    for (const Position &direction : directions) {
      const Position next{current.row + direction.row,
                          current.col + direction.col};
      if (next.row < 0 || next.row >= kBoardSize || next.col < 0 ||
          next.col >= kBoardSize) {
        continue;
      }
      const int next_index = next.row * kBoardSize + next.col;
      if (distances[next_index] <= next_distance ||
          is_passage_blocked(state, current.row, current.col, next.row,
                             next.col)) {
        continue;
      }
      distances[next_index] = next_distance;
      queue[tail++] = next;
    }
  }
  return distances;
}

void encode_into(const State &state, int input_channels,
                 std::vector<float> &encoded, std::size_t batch_offset) {
  const State canonical = canonicalize_state(state);
  auto set = [&encoded, batch_offset](int plane, int row, int col,
                                      float value) {
    encoded[batch_offset + offset(plane, row, col)] = value;
  };
  auto fill = [&encoded, batch_offset](int plane, float value) {
    const auto begin = encoded.begin() +
                       static_cast<std::ptrdiff_t>(batch_offset +
                                                   plane * kBoardArea);
    std::fill(begin, begin + kBoardArea, value);
  };

  set(0, canonical.pawns[0].row, canonical.pawns[0].col, 1.0F);
  set(1, canonical.pawns[1].row, canonical.pawns[1].col, 1.0F);
  for (const Wall &wall : canonical.horizontal_walls) {
    set(2, wall.row, wall.col, 1.0F);
  }
  for (const Wall &wall : canonical.vertical_walls) {
    set(3, wall.row, wall.col, 1.0F);
  }

  fill(4, static_cast<float>(std::clamp(canonical.walls_remaining[0], 0,
                                        kStartingWalls)) /
              static_cast<float>(kStartingWalls));
  fill(5, static_cast<float>(std::clamp(canonical.walls_remaining[1], 0,
                                        kStartingWalls)) /
              static_cast<float>(kStartingWalls));

  for (const Position &move : legal_pawn_moves(canonical)) {
    set(6, move.row, move.col, 1.0F);
  }

  for (int plane = 7; plane <= 8; ++plane) {
    const int goal_row = plane == 7 ? 0 : kBoardSize - 1;
    const auto distances = goal_distance_map(canonical, goal_row);
    for (int row = 0; row < kBoardSize; ++row) {
      for (int col = 0; col < kBoardSize; ++col) {
        const int distance = std::min(
            distances[row * kBoardSize + col], kDistanceScale);
        set(plane, row, col,
            static_cast<float>(distance) / static_cast<float>(kDistanceScale));
      }
    }
  }

  for (int col = 0; col < kBoardSize; ++col) {
    set(9, 0, col, 1.0F);
    set(10, kBoardSize - 1, col, 1.0F);
  }
  fill(11, static_cast<float>(std::clamp(canonical.move_number, 0,
                                         kMaxHalfMoves)) /
               static_cast<float>(kMaxHalfMoves));
  fill(12, static_cast<float>(std::clamp(current_repetition_count(canonical),
                                         0, kRepetitionLimit)) /
               static_cast<float>(kRepetitionLimit));
  fill(13, 1.0F);

  if (input_channels != kPhase4InputChannels) {
    return;
  }
  for (const Position &move : legal_pawn_moves(state)) {
    const int action_index = encode_move(move.row, move.col);
    const State child = apply_legal_action_index(state, action_index);
    const int canonical_index =
        to_canonical_action_index(action_index, state.current_player);
    const int row = canonical_index / kBoardSize;
    const int col = canonical_index % kBoardSize;
    if (child.termination_reason == TerminationReason::Repetition) {
      set(15, row, col, 1.0F);
    } else if (current_repetition_count(child) == 2) {
      set(14, row, col, 1.0F);
    }
  }
}

void require_input_channels(int input_channels) {
  if (input_channels != kLegacyInputChannels &&
      input_channels != kPhase4InputChannels) {
    throw std::invalid_argument("input_channels must be 14 or 16");
  }
}

} // namespace

std::vector<float> encode_state(const State &state, int input_channels) {
  require_input_channels(input_channels);
  std::vector<float> encoded(
      static_cast<std::size_t>(input_channels * kBoardArea), 0.0F);
  encode_into(state, input_channels, encoded, 0);
  return encoded;
}

std::vector<float> encode_states(const std::vector<State> &states,
                                 int input_channels) {
  require_input_channels(input_channels);
  const std::size_t state_size =
      static_cast<std::size_t>(input_channels * kBoardArea);
  std::vector<float> encoded(states.size() * state_size, 0.0F);
  for (std::size_t index = 0; index < states.size(); ++index) {
    encode_into(states[index], input_channels, encoded, index * state_size);
  }
  return encoded;
}

} // namespace quoridor
