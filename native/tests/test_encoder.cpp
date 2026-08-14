#include "quoridor/encoder.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using quoridor::State;

constexpr int kBoardArea = quoridor::kBoardSize * quoridor::kBoardSize;

std::size_t offset(int plane, int row, int col) {
  return static_cast<std::size_t>(plane * kBoardArea +
                                  row * quoridor::kBoardSize + col);
}

void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_near(float actual, float expected, const std::string &message) {
  if (std::abs(actual - expected) > 1.0e-4F) {
    throw std::runtime_error(message);
  }
}

float plane_sum(const std::vector<float> &encoded, int plane,
                std::size_t batch_offset = 0) {
  const auto begin = encoded.begin() +
                     static_cast<std::ptrdiff_t>(batch_offset +
                                                 plane * kBoardArea);
  return std::accumulate(begin, begin + kBoardArea, 0.0F);
}

void test_initial_state_contract() {
  const State initial = quoridor::create_initial_state();
  const std::vector<float> encoded = quoridor::encode_state(initial);
  check(encoded.size() ==
            static_cast<std::size_t>(quoridor::kPhase4InputChannels *
                                     kBoardArea),
        "Phase 4 encoder shape mismatch");
  check(encoded[offset(0, 8, 4)] == 1.0F,
        "self pawn plane mismatch");
  check(encoded[offset(1, 0, 4)] == 1.0F,
        "opponent pawn plane mismatch");
  check_near(plane_sum(encoded, 4), 81.0F,
             "self walls plane mismatch");
  check_near(plane_sum(encoded, 5), 81.0F,
             "opponent walls plane mismatch");
  check_near(plane_sum(encoded, 6), 3.0F,
             "legal pawn targets plane mismatch");
  check_near(encoded[offset(7, 8, 4)], 0.1F,
             "self goal distance mismatch");
  check_near(encoded[offset(8, 0, 4)], 0.1F,
             "opponent goal distance mismatch");
  check_near(plane_sum(encoded, 12), 27.0F,
             "repetition count plane mismatch");
  check_near(plane_sum(encoded, 13), 81.0F, "ones plane mismatch");
  check_near(plane_sum(encoded, 14), 0.0F,
             "initial second-occurrence targets must be empty");
  check_near(plane_sum(encoded, 15), 0.0F,
             "initial repetition-loss targets must be empty");

  const std::vector<float> legacy =
      quoridor::encode_state(initial, quoridor::kLegacyInputChannels);
  check(legacy.size() ==
            static_cast<std::size_t>(quoridor::kLegacyInputChannels *
                                     kBoardArea),
        "legacy encoder shape mismatch");
  check(std::equal(legacy.begin(), legacy.end(), encoded.begin()),
        "legacy planes must be the prefix of the Phase 4 contract");
}

void test_repetition_target_planes() {
  const State initial = quoridor::create_initial_state();
  const int action = quoridor::encode_move(8, 3);
  const State child = quoridor::apply_action_index(initial, action);

  State second = initial;
  second.move_number = 1;
  second.position_counts = {{quoridor::position_hash(initial), 1},
                            {quoridor::position_hash(child), 1}};
  const std::vector<float> second_encoded = quoridor::encode_state(second);
  check(second_encoded[offset(14, 8, 3)] == 1.0F,
        "second-occurrence pawn target mismatch");
  check_near(plane_sum(second_encoded, 15), 0.0F,
             "second occurrence must not be marked as a loss");

  State third = second;
  third.move_number = 2;
  third.position_counts.set(quoridor::position_hash(child), 2);
  const std::vector<float> third_encoded = quoridor::encode_state(third);
  check_near(plane_sum(third_encoded, 14), 0.0F,
             "third occurrence must not remain in the second plane");
  check(third_encoded[offset(15, 8, 3)] == 1.0F,
        "repetition-loss pawn target mismatch");
}

void test_batch_layout() {
  const State first = quoridor::create_initial_state();
  const State second = quoridor::create_initial_state(1);
  const std::vector<float> first_encoded = quoridor::encode_state(first);
  const std::vector<float> second_encoded = quoridor::encode_state(second);
  const std::vector<float> batch = quoridor::encode_states({first, second});
  check(batch.size() == first_encoded.size() + second_encoded.size(),
        "batch encoder size mismatch");
  check(std::equal(first_encoded.begin(), first_encoded.end(), batch.begin()),
        "first batch entry mismatch");
  check(std::equal(second_encoded.begin(), second_encoded.end(),
                   batch.begin() +
                       static_cast<std::ptrdiff_t>(first_encoded.size())),
        "second batch entry mismatch");
}

} // namespace

int main() {
  try {
    test_initial_state_contract();
    test_repetition_target_planes();
    test_batch_layout();
  } catch (const std::exception &error) {
    std::cerr << "native encoder test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "native encoder tests passed\n";
  return 0;
}
