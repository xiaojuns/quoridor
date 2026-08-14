#pragma once

#include <vector>

#include "quoridor/game.hpp"

namespace quoridor {

inline constexpr int kLegacyInputChannels = 14;
inline constexpr int kPhase4InputChannels = 16;

std::vector<float> encode_state(const State &state,
                                int input_channels = kPhase4InputChannels);
std::vector<float>
encode_states(const std::vector<State> &states,
              int input_channels = kPhase4InputChannels);

} // namespace quoridor
