#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include "quoridor/game.hpp"

namespace quoridor {

struct Evaluation {
  std::array<double, kActionSpaceSize> policy_logits{};
  double value = 0.0;
};

using EvaluateCallback = std::function<Evaluation(const State&)>;
// Inputs are canonical states; outputs must preserve their order and cardinality.
using BatchEvaluateCallback =
    std::function<std::vector<Evaluation>(const std::vector<State>&)>;

struct MCTSConfig {
  std::size_t simulations = 800;
  double c_base = 19652.0;
  double c_init = 1.25;
  double dirichlet_epsilon = 0.25;
  std::optional<double> dirichlet_alpha = std::nullopt;
  double temperature = 1.0;
  bool add_root_noise = true;
  std::uint64_t seed = 0;
};

struct EdgeStatistics {
  int action_index = -1;
  double prior = 0.0;
  std::uint64_t visits = 0;
  double value_sum = 0.0;
  double q = 0.0;
};

struct SearchResult {
  std::optional<int> action_index;
  std::array<double, kActionSpaceSize> policy{};
  std::array<std::uint64_t, kActionSpaceSize> visit_counts{};
  double root_value = 0.0;
  std::vector<EdgeStatistics> root_edges;
};

class MCTS {
 public:
  using SessionId = std::uint64_t;

  explicit MCTS(EvaluateCallback evaluator, MCTSConfig config = {});
  explicit MCTS(BatchEvaluateCallback evaluator, MCTSConfig config = {});
  ~MCTS();

  MCTS(const MCTS&) = delete;
  MCTS& operator=(const MCTS&) = delete;

  SearchResult run(const State& state);
  SearchResult run(const State& state, double temperature,
                   bool add_root_noise);
  std::vector<SearchResult> run_batch(const std::vector<State>& states);
  // `temperatures` contains one sampling temperature per root.
  std::vector<SearchResult> run_batch(
      const std::vector<State>& states,
      const std::vector<double>& temperatures,
      bool add_root_noise);

  SessionId create_session(const State& state);
  // Returns true only when the playable position and complete history match.
  // A mismatch replaces the retained tree with the supplied authoritative state.
  bool sync_session(SessionId session_id, const State& state);
  SearchResult run_session(SessionId session_id, const State& state);
  SearchResult run_session(SessionId session_id, const State& state,
                           double temperature, bool add_root_noise);
  std::vector<SearchResult> run_sessions(
      const std::vector<SessionId>& session_ids,
      const std::vector<State>& states);
  std::vector<SearchResult> run_sessions(
      const std::vector<SessionId>& session_ids,
      const std::vector<State>& states,
      const std::vector<double>& temperatures,
      bool add_root_noise);
  // Returns true when the exact child subtree was promoted, false after a safe
  // rebuild. Promotion destroys the old root and all unselected siblings.
  bool advance_session(SessionId session_id, int action_index,
                       const State& next_state);
  bool release_session(SessionId session_id);
  void clear_sessions();
  std::size_t session_count() const noexcept;
  std::size_t session_node_count(SessionId session_id) const;

 private:
  struct Session;

  std::vector<Evaluation> evaluate_batch(
      const std::vector<State>& canonical_states) const;
  std::vector<SearchResult> run_roots(
      const std::vector<Session*>& sessions,
      const std::vector<double>& temperatures,
      bool add_root_noise);

  EvaluateCallback evaluator_;
  BatchEvaluateCallback batch_evaluator_;
  MCTSConfig config_;
  std::mt19937_64 random_;
  std::unordered_map<SessionId, std::unique_ptr<Session>> sessions_;
  SessionId next_session_id_ = 1;
};

}  // namespace quoridor
