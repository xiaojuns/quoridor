#include "quoridor/mcts.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using quoridor::BatchEvaluateCallback;
using quoridor::EdgeStatistics;
using quoridor::EvaluateCallback;
using quoridor::Evaluation;
using quoridor::MCTS;
using quoridor::MCTSConfig;
using quoridor::SearchResult;
using quoridor::State;

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

void check_near(double actual, double expected, const std::string& message,
                double tolerance = 1e-10) {
  check(std::abs(actual - expected) <= tolerance, message);
}

Evaluation evaluation_with_preference(int action_index, double value = 0.0) {
  Evaluation evaluation;
  evaluation.policy_logits.fill(-100.0);
  evaluation.policy_logits[static_cast<std::size_t>(action_index)] = 100.0;
  evaluation.value = value;
  return evaluation;
}

Evaluation uniform_evaluation(double value = 0.0) {
  Evaluation evaluation;
  evaluation.policy_logits.fill(0.0);
  evaluation.value = value;
  return evaluation;
}

MCTSConfig config(std::size_t simulations, std::uint64_t seed = 7,
                  bool noise = false, double temperature = 0.0) {
  MCTSConfig result;
  result.simulations = simulations;
  result.seed = seed;
  result.add_root_noise = noise;
  result.temperature = temperature;
  return result;
}

const EdgeStatistics& edge_for(const SearchResult& result, int action_index) {
  const auto iterator = std::find_if(
      result.root_edges.begin(), result.root_edges.end(),
      [action_index](const EdgeStatistics& edge) {
        return edge.action_index == action_index;
      });
  if (iterator == result.root_edges.end()) {
    throw std::runtime_error("missing root edge");
  }
  return *iterator;
}

void test_complete_legal_expansion_and_masking() {
  const State state = quoridor::create_initial_state();
  const std::vector<int> legal = quoridor::legal_action_indices(state);
  check(std::find(legal.begin(), legal.end(), 0) == legal.end(),
        "chosen illegal test action must be illegal");

  MCTS search(
      EvaluateCallback([](const State&) {
        return evaluation_with_preference(0);
      }),
      config(1));
  const SearchResult result = search.run(state);

  check(result.root_edges.size() == legal.size(),
        "root must retain every legal action");
  check(result.root_edges.size() == 131,
        "initial root must contain 131 actions");
  const auto wall_count = std::count_if(
      result.root_edges.begin(), result.root_edges.end(),
      [](const EdgeStatistics& edge) { return edge.action_index >= 81; });
  check(wall_count == 128, "root must not prune any initial wall action");
  check(result.action_index.has_value(), "non-terminal search must choose an action");
  check(std::find(legal.begin(), legal.end(), *result.action_index) != legal.end(),
        "selected action must be legal");
  check(result.policy[0] == 0.0, "illegal action must have zero probability");

  double prior_sum = 0.0;
  for (const auto& edge : result.root_edges) {
    prior_sum += edge.prior;
  }
  check_near(prior_sum, 1.0, "legal priors must normalize to one");
}

void test_value_sign_backup() {
  State state = quoridor::create_initial_state();
  state.walls_remaining = {{0, 0}};
  const int action = quoridor::encode_move(7, 4);
  int calls = 0;
  MCTS search(
      EvaluateCallback([&calls, action](const State&) {
        const double value = calls == 0 ? 0.0 : 0.75;
        ++calls;
        return evaluation_with_preference(action, value);
      }),
      config(1));

  const SearchResult result = search.run(state);
  const EdgeStatistics& edge = edge_for(result, action);
  check(edge.visits == 1, "selected edge must receive one visit");
  check_near(edge.value_sum, -0.75,
             "child current-player value must negate at parent");
  check_near(edge.q, -0.75, "edge Q must use the parent perspective");
  check_near(result.root_value, -0.75,
             "root value must use the root player perspective");
  check(calls == 2, "non-terminal child must be evaluated exactly once");
}

void test_player_one_canonical_immediate_win() {
  State state;
  state.pawns = {{quoridor::Position{8, 0}, quoridor::Position{7, 4}}};
  state.walls_remaining = {{0, 0}};
  state.current_player = 1;
  state.position_counts.clear();
  const int winning_action = quoridor::encode_move(8, 4);
  const int canonical_action =
      quoridor::to_canonical_action_index(winning_action, 1);
  int calls = 0;

  MCTS search(
      EvaluateCallback([&calls, canonical_action](const State& canonical) {
        ++calls;
        check(canonical.current_player == 0,
              "evaluator must receive a canonical player-zero state");
        return evaluation_with_preference(canonical_action);
      }),
      config(4));
  const SearchResult result = search.run(state);

  check(result.action_index == winning_action,
        "canonical policy must map back to player one's raw winning action");
  const EdgeStatistics& edge = edge_for(result, winning_action);
  check(edge.visits == 4, "immediate win must receive all visits");
  check_near(edge.q, 1.0, "immediate win must have exact Q=1");
  check_near(result.root_value, 1.0, "immediate win root value must be one");
  check(calls == 1, "terminal children must not invoke the evaluator");
}

void test_terminal_and_batch_evaluators() {
  State state;
  state.pawns = {{quoridor::Position{1, 4}, quoridor::Position{0, 0}}};
  state.walls_remaining = {{0, 0}};
  const State ended =
      quoridor::apply_action_index(state, quoridor::encode_move(0, 4));
  int single_calls = 0;
  MCTS terminal_search(
      EvaluateCallback([&single_calls](const State&) {
        ++single_calls;
        return uniform_evaluation();
      }),
      config(8));
  const SearchResult terminal = terminal_search.run(ended);
  check(!terminal.action_index.has_value(),
        "terminal root must not return an action");
  check_near(terminal.root_value, -1.0,
             "terminal value must use the current player perspective");
  check(single_calls == 0, "terminal root must not call a single evaluator");

  State winning_state;
  winning_state.pawns =
      {{quoridor::Position{1, 4}, quoridor::Position{0, 0}}};
  winning_state.walls_remaining = {{0, 0}};
  const int winning_action = quoridor::encode_move(0, 4);
  int batch_calls = 0;
  MCTS batch_search(
      BatchEvaluateCallback([&batch_calls, winning_action](
                                const std::vector<State>& states) {
        ++batch_calls;
        check(states.size() == 1,
              "sequential search must request a singleton evaluation batch");
        return std::vector<Evaluation>{
            evaluation_with_preference(winning_action),
        };
      }),
      config(2));
  const SearchResult batch_result = batch_search.run(winning_state);
  check(batch_result.action_index == winning_action,
        "batch evaluator must drive the same immediate win");
  check(batch_calls == 1,
        "terminal children must not add batch evaluator calls");
}

void test_multiple_roots_use_real_batches() {
  State first = quoridor::create_initial_state();
  first.walls_remaining = {{0, 0}};
  State second = first;
  const int preferred_action = quoridor::encode_move(7, 4);
  std::vector<std::size_t> batch_sizes;

  MCTS search(
      BatchEvaluateCallback([&batch_sizes, preferred_action](
                                const std::vector<State>& states) {
        batch_sizes.push_back(states.size());
        std::vector<Evaluation> evaluations;
        evaluations.reserve(states.size());
        for (const State& state : states) {
          check(state.current_player == 0,
                "every batched evaluator state must be canonical");
          evaluations.push_back(evaluation_with_preference(
              preferred_action, state.move_number == 0 ? 0.0 : 0.6));
        }
        return evaluations;
      }),
      config(1));

  const std::vector<SearchResult> results = search.run_batch(
      std::vector<State>{first, second}, std::vector<double>{0.0, 0.0}, false);

  check(results.size() == 2, "batch search must return one result per root");
  check(std::any_of(batch_sizes.begin(), batch_sizes.end(),
                    [](std::size_t size) { return size > 1; }),
        "batch evaluator must receive more than one state in one call");
  check(batch_sizes.size() == 2 && batch_sizes[0] == 2 && batch_sizes[1] == 2,
        "root and leaf evaluations must each form a two-state batch");

  for (std::size_t index = 0; index < results.size(); ++index) {
    const SearchResult& result = results[index];
    const std::uint64_t total_visits = std::accumulate(
        result.visit_counts.begin(), result.visit_counts.end(),
        std::uint64_t{0});
    check(total_visits == 1,
          "every root must independently receive the simulation budget");
    check(result.action_index.has_value(),
          "every non-terminal root must return an action");
    const std::vector<int> legal =
        quoridor::legal_action_indices(index == 0 ? first : second);
    check(std::find(legal.begin(), legal.end(), *result.action_index) !=
              legal.end(),
          "every batched root action must be legal");
    const EdgeStatistics& edge = edge_for(result, preferred_action);
    check(edge.visits == 1, "preferred edge must receive the root visit");
    check_near(edge.q, -0.6,
               "batched child value must negate into the root perspective");
    check_near(result.root_value, -0.6,
               "batched root value must preserve backup sign semantics");
  }
}

void test_temperature_and_seed_determinism() {
  State state = quoridor::create_initial_state();
  state.walls_remaining = {{0, 0}};
  const MCTSConfig settings = config(24, 91, true, 1.0);
  const EvaluateCallback evaluator = [](const State&) {
    return uniform_evaluation();
  };

  MCTS first_search(evaluator, settings);
  MCTS second_search(evaluator, settings);
  const SearchResult first = first_search.run(state);
  const SearchResult second = second_search.run(state);

  check(first.visit_counts == second.visit_counts,
        "same seed must reproduce visit counts");
  check(first.policy == second.policy,
        "same seed must reproduce visit policy");
  check(first.action_index == second.action_index,
        "same seed must reproduce sampled action");
  check(first.root_edges.size() == second.root_edges.size(),
        "same root must expose the same number of edges");
  for (std::size_t index = 0; index < first.root_edges.size(); ++index) {
    check_near(first.root_edges[index].prior, second.root_edges[index].prior,
               "same seed must reproduce root noise");
  }

  const std::uint64_t total_visits = std::accumulate(
      first.visit_counts.begin(), first.visit_counts.end(), std::uint64_t{0});
  check(total_visits == settings.simulations,
        "root visit counts must equal simulation budget");
  const double probability_sum =
      std::accumulate(first.policy.begin(), first.policy.end(), 0.0);
  check_near(probability_sum, 1.0, "temperature policy must normalize");
  for (std::size_t action = 0; action < first.policy.size(); ++action) {
    check_near(first.policy[action],
               static_cast<double>(first.visit_counts[action]) /
                   static_cast<double>(total_visits),
               "temperature one policy must equal normalized visits");
  }
}

void test_session_promotes_child_and_reuses_search() {
  State state = quoridor::create_initial_state();
  state.walls_remaining = {{0, 0}};
  const int preferred_action = quoridor::encode_move(7, 4);
  int move_one_evaluations = 0;

  MCTS search(
      EvaluateCallback([&move_one_evaluations, preferred_action](
                           const State& canonical) {
        if (canonical.move_number == 1) {
          ++move_one_evaluations;
        }
        return evaluation_with_preference(preferred_action);
      }),
      config(4));
  const MCTS::SessionId session_id = search.create_session(state);
  const SearchResult first = search.run_session(session_id, state);

  check(first.action_index == preferred_action,
        "session search must choose the preferred action");
  check(move_one_evaluations == 1,
        "selected child must be evaluated once during the first decision");
  const std::size_t nodes_before_advance =
      search.session_node_count(session_id);
  const State next = quoridor::apply_action_index(state, preferred_action);
  check(search.advance_session(session_id, preferred_action, next),
        "advance must promote an exact selected child");
  const std::size_t nodes_after_advance = search.session_node_count(session_id);
  check(nodes_after_advance < nodes_before_advance,
        "advance must release the old root and sibling branches");

  const SearchResult second = search.run_session(session_id, next);
  const std::uint64_t retained_visits = std::accumulate(
      second.visit_counts.begin(), second.visit_counts.end(), std::uint64_t{0});
  check(retained_visits > 4,
        "promoted root must retain child visits before adding simulations");
  check(move_one_evaluations == 1,
        "promoted expanded root must not be evaluated again");
  check(search.sync_session(session_id, next),
        "an exact state including history must preserve the session tree");
  check(search.release_session(session_id),
        "release must remove an existing session");
  check(!search.release_session(session_id),
        "release must report an already removed session");
  check(search.session_count() == 0,
        "released session memory must no longer be retained");
}

void test_session_history_mismatch_rebuilds_safely() {
  State state = quoridor::create_initial_state();
  state.walls_remaining = {{0, 0}};
  const int preferred_action = quoridor::encode_move(7, 4);
  MCTS search(
      EvaluateCallback([preferred_action](const State&) {
        return evaluation_with_preference(preferred_action);
      }),
      config(3));
  const MCTS::SessionId session_id = search.create_session(state);
  (void)search.run_session(session_id, state);
  check(search.session_node_count(session_id) > 1,
        "searched session must contain a reusable subtree");

  State mismatched_root = state;
  mismatched_root.position_counts.set("unrelated-history", 2);
  check(quoridor::position_hash(mismatched_root) ==
            quoridor::position_hash(state),
        "history mismatch fixture must preserve the playable-position hash");
  check(!search.sync_session(session_id, mismatched_root),
        "same hash with different repetition history must rebuild the root");
  check(search.session_node_count(session_id) == 1,
        "history mismatch rebuild must discard all stale nodes");
  check(search.sync_session(session_id, mismatched_root),
        "rebuilt session must then match the authoritative state exactly");

  const MCTS::SessionId advance_id = search.create_session(state);
  const SearchResult advance_result = search.run_session(advance_id, state);
  check(advance_result.action_index.has_value(),
        "session fixture must choose an action before advance");
  State next = quoridor::apply_action_index(state, *advance_result.action_index);
  next.position_counts.set("different-next-history", 2);
  check(!search.advance_session(advance_id, *advance_result.action_index, next),
        "advance must reject a child with mismatched repetition history");
  check(search.session_node_count(advance_id) == 1,
        "mismatched advance must safely rebuild from the supplied state");

  search.clear_sessions();
  check(search.session_count() == 0,
        "clear_sessions must release every retained tree");
}

void test_session_root_noise_is_cached_per_decision() {
  State state = quoridor::create_initial_state();
  state.walls_remaining = {{0, 0}};
  MCTS search(
      EvaluateCallback([](const State&) { return uniform_evaluation(); }),
      config(12, 1234, true, 1.0));
  const MCTS::SessionId session_id = search.create_session(state);

  const SearchResult first = search.run_session(session_id, state, 1.0, true);
  const SearchResult second = search.run_session(session_id, state, 1.0, true);
  check(first.root_edges.size() == second.root_edges.size(),
        "repeated decision must expose the same root actions");
  for (std::size_t index = 0; index < first.root_edges.size(); ++index) {
    check_near(first.root_edges[index].prior, second.root_edges[index].prior,
               "root noise must be sampled once and never accumulate");
  }
  const std::uint64_t second_visits = std::accumulate(
      second.visit_counts.begin(), second.visit_counts.end(), std::uint64_t{0});
  check(second_visits == 24,
        "repeated session search must add simulations to the existing root");

  bool changed_noise_rejected = false;
  try {
    (void)search.run_session(session_id, state, 1.0, false);
  } catch (const std::invalid_argument&) {
    changed_noise_rejected = true;
  }
  check(changed_noise_rejected,
        "noise mode must not change during one persistent decision");

  check(second.action_index.has_value(),
        "noisy session search must select an action");
  const State next =
      quoridor::apply_action_index(state, *second.action_index);
  check(search.advance_session(session_id, *second.action_index, next),
        "sampled visited action must have a promotable child");
  const SearchResult next_first =
      search.run_session(session_id, next, 1.0, true);
  const SearchResult next_second =
      search.run_session(session_id, next, 1.0, true);
  for (std::size_t index = 0; index < next_first.root_edges.size(); ++index) {
    check_near(next_first.root_edges[index].prior,
               next_second.root_edges[index].prior,
               "advanced root must also cache exactly one noise sample");
  }
}

void test_multiple_sessions_preserve_batching() {
  State first = quoridor::create_initial_state();
  first.walls_remaining = {{0, 0}};
  State second = first;
  std::vector<std::size_t> batch_sizes;
  MCTS search(
      BatchEvaluateCallback([&batch_sizes](const std::vector<State>& states) {
        batch_sizes.push_back(states.size());
        return std::vector<Evaluation>(states.size(), uniform_evaluation());
      }),
      config(2));
  const MCTS::SessionId first_id = search.create_session(first);
  const MCTS::SessionId second_id = search.create_session(second);
  const std::vector<SearchResult> results = search.run_sessions(
      std::vector<MCTS::SessionId>{first_id, second_id},
      std::vector<State>{first, second}, std::vector<double>{0.0, 0.0}, false);

  check(results.size() == 2,
        "multi-session search must return one result per retained root");
  check(batch_sizes.size() == 3 &&
            std::all_of(batch_sizes.begin(), batch_sizes.end(),
                        [](std::size_t size) { return size == 2; }),
        "multi-session root and leaf inference must remain batched");
  check(search.session_count() == 2,
        "multi-session search must retain both roots until release");

  bool duplicate_rejected = false;
  try {
    (void)search.run_sessions(
        std::vector<MCTS::SessionId>{first_id, first_id},
        std::vector<State>{first, first}, std::vector<double>{0.0, 0.0}, false);
  } catch (const std::invalid_argument&) {
    duplicate_rejected = true;
  }
  check(duplicate_rejected,
        "one session root must not appear twice in the same batch");
  search.clear_sessions();
  check(search.session_count() == 0,
        "multi-session clear must release every root");
}

}  // namespace

int main() {
  test_complete_legal_expansion_and_masking();
  test_value_sign_backup();
  test_player_one_canonical_immediate_win();
  test_terminal_and_batch_evaluators();
  test_multiple_roots_use_real_batches();
  test_temperature_and_seed_determinism();
  test_session_promotes_child_and_reuses_search();
  test_session_history_mismatch_rebuilds_safely();
  test_session_root_noise_is_cached_per_decision();
  test_multiple_sessions_preserve_batching();

  if (failures != 0) {
    std::cerr << failures << " MCTS test(s) failed\n";
    return 1;
  }
  std::cout << "native MCTS tests passed\n";
  return 0;
}
