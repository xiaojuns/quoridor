#include "quoridor/mcts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace quoridor {
namespace {

struct Node;

struct Edge {
  int action_index = -1;
  double base_prior = 0.0;
  std::uint64_t visits = 0;
  double value_sum = 0.0;
  std::unique_ptr<Node> child;

  double q() const {
    return visits == 0 ? 0.0 : value_sum / static_cast<double>(visits);
  }
};

struct Node {
  explicit Node(State initial_state) : state(std::move(initial_state)) {}

  State state;
  std::vector<Edge> edges;
  bool expanded = false;
  double network_value = 0.0;

  std::uint64_t visits() const {
    std::uint64_t total = 0;
    for (const auto& edge : edges) {
      total += edge.visits;
    }
    return total;
  }
};

void validate_config(const MCTSConfig& config) {
  if (config.simulations == 0) {
    throw std::invalid_argument("simulations must be positive");
  }
  if (!std::isfinite(config.c_base) || config.c_base <= 0.0) {
    throw std::invalid_argument("c_base must be finite and positive");
  }
  if (!std::isfinite(config.c_init) || config.c_init < 0.0) {
    throw std::invalid_argument("c_init must be finite and non-negative");
  }
  if (!std::isfinite(config.dirichlet_epsilon) ||
      config.dirichlet_epsilon < 0.0 || config.dirichlet_epsilon > 1.0) {
    throw std::invalid_argument("dirichlet_epsilon must be between zero and one");
  }
  if (config.dirichlet_alpha.has_value() &&
      (!std::isfinite(*config.dirichlet_alpha) ||
       *config.dirichlet_alpha <= 0.0)) {
    throw std::invalid_argument(
        "dirichlet_alpha must be finite and positive or unset");
  }
  if (!std::isfinite(config.temperature) || config.temperature < 0.0) {
    throw std::invalid_argument("temperature must be finite and non-negative");
  }
}

std::vector<double> legal_softmax(const std::vector<double>& logits) {
  if (logits.empty()) {
    throw std::invalid_argument("cannot normalize an empty legal policy");
  }

  std::size_t positive_infinities = 0;
  for (const double value : logits) {
    if (value == std::numeric_limits<double>::infinity()) {
      ++positive_infinities;
    }
  }
  if (positive_infinities != 0) {
    std::vector<double> probabilities(logits.size(), 0.0);
    const double probability = 1.0 / static_cast<double>(positive_infinities);
    for (std::size_t index = 0; index < logits.size(); ++index) {
      if (logits[index] == std::numeric_limits<double>::infinity()) {
        probabilities[index] = probability;
      }
    }
    return probabilities;
  }

  double maximum = -std::numeric_limits<double>::infinity();
  for (const double value : logits) {
    if (std::isfinite(value)) {
      maximum = std::max(maximum, value);
    }
  }
  if (!std::isfinite(maximum)) {
    return std::vector<double>(logits.size(), 1.0 / logits.size());
  }

  std::vector<double> weights(logits.size(), 0.0);
  double total = 0.0;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (std::isfinite(logits[index])) {
      weights[index] = std::exp(logits[index] - maximum);
      total += weights[index];
    }
  }
  if (!std::isfinite(total) || total <= 0.0) {
    return std::vector<double>(logits.size(), 1.0 / logits.size());
  }
  for (double& weight : weights) {
    weight /= total;
  }
  return weights;
}

Edge& select_edge(Node& node, const MCTSConfig& config,
                  const std::vector<double>* decision_priors = nullptr) {
  if (decision_priors != nullptr &&
      decision_priors->size() != node.edges.size()) {
    throw std::logic_error("decision priors do not match root edges");
  }
  const std::uint64_t parent_visits = node.visits();
  const double dynamic_c =
      std::log((static_cast<double>(parent_visits) + config.c_base + 1.0) /
               config.c_base) +
      config.c_init;
  const double parent_scale =
      std::sqrt(static_cast<double>(parent_visits) + 1.0);

  Edge* best = nullptr;
  double best_score = -std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < node.edges.size(); ++index) {
    Edge& edge = node.edges[index];
    const double prior = decision_priors == nullptr
                             ? edge.base_prior
                             : (*decision_priors)[index];
    const double exploration = dynamic_c * prior * parent_scale /
                               (1.0 + static_cast<double>(edge.visits));
    const double score = edge.q() + exploration;
    if (best == nullptr || score > best_score ||
        (score == best_score && edge.action_index < best->action_index)) {
      best = &edge;
      best_score = score;
    }
  }
  if (best == nullptr) {
    throw std::logic_error("cannot select from an unexpanded node");
  }
  return *best;
}

std::array<double, kActionSpaceSize> visit_policy(const Node& root,
                                                  double temperature,
                                                  const std::vector<double>&
                                                      decision_priors) {
  if (decision_priors.size() != root.edges.size()) {
    throw std::logic_error("decision priors do not match root edges");
  }
  std::array<double, kActionSpaceSize> policy{};
  if (temperature == 0.0) {
    const Edge* best = nullptr;
    std::size_t best_index = 0;
    for (std::size_t index = 0; index < root.edges.size(); ++index) {
      const Edge& edge = root.edges[index];
      if (best == nullptr || edge.visits > best->visits ||
          (edge.visits == best->visits &&
           decision_priors[index] > decision_priors[best_index]) ||
          (edge.visits == best->visits &&
           decision_priors[index] == decision_priors[best_index] &&
           edge.action_index < best->action_index)) {
        best = &edge;
        best_index = index;
      }
    }
    if (best == nullptr) {
      throw std::logic_error("root has no actions");
    }
    policy[static_cast<std::size_t>(best->action_index)] = 1.0;
    return policy;
  }

  double maximum = -std::numeric_limits<double>::infinity();
  for (const auto& edge : root.edges) {
    if (edge.visits != 0) {
      maximum = std::max(
          maximum, std::log(static_cast<double>(edge.visits)) / temperature);
    }
  }
  if (!std::isfinite(maximum)) {
    for (std::size_t index = 0; index < root.edges.size(); ++index) {
      policy[static_cast<std::size_t>(root.edges[index].action_index)] =
          decision_priors[index];
    }
    return policy;
  }

  double total = 0.0;
  for (const auto& edge : root.edges) {
    if (edge.visits == 0) {
      continue;
    }
    const double weight =
        std::exp(std::log(static_cast<double>(edge.visits)) / temperature -
                 maximum);
    policy[static_cast<std::size_t>(edge.action_index)] = weight;
    total += weight;
  }
  for (double& probability : policy) {
    probability /= total;
  }
  return policy;
}

int sample_action(const Node& root,
                  const std::array<double, kActionSpaceSize>& policy,
                  double temperature, std::mt19937_64& random) {
  if (temperature == 0.0) {
    for (int action = 0; action < kActionSpaceSize; ++action) {
      if (policy[static_cast<std::size_t>(action)] == 1.0) {
        return action;
      }
    }
    throw std::logic_error("zero-temperature policy has no selected action");
  }

  std::array<double, kActionSpaceSize> weights = policy;
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  const int action = distribution(random);
  if (std::none_of(root.edges.begin(), root.edges.end(),
                   [action](const Edge& edge) {
                     return edge.action_index == action;
                   })) {
    throw std::logic_error("sampled an illegal action");
  }
  return action;
}

double expand_node(Node& node, const Evaluation& evaluation) {
  const std::vector<int> legal = legal_action_indices(node.state);
  if (legal.empty()) {
    throw std::logic_error("non-terminal state has no legal actions");
  }

  const int original_player = node.state.current_player;
  std::vector<double> legal_logits;
  legal_logits.reserve(legal.size());
  for (const int action : legal) {
    if (action < 0 || action >= kActionSpaceSize) {
      throw std::logic_error("legal action is outside the action space");
    }
    const int canonical_action =
        to_canonical_action_index(action, original_player);
    legal_logits.push_back(
        evaluation.policy_logits[static_cast<std::size_t>(canonical_action)]);
  }

  const std::vector<double> priors = legal_softmax(legal_logits);
  node.edges.reserve(legal.size());
  for (std::size_t index = 0; index < legal.size(); ++index) {
    Edge edge;
    edge.action_index = legal[index];
    edge.base_prior = priors[index];
    node.edges.push_back(std::move(edge));
  }
  node.expanded = true;
  node.network_value = evaluation.value;
  return evaluation.value;
}

std::vector<double> make_decision_priors(const Node& root,
                                         const MCTSConfig& config,
                                         bool add_root_noise,
                                         std::mt19937_64& random) {
  std::vector<double> priors;
  priors.reserve(root.edges.size());
  for (const Edge& edge : root.edges) {
    priors.push_back(edge.base_prior);
  }
  if (!add_root_noise || config.dirichlet_epsilon == 0.0) {
    return priors;
  }
  const double alpha = config.dirichlet_alpha.value_or(
      std::min(0.30, std::max(0.03, 10.0 / root.edges.size())));
  std::gamma_distribution<double> gamma(alpha, 1.0);
  std::vector<double> noise(root.edges.size());
  double total = 0.0;
  for (double& sample : noise) {
    sample = gamma(random);
    total += sample;
  }
  if (!std::isfinite(total) || total <= 0.0) {
    std::fill(noise.begin(), noise.end(), 1.0 / noise.size());
  } else {
    for (double& sample : noise) {
      sample /= total;
    }
  }
  for (std::size_t index = 0; index < root.edges.size(); ++index) {
    priors[index] =
        (1.0 - config.dirichlet_epsilon) * priors[index] +
        config.dirichlet_epsilon * noise[index];
  }
  return priors;
}

void backup(std::vector<Edge*>& path, double value) {
  for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
    value = -value;
    Edge& edge = **iterator;
    ++edge.visits;
    edge.value_sum += value;
  }
}

SearchResult make_search_result(const Node& root, double temperature,
                                const std::vector<double>& decision_priors,
                                std::mt19937_64& random) {
  SearchResult result;
  result.policy = visit_policy(root, temperature, decision_priors);
  result.action_index = sample_action(root, result.policy, temperature, random);
  std::uint64_t total_visits = 0;
  double total_value = 0.0;
  result.root_edges.reserve(root.edges.size());
  for (std::size_t index = 0; index < root.edges.size(); ++index) {
    const Edge& edge = root.edges[index];
    result.visit_counts[static_cast<std::size_t>(edge.action_index)] =
        edge.visits;
    total_visits += edge.visits;
    total_value += edge.value_sum;
    result.root_edges.push_back(EdgeStatistics{
        edge.action_index,
        decision_priors[index],
        edge.visits,
        edge.value_sum,
        edge.q(),
    });
  }
  result.root_value = total_visits == 0
                          ? root.network_value
                          : total_value / static_cast<double>(total_visits);
  return result;
}

struct PendingSimulation {
  Node* leaf = nullptr;
  std::vector<Edge*> path;
};

bool same_state_identity(const State& left, const State& right) {
  return position_hash(left) == position_hash(right) && left == right;
}

std::size_t node_count(const Node& node) {
  std::size_t count = 1;
  for (const Edge& edge : node.edges) {
    if (edge.child) {
      count += node_count(*edge.child);
    }
  }
  return count;
}

}  // namespace

struct MCTS::Session {
  explicit Session(State state)
      : root(std::make_unique<Node>(std::move(state))) {}

  void reset_decision() {
    decision_priors.clear();
    decision_noise.reset();
  }

  std::unique_ptr<Node> root;
  std::vector<double> decision_priors;
  std::optional<bool> decision_noise;
};

MCTS::MCTS(EvaluateCallback evaluator, MCTSConfig config)
    : evaluator_(std::move(evaluator)),
      config_(config),
      random_(config.seed) {
  if (!evaluator_) {
    throw std::invalid_argument("evaluator callback must not be empty");
  }
  validate_config(config_);
}

MCTS::MCTS(BatchEvaluateCallback evaluator, MCTSConfig config)
    : batch_evaluator_(std::move(evaluator)),
      config_(config),
      random_(config.seed) {
  if (!batch_evaluator_) {
    throw std::invalid_argument("batch evaluator callback must not be empty");
  }
  validate_config(config_);
}

MCTS::~MCTS() = default;

std::vector<Evaluation> MCTS::evaluate_batch(
    const std::vector<State>& canonical_states) const {
  if (canonical_states.empty()) {
    return {};
  }

  std::vector<Evaluation> results;
  if (evaluator_) {
    results.reserve(canonical_states.size());
    for (const State& state : canonical_states) {
      results.push_back(evaluator_(state));
    }
  } else {
    results = batch_evaluator_(canonical_states);
    if (results.size() != canonical_states.size()) {
      throw std::runtime_error(
          "batch evaluator must return one result for every input state");
    }
  }

  for (Evaluation& result : results) {
    if (!std::isfinite(result.value)) {
      throw std::runtime_error("evaluator value must be finite");
    }
    result.value = std::max(-1.0, std::min(1.0, result.value));
  }
  return results;
}

SearchResult MCTS::run(const State& state) {
  return run_batch(std::vector<State>{state},
                   std::vector<double>{config_.temperature},
                   config_.add_root_noise)
      .front();
}

SearchResult MCTS::run(const State& state, double temperature,
                       bool add_root_noise) {
  return run_batch(std::vector<State>{state},
                   std::vector<double>{temperature}, add_root_noise)
      .front();
}

std::vector<SearchResult> MCTS::run_batch(
    const std::vector<State>& states) {
  return run_batch(states,
                   std::vector<double>(states.size(), config_.temperature),
                   config_.add_root_noise);
}

std::vector<SearchResult> MCTS::run_batch(
    const std::vector<State>& states,
    const std::vector<double>& temperatures,
    bool add_root_noise) {
  std::vector<std::unique_ptr<Session>> owned_sessions;
  std::vector<Session*> session_pointers;
  owned_sessions.reserve(states.size());
  session_pointers.reserve(states.size());
  for (const State& state : states) {
    owned_sessions.push_back(std::make_unique<Session>(state));
    session_pointers.push_back(owned_sessions.back().get());
  }
  return run_roots(session_pointers, temperatures, add_root_noise);
}

std::vector<SearchResult> MCTS::run_roots(
    const std::vector<Session*>& sessions,
    const std::vector<double>& temperatures,
    bool add_root_noise) {
  if (sessions.size() != temperatures.size()) {
    throw std::invalid_argument(
        "temperatures must contain one value for every root state");
  }
  for (const double temperature : temperatures) {
    if (!std::isfinite(temperature) || temperature < 0.0) {
      throw std::invalid_argument("temperature must be finite and non-negative");
    }
  }
  if (sessions.empty()) {
    return {};
  }

  std::unordered_set<const Session*> unique_sessions;
  for (const Session* session : sessions) {
    if (session == nullptr) {
      throw std::invalid_argument("sessions must not contain null roots");
    }
    if (!unique_sessions.insert(session).second) {
      throw std::invalid_argument("session ids must be unique within a batch");
    }
    if (session->decision_noise.has_value() &&
        *session->decision_noise != add_root_noise) {
      throw std::invalid_argument(
          "root noise setting cannot change before advancing the session");
    }
  }

  std::vector<SearchResult> results(sessions.size());
  std::vector<std::size_t> root_indices;
  std::vector<State> canonical_roots;
  root_indices.reserve(sessions.size());
  canonical_roots.reserve(sessions.size());

  for (std::size_t index = 0; index < sessions.size(); ++index) {
    Session& session = *sessions[index];
    const State& state = session.root->state;
    if (is_terminal(state)) {
      const auto value = winner_value(state);
      if (!value.has_value()) {
        throw std::logic_error("terminal state has no winner value");
      }
      results[index].root_value = *value;
      continue;
    }
    root_indices.push_back(index);
    if (!session.root->expanded) {
      canonical_roots.push_back(canonicalize_state(state));
    }
  }

  const std::vector<Evaluation> root_evaluations =
      evaluate_batch(canonical_roots);
  std::size_t evaluation_index = 0;
  for (const std::size_t root_index : root_indices) {
    Session& session = *sessions[root_index];
    if (!session.root->expanded) {
      expand_node(*session.root, root_evaluations[evaluation_index]);
      ++evaluation_index;
    }
    if (!session.decision_noise.has_value()) {
      session.decision_priors = make_decision_priors(
          *session.root, config_, add_root_noise, random_);
      session.decision_noise = add_root_noise;
    }
  }
  if (evaluation_index != root_evaluations.size()) {
    throw std::logic_error("root evaluations were not consumed exactly once");
  }

  for (std::size_t simulation = 0; simulation < config_.simulations;
       ++simulation) {
    std::vector<PendingSimulation> pending;
    std::vector<State> canonical_leaves;
    pending.reserve(root_indices.size());
    canonical_leaves.reserve(root_indices.size());

    for (const std::size_t root_index : root_indices) {
      Session& session = *sessions[root_index];
      Node* const root = session.root.get();
      Node* node = root;
      PendingSimulation trajectory;
      while (node->expanded) {
        const std::vector<double>* priors =
            node == root ? &session.decision_priors : nullptr;
        Edge& edge = select_edge(*node, config_, priors);
        trajectory.path.push_back(&edge);
        if (!edge.child) {
          edge.child = std::make_unique<Node>(
              apply_legal_action_index(node->state, edge.action_index));
        }
        node = edge.child.get();
        if (is_terminal(node->state)) {
          break;
        }
      }

      if (is_terminal(node->state)) {
        const auto value = winner_value(node->state);
        if (!value.has_value()) {
          throw std::logic_error("terminal state has no winner value");
        }
        backup(trajectory.path, *value);
      } else {
        trajectory.leaf = node;
        canonical_leaves.push_back(canonicalize_state(node->state));
        pending.push_back(std::move(trajectory));
      }
    }

    const std::vector<Evaluation> leaf_evaluations =
        evaluate_batch(canonical_leaves);
    for (std::size_t index = 0; index < pending.size(); ++index) {
      const double value =
          expand_node(*pending[index].leaf, leaf_evaluations[index]);
      backup(pending[index].path, value);
    }
  }

  for (const std::size_t root_index : root_indices) {
    const Session& session = *sessions[root_index];
    results[root_index] = make_search_result(
        *session.root, temperatures[root_index], session.decision_priors,
        random_);
  }
  return results;
}

MCTS::SessionId MCTS::create_session(const State& state) {
  if (next_session_id_ == 0) {
    throw std::overflow_error("MCTS session id space is exhausted");
  }
  const SessionId session_id = next_session_id_++;
  sessions_.emplace(session_id, std::make_unique<Session>(state));
  return session_id;
}

bool MCTS::sync_session(SessionId session_id, const State& state) {
  const auto iterator = sessions_.find(session_id);
  if (iterator == sessions_.end()) {
    throw std::out_of_range("unknown MCTS session id");
  }
  Session& session = *iterator->second;
  if (same_state_identity(session.root->state, state)) {
    return true;
  }
  session.root = std::make_unique<Node>(state);
  session.reset_decision();
  return false;
}

SearchResult MCTS::run_session(SessionId session_id, const State& state) {
  return run_session(session_id, state, config_.temperature,
                     config_.add_root_noise);
}

SearchResult MCTS::run_session(SessionId session_id, const State& state,
                               double temperature, bool add_root_noise) {
  return run_sessions(std::vector<SessionId>{session_id},
                      std::vector<State>{state},
                      std::vector<double>{temperature}, add_root_noise)
      .front();
}

std::vector<SearchResult> MCTS::run_sessions(
    const std::vector<SessionId>& session_ids,
    const std::vector<State>& states) {
  return run_sessions(session_ids, states,
                      std::vector<double>(states.size(), config_.temperature),
                      config_.add_root_noise);
}

std::vector<SearchResult> MCTS::run_sessions(
    const std::vector<SessionId>& session_ids,
    const std::vector<State>& states,
    const std::vector<double>& temperatures,
    bool add_root_noise) {
  if (session_ids.size() != states.size()) {
    throw std::invalid_argument(
        "session ids must contain one value for every root state");
  }
  if (temperatures.size() != states.size()) {
    throw std::invalid_argument(
        "temperatures must contain one value for every root state");
  }
  for (const double temperature : temperatures) {
    if (!std::isfinite(temperature) || temperature < 0.0) {
      throw std::invalid_argument("temperature must be finite and non-negative");
    }
  }

  std::unordered_set<SessionId> unique_ids;
  std::vector<Session*> session_pointers;
  session_pointers.reserve(session_ids.size());
  for (const SessionId session_id : session_ids) {
    if (!unique_ids.insert(session_id).second) {
      throw std::invalid_argument("session ids must be unique within a batch");
    }
    const auto iterator = sessions_.find(session_id);
    if (iterator == sessions_.end()) {
      throw std::out_of_range("unknown MCTS session id");
    }
    const std::size_t index = session_pointers.size();
    const Session& session = *iterator->second;
    if (same_state_identity(session.root->state, states[index]) &&
        session.decision_noise.has_value() &&
        *session.decision_noise != add_root_noise) {
      throw std::invalid_argument(
          "root noise setting cannot change before advancing the session");
    }
    session_pointers.push_back(iterator->second.get());
  }
  for (std::size_t index = 0; index < states.size(); ++index) {
    sync_session(session_ids[index], states[index]);
  }
  return run_roots(session_pointers, temperatures, add_root_noise);
}

bool MCTS::advance_session(SessionId session_id, int action_index,
                           const State& next_state) {
  const auto iterator = sessions_.find(session_id);
  if (iterator == sessions_.end()) {
    throw std::out_of_range("unknown MCTS session id");
  }
  Session& session = *iterator->second;

  bool transition_matches = false;
  if (!is_terminal(session.root->state)) {
    try {
      const State expected = apply_action_index(session.root->state, action_index);
      transition_matches = same_state_identity(expected, next_state);
    } catch (const std::invalid_argument&) {
      transition_matches = false;
    }
  }

  if (transition_matches) {
    const auto edge = std::find_if(
        session.root->edges.begin(), session.root->edges.end(),
        [action_index](const Edge& candidate) {
          return candidate.action_index == action_index;
        });
    if (edge != session.root->edges.end() && edge->child &&
        same_state_identity(edge->child->state, next_state)) {
      std::unique_ptr<Node> promoted = std::move(edge->child);
      session.root = std::move(promoted);
      session.reset_decision();
      return true;
    }
  }

  session.root = std::make_unique<Node>(next_state);
  session.reset_decision();
  return false;
}

bool MCTS::release_session(SessionId session_id) {
  return sessions_.erase(session_id) != 0;
}

void MCTS::clear_sessions() { sessions_.clear(); }

std::size_t MCTS::session_count() const noexcept { return sessions_.size(); }

std::size_t MCTS::session_node_count(SessionId session_id) const {
  const auto iterator = sessions_.find(session_id);
  if (iterator == sessions_.end()) {
    throw std::out_of_range("unknown MCTS session id");
  }
  return node_count(*iterator->second->root);
}

}  // namespace quoridor
