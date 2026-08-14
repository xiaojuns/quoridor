#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "quoridor/encoder.hpp"
#include "quoridor/game.hpp"
#include "quoridor/mcts.hpp"

namespace py = pybind11;

namespace quoridor {
namespace {

int as_int(const py::handle value, const char* label) {
  if (PyBool_Check(value.ptr()) || !PyLong_Check(value.ptr())) {
    throw py::type_error(std::string(label) + " must be an integer");
  }
  return py::cast<int>(value);
}

Position position_from_sequence(const py::handle value, const char* label) {
  if (PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr()) ||
      !PySequence_Check(value.ptr())) {
    throw py::type_error(std::string(label) + " must be a two-item sequence");
  }
  const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
  if (py::len(sequence) != 2) {
    throw py::value_error(std::string(label) + " must contain row and col");
  }
  return Position{as_int(sequence[0], label), as_int(sequence[1], label)};
}

Position position_from_dict(const py::handle value, const char* label) {
  if (!PyDict_Check(value.ptr())) {
    throw py::type_error(std::string(label) + " must be a mapping");
  }
  const py::dict position = py::reinterpret_borrow<py::dict>(value);
  if (!position.contains("row") || !position.contains("col")) {
    throw py::value_error(std::string(label) + " must contain row and col");
  }
  return Position{as_int(position["row"], label),
                  as_int(position["col"], label)};
}

Wall wall_from_key(const py::handle value) {
  if (!PyUnicode_Check(value.ptr())) {
    throw py::type_error("wall keys must be strings");
  }
  const std::string key = py::cast<std::string>(value);
  const std::size_t separator = key.find(',');
  if (separator == std::string::npos || key.find(',', separator + 1) !=
                                           std::string::npos) {
    throw py::value_error("invalid wall key: " + key);
  }
  try {
    std::size_t row_end = 0;
    std::size_t col_end = 0;
    const int row = std::stoi(key.substr(0, separator), &row_end);
    const int col = std::stoi(key.substr(separator + 1), &col_end);
    if (row_end != separator || col_end != key.size() - separator - 1) {
      throw py::value_error("invalid wall key: " + key);
    }
    if (row < 0 || row >= kWallGridSize || col < 0 ||
        col >= kWallGridSize) {
      throw py::value_error("wall coordinate is outside 0..7");
    }
    return Wall{row, col};
  } catch (const std::invalid_argument&) {
    throw py::value_error("invalid wall key: " + key);
  } catch (const std::out_of_range&) {
    throw py::value_error("invalid wall key: " + key);
  }
}

TerminationReason parse_termination_reason(const py::handle value) {
  if (value.is_none()) {
    return TerminationReason::None;
  }
  const std::string reason = py::cast<std::string>(value);
  if (reason == "goal") {
    return TerminationReason::Goal;
  }
  if (reason == "repetition") {
    return TerminationReason::Repetition;
  }
  if (reason == "move-limit") {
    return TerminationReason::MoveLimit;
  }
  throw py::value_error("unknown termination reason: " + reason);
}

py::object termination_reason_to_python(const TerminationReason reason) {
  if (reason == TerminationReason::None) {
    return py::none();
  }
  return py::str(termination_reason_name(reason));
}

template <typename Parser>
std::array<Position, 2> parse_pair(const py::handle value, const char* label,
                                   Parser parser) {
  if (PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr()) ||
      !PySequence_Check(value.ptr())) {
    throw py::type_error(std::string(label) + " must be a two-item sequence");
  }
  const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
  if (py::len(sequence) != 2) {
    throw py::value_error(std::string(label) + " must contain two entries");
  }
  return {{parser(sequence[0], label), parser(sequence[1], label)}};
}

std::array<int, 2> integer_pair(const py::handle value, const char* label) {
  if (PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr()) ||
      !PySequence_Check(value.ptr())) {
    throw py::type_error(std::string(label) + " must be a two-item sequence");
  }
  const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
  if (py::len(sequence) != 2) {
    throw py::value_error(std::string(label) + " must contain two entries");
  }
  return {{as_int(sequence[0], label), as_int(sequence[1], label)}};
}

std::set<Wall> walls_from_python(const py::handle value) {
  std::set<Wall> walls;
  for (const py::handle item : py::reinterpret_borrow<py::iterable>(value)) {
    walls.insert(position_from_sequence(item, "wall"));
  }
  return walls;
}

std::set<Wall> walls_from_keys(const py::handle value) {
  std::set<Wall> walls;
  for (const py::handle item : py::reinterpret_borrow<py::iterable>(value)) {
    walls.insert(wall_from_key(item));
  }
  return walls;
}

State state_from_python(const py::handle source) {
  State state;
  state.pawns = parse_pair(source.attr("pawns"), "pawns",
                           position_from_sequence);
  state.walls_remaining =
      integer_pair(source.attr("walls_remaining"), "walls_remaining");
  state.horizontal_walls =
      walls_from_python(source.attr("horizontal_walls"));
  state.vertical_walls = walls_from_python(source.attr("vertical_walls"));
  state.current_player = as_int(source.attr("current_player"), "current_player");

  const py::object winner = source.attr("winner");
  state.winner = winner.is_none() ? kNoWinner : as_int(winner, "winner");
  state.move_number = as_int(source.attr("move_number"), "move_number");
  state.termination_reason =
      parse_termination_reason(source.attr("termination_reason"));

  const py::dict counts(source.attr("position_counts"));
  PositionCounts::Map native_counts;
  for (const auto& item : counts) {
    const std::string key = py::cast<std::string>(item.first);
    native_counts.emplace(key, as_int(item.second, "repetition count"));
  }
  state.position_counts.assign(std::move(native_counts));
  validate_state(state);
  return state;
}

py::array_t<float> encode_states_from_python(const py::sequence& sources,
                                             const int input_channels) {
  std::vector<State> states;
  states.reserve(static_cast<std::size_t>(py::len(sources)));
  for (const py::handle source : sources) {
    states.push_back(py::isinstance<State>(source)
                         ? py::cast<State>(source)
                         : state_from_python(source));
  }

  std::vector<float> encoded;
  {
    py::gil_scoped_release release;
    encoded = encode_states(states, input_channels);
  }
  const std::vector<py::ssize_t> shape{
      static_cast<py::ssize_t>(states.size()),
      static_cast<py::ssize_t>(input_channels),
      static_cast<py::ssize_t>(kBoardSize),
      static_cast<py::ssize_t>(kBoardSize),
  };
  py::array_t<float> output(shape);
  std::copy(encoded.begin(), encoded.end(), output.mutable_data());
  return output;
}

State state_from_dict(const py::dict& dto) {
  const int version = dto.contains("codecVersion")
                          ? as_int(dto["codecVersion"], "codecVersion")
                          : kCodecVersion;
  if (version != kCodecVersion) {
    throw py::value_error("unsupported codec version");
  }

  for (const char* key : {"pawns", "wallsRemaining", "horizontalWalls",
                          "verticalWalls", "currentPlayer"}) {
    if (!dto.contains(key)) {
      throw py::value_error(std::string("state DTO is missing ") + key);
    }
  }

  State state;
  state.pawns =
      parse_pair(dto["pawns"], "pawns", position_from_dict);
  state.walls_remaining =
      integer_pair(dto["wallsRemaining"], "wallsRemaining");
  state.horizontal_walls = walls_from_keys(dto["horizontalWalls"]);
  state.vertical_walls = walls_from_keys(dto["verticalWalls"]);
  state.current_player = as_int(dto["currentPlayer"], "currentPlayer");
  if (dto.contains("winner") && !dto["winner"].is_none()) {
    state.winner = as_int(dto["winner"], "winner");
  }
  if (dto.contains("moveNumber")) {
    state.move_number = as_int(dto["moveNumber"], "moveNumber");
  }
  if (dto.contains("terminationReason")) {
    state.termination_reason = parse_termination_reason(dto["terminationReason"]);
  }
  validate_state(state);
  const int repetition_count =
      dto.contains("repetitionCount")
          ? as_int(dto["repetitionCount"], "repetitionCount")
          : 1;
  if (repetition_count <= 0) {
    throw py::value_error("repetitionCount must be positive");
  }
  state.position_counts = {{position_hash(state), repetition_count}};
  return state;
}

py::tuple position_to_python(const Position& position) {
  return py::make_tuple(position.row, position.col);
}

py::object state_to_python(const State& state) {
  py::list horizontal;
  for (const Wall& wall : state.horizontal_walls) {
    horizontal.append(position_to_python(wall));
  }
  py::list vertical;
  for (const Wall& wall : state.vertical_walls) {
    vertical.append(position_to_python(wall));
  }
  py::dict counts;
  for (const auto& entry : state.position_counts.materialize()) {
    counts[py::str(entry.first)] = entry.second;
  }
  const py::object state_class = py::module_::import("quoridor.game").attr("State");
  return state_class(
      py::arg("pawns") =
          py::make_tuple(position_to_python(state.pawns[0]),
                         position_to_python(state.pawns[1])),
      py::arg("walls_remaining") =
          py::make_tuple(state.walls_remaining[0], state.walls_remaining[1]),
      py::arg("horizontal_walls") =
          py::module_::import("builtins").attr("frozenset")(horizontal),
      py::arg("vertical_walls") =
          py::module_::import("builtins").attr("frozenset")(vertical),
      py::arg("current_player") = state.current_player,
      py::arg("winner") =
          state.winner == kNoWinner ? py::none() : py::cast(state.winner),
      py::arg("move_number") = state.move_number,
      py::arg("position_counts") = counts,
      py::arg("termination_reason") =
          termination_reason_to_python(state.termination_reason));
}

py::dict state_to_dict(const State& state) {
  py::list pawns;
  for (const Position& pawn : state.pawns) {
    py::dict item;
    item["row"] = pawn.row;
    item["col"] = pawn.col;
    pawns.append(std::move(item));
  }
  py::list remaining;
  remaining.append(state.walls_remaining[0]);
  remaining.append(state.walls_remaining[1]);
  py::list horizontal;
  for (const Wall& wall : state.horizontal_walls) {
    horizontal.append(py::str(std::to_string(wall.row) + "," +
                              std::to_string(wall.col)));
  }
  py::list vertical;
  for (const Wall& wall : state.vertical_walls) {
    vertical.append(py::str(std::to_string(wall.row) + "," +
                            std::to_string(wall.col)));
  }

  py::dict dto;
  dto["codecVersion"] = kCodecVersion;
  dto["pawns"] = std::move(pawns);
  dto["wallsRemaining"] = std::move(remaining);
  dto["horizontalWalls"] = std::move(horizontal);
  dto["verticalWalls"] = std::move(vertical);
  dto["currentPlayer"] = state.current_player;
  dto["winner"] =
      state.winner == kNoWinner ? py::none() : py::cast(state.winner);
  dto["moveNumber"] = state.move_number;
  dto["terminationReason"] =
      termination_reason_to_python(state.termination_reason);
  dto["repetitionCount"] = current_repetition_count(state);
  return dto;
}

Evaluation parse_evaluation(const py::handle result) {
  if (PyUnicode_Check(result.ptr()) || PyBytes_Check(result.ptr()) ||
      !PySequence_Check(result.ptr())) {
    throw py::type_error("evaluator must return (policy_logits, value)");
  }
  const py::sequence pair = py::reinterpret_borrow<py::sequence>(result);
  if (py::len(pair) != 2) {
    throw py::value_error("evaluator must return (policy_logits, value)");
  }
  const py::handle raw_logits = pair[0];
  if (PyUnicode_Check(raw_logits.ptr()) || PyBytes_Check(raw_logits.ptr()) ||
      !PySequence_Check(raw_logits.ptr())) {
    throw py::type_error("policy logits must be a sequence");
  }
  const py::sequence logits = py::reinterpret_borrow<py::sequence>(raw_logits);
  if (py::len(logits) != kActionSpaceSize) {
    throw py::value_error("policy logits must contain 209 values");
  }
  Evaluation evaluation;
  for (int index = 0; index < kActionSpaceSize; ++index) {
    evaluation.policy_logits[static_cast<std::size_t>(index)] =
        py::cast<double>(logits[index]);
  }
  evaluation.value = py::cast<double>(pair[1]);
  return evaluation;
}

class PythonMCTS {
 public:
  PythonMCTS(py::object evaluator, const std::size_t simulations,
             const double c_base, const double c_init,
             const double dirichlet_epsilon,
             const std::optional<double> dirichlet_alpha,
             const double temperature, const bool add_root_noise,
             const std::uint64_t seed)
      : evaluator_(std::move(evaluator)),
        default_temperature_(temperature),
        default_root_noise_(add_root_noise) {
    MCTSConfig config;
    config.simulations = simulations;
    config.c_base = c_base;
    config.c_init = c_init;
    config.dirichlet_epsilon = dirichlet_epsilon;
    config.dirichlet_alpha = dirichlet_alpha;
    config.temperature = temperature;
    config.add_root_noise = add_root_noise;
    config.seed = seed;
    search_ = std::make_unique<MCTS>(
        BatchEvaluateCallback([this](const std::vector<State>& states) {
          return evaluate(states);
        }),
        config);
  }

  PythonMCTS(const PythonMCTS&) = delete;
  PythonMCTS& operator=(const PythonMCTS&) = delete;

  SearchResult run(const py::object& source,
                   const std::optional<double> temperature,
                   const std::optional<bool> add_root_noise) {
    const State state = py::isinstance<State>(source)
                            ? py::cast<State>(source)
                            : state_from_python(source);
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    if (temperature.has_value() || add_root_noise.has_value()) {
      return search_->run(state, temperature.value_or(default_temperature_),
                          add_root_noise.value_or(default_root_noise_));
    }
    return search_->run(state);
  }

  std::vector<SearchResult> run_batch(
      const py::sequence& sources,
      const std::optional<std::vector<double>>& temperatures,
      const std::optional<bool> add_root_noise) {
    std::vector<State> states;
    states.reserve(static_cast<std::size_t>(py::len(sources)));
    for (const py::handle source : sources) {
      states.push_back(py::isinstance<State>(source)
                           ? py::cast<State>(source)
                           : state_from_python(source));
    }
    std::vector<double> selected_temperatures = temperatures.value_or(
        std::vector<double>(states.size(), default_temperature_));
    const bool selected_noise = add_root_noise.value_or(default_root_noise_);
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    return search_->run_batch(states, selected_temperatures, selected_noise);
  }

  std::uint64_t create_session(const py::object& source) {
    const State state = py::isinstance<State>(source)
                            ? py::cast<State>(source)
                            : state_from_python(source);
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    return search_->create_session(state);
  }

  bool sync_session(const std::uint64_t session_id,
                    const py::object& source) {
    const State state = py::isinstance<State>(source)
                            ? py::cast<State>(source)
                            : state_from_python(source);
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    return search_->sync_session(session_id, state);
  }

  SearchResult run_session(const std::uint64_t session_id,
                           const py::object& source,
                           const std::optional<double> temperature,
                           const std::optional<bool> add_root_noise) {
    const State state = py::isinstance<State>(source)
                            ? py::cast<State>(source)
                            : state_from_python(source);
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    return search_->run_session(
        session_id, state, temperature.value_or(default_temperature_),
        add_root_noise.value_or(default_root_noise_));
  }

  std::vector<SearchResult> run_sessions(
      const std::vector<std::uint64_t>& session_ids,
      const py::sequence& sources,
      const std::optional<std::vector<double>>& temperatures,
      const std::optional<bool> add_root_noise) {
    std::vector<State> states;
    states.reserve(static_cast<std::size_t>(py::len(sources)));
    for (const py::handle source : sources) {
      states.push_back(py::isinstance<State>(source)
                           ? py::cast<State>(source)
                           : state_from_python(source));
    }
    std::vector<double> selected_temperatures = temperatures.value_or(
        std::vector<double>(states.size(), default_temperature_));
    const bool selected_noise = add_root_noise.value_or(default_root_noise_);
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    return search_->run_sessions(session_ids, states, selected_temperatures,
                                 selected_noise);
  }

  bool advance_session(const std::uint64_t session_id, const int action_index,
                       const py::object& source) {
    const State state = py::isinstance<State>(source)
                            ? py::cast<State>(source)
                            : state_from_python(source);
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    return search_->advance_session(session_id, action_index, state);
  }

  bool release_session(const std::uint64_t session_id) {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    return search_->release_session(session_id);
  }

  void clear_sessions() {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    search_->clear_sessions();
  }

  std::size_t session_count() {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    return search_->session_count();
  }

  std::size_t session_node_count(const std::uint64_t session_id) {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(search_mutex_);
    return search_->session_node_count(session_id);
  }

 private:
  std::vector<Evaluation> evaluate(const std::vector<State>& states) {
    py::gil_scoped_acquire acquire;
    std::vector<Evaluation> results;
    results.reserve(states.size());
    if (states.size() > 1 && py::hasattr(evaluator_, "evaluate_batch")) {
      py::list inputs;
      for (const State& state : states) {
        inputs.append(state_to_python(state));
      }
      const py::object raw_batch = evaluator_.attr("evaluate_batch")(inputs);
      if (!PySequence_Check(raw_batch.ptr())) {
        throw py::type_error("evaluate_batch must return a sequence");
      }
      const py::sequence batch =
          py::reinterpret_borrow<py::sequence>(raw_batch);
      if (static_cast<std::size_t>(py::len(batch)) != states.size()) {
        throw py::value_error(
            "evaluate_batch must return one (policy, value) pair per state");
      }
      for (const py::handle item : batch) {
        results.push_back(parse_evaluation(item));
      }
      return results;
    }
    for (const State& state : states) {
      results.push_back(
          parse_evaluation(evaluator_.attr("evaluate")(state_to_python(state))));
    }
    return results;
  }

  py::object evaluator_;
  std::unique_ptr<MCTS> search_;
  std::mutex search_mutex_;
  double default_temperature_ = 1.0;
  bool default_root_noise_ = true;
};

}  // namespace
}  // namespace quoridor

PYBIND11_MODULE(quoridor_native, module) {
  using namespace quoridor;
  module.doc() = "C++17 Quoridor rules and AlphaZero MCTS";
  module.attr("CODEC_VERSION") = kCodecVersion;
  module.attr("ACTION_SPACE_SIZE") = kActionSpaceSize;
  module.attr("LEGACY_INPUT_CHANNELS") = kLegacyInputChannels;
  module.attr("PHASE4_INPUT_CHANNELS") = kPhase4InputChannels;

  py::class_<State>(module, "NativeState")
      .def("__repr__", [](const State& state) {
        return "NativeState(" + position_hash(state) + ")";
      });

  py::class_<EdgeStatistics>(module, "EdgeStatistics")
      .def_readonly("action_index", &EdgeStatistics::action_index)
      .def_readonly("prior", &EdgeStatistics::prior)
      .def_readonly("visits", &EdgeStatistics::visits)
      .def_readonly("value_sum", &EdgeStatistics::value_sum)
      .def_readonly("q", &EdgeStatistics::q);

  py::class_<SearchResult>(module, "SearchResult")
      .def_readonly("action_index", &SearchResult::action_index)
      .def_property_readonly("policy", [](const SearchResult& result) {
        return std::vector<double>(result.policy.begin(), result.policy.end());
      })
      .def_property_readonly("visit_counts", [](const SearchResult& result) {
        return std::vector<std::uint64_t>(result.visit_counts.begin(),
                                          result.visit_counts.end());
      })
      .def_readonly("root_value", &SearchResult::root_value)
      .def_readonly("root_edges", &SearchResult::root_edges);

  module.def("state_from_dict", &state_from_dict, py::arg("dto"));
  module.def("state_to_dict", &state_to_dict, py::arg("state"));
  module.def("state_from_python", &state_from_python, py::arg("state"));
  module.def("state_to_python", &state_to_python, py::arg("state"));
  module.def("encode_states", &encode_states_from_python, py::arg("states"),
             py::arg("input_channels") = kPhase4InputChannels);
  module.def("legal_action_indices", &legal_action_indices,
             py::arg("state"), py::arg("player") = std::nullopt,
             py::call_guard<py::gil_scoped_release>());
  module.def("apply_action_index", &apply_action_index, py::arg("state"),
             py::arg("action_index"),
             py::call_guard<py::gil_scoped_release>());
  module.def("apply_legal_action_index", &apply_legal_action_index,
             py::arg("state"), py::arg("action_index"),
             py::call_guard<py::gil_scoped_release>());
  module.def("apply_legal_action_position_hash",
             &apply_legal_action_position_hash, py::arg("state"),
             py::arg("action_index"),
             py::call_guard<py::gil_scoped_release>());
  module.def("position_hash", &position_hash, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());
  module.def("canonicalize_state", &canonicalize_state, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());
  module.def("rotate_and_swap_state", &rotate_and_swap_state, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());
  module.def("mirror_state", &mirror_state, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());
  module.def("to_canonical_action_index", &to_canonical_action_index,
             py::arg("index"), py::arg("original_current_player"),
             py::call_guard<py::gil_scoped_release>());
  module.def("mirror_action_index", &mirror_action_index, py::arg("index"),
             py::call_guard<py::gil_scoped_release>());
  module.def("is_terminal", &is_terminal, py::arg("state"),
             py::call_guard<py::gil_scoped_release>());

  py::class_<PythonMCTS>(module, "MCTS")
      .def(py::init<py::object, std::size_t, double, double, double,
                    std::optional<double>, double, bool, std::uint64_t>(),
           py::arg("evaluator"), py::arg("simulations") = 800,
           py::arg("c_base") = 19652.0, py::arg("c_init") = 1.25,
           py::arg("dirichlet_epsilon") = 0.25,
           py::arg("dirichlet_alpha") = std::nullopt,
           py::arg("temperature") = 1.0,
           py::arg("add_root_noise") = true, py::arg("seed") = 0)
      .def("run", &PythonMCTS::run, py::arg("state"),
           py::arg("temperature") = std::nullopt,
           py::arg("add_root_noise") = std::nullopt)
      .def("run_batch", &PythonMCTS::run_batch, py::arg("states"),
           py::arg("temperatures") = std::nullopt,
           py::arg("add_root_noise") = std::nullopt)
      .def("create_session", &PythonMCTS::create_session, py::arg("state"))
      .def("sync_session", &PythonMCTS::sync_session,
           py::arg("session_id"), py::arg("state"))
      .def("run_session", &PythonMCTS::run_session,
           py::arg("session_id"), py::arg("state"),
           py::arg("temperature") = std::nullopt,
           py::arg("add_root_noise") = std::nullopt)
      .def("run_sessions", &PythonMCTS::run_sessions,
           py::arg("session_ids"), py::arg("states"),
           py::arg("temperatures") = std::nullopt,
           py::arg("add_root_noise") = std::nullopt)
      .def("advance_session", &PythonMCTS::advance_session,
           py::arg("session_id"), py::arg("action_index"),
           py::arg("next_state"))
      .def("release_session", &PythonMCTS::release_session,
           py::arg("session_id"))
      .def("clear_sessions", &PythonMCTS::clear_sessions)
      .def_property_readonly("session_count", &PythonMCTS::session_count)
      .def("session_node_count", &PythonMCTS::session_node_count,
           py::arg("session_id"));
}
