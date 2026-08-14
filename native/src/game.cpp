#include "quoridor/game.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace quoridor {
namespace {

constexpr std::array<Position, 4> kDirections{{
    Position{-1, 0},
    Position{1, 0},
    Position{0, -1},
    Position{0, 1},
}};

bool inside(int row, int col) noexcept {
  return row >= 0 && row < kBoardSize && col >= 0 && col < kBoardSize;
}

void require_player(int player) {
  if (player != 0 && player != 1) {
    throw std::invalid_argument("player must be 0 or 1");
  }
}

void require_coordinate(int value, int size, const char *label) {
  if (value < 0 || value >= size) {
    std::ostringstream message;
    message << label << " must be between 0 and " << (size - 1);
    throw std::invalid_argument(message.str());
  }
}

int selected_player(const State &state, std::optional<int> player) {
  const int result = player.value_or(state.current_player);
  require_player(result);
  return result;
}

bool contains(const std::set<Position> &positions, int row, int col) {
  return positions.find(Position{row, col}) != positions.end();
}

std::uint64_t wall_mask(const std::set<Wall> &walls) {
  std::uint64_t mask = 0;
  for (const Wall &wall : walls) {
    if (wall.row < 0 || wall.row >= kWallGridSize || wall.col < 0 ||
        wall.col >= kWallGridSize) {
      throw std::invalid_argument("wall coordinate is outside 0..7");
    }
    mask |= std::uint64_t{1}
            << static_cast<unsigned>(wall.row * kWallGridSize + wall.col);
  }
  return mask;
}

bool mask_contains(std::uint64_t mask, int row, int col) noexcept {
  if (row < 0 || row >= kWallGridSize || col < 0 ||
      col >= kWallGridSize) {
    return false;
  }
  const unsigned index =
      static_cast<unsigned>(row * kWallGridSize + col);
  return (mask & (std::uint64_t{1} << index)) != 0;
}

bool passage_blocked_with_masks(std::uint64_t horizontal,
                                std::uint64_t vertical, int from_row,
                                int from_col, int to_row, int to_col) noexcept {
  if (!inside(from_row, from_col) || !inside(to_row, to_col) ||
      std::abs(to_row - from_row) + std::abs(to_col - from_col) != 1) {
    return true;
  }
  if (to_row != from_row) {
    const int boundary_row = std::min(from_row, to_row);
    return mask_contains(horizontal, boundary_row, from_col) ||
           mask_contains(horizontal, boundary_row, from_col - 1);
  }
  const int boundary_col = std::min(from_col, to_col);
  return mask_contains(vertical, from_row, boundary_col) ||
         mask_contains(vertical, from_row - 1, boundary_col);
}

double shortest_path_with_masks(Position start, int goal_row,
                                std::uint64_t horizontal,
                                std::uint64_t vertical) {
  if (start.row == goal_row) {
    return 0.0;
  }
  std::array<int, kBoardSize * kBoardSize> distances;
  distances.fill(-1);
  const auto board_index = [](const Position &position) {
    return position.row * kBoardSize + position.col;
  };
  std::array<Position, kBoardSize * kBoardSize> queue;
  std::size_t head = 0;
  std::size_t tail = 0;
  queue[tail++] = start;
  distances[board_index(start)] = 0;

  while (head < tail) {
    const Position current = queue[head++];
    const int next_distance = distances[board_index(current)] + 1;
    for (const Position &direction : kDirections) {
      const Position destination{current.row + direction.row,
                                 current.col + direction.col};
      if (!inside(destination.row, destination.col) ||
          distances[board_index(destination)] >= 0 ||
          passage_blocked_with_masks(horizontal, vertical, current.row,
                                     current.col, destination.row,
                                     destination.col)) {
        continue;
      }
      if (destination.row == goal_row) {
        return static_cast<double>(next_distance);
      }
      distances[board_index(destination)] = next_distance;
      queue[tail++] = destination;
    }
  }
  return std::numeric_limits<double>::infinity();
}

void validate_action(const Action &action) {
  if (action.type == ActionType::Move) {
    require_coordinate(action.row, kBoardSize, "move row");
    require_coordinate(action.col, kBoardSize, "move col");
    return;
  }
  if (action.type != ActionType::Wall ||
      (action.orientation != Orientation::Horizontal &&
       action.orientation != Orientation::Vertical)) {
    throw std::invalid_argument("invalid action type or wall orientation");
  }
  require_coordinate(action.row, kWallGridSize, "wall row");
  require_coordinate(action.col, kWallGridSize, "wall col");
}

void append_wall_array(std::ostringstream &output,
                       const std::set<Wall> &walls) {
  output << '[';
  bool first = true;
  for (const Wall &wall : walls) {
    if (!first) {
      output << ',';
    }
    first = false;
    output << '\"' << wall.row << ',' << wall.col << '\"';
  }
  output << ']';
}

Position rotate_pawn(const Position &pawn) {
  return Position{kBoardSize - 1 - pawn.row, kBoardSize - 1 - pawn.col};
}

Wall rotate_wall(const Wall &wall) {
  return Wall{kWallGridSize - 1 - wall.row, kWallGridSize - 1 - wall.col};
}

Position mirror_pawn(const Position &pawn) {
  return Position{pawn.row, kBoardSize - 1 - pawn.col};
}

Wall mirror_wall(const Wall &wall) {
  return Wall{wall.row, kWallGridSize - 1 - wall.col};
}

std::set<Wall> transform_walls(const std::set<Wall> &source,
                               Wall (*transform)(const Wall &)) {
  std::set<Wall> result;
  for (const Wall &wall : source) {
    result.insert(transform(wall));
  }
  return result;
}

enum class HistoryTransform {
  Identity,
  RotateAndSwap,
  Mirror,
};

class PositionHashParser {
 public:
  explicit PositionHashParser(const std::string &value) : value_(value) {}

  State parse() {
    State result;
    require('[');
    const int version = integer();
    if (version != kCodecVersion) {
      throw std::invalid_argument(
          "position_counts contains an unsupported position hash");
    }
    comma();
    result.current_player = integer();
    comma();
    result.pawns[0].row = integer();
    comma();
    result.pawns[0].col = integer();
    comma();
    result.pawns[1].row = integer();
    comma();
    result.pawns[1].col = integer();
    comma();
    result.walls_remaining[0] = integer();
    comma();
    result.walls_remaining[1] = integer();
    comma();
    result.horizontal_walls = wall_array();
    comma();
    result.vertical_walls = wall_array();
    require(']');
    whitespace();
    if (offset_ != value_.size()) {
      invalid_hash();
    }
    validate_state(result);
    return result;
  }

 private:
  [[noreturn]] static void invalid_hash() {
    throw std::invalid_argument(
        "position_counts contains an invalid position hash");
  }

  void whitespace() {
    while (offset_ < value_.size() &&
           (value_[offset_] == ' ' || value_[offset_] == '\t' ||
            value_[offset_] == '\n' || value_[offset_] == '\r')) {
      ++offset_;
    }
  }

  void require(char expected) {
    whitespace();
    if (offset_ >= value_.size() || value_[offset_] != expected) {
      invalid_hash();
    }
    ++offset_;
  }

  void comma() { require(','); }

  int integer() {
    whitespace();
    const char *first = value_.data() + offset_;
    const char *last = value_.data() + value_.size();
    int result = 0;
    const auto parsed = std::from_chars(first, last, result);
    if (parsed.ec != std::errc{} || parsed.ptr == first) {
      invalid_hash();
    }
    offset_ = static_cast<std::size_t>(parsed.ptr - value_.data());
    return result;
  }

  static int coordinate_component(const std::string &value) {
    int result = 0;
    const char *first = value.data();
    const char *last = first + value.size();
    const auto parsed = std::from_chars(first, last, result);
    if (parsed.ec != std::errc{} || parsed.ptr != last) {
      invalid_hash();
    }
    return result;
  }

  Wall wall() {
    require('"');
    const std::size_t start = offset_;
    while (offset_ < value_.size() && value_[offset_] != '"') {
      if (value_[offset_] == '\\') {
        invalid_hash();
      }
      ++offset_;
    }
    if (offset_ >= value_.size()) {
      invalid_hash();
    }
    const std::string key = value_.substr(start, offset_ - start);
    ++offset_;
    const std::size_t separator = key.find(',');
    if (separator == std::string::npos ||
        key.find(',', separator + 1) != std::string::npos) {
      invalid_hash();
    }
    const int row = coordinate_component(key.substr(0, separator));
    const int col = coordinate_component(key.substr(separator + 1));
    require_coordinate(row, kWallGridSize, "wall row");
    require_coordinate(col, kWallGridSize, "wall col");
    return Wall{row, col};
  }

  std::set<Wall> wall_array() {
    std::set<Wall> result;
    require('[');
    whitespace();
    if (offset_ < value_.size() && value_[offset_] == ']') {
      ++offset_;
      return result;
    }
    while (true) {
      if (!result.insert(wall()).second) {
        invalid_hash();
      }
      whitespace();
      if (offset_ < value_.size() && value_[offset_] == ']') {
        ++offset_;
        return result;
      }
      comma();
    }
  }

  const std::string &value_;
  std::size_t offset_ = 0;
};

std::string transform_position_hash(const std::string &value,
                                    HistoryTransform transform) {
  if (transform == HistoryTransform::Identity) {
    return value;
  }

  const State decoded = PositionHashParser(value).parse();
  State transformed;
  transformed.current_player = decoded.current_player;
  transformed.pawns = decoded.pawns;
  transformed.walls_remaining = decoded.walls_remaining;
  transformed.horizontal_walls = decoded.horizontal_walls;
  transformed.vertical_walls = decoded.vertical_walls;
  if (transform == HistoryTransform::RotateAndSwap) {
    transformed.current_player = 1 - decoded.current_player;
    transformed.pawns = std::array<Position, 2>{
        {rotate_pawn(decoded.pawns[1]), rotate_pawn(decoded.pawns[0])}};
    transformed.walls_remaining = std::array<int, 2>{
        {decoded.walls_remaining[1], decoded.walls_remaining[0]}};
    transformed.horizontal_walls =
        transform_walls(decoded.horizontal_walls, rotate_wall);
    transformed.vertical_walls =
        transform_walls(decoded.vertical_walls, rotate_wall);
  } else {
    transformed.pawns = std::array<Position, 2>{
        {mirror_pawn(decoded.pawns[0]), mirror_pawn(decoded.pawns[1])}};
    transformed.horizontal_walls =
        transform_walls(decoded.horizontal_walls, mirror_wall);
    transformed.vertical_walls =
        transform_walls(decoded.vertical_walls, mirror_wall);
  }
  return position_hash(transformed);
}

State transformed_state(const State &source, std::array<Position, 2> pawns,
                        std::array<int, 2> walls_remaining,
                        std::set<Wall> horizontal_walls,
                        std::set<Wall> vertical_walls, int current_player,
                        int winner, HistoryTransform history_transform) {
  const int repetition_count = current_repetition_count(source);
  State result;
  result.pawns = std::move(pawns);
  result.walls_remaining = walls_remaining;
  result.horizontal_walls = std::move(horizontal_walls);
  result.vertical_walls = std::move(vertical_walls);
  result.current_player = current_player;
  result.winner = winner;
  result.move_number = source.move_number;
  result.termination_reason = source.termination_reason;
  PositionCounts::Map counts;
  for (const auto &entry : source.position_counts.materialize()) {
    if (entry.first.empty()) {
      throw std::invalid_argument(
          "position_counts keys must be non-empty strings");
    }
    if (entry.second <= 0) {
      throw std::invalid_argument(
          "position_counts values must be positive integers");
    }
    const std::string transformed_hash =
        transform_position_hash(entry.first, history_transform);
    if (!counts.emplace(transformed_hash, entry.second).second) {
      throw std::invalid_argument(
          "transformed repetition history contains a collision");
    }
  }
  const std::string current_hash = position_hash(result);
  if (counts.find(current_hash) == counts.end()) {
    counts.emplace(current_hash, repetition_count);
  }
  result.position_counts.assign(std::move(counts));
  return result;
}

Action rotate_action(const Action &action) {
  validate_action(action);
  Action transformed = action;
  if (action.type == ActionType::Move) {
    transformed.row = kBoardSize - 1 - action.row;
    transformed.col = kBoardSize - 1 - action.col;
  } else {
    transformed.row = kWallGridSize - 1 - action.row;
    transformed.col = kWallGridSize - 1 - action.col;
  }
  return transformed;
}

} // namespace

bool Position::operator==(const Position &other) const noexcept {
  return row == other.row && col == other.col;
}

bool Position::operator!=(const Position &other) const noexcept {
  return !(*this == other);
}

bool Position::operator<(const Position &other) const noexcept {
  return row < other.row || (row == other.row && col < other.col);
}

Action Action::move(int row, int col) {
  return Action{ActionType::Move, Orientation::None, row, col};
}

Action Action::wall(Orientation orientation, int row, int col) {
  if (orientation != Orientation::Horizontal &&
      orientation != Orientation::Vertical) {
    throw std::invalid_argument(
        "wall orientation must be horizontal or vertical");
  }
  return Action{ActionType::Wall, orientation, row, col};
}

bool Action::operator==(const Action &other) const noexcept {
  return type == other.type && orientation == other.orientation &&
         row == other.row && col == other.col;
}

bool Action::operator!=(const Action &other) const noexcept {
  return !(*this == other);
}

struct PositionCounts::Delta {
  std::string key;
  int increment = 0;
  std::shared_ptr<const Delta> previous;
};

PositionCounts::PositionCounts() : base_(std::make_shared<const Map>()) {}

PositionCounts::PositionCounts(Map counts)
    : base_(std::make_shared<const Map>(std::move(counts))) {}

PositionCounts::PositionCounts(std::initializer_list<Value> counts)
    : PositionCounts(Map(counts)) {}

PositionCounts &
PositionCounts::operator=(std::initializer_list<Value> counts) {
  assign(Map(counts));
  return *this;
}

void PositionCounts::clear() {
  base_ = std::make_shared<const Map>();
  deltas_.reset();
}

void PositionCounts::assign(Map counts) {
  base_ = std::make_shared<const Map>(std::move(counts));
  deltas_.reset();
}

void PositionCounts::set(const std::string &key, int value) {
  Map counts = materialize();
  counts[key] = value;
  assign(std::move(counts));
}

int PositionCounts::get(const std::string &key) const {
  int count = 0;
  const auto found = base_->find(key);
  if (found != base_->end()) {
    count = found->second;
  }
  for (auto delta = deltas_; delta; delta = delta->previous) {
    if (delta->key == key) {
      count += delta->increment;
    }
  }
  return count;
}

int PositionCounts::increment(const std::string &key) {
  const int count = get(key) + 1;
  deltas_ = std::make_shared<const Delta>(Delta{key, 1, deltas_});
  return count;
}

PositionCounts::Map PositionCounts::materialize() const {
  Map result = *base_;
  for (auto delta = deltas_; delta; delta = delta->previous) {
    result[delta->key] += delta->increment;
  }
  return result;
}

bool PositionCounts::operator==(const PositionCounts &other) const {
  return materialize() == other.materialize();
}

bool PositionCounts::operator!=(const PositionCounts &other) const {
  return !(*this == other);
}

bool State::operator==(const State &other) const {
  return pawns == other.pawns && walls_remaining == other.walls_remaining &&
         horizontal_walls == other.horizontal_walls &&
         vertical_walls == other.vertical_walls &&
         current_player == other.current_player && winner == other.winner &&
         move_number == other.move_number &&
         position_counts == other.position_counts &&
         termination_reason == other.termination_reason;
}

bool State::operator!=(const State &other) const { return !(*this == other); }

void validate_state(const State &state) {
  require_player(state.current_player);
  if (state.winner != kNoWinner) {
    require_player(state.winner);
  }
  for (const Position &pawn : state.pawns) {
    if (!inside(pawn.row, pawn.col)) {
      throw std::invalid_argument("pawns must contain in-board coordinates");
    }
  }
  if (state.walls_remaining[0] < 0 || state.walls_remaining[1] < 0) {
    throw std::invalid_argument("walls_remaining must be non-negative");
  }
  if (state.move_number < 0) {
    throw std::invalid_argument("move_number must be non-negative");
  }
  wall_mask(state.horizontal_walls);
  wall_mask(state.vertical_walls);
}

State create_initial_state(int starting_player) {
  require_player(starting_player);
  State state;
  state.current_player = starting_player;
  state.position_counts = {{position_hash(state), 1}};
  return state;
}

Orientation parse_orientation(char orientation) {
  if (orientation == 'h') {
    return Orientation::Horizontal;
  }
  if (orientation == 'v') {
    return Orientation::Vertical;
  }
  throw std::invalid_argument("wall orientation must be 'h' or 'v'");
}

char orientation_code(Orientation orientation) {
  if (orientation == Orientation::Horizontal) {
    return 'h';
  }
  if (orientation == Orientation::Vertical) {
    return 'v';
  }
  throw std::invalid_argument("action does not have a wall orientation");
}

std::string termination_reason_name(TerminationReason reason) {
  switch (reason) {
  case TerminationReason::None:
    return "";
  case TerminationReason::Goal:
    return "goal";
  case TerminationReason::Repetition:
    return "repetition";
  case TerminationReason::MoveLimit:
    return "move-limit";
  }
  throw std::invalid_argument("unknown termination reason");
}

int encode_move(int row, int col) {
  require_coordinate(row, kBoardSize, "move row");
  require_coordinate(col, kBoardSize, "move col");
  return kMoveActionOffset + row * kBoardSize + col;
}

int encode_wall(Orientation orientation, int row, int col) {
  if (orientation != Orientation::Horizontal &&
      orientation != Orientation::Vertical) {
    throw std::invalid_argument(
        "wall orientation must be horizontal or vertical");
  }
  require_coordinate(row, kWallGridSize, "wall row");
  require_coordinate(col, kWallGridSize, "wall col");
  const int offset = orientation == Orientation::Horizontal
                         ? kHorizontalWallActionOffset
                         : kVerticalWallActionOffset;
  return offset + row * kWallGridSize + col;
}

int encode_wall(char orientation, int row, int col) {
  return encode_wall(parse_orientation(orientation), row, col);
}

int encode_action(const Action &action) {
  validate_action(action);
  if (action.type == ActionType::Move) {
    return encode_move(action.row, action.col);
  }
  return encode_wall(action.orientation, action.row, action.col);
}

Action decode_action(int index) {
  if (index < 0 || index >= kActionSpaceSize) {
    throw std::invalid_argument("action index must be between 0 and 208");
  }
  if (index < kHorizontalWallActionOffset) {
    return Action::move(index / kBoardSize, index % kBoardSize);
  }

  const bool horizontal = index < kVerticalWallActionOffset;
  const int offset =
      horizontal ? kHorizontalWallActionOffset : kVerticalWallActionOffset;
  const int wall_index = index - offset;
  return Action::wall(horizontal ? Orientation::Horizontal
                                 : Orientation::Vertical,
                      wall_index / kWallGridSize, wall_index % kWallGridSize);
}

std::string position_hash(const State &state) {
  validate_state(state);
  std::ostringstream output;
  output << '[' << kCodecVersion << ',' << state.current_player << ','
         << state.pawns[0].row << ',' << state.pawns[0].col << ','
         << state.pawns[1].row << ',' << state.pawns[1].col << ','
         << state.walls_remaining[0] << ',' << state.walls_remaining[1] << ',';
  append_wall_array(output, state.horizontal_walls);
  output << ',';
  append_wall_array(output, state.vertical_walls);
  output << ']';
  return output.str();
}

int current_repetition_count(const State &state) {
  const int count = state.position_counts.get(position_hash(state));
  return count > 0 ? count : 1;
}

State rotate_and_swap_state(const State &state) {
  validate_state(state);
  const int winner = state.winner == kNoWinner ? kNoWinner : 1 - state.winner;
  return transformed_state(
      state,
      std::array<Position, 2>{
          {rotate_pawn(state.pawns[1]), rotate_pawn(state.pawns[0])}},
      std::array<int, 2>{{state.walls_remaining[1], state.walls_remaining[0]}},
      transform_walls(state.horizontal_walls, rotate_wall),
      transform_walls(state.vertical_walls, rotate_wall),
      1 - state.current_player, winner,
      HistoryTransform::RotateAndSwap);
}

State canonicalize_state(const State &state) {
  validate_state(state);
  if (state.current_player == 1) {
    return rotate_and_swap_state(state);
  }
  return transformed_state(state, state.pawns, state.walls_remaining,
                           state.horizontal_walls, state.vertical_walls, 0,
                           state.winner, HistoryTransform::Identity);
}

Action to_canonical_action(const Action &action, int original_current_player) {
  require_player(original_current_player);
  validate_action(action);
  return original_current_player == 0 ? action : rotate_action(action);
}

Action from_canonical_action(const Action &action,
                             int original_current_player) {
  return to_canonical_action(action, original_current_player);
}

int to_canonical_action_index(int index, int original_current_player) {
  return encode_action(
      to_canonical_action(decode_action(index), original_current_player));
}

int from_canonical_action_index(int index, int original_current_player) {
  return to_canonical_action_index(index, original_current_player);
}

State mirror_state(const State &state) {
  validate_state(state);
  return transformed_state(
      state,
      std::array<Position, 2>{
          {mirror_pawn(state.pawns[0]), mirror_pawn(state.pawns[1])}},
      state.walls_remaining,
      transform_walls(state.horizontal_walls, mirror_wall),
      transform_walls(state.vertical_walls, mirror_wall), state.current_player,
      state.winner, HistoryTransform::Mirror);
}

Action mirror_action(const Action &action) {
  validate_action(action);
  Action transformed = action;
  transformed.col = action.type == ActionType::Move
                        ? kBoardSize - 1 - action.col
                        : kWallGridSize - 1 - action.col;
  return transformed;
}

int mirror_action_index(int index) {
  return encode_action(mirror_action(decode_action(index)));
}

bool is_passage_blocked(const State &state, int from_row, int from_col,
                        int to_row, int to_col) {
  if (!inside(from_row, from_col) || !inside(to_row, to_col) ||
      std::abs(to_row - from_row) + std::abs(to_col - from_col) != 1) {
    return true;
  }

  if (to_row != from_row) {
    const int boundary_row = std::min(from_row, to_row);
    return contains(state.horizontal_walls, boundary_row, from_col) ||
           contains(state.horizontal_walls, boundary_row, from_col - 1);
  }

  const int boundary_col = std::min(from_col, to_col);
  return contains(state.vertical_walls, from_row, boundary_col) ||
         contains(state.vertical_walls, from_row - 1, boundary_col);
}

std::vector<Position> legal_pawn_moves(const State &state,
                                       std::optional<int> player) {
  const int selected = selected_player(state, player);
  if (get_winner(state) != kNoWinner) {
    return {};
  }

  const Position pawn = state.pawns[selected];
  const Position opponent = state.pawns[1 - selected];
  std::vector<Position> moves;

  for (const Position &direction : kDirections) {
    const Position adjacent{pawn.row + direction.row, pawn.col + direction.col};
    if (is_passage_blocked(state, pawn.row, pawn.col, adjacent.row,
                           adjacent.col)) {
      continue;
    }
    if (adjacent != opponent) {
      moves.push_back(adjacent);
      continue;
    }

    const Position jump{opponent.row + direction.row,
                        opponent.col + direction.col};
    if (!is_passage_blocked(state, opponent.row, opponent.col, jump.row,
                            jump.col)) {
      moves.push_back(jump);
      continue;
    }

    const std::array<Position, 2> side_directions =
        direction.row == 0
            ? std::array<Position, 2>{{Position{-1, 0}, Position{1, 0}}}
            : std::array<Position, 2>{{Position{0, -1}, Position{0, 1}}};
    for (const Position &side : side_directions) {
      const Position diagonal{opponent.row + side.row, opponent.col + side.col};
      if (!is_passage_blocked(state, opponent.row, opponent.col, diagonal.row,
                              diagonal.col)) {
        moves.push_back(diagonal);
      }
    }
  }
  return moves;
}

bool is_wall_structurally_valid(const State &state, Orientation orientation,
                                int row, int col) {
  if ((orientation != Orientation::Horizontal &&
       orientation != Orientation::Vertical) ||
      row < 0 || row >= kWallGridSize || col < 0 || col >= kWallGridSize) {
    return false;
  }

  if (orientation == Orientation::Horizontal) {
    for (int candidate : {col - 1, col, col + 1}) {
      if (contains(state.horizontal_walls, row, candidate)) {
        return false;
      }
    }
    return !contains(state.vertical_walls, row, col);
  }

  for (int candidate : {row - 1, row, row + 1}) {
    if (contains(state.vertical_walls, candidate, col)) {
      return false;
    }
  }
  return !contains(state.horizontal_walls, row, col);
}

bool is_wall_structurally_valid(const State &state, char orientation, int row,
                                int col) {
  if (orientation != 'h' && orientation != 'v') {
    return false;
  }
  return is_wall_structurally_valid(state,
                                    orientation == 'h' ? Orientation::Horizontal
                                                       : Orientation::Vertical,
                                    row, col);
}

double shortest_path_distance(const State &state, int player) {
  require_player(player);
  const int goal_row = player == 0 ? 0 : kBoardSize - 1;
  return shortest_path_with_masks(state.pawns[player], goal_row,
                                  wall_mask(state.horizontal_walls),
                                  wall_mask(state.vertical_walls));
}

bool has_path_to_goal(const State &state, int player) {
  return !std::isinf(shortest_path_distance(state, player));
}

bool is_legal_wall(const State &state, Orientation orientation, int row,
                   int col, std::optional<int> player) {
  int selected;
  try {
    selected = selected_player(state, player);
  } catch (const std::invalid_argument &) {
    return false;
  }
  if (get_winner(state) != kNoWinner || state.walls_remaining[selected] <= 0 ||
      !is_wall_structurally_valid(state, orientation, row, col)) {
    return false;
  }

  std::uint64_t horizontal = wall_mask(state.horizontal_walls);
  std::uint64_t vertical = wall_mask(state.vertical_walls);
  const std::uint64_t added = std::uint64_t{1}
                              << static_cast<unsigned>(row * kWallGridSize + col);
  if (orientation == Orientation::Horizontal) {
    horizontal |= added;
  } else {
    vertical |= added;
  }
  return !std::isinf(shortest_path_with_masks(state.pawns[0], 0, horizontal,
                                               vertical)) &&
         !std::isinf(shortest_path_with_masks(state.pawns[1], kBoardSize - 1,
                                               horizontal, vertical));
}

bool is_legal_wall(const State &state, char orientation, int row, int col,
                   std::optional<int> player) {
  if (orientation != 'h' && orientation != 'v') {
    return false;
  }
  return is_legal_wall(state,
                       orientation == 'h' ? Orientation::Horizontal
                                          : Orientation::Vertical,
                       row, col, player);
}

std::vector<int> legal_wall_indices(const State &state,
                                    std::optional<int> player) {
  const int selected = selected_player(state, player);
  if (get_winner(state) != kNoWinner || state.walls_remaining[selected] <= 0) {
    return {};
  }

  std::vector<int> indices;
  for (Orientation orientation :
       {Orientation::Horizontal, Orientation::Vertical}) {
    for (int row = 0; row < kWallGridSize; ++row) {
      for (int col = 0; col < kWallGridSize; ++col) {
        if (is_legal_wall(state, orientation, row, col, selected)) {
          indices.push_back(encode_wall(orientation, row, col));
        }
      }
    }
  }
  return indices;
}

std::vector<int> legal_action_indices(const State &state,
                                      std::optional<int> player) {
  const int selected = selected_player(state, player);
  if (get_winner(state) != kNoWinner) {
    return {};
  }

  std::vector<int> indices;
  for (const Position &move : legal_pawn_moves(state, selected)) {
    indices.push_back(encode_move(move.row, move.col));
  }
  const std::vector<int> walls = legal_wall_indices(state, selected);
  indices.insert(indices.end(), walls.begin(), walls.end());
  std::sort(indices.begin(), indices.end());
  return indices;
}

int get_winner(const State &state) {
  if (state.winner != kNoWinner) {
    return state.winner;
  }
  if (state.pawns[0].row == 0) {
    return 0;
  }
  if (state.pawns[1].row == kBoardSize - 1) {
    return 1;
  }
  return kNoWinner;
}

bool is_terminal(const State &state) { return get_winner(state) != kNoWinner; }

std::optional<double> winner_value(const State &state,
                                   std::optional<int> perspective_player) {
  const int perspective = selected_player(state, perspective_player);
  const int winner = get_winner(state);
  if (winner == kNoWinner) {
    return std::nullopt;
  }
  return winner == perspective ? 1.0 : -1.0;
}

namespace {

State apply_known_action(const State &state, const Action &action,
                         std::string *resulting_hash = nullptr) {
  const int player = state.current_player;
  State next = state;
  if (action.type == ActionType::Move) {
    next.pawns[player] = Position{action.row, action.col};
  } else {
    if (action.orientation == Orientation::Horizontal) {
      next.horizontal_walls.insert(Wall{action.row, action.col});
    } else {
      next.vertical_walls.insert(Wall{action.row, action.col});
    }
    --next.walls_remaining[player];
  }

  next.current_player = 1 - player;
  next.winner = kNoWinner;
  ++next.move_number;
  next.termination_reason = TerminationReason::None;

  const int goal_winner = get_winner(next);
  if (goal_winner != kNoWinner) {
    next.winner = goal_winner;
    next.termination_reason = TerminationReason::Goal;
    if (resulting_hash != nullptr) {
      *resulting_hash = position_hash(next);
    }
    return next;
  }

  const std::string next_hash = position_hash(next);
  if (resulting_hash != nullptr) {
    *resulting_hash = next_hash;
  }
  const int occurrence = next.position_counts.increment(next_hash);
  if (occurrence >= kRepetitionLimit) {
    next.winner = 1 - player;
    next.termination_reason = TerminationReason::Repetition;
    return next;
  }
  if (next.move_number >= kMaxHalfMoves) {
    next.winner = 1 - player;
    next.termination_reason = TerminationReason::MoveLimit;
  }
  return next;
}

} // namespace

State apply_action_index(const State &state, int action_index) {
  validate_state(state);
  if (is_terminal(state)) {
    throw std::invalid_argument(
        "cannot apply an action after the game has ended");
  }

  const Action action = decode_action(action_index);
  const int player = state.current_player;

  if (action.type == ActionType::Move) {
    const Position destination{action.row, action.col};
    const std::vector<Position> moves = legal_pawn_moves(state, player);
    if (std::find(moves.begin(), moves.end(), destination) == moves.end()) {
      std::ostringstream message;
      message << "illegal pawn move to " << action.row << ',' << action.col;
      throw std::invalid_argument(message.str());
    }
  } else {
    if (!is_legal_wall(state, action.orientation, action.row, action.col,
                       player)) {
      throw std::invalid_argument("illegal wall placement");
    }
  }
  return apply_known_action(state, action);
}

State apply_legal_action_index(const State &state, int action_index) {
  if (is_terminal(state)) {
    throw std::invalid_argument(
        "cannot apply an action after the game has ended");
  }
  return apply_known_action(state, decode_action(action_index));
}

std::string apply_legal_action_position_hash(const State &state,
                                             int action_index) {
  if (is_terminal(state)) {
    throw std::invalid_argument(
        "cannot apply an action after the game has ended");
  }
  std::string resulting_hash;
  apply_known_action(state, decode_action(action_index), &resulting_hash);
  return resulting_hash;
}

} // namespace quoridor
