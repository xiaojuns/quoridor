#pragma once

#include <array>
#include <initializer_list>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace quoridor {

inline constexpr int kCodecVersion = 1;
inline constexpr int kBoardSize = 9;
inline constexpr int kWallGridSize = 8;
inline constexpr int kStartingWalls = 10;
inline constexpr int kRepetitionLimit = 3;
inline constexpr int kMaxHalfMoves = 256;

inline constexpr int kMoveActionOffset = 0;
inline constexpr int kMoveActionCount = 81;
inline constexpr int kHorizontalWallActionOffset = 81;
inline constexpr int kWallActionCount = 64;
inline constexpr int kVerticalWallActionOffset = 145;
inline constexpr int kActionSpaceSize = 209;
inline constexpr int kNoWinner = -1;

struct Position {
  int row = 0;
  int col = 0;

  bool operator==(const Position &other) const noexcept;
  bool operator!=(const Position &other) const noexcept;
  bool operator<(const Position &other) const noexcept;
};

using Wall = Position;

enum class ActionType {
  Move,
  Wall,
};

enum class Orientation {
  None,
  Horizontal,
  Vertical,
};

struct Action {
  ActionType type = ActionType::Move;
  Orientation orientation = Orientation::None;
  int row = 0;
  int col = 0;

  static Action move(int row, int col);
  static Action wall(Orientation orientation, int row, int col);
  bool operator==(const Action &other) const noexcept;
  bool operator!=(const Action &other) const noexcept;
};

enum class TerminationReason {
  None,
  Goal,
  Repetition,
  MoveLimit,
};

class PositionCounts {
 public:
  using Map = std::unordered_map<std::string, int>;
  using Value = Map::value_type;

  PositionCounts();
  explicit PositionCounts(Map counts);
  PositionCounts(std::initializer_list<Value> counts);
  PositionCounts &operator=(std::initializer_list<Value> counts);

  void clear();
  void assign(Map counts);
  void set(const std::string &key, int value);
  int get(const std::string &key) const;
  int increment(const std::string &key);
  Map materialize() const;

  bool operator==(const PositionCounts &other) const;
  bool operator!=(const PositionCounts &other) const;

 private:
  struct Delta;
  std::shared_ptr<const Map> base_;
  std::shared_ptr<const Delta> deltas_;
};

struct State {
  std::array<Position, 2> pawns{{Position{8, 4}, Position{0, 4}}};
  std::array<int, 2> walls_remaining{{kStartingWalls, kStartingWalls}};
  std::set<Wall> horizontal_walls;
  std::set<Wall> vertical_walls;
  int current_player = 0;
  int winner = kNoWinner;
  int move_number = 0;
  PositionCounts position_counts;
  TerminationReason termination_reason = TerminationReason::None;

  bool operator==(const State &other) const;
  bool operator!=(const State &other) const;
};

void validate_state(const State &state);
State create_initial_state(int starting_player = 0);

Orientation parse_orientation(char orientation);
char orientation_code(Orientation orientation);
std::string termination_reason_name(TerminationReason reason);

int encode_move(int row, int col);
int encode_wall(Orientation orientation, int row, int col);
int encode_wall(char orientation, int row, int col);
int encode_action(const Action &action);
Action decode_action(int index);

std::string position_hash(const State &state);
int current_repetition_count(const State &state);
State rotate_and_swap_state(const State &state);
State canonicalize_state(const State &state);
Action to_canonical_action(const Action &action, int original_current_player);
Action from_canonical_action(const Action &action, int original_current_player);
int to_canonical_action_index(int index, int original_current_player);
int from_canonical_action_index(int index, int original_current_player);
State mirror_state(const State &state);
Action mirror_action(const Action &action);
int mirror_action_index(int index);

bool is_passage_blocked(const State &state, int from_row, int from_col,
                        int to_row, int to_col);
std::vector<Position>
legal_pawn_moves(const State &state, std::optional<int> player = std::nullopt);
bool is_wall_structurally_valid(const State &state, Orientation orientation,
                                int row, int col);
bool is_wall_structurally_valid(const State &state, char orientation, int row,
                                int col);
double shortest_path_distance(const State &state, int player);
bool has_path_to_goal(const State &state, int player);
bool is_legal_wall(const State &state, Orientation orientation, int row,
                   int col, std::optional<int> player = std::nullopt);
bool is_legal_wall(const State &state, char orientation, int row, int col,
                   std::optional<int> player = std::nullopt);
std::vector<int> legal_wall_indices(const State &state,
                                    std::optional<int> player = std::nullopt);
std::vector<int> legal_action_indices(const State &state,
                                      std::optional<int> player = std::nullopt);
int get_winner(const State &state);
bool is_terminal(const State &state);
std::optional<double>
winner_value(const State &state,
             std::optional<int> perspective_player = std::nullopt);
State apply_action_index(const State &state, int action_index);
// Training hot path: the caller must have obtained the action from
// legal_action_indices(state). The transition and terminal rules remain exact.
State apply_legal_action_index(const State &state, int action_index);
std::string apply_legal_action_position_hash(const State &state,
                                             int action_index);

} // namespace quoridor
