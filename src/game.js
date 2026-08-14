import { positionHash } from "./alphazero/state-codec.js";

export const BOARD_SIZE = 9;
export const STARTING_WALLS = 10;
export const REPETITION_LIMIT = 3;
export const MAX_HALF_MOVES = 256;

const WALL_GRID_SIZE = BOARD_SIZE - 1;
const DIRECTIONS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

function isPlayer(player) {
  return player === 0 || player === 1;
}

function isIntegerInRange(value, min, max) {
  return Number.isInteger(value) && value >= min && value <= max;
}

function isInsideBoard(row, col) {
  return (
    isIntegerInRange(row, 0, BOARD_SIZE - 1) &&
    isIntegerInRange(col, 0, BOARD_SIZE - 1)
  );
}

function wallKey(row, col) {
  return `${row},${col}`;
}

function assertPlayer(player) {
  if (!isPlayer(player)) {
    throw new RangeError(`Invalid player: ${player}`);
  }
}

export function createInitialState(startingPlayer = 0) {
  assertPlayer(startingPlayer);

  const state = {
    pawns: [
      { row: BOARD_SIZE - 1, col: Math.floor(BOARD_SIZE / 2) },
      { row: 0, col: Math.floor(BOARD_SIZE / 2) },
    ],
    wallsRemaining: [STARTING_WALLS, STARTING_WALLS],
    horizontalWalls: new Set(),
    verticalWalls: new Set(),
    currentPlayer: startingPlayer,
    winner: null,
    moveNumber: 0,
    positionCounts: new Map(),
    terminationReason: null,
  };

  state.positionCounts.set(positionHash(state), 1);
  return state;
}

export function cloneState(state) {
  return {
    pawns: state.pawns.map(({ row, col }) => ({ row, col })),
    wallsRemaining: [...state.wallsRemaining],
    horizontalWalls: new Set(state.horizontalWalls),
    verticalWalls: new Set(state.verticalWalls),
    currentPlayer: state.currentPlayer,
    winner: state.winner,
    moveNumber: state.moveNumber,
    positionCounts: new Map(state.positionCounts ?? []),
    terminationReason: state.terminationReason ?? null,
  };
}

function cloneBoardProbe(state) {
  return {
    pawns: state.pawns,
    horizontalWalls: new Set(state.horizontalWalls),
    verticalWalls: new Set(state.verticalWalls),
  };
}

/**
 * Returns whether a wall blocks movement between two cardinally adjacent cells.
 * Invalid and out-of-board steps are treated as blocked.
 */
export function isPassageBlocked(state, fromRow, fromCol, toRow, toCol) {
  if (
    !isInsideBoard(fromRow, fromCol) ||
    !isInsideBoard(toRow, toCol) ||
    Math.abs(toRow - fromRow) + Math.abs(toCol - fromCol) !== 1
  ) {
    return true;
  }

  if (toRow !== fromRow) {
    const boundaryRow = Math.min(fromRow, toRow);
    return (
      state.horizontalWalls.has(wallKey(boundaryRow, fromCol)) ||
      state.horizontalWalls.has(wallKey(boundaryRow, fromCol - 1))
    );
  }

  const boundaryCol = Math.min(fromCol, toCol);
  return (
    state.verticalWalls.has(wallKey(fromRow, boundaryCol)) ||
    state.verticalWalls.has(wallKey(fromRow - 1, boundaryCol))
  );
}

export function getLegalPawnMoves(state, player = state.currentPlayer) {
  assertPlayer(player);
  if (state.winner !== null) {
    return [];
  }

  const pawn = state.pawns[player];
  const opponent = state.pawns[1 - player];
  const moves = [];

  for (const [rowDelta, colDelta] of DIRECTIONS) {
    const adjacentRow = pawn.row + rowDelta;
    const adjacentCol = pawn.col + colDelta;

    if (
      isPassageBlocked(
        state,
        pawn.row,
        pawn.col,
        adjacentRow,
        adjacentCol,
      )
    ) {
      continue;
    }

    if (adjacentRow !== opponent.row || adjacentCol !== opponent.col) {
      moves.push({ row: adjacentRow, col: adjacentCol });
      continue;
    }

    const jumpRow = opponent.row + rowDelta;
    const jumpCol = opponent.col + colDelta;
    if (
      !isPassageBlocked(
        state,
        opponent.row,
        opponent.col,
        jumpRow,
        jumpCol,
      )
    ) {
      moves.push({ row: jumpRow, col: jumpCol });
      continue;
    }

    // A diagonal is legal only when the direct jump is stopped by a wall or edge.
    const sideDirections =
      rowDelta === 0
        ? [
            [-1, 0],
            [1, 0],
          ]
        : [
            [0, -1],
            [0, 1],
          ];

    for (const [sideRowDelta, sideColDelta] of sideDirections) {
      const diagonalRow = opponent.row + sideRowDelta;
      const diagonalCol = opponent.col + sideColDelta;
      if (
        !isPassageBlocked(
          state,
          opponent.row,
          opponent.col,
          diagonalRow,
          diagonalCol,
        )
      ) {
        moves.push({ row: diagonalRow, col: diagonalCol });
      }
    }
  }

  return moves;
}

export function isWallStructurallyValid(state, orientation, row, col) {
  if (
    (orientation !== "h" && orientation !== "v") ||
    !isIntegerInRange(row, 0, WALL_GRID_SIZE - 1) ||
    !isIntegerInRange(col, 0, WALL_GRID_SIZE - 1)
  ) {
    return false;
  }

  if (orientation === "h") {
    if (
      state.horizontalWalls.has(wallKey(row, col - 1)) ||
      state.horizontalWalls.has(wallKey(row, col)) ||
      state.horizontalWalls.has(wallKey(row, col + 1))
    ) {
      return false;
    }

    return !state.verticalWalls.has(wallKey(row, col));
  }

  if (
    state.verticalWalls.has(wallKey(row - 1, col)) ||
    state.verticalWalls.has(wallKey(row, col)) ||
    state.verticalWalls.has(wallKey(row + 1, col))
  ) {
    return false;
  }

  return !state.horizontalWalls.has(wallKey(row, col));
}

export function hasPathToGoal(state, player) {
  return Number.isFinite(shortestPathDistance(state, player));
}

export function isLegalWall(
  state,
  orientation,
  row,
  col,
  player = state.currentPlayer,
) {
  if (
    !isPlayer(player) ||
    state.winner !== null ||
    state.wallsRemaining[player] <= 0 ||
    !isWallStructurallyValid(state, orientation, row, col)
  ) {
    return false;
  }

  const stateWithWall = cloneBoardProbe(state);
  const walls =
    orientation === "h"
      ? stateWithWall.horizontalWalls
      : stateWithWall.verticalWalls;
  walls.add(wallKey(row, col));

  return hasPathToGoal(stateWithWall, 0) && hasPathToGoal(stateWithWall, 1);
}

export function getLegalWallPlacements(state, player = state.currentPlayer) {
  assertPlayer(player);
  if (state.winner !== null || state.wallsRemaining[player] <= 0) {
    return [];
  }

  const placements = [];
  for (const orientation of ["h", "v"]) {
    for (let row = 0; row < WALL_GRID_SIZE; row += 1) {
      for (let col = 0; col < WALL_GRID_SIZE; col += 1) {
        if (isLegalWall(state, orientation, row, col, player)) {
          placements.push({ type: "wall", orientation, row, col });
        }
      }
    }
  }

  return placements;
}

export function getLegalActions(state, player = state.currentPlayer) {
  assertPlayer(player);
  if (state.winner !== null) {
    return [];
  }

  const pawnMoves = getLegalPawnMoves(state, player).map(({ row, col }) => ({
    type: "move",
    row,
    col,
  }));

  return [...pawnMoves, ...getLegalWallPlacements(state, player)];
}

export function getWinner(state) {
  if (state.winner !== null && state.winner !== undefined) {
    return state.winner;
  }
  if (state.pawns[0].row === 0) {
    return 0;
  }
  if (state.pawns[1].row === BOARD_SIZE - 1) {
    return 1;
  }
  return null;
}

export function applyAction(state, action) {
  if (getWinner(state) !== null) {
    throw new Error("Cannot apply an action after the game has ended");
  }
  if (!action || typeof action !== "object") {
    throw new TypeError("Action must be an object");
  }

  const player = state.currentPlayer;
  const nextState = cloneState(state);

  if (action.type === "move") {
    const legal = getLegalPawnMoves(state, player).some(
      ({ row, col }) => row === action.row && col === action.col,
    );
    if (!legal) {
      throw new Error(`Illegal pawn move to ${action.row},${action.col}`);
    }
    nextState.pawns[player] = { row: action.row, col: action.col };
  } else if (action.type === "wall") {
    if (
      !isLegalWall(
        state,
        action.orientation,
        action.row,
        action.col,
        player,
      )
    ) {
      throw new Error(
        `Illegal wall placement: ${action.orientation} ${action.row},${action.col}`,
      );
    }

    const walls =
      action.orientation === "h"
        ? nextState.horizontalWalls
        : nextState.verticalWalls;
    walls.add(wallKey(action.row, action.col));
    nextState.wallsRemaining[player] -= 1;
  } else {
    throw new Error(`Unknown action type: ${action.type}`);
  }

  nextState.moveNumber += 1;
  nextState.currentPlayer = 1 - player;

  const goalWinner = getWinner(nextState);
  if (goalWinner !== null) {
    nextState.winner = goalWinner;
    nextState.terminationReason = "goal";
    return nextState;
  }

  const hash = positionHash(nextState);
  const positionCount = (nextState.positionCounts.get(hash) ?? 0) + 1;
  nextState.positionCounts.set(hash, positionCount);

  if (positionCount >= REPETITION_LIMIT) {
    nextState.winner = 1 - player;
    nextState.terminationReason = "repetition";
  } else if (nextState.moveNumber >= MAX_HALF_MOVES) {
    nextState.winner = 1 - player;
    nextState.terminationReason = "move-limit";
  }

  return nextState;
}

export function shortestPathDistance(state, player) {
  assertPlayer(player);

  const start = state.pawns[player];
  const goalRow = player === 0 ? 0 : BOARD_SIZE - 1;
  if (start.row === goalRow) {
    return 0;
  }

  const distances = Array.from({ length: BOARD_SIZE }, () =>
    Array(BOARD_SIZE).fill(-1),
  );
  const queue = [{ row: start.row, col: start.col }];
  distances[start.row][start.col] = 0;

  for (let index = 0; index < queue.length; index += 1) {
    const current = queue[index];
    const nextDistance = distances[current.row][current.col] + 1;

    for (const [rowDelta, colDelta] of DIRECTIONS) {
      const row = current.row + rowDelta;
      const col = current.col + colDelta;
      if (
        !isInsideBoard(row, col) ||
        distances[row][col] !== -1 ||
        isPassageBlocked(state, current.row, current.col, row, col)
      ) {
        continue;
      }

      if (row === goalRow) {
        return nextDistance;
      }

      distances[row][col] = nextDistance;
      queue.push({ row, col });
    }
  }

  return Infinity;
}

export function serializeState(state) {
  const sortedPositionCounts = [...(state.positionCounts ?? new Map())].sort(
    ([firstHash], [secondHash]) =>
      String(firstHash).localeCompare(String(secondHash)),
  );

  return JSON.stringify([
    state.currentPlayer,
    state.winner,
    state.moveNumber,
    state.terminationReason ?? null,
    state.pawns[0].row,
    state.pawns[0].col,
    state.pawns[1].row,
    state.pawns[1].col,
    state.wallsRemaining[0],
    state.wallsRemaining[1],
    [...state.horizontalWalls].sort(),
    [...state.verticalWalls].sort(),
    sortedPositionCounts,
  ]);
}
