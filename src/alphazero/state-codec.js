export const CODEC_VERSION = 1;

const BOARD_LAST_INDEX = 8;
const WALL_LAST_INDEX = 7;

function assertPlayer(player) {
  if (player !== 0 && player !== 1) {
    throw new RangeError(`Invalid player: ${player}`);
  }
}

function clonePawn(pawn) {
  return { row: pawn.row, col: pawn.col };
}

function sortedWallKeys(walls) {
  return [...(walls ?? [])].map(String).sort();
}

function parseWallKey(key) {
  const [row, col] = String(key).split(",").map(Number);
  if (!Number.isInteger(row) || !Number.isInteger(col)) {
    throw new TypeError(`Invalid wall key: ${key}`);
  }
  return { row, col };
}

function transformWallKeys(walls, transform) {
  const transformed = sortedWallKeys(walls).map((key) => {
    const { row, col } = parseWallKey(key);
    const next = transform(row, col);
    return `${next.row},${next.col}`;
  });
  return new Set(transformed.sort());
}

function normalizeWinner(winner) {
  return winner === 0 || winner === 1 ? winner : winner ?? null;
}

/**
 * Returns a deterministic hash of the playable board position.
 *
 * Turn counters, terminal metadata, repetition metadata, and full history are
 * deliberately excluded so repeated board positions receive the same key.
 */
export function positionHash(state) {
  assertPlayer(state.currentPlayer);
  return JSON.stringify([
    CODEC_VERSION,
    state.currentPlayer,
    state.pawns[0].row,
    state.pawns[0].col,
    state.pawns[1].row,
    state.pawns[1].col,
    state.wallsRemaining[0],
    state.wallsRemaining[1],
    sortedWallKeys(state.horizontalWalls),
    sortedWallKeys(state.verticalWalls),
  ]);
}

function currentRepetitionCount(state) {
  const key = positionHash(state);
  const counts = state.positionCounts;
  let count;

  if (counts instanceof Map) {
    count = counts.get(key);
  } else if (counts && typeof counts === "object") {
    count = counts[key];
  }

  count ??= state.repetitionCount ?? state.currentRepetitionCount;
  return Number.isInteger(count) && count > 0 ? count : 1;
}

function createTransformedState(state, fields) {
  const transformed = {
    pawns: fields.pawns.map(clonePawn),
    wallsRemaining: [...fields.wallsRemaining],
    horizontalWalls: new Set(sortedWallKeys(fields.horizontalWalls)),
    verticalWalls: new Set(sortedWallKeys(fields.verticalWalls)),
    currentPlayer: fields.currentPlayer,
    winner: normalizeWinner(fields.winner),
    moveNumber: Number.isInteger(state.moveNumber) ? state.moveNumber : 0,
    terminationReason: state.terminationReason ?? null,
  };

  const repetitionCount = currentRepetitionCount(state);
  transformed.repetitionCount = repetitionCount;
  transformed.positionCounts = new Map([
    [positionHash(transformed), repetitionCount],
  ]);
  return transformed;
}

function rotatePawn(pawn) {
  return {
    row: BOARD_LAST_INDEX - pawn.row,
    col: BOARD_LAST_INDEX - pawn.col,
  };
}

function rotateWalls(walls) {
  return transformWallKeys(walls, (row, col) => ({
    row: WALL_LAST_INDEX - row,
    col: WALL_LAST_INDEX - col,
  }));
}

/**
 * Returns a fresh rule-engine-compatible state whose side to move is player 0,
 * positioned at the bottom and moving upward. Wall collections are fresh Sets.
 *
 * Historical repetition counts cannot be transformed faithfully without the
 * historical states. The returned Map therefore contains only the transformed
 * current position and its source count; the same count is also exposed as the
 * scalar `repetitionCount` for model input.
 */
export function canonicalizeState(state) {
  assertPlayer(state.currentPlayer);
  if (state.currentPlayer === 0) {
    return createTransformedState(state, {
      pawns: state.pawns,
      wallsRemaining: state.wallsRemaining,
      horizontalWalls: state.horizontalWalls,
      verticalWalls: state.verticalWalls,
      currentPlayer: 0,
      winner: state.winner,
    });
  }

  const winner = normalizeWinner(state.winner);
  return createTransformedState(state, {
    pawns: [rotatePawn(state.pawns[1]), rotatePawn(state.pawns[0])],
    wallsRemaining: [state.wallsRemaining[1], state.wallsRemaining[0]],
    horizontalWalls: rotateWalls(state.horizontalWalls),
    verticalWalls: rotateWalls(state.verticalWalls),
    currentPlayer: 0,
    winner: winner === 0 || winner === 1 ? 1 - winner : winner,
  });
}

function transformAction(action, transformPawn, transformWall) {
  if (!action || typeof action !== "object") {
    throw new TypeError("Action must be an object");
  }

  if (action.type === "move") {
    const position = transformPawn(action.row, action.col);
    return { ...action, row: position.row, col: position.col };
  }
  if (action.type === "wall") {
    const position = transformWall(action.row, action.col);
    return { ...action, row: position.row, col: position.col };
  }
  throw new TypeError(`Unknown action type: ${action.type}`);
}

function rotateAction(action) {
  return transformAction(
    action,
    (row, col) => ({
      row: BOARD_LAST_INDEX - row,
      col: BOARD_LAST_INDEX - col,
    }),
    (row, col) => ({
      row: WALL_LAST_INDEX - row,
      col: WALL_LAST_INDEX - col,
    }),
  );
}

/** Maps an action from the original board into its side-to-move frame. */
export function toCanonicalAction(action, originalCurrentPlayer) {
  assertPlayer(originalCurrentPlayer);
  return originalCurrentPlayer === 0 ? { ...action } : rotateAction(action);
}

/** Maps an action from the side-to-move frame back to the original board. */
export function fromCanonicalAction(action, originalCurrentPlayer) {
  assertPlayer(originalCurrentPlayer);
  return originalCurrentPlayer === 0 ? { ...action } : rotateAction(action);
}

/**
 * Reflects a state left-to-right and returns fresh Sets and a current-only
 * repetition Map, using the same metadata policy as `canonicalizeState`.
 */
export function mirrorState(state) {
  assertPlayer(state.currentPlayer);
  return createTransformedState(state, {
    pawns: state.pawns.map((pawn) => ({
      row: pawn.row,
      col: BOARD_LAST_INDEX - pawn.col,
    })),
    wallsRemaining: state.wallsRemaining,
    horizontalWalls: transformWallKeys(state.horizontalWalls, (row, col) => ({
      row,
      col: WALL_LAST_INDEX - col,
    })),
    verticalWalls: transformWallKeys(state.verticalWalls, (row, col) => ({
      row,
      col: WALL_LAST_INDEX - col,
    })),
    currentPlayer: state.currentPlayer,
    winner: state.winner,
  });
}

/** Reflects a move destination or wall anchor left-to-right. */
export function mirrorAction(action) {
  return transformAction(
    action,
    (row, col) => ({ row, col: BOARD_LAST_INDEX - col }),
    (row, col) => ({ row, col: WALL_LAST_INDEX - col }),
  );
}

/**
 * Creates a deterministic JSON-safe DTO. Sets become sorted arrays and full
 * repetition history is reduced to the current position's scalar count.
 */
export function stateToSerializable(state) {
  assertPlayer(state.currentPlayer);
  return {
    codecVersion: CODEC_VERSION,
    pawns: state.pawns.map(clonePawn),
    wallsRemaining: [...state.wallsRemaining],
    horizontalWalls: sortedWallKeys(state.horizontalWalls),
    verticalWalls: sortedWallKeys(state.verticalWalls),
    currentPlayer: state.currentPlayer,
    winner: normalizeWinner(state.winner),
    moveNumber: Number.isInteger(state.moveNumber) ? state.moveNumber : 0,
    terminationReason: state.terminationReason ?? null,
    repetitionCount: currentRepetitionCount(state),
  };
}
