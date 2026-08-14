import { getLegalActions } from "../game.js";

export const BOARD_SIZE = 9;
export const WALL_GRID_SIZE = BOARD_SIZE - 1;

export const MOVE_ACTION_OFFSET = 0;
export const MOVE_ACTION_COUNT = BOARD_SIZE * BOARD_SIZE;

export const HORIZONTAL_WALL_ACTION_OFFSET =
  MOVE_ACTION_OFFSET + MOVE_ACTION_COUNT;
export const WALL_ACTION_COUNT = WALL_GRID_SIZE * WALL_GRID_SIZE;

export const VERTICAL_WALL_ACTION_OFFSET =
  HORIZONTAL_WALL_ACTION_OFFSET + WALL_ACTION_COUNT;
export const ACTION_SPACE_SIZE =
  VERTICAL_WALL_ACTION_OFFSET + WALL_ACTION_COUNT;

function assertActionObject(action) {
  if (action === null || typeof action !== "object" || Array.isArray(action)) {
    throw new TypeError("Action must be an object");
  }
}

function assertCoordinate(value, name, size) {
  if (!Number.isInteger(value)) {
    throw new TypeError(`${name} must be an integer`);
  }
  if (value < 0 || value >= size) {
    throw new RangeError(`${name} must be between 0 and ${size - 1}`);
  }
}

function assertActionIndex(index) {
  if (!Number.isInteger(index)) {
    throw new TypeError("Action index must be an integer");
  }
  if (index < 0 || index >= ACTION_SPACE_SIZE) {
    throw new RangeError(
      `Action index must be between 0 and ${ACTION_SPACE_SIZE - 1}`,
    );
  }
}

export function encodeAction(action) {
  assertActionObject(action);

  if (action.type === "move") {
    assertCoordinate(action.row, "Move row", BOARD_SIZE);
    assertCoordinate(action.col, "Move col", BOARD_SIZE);
    return MOVE_ACTION_OFFSET + action.row * BOARD_SIZE + action.col;
  }

  if (action.type === "wall") {
    if (action.orientation !== "h" && action.orientation !== "v") {
      throw new TypeError('Wall orientation must be "h" or "v"');
    }
    assertCoordinate(action.row, "Wall row", WALL_GRID_SIZE);
    assertCoordinate(action.col, "Wall col", WALL_GRID_SIZE);

    const offset =
      action.orientation === "h"
        ? HORIZONTAL_WALL_ACTION_OFFSET
        : VERTICAL_WALL_ACTION_OFFSET;
    return offset + action.row * WALL_GRID_SIZE + action.col;
  }

  throw new TypeError('Action type must be "move" or "wall"');
}

export function decodeAction(index) {
  assertActionIndex(index);

  if (index < HORIZONTAL_WALL_ACTION_OFFSET) {
    return {
      type: "move",
      row: Math.floor(index / BOARD_SIZE),
      col: index % BOARD_SIZE,
    };
  }

  const orientation =
    index < VERTICAL_WALL_ACTION_OFFSET ? "h" : "v";
  const offset =
    orientation === "h"
      ? HORIZONTAL_WALL_ACTION_OFFSET
      : VERTICAL_WALL_ACTION_OFFSET;
  const wallIndex = index - offset;

  return {
    type: "wall",
    orientation,
    row: Math.floor(wallIndex / WALL_GRID_SIZE),
    col: wallIndex % WALL_GRID_SIZE,
  };
}

export function getLegalActionIndices(state) {
  if (state === null || typeof state !== "object" || Array.isArray(state)) {
    throw new TypeError("State must be an object");
  }

  return getLegalActions(state).map(encodeAction);
}

export function createLegalActionMask(state) {
  const mask = new Uint8Array(ACTION_SPACE_SIZE);
  for (const index of getLegalActionIndices(state)) {
    mask[index] = 1;
  }
  return mask;
}
