import test from "node:test";
import assert from "node:assert/strict";

import {
  BOARD_SIZE,
  STARTING_WALLS,
  applyAction,
  cloneState,
  createInitialState,
  getLegalActions,
  getLegalPawnMoves,
  getLegalWallPlacements,
  getWinner,
  hasPathToGoal,
  isLegalWall,
  isPassageBlocked,
  isWallStructurallyValid,
  serializeState,
  shortestPathDistance,
} from "../src/game.js";

function sortedMoves(moves) {
  return moves
    .map(({ row, col }) => `${row},${col}`)
    .sort();
}

test("creates the standard initial state for either starting player", () => {
  const state = createInitialState(1);

  assert.equal(BOARD_SIZE, 9);
  assert.equal(STARTING_WALLS, 10);
  assert.deepEqual(state.pawns, [
    { row: 8, col: 4 },
    { row: 0, col: 4 },
  ]);
  assert.deepEqual(state.wallsRemaining, [10, 10]);
  assert.equal(state.currentPlayer, 1);
  assert.equal(state.winner, null);
  assert.equal(state.moveNumber, 0);
  assert.equal(state.horizontalWalls.size, 0);
  assert.equal(state.verticalWalls.size, 0);
  assert.throws(() => createInitialState(2), /Invalid player/);
});

test("cloning a state copies all mutable collections and pawn positions", () => {
  const state = createInitialState();
  state.horizontalWalls.add("2,3");
  const cloned = cloneState(state);

  cloned.pawns[0].row = 7;
  cloned.wallsRemaining[0] -= 1;
  cloned.horizontalWalls.add("4,5");

  assert.deepEqual(state.pawns[0], { row: 8, col: 4 });
  assert.equal(state.wallsRemaining[0], 10);
  assert.deepEqual([...state.horizontalWalls], ["2,3"]);
});

test("initial pawn moves and path distances are correct", () => {
  const state = createInitialState();

  assert.deepEqual(sortedMoves(getLegalPawnMoves(state, 0)), [
    "7,4",
    "8,3",
    "8,5",
  ]);
  assert.deepEqual(sortedMoves(getLegalPawnMoves(state, 1)), [
    "0,3",
    "0,5",
    "1,4",
  ]);
  assert.equal(shortestPathDistance(state, 0), 8);
  assert.equal(shortestPathDistance(state, 1), 8);
  assert.equal(getLegalWallPlacements(state).length, 128);
  assert.equal(getLegalActions(state).length, 131);
});

test("a pawn jumps directly over an adjacent opponent when the route is open", () => {
  const state = createInitialState();
  state.pawns = [
    { row: 4, col: 4 },
    { row: 3, col: 4 },
  ];

  assert.deepEqual(sortedMoves(getLegalPawnMoves(state, 0)), [
    "2,4",
    "4,3",
    "4,5",
    "5,4",
  ]);
});

test("a blocked direct jump permits diagonals around the opponent", () => {
  const state = createInitialState();
  state.pawns = [
    { row: 4, col: 4 },
    { row: 3, col: 4 },
  ];
  state.horizontalWalls.add("2,4");

  assert.deepEqual(sortedMoves(getLegalPawnMoves(state, 0)), [
    "3,3",
    "3,5",
    "4,3",
    "4,5",
    "5,4",
  ]);

  state.verticalWalls.add("3,3");
  assert.deepEqual(sortedMoves(getLegalPawnMoves(state, 0)), [
    "3,5",
    "4,5",
    "5,4",
  ]);
});

test("the board edge also turns a direct jump into diagonal moves", () => {
  const state = createInitialState();
  state.pawns = [
    { row: 1, col: 4 },
    { row: 0, col: 4 },
  ];

  assert.deepEqual(sortedMoves(getLegalPawnMoves(state, 0)), [
    "0,3",
    "0,5",
    "1,3",
    "1,5",
    "2,4",
  ]);
});

test("walls block the exact two passages they cover", () => {
  const state = createInitialState();
  state.pawns = [
    { row: 4, col: 4 },
    { row: 0, col: 0 },
  ];
  state.horizontalWalls.add("3,4");
  state.verticalWalls.add("4,4");

  assert.equal(isPassageBlocked(state, 4, 4, 3, 4), true);
  assert.equal(isPassageBlocked(state, 4, 5, 3, 5), true);
  assert.equal(isPassageBlocked(state, 4, 4, 4, 5), true);
  assert.equal(isPassageBlocked(state, 5, 4, 5, 5), true);
  assert.equal(isPassageBlocked(state, 4, 4, 5, 4), false);
  assert.deepEqual(sortedMoves(getLegalPawnMoves(state, 0)), ["4,3", "5,4"]);
});

test("wall structure rejects bounds, overlap, and perpendicular crossing", () => {
  const state = createInitialState();
  state.horizontalWalls.add("3,3");

  assert.equal(isWallStructurallyValid(state, "h", 3, 3), false);
  assert.equal(isWallStructurallyValid(state, "h", 3, 2), false);
  assert.equal(isWallStructurallyValid(state, "h", 3, 4), false);
  assert.equal(isWallStructurallyValid(state, "h", 3, 1), true);
  assert.equal(isWallStructurallyValid(state, "v", 3, 3), false);
  assert.equal(isWallStructurallyValid(state, "v", 3, 4), true);
  assert.equal(isWallStructurallyValid(state, "x", 3, 3), false);
  assert.equal(isWallStructurallyValid(state, "v", -1, 0), false);
  assert.equal(isWallStructurallyValid(state, "v", 7, 8), false);
  assert.equal(isWallStructurallyValid(state, "v", 1.5, 2), false);

  const verticalState = createInitialState();
  verticalState.verticalWalls.add("4,5");
  assert.equal(isWallStructurallyValid(verticalState, "v", 3, 5), false);
  assert.equal(isWallStructurallyValid(verticalState, "v", 5, 5), false);
  assert.equal(isWallStructurallyValid(verticalState, "v", 2, 5), true);
});

test("a structurally valid wall is illegal when it removes a player's last path", () => {
  const state = createInitialState();
  state.pawns = [
    { row: 4, col: 4 },
    { row: 0, col: 0 },
  ];
  state.horizontalWalls = new Set(["3,4", "4,3"]);
  state.verticalWalls = new Set(["3,3"]);

  assert.equal(hasPathToGoal(state, 0), true);
  assert.equal(isWallStructurallyValid(state, "v", 4, 4), true);
  assert.equal(isLegalWall(state, "v", 4, 4), false);

  const sealed = cloneState(state);
  sealed.verticalWalls.add("4,4");
  assert.equal(shortestPathDistance(sealed, 0), Infinity);
  assert.equal(hasPathToGoal(sealed, 0), false);
});

test("a player without wall resources cannot place a wall", () => {
  const state = createInitialState();
  state.wallsRemaining[0] = 0;

  assert.equal(isLegalWall(state, "h", 3, 3), false);
  assert.deepEqual(getLegalWallPlacements(state), []);
  assert.equal(getLegalActions(state).length, 3);
});

test("applying a wall is immutable, consumes a wall, and switches turns", () => {
  const state = createInitialState();
  const next = applyAction(state, {
    type: "wall",
    orientation: "h",
    row: 3,
    col: 3,
  });

  assert.equal(next.horizontalWalls.has("3,3"), true);
  assert.deepEqual(next.wallsRemaining, [9, 10]);
  assert.equal(next.currentPlayer, 1);
  assert.equal(next.moveNumber, 1);
  assert.equal(state.horizontalWalls.size, 0);
  assert.deepEqual(state.wallsRemaining, [10, 10]);
  assert.equal(state.currentPlayer, 0);

  assert.throws(
    () =>
      applyAction(next, {
        type: "wall",
        orientation: "v",
        row: 3,
        col: 3,
      }),
    /Illegal wall placement/,
  );
});

test("applying a winning pawn move records the winner and ends the game", () => {
  const state = createInitialState();
  state.pawns = [
    { row: 1, col: 4 },
    { row: 0, col: 0 },
  ];
  state.moveNumber = 20;

  const won = applyAction(state, { type: "move", row: 0, col: 4 });

  assert.equal(getWinner(won), 0);
  assert.equal(won.winner, 0);
  assert.equal(won.currentPlayer, 1);
  assert.equal(won.moveNumber, 21);
  assert.deepEqual(getLegalActions(won), []);
  assert.throws(
    () => applyAction(won, { type: "move", row: 0, col: 1 }),
    /game has ended/,
  );
});

test("illegal pawn actions are rejected without modifying the input", () => {
  const state = createInitialState();

  assert.throws(
    () => applyAction(state, { type: "move", row: 6, col: 4 }),
    /Illegal pawn move/,
  );
  assert.throws(() => applyAction(state, { type: "dance" }), /Unknown action/);
  assert.deepEqual(state.pawns[0], { row: 8, col: 4 });
  assert.equal(state.moveNumber, 0);
});

test("serialization is stable regardless of Set insertion order", () => {
  const first = createInitialState();
  first.horizontalWalls = new Set(["6,6", "1,2"]);
  first.verticalWalls = new Set(["5,3", "0,1"]);

  const second = createInitialState();
  second.horizontalWalls = new Set(["1,2", "6,6"]);
  second.verticalWalls = new Set(["0,1", "5,3"]);

  assert.equal(serializeState(first), serializeState(second));
});
