import test from "node:test";
import assert from "node:assert/strict";

import {
  MAX_HALF_MOVES,
  REPETITION_LIMIT,
  applyAction,
  cloneState,
  createInitialState,
  getLegalActions,
  getWinner,
  serializeState,
} from "../src/game.js";
import { positionHash } from "../src/alphazero/state-codec.js";

test("initial state and clones carry independent end-condition metadata", () => {
  const state = createInitialState(1);
  const initialHash = positionHash(state);
  const clone = cloneState(state);

  assert.equal(REPETITION_LIMIT, 3);
  assert.equal(MAX_HALF_MOVES, 256);
  assert.equal(state.terminationReason, null);
  assert.deepEqual([...state.positionCounts], [[initialHash, 1]]);
  assert.notEqual(clone.positionCounts, state.positionCounts);

  clone.positionCounts.set(initialHash, 2);
  clone.terminationReason = "repetition";
  assert.equal(state.positionCounts.get(initialHash), 1);
  assert.equal(state.terminationReason, null);
});

test("the mover loses when an unchanged position occurs for the third time", () => {
  let state = createInitialState();
  const cycle = [
    { type: "move", row: 8, col: 3 },
    { type: "move", row: 0, col: 3 },
    { type: "move", row: 8, col: 4 },
    { type: "move", row: 0, col: 4 },
  ];

  for (const action of [...cycle, ...cycle]) {
    state = applyAction(state, action);
  }

  assert.equal(state.moveNumber, 8);
  assert.equal(state.positionCounts.get(positionHash(state)), 3);
  assert.equal(state.winner, 0);
  assert.equal(getWinner(state), 0);
  assert.equal(state.terminationReason, "repetition");
});

test("the mover loses on the 256th half-move", () => {
  const state = createInitialState();
  state.moveNumber = MAX_HALF_MOVES - 1;

  const ended = applyAction(state, { type: "move", row: 8, col: 3 });

  assert.equal(ended.moveNumber, MAX_HALF_MOVES);
  assert.equal(ended.winner, 1);
  assert.equal(getWinner(ended), 1);
  assert.equal(ended.terminationReason, "move-limit");
});

test("repetition is the reason when both penalty limits trigger together", () => {
  let state = createInitialState();
  const almostTwoCycles = [
    { type: "move", row: 8, col: 3 },
    { type: "move", row: 0, col: 3 },
    { type: "move", row: 8, col: 4 },
    { type: "move", row: 0, col: 4 },
    { type: "move", row: 8, col: 3 },
    { type: "move", row: 0, col: 3 },
    { type: "move", row: 8, col: 4 },
  ];
  for (const action of almostTwoCycles) {
    state = applyAction(state, action);
  }
  state.moveNumber = MAX_HALF_MOVES - 1;

  const ended = applyAction(state, { type: "move", row: 0, col: 4 });

  assert.equal(ended.winner, 0);
  assert.equal(ended.moveNumber, MAX_HALF_MOVES);
  assert.equal(ended.terminationReason, "repetition");
});

test("reaching the goal takes precedence over repetition and move limits", () => {
  const state = createInitialState();
  state.pawns = [
    { row: 1, col: 4 },
    { row: 0, col: 0 },
  ];
  state.moveNumber = MAX_HALF_MOVES - 1;

  const resultingPosition = cloneState(state);
  resultingPosition.pawns[0] = { row: 0, col: 4 };
  resultingPosition.currentPlayer = 1;
  const resultingHash = positionHash(resultingPosition);
  state.positionCounts.set(resultingHash, REPETITION_LIMIT - 1);

  const ended = applyAction(state, { type: "move", row: 0, col: 4 });

  assert.equal(ended.winner, 0);
  assert.equal(getWinner(ended), 0);
  assert.equal(ended.terminationReason, "goal");
  assert.equal(ended.positionCounts.get(resultingHash), REPETITION_LIMIT - 1);
});

test("penalty termination blocks every later action", () => {
  const state = createInitialState();
  state.moveNumber = MAX_HALF_MOVES - 1;
  const ended = applyAction(state, { type: "move", row: 8, col: 3 });

  assert.deepEqual(getLegalActions(ended), []);
  assert.throws(
    () => applyAction(ended, { type: "move", row: 0, col: 3 }),
    /game has ended/,
  );
});

test("serialization distinguishes repetition history and remains order-stable", () => {
  const first = createInitialState();
  const second = cloneState(first);
  second.positionCounts.set("historical-position", 1);

  assert.notEqual(serializeState(first), serializeState(second));

  const reordered = cloneState(second);
  reordered.positionCounts = new Map(
    [...second.positionCounts].reverse(),
  );
  assert.equal(serializeState(second), serializeState(reordered));

  reordered.terminationReason = "move-limit";
  assert.notEqual(serializeState(second), serializeState(reordered));
});
