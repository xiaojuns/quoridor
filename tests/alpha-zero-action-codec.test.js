import assert from "node:assert/strict";
import test from "node:test";

import {
  applyAction,
  createInitialState,
  getLegalActions,
} from "../src/game.js";
import {
  ACTION_SPACE_SIZE,
  HORIZONTAL_WALL_ACTION_OFFSET,
  MOVE_ACTION_COUNT,
  VERTICAL_WALL_ACTION_OFFSET,
  WALL_ACTION_COUNT,
  createLegalActionMask,
  decodeAction,
  encodeAction,
  getLegalActionIndices,
} from "../src/alphazero/action-codec.js";

test("the fixed action layout has 209 unique round-trippable actions", () => {
  assert.equal(MOVE_ACTION_COUNT, 81);
  assert.equal(HORIZONTAL_WALL_ACTION_OFFSET, 81);
  assert.equal(WALL_ACTION_COUNT, 64);
  assert.equal(VERTICAL_WALL_ACTION_OFFSET, 145);
  assert.equal(ACTION_SPACE_SIZE, 209);

  const encoded = new Set();
  for (let index = 0; index < ACTION_SPACE_SIZE; index += 1) {
    const action = decodeAction(index);
    assert.equal(encodeAction(action), index);
    encoded.add(encodeAction(action));
  }

  assert.equal(encoded.size, ACTION_SPACE_SIZE);
});

test("action range boundaries map to the documented move and wall coordinates", () => {
  assert.deepEqual(decodeAction(0), { type: "move", row: 0, col: 0 });
  assert.deepEqual(decodeAction(80), { type: "move", row: 8, col: 8 });
  assert.deepEqual(decodeAction(81), {
    type: "wall",
    orientation: "h",
    row: 0,
    col: 0,
  });
  assert.deepEqual(decodeAction(144), {
    type: "wall",
    orientation: "h",
    row: 7,
    col: 7,
  });
  assert.deepEqual(decodeAction(145), {
    type: "wall",
    orientation: "v",
    row: 0,
    col: 0,
  });
  assert.deepEqual(decodeAction(208), {
    type: "wall",
    orientation: "v",
    row: 7,
    col: 7,
  });
});

test("encoding and decoding reject malformed types and out-of-range values", () => {
  for (const action of [null, undefined, "move", [], 12]) {
    assert.throws(() => encodeAction(action), TypeError);
  }

  assert.throws(() => encodeAction({ type: "jump", row: 0, col: 0 }), TypeError);
  assert.throws(() => encodeAction({ type: "move", row: "0", col: 0 }), TypeError);
  assert.throws(() => encodeAction({ type: "move", row: 0.5, col: 0 }), TypeError);
  assert.throws(() => encodeAction({ type: "move", row: -1, col: 0 }), RangeError);
  assert.throws(() => encodeAction({ type: "move", row: 0, col: 9 }), RangeError);
  assert.throws(
    () => encodeAction({ type: "wall", orientation: "x", row: 0, col: 0 }),
    TypeError,
  );
  assert.throws(
    () => encodeAction({ type: "wall", orientation: "h", row: 8, col: 0 }),
    RangeError,
  );
  assert.throws(
    () => encodeAction({ type: "wall", orientation: "v", row: 0, col: -1 }),
    RangeError,
  );

  for (const index of [undefined, "0", NaN, Infinity, 0.5]) {
    assert.throws(() => decodeAction(index), TypeError);
  }
  assert.throws(() => decodeAction(-1), RangeError);
  assert.throws(() => decodeAction(ACTION_SPACE_SIZE), RangeError);
});

test("the initial legal mask exactly matches all engine actions", () => {
  const state = createInitialState();
  const engineIndices = getLegalActions(state).map(encodeAction);
  const legalIndices = getLegalActionIndices(state);
  const mask = createLegalActionMask(state);
  const maskedIndices = [...mask.keys()].filter((index) => mask[index] === 1);

  assert.ok(mask instanceof Uint8Array);
  assert.equal(mask.length, ACTION_SPACE_SIZE);
  assert.equal(mask.reduce((sum, value) => sum + value, 0), 131);
  assert.deepEqual(legalIndices, engineIndices);
  assert.deepEqual(maskedIndices, [...engineIndices].sort((a, b) => a - b));
});

test("terminal positions have no legal action indices or mask entries", () => {
  const state = createInitialState();
  state.pawns = [
    { row: 1, col: 4 },
    { row: 0, col: 0 },
  ];
  const terminal = applyAction(state, { type: "move", row: 0, col: 4 });
  const mask = createLegalActionMask(terminal);

  assert.deepEqual(getLegalActionIndices(terminal), []);
  assert.equal(mask.reduce((sum, value) => sum + value, 0), 0);
  assert.deepEqual([...mask], Array(ACTION_SPACE_SIZE).fill(0));
});
