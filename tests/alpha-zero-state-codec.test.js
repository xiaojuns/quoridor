import assert from "node:assert/strict";
import test from "node:test";

import {
  applyAction,
  createInitialState,
  getLegalActions,
} from "../src/game.js";
import {
  CODEC_VERSION,
  canonicalizeState,
  fromCanonicalAction,
  mirrorAction,
  mirrorState,
  positionHash,
  stateToSerializable,
  toCanonicalAction,
} from "../src/alphazero/state-codec.js";

function comparableState(state) {
  const repetitionCount =
    state.positionCounts?.get(positionHash(state)) ?? state.repetitionCount;
  return {
    pawns: state.pawns,
    wallsRemaining: state.wallsRemaining,
    horizontalWalls: [...state.horizontalWalls].sort(),
    verticalWalls: [...state.verticalWalls].sort(),
    currentPlayer: state.currentPlayer,
    winner: state.winner,
    moveNumber: state.moveNumber,
    terminationReason: state.terminationReason ?? null,
    repetitionCount,
  };
}

function makeAsymmetricState(currentPlayer = 0) {
  const state = createInitialState(currentPlayer);
  state.pawns = [
    { row: 6, col: 2 },
    { row: 2, col: 5 },
  ];
  state.wallsRemaining = [8, 7];
  state.horizontalWalls = new Set(["1,2", "5,4"]);
  state.verticalWalls = new Set(["0,6", "4,1"]);
  state.moveNumber = 9;
  state.terminationReason = null;
  state.positionCounts = new Map([[positionHash(state), 2], ["old", 3]]);
  return state;
}

test("positionHash is stable and contains only the playable position", () => {
  const first = makeAsymmetricState(1);
  const second = makeAsymmetricState(1);
  second.horizontalWalls = new Set(["5,4", "1,2"]);
  second.verticalWalls = new Set(["4,1", "0,6"]);
  second.moveNumber = 999;
  second.winner = 0;
  second.terminationReason = "threefold-repetition";
  second.history = [{ ignored: true }];
  second.positionCounts = new Map([["different-history", 20]]);

  assert.equal(positionHash(first), positionHash(second));

  second.wallsRemaining[1] -= 1;
  assert.notEqual(positionHash(first), positionHash(second));
});

test("player 0 canonicalization is an independent identity transform", () => {
  const state = makeAsymmetricState(0);
  const canonical = canonicalizeState(state);

  assert.deepEqual(comparableState(canonical), comparableState({
    ...state,
    repetitionCount: 2,
  }));
  assert.notEqual(canonical, state);
  assert.notEqual(canonical.pawns, state.pawns);
  assert.notEqual(canonical.horizontalWalls, state.horizontalWalls);
  assert.notEqual(canonical.verticalWalls, state.verticalWalls);
  assert.notEqual(canonical.positionCounts, state.positionCounts);
  assert.deepEqual([...canonical.positionCounts], [[positionHash(canonical), 2]]);
});

test("player 1 canonicalization swaps players and rotates coordinates", () => {
  const state = makeAsymmetricState(1);
  state.winner = 1;
  const canonical = canonicalizeState(state);

  assert.equal(canonical.currentPlayer, 0);
  assert.deepEqual(canonical.pawns, [
    { row: 6, col: 3 },
    { row: 2, col: 6 },
  ]);
  assert.deepEqual(canonical.wallsRemaining, [7, 8]);
  assert.deepEqual([...canonical.horizontalWalls].sort(), ["2,3", "6,5"]);
  assert.deepEqual([...canonical.verticalWalls].sort(), ["3,6", "7,1"]);
  assert.equal(canonical.winner, 0);
  assert.equal(canonical.repetitionCount, 2);

  const move = { type: "move", row: 3, col: 5 };
  const wall = { type: "wall", orientation: "h", row: 2, col: 6 };
  assert.deepEqual(toCanonicalAction(move, 1), {
    type: "move",
    row: 5,
    col: 3,
  });
  assert.deepEqual(toCanonicalAction(wall, 1), {
    type: "wall",
    orientation: "h",
    row: 5,
    col: 1,
  });
  assert.deepEqual(fromCanonicalAction(toCanonicalAction(move, 1), 1), move);
  assert.deepEqual(fromCanonicalAction(toCanonicalAction(wall, 1), 1), wall);
});

test("mirror transforms are involutions and do not mutate their inputs", () => {
  const state = makeAsymmetricState(1);
  const before = stateToSerializable(state);
  const mirrored = mirrorState(state);
  const restored = mirrorState(mirrored);

  assert.deepEqual(comparableState(restored), {
    ...comparableState(state),
    repetitionCount: 2,
  });
  assert.deepEqual(stateToSerializable(state), before);
  assert.notEqual(mirrored.horizontalWalls, state.horizontalWalls);
  assert.deepEqual(mirrorAction(mirrorAction({ type: "move", row: 5, col: 2 })), {
    type: "move",
    row: 5,
    col: 2,
  });
  assert.deepEqual(
    mirrorAction(mirrorAction({ type: "wall", orientation: "v", row: 4, col: 1 })),
    { type: "wall", orientation: "v", row: 4, col: 1 },
  );
});

test("canonical and mirror transforms are equivariant with legal transitions", () => {
  const original = createInitialState(1);
  const originalAction = { type: "move", row: 1, col: 4 };
  const canonical = canonicalizeState(original);
  const canonicalAction = toCanonicalAction(originalAction, original.currentPlayer);

  assert.ok(
    getLegalActions(canonical).some(
      (action) => JSON.stringify(action) === JSON.stringify(canonicalAction),
    ),
  );

  const originalNext = applyAction(original, originalAction);
  const canonicalNext = applyAction(canonical, canonicalAction);
  assert.deepEqual(
    comparableState(canonicalizeState(originalNext)),
    comparableState(canonicalizeState(canonicalNext)),
  );

  const wallState = createInitialState(0);
  const wallAction = { type: "wall", orientation: "v", row: 3, col: 1 };
  const mirroredState = mirrorState(wallState);
  const mirroredAction = mirrorAction(wallAction);
  assert.deepEqual(
    comparableState(mirrorState(applyAction(wallState, wallAction))),
    comparableState(applyAction(mirroredState, mirroredAction)),
  );
});

test("stateToSerializable is deterministic, JSON-safe, and omits history maps", () => {
  const state = makeAsymmetricState(0);
  const dto = stateToSerializable(state);

  assert.equal(dto.codecVersion, CODEC_VERSION);
  assert.deepEqual(dto.horizontalWalls, ["1,2", "5,4"]);
  assert.deepEqual(dto.verticalWalls, ["0,6", "4,1"]);
  assert.equal(dto.repetitionCount, 2);
  assert.equal("positionCounts" in dto, false);
  assert.equal("history" in dto, false);
  assert.doesNotThrow(() => JSON.stringify(dto));
  assert.deepEqual(stateToSerializable(state), dto);
});
