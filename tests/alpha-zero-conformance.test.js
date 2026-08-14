import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  applyAction,
  getLegalActions,
} from "../src/game.js";
import {
  ACTION_SPACE_SIZE,
  createLegalActionMask,
  decodeAction,
  encodeAction,
  getLegalActionIndices,
} from "../src/alphazero/action-codec.js";
import {
  canonicalizeState,
  mirrorAction,
  mirrorState,
  positionHash,
  toCanonicalAction,
} from "../src/alphazero/state-codec.js";
import {
  FIXTURE_POSITION_COUNT,
  FIXTURE_SCHEMA_VERSION,
  buildConformanceFixture,
  serializeConformanceFixture,
} from "../scripts/generate-alpha-zero-fixtures.js";

const FIXTURE_URL = new URL(
  "./fixtures/alpha-zero-conformance-v1.json",
  import.meta.url,
);

function stateFromDto(dto) {
  const state = {
    pawns: dto.pawns.map(({ row, col }) => ({ row, col })),
    wallsRemaining: [...dto.wallsRemaining],
    horizontalWalls: new Set(dto.horizontalWalls),
    verticalWalls: new Set(dto.verticalWalls),
    currentPlayer: dto.currentPlayer,
    winner: dto.winner,
    moveNumber: dto.moveNumber,
    terminationReason: dto.terminationReason,
    repetitionCount: dto.repetitionCount,
    positionCounts: new Map(),
  };
  state.positionCounts.set(positionHash(state), dto.repetitionCount);
  return state;
}

function sorted(values) {
  return [...values].sort((left, right) => left - right);
}

test("the committed conformance fixture is byte-for-byte reproducible", async () => {
  const committed = await readFile(FIXTURE_URL, "utf8");
  assert.equal(committed, serializeConformanceFixture(buildConformanceFixture()));
});

test("all fixture positions lock masks, transforms, and sampled transitions", async () => {
  const fixture = JSON.parse(await readFile(FIXTURE_URL, "utf8"));

  assert.equal(fixture.schemaVersion, FIXTURE_SCHEMA_VERSION);
  assert.equal(fixture.actionSpaceSize, ACTION_SPACE_SIZE);
  assert.equal(fixture.positionCount, FIXTURE_POSITION_COUNT);
  assert.equal(fixture.positions.length, FIXTURE_POSITION_COUNT);
  assert.deepEqual(fixture.endRules.priority, ["goal", "repetition", "move-limit"]);

  for (const entry of fixture.positions) {
    const state = stateFromDto(entry.state);
    const canonical = canonicalizeState(state);
    const mirrored = mirrorState(state);
    const legalActions = getLegalActions(state);
    const mask = createLegalActionMask(state);

    assert.equal(positionHash(state), entry.positionHash, entry.id);
    assert.deepEqual(sorted(getLegalActionIndices(state)), entry.legalActionIndices, entry.id);
    assert.deepEqual(
      [...mask.keys()].filter((index) => mask[index] === 1),
      entry.legalActionIndices,
      entry.id,
    );

    assert.equal(positionHash(canonical), entry.canonical.positionHash, entry.id);
    assert.deepEqual(
      sorted(getLegalActionIndices(canonical)),
      entry.canonical.legalActionIndices,
      entry.id,
    );
    assert.deepEqual(
      sorted(legalActions.map((action) => encodeAction(
        toCanonicalAction(action, state.currentPlayer),
      ))),
      entry.canonical.legalActionIndices,
      entry.id,
    );

    assert.equal(positionHash(mirrored), entry.mirrored.positionHash, entry.id);
    assert.deepEqual(
      sorted(getLegalActionIndices(mirrored)),
      entry.mirrored.legalActionIndices,
      entry.id,
    );
    assert.deepEqual(
      sorted(legalActions.map((action) => encodeAction(mirrorAction(action)))),
      entry.mirrored.legalActionIndices,
      entry.id,
    );

    const action = decodeAction(entry.sample.actionIndex);
    assert.ok(entry.legalActionIndices.includes(entry.sample.actionIndex), entry.id);
    assert.equal(
      encodeAction(toCanonicalAction(action, state.currentPlayer)),
      entry.sample.canonicalActionIndex,
      entry.id,
    );
    assert.equal(
      encodeAction(mirrorAction(action)),
      entry.sample.mirroredActionIndex,
      entry.id,
    );

    const next = applyAction(state, action);
    assert.equal(positionHash(next), entry.sample.nextPositionHash, entry.id);
    assert.equal(
      positionHash(canonicalizeState(next)),
      entry.sample.canonicalNextPositionHash,
      entry.id,
    );
    assert.equal(
      positionHash(mirrorState(next)),
      entry.sample.mirroredNextPositionHash,
      entry.id,
    );

    const canonicalNext = applyAction(
      canonical,
      decodeAction(entry.sample.canonicalActionIndex),
    );
    assert.equal(
      positionHash(canonicalizeState(canonicalNext)),
      entry.sample.canonicalNextPositionHash,
      entry.id,
    );

    const mirroredNext = applyAction(
      mirrored,
      decodeAction(entry.sample.mirroredActionIndex),
    );
    assert.equal(
      positionHash(mirroredNext),
      entry.sample.mirroredNextPositionHash,
      entry.id,
    );
  }
});
