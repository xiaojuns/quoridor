import { writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolve } from "node:path";

import {
  applyAction,
  createInitialState,
  getLegalActions,
  getWinner,
} from "../src/game.js";
import {
  ACTION_SPACE_SIZE,
  encodeAction,
  getLegalActionIndices,
} from "../src/alphazero/action-codec.js";
import {
  CODEC_VERSION,
  canonicalizeState,
  mirrorAction,
  mirrorState,
  positionHash,
  stateToSerializable,
  toCanonicalAction,
} from "../src/alphazero/state-codec.js";

export const FIXTURE_SCHEMA_VERSION = 1;
export const FIXTURE_SEED = 0x5eed1234;
export const FIXTURE_POSITION_COUNT = 48;

const DEFAULT_OUTPUT_URL = new URL(
  "../tests/fixtures/alpha-zero-conformance-v1.json",
  import.meta.url,
);

function createRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

function sortedIndices(state) {
  return getLegalActionIndices(state).sort((left, right) => left - right);
}

function buildPositionFixture(state, index, random) {
  const originalPlayer = state.currentPlayer;
  const legalActions = getLegalActions(state);
  const canonical = canonicalizeState(state);
  const mirrored = mirrorState(state);
  const sampleAction = legalActions[Math.floor(random() * legalActions.length)];
  const canonicalAction = toCanonicalAction(sampleAction, originalPlayer);
  const mirroredAction = mirrorAction(sampleAction);

  const next = applyAction(state, sampleAction);
  const canonicalNext = canonicalizeState(next);
  const mirroredNext = mirrorState(next);

  return {
    id: `reachable-${String(index).padStart(3, "0")}`,
    state: stateToSerializable(state),
    positionHash: positionHash(state),
    legalActionIndices: sortedIndices(state),
    canonical: {
      state: stateToSerializable(canonical),
      positionHash: positionHash(canonical),
      legalActionIndices: sortedIndices(canonical),
    },
    mirrored: {
      state: stateToSerializable(mirrored),
      positionHash: positionHash(mirrored),
      legalActionIndices: sortedIndices(mirrored),
    },
    sample: {
      actionIndex: encodeAction(sampleAction),
      canonicalActionIndex: encodeAction(canonicalAction),
      mirroredActionIndex: encodeAction(mirroredAction),
      nextPositionHash: positionHash(next),
      canonicalNextPositionHash: positionHash(canonicalNext),
      mirroredNextPositionHash: positionHash(mirroredNext),
    },
  };
}

export function buildConformanceFixture() {
  const random = createRandom(FIXTURE_SEED);
  const positions = [];
  let gameIndex = 0;
  let state = createInitialState(gameIndex % 2);

  while (positions.length < FIXTURE_POSITION_COUNT) {
    if (getWinner(state) !== null) {
      gameIndex += 1;
      state = createInitialState(gameIndex % 2);
      continue;
    }

    positions.push(buildPositionFixture(state, positions.length, random));
    const legalActions = getLegalActions(state);
    const action = legalActions[Math.floor(random() * legalActions.length)];
    state = applyAction(state, action);
  }

  return {
    schemaVersion: FIXTURE_SCHEMA_VERSION,
    codecVersion: CODEC_VERSION,
    actionSpaceSize: ACTION_SPACE_SIZE,
    seed: FIXTURE_SEED,
    positionCount: FIXTURE_POSITION_COUNT,
    endRules: {
      repetition: "mover-loses-on-third-occurrence",
      maxHalfMoves: 256,
      moveLimit: "mover-loses-on-limit",
      priority: ["goal", "repetition", "move-limit"],
    },
    positions,
  };
}

export function serializeConformanceFixture(fixture = buildConformanceFixture()) {
  return `${JSON.stringify(fixture, null, 2)}\n`;
}

export async function writeConformanceFixture(outputPath) {
  const destination = outputPath
    ? resolve(outputPath)
    : fileURLToPath(DEFAULT_OUTPUT_URL);
  await writeFile(destination, serializeConformanceFixture(), "utf8");
  return destination;
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : null;
if (invokedPath === import.meta.url) {
  const destination = await writeConformanceFixture(process.argv[2]);
  console.log(`Wrote ${FIXTURE_POSITION_COUNT} positions to ${destination}`);
}
