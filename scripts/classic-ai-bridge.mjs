#!/usr/bin/env node

import readline from 'node:readline';

import { analyzeAIMoves, chooseAIMove } from '../src/ai.js';
import { getLegalActions, getWinner } from '../src/game.js';
import { CODEC_VERSION, positionHash } from '../src/alphazero/state-codec.js';

const PROTOCOL_VERSION = 1;
const ENGINE = 'classic-negamax';
const HARD_TIME_LIMIT_MS = 850;
const HARD_MAX_DEPTH = 5;
const NODE_BUDGET_TIME_LIMIT_MS = 2_147_483_647;

class RequestError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'RequestError';
    this.code = code;
  }
}

function writeMessage(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function requestError(condition, code, message) {
  if (!condition) throw new RequestError(code, message);
}

function integer(value, label, minimum, maximum = Number.MAX_SAFE_INTEGER) {
  requestError(
    Number.isSafeInteger(value) && value >= minimum && value <= maximum,
    'invalid-request',
    `${label} must be an integer in [${minimum}, ${maximum}]`,
  );
  return value;
}

function player(value, label) {
  return integer(value, label, 0, 1);
}

function position(value, label) {
  requestError(
    value !== null && typeof value === 'object' && !Array.isArray(value),
    'invalid-state',
    `${label} must be an object`,
  );
  return {
    row: integer(value.row, `${label}.row`, 0, 8),
    col: integer(value.col, `${label}.col`, 0, 8),
  };
}

function wallKey(value, label) {
  requestError(typeof value === 'string', 'invalid-state', `${label} must be a string`);
  const match = /^(0|[1-7]),(0|[1-7])$/.exec(value);
  requestError(Boolean(match), 'invalid-state', `${label} is not a canonical wall key`);
  return value;
}

function wallKeys(value, label) {
  requestError(Array.isArray(value), 'invalid-state', `${label} must be an array`);
  const walls = value.map((wall, index) => wallKey(wall, `${label}[${index}]`));
  const sorted = [...walls].sort();
  requestError(
    walls.every((wall, index) => wall === sorted[index]) && new Set(walls).size === walls.length,
    'invalid-state',
    `${label} must be sorted and unique`,
  );
  return new Set(walls);
}

function positionCounts(value) {
  requestError(
    Array.isArray(value) && value.length > 0,
    'invalid-state',
    'state.positionCounts must be a non-empty array',
  );
  const counts = new Map();
  let previousHash = null;
  let occurrences = 0;
  for (let index = 0; index < value.length; index += 1) {
    const entry = value[index];
    requestError(
      entry !== null && typeof entry === 'object' && !Array.isArray(entry),
      'invalid-state',
      `state.positionCounts[${index}] must be an object`,
    );
    requestError(
      typeof entry.hash === 'string' && entry.hash.length > 0,
      'invalid-state',
      `state.positionCounts[${index}].hash must be non-empty`,
    );
    const count = integer(entry.count, `state.positionCounts[${index}].count`, 1, 2);
    requestError(
      previousHash === null || previousHash.localeCompare(entry.hash) < 0,
      'invalid-state',
      'state.positionCounts must be sorted and unique',
    );
    counts.set(entry.hash, count);
    previousHash = entry.hash;
    occurrences += count;
  }
  return { counts, occurrences };
}

function restoreState(payload, expectedHash) {
  requestError(
    payload !== null && typeof payload === 'object' && !Array.isArray(payload),
    'invalid-state',
    'state must be an object',
  );
  requestError(
    payload.codecVersion === CODEC_VERSION,
    'invalid-state',
    `state.codecVersion must be ${CODEC_VERSION}`,
  );
  requestError(
    Array.isArray(payload.pawns) && payload.pawns.length === 2,
    'invalid-state',
    'state.pawns must contain two positions',
  );
  requestError(
    Array.isArray(payload.wallsRemaining) && payload.wallsRemaining.length === 2,
    'invalid-state',
    'state.wallsRemaining must contain two counts',
  );
  const pawns = [
    position(payload.pawns[0], 'state.pawns[0]'),
    position(payload.pawns[1], 'state.pawns[1]'),
  ];
  requestError(
    pawns[0].row !== pawns[1].row || pawns[0].col !== pawns[1].col,
    'invalid-state',
    'state pawns must occupy different squares',
  );
  const moveNumber = integer(payload.moveNumber, 'state.moveNumber', 0, 255);
  const { counts, occurrences } = positionCounts(payload.positionCounts);
  requestError(
    occurrences === moveNumber + 1,
    'invalid-state',
    'complete positionCounts occurrences must equal moveNumber + 1',
  );
  requestError(payload.winner === null, 'invalid-state', 'bridge states must be non-terminal');
  requestError(
    payload.terminationReason === null,
    'invalid-state',
    'bridge states must not have a termination reason',
  );

  const state = {
    pawns,
    wallsRemaining: [
      integer(payload.wallsRemaining[0], 'state.wallsRemaining[0]', 0, 10),
      integer(payload.wallsRemaining[1], 'state.wallsRemaining[1]', 0, 10),
    ],
    horizontalWalls: wallKeys(payload.horizontalWalls, 'state.horizontalWalls'),
    verticalWalls: wallKeys(payload.verticalWalls, 'state.verticalWalls'),
    currentPlayer: player(payload.currentPlayer, 'state.currentPlayer'),
    winner: null,
    moveNumber,
    positionCounts: counts,
    terminationReason: null,
  };
  const actualHash = positionHash(state);
  requestError(
    typeof expectedHash === 'string' && expectedHash === actualHash,
    'position-hash-mismatch',
    `request position hash does not match JavaScript state: ${actualHash}`,
  );
  requestError(
    counts.has(actualHash),
    'invalid-state',
    'complete positionCounts omits the current position',
  );
  requestError(
    payload.repetitionCount === counts.get(actualHash),
    'invalid-state',
    'state.repetitionCount does not match complete positionCounts',
  );
  requestError(getWinner(state) === null, 'invalid-state', 'bridge states must be non-terminal');
  requestError(getLegalActions(state).length > 0, 'invalid-state', 'state has no legal action');
  return { state, actualHash, historyEntries: counts.size, historyOccurrences: occurrences };
}

function parseBudget(value) {
  if (value === undefined) {
    return { kind: 'time', limit: HARD_TIME_LIMIT_MS };
  }
  requestError(
    value !== null && typeof value === 'object' && !Array.isArray(value),
    'invalid-request',
    'budget must be an object',
  );
  if (value.kind === 'time') {
    return {
      kind: 'time',
      limit: integer(value.timeLimitMs, 'budget.timeLimitMs', 0),
    };
  }
  if (value.kind === 'nodes') {
    return {
      kind: 'nodes',
      limit: integer(value.nodeLimit, 'budget.nodeLimit', 0),
    };
  }
  throw new RequestError('invalid-request', 'budget.kind must be "time" or "nodes"');
}

function handleChoose(request) {
  const budget = parseBudget(request.budget);
  const maxDepth =
    request.maxDepth === undefined
      ? HARD_MAX_DEPTH
      : integer(request.maxDepth, 'maxDepth', 1, 12);
  const restored = restoreState(request.state, request.positionHash);
  const options = {
    difficulty: 'hard',
    maxDepth,
  };
  if (budget.kind === 'nodes') {
    options.nodeLimit = budget.limit;
    options.timeLimitMs = NODE_BUDGET_TIME_LIMIT_MS;
  } else {
    options.timeLimitMs = budget.limit;
  }

  const result = chooseAIMove(restored.state, options);
  requestError(
    result.action !== null && typeof result.action === 'object',
    'engine-error',
    'classic AI returned no action for a non-terminal state',
  );
  const afterHash = positionHash(restored.state);
  requestError(
    afterHash === restored.actualHash,
    'engine-error',
    'classic AI mutated the request state',
  );
  return {
    version: PROTOCOL_VERSION,
    type: 'result',
    id: request.id,
    engine: ENGINE,
    positionHash: afterHash,
    historyEntries: restored.historyEntries,
    historyOccurrences: restored.historyOccurrences,
    budget,
    action: result.action,
    score: result.score,
    depth: result.depth,
    nodes: result.nodes,
    elapsedMs: result.elapsedMs,
  };
}

function handleAnalyze(request) {
  const budget = parseBudget(request.budget);
  const maxDepth =
    request.maxDepth === undefined
      ? HARD_MAX_DEPTH
      : integer(request.maxDepth, 'maxDepth', 1, 12);
  const restored = restoreState(request.state, request.positionHash);
  const options = { difficulty: 'hard', maxDepth };
  if (budget.kind === 'nodes') {
    options.nodeLimit = budget.limit;
    options.timeLimitMs = NODE_BUDGET_TIME_LIMIT_MS;
  } else {
    options.timeLimitMs = budget.limit;
  }
  const result = analyzeAIMoves(restored.state, options);
  requestError(
    result.action !== null && typeof result.action === 'object',
    'engine-error',
    'classic AI returned no action for a non-terminal state',
  );
  requestError(
    result.rootScoreKind === 'alpha-beta-return-v1' && Array.isArray(result.rootScores),
    'engine-error',
    'classic AI returned malformed root analysis',
  );
  const afterHash = positionHash(restored.state);
  requestError(afterHash === restored.actualHash, 'engine-error', 'classic AI mutated the request state');
  return {
    version: PROTOCOL_VERSION,
    type: 'analysis',
    id: request.id,
    engine: ENGINE,
    positionHash: afterHash,
    historyEntries: restored.historyEntries,
    historyOccurrences: restored.historyOccurrences,
    budget,
    action: result.action,
    score: result.score,
    depth: result.depth,
    nodes: result.nodes,
    elapsedMs: result.elapsedMs,
    rootScoreKind: result.rootScoreKind,
    rootScores: result.rootScores,
  };
}

function errorResponse(id, error) {
  const source = error instanceof Error ? error : new Error(String(error));
  return {
    version: PROTOCOL_VERSION,
    type: 'error',
    id: Number.isSafeInteger(id) ? id : null,
    error: {
      code: source instanceof RequestError ? source.code : 'internal-error',
      message: source.message,
    },
  };
}

function handleLine(line) {
  let request;
  try {
    request = JSON.parse(line);
  } catch (error) {
    writeMessage(errorResponse(null, new RequestError('invalid-json', error.message)));
    return false;
  }

  try {
    requestError(
      request !== null && typeof request === 'object' && !Array.isArray(request),
      'invalid-request',
      'request must be an object',
    );
    requestError(
      request.version === PROTOCOL_VERSION,
      'invalid-request',
      `request.version must be ${PROTOCOL_VERSION}`,
    );
    integer(request.id, 'request.id', 1);
    if (request.type === 'close') {
      writeMessage({
        version: PROTOCOL_VERSION,
        type: 'closed',
        id: request.id,
      });
      return true;
    }
    requestError(
      request.type === 'choose' || request.type === 'analyze',
      'invalid-request',
      'unsupported request type',
    );
    writeMessage(request.type === 'analyze' ? handleAnalyze(request) : handleChoose(request));
  } catch (error) {
    writeMessage(errorResponse(request?.id, error));
  }
  return false;
}

async function main() {
  const input = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
    terminal: false,
  });
  writeMessage({
    version: PROTOCOL_VERSION,
    type: 'ready',
    engine: ENGINE,
    hardDefaults: {
      timeLimitMs: HARD_TIME_LIMIT_MS,
      maxDepth: HARD_MAX_DEPTH,
    },
  });
  for await (const line of input) {
    if (handleLine(line)) {
      input.close();
      break;
    }
  }
}

main().catch((error) => {
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exitCode = 1;
});
