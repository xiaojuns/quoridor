import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyAction,
  createInitialState,
  getLegalActions,
  getWinner,
  shortestPathDistance,
} from '../src/game.js';
import {
  WIN_SCORE,
  analyzeAIMoves,
  chooseAIMove,
  getCandidateActions,
  rankWallCandidates,
} from '../src/ai.js';

function containsAction(actions, expected) {
  return actions.some((action) => JSON.stringify(action) === JSON.stringify(expected));
}

test('AI returns legal actions and useful search statistics', () => {
  const state = createInitialState();
  const legalActions = getLegalActions(state);
  const result = chooseAIMove(state, {
    difficulty: 'medium',
    maxDepth: 2,
    timeLimitMs: 2_000,
  });

  assert.ok(containsAction(legalActions, result.action));
  assert.equal(typeof result.score, 'number');
  assert.equal(result.depth, 2);
  assert.ok(Number.isInteger(result.nodes) && result.nodes > 0);
  assert.ok(Number.isFinite(result.elapsedMs) && result.elapsedMs >= 0);
});

test('AI always takes an immediate win, even with a zero search budget', () => {
  const state = createInitialState();
  state.pawns[0] = { row: 1, col: 4 };
  state.pawns[1] = { row: 0, col: 0 };
  state.currentPlayer = 0;

  const result = chooseAIMove(state, {
    difficulty: 'hard',
    timeLimitMs: 0,
  });
  const next = applyAction(state, result.action);

  assert.deepEqual(result.action, { type: 'move', row: 0, col: 4 });
  assert.equal(getWinner(next), 0);
  assert.equal(result.score, WIN_SCORE - 1);
  assert.equal(result.depth, 1);
});

test('two-ply search blocks an opponent who would win next turn', () => {
  const state = createInitialState();
  state.pawns = [
    { row: 6, col: 4 },
    { row: 7, col: 4 },
  ];
  state.currentPlayer = 0;

  assert.equal(shortestPathDistance(state, 1), 1);
  const result = chooseAIMove(state, {
    difficulty: 'medium',
    maxDepth: 2,
    timeLimitMs: 2_000,
  });
  const next = applyAction(state, result.action);

  assert.equal(result.action.type, 'wall');
  assert.ok(shortestPathDistance(next, 1) > 1);
  assert.ok(containsAction(getLegalActions(state), result.action));
});

test('wall ranking is tactical and fixed-depth search is deterministic', () => {
  const state = createInitialState();
  state.pawns = [
    { row: 6, col: 4 },
    { row: 7, col: 4 },
  ];
  state.currentPlayer = 0;

  const legalWalls = getLegalActions(state).filter((action) => action.type === 'wall');
  const ranked = rankWallCandidates(state, legalWalls, 0, { prefilterLimit: 128 });
  assert.equal(ranked[0].action.type, 'wall');
  assert.ok(ranked[0].opponentDelta > 0);
  assert.equal(ranked[0].ownDelta, 0);

  const candidates = getCandidateActions(state, {
    difficulty: 'medium',
    legalActions: getLegalActions(state),
  });
  assert.ok(candidates.some((action) => action.type === 'move'));
  assert.ok(candidates.some((action) => action.type === 'wall'));

  const options = { difficulty: 'medium', maxDepth: 2, timeLimitMs: 2_000 };
  const first = chooseAIMove(state, options);
  const second = chooseAIMove(state, options);
  assert.deepEqual(first.action, second.action);
  assert.equal(first.score, second.score);
  assert.equal(first.depth, second.depth);
});

test('node budget is deterministic and does not exceed its explicit limit', () => {
  const state = createInitialState();
  const options = {
    difficulty: 'hard',
    maxDepth: 5,
    timeLimitMs: 60_000,
    nodeLimit: 64,
  };

  const first = chooseAIMove(state, options);
  const second = chooseAIMove(state, options);

  assert.equal(first.nodes, 64);
  assert.equal(second.nodes, 64);
  assert.deepEqual(first.action, second.action);
  assert.equal(first.score, second.score);
  assert.equal(first.depth, second.depth);
  assert.ok(containsAction(getLegalActions(state), first.action));

  const zero = chooseAIMove(state, { ...options, nodeLimit: 0 });
  assert.equal(zero.nodes, 0);
  assert.equal(zero.depth, 0);
  assert.ok(containsAction(getLegalActions(state), zero.action));
});

test('root analysis exposes actual searched scores and preserves classic choice', () => {
  const state = createInitialState();
  const options = {
    difficulty: 'hard',
    maxDepth: 5,
    timeLimitMs: 60_000,
    nodeLimit: 587,
  };
  const choice = chooseAIMove(state, options);
  const analysis = analyzeAIMoves(state, options);

  assert.deepEqual(analysis.action, choice.action);
  assert.equal(analysis.score, choice.score);
  assert.equal(analysis.depth, choice.depth);
  assert.equal(analysis.nodes, choice.nodes);
  assert.equal(analysis.rootScoreKind, 'alpha-beta-return-v1');
  assert.equal(analysis.rootScores.length, getLegalActions(state).length);
  assert.ok(analysis.rootScores.some((entry) => entry.searched && Number.isFinite(entry.score)));
  assert.ok(analysis.rootScores.some((entry) => !entry.searched && entry.score === null));
});
