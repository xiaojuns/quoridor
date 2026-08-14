import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ABORT_ERROR_NAME,
  Agent,
  createAbortError,
  isAbortError,
  normalizeAgentResult,
  throwIfAborted,
} from '../src/agents/agent.js';
import {
  CLASSIC_AGENT_ENGINE,
  ClassicAgent,
  classicAgent,
  createClassicAgent,
} from '../src/agents/classic-agent.js';
import { createInitialState, getLegalActions } from '../src/game.js';

function containsAction(actions, expected) {
  return actions.some((action) => JSON.stringify(action) === JSON.stringify(expected));
}

test('normalizes all agent result fields', () => {
  assert.deepEqual(normalizeAgentResult(), {
    action: null,
    value: null,
    simulations: 0,
    nodes: 0,
    elapsedMs: 0,
    engine: null,
    depth: 0,
    score: null,
  });

  const action = { type: 'move', row: 7, col: 4 };
  assert.deepEqual(
    normalizeAgentResult(
      {
        action,
        value: 0.25,
        simulations: 64,
        nodes: 123,
        elapsedMs: 4.5,
        engine: 'test-engine',
        depth: 3,
        score: -42,
      },
    ),
    {
      action,
      value: 0.25,
      simulations: 64,
      nodes: 123,
      elapsedMs: 4.5,
      engine: 'test-engine',
      depth: 3,
      score: -42,
    },
  );
});

test('provides one AbortError shape and detection helper', () => {
  const error = createAbortError('search cancelled');
  assert.equal(error.name, ABORT_ERROR_NAME);
  assert.equal(error.message, 'search cancelled');
  assert.equal(isAbortError(error), true);
  assert.equal(isAbortError({ code: 'ABORT_ERR' }), true);
  assert.equal(isAbortError(new Error('failure')), false);

  const controller = new AbortController();
  controller.abort('new game');
  assert.throws(
    () => throwIfAborted(controller.signal),
    (thrown) => isAbortError(thrown) && thrown.message === 'new game',
  );
});

test('base Agent documents an asynchronous choose contract', async () => {
  const result = new Agent().choose({});
  assert.equal(typeof result.then, 'function');
  await assert.rejects(result, /must be implemented/);
});

test('classic adapter preserves Negamax output in the unified shape', async () => {
  const state = createInitialState();
  const result = await classicAgent.choose(state, {
    difficulty: 'easy',
    timeLimitMs: 1_000,
  });

  assert.ok(containsAction(getLegalActions(state), result.action));
  assert.equal(result.engine, CLASSIC_AGENT_ENGINE);
  assert.equal(result.value, null);
  assert.equal(result.simulations, 0);
  assert.equal(typeof result.score, 'number');
  assert.ok(Number.isInteger(result.depth) && result.depth >= 0);
  assert.ok(Number.isInteger(result.nodes) && result.nodes > 0);
  assert.ok(Number.isFinite(result.elapsedMs) && result.elapsedMs >= 0);
});

test('classic adapter accepts an injected sync or async search function', async () => {
  const expected = {
    action: { type: 'move', row: 1, col: 2 },
    score: 17,
    depth: 4,
    nodes: 55,
    elapsedMs: 6,
  };
  const calls = [];
  const agent = createClassicAgent({
    search: async (...args) => {
      calls.push(args);
      return expected;
    },
  });
  const state = { marker: 'state' };
  const options = { marker: 'options' };
  const controller = new AbortController();
  const result = await agent.choose(state, options, controller.signal);

  assert.equal(agent instanceof ClassicAgent, true);
  assert.deepEqual(calls, [[state, options, controller.signal]]);
  assert.deepEqual(result, {
    ...expected,
    value: null,
    simulations: 0,
    engine: CLASSIC_AGENT_ENGINE,
  });
});

test('classic adapter checks cancellation before starting search', async () => {
  let calls = 0;
  const agent = createClassicAgent({
    search() {
      calls += 1;
      return {};
    },
  });
  const controller = new AbortController();
  controller.abort();

  await assert.rejects(agent.choose({}, {}, controller.signal), isAbortError);
  assert.equal(calls, 0);
});

test('classic adapter checks cancellation after search completion', async () => {
  const controller = new AbortController();
  const agent = createClassicAgent({
    search() {
      controller.abort('stale request');
      return { action: { type: 'move', row: 0, col: 0 } };
    },
  });

  await assert.rejects(
    agent.choose({}, {}, controller.signal),
    (error) => isAbortError(error) && error.message === 'stale request',
  );
});

test('classic adapter validates search injection and propagates failures', async () => {
  assert.throws(() => createClassicAgent({ search: null }), TypeError);

  const failure = new Error('search failed');
  const agent = createClassicAgent({
    search() {
      throw failure;
    },
  });
  await assert.rejects(agent.choose({}), (error) => error === failure);
});
