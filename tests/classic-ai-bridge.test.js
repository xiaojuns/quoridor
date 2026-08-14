import assert from 'node:assert/strict';
import { once } from 'node:events';
import { spawn, spawnSync } from 'node:child_process';
import readline from 'node:readline';
import test from 'node:test';

import { applyAction, createInitialState, getLegalActions } from '../src/game.js';
import { CODEC_VERSION, positionHash } from '../src/alphazero/state-codec.js';

class JsonLineReader {
  constructor(stream) {
    this.lines = [];
    this.waiters = [];
    this.closed = false;
    this.input = readline.createInterface({ input: stream, crlfDelay: Infinity });
    this.input.on('line', (line) => {
      const waiter = this.waiters.shift();
      if (waiter) waiter.resolve(line);
      else this.lines.push(line);
    });
    this.input.on('close', () => {
      this.closed = true;
      for (const waiter of this.waiters.splice(0)) {
        waiter.reject(new Error('bridge stdout closed'));
      }
    });
  }

  async next(timeoutMs = 3_000) {
    let line;
    if (this.lines.length > 0) {
      line = this.lines.shift();
    } else {
      if (this.closed) throw new Error('bridge stdout closed');
      line = await new Promise((resolve, reject) => {
        const waiter = { resolve, reject };
        this.waiters.push(waiter);
        const timer = setTimeout(() => {
          const index = this.waiters.indexOf(waiter);
          if (index >= 0) this.waiters.splice(index, 1);
          reject(new Error('timed out waiting for bridge response'));
        }, timeoutMs);
        waiter.resolve = (value) => {
          clearTimeout(timer);
          resolve(value);
        };
        waiter.reject = (error) => {
          clearTimeout(timer);
          reject(error);
        };
      });
    }
    return JSON.parse(line);
  }
}

function serializableState(state) {
  const hash = positionHash(state);
  return {
    codecVersion: CODEC_VERSION,
    pawns: state.pawns.map(({ row, col }) => ({ row, col })),
    wallsRemaining: [...state.wallsRemaining],
    horizontalWalls: [...state.horizontalWalls].sort(),
    verticalWalls: [...state.verticalWalls].sort(),
    currentPlayer: state.currentPlayer,
    winner: state.winner,
    moveNumber: state.moveNumber,
    terminationReason: state.terminationReason,
    repetitionCount: state.positionCounts.get(hash),
    positionCounts: [...state.positionCounts]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([position, count]) => ({ hash: position, count })),
  };
}

function send(child, message) {
  child.stdin.write(`${JSON.stringify(message)}\n`);
}

test('classic bridge preserves full history, checks hashes, and remains persistent', async (t) => {
  const probe = spawnSync(process.execPath, ['-e', ''], { encoding: 'utf8' });
  if (probe.error?.code === 'EPERM') {
    t.skip('sandbox forbids Node from spawning the bridge child process');
    return;
  }
  const env = { ...process.env };
  delete env.NODE_TEST_CONTEXT;
  const child = spawn(process.execPath, ['scripts/classic-ai-bridge.mjs'], {
    cwd: process.cwd(),
    stdio: ['pipe', 'pipe', 'pipe'],
    env,
  });
  const reader = new JsonLineReader(child.stdout);
  let stderr = '';
  let stage = 'ready';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => {
    stderr += chunk;
  });

  try {
    const ready = await reader.next();
    assert.deepEqual(ready.hardDefaults, { timeLimitMs: 850, maxDepth: 5 });
    assert.equal(ready.engine, 'classic-negamax');

    let state = createInitialState();
    for (const action of [
      { type: 'move', row: 8, col: 3 },
      { type: 'move', row: 0, col: 3 },
      { type: 'move', row: 8, col: 4 },
      { type: 'move', row: 0, col: 4 },
    ]) {
      state = applyAction(state, action);
    }
    assert.equal(state.positionCounts.get(positionHash(state)), 2);

    send(child, {
      version: 1,
      type: 'choose',
      id: 1,
      positionHash: positionHash(state),
      state: serializableState(state),
      budget: { kind: 'nodes', nodeLimit: 48 },
      maxDepth: 5,
    });
    stage = 'first result';
    const result = await reader.next();
    assert.equal(result.type, 'result');
    assert.equal(result.id, 1);
    assert.equal(result.positionHash, positionHash(state));
    assert.equal(result.historyEntries, state.positionCounts.size);
    assert.equal(
      result.historyOccurrences,
      [...state.positionCounts.values()].reduce((total, count) => total + count, 0),
    );
    assert.equal(result.nodes, 48);
    assert.ok(
      getLegalActions(state).some(
        (action) => JSON.stringify(action) === JSON.stringify(result.action),
      ),
    );

    send(child, {
      version: 1,
      type: 'choose',
      id: 2,
      positionHash: 'not-the-python-hash',
      state: serializableState(state),
      budget: { kind: 'nodes', nodeLimit: 8 },
    });
    stage = 'hash mismatch';
    const mismatch = await reader.next();
    assert.equal(mismatch.type, 'error');
    assert.equal(mismatch.id, 2);
    assert.equal(mismatch.error.code, 'position-hash-mismatch');

    send(child, {
      version: 1,
      type: 'choose',
      id: 3,
      positionHash: positionHash(state),
      state: serializableState(state),
      budget: { kind: 'nodes', nodeLimit: 8 },
    });
    stage = 'result after error';
    const afterError = await reader.next();
    assert.equal(afterError.type, 'result');
    assert.equal(afterError.id, 3);

    const exit = once(child, 'exit');
    send(child, { version: 1, type: 'close', id: 4 });
    stage = 'close response';
    const closed = await reader.next();
    assert.deepEqual(closed, { version: 1, type: 'closed', id: 4 });
    const [exitCode] = await exit;
    assert.equal(exitCode, 0, stderr);
  } catch (error) {
    error.message =
      `${error.message}; stage: ${stage}; ` +
      `exit: ${child.exitCode}/${child.signalCode}; bridge stderr: ${stderr}`;
    throw error;
  } finally {
    if (child.exitCode === null) child.kill('SIGKILL');
  }
});
