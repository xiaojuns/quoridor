import assert from "node:assert/strict";
import test from "node:test";

import { isAbortError } from "../src/agents/agent.js";
import { ClassicWorkerAgent } from "../src/agents/worker-agent.js";

class FakeWorker {
  constructor() {
    this.messages = [];
    this.terminateCount = 0;
    this.postMessageError = null;
    this.listeners = new Map();
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }

  postMessage(message) {
    if (this.postMessageError) {
      throw this.postMessageError;
    }
    this.messages.push(message);
  }

  terminate() {
    this.terminateCount += 1;
  }

  emit(type, event) {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event);
    }
  }

  respond(data) {
    this.emit("message", { data });
  }
}

function createHarness(agentOptions = {}) {
  const workers = [];
  const factoryCalls = [];
  const workerFactory = (workerUrl, options) => {
    const worker = new FakeWorker();
    workers.push(worker);
    factoryCalls.push({ workerUrl, options });
    return worker;
  };
  const agent = new ClassicWorkerAgent({ workerFactory, ...agentOptions });
  return { agent, workers, factoryCalls };
}

function resultMessage(requestId, overrides = {}) {
  return {
    type: "result",
    requestId,
    result: {
      action: { type: "move", row: 7, col: 4 },
      nodes: 23,
      elapsedMs: 4,
      depth: 2,
      score: 17,
      engine: "classic-negamax",
      ...overrides,
    },
  };
}

test("choose lazily creates and reuses a module worker", async () => {
  const { agent, workers, factoryCalls } = createHarness();
  assert.equal(agent.engine, "classic-negamax-worker");
  assert.equal(workers.length, 0);

  const first = agent.choose({ currentPlayer: 0 }, { difficulty: "medium" });
  assert.equal(workers.length, 1);
  assert.deepEqual(factoryCalls[0].options, { type: "module" });
  assert.deepEqual(workers[0].messages[0], {
    type: "choose",
    requestId: 1,
    state: { currentPlayer: 0 },
    options: { difficulty: "medium" },
  });
  workers[0].respond(resultMessage(1));

  const firstResult = await first;
  assert.deepEqual(firstResult.action, { type: "move", row: 7, col: 4 });
  assert.equal(firstResult.nodes, 23);
  assert.equal(firstResult.engine, "classic-negamax-worker");

  const second = agent.choose({ currentPlayer: 1 });
  assert.equal(workers.length, 1);
  assert.equal(workers[0].messages[1].requestId, 2);
  workers[0].respond(resultMessage(2, { nodes: 31 }));
  assert.equal((await second).nodes, 31);
});

test("a newer choose aborts the old request and replaces its busy worker", async () => {
  const { agent, workers } = createHarness();
  const oldRequest = agent.choose({ position: "old" });
  const oldRejection = assert.rejects(oldRequest, isAbortError);
  const oldWorker = workers[0];

  const currentRequest = agent.choose({ position: "current" });
  await oldRejection;
  assert.equal(oldWorker.terminateCount, 1);
  assert.equal(workers.length, 2);

  oldWorker.respond(resultMessage(1, { nodes: 999 }));
  workers[1].respond(resultMessage(2, { nodes: 42 }));
  assert.equal((await currentRequest).nodes, 42);
});

test("stale request ids are ignored while the current request remains pending", async () => {
  const { agent, workers } = createHarness();
  const first = agent.choose({ turn: 1 });
  workers[0].respond(resultMessage(1));
  await first;

  let settled = false;
  const second = agent.choose({ turn: 2 }).finally(() => {
    settled = true;
  });
  workers[0].respond(resultMessage(1, { nodes: 1000 }));
  await Promise.resolve();
  assert.equal(settled, false);

  workers[0].respond(resultMessage(2, { nodes: 7 }));
  assert.equal((await second).nodes, 7);
});

test("abort terminates an active worker and the next choose recreates it", async () => {
  const { agent, workers } = createHarness();
  const alreadyAborted = new AbortController();
  alreadyAborted.abort("not started");
  await assert.rejects(
    agent.choose({}, {}, alreadyAborted.signal),
    isAbortError,
  );
  assert.equal(workers.length, 0);

  const controller = new AbortController();
  const pending = agent.choose({}, {}, controller.signal);
  const rejection = assert.rejects(pending, isAbortError);
  controller.abort("cancel search");
  await rejection;
  assert.equal(workers[0].terminateCount, 1);

  const next = agent.choose({});
  assert.equal(workers.length, 2);
  workers[1].respond(resultMessage(2));
  await next;
});

test("a hard timeout rejects, terminates, and lazily replaces the worker", async () => {
  const { agent, workers } = createHarness({ requestTimeoutMs: 10 });
  const timedOut = agent.choose({ slow: true });

  await assert.rejects(timedOut, {
    name: "TimeoutError",
    message: /within 10 ms/,
  });
  assert.equal(workers[0].terminateCount, 1);

  const next = agent.choose({ slow: false });
  assert.equal(workers.length, 2);
  workers[1].respond(resultMessage(2));
  await next;
});

test("request timeouts can be disabled and constructor values are validated", async () => {
  assert.throws(
    () => createHarness({ requestTimeoutMs: -1 }),
    /requestTimeoutMs/,
  );
  assert.throws(
    () => createHarness({ requestTimeoutMs: Infinity }),
    /requestTimeoutMs/,
  );

  const { agent, workers } = createHarness({ requestTimeoutMs: 0 });
  let settled = false;
  const pending = agent.choose({}).finally(() => {
    settled = true;
  });
  await new Promise((resolve) => setTimeout(resolve, 15));
  assert.equal(settled, false);
  workers[0].respond(resultMessage(1));
  await pending;
});

test("worker, remote, postMessage, and protocol errors reject predictably", async () => {
  const { agent, workers } = createHarness();

  const crashed = agent.choose({ case: "worker error" });
  const crashError = new Error("worker crashed");
  workers[0].emit("error", { error: crashError });
  await assert.rejects(crashed, (error) => error === crashError);
  assert.equal(workers[0].terminateCount, 1);

  const remote = agent.choose({ case: "remote error" });
  workers[1].respond({
    type: "error",
    requestId: 2,
    error: { name: "RangeError", message: "bad search option" },
  });
  await assert.rejects(
    remote,
    (error) => error.name === "RangeError" && error.message === "bad search option",
  );
  assert.equal(workers[1].terminateCount, 0);

  const malformed = agent.choose({ case: "bad protocol" });
  workers[1].respond({ type: "result", requestId: 3, result: null });
  await assert.rejects(malformed, { name: "WorkerProtocolError" });
  assert.equal(workers[1].terminateCount, 1);

  const postFailure = new Error("could not clone state");
  const nextWorkerIndex = workers.length;
  const invalidStateRequest = agent.choose({ case: "clone error" });
  // postMessage already ran, so exercise a synchronous failure on a fresh request.
  workers[nextWorkerIndex].respond(resultMessage(4));
  await invalidStateRequest;
  workers[nextWorkerIndex].postMessageError = postFailure;
  await assert.rejects(agent.choose({ case: "clone error" }), (error) => error === postFailure);
  assert.equal(workers[nextWorkerIndex].terminateCount, 1);
});

test("dispose is idempotent, aborts pending work, and prevents future requests", async () => {
  const { agent, workers } = createHarness();
  const pending = agent.choose({});
  const rejection = assert.rejects(pending, isAbortError);

  agent.dispose();
  agent.dispose();
  await rejection;
  assert.equal(workers[0].terminateCount, 1);
  await assert.rejects(agent.choose({}), /disposed/);
  assert.equal(workers.length, 1);
});
