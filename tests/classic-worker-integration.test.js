import assert from "node:assert/strict";
import test from "node:test";
import { Worker } from "node:worker_threads";

import {
  applyAction,
  createInitialState,
  getLegalActions,
} from "../src/game.js";

function waitForWorkerMessage(worker, timeoutMs = 5_000) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error("Timed out waiting for classic worker response"));
    }, timeoutMs);
    const onMessage = (message) => {
      cleanup();
      resolve(message);
    };
    const onError = (error) => {
      cleanup();
      reject(error);
    };
    const cleanup = () => {
      clearTimeout(timeout);
      worker.off("message", onMessage);
      worker.off("error", onError);
    };

    worker.once("message", onMessage);
    worker.once("error", onError);
  });
}

test("the real worker entry receives a structured-cloned state and returns a legal move", async () => {
  const worker = new Worker(
    new URL("./helpers/browser-worker-bootstrap.js", import.meta.url),
    { type: "module" },
  );
  const state = createInitialState();

  try {
    const responsePromise = waitForWorkerMessage(worker);
    worker.postMessage({
      type: "choose",
      requestId: 17,
      state,
      options: {
        difficulty: "easy",
        maxDepth: 1,
        timeLimitMs: 1_000,
      },
    });
    const response = await responsePromise;

    assert.equal(response.type, "result");
    assert.equal(response.requestId, 17);
    assert.equal(response.result.engine, "classic-negamax");
    assert.ok(
      getLegalActions(state).some(
        (action) => JSON.stringify(action) === JSON.stringify(response.result.action),
      ),
    );
    assert.doesNotThrow(() => applyAction(state, response.result.action));
    assert.ok(response.result.nodes > 0);
  } finally {
    await worker.terminate();
  }
});
