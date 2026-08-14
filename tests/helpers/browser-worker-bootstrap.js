import { parentPort } from "node:worker_threads";

if (!parentPort) {
  throw new Error("Browser worker bootstrap requires a parent port");
}

const messageListeners = new Set();

globalThis.self = {
  addEventListener(type, listener) {
    if (type !== "message") {
      throw new Error(`Unsupported worker event type: ${type}`);
    }
    messageListeners.add(listener);
  },
  postMessage(message) {
    parentPort.postMessage(message);
  },
};

await import("../../src/workers/classic-ai.worker.js");

parentPort.on("message", (data) => {
  for (const listener of messageListeners) {
    listener({ data });
  }
});
