import {
  Agent,
  createAbortError,
  normalizeAgentResult,
} from "./agent.js";

const DEFAULT_WORKER_URL = new URL(
  "../workers/classic-ai.worker.js",
  import.meta.url,
);

function defaultWorkerFactory(workerUrl) {
  if (typeof Worker !== "function") {
    throw new Error("Web Workers are not available in this environment");
  }
  return new Worker(workerUrl, { type: "module" });
}

function protocolError(message, cause) {
  const error = new Error(`Classic AI worker protocol error: ${message}`);
  error.name = "WorkerProtocolError";
  if (cause !== undefined) {
    error.cause = cause;
  }
  return error;
}

function timeoutError(timeoutMs) {
  const error = new Error(
    `Classic AI worker did not respond within ${timeoutMs} ms`,
  );
  error.name = "TimeoutError";
  return error;
}

function deserializeWorkerError(payload) {
  if (
    payload === null ||
    typeof payload !== "object" ||
    typeof payload.message !== "string"
  ) {
    throw protocolError("invalid error payload");
  }

  const error = new Error(payload.message);
  error.name = typeof payload.name === "string" ? payload.name : "Error";
  if (typeof payload.stack === "string") {
    error.stack = payload.stack;
  }
  return error;
}

export class ClassicWorkerAgent extends Agent {
  engine = "classic-negamax-worker";

  #workerFactory;
  #workerUrl;
  #worker = null;
  #workerListeners = null;
  #pending = null;
  #nextRequestId = 1;
  #disposed = false;
  #requestTimeoutMs;

  constructor({
    workerFactory = defaultWorkerFactory,
    workerUrl = DEFAULT_WORKER_URL,
    requestTimeoutMs = 5000,
  } = {}) {
    super();
    if (typeof workerFactory !== "function") {
      throw new TypeError("workerFactory must be a function");
    }
    if (
      requestTimeoutMs !== null &&
      (typeof requestTimeoutMs !== "number" ||
        !Number.isFinite(requestTimeoutMs) ||
        requestTimeoutMs < 0)
    ) {
      throw new RangeError("requestTimeoutMs must be a non-negative finite number or null");
    }
    this.#workerFactory = workerFactory;
    this.#workerUrl = workerUrl;
    this.#requestTimeoutMs = requestTimeoutMs;
  }

  choose(state, options = {}, signal) {
    if (this.#disposed) {
      return Promise.reject(new Error("ClassicWorkerAgent has been disposed"));
    }
    if (signal?.aborted) {
      return Promise.reject(createAbortError(signal.reason));
    }

    if (this.#pending) {
      this.#rejectPending(createAbortError("Superseded by a newer AI request"));
      this.#terminateWorker();
    }

    let worker;
    try {
      worker = this.#ensureWorker();
    } catch (error) {
      return Promise.reject(error);
    }

    const requestId = this.#nextRequestId;
    this.#nextRequestId += 1;

    return new Promise((resolve, reject) => {
      const onAbort = () => {
        if (this.#pending?.requestId !== requestId) {
          return;
        }
        this.#rejectPending(createAbortError(signal?.reason));
        this.#terminateWorker();
      };

      this.#pending = {
        requestId,
        resolve,
        reject,
        signal,
        onAbort,
        timeoutId: null,
      };
      signal?.addEventListener("abort", onAbort, { once: true });

      if (this.#requestTimeoutMs > 0) {
        this.#pending.timeoutId = setTimeout(() => {
          if (this.#pending?.requestId !== requestId) {
            return;
          }
          this.#rejectPending(timeoutError(this.#requestTimeoutMs));
          this.#terminateWorker();
        }, this.#requestTimeoutMs);
      }

      try {
        worker.postMessage({ type: "choose", requestId, state, options });
      } catch (error) {
        this.#rejectPending(error);
        this.#terminateWorker();
      }
    });
  }

  dispose() {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    if (this.#pending) {
      this.#rejectPending(createAbortError("ClassicWorkerAgent was disposed"));
    }
    this.#terminateWorker();
  }

  #ensureWorker() {
    if (this.#worker) {
      return this.#worker;
    }

    const worker = this.#workerFactory(this.#workerUrl, { type: "module" });
    if (
      !worker ||
      typeof worker.postMessage !== "function" ||
      typeof worker.terminate !== "function"
    ) {
      throw new TypeError("workerFactory must return a Worker-compatible object");
    }

    const listeners = {
      message: (event) => this.#handleMessage(worker, event),
      error: (event) => this.#handleWorkerError(worker, event),
      messageerror: () =>
        this.#failProtocol(worker, "the response could not be deserialized"),
    };

    if (typeof worker.addEventListener === "function") {
      for (const [type, listener] of Object.entries(listeners)) {
        worker.addEventListener(type, listener);
      }
    } else {
      worker.onmessage = listeners.message;
      worker.onerror = listeners.error;
      worker.onmessageerror = listeners.messageerror;
    }

    this.#worker = worker;
    this.#workerListeners = listeners;
    return worker;
  }

  #handleMessage(worker, event) {
    if (worker !== this.#worker || !this.#pending) {
      return;
    }

    const message = event?.data;
    if (message === null || typeof message !== "object") {
      this.#failProtocol(worker, "response must be an object");
      return;
    }
    if (!Number.isInteger(message.requestId)) {
      this.#failProtocol(worker, "response requestId must be an integer");
      return;
    }
    if (message.requestId !== this.#pending.requestId) {
      return;
    }

    if (message.type === "result") {
      if (
        message.result === null ||
        typeof message.result !== "object" ||
        Array.isArray(message.result)
      ) {
        this.#failProtocol(worker, "invalid result payload");
        return;
      }
      try {
        this.#resolvePending(normalizeAgentResult(
          { ...message.result, engine: this.engine },
          { engine: this.engine },
        ));
      } catch (error) {
        this.#failProtocol(worker, "invalid result payload", error);
      }
      return;
    }

    if (message.type === "error") {
      try {
        this.#rejectPending(deserializeWorkerError(message.error));
      } catch (error) {
        this.#failProtocol(worker, "invalid error response", error);
      }
      return;
    }

    this.#failProtocol(worker, `unknown response type: ${String(message.type)}`);
  }

  #handleWorkerError(worker, event) {
    if (worker !== this.#worker) {
      return;
    }
    const error =
      event?.error instanceof Error
        ? event.error
        : new Error(event?.message || "Classic AI worker failed");
    if (this.#pending) {
      this.#rejectPending(error);
    }
    this.#terminateWorker();
  }

  #failProtocol(worker, message, cause) {
    if (worker !== this.#worker) {
      return;
    }
    if (this.#pending) {
      this.#rejectPending(protocolError(message, cause));
    }
    this.#terminateWorker();
  }

  #resolvePending(result) {
    const pending = this.#takePending();
    pending?.resolve(result);
  }

  #rejectPending(error) {
    const pending = this.#takePending();
    pending?.reject(error);
  }

  #takePending() {
    const pending = this.#pending;
    if (!pending) {
      return null;
    }
    this.#pending = null;
    pending.signal?.removeEventListener("abort", pending.onAbort);
    if (pending.timeoutId !== null) {
      clearTimeout(pending.timeoutId);
    }
    return pending;
  }

  #terminateWorker() {
    const worker = this.#worker;
    if (!worker) {
      return;
    }

    const listeners = this.#workerListeners;
    this.#worker = null;
    this.#workerListeners = null;
    if (listeners && typeof worker.removeEventListener === "function") {
      for (const [type, listener] of Object.entries(listeners)) {
        worker.removeEventListener(type, listener);
      }
    } else {
      worker.onmessage = null;
      worker.onerror = null;
      worker.onmessageerror = null;
    }
    try {
      worker.terminate();
    } catch {
      // The transport is already detached; termination is best-effort.
    }
  }
}

export { ClassicWorkerAgent as WorkerAgent };
export default ClassicWorkerAgent;
