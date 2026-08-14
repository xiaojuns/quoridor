import {
  isAbortError,
  normalizeAgentResult,
} from "../agents/agent.js";
import { ClassicAgent } from "../agents/classic-agent.js";

const agent = new ClassicAgent();

function serializeError(error) {
  const source = error instanceof Error ? error : new Error(String(error));
  return {
    name: isAbortError(source) ? "AbortError" : source.name || "Error",
    message: source.message,
    stack: source.stack,
  };
}

function postError(requestId, error) {
  self.postMessage({
    type: "error",
    requestId,
    error: serializeError(error),
  });
}

self.addEventListener("message", async (event) => {
  const message = event?.data;
  const requestId = message?.requestId;

  if (
    message === null ||
    typeof message !== "object" ||
    message.type !== "choose" ||
    !Number.isInteger(requestId)
  ) {
    postError(
      Number.isInteger(requestId) ? requestId : null,
      new TypeError("Invalid classic AI worker request"),
    );
    return;
  }

  try {
    const result = await agent.choose(
      message.state,
      message.options && typeof message.options === "object"
        ? message.options
        : {},
    );
    self.postMessage({
      type: "result",
      requestId,
      result: normalizeAgentResult(result, { engine: agent.engine }),
    });
  } catch (error) {
    postError(requestId, error);
  }
});
