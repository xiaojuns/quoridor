export const ABORT_ERROR_NAME = 'AbortError';

const DEFAULT_ABORT_MESSAGE = 'The operation was aborted';

function finiteNumber(value, fallback) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function nonNegativeNumber(value, fallback = 0) {
  const number = finiteNumber(value, fallback);
  return number >= 0 ? number : fallback;
}

function nullableNumber(value, fallback = null) {
  return finiteNumber(value, finiteNumber(fallback, null));
}

/**
 * Creates the one cancellation error shape shared by agents and workers.
 */
export function createAbortError(reason) {
  if (isAbortError(reason)) return reason;

  const message =
    typeof reason === 'string' && reason.length > 0
      ? reason
      : reason instanceof Error && reason.message
        ? reason.message
        : DEFAULT_ABORT_MESSAGE;

  if (typeof DOMException === 'function') {
    return new DOMException(message, ABORT_ERROR_NAME);
  }

  const error = new Error(message);
  error.name = ABORT_ERROR_NAME;
  return error;
}

export function isAbortError(error) {
  return Boolean(
    error &&
      (error.name === ABORT_ERROR_NAME || error.code === 'ABORT_ERR'),
  );
}

export function throwIfAborted(signal) {
  if (signal?.aborted) {
    throw createAbortError(signal.reason);
  }
}

/**
 * Returns the stable result shape consumed by the UI and benchmark tooling.
 */
export function normalizeAgentResult(result = {}, defaults = {}) {
  const source = result && typeof result === 'object' ? result : {};
  const fallback = defaults && typeof defaults === 'object' ? defaults : {};
  const engine =
    typeof source.engine === 'string' && source.engine.length > 0
      ? source.engine
      : typeof fallback.engine === 'string' && fallback.engine.length > 0
        ? fallback.engine
        : null;

  return {
    action: source.action ?? fallback.action ?? null,
    value: nullableNumber(source.value, fallback.value),
    simulations: nonNegativeNumber(
      source.simulations,
      nonNegativeNumber(fallback.simulations),
    ),
    nodes: nonNegativeNumber(source.nodes, nonNegativeNumber(fallback.nodes)),
    elapsedMs: nonNegativeNumber(
      source.elapsedMs,
      nonNegativeNumber(fallback.elapsedMs),
    ),
    engine,
    depth: nonNegativeNumber(source.depth, nonNegativeNumber(fallback.depth)),
    score: nullableNumber(source.score, fallback.score),
  };
}

/**
 * Base contract for every browser AI implementation.
 */
export class Agent {
  async choose(_state, _options = {}, _signal) {
    throw new Error('Agent.choose() must be implemented');
  }
}
