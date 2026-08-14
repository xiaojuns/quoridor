import { chooseAIMove } from '../ai.js';
import {
  Agent,
  createAbortError,
  normalizeAgentResult,
  throwIfAborted,
} from './agent.js';

export const CLASSIC_AGENT_ENGINE = 'classic-negamax';

export class ClassicAgent extends Agent {
  constructor({ search = chooseAIMove } = {}) {
    super();
    if (typeof search !== 'function') {
      throw new TypeError('ClassicAgent search must be a function');
    }
    this.search = search;
  }

  async choose(state, options = {}, signal) {
    throwIfAborted(signal);

    let result;
    try {
      result = await this.search(state, options, signal);
    } catch (error) {
      if (signal?.aborted) throw createAbortError(signal.reason);
      throw error;
    }

    throwIfAborted(signal);
    return normalizeAgentResult(
      { ...result, engine: CLASSIC_AGENT_ENGINE },
      {
        value: null,
        simulations: 0,
        nodes: 0,
        elapsedMs: 0,
        depth: 0,
        score: null,
      },
    );
  }
}

export function createClassicAgent(options) {
  return new ClassicAgent(options);
}

export const classicAgent = createClassicAgent();
