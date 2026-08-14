import {
  BOARD_SIZE,
  applyAction,
  cloneState,
  getLegalActions,
  getLegalPawnMoves,
  getWinner,
  serializeState,
  shortestPathDistance,
} from './game.js';

export const WIN_SCORE = 1_000_000;

const DIFFICULTY = Object.freeze({
  easy: {
    maxDepth: 1,
    timeLimitMs: 80,
    wallLimit: 6,
    deepWallLimit: 4,
    prefilterLimit: 48,
  },
  medium: {
    maxDepth: 3,
    timeLimitMs: 350,
    wallLimit: 9,
    deepWallLimit: 6,
    prefilterLimit: 64,
  },
  hard: {
    maxDepth: 5,
    timeLimitMs: 850,
    wallLimit: 13,
    deepWallLimit: 9,
    prefilterLimit: 96,
  },
});

const TT_EXACT = 0;
const TT_LOWER = 1;
const TT_UPPER = 2;
const TIMEOUT = Symbol('AI_SEARCH_TIMEOUT');

function now() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function currentPlayerOf(state) {
  return Number.isInteger(state?.currentPlayer) ? state.currentPlayer : 0;
}

function opponentOf(player) {
  return player === 0 ? 1 : 0;
}

function samePlayer(winner, player) {
  if (winner === player) return true;
  if (winner && typeof winner === 'object') {
    return winner.index === player || winner.player === player || winner.id === player;
  }
  return false;
}

function remainingWalls(state, player) {
  const playerState = state?.players?.[player];
  const value =
    playerState?.wallsRemaining ??
    playerState?.remainingWalls ??
    playerState?.walls ??
    state?.wallsRemaining?.[player] ??
    state?.wallCounts?.[player] ??
    0;

  return Number.isFinite(value) ? value : 0;
}

function playerPosition(state, player) {
  const playerState = state?.players?.[player];
  const position = playerState?.position ?? state?.pawns?.[player] ?? playerState;

  if (Array.isArray(position) && position.length >= 2) {
    return { row: position[0], col: position[1] };
  }
  if (Number.isFinite(position?.row) && Number.isFinite(position?.col)) {
    return { row: position.row, col: position.col };
  }
  return null;
}

function actionPosition(action) {
  const position = action?.position ?? action?.to ?? action?.wall ?? action;
  if (Array.isArray(position) && position.length >= 2) {
    return { row: position[0], col: position[1] };
  }
  if (Number.isFinite(position?.row) && Number.isFinite(position?.col)) {
    return { row: position.row, col: position.col };
  }
  return null;
}

function actionKey(action) {
  if (action === null || action === undefined) return '';
  return JSON.stringify(action);
}

function actionsEqual(left, right) {
  return left === right || actionKey(left) === actionKey(right);
}

function isWallAction(action) {
  return (
    action?.type === 'wall' ||
    action?.type === 'placeWall' ||
    action?.orientation === 'horizontal' ||
    action?.orientation === 'vertical' ||
    action?.orientation === 'h' ||
    action?.orientation === 'v'
  );
}

function nextState(state, action) {
  return applyAction(state, action);
}

function finiteDistance(value) {
  return Number.isFinite(value) ? value : BOARD_SIZE * BOARD_SIZE;
}

function distanceFor(state, player) {
  return finiteDistance(shortestPathDistance(state, player));
}

function legalPawnMoveCount(state, player) {
  // Cloning and setting the turn keeps this compatible with engines whose pawn
  // move generator only examines state.currentPlayer.
  const probe = cloneState(state);
  if ('currentPlayer' in probe) probe.currentPlayer = player;

  try {
    const moves = getLegalPawnMoves(probe, player);
    return Array.isArray(moves) ? moves.length : 0;
  } catch {
    return 0;
  }
}

/**
 * Scores a position from `player`'s point of view. Larger values are better.
 */
export function evaluateState(state, player = currentPlayerOf(state), ply = 0) {
  const winner = getWinner(state);
  if (winner !== null && winner !== undefined) {
    return samePlayer(winner, player) ? WIN_SCORE - ply : -WIN_SCORE + ply;
  }

  const opponent = opponentOf(player);
  const ownDistance = distanceFor(state, player);
  const opponentDistance = distanceFor(state, opponent);
  const wallBalance = remainingWalls(state, player) - remainingWalls(state, opponent);
  const mobilityBalance =
    legalPawnMoveCount(state, player) - legalPawnMoveCount(state, opponent);

  return (
    120 * (opponentDistance - ownDistance) +
    7 * wallBalance +
    3 * mobilityBalance
  );
}

function geometricWallScore(state, action, player) {
  const wall = actionPosition(action);
  const opponent = playerPosition(state, opponentOf(player));
  if (!wall || !opponent) return 0;

  const distance = Math.abs(wall.row - opponent.row) + Math.abs(wall.col - opponent.col);
  const centrality = BOARD_SIZE - Math.abs(wall.col - (BOARD_SIZE - 1) / 2);
  return 24 - 3 * distance + centrality * 0.2;
}

function checkDeadline(context, force = false) {
  if (!context || !Number.isFinite(context.deadline)) return;
  if (force || (context.nodes & 31) === 0) {
    if (now() >= context.deadline) throw TIMEOUT;
  }
}

function consumeNode(context) {
  if (Number.isFinite(context?.nodeLimit) && context.nodes >= context.nodeLimit) {
    throw TIMEOUT;
  }
  context.nodes += 1;
}

/**
 * Ranks legal wall actions by the shortest-path change they cause. The result
 * contains diagnostics as well as the action so tests and UI tooling can inspect
 * why a wall was preferred.
 */
export function rankWallCandidates(
  state,
  wallActions,
  player = currentPlayerOf(state),
  options = {},
) {
  if (!Array.isArray(wallActions) || wallActions.length === 0) return [];

  const opponent = opponentOf(player);
  const ownBefore = distanceFor(state, player);
  const opponentBefore = distanceFor(state, opponent);
  const prefilterLimit = Math.max(
    1,
    Math.floor(options.prefilterLimit ?? wallActions.length),
  );
  const context = options.context;

  const prefiltered = wallActions
    .map((action) => ({
      action,
      geometry: geometricWallScore(state, action, player),
      key: actionKey(action),
    }))
    .sort((left, right) => right.geometry - left.geometry || left.key.localeCompare(right.key))
    .slice(0, prefilterLimit);

  return prefiltered
    .map(({ action, geometry, key }) => {
      checkDeadline(context, true);
      const child = nextState(state, action);
      const ownAfter = distanceFor(child, player);
      const opponentAfter = distanceFor(child, opponent);
      const ownDelta = ownAfter - ownBefore;
      const opponentDelta = opponentAfter - opponentBefore;
      const emergencyBonus = opponentBefore <= 2 && opponentDelta > 0 ? 220 : 0;
      const score =
        240 * opponentDelta -
        210 * ownDelta +
        emergencyBonus +
        geometry;

      return { action, score, opponentDelta, ownDelta, key };
    })
    .sort((left, right) => right.score - left.score || left.key.localeCompare(right.key));
}

function actionOrderingScore(state, action, player, rankedWalls, context) {
  checkDeadline(context, true);
  const key = actionKey(action);
  const wallRank = rankedWalls.get(key);
  if (wallRank) return wallRank.score;

  const opponent = opponentOf(player);
  const ownBefore = distanceFor(state, player);
  const opponentBefore = distanceFor(state, opponent);
  const child = nextState(state, action);
  const winner = getWinner(child);
  if (winner !== null && winner !== undefined && samePlayer(winner, player)) {
    return WIN_SCORE;
  }

  const ownAfter = distanceFor(child, player);
  const opponentAfter = distanceFor(child, opponent);
  return 320 * (ownBefore - ownAfter) + 24 * (opponentAfter - opponentBefore);
}

/**
 * Keeps every legal pawn move and only the strongest wall candidates. Returned
 * actions are also ordered to make alpha-beta cutoffs happen earlier.
 */
export function getCandidateActions(state, options = {}) {
  const difficulty = DIFFICULTY[options.difficulty] ? options.difficulty : 'medium';
  const config = DIFFICULTY[difficulty];
  const player = currentPlayerOf(state);
  const ply = Math.max(0, options.ply ?? 0);
  const legalActions = options.legalActions ?? getLegalActions(state);
  if (!Array.isArray(legalActions) || legalActions.length === 0) return [];

  const pawnActions = [];
  const wallActions = [];
  for (const action of legalActions) {
    (isWallAction(action) ? wallActions : pawnActions).push(action);
  }

  const wallLimit = Math.max(
    0,
    Math.floor(
      options.wallLimit ?? (ply === 0 ? config.wallLimit : config.deepWallLimit),
    ),
  );
  const rankedWalls = rankWallCandidates(state, wallActions, player, {
    prefilterLimit: options.prefilterLimit ?? config.prefilterLimit,
    context: options.context,
  });
  const selectedWalls = rankedWalls.slice(0, wallLimit);
  const wallScores = new Map(selectedWalls.map((entry) => [entry.key, entry]));
  const preferredKey = actionKey(options.preferredAction);

  return [...pawnActions, ...selectedWalls.map((entry) => entry.action)]
    .map((action) => ({
      action,
      key: actionKey(action),
      score: actionOrderingScore(state, action, player, wallScores, options.context),
    }))
    .sort((left, right) => {
      if (preferredKey) {
        if (left.key === preferredKey && right.key !== preferredKey) return -1;
        if (right.key === preferredKey && left.key !== preferredKey) return 1;
      }
      return right.score - left.score || left.key.localeCompare(right.key);
    })
    .map((entry) => entry.action);
}

function transpositionKey(state) {
  const serialized = serializeState(state);
  return typeof serialized === 'string' ? serialized : JSON.stringify(serialized);
}

function negamax(state, depth, alpha, beta, ply, context) {
  consumeNode(context);
  checkDeadline(context);

  const player = currentPlayerOf(state);
  const winner = getWinner(state);
  if (winner !== null && winner !== undefined) {
    return samePlayer(winner, player) ? WIN_SCORE - ply : -WIN_SCORE + ply;
  }
  if (depth <= 0) return evaluateState(state, player, ply);

  const key = transpositionKey(state);
  const cached = context.table.get(key);
  const originalAlpha = alpha;
  const originalBeta = beta;
  if (cached && cached.depth >= depth) {
    if (cached.flag === TT_EXACT) return cached.score;
    if (cached.flag === TT_LOWER) alpha = Math.max(alpha, cached.score);
    if (cached.flag === TT_UPPER) beta = Math.min(beta, cached.score);
    if (alpha >= beta) return cached.score;
  }

  const actions = getCandidateActions(state, {
    difficulty: context.difficulty,
    ply,
    preferredAction: cached?.bestAction,
    context,
  });
  if (actions.length === 0) return evaluateState(state, player, ply);

  let bestScore = -Infinity;
  let bestAction = actions[0];
  for (const action of actions) {
    checkDeadline(context, true);
    const child = nextState(state, action);
    const childWinner = getWinner(child);
    const score =
      childWinner !== null && childWinner !== undefined && samePlayer(childWinner, player)
        ? WIN_SCORE - (ply + 1)
        : -negamax(child, depth - 1, -beta, -alpha, ply + 1, context);

    if (score > bestScore) {
      bestScore = score;
      bestAction = action;
    }
    alpha = Math.max(alpha, score);
    if (alpha >= beta) break;
  }

  let flag = TT_EXACT;
  if (bestScore <= originalAlpha) flag = TT_UPPER;
  else if (bestScore >= originalBeta) flag = TT_LOWER;
  context.table.set(key, { depth, score: bestScore, flag, bestAction });
  return bestScore;
}

function searchRoot(state, depth, context, preferredAction) {
  const player = currentPlayerOf(state);
  const actions = getCandidateActions(state, {
    difficulty: context.difficulty,
    ply: 0,
    preferredAction,
    context,
  });
  if (actions.length === 0) return { action: null, score: evaluateState(state, player) };

  let alpha = -Infinity;
  const beta = Infinity;
  let bestAction = actions[0];
  let bestScore = -Infinity;

  for (const action of actions) {
    checkDeadline(context, true);
    const child = nextState(state, action);
    const winner = getWinner(child);
    const score =
      winner !== null && winner !== undefined && samePlayer(winner, player)
        ? WIN_SCORE - 1
        : -negamax(child, depth - 1, -beta, -alpha, 1, context);
    if (score > bestScore) {
      bestScore = score;
      bestAction = action;
    }
    alpha = Math.max(alpha, score);
  }

  return { action: bestAction, score: bestScore };
}

function chooseWeighted(ranked, random) {
  const choices = ranked.slice(0, Math.min(3, ranked.length));
  if (choices.length <= 1) return choices[0];

  const weights = choices.length === 2 ? [0.74, 0.26] : [0.64, 0.24, 0.12];
  let value = Math.max(0, Math.min(0.999999, Number(random()) || 0));
  for (let index = 0; index < choices.length; index += 1) {
    value -= weights[index];
    if (value < 0) return choices[index];
  }
  return choices[0];
}

function easyMove(state, legalActions, context, random) {
  const player = currentPlayerOf(state);
  const candidates = getCandidateActions(state, {
    difficulty: 'easy',
    legalActions,
    context,
  });
  const ranked = candidates
    .map((action) => {
      checkDeadline(context, true);
      consumeNode(context);
      const child = nextState(state, action);
      const winner = getWinner(child);
      return {
        action,
        score:
          winner !== null && winner !== undefined && samePlayer(winner, player)
            ? WIN_SCORE - 1
            : evaluateState(child, player, 1),
      };
    })
    .sort(
      (left, right) =>
        right.score - left.score || actionKey(left.action).localeCompare(actionKey(right.action)),
    );

  const immediateWin = ranked.find((entry) => entry.score >= WIN_SCORE - 1);
  return immediateWin ?? chooseWeighted(ranked, random);
}

/**
 * Chooses a legal action synchronously.
 *
 * Options:
 * - difficulty: "easy" | "medium" | "hard" (default: "medium")
 * - timeLimitMs: wall-clock search budget
 * - random: injected RNG used by easy mode
 * - maxDepth: optional cap, mainly useful for deterministic tests
 * - nodeLimit: optional deterministic search-node budget
 */
export function chooseAIMove(state, options = {}) {
  const startedAt = now();
  const difficulty = DIFFICULTY[options.difficulty] ? options.difficulty : 'medium';
  const config = DIFFICULTY[difficulty];
  const requestedLimit = Number(options.timeLimitMs);
  const timeLimitMs = Number.isFinite(requestedLimit)
    ? Math.max(0, requestedLimit)
    : config.timeLimitMs;
  const requestedDepth = Number(options.maxDepth);
  const maxDepth = Number.isFinite(requestedDepth)
    ? Math.max(1, Math.min(12, Math.floor(requestedDepth)))
    : config.maxDepth;
  const requestedNodeLimit = Number(options.nodeLimit);
  const nodeLimit = Number.isFinite(requestedNodeLimit)
    ? Math.max(0, Math.floor(requestedNodeLimit))
    : Infinity;
  const random = typeof options.random === 'function' ? options.random : Math.random;
  const legalActions = getLegalActions(state);

  if (!Array.isArray(legalActions) || legalActions.length === 0) {
    return {
      action: null,
      score: evaluateState(state, currentPlayerOf(state)),
      depth: 0,
      nodes: 0,
      elapsedMs: now() - startedAt,
    };
  }

  // A legal fallback exists before any expensive ranking or search begins.
  let bestAction = legalActions[0];
  let bestScore = evaluateState(state, currentPlayerOf(state));
  let completedDepth = 0;
  const context = {
    difficulty,
    deadline: startedAt + timeLimitMs,
    nodeLimit,
    nodes: 0,
    table: new Map(),
  };

  try {
    // Winning pawn moves are cheap to detect and must never be lost to wall
    // pruning or an extremely small wall-clock budget. A caller-supplied node
    // budget still accounts for them.
    const player = currentPlayerOf(state);
    for (const action of legalActions) {
      if (isWallAction(action)) continue;
      consumeNode(context);
      const child = nextState(state, action);
      const winner = getWinner(child);
      if (winner !== null && winner !== undefined && samePlayer(winner, player)) {
        return {
          action,
          score: WIN_SCORE - 1,
          depth: 1,
          nodes: context.nodes,
          elapsedMs: now() - startedAt,
        };
      }
    }

    if (difficulty === 'easy') {
      const result = easyMove(state, legalActions, context, random);
      if (result) {
        bestAction = result.action;
        bestScore = result.score;
        completedDepth = 1;
      }
    } else {
      for (let depth = 1; depth <= maxDepth; depth += 1) {
        checkDeadline(context, true);
        const result = searchRoot(state, depth, context, bestAction);
        if (result.action) {
          bestAction = result.action;
          bestScore = result.score;
          completedDepth = depth;
        }
        if (Math.abs(bestScore) >= WIN_SCORE - depth) break;
      }
    }
  } catch (error) {
    if (error !== TIMEOUT) throw error;
  }

  // Candidate actions came from legalActions, but retain this final guard in case
  // an engine implementation returns fresh action objects on each call.
  if (!legalActions.some((action) => actionsEqual(action, bestAction))) {
    bestAction = legalActions[0];
  }

  return {
    action: bestAction,
    score: bestScore,
    depth: completedDepth,
    nodes: context.nodes,
    elapsedMs: now() - startedAt,
  };
}
