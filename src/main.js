import {
  applyAction,
  createInitialState,
  getLegalPawnMoves,
  getWinner,
  isLegalWall,
} from "./game.js";
import { isAbortError } from "./agents/agent.js";
import { ClassicWorkerAgent } from "./agents/worker-agent.js";

const BOARD_SIZE = 9;
const WALL_GRID_SIZE = BOARD_SIZE - 1;
const CELL_PERCENT = 9.333333;
const STEP_PERCENT = 11.333333;
const PLAYER_COLORS = ["红方", "蓝方"];
const COLUMN_LABELS = "ABCDEFGHI";

const elements = {
  board: document.querySelector("#board"),
  boardGrid: document.querySelector("#board-grid"),
  wallSlots: document.querySelector("#wall-slots"),
  placedWalls: document.querySelector("#placed-walls"),
  goalLabelTop: document.querySelector("#goal-label-top"),
  goalLabelBottom: document.querySelector("#goal-label-bottom"),
  boardFeedback: document.querySelector("#board-feedback"),
  gameHeading: document.querySelector("#game-heading"),
  matchLabel: document.querySelector("#match-label"),
  thinkingIndicator: document.querySelector("#thinking-indicator"),
  thinkingLabel: document.querySelector("#thinking-label"),
  modeBadge: document.querySelector("#mode-badge"),
  playerRows: [
    document.querySelector("#player-1-row"),
    document.querySelector("#player-2-row"),
  ],
  playerNames: [
    document.querySelector("#player-1-name"),
    document.querySelector("#player-2-name"),
  ],
  playerDirections: [
    document.querySelector("#player-1-direction"),
    document.querySelector("#player-2-direction"),
  ],
  wallCounts: [
    document.querySelector("#player-1-walls"),
    document.querySelector("#player-2-walls"),
  ],
  moveModeButton: document.querySelector("#move-mode-button"),
  wallModeButton: document.querySelector("#wall-mode-button"),
  wallOptions: document.querySelector("#wall-options"),
  orientationButtons: [...document.querySelectorAll("[data-orientation]")],
  moveCount: document.querySelector("#move-count"),
  lastAction: document.querySelector("#last-action"),
  aiMetrics: document.querySelector("#ai-metrics"),
  aiMetricsValue: document.querySelector("#ai-metrics-value"),
  undoButton: document.querySelector("#undo-button"),
  restartButton: document.querySelector("#restart-button"),
  newGameButton: document.querySelector("#new-game-button"),
  setupDialog: document.querySelector("#setup-dialog"),
  setupForm: document.querySelector("#setup-form"),
  aiSetup: document.querySelector("#ai-setup"),
  humanSideSetup: document.querySelector("#human-side-setup"),
  cancelSetupButton: document.querySelector("#cancel-setup-button"),
  gameOverDialog: document.querySelector("#game-over-dialog"),
  winnerTitle: document.querySelector("#winner-title"),
  winnerSummary: document.querySelector("#winner-summary"),
  winnerToken: document.querySelector("#winner-token"),
  reviewButton: document.querySelector("#review-button"),
  playAgainButton: document.querySelector("#play-again-button"),
  toast: document.querySelector("#toast"),
};

let state = createInitialState();
let history = [{ state, action: null, actor: null }];
let actionMode = "move";
let wallOrientation = "h";
let isThinking = false;
let gameStarted = false;
let sessionVersion = 0;
let toastTimer = null;
let pendingAIFrame = null;
let activeAIController = null;
let aiAgent = null;

const settings = {
  mode: "human",
  humanPlayer: 0,
  difficulty: "medium",
  bottomPlayer: 0,
};

function positionKey(row, col) {
  return `${row},${col}`;
}

function parseWallKey(key) {
  const [row, col] = key.split(",").map(Number);
  return { row, col };
}

function playerName(player) {
  if (settings.mode === "human") return PLAYER_COLORS[player];
  if (settings.mode === "ai-vs-ai") return `AI · ${PLAYER_COLORS[player]}`;
  return player === settings.humanPlayer
    ? `你 · ${PLAYER_COLORS[player]}`
    : `AI · ${PLAYER_COLORS[player]}`;
}

function isHumanTurn() {
  if (settings.mode === "ai-vs-ai") return false;
  return settings.mode === "human" || state.currentPlayer === settings.humanPlayer;
}

function canAcceptInput() {
  return gameStarted && !isThinking && getWinner(state) === null && isHumanTurn();
}

function formatCoordinate(row, col) {
  return `${COLUMN_LABELS[col]}${row + 1}`;
}

function formatAction(action, actor) {
  if (!action) return "等待开局";
  const who = playerName(actor);
  if (action.type === "move") {
    return `${who}移动至 ${formatCoordinate(action.row, action.col)}`;
  }
  const direction = action.orientation === "h" ? "横墙" : "纵墙";
  return `${who}放置${direction} ${formatCoordinate(action.row, action.col)}`;
}

function setFeedback(message, isError = false) {
  elements.boardFeedback.textContent = message;
  elements.boardFeedback.classList.toggle("is-error", isError);
}

function terminationMessage(winner, { summary = false } = {}) {
  const loser = 1 - winner;
  if (state.terminationReason === "repetition") {
    return summary
      ? `${state.moveNumber} 步后，${playerName(loser)}触发三次重复判负`
      : `${playerName(winner)}获胜，${playerName(loser)}触发三次重复`;
  }
  if (state.terminationReason === "move-limit") {
    return summary
      ? `${state.moveNumber} 步后，${playerName(loser)}触发 256 个半回合上限判负`
      : `${playerName(winner)}获胜，${playerName(loser)}触发步数上限`;
  }
  return summary
    ? `${state.moveNumber} 步后率先抵达对边`
    : `${playerName(winner)}已抵达对边`;
}

function defaultFeedback() {
  const winner = getWinner(state);
  if (winner !== null) return terminationMessage(winner);
  if (isThinking) return `${playerName(state.currentPlayer)}正在评估路径与墙位`;
  if (!isHumanTurn()) return `等待${playerName(state.currentPlayer)}行动`;
  if (actionMode === "wall") {
    if (state.wallsRemaining[state.currentPlayer] <= 0) return "当前玩家已没有可用墙壁";
    return wallOrientation === "h" ? "选择横墙位置" : "选择纵墙位置";
  }
  return `${playerName(state.currentPlayer)}请选择移动位置`;
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 3600);
}

function setElementPosition(element, orientation, row, col) {
  if (orientation === "h") {
    element.style.left = `${col * STEP_PERCENT}%`;
    element.style.top = `${CELL_PERCENT + row * STEP_PERCENT}%`;
  } else {
    element.style.left = `${CELL_PERCENT + col * STEP_PERCENT}%`;
    element.style.top = `${row * STEP_PERCENT}%`;
  }
}

function createBoardCells() {
  const fragment = document.createDocumentFragment();
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.type = "button";
      cell.dataset.cell = "";
      cell.dataset.row = String(row);
      cell.dataset.col = String(col);
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-rowindex", String(row + 1));
      cell.setAttribute("aria-colindex", String(col + 1));
      cell.tabIndex = -1;
      fragment.append(cell);
    }
  }
  elements.boardGrid.replaceChildren(fragment);
}

function renderBoardCells(legalMoves) {
  const legalKeys = new Set(legalMoves.map(({ row, col }) => positionKey(row, col)));
  const inputEnabled = canAcceptInput() && actionMode === "move";
  const currentPawn = state.pawns[state.currentPlayer];
  let tabStopAssigned = false;

  for (const cell of elements.boardGrid.children) {
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    const key = positionKey(row, col);
    const occupant = state.pawns.findIndex(
      (pawn) => pawn.row === row && pawn.col === col,
    );
    const isLegal = inputEnabled && legalKeys.has(key);

    cell.replaceChildren();
    cell.classList.toggle("is-legal", isLegal);
    cell.classList.toggle(
      "is-current",
      occupant === state.currentPlayer && getWinner(state) === null,
    );
    cell.setAttribute("aria-disabled", String(!isLegal));

    let label = `${formatCoordinate(row, col)}，第 ${row + 1} 行第 ${col + 1} 列`;
    if (occupant >= 0) {
      const pawn = document.createElement("span");
      pawn.className = `pawn pawn-${occupant === 0 ? "red" : "blue"}`;
      pawn.setAttribute("aria-hidden", "true");
      cell.append(pawn);
      label += `，${playerName(occupant)}棋子`;
    }
    if (isLegal) label += "，可移动至此";
    cell.setAttribute("aria-label", label);

    const preferredTabStop =
      (inputEnabled && isLegal && !tabStopAssigned) ||
      (!inputEnabled && row === currentPawn.row && col === currentPawn.col);
    cell.tabIndex = preferredTabStop ? 0 : -1;
    if (preferredTabStop) tabStopAssigned = true;
  }
}

function renderPlacedWalls() {
  const fragment = document.createDocumentFragment();
  for (const [orientation, walls] of [
    ["h", state.horizontalWalls],
    ["v", state.verticalWalls],
  ]) {
    for (const key of walls) {
      const { row, col } = parseWallKey(key);
      const wall = document.createElement("span");
      wall.className = `wall is-${orientation === "h" ? "horizontal" : "vertical"}`;
      setElementPosition(wall, orientation, row, col);
      fragment.append(wall);
    }
  }
  elements.placedWalls.replaceChildren(fragment);
}

function renderWallSlots() {
  const fragment = document.createDocumentFragment();
  const inputEnabled = canAcceptInput() && actionMode === "wall";

  if (!inputEnabled) {
    elements.wallSlots.replaceChildren();
    return;
  }

  for (let row = 0; row < WALL_GRID_SIZE; row += 1) {
    for (let col = 0; col < WALL_GRID_SIZE; col += 1) {
      const legal = isLegalWall(state, wallOrientation, row, col);
      const slot = document.createElement("button");
      const direction = wallOrientation === "h" ? "横墙" : "纵墙";
      slot.className = `wall-slot is-${wallOrientation === "h" ? "horizontal" : "vertical"} ${legal ? "is-legal" : "is-invalid"}`;
      slot.type = "button";
      slot.dataset.wallSlot = "";
      slot.dataset.orientation = wallOrientation;
      slot.dataset.row = String(row);
      slot.dataset.col = String(col);
      slot.dataset.legal = String(legal);
      slot.tabIndex = legal ? 0 : -1;
      slot.setAttribute("aria-disabled", String(!legal));
      slot.setAttribute(
        "aria-label",
        `${formatCoordinate(row, col)} 放置${direction}${legal ? "" : "，此处不可放置"}`,
      );
      const hitArea = document.createElement("span");
      hitArea.className = "wall-slot-hit";
      hitArea.setAttribute("aria-hidden", "true");
      slot.append(hitArea);
      setElementPosition(slot, wallOrientation, row, col);
      fragment.append(slot);
    }
  }
  elements.wallSlots.replaceChildren(fragment);
}

function updateLastAction() {
  const entry = history.at(-1);
  const mark = elements.lastAction.querySelector(".last-action-mark");
  const label = elements.lastAction.querySelector("strong");
  mark.textContent = entry.action?.type === "move" ? "●" : entry.action ? "▰" : "·";
  mark.classList.toggle("is-red", entry.actor === 0);
  mark.classList.toggle("is-blue", entry.actor === 1);
  label.textContent = formatAction(entry.action, entry.actor);
}

function hasUndoableHumanRound() {
  if (history.length <= 1) return false;
  if (settings.mode === "human" || settings.mode === "ai-vs-ai") return true;
  return history.some(
    (entry, index) => index > 0 && entry.actor === settings.humanPlayer,
  );
}

function renderControls() {
  const winner = getWinner(state);
  const current = state.currentPlayer;
  const inputEnabled = canAcceptInput();
  const difficultyLabels = { easy: "简单", medium: "标准", hard: "困难" };

  elements.playerNames.forEach((element, player) => {
    element.textContent = playerName(player);
    elements.wallCounts[player].textContent = String(state.wallsRemaining[player]);
    elements.playerRows[player].classList.toggle(
      "is-current",
      winner === null && current === player,
    );
  });

  elements.gameHeading.textContent =
    winner === null ? `${playerName(current)}行动` : `${playerName(winner)}获胜`;
  if (settings.mode === "human") {
    elements.matchLabel.textContent = `本地双人 · 第 ${state.moveNumber + 1} 回合`;
    elements.modeBadge.textContent = "双人";
  } else if (settings.mode === "ai-vs-ai") {
    elements.matchLabel.textContent = `AI 对弈 · ${difficultyLabels[settings.difficulty]} · 第 ${state.moveNumber + 1} 回合`;
    elements.modeBadge.textContent = "AI 对弈";
  } else {
    elements.matchLabel.textContent = `人机对战 · ${difficultyLabels[settings.difficulty]} · 第 ${state.moveNumber + 1} 回合`;
    elements.modeBadge.textContent = difficultyLabels[settings.difficulty];
  }
  elements.moveCount.textContent = `${state.moveNumber} 步`;
  elements.thinkingIndicator.hidden = !isThinking;
  elements.thinkingLabel.textContent = `${playerName(current)}思考中`;

  elements.moveModeButton.disabled = !inputEnabled;
  elements.wallModeButton.disabled =
    !inputEnabled || state.wallsRemaining[current] <= 0;
  elements.moveModeButton.classList.toggle("is-active", actionMode === "move");
  elements.wallModeButton.classList.toggle("is-active", actionMode === "wall");
  elements.moveModeButton.setAttribute("aria-selected", String(actionMode === "move"));
  elements.wallModeButton.setAttribute("aria-selected", String(actionMode === "wall"));
  elements.wallOptions.hidden = actionMode !== "wall" || !inputEnabled;

  for (const button of elements.orientationButtons) {
    const active = button.dataset.orientation === wallOrientation;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = !inputEnabled;
  }

  elements.undoButton.disabled = !hasUndoableHumanRound();
  elements.restartButton.disabled = !gameStarted;
  elements.newGameButton.disabled = false;
  updateLastAction();
}

function renderPerspective() {
  const isRotated = settings.bottomPlayer === 1;
  elements.board.classList.toggle("is-rotated", isRotated);
  elements.board.dataset.bottomPlayer = String(settings.bottomPlayer);
  elements.goalLabelTop.textContent = isRotated ? "蓝方目标" : "红方目标";
  elements.goalLabelBottom.textContent = isRotated ? "红方目标" : "蓝方目标";
  elements.playerDirections[0].textContent = isRotated
    ? "向下抵达对边"
    : "向上抵达对边";
  elements.playerDirections[1].textContent = isRotated
    ? "向上抵达对边"
    : "向下抵达对边";
  elements.board.setAttribute(
    "aria-label",
    isRotated
      ? "九乘九步步为营棋盘，蓝方在下，红方在上"
      : "九乘九步步为营棋盘，红方在下，蓝方在上",
  );
}

function render() {
  renderPerspective();
  const legalMoves =
    canAcceptInput() && actionMode === "move" ? getLegalPawnMoves(state) : [];
  renderBoardCells(legalMoves);
  renderPlacedWalls();
  renderWallSlots();
  renderControls();
  setFeedback(defaultFeedback());
}

function showGameOver(winner) {
  const completedVersion = sessionVersion;
  elements.winnerTitle.textContent = `${playerName(winner)}获胜`;
  const summary = terminationMessage(winner, { summary: true });
  elements.winnerSummary.textContent = summary;
  elements.winnerToken.classList.toggle("is-blue", winner === 1);
  window.setTimeout(() => {
    if (
      completedVersion === sessionVersion &&
      !elements.gameOverDialog.open &&
      getWinner(state) !== null
    ) {
      elements.gameOverDialog.showModal();
    }
  }, 180);
}

function commitAction(action, source = "human") {
  const actor = state.currentPlayer;
  try {
    state = applyAction(state, action);
    history.push({ state, action, actor });
    actionMode = "move";
    render();

    const winner = getWinner(state);
    if (winner !== null) {
      showGameOver(winner);
      return;
    }

    scheduleAITurn();
  } catch (error) {
    console.error("无法执行游戏动作", error);
    setFeedback("该位置当前不可用", true);
    showToast("操作未完成，请选择其他位置");
    if (source === "ai") scheduleAITurn();
  }
}

function handleCellClick(cell) {
  const row = Number(cell.dataset.row);
  const col = Number(cell.dataset.col);
  if (!canAcceptInput()) {
    setFeedback(isThinking ? "请等待 AI 完成行动" : defaultFeedback(), true);
    return;
  }
  if (actionMode !== "move") {
    setFeedback("当前为放墙模式", true);
    return;
  }
  const legal = getLegalPawnMoves(state).some(
    (move) => move.row === row && move.col === col,
  );
  if (!legal) {
    setFeedback("棋子不能移动到这里", true);
    return;
  }
  commitAction({ type: "move", row, col });
}

function handleWallClick(slot) {
  if (!canAcceptInput() || actionMode !== "wall") return;
  const orientation = slot.dataset.orientation;
  const row = Number(slot.dataset.row);
  const col = Number(slot.dataset.col);
  if (!isLegalWall(state, orientation, row, col)) {
    setFeedback("墙壁不能重叠、交叉或封死任一玩家", true);
    return;
  }
  commitAction({ type: "wall", orientation, row, col });
}

function switchActionMode(mode) {
  if (!canAcceptInput()) return;
  if (mode === "wall" && state.wallsRemaining[state.currentPlayer] <= 0) {
    setFeedback("当前玩家已没有可用墙壁", true);
    return;
  }
  actionMode = mode;
  render();
}

function getAIAgent() {
  aiAgent ??= new ClassicWorkerAgent();
  return aiAgent;
}

function cancelAITurn(reason = "AI turn cancelled") {
  if (pendingAIFrame !== null) {
    window.cancelAnimationFrame(pendingAIFrame);
    pendingAIFrame = null;
  }
  if (activeAIController !== null) {
    const controller = activeAIController;
    activeAIController = null;
    controller.abort(reason);
  }
  isThinking = false;
}

function shouldAITakeTurn() {
  if (
    !gameStarted ||
    elements.setupDialog.open ||
    getWinner(state) !== null ||
    settings.mode === "human"
  ) {
    return false;
  }
  return settings.mode === "ai-vs-ai" || !isHumanTurn();
}

function beginAITurn(scheduledVersion) {
  if (
    scheduledVersion !== sessionVersion ||
    isThinking ||
    !shouldAITakeTurn()
  ) {
    return;
  }

  isThinking = true;
  actionMode = "move";
  render();

  // Start the worker after the thinking state has painted.
  pendingAIFrame = window.requestAnimationFrame(() => {
    pendingAIFrame = null;
    void runAITurn(scheduledVersion);
  });
}

function scheduleAITurn() {
  if (!shouldAITakeTurn() || isThinking || pendingAIFrame !== null) return;
  beginAITurn(sessionVersion);
}

async function runAITurn(scheduledVersion) {
  if (scheduledVersion !== sessionVersion || !isThinking) return;

  const controller = new AbortController();
  activeAIController = controller;
  let result;
  try {
    result = await getAIAgent().choose(
      state,
      { difficulty: settings.difficulty },
      controller.signal,
    );
  } catch (error) {
    if (
      isAbortError(error) ||
      controller.signal.aborted ||
      scheduledVersion !== sessionVersion
    ) {
      return;
    }
    console.error("AI 搜索失败", error);
    const fallback = getLegalPawnMoves(state)[0];
    if (!fallback) {
      isThinking = false;
      render();
      showToast("AI 无法完成行动，请重新开始本局");
      return;
    }
    result = {
      action: { type: "move", row: fallback.row, col: fallback.col },
      score: 0,
      depth: 0,
      nodes: 0,
      elapsedMs: 0,
    };
    showToast("AI 搜索异常，已使用备用移动");
  } finally {
    if (activeAIController === controller) activeAIController = null;
  }

  if (
    scheduledVersion !== sessionVersion ||
    controller.signal.aborted ||
    !isThinking
  ) {
    return;
  }
  isThinking = false;
  elements.aiMetrics.hidden = false;
  elements.aiMetricsValue.textContent = `深度 ${result.depth} · ${Number(result.nodes).toLocaleString("zh-CN")} 节点 · ${Math.round(result.elapsedMs)}ms`;

  if (!result.action) {
    render();
    showToast("AI 没有找到可执行动作");
    return;
  }
  commitAction(result.action, "ai");
}

function undo() {
  if (!hasUndoableHumanRound()) return;
  cancelAITurn("Undo requested");
  sessionVersion += 1;

  if (settings.mode === "human" || settings.mode === "ai-vs-ai") {
    history.pop();
  } else {
    while (history.length > 1) {
      const removed = history.pop();
      if (removed.actor === settings.humanPlayer) break;
    }
  }

  state = history.at(-1).state;
  elements.aiMetrics.hidden = true;
  actionMode = "move";
  if (elements.gameOverDialog.open) elements.gameOverDialog.close();
  render();
  scheduleAITurn();
}

function restartGame() {
  if (!gameStarted) return;
  cancelAITurn("Game restarted");
  sessionVersion += 1;
  if (elements.gameOverDialog.open) elements.gameOverDialog.close();
  state = createInitialState(0);
  history = [{ state, action: null, actor: null }];
  elements.aiMetrics.hidden = true;
  actionMode = "move";
  isThinking = false;
  render();
  scheduleAITurn();
}

function openSetup() {
  cancelAITurn("Game setup opened");
  if (gameStarted) sessionVersion += 1;
  elements.cancelSetupButton.hidden = !gameStarted;
  const modeInput = elements.setupForm.querySelector(
    `input[name="game-mode"][value="${settings.mode}"]`,
  );
  if (modeInput) modeInput.checked = true;
  const humanSideInput = elements.setupForm.querySelector(
    `input[name="human-side"][value="${settings.humanPlayer}"]`,
  );
  if (humanSideInput) humanSideInput.checked = true;
  const difficultyInput = elements.setupForm.querySelector(
    `input[name="difficulty"][value="${settings.difficulty}"]`,
  );
  if (difficultyInput) difficultyInput.checked = true;
  const bottomPlayerInput = elements.setupForm.querySelector(
    `input[name="bottom-player"][value="${settings.bottomPlayer}"]`,
  );
  if (bottomPlayerInput) bottomPlayerInput.checked = true;
  updateSetupVisibility(settings.mode);
  if (!elements.setupDialog.open) elements.setupDialog.showModal();
}

function startGameFromSetup() {
  const formData = new FormData(elements.setupForm);
  const requestedMode = String(formData.get("game-mode") ?? "human");
  const requestedHumanPlayer = Number(formData.get("human-side") ?? 0);
  const requestedDifficulty = String(formData.get("difficulty") ?? "medium");
  const requestedBottomPlayer = Number(formData.get("bottom-player") ?? 0);
  settings.mode = ["human", "ai", "ai-vs-ai"].includes(requestedMode)
    ? requestedMode
    : "human";
  settings.humanPlayer = requestedHumanPlayer === 1 ? 1 : 0;
  settings.difficulty = ["easy", "medium", "hard"].includes(requestedDifficulty)
    ? requestedDifficulty
    : "medium";
  settings.bottomPlayer = requestedBottomPlayer === 1 ? 1 : 0;

  cancelAITurn("New game started");
  sessionVersion += 1;
  state = createInitialState(0);
  history = [{ state, action: null, actor: null }];
  elements.aiMetrics.hidden = true;
  actionMode = "move";
  wallOrientation = "h";
  isThinking = false;
  gameStarted = true;
  elements.setupDialog.close();
  if (elements.gameOverDialog.open) elements.gameOverDialog.close();
  render();
  scheduleAITurn();
}

function updateSetupVisibility(mode) {
  const usesAI = mode === "ai" || mode === "ai-vs-ai";
  elements.aiSetup.hidden = !usesAI;
  elements.humanSideSetup.hidden = mode !== "ai";
}

function closeSetupAndResume() {
  if (!gameStarted) return;
  elements.setupDialog.close();
  render();
  scheduleAITurn();
}

function handleBoardKeydown(event) {
  const cell = event.target.closest("[data-cell]");
  if (!cell) return;
  const offsets = {
    ArrowUp: [-1, 0],
    ArrowDown: [1, 0],
    ArrowLeft: [0, -1],
    ArrowRight: [0, 1],
  };
  const offset = offsets[event.key];
  if (!offset) return;

  event.preventDefault();
  const visualDirection = settings.bottomPlayer === 1 ? -1 : 1;
  const row = Math.max(
    0,
    Math.min(
      BOARD_SIZE - 1,
      Number(cell.dataset.row) + offset[0] * visualDirection,
    ),
  );
  const col = Math.max(
    0,
    Math.min(
      BOARD_SIZE - 1,
      Number(cell.dataset.col) + offset[1] * visualDirection,
    ),
  );
  const target = elements.boardGrid.querySelector(
    `[data-cell][data-row="${row}"][data-col="${col}"]`,
  );
  if (target) {
    cell.tabIndex = -1;
    target.tabIndex = 0;
    target.focus();
  }
}

elements.board.addEventListener("click", (event) => {
  const cell = event.target.closest("[data-cell]");
  if (cell) {
    handleCellClick(cell);
    return;
  }
  const wallSlot = event.target.closest("[data-wall-slot]");
  if (wallSlot) handleWallClick(wallSlot);
});

elements.board.addEventListener("keydown", handleBoardKeydown);

elements.wallSlots.addEventListener("pointerover", (event) => {
  const slot = event.target.closest("[data-wall-slot]");
  if (!slot) return;
  const direction = slot.dataset.orientation === "h" ? "横墙" : "纵墙";
  const coordinate = formatCoordinate(Number(slot.dataset.row), Number(slot.dataset.col));
  setFeedback(
    slot.dataset.legal === "true"
      ? `可在 ${coordinate} 放置${direction}`
      : "此处墙壁会重叠、交叉或阻断全部路径",
    slot.dataset.legal !== "true",
  );
});

elements.wallSlots.addEventListener("pointerout", (event) => {
  if (event.relatedTarget && elements.wallSlots.contains(event.relatedTarget)) return;
  setFeedback(defaultFeedback());
});

elements.moveModeButton.addEventListener("click", () => switchActionMode("move"));
elements.wallModeButton.addEventListener("click", () => switchActionMode("wall"));

for (const button of elements.orientationButtons) {
  button.addEventListener("click", () => {
    wallOrientation = button.dataset.orientation;
    render();
  });
}

elements.undoButton.addEventListener("click", undo);
elements.restartButton.addEventListener("click", restartGame);
elements.newGameButton.addEventListener("click", openSetup);

elements.setupForm.addEventListener("change", (event) => {
  if (event.target.name === "game-mode") {
    updateSetupVisibility(event.target.value);
  }
});

elements.setupForm.addEventListener("submit", (event) => {
  event.preventDefault();
  startGameFromSetup();
});

elements.cancelSetupButton.addEventListener("click", () => {
  closeSetupAndResume();
});

elements.setupDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  closeSetupAndResume();
});

elements.reviewButton.addEventListener("click", () => elements.gameOverDialog.close());
elements.playAgainButton.addEventListener("click", () => {
  elements.gameOverDialog.close();
  restartGame();
});

window.addEventListener("beforeunload", () => {
  cancelAITurn("Page closed");
  aiAgent?.dispose();
});

createBoardCells();
render();
openSetup();

// Expose a read-only smoke-test surface without coupling the UI to a test runner.
window.quoridorUI = Object.freeze({
  getState: () => state,
  getSettings: () => ({ ...settings }),
  getRuntime: () => ({
    agentEngine: aiAgent?.engine ?? "classic-negamax-worker",
    hasPendingAITurn: pendingAIFrame !== null,
    hasActiveAIRequest: activeAIController !== null,
    isThinking,
    sessionVersion,
  }),
});
