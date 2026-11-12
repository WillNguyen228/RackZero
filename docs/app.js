"use strict";

import {
  BOARD_SIZE,
  CENTER,
  MAX_SCORELESS_TURNS,
  PREMIUMS,
  RACK_SIZE,
  analyzeMove,
  commitMove as engineCommitMove,
  createAssignedBlank,
  createGameState,
  currentPlayer,
  currentAgent,
  exchangeTiles as engineExchangeTiles,
  normalizePlayerNames,
  passTurn as enginePassTurn,
  validateSavedState,
  winners
} from "./engine.js";
import { createGameStorage } from "./storage.js";
import { validateModel } from "./dqn.js";

const THEME_KEY = "masterchief-scrabble-theme";
const gameStorage = createGameStorage(window.localStorage);

const boardElement = document.getElementById("board");
const scoreboardElement = document.getElementById("scoreboard");
const rackElement = document.getElementById("rack");
const bagCountElement = document.getElementById("bag-count");
const scorelessElement = document.getElementById("scoreless-count");
const turnTitleElement = document.getElementById("turn-title");
const rackPlayerElement = document.getElementById("rack-player");
const modeLabelElement = document.getElementById("mode-label");
const rackHelpElement = document.getElementById("rack-help");
const statusElement = document.getElementById("status");
const historyElement = document.getElementById("history");
const dictionaryStateElement = document.getElementById("dictionary-state");
const commitButton = document.getElementById("commit-move");
const recallButton = document.getElementById("recall-tiles");
const exchangeButton = document.getElementById("exchange-tiles");
const passButton = document.getElementById("pass-turn");
const newGameButton = document.getElementById("new-game");
const saveButton = document.getElementById("save-game");
const loadButton = document.getElementById("load-game");
const themeButton = document.getElementById("theme-toggle");
const newGameDialog = document.getElementById("new-game-dialog");
const newGameForm = document.getElementById("new-game-form");
const dialogCloseButton = document.getElementById("dialog-close");
const cancelNewGameButton = document.getElementById("cancel-new-game");
const playerCountElement = document.getElementById("player-count");
const playerNameFields = [1, 2, 3, 4].map(index => document.getElementById(`player-name-${index}`));
const newGameError = document.getElementById("new-game-error");
const opponentElement = document.getElementById("opponent");
const playAgentButton = document.getElementById("play-agent");
const retryAgentButton = document.getElementById("retry-agent");
const agentsMatchButton = document.getElementById("agents-match");
const agentMatchFields = document.getElementById("agent-match-fields");
const agentSelectors = [document.getElementById("agent-one"), document.getElementById("agent-two")];
const matchControls = document.getElementById("match-controls");
const pauseMatchButton = document.getElementById("pause-match");
const stepMatchButton = document.getElementById("step-match");
const quackleStateElement = document.getElementById("quackle-state");
const dqnStateElement = document.getElementById("dqn-state");

let dictionary = new Set();
let dictionaryReady = false;
let state = null;
let pending = [];
let activeRackIndex = null;
let exchangeMode = false;
let exchangeSelection = new Set();
let agentWorker = null;
let agentRequest = 0;
let agentThinking = false;
let agentError = false;
let agentTimer = null;
let agentStep = false;
let nativeRequest = null;
let quackleReady = false;
let quackleChecked = false;
let dqnModel = null;
let dqnChecked = false;

function isAgentTurn() {
  return state && currentAgent(state) !== null && !state.gameOver;
}

function agentName(opponent = currentAgent(state)) {
  return { dp: "DP", greedy: "Greedy", quackle: "Quackle", dqn: "DQN" }[opponent] || "Agent";
}

function cancelAgent() {
  clearTimeout(agentTimer);
  agentTimer = null;
  agentStep = false;
  agentRequest += 1;
  nativeRequest?.abort();
  nativeRequest = null;
  agentWorker?.terminate();
  agentWorker = null;
  agentThinking = false;
  agentError = false;
}

function failAgent(message) {
  cancelAgent();
  agentError = true;
  setStatus(`${agentName()} could not complete its turn: ${message}. Retry or start/load another game.`, "error");
  renderAll();
}

function scheduleAgentTurn() {
  if (!isAgentTurn() || !dictionaryReady || agentThinking || agentError || agentTimer !== null) return;
  if (currentAgent(state) === "quackle") {
    if (!quackleChecked) return;
    if (!quackleReady) {
      failAgent("Native Quackle is unavailable. Open the game through python -m scrabble_gym.quackle_server");
      return;
    }
  }
  if (currentAgent(state) === "dqn") {
    if (!dqnChecked) return;
    if (!dqnModel) {
      failAgent("DQN checkpoint unavailable. Train a model and use the local Quackle server");
      return;
    }
  }
  if (state.opponent === "agents") {
    if (state.agentPaused && !agentStep) return;
    agentTimer = setTimeout(() => {
      agentTimer = null;
      startAgentTurn();
      renderAll();
    }, agentStep ? 0 : 650);
    return;
  }
  startAgentTurn();
}

function startAgentTurn() {
  agentStep = false;
  agentThinking = true;
  setStatus(`${agentName()} is thinking…`);
  const id = ++agentRequest;
  const position = { board: state.board, rack: currentPlayer(state).rack, bagSize: state.bag.length };
  if (currentAgent(state) === "quackle") {
    nativeRequest = new AbortController();
    fetch("/api/quackle/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(position),
      signal: nativeRequest.signal
    }).then(async response => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Quackle server returned HTTP ${response.status}`);
      receiveAgentResult({ ...data, id });
    }).catch(error => {
      if (id !== agentRequest) return;
      failAgent(error.message);
    });
    return;
  }
  try {
    if (!agentWorker) {
      agentWorker = new Worker(new URL("./agent-worker.js", import.meta.url), { type: "module" });
      agentWorker.onmessage = ({ data }) => receiveAgentResult(data);
      agentWorker.onerror = event => {
        event.preventDefault();
        failAgent(event.message || "Agent worker failed to load");
      };
      agentWorker.postMessage({ type: "initialize", words: [...dictionary], dqnModel });
    }
    if (currentAgent(state) === "dqn") {
      position.scores = [currentPlayer(state).score, state.players[1 - state.currentPlayer].score];
      position.opponentRackSize = state.players[1 - state.currentPlayer].rack.length;
      position.scorelessTurns = state.scorelessTurns;
    }
    // Send only public board information and the acting agent's own rack.
    agentWorker.postMessage({
      type: "move", id, opponent: currentAgent(state),
      position
    });
  } catch (error) {
    failAgent(error.message);
  }
}

function receiveAgentResult(data) {
  if (data.id !== undefined && data.id !== agentRequest) return;
  if (!isAgentTurn() || !agentThinking) return;
  if (data.error) {
    failAgent(data.error);
    return;
  }
  try {
    const action = data.action;
    const name = currentPlayer(state).name;
    const suffix = state.opponent === "agents" ? "" : " Your turn.";
    let summary;
    if (action.kind === "place") {
      const preview = analyzeMove(state, action.placements, dictionary);
      if (preview.score !== action.score) throw new Error("Agent and browser scores disagree; no move committed.");
      const result = engineCommitMove(state, action.placements, dictionary);
      summary = `${name} played ${result.words.join(", ")} for ${result.score} points.${suffix}`;
    } else if (action.kind === "exchange") {
      engineExchangeTiles(state, action.rackIndexes);
      summary = `${name} exchanged ${action.rackIndexes.length} tiles.${suffix}`;
    } else if (action.kind === "pass") {
      enginePassTurn(state);
      summary = `${name} passed.${suffix}`;
    } else {
      throw new Error("Unknown agent action.");
    }
    nativeRequest = null;
    agentThinking = false;
    clearTransientState();
    autoSave();
    if (state.gameOver) announceGameOver();
    else setStatus(summary, "success");
    renderAll();
  } catch (error) {
    failAgent(error.message);
  }
}

async function checkQuackle() {
  try {
    const response = await fetch("/api/quackle", { cache: "no-store" });
    if (response.status === 404) {
      quackleStateElement.textContent = "Quackle is local-only. Start python -m scrabble_gym.quackle_server to enable it.";
    } else {
      const data = await response.json();
      if (!response.ok || data.available !== true) throw new Error(data.error || "Native engine unavailable");
      quackleReady = true;
      quackleStateElement.textContent = "Native Quackle static policy connected (shared demo lexicon).";
    }
  } catch (error) {
    quackleStateElement.textContent = `Quackle unavailable: ${error.message}. Use the local Quackle server.`;
  }
  quackleChecked = true;
  document.querySelectorAll('option[value="quackle"]').forEach(option => { option.disabled = !quackleReady; });
  renderAll();
}

async function checkDQN() {
  dqnModel = null;
  try {
    const response = await fetch("/api/dqn/model", { cache: "no-store" });
    if (response.status === 404) {
      dqnStateElement.textContent = "DQN needs a trained checkpoint and the local Quackle server.";
    } else {
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      const wordsResponse = await fetch("words.txt", { cache: "no-store" });
      if (!wordsResponse.ok) throw new Error("Cannot verify DQN dictionary");
      const digest = await crypto.subtle.digest("SHA-256", await wordsResponse.arrayBuffer());
      const hash = [...new Uint8Array(digest)].map(n => n.toString(16).padStart(2, "0")).join("");
      if (hash !== data.dictionary_sha256) throw new Error("DQN dictionary mismatch; retrain the model");
      dqnModel = validateModel(data);
      dqnStateElement.textContent = `DQN checkpoint loaded: ${data.episodes} training episodes, ${data.updates} updates. Training amount is not a strength guarantee.`;
    }
  } catch (error) {
    dqnStateElement.textContent = `DQN unavailable: ${error.message}`;
  }
  dqnChecked = true;
  agentWorker?.postMessage({ type: "dqn-model", model: dqnModel });
  document.querySelectorAll('option[value="dqn"]').forEach(option => { option.disabled = !dqnModel; });
  renderAll();
}

function initializeState() {
  const restored = gameStorage.readAuto();
  if (restored) {
    state = restored;
    setStatus("Automatic save restored from this browser.", "success");
  } else {
    state = createGameState();
    gameStorage.writeAuto(state);
    setStatus("Player 1 starts. Build the first word through the center star.");
  }
  clearTransientState();
  renderAll();
}

function clearTransientState() {
  pending = [];
  activeRackIndex = null;
  exchangeMode = false;
  exchangeSelection = new Set();
}

function key(row, col) {
  return `${row},${col}`;
}

function pendingAt(row, col) {
  return pending.find(item => item.row === row && item.col === col) || null;
}

function premiumClass(type) {
  return {
    DL: "premium-dl",
    TL: "premium-tl",
    DW: "premium-dw",
    TW: "premium-tw",
    CENTER: "premium-center"
  }[type] || "";
}

function premiumLabel(type) {
  return { DL: "DL", TL: "TL", DW: "DW", TW: "TW", CENTER: "★" }[type] || "";
}

function tileHtml(tile, pendingTile = false) {
  const letter = tile.letter === "?" ? "_" : escapeHtml(tile.letter);
  const blankMark = tile.blank ? '<span class="blank-mark">BLANK</span>' : "";
  return `<span class="board-tile${pendingTile ? " pending" : ""}"><span>${letter}</span><span class="tile-points">${tile.points}</span>${blankMark}</span>`;
}

function renderBoard() {
  boardElement.innerHTML = "";
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.disabled = state.gameOver || isAgentTurn();
      const premium = PREMIUMS[row][col];
      const fixedTile = state.board[row][col];
      const temp = pendingAt(row, col);
      button.className = `cell ${premiumClass(premium)}`.trim();
      button.dataset.row = row;
      button.dataset.col = col;
      button.setAttribute("role", "gridcell");

      if (temp) {
        button.innerHTML = tileHtml(temp.tile, true);
        button.setAttribute("aria-label", `Pending ${temp.tile.letter} at row ${row + 1}, column ${col + 1}`);
      } else if (fixedTile) {
        button.innerHTML = tileHtml(fixedTile, false);
        button.setAttribute("aria-label", `${fixedTile.letter} at row ${row + 1}, column ${col + 1}`);
      } else {
        button.innerHTML = premium ? `<span class="premium-label">${premiumLabel(premium)}</span>` : "";
        button.setAttribute("aria-label", `Empty row ${row + 1}, column ${col + 1}`);
      }

      button.addEventListener("click", () => handleBoardClick(row, col));
      boardElement.appendChild(button);
    }
  }
}

function handleBoardClick(row, col) {
  if (state.gameOver || isAgentTurn()) return;

  const tempIndex = pending.findIndex(item => item.row === row && item.col === col);
  if (tempIndex >= 0) {
    const [removed] = pending.splice(tempIndex, 1);
    activeRackIndex = removed.rackIndex;
    setStatus(`${removed.tile.letter} returned to the rack.`);
    renderAll();
    return;
  }

  if (state.board[row][col]) {
    setStatus("That square is already occupied.", "warning");
    return;
  }
  if (exchangeMode) {
    setStatus("Finish or cancel exchange mode before placing tiles.", "warning");
    return;
  }
  if (activeRackIndex === null) {
    setStatus("Select a rack tile first.", "warning");
    return;
  }
  if (pending.some(item => item.rackIndex === activeRackIndex)) {
    setStatus("That rack tile is already pending on the board.", "warning");
    return;
  }

  const rackTile = currentPlayer(state).rack[activeRackIndex];
  if (!rackTile) {
    activeRackIndex = null;
    renderAll();
    return;
  }

  let placedTile = { ...rackTile };
  if (rackTile.blank) {
    const assigned = window.prompt("Assign the blank tile a letter from A to Z:", "A");
    if (assigned === null) return;
    try {
      placedTile = createAssignedBlank(assigned);
    } catch (error) {
      setStatus(error.message, "error");
      return;
    }
  }

  pending.push({ row, col, rackIndex: activeRackIndex, tile: placedTile });
  activeRackIndex = null;
  setStatus("Tile staged. Add more tiles or commit the move.");
  renderAll();
}

function renderRack() {
  rackElement.innerHTML = "";
  if (isAgentTurn()) return;
  const player = currentPlayer(state);
  player.rack.forEach((tile, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rack-tile";
    const used = pending.some(item => item.rackIndex === index);
    if (used) button.classList.add("used");
    if (!exchangeMode && activeRackIndex === index) button.classList.add("active");
    if (exchangeMode && exchangeSelection.has(index)) button.classList.add("exchange");
    button.disabled = used;
    const letter = tile.letter === "?" ? "_" : escapeHtml(tile.letter);
    button.innerHTML = `<span>${letter}</span><span class="tile-points">${tile.points}</span>${tile.blank ? '<span class="blank-mark">BLANK</span>' : ""}`;
    button.setAttribute("aria-label", tile.blank ? "Blank tile" : `${tile.letter}, ${tile.points} points`);

    button.addEventListener("click", () => {
      if (exchangeMode) {
        if (exchangeSelection.has(index)) exchangeSelection.delete(index);
        else exchangeSelection.add(index);
        renderRack();
        updateActionState();
      } else {
        activeRackIndex = activeRackIndex === index ? null : index;
        renderRack();
      }
    });
    rackElement.appendChild(button);
  });
}

function renderScoreboard() {
  scoreboardElement.innerHTML = "";
  state.players.forEach((player, index) => {
    const row = document.createElement("div");
    row.className = `player-score${index === state.currentPlayer && !state.gameOver ? " current" : ""}`;
    const name = document.createElement("span");
    name.textContent = player.name;
    const score = document.createElement("strong");
    score.textContent = player.score;
    row.append(name, score);
    scoreboardElement.appendChild(row);
  });
}

function renderHistory() {
  historyElement.innerHTML = "";
  if (!state.history.length) {
    const li = document.createElement("li");
    li.textContent = "No completed turns yet.";
    historyElement.appendChild(li);
    return;
  }

  state.history.slice().reverse().slice(0, 20).forEach(entry => {
    const li = document.createElement("li");
    const strong = document.createElement("strong");
    strong.textContent = entry.player;
    li.appendChild(strong);

    if (entry.type === "move") {
      li.appendChild(document.createTextNode(` ${entry.words.join(", ")} `));
      const score = document.createElement("span");
      score.className = "score";
      score.textContent = `+${entry.score}`;
      li.appendChild(score);
    } else if (entry.type === "exchange") {
      li.appendChild(document.createTextNode(` exchanged ${entry.count} tile${entry.count === 1 ? "" : "s"}.`));
    } else if (entry.type === "pass") {
      li.appendChild(document.createTextNode(" passed."));
    }
    historyElement.appendChild(li);
  });
}

function renderStats() {
  const player = currentPlayer(state);
  bagCountElement.textContent = state.bag.length;
  scorelessElement.textContent = `${state.scorelessTurns} / ${MAX_SCORELESS_TURNS}`;
  turnTitleElement.textContent = state.gameOver ? "Game over" : `${player.name}'s turn`;
  if (["greedy", "dp", "quackle", "dqn"].includes(state.opponent)) turnTitleElement.textContent += ` · vs ${agentName(state.opponent)}`;
  if (state.opponent === "agents") {
    turnTitleElement.textContent += ` · ${state.agents.map(agent => agentName(agent)).join(" vs ")}`;
  }
  rackPlayerElement.textContent = player.name;

  if (isAgentTurn()) {
    modeLabelElement.textContent = agentError || state.agentPaused ? "Agent paused" : "Agent turn";
    rackHelpElement.textContent = state.opponent === "agents"
      ? "Spectator mode: agent racks stay hidden. Pause to inspect the board or step one turn."
      : `${agentName()}'s rack is hidden. Your rack returns after its turn.`;
  } else if (exchangeMode) {
    modeLabelElement.textContent = "Exchange mode";
    rackHelpElement.textContent = "Select one or more rack tiles, then click Confirm exchange. Exchanges require at least seven tiles in the bag.";
  } else {
    modeLabelElement.textContent = "Placement mode";
    rackHelpElement.textContent = "Select a rack tile, then click an empty board square. Click a pending board tile to return it.";
  }
}

function updateActionState() {
  const locked = state.gameOver || isAgentTurn();
  commitButton.disabled = locked || !pending.length || !dictionaryReady || exchangeMode;
  recallButton.disabled = locked || (!pending.length && !exchangeMode);
  passButton.disabled = locked || pending.length > 0 || exchangeMode;
  exchangeButton.disabled = locked || pending.length > 0 || state.bag.length < RACK_SIZE;
  retryAgentButton.hidden = !agentError || !isAgentTurn();
  retryAgentButton.textContent = `Retry ${agentName()} turn`;
  retryAgentButton.disabled = !dictionaryReady;
  matchControls.hidden = state.opponent !== "agents";
  pauseMatchButton.textContent = state.agentPaused ? "Resume" : "Pause";
  pauseMatchButton.disabled = state.gameOver;
  stepMatchButton.disabled = state.gameOver || !state.agentPaused || !dictionaryReady
    || agentThinking || agentTimer !== null || agentError;
  exchangeButton.textContent = exchangeMode ? "Confirm exchange" : "Exchange tiles";
  loadButton.disabled = !gameStorage.hasManual();
}

function renderAll() {
  scheduleAgentTurn();
  renderBoard();
  renderRack();
  renderScoreboard();
  renderHistory();
  renderStats();
  updateActionState();
}

function commitMove() {
  if (state.gameOver || isAgentTurn()) return;
  if (!dictionaryReady) {
    setStatus("Dictionary is still loading.", "warning");
    return;
  }

  let result;
  try {
    result = engineCommitMove(state, pending, dictionary);
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }

  const summary = `${result.words.join(", ")} scored ${result.score} point${result.score === 1 ? "" : "s"}${result.bingo ? " including a 50-point bingo bonus" : ""}.`;
  clearTransientState();
  if (state.gameOver) announceGameOver();
  else setStatus(summary, "success");
  autoSave();
  renderAll();
}

function startOrConfirmExchange() {
  if (state.gameOver || isAgentTurn()) return;
  if (!exchangeMode) {
    exchangeMode = true;
    exchangeSelection.clear();
    activeRackIndex = null;
    setStatus("Exchange mode active. Select rack tiles, then click Confirm exchange.");
    renderAll();
    return;
  }

  if (!exchangeSelection.size) {
    setStatus("Select at least one rack tile to exchange.", "warning");
    return;
  }

  const playerName = currentPlayer(state).name;
  const count = exchangeSelection.size;
  try {
    engineExchangeTiles(state, [...exchangeSelection]);
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }

  clearTransientState();
  if (state.gameOver) announceGameOver();
  else setStatus(`${playerName} exchanged ${count} tile${count === 1 ? "" : "s"}.`);
  autoSave();
  renderAll();
}

function passTurn() {
  if (state.gameOver || isAgentTurn()) return;
  if (pending.length || exchangeMode) {
    setStatus("Recall pending tiles or leave exchange mode before passing.", "warning");
    return;
  }
  const playerName = currentPlayer(state).name;
  try {
    enginePassTurn(state);
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }
  if (state.gameOver) announceGameOver();
  else setStatus(`${playerName} passed.`);
  autoSave();
  renderAll();
}

function announceGameOver() {
  const top = winners(state);
  if (top.length === 1) {
    setStatus(`Game over. ${top[0].name} wins with ${top[0].score} points.`, "success");
  } else {
    setStatus(`Game over. Tie between ${top.map(player => player.name).join(", ")} at ${top[0].score} points.`, "success");
  }
}

function recallTiles() {
  if (state.gameOver || isAgentTurn()) return;
  if (exchangeMode) {
    exchangeMode = false;
    exchangeSelection.clear();
    setStatus("Exchange mode cancelled.");
  } else if (pending.length) {
    pending = [];
    activeRackIndex = null;
    setStatus("Pending tiles returned to the rack.");
  }
  renderAll();
}

function saveGame() {
  if (!validateSavedState(state)) {
    setStatus("The current game state could not be saved safely.", "error");
    return;
  }
  gameStorage.writeManual(state);
  setStatus("Manual save created in this browser.", "success");
  updateActionState();
}

function autoSave() {
  if (validateSavedState(state)) gameStorage.writeAuto(state);
}

function loadManualSave() {
  const loaded = gameStorage.readManual();
  if (!loaded) {
    setStatus("No valid manual save was found in this browser.", "warning");
    updateActionState();
    return;
  }
  cancelAgent();
  state = loaded;
  clearTransientState();
  autoSave();
  setStatus("Manual save loaded.", "success");
  renderAll();
}


function setStatus(message, type = "info") {
  statusElement.textContent = message;
  statusElement.className = `status${type === "info" ? "" : ` ${type}`}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadDictionary() {
  dictionaryStateElement.textContent = "Loading dictionary…";
  dictionaryStateElement.className = "dictionary-state";
  try {
    const response = await fetch("words.txt", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    dictionary = new Set(text.split(/\s+/).map(word => word.trim().toUpperCase()).filter(Boolean));
    dictionaryReady = true;
    dictionaryStateElement.textContent = `${dictionary.size.toLocaleString()} offline words loaded`;
    dictionaryStateElement.classList.add("ready");
    if (!statusElement.textContent || statusElement.textContent === "Loading…") {
      setStatus("Dictionary ready. Select a tile and start playing.");
    }
  } catch (error) {
    dictionaryReady = false;
    dictionaryStateElement.textContent = "Dictionary failed to load";
    dictionaryStateElement.classList.add("error");
    setStatus(`Dictionary could not be loaded: ${error.message}. Serve the site through GitHub Pages or another HTTP server.`, "error");
  }
  renderAll();
}

function applyTheme(theme) {
  const normalized = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = normalized;
  localStorage.setItem(THEME_KEY, normalized);
  themeButton.textContent = normalized === "dark" ? "Light mode" : "Dark mode";
  themeButton.setAttribute("aria-label", `Switch to ${normalized === "dark" ? "light" : "dark"} mode`);
}

function initializeTheme() {
  applyTheme(localStorage.getItem(THEME_KEY) || "dark");
}

function toggleTheme() {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
}

function openNewGameDialog() {
  const names = state?.players?.map(player => player.name) ?? ["Player 1", "Player 2"];
  playerCountElement.value = String(Math.min(Math.max(names.length, 2), 4));
  playerNameFields.forEach((input, index) => { input.value = names[index] || `Player ${index + 1}`; });
  opponentElement.value = state?.opponent || "human";
  agentSelectors.forEach((select, index) => {
    select.value = state?.agents?.[index] || (index === 0 ? "greedy" : "dp");
  });
  newGameError.textContent = "";
  updatePlayerNameFields();
  if (typeof newGameDialog.showModal === "function") newGameDialog.showModal();
  else newGameDialog.setAttribute("open", "");
}

function closeNewGameDialog() {
  if (typeof newGameDialog.close === "function") newGameDialog.close();
  else newGameDialog.removeAttribute("open");
}

function updatePlayerNameFields() {
  const match = opponentElement.value === "agents";
  const againstAgent = ["greedy", "dp", "quackle", "dqn", "agents"].includes(opponentElement.value);
  agentMatchFields.hidden = !match;
  agentSelectors.forEach(select => { select.disabled = !match; });
  playerCountElement.disabled = againstAgent;
  if (againstAgent) playerCountElement.value = "2";
  if (match) {
    agentSelectors.forEach((select, index) => {
      playerNameFields[index].value = `${agentName(select.value)} ${index + 1}`;
    });
  } else if (againstAgent) playerNameFields[1].value = agentName(opponentElement.value);
  const count = Number(playerCountElement.value);
  playerNameFields.forEach((input, index) => {
    const field = input.closest("label");
    const visible = index < count;
    field.hidden = !visible;
    input.disabled = !visible || match || (againstAgent && index === 1);
    input.required = visible;
  });
}

function startConfiguredGame(event) {
  event.preventDefault();
  const count = Number(playerCountElement.value);
  const names = playerNameFields.slice(0, count).map(input => input.value);
  try {
    if (!["human", "greedy", "dp", "quackle", "dqn", "agents"].includes(opponentElement.value)) {
      throw new Error("That opponent is not implemented yet.");
    }
    const normalized = normalizePlayerNames(names);
    const nextState = createGameState(normalized);
    nextState.opponent = opponentElement.value;
    if (nextState.opponent === "agents") {
      const agents = agentSelectors.map(select => select.value);
      if (!agents.every(agent => ["greedy", "dp", "quackle", "dqn"].includes(agent))) {
        throw new Error("Choose an implemented agent for both players.");
      }
      nextState.agents = agents;
      nextState.agentPaused = false;
    }
    if ((nextState.opponent === "quackle" || nextState.agents?.includes("quackle")) && !quackleReady) {
      throw new Error("Quackle requires the local native-engine server.");
    }
    if ((nextState.opponent === "dqn" || nextState.agents?.includes("dqn")) && !dqnModel) {
      throw new Error("DQN requires a compatible trained checkpoint from the local server.");
    }
    cancelAgent();
    state = nextState;
  } catch (error) {
    newGameError.textContent = error.message;
    return;
  }

  clearTransientState();
  gameStorage.writeAuto(state);
  closeNewGameDialog();
  setStatus(state.opponent === "agents"
    ? "Agent match started. Pause to inspect the board or step through turns."
    : `${state.players[0].name} starts. Build the first word through the center star.`, "success");
  renderAll();
}

commitButton.addEventListener("click", commitMove);
recallButton.addEventListener("click", recallTiles);
exchangeButton.addEventListener("click", startOrConfirmExchange);
passButton.addEventListener("click", passTurn);
newGameButton.addEventListener("click", openNewGameDialog);
playAgentButton.addEventListener("click", () => {
  openNewGameDialog();
  opponentElement.value = "greedy";
  updatePlayerNameFields();
});
opponentElement.addEventListener("change", updatePlayerNameFields);
agentsMatchButton.addEventListener("click", () => {
  openNewGameDialog();
  opponentElement.value = "agents";
  updatePlayerNameFields();
});
agentSelectors.forEach(select => select.addEventListener("change", updatePlayerNameFields));
pauseMatchButton.addEventListener("click", () => {
  if (state.opponent !== "agents" || state.gameOver) return;
  state.agentPaused = !state.agentPaused;
  if (state.agentPaused) cancelAgent();
  setStatus(state.agentPaused ? "Match paused. Step one turn or resume playback." : "Match resumed.");
  autoSave();
  renderAll();
});
stepMatchButton.addEventListener("click", () => {
  if (state.opponent !== "agents" || !state.agentPaused || state.gameOver
    || agentThinking || agentTimer !== null || agentError) return;
  agentStep = true;
  renderAll();
});
retryAgentButton.addEventListener("click", () => {
  cancelAgent();
  if (state.opponent === "agents" && state.agentPaused) agentStep = true;
  if (currentAgent(state) === "quackle") {
    quackleChecked = false;
    quackleReady = false;
    checkQuackle();
  } else if (currentAgent(state) === "dqn") {
    dqnChecked = false;
    checkDQN();
  } else renderAll();
});
saveButton.addEventListener("click", saveGame);
loadButton.addEventListener("click", loadManualSave);
themeButton.addEventListener("click", toggleTheme);
playerCountElement.addEventListener("change", updatePlayerNameFields);
newGameForm.addEventListener("submit", startConfiguredGame);
dialogCloseButton.addEventListener("click", closeNewGameDialog);
cancelNewGameButton.addEventListener("click", closeNewGameDialog);
newGameDialog.addEventListener("cancel", event => {
  event.preventDefault();
  closeNewGameDialog();
});

initializeTheme();
initializeState();
loadDictionary();
checkQuackle();
checkDQN();
