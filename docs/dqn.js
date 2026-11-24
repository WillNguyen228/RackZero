"use strict";

import { createCandidateGenerator } from "./agents.js";
import { PREMIUMS } from "./engine.js";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ?";
export const CELL_SIZE = 35;
export const STATE_SIZE = 225 * CELL_SIZE + 32;
export const ACTION_SIZE = 225 * CELL_SIZE + 58;
const HIDDEN = [256, 128, 64, 1];

function cellsFeatures(board) {
  const features = [];
  for (let r = 0; r < 15; r += 1) {
    for (let c = 0; c < 15; c += 1) {
      const tile = board[r][c];
      const code = tile ? LETTERS.indexOf(tile.letter) + 1 : 0;
      for (let letter = 1; letter <= 26; letter += 1) features.push(Number(code === letter));
      features.push(Number(Boolean(tile?.blank)));
      features.push(tile ? r / 14 : 0, tile ? c / 14 : 0);
      for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        features.push(Number(Boolean(board[r + dr]?.[c + dc])));
      }
      const premium = tile ? PREMIUMS[r][c] : "";
      features.push(premium === "TW" ? 1 : ["DW", "CENTER"].includes(premium) ? 0.5 : 0);
      features.push(premium === "TL" ? 1 : premium === "DL" ? 0.5 : 0);
    }
  }
  return features;
}

function rackCounts(rack) {
  const counts = Array(27).fill(0);
  for (const tile of rack) counts[LETTERS.indexOf(tile.letter)] += 1;
  return counts;
}

export function encodeState(position) {
  return [
    ...cellsFeatures(position.board), ...rackCounts(position.rack).map(n => n / 7),
    position.scores[0] / 1000, position.scores[1] / 1000,
    position.bagSize / 100, position.opponentRackSize / 7, position.scorelessTurns / 6
  ];
}

export function encodeAction(position, action) {
  const board = Array.from({ length: 15 }, () => Array(15).fill(null));
  const leave = rackCounts(position.rack);
  const exchanged = Array(27).fill(0);
  if (action.kind === "place") {
    for (const p of action.placements) {
      board[p.row][p.col] = p.tile;
      leave[LETTERS.indexOf(p.tile.blank ? "?" : p.tile.letter)] -= 1;
    }
  } else if (action.kind === "exchange") {
    for (const index of action.rackIndexes) {
      const code = LETTERS.indexOf(position.rack[index].letter);
      exchanged[code] += 1;
      leave[code] -= 1;
    }
  }
  return [
    ...cellsFeatures(board), ...leave.map(n => n / 7), ...exchanged.map(n => n / 7),
    ...["place", "exchange", "pass"].map(kind => Number(action.kind === kind)),
    (action.score || 0) / 100
  ];
}

export function validateModel(model) {
  if (model?.schema !== 3 || model.state_size !== STATE_SIZE || model.action_size !== ACTION_SIZE
    || !Number.isInteger(model.episodes) || model.episodes < 1
    || !Number.isInteger(model.updates) || model.updates < 1
    || !(model.top_candidates == null
      || (Number.isInteger(model.top_candidates) && model.top_candidates > 0))
    || !/^[a-f0-9]{64}$/.test(model.dictionary_sha256)
    || !Array.isArray(model.layers) || model.layers.length !== HIDDEN.length) {
    throw new Error("Missing or incompatible DQN checkpoint. Train and export a policy first.");
  }
  let input = STATE_SIZE + ACTION_SIZE;
  for (const [index, layer] of model.layers.entries()) {
    const output = HIDDEN[index];
    if (!Array.isArray(layer.bias) || layer.bias.length !== output
      || !layer.bias.every(Number.isFinite) || !Array.isArray(layer.weight)
      || layer.weight.length !== output || !layer.weight.every(row =>
        Array.isArray(row) && row.length === input && row.every(Number.isFinite))) {
      throw new Error("Invalid DQN network weights.");
    }
    input = output;
  }
  return model;
}

function dense(input, layer, relu) {
  const nonzero = [];
  for (let j = 0; j < input.length; j += 1) if (input[j] !== 0) nonzero.push(j);
  return layer.weight.map((row, i) => {
    let value = layer.bias[i];
    for (const j of nonzero) value += row[j] * input[j];
    return relu ? Math.max(0, value) : value;
  });
}

export function qValue(model, state, action) {
  let values = state.concat(action);
  model.layers.forEach((layer, i) => {
    values = dense(values, layer, i < model.layers.length - 1);
  });
  if (!Number.isFinite(values[0])) throw new Error("Nonfinite DQN output.");
  // The network is a residual on top of the immediate score feature (Python QNetwork).
  return values[0] + action.at(-1);
}

// Mirrors Python Candidates.top_indexes: highest scores first, then place/exchange/pass.
export function topCandidates(actions, limit) {
  if (limit == null || actions.length <= limit) return actions;
  const rank = { place: 0, exchange: 1, pass: 2 };
  const kept = actions.map((action, index) => ({ action, index }))
    .sort((a, b) => ((b.action.kind === "place" ? b.action.score : 0)
      - (a.action.kind === "place" ? a.action.score : 0))
      || rank[a.action.kind] - rank[b.action.kind] || a.index - b.index)
    .slice(0, limit);
  const pass = actions.findIndex(action => action.kind === "pass");
  if (pass >= 0 && !kept.some(item => item.index === pass)) {
    kept[kept.length - 1] = { action: actions[pass], index: pass };
  }
  return kept.sort((a, b) => a.index - b.index).map(item => item.action);
}

export function createDQNAgent(dictionary, model) {
  validateModel(model);
  const generate = createCandidateGenerator(dictionary);
  return position => {
    const state = encodeState(position);
    const actions = topCandidates(generate(position), model.top_candidates);
    let best = null, bestValue = -Infinity;
    for (const action of actions) {
      const value = qValue(model, state, encodeAction(position, action));
      if (value > bestValue) {
        best = action;
        bestValue = value;
      }
    }
    return { ...best, qValue: bestValue };
  };
}
