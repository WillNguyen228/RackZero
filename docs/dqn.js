"use strict";

import { createCandidateGenerator } from "./agents.js";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ?";
export const STATE_SIZE = 1382;
export const ACTION_SIZE = 1408;

function cellsFeatures(board) {
  const features = [];
  for (const row of board) {
    for (const tile of row) {
      const code = tile ? LETTERS.indexOf(tile.letter) + 1 : 0;
      for (let bit = 0; bit < 5; bit += 1) features.push((code >> bit) & 1);
      features.push(Number(Boolean(tile?.blank)));
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
  if (model?.schema !== 1 || model.state_size !== STATE_SIZE || model.action_size !== ACTION_SIZE
    || !Number.isInteger(model.episodes) || model.episodes < 1
    || !Number.isInteger(model.updates) || model.updates < 1
    || !/^[a-f0-9]{64}$/.test(model.dictionary_sha256)
    || !Array.isArray(model.layers) || model.layers.length !== 3) {
    throw new Error("Missing or incompatible DQN checkpoint. Train and export a policy first.");
  }
  let input = STATE_SIZE + ACTION_SIZE;
  for (const [index, layer] of model.layers.entries()) {
    const output = [64, 32, 1][index];
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
  return layer.weight.map((row, i) => {
    let value = layer.bias[i];
    for (let j = 0; j < input.length; j += 1) {
      if (input[j] !== 0) value += row[j] * input[j];
    }
    return relu ? Math.max(0, value) : value;
  });
}

export function qValue(model, state, action) {
  let values = state.concat(action);
  model.layers.forEach((layer, i) => { values = dense(values, layer, i < 2); });
  if (!Number.isFinite(values[0])) throw new Error("Nonfinite DQN output.");
  return values[0];
}

export function createDQNAgent(dictionary, model) {
  validateModel(model);
  const generate = createCandidateGenerator(dictionary);
  return position => {
    const state = encodeState(position);
    const actions = generate(position);
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
