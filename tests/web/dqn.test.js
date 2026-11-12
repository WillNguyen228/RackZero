import test from "node:test";
import assert from "node:assert/strict";

import { createCandidateGenerator } from "../../docs/agents.js";
import {
  ACTION_SIZE, STATE_SIZE, createDQNAgent, encodeAction, encodeState, validateModel
} from "../../docs/dqn.js";
import { analyzeMove, createAssignedBlank, createGameState, createTile } from "../../docs/engine.js";

function model() {
  const sizes = [STATE_SIZE + ACTION_SIZE, 64, 32, 1];
  return {
    schema: 1, dictionary_sha256: "a".repeat(64), episodes: 2, updates: 3,
    state_size: STATE_SIZE, action_size: ACTION_SIZE,
    layers: sizes.slice(1).map((output, i) => ({
      weight: Array.from({ length: output }, () => Array(sizes[i]).fill(0)),
      bias: Array(output).fill(0)
    }))
  };
}

function position(rack = "AT?") {
  return {
    board: createGameState().board, rack: [...rack].map(createTile), bagSize: 86,
    scores: [10, 20], opponentRackSize: 7, scorelessTurns: 0
  };
}

test("DQN candidates include every exchange multiset and pass, even with placements", () => {
  const generate = createCandidateGenerator(new Set(["AT", "CAT"]));
  const state = position("AAT");
  const before = JSON.stringify(state);
  const actions = generate(state);
  assert.ok(actions.some(a => a.kind === "place"));
  const exchanges = actions.filter(a => a.kind === "exchange");
  assert.equal(exchanges.length, 5);
  assert.equal(new Set(exchanges.map(a => a.rackIndexes.map(i => state.rack[i].letter).sort().join(""))).size, 5);
  assert.equal(actions.filter(a => a.kind === "pass").length, 1);
  assert.equal(JSON.stringify(state), before);
  assert.equal(generate({ ...state, bagSize: 6 }).filter(a => a.kind === "exchange").length, 0);
});

test("DQN feature schema preserves physical blanks, rack leave and public perspective", () => {
  const state = position();
  state.board[7][7] = createAssignedBlank("Z");
  const s = encodeState(state);
  assert.equal(s.length, STATE_SIZE);
  assert.equal(s[112 * 6 + 5], 1);
  assert.deepEqual(s.slice(-5), [0.01, 0.02, 0.86, 1, 0]);
  const action = {
    kind: "place", score: 2,
    placements: [{ row: 7, col: 8, rackIndex: 2, tile: createAssignedBlank("A") }]
  };
  const a = encodeAction(state, action);
  assert.equal(a.length, ACTION_SIZE);
  assert.equal(a[113 * 6 + 5], 1);
  assert.equal(a[1350 + 26], 0);
  assert.equal(a.at(-1), 0.02);
});

test("DQN chooses by learned Q values across legal moves, not changing action indices", () => {
  const weights = model();
  weights.layers[0].weight[0][STATE_SIZE + ACTION_SIZE - 1] = 1;
  weights.layers[1].weight[0][0] = 1;
  weights.layers[2].weight[0][0] = 1;
  const words = new Set(["AT", "CAT"]);
  const state = position();
  const action = createDQNAgent(words, weights)(state);
  assert.equal(action.kind, "place");
  const engine = createGameState();
  engine.board = state.board;
  engine.players[0].rack = state.rack;
  assert.equal(analyzeMove(engine, action.placements, words).score, action.score);
  assert.ok(Math.abs(action.qValue - action.score / 100) < 1e-6);
});

test("DQN requires real updates and finite correctly shaped weights", () => {
  const weights = model();
  assert.equal(validateModel(weights), weights);
  for (const patch of [
    { schema: 2 }, { updates: 0 }, { episodes: 0 }, { dictionary_sha256: "wrong" },
    { state_size: STATE_SIZE + 1 }, { layers: [] }
  ]) assert.throws(() => validateModel({ ...weights, ...patch }));
  weights.layers[0].weight[0][0] = NaN;
  assert.throws(() => validateModel(weights));
  assert.throws(() => createDQNAgent(new Set(["AT"]), null));
});

test("an existing Greedy worker accepts a DQN model loaded after initialization", async () => {
  const previous = globalThis.self;
  const messages = [];
  globalThis.self = { postMessage: message => messages.push(message) };
  try {
    await import("../../docs/agent-worker.js");
    const worker = globalThis.self;
    worker.onmessage({ data: { type: "initialize", words: ["AT"] } });
    worker.onmessage({ data: { type: "move", id: 1, opponent: "greedy", position: position() } });
    assert.equal(messages.at(-1).action.kind, "place");
    worker.onmessage({ data: { type: "dqn-model", model: model() } });
    worker.onmessage({ data: { type: "move", id: 2, opponent: "dqn", position: position() } });
    assert.equal(messages.at(-1).id, 2);
    assert.equal(messages.at(-1).action.kind, "place");
    worker.onmessage({ data: { type: "dqn-model", model: null } });
    worker.onmessage({ data: { type: "move", id: 3, opponent: "dqn", position: position() } });
    assert.match(messages.at(-1).error, /checkpoint/);
  } finally {
    if (previous === undefined) delete globalThis.self;
    else globalThis.self = previous;
  }
});
