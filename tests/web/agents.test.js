import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { createDPAgent, createGreedyAgent } from "../../docs/agents.js";
import {
  analyzeMove, commitMove, createAssignedBlank, createGameState, createTile,
  exchangeTiles, passTurn, validateSavedState, currentAgent
} from "../../docs/engine.js";
import { createGameStorage } from "../../docs/storage.js";

function choose(agent, state) {
  return agent({
    board: state.board,
    rack: state.players[state.currentPlayer].rack,
    bagSize: state.bag.length
  });
}

function apply(state, action, dictionary) {
  if (action.kind === "place") return commitMove(state, action.placements, dictionary);
  if (action.kind === "exchange") return exchangeTiles(state, action.rackIndexes, () => 0.5);
  return passTurn(state);
}

test("greedy selects the maximum immediate score without mutating its input", () => {
  const state = createGameState();
  state.players[0].rack = [..."CATS"].map(createTile);
  const dictionary = new Set(["AT", "CAT", "CATS", "SAT", "SCAT", "ACT", "CAST"]);
  const before = JSON.stringify(state);
  const agent = createGreedyAgent(dictionary);
  const action = choose(agent, state);
  assert.equal(action.kind, "place");
  assert.equal(action.score, 12);
  assert.equal(analyzeMove(state, action.placements, dictionary).score, action.score);
  assert.equal(JSON.stringify(state), before);
  assert.deepEqual(choose(agent, state), action);
});

test("blank assignments are legal, worth zero, and retain rack indices", () => {
  const state = createGameState();
  state.players[0].rack = ["?", "A", "T"].map(createTile);
  const dictionary = new Set(["CAT"]);
  const action = choose(createGreedyAgent(dictionary), state);
  assert.equal(action.kind, "place");
  assert.equal(action.score, 4);
  const blank = action.placements.find(p => p.tile.blank);
  assert.deepEqual(blank.tile, createAssignedBlank("C"));
  assert.equal(blank.rackIndex, 0);
  assert.equal(new Set(action.placements.map(p => p.rackIndex)).size, 3);
  assert.equal(commitMove(state, action.placements, dictionary).score, 4);
});

test("greedy includes the engine's seven-tile bingo bonus", () => {
  const state = createGameState();
  state.players[0].rack = [..."READING"].map(createTile);
  const dictionary = new Set(["READING"]);
  const action = choose(createGreedyAgent(dictionary), state);
  assert.equal(action.kind, "place");
  assert.equal(action.placements.length, 7);
  assert.equal(action.bingo, true);
  assert.ok(action.score >= 50);
  assert.equal(commitMove(state, action.placements, dictionary).score, action.score);
});

test("greedy exchanges all rack tiles when blocked, or passes when exchange is unavailable", () => {
  const state = createGameState();
  state.players[0].rack = [..."ZZZZZZZ"].map(createTile);
  const agent = createGreedyAgent(new Set(["AT"]));
  assert.deepEqual(choose(agent, state), { kind: "exchange", rackIndexes: [0, 1, 2, 3, 4, 5, 6] });
  state.bag = state.bag.slice(0, 6);
  assert.deepEqual(choose(agent, state), { kind: "pass" });
});

test("greedy uses existing tiles and validates perpendicular cross words", () => {
  const state = createGameState();
  state.board[7][7] = createTile("A");
  state.board[6][8] = createTile("I");
  state.players[0].rack = ["T", "S", "?"].map(createTile);
  const dictionary = new Set(["AT", "AS", "IT", "IS", "SAT", "TAT"]);
  const action = choose(createGreedyAgent(dictionary), state);
  assert.equal(action.kind, "place");
  assert.equal(analyzeMove(state, action.placements, dictionary).score, action.score);
  assert.equal(state.board[7][7].letter, "A");
});

test("greedy score matches exhaustive two-letter placement search, including blanks and hooks", () => {
  const dictionary = new Set(["AT", "AS", "IS", "IT", "TA", "TI"]);
  const state = createGameState();
  state.players[0].rack = ["A", "T", "?", "S"].map(createTile);
  const agent = createGreedyAgent(dictionary);
  for (const withBoard of [false, true]) {
    if (withBoard) {
      state.board[7][7] = createTile("A");
      state.board[6][8] = createTile("I");
    }
    let bestScore = -1;
    for (const word of dictionary) {
      for (const [dr, dc] of [[0, 1], [1, 0]]) {
        for (let r = 0; r + dr < 15; r += 1) {
          for (let c = 0; c + dc < 15; c += 1) {
            for (let mask = 0; mask < 4; mask += 1) {
              const used = new Set();
              const placements = [];
              let valid = true;
              for (let i = 0; i < 2; i += 1) {
                const row = r + i * dr, col = c + i * dc;
                const fixed = state.board[row][col];
                if (fixed) {
                  if (fixed.letter !== word[i] || mask & (1 << i)) valid = false;
                  continue;
                }
                const letter = mask & (1 << i) ? "?" : word[i];
                const index = state.players[0].rack.findIndex((t, j) => !used.has(j) && t.letter === letter);
                if (index < 0) { valid = false; break; }
                used.add(index);
                placements.push({
                  row, col, rackIndex: index,
                  tile: letter === "?" ? createAssignedBlank(word[i]) : createTile(letter)
                });
              }
              if (!valid) continue;
              try {
                bestScore = Math.max(bestScore, analyzeMove(state, placements, dictionary).score);
              } catch (error) {
                assert.ok(error instanceof Error);
              }
            }
          }
        }
      }
    }
    assert.ok(bestScore >= 0);
    assert.equal(choose(agent, state).score, bestScore);
  }
});

test("agent games save opponent metadata while old multiplayer saves remain valid", () => {
  const memory = new Map();
  const storage = createGameStorage({
    getItem: key => memory.get(key) ?? null,
    setItem: (key, value) => memory.set(key, value),
    removeItem: key => memory.delete(key)
  });
  const state = createGameState(["Human", "Greedy"]);
  assert.equal(validateSavedState(state), true);
  state.opponent = "greedy";
  passTurn(state);
  storage.writeAuto(state);
  assert.equal(storage.readAuto().opponent, "greedy");
  assert.equal(storage.readAuto().currentPlayer, 1);
  storage.writeManual(state);
  assert.equal(storage.readManual().opponent, "greedy");
  state.opponent = "dp";
  assert.equal(storage.writeAuto(state), true);
  assert.equal(storage.readAuto().opponent, "dp");
  assert.equal(storage.writeManual(state), true);
  assert.equal(storage.readManual().opponent, "dp");
  state.opponent = "quackle";
  assert.equal(storage.writeAuto(state), true);
  assert.equal(storage.readAuto().opponent, "quackle");
  assert.equal(storage.writeManual(state), true);
  assert.equal(storage.readManual().opponent, "quackle");
  state.opponent = "dqn";
  assert.equal(storage.writeAuto(state), true);
  assert.equal(storage.readAuto().opponent, "dqn");
  assert.equal(storage.writeManual(state), true);
  assert.equal(storage.readManual().opponent, "dqn");
  for (const opponent of [null, {}, "unknown"]) {
    assert.equal(validateSavedState({ ...state, opponent }), false);
  }
  const multiplayer = createGameState(["One", "Two", "Three"]);
  multiplayer.opponent = "greedy";
  assert.equal(validateSavedState(multiplayer), false);
  multiplayer.opponent = "dp";
  assert.equal(validateSavedState(multiplayer), false);
});

test("agent match routing and pause state survive auto and manual saves", () => {
  const memory = new Map();
  const storage = createGameStorage({
    getItem: key => memory.get(key) ?? null,
    setItem: (key, value) => memory.set(key, value),
    removeItem: key => memory.delete(key)
  });
  const state = createGameState(["Greedy 1", "DP 2"]);
  state.opponent = "agents";
  state.agents = ["greedy", "dp"];
  state.agentPaused = true;
  assert.equal(validateSavedState(state), true);
  assert.equal(currentAgent(state), "greedy");
  passTurn(state);
  assert.equal(currentAgent(state), "dp");
  storage.writeAuto(state);
  storage.writeManual(state);
  for (const restored of [storage.readAuto(), storage.readManual()]) {
    assert.deepEqual(restored.agents, ["greedy", "dp"]);
    assert.equal(restored.agentPaused, true);
    assert.equal(currentAgent(restored), "dp");
  }
  for (const patch of [
    { agents: ["greedy"] }, { agents: ["greedy", "unknown"] },
    { agents: "greedy" }, { agentPaused: undefined }, { agentPaused: "true" },
    { players: [...state.players, { ...state.players[0], name: "Third" }] },
    { opponent: "human" }
  ]) assert.equal(validateSavedState({ ...state, ...patch }), false);
  for (const agents of [
    ["greedy", "greedy"], ["dp", "dp"], ["dp", "greedy"],
    ["quackle", "greedy"], ["dp", "quackle"], ["quackle", "quackle"],
    ["dqn", "quackle"], ["greedy", "dqn"], ["dqn", "dqn"]
  ]) {
    assert.equal(validateSavedState({ ...state, agents }), true);
    state.agents = agents;
    state.currentPlayer = 0;
    assert.equal(currentAgent(state), agents[0]);
    state.currentPlayer = 1;
    assert.equal(currentAgent(state), agents[1]);
    storage.writeManual(state);
    assert.deepEqual(storage.readManual().agents, agents);
  }
});

test("old human and single-agent games route turns as before", () => {
  const state = createGameState();
  assert.equal(currentAgent(state), null);
  for (const opponent of ["human", "greedy", "dp", "quackle", "dqn"]) {
    state.opponent = opponent;
    state.currentPlayer = 0;
    assert.equal(currentAgent(state), null);
    state.currentPlayer = 1;
    assert.equal(currentAgent(state), opponent === "human" ? null : opponent);
  }
});

test("mixed-agent match alternates the selected policies and completes legally", () => {
  const dictionary = new Set(readFileSync(new URL("../../docs/words.txt", import.meta.url), "utf8").split(/\s+/).filter(Boolean));
  const agents = { greedy: createGreedyAgent(dictionary), dp: createDPAgent(dictionary) };
  const state = createGameState(["Greedy 1", "DP 2"], () => 0.5);
  Object.assign(state, { opponent: "agents", agents: ["greedy", "dp"], agentPaused: false });
  let turns = 0;
  const acted = new Set();
  while (!state.gameOver && turns < 200) {
    const policy = currentAgent(state);
    acted.add(policy);
    apply(state, choose(agents[policy], state), dictionary);
    assert.equal(validateSavedState(state), true);
    turns += 1;
  }
  assert.deepEqual([...acted].sort(), ["dp", "greedy"]);
  assert.equal(state.gameOver, true);
});

test("depth-one DP agrees with Greedy; lookahead changes its choice without mutating state", () => {
  const dictionary = new Set([
    "AT", "CAT", "CATS", "SAT", "SCAT", "ACT", "CAST", "AS", "TA", "AA",
    "TAT", "TATS", "CAR", "CARS", "ART", "ARTS", "RAT", "RATS", "TAR",
    "TARS", "STAR", "START", "TART"
  ]);
  const state = createGameState();
  state.players[0].rack = [..."CATSRAT"].map(createTile);
  const before = JSON.stringify(state);
  const greedy = choose(createGreedyAgent(dictionary), state);
  const shallow = choose(createDPAgent(dictionary, { depth: 1 }), state);
  const agent = createDPAgent(dictionary);
  const deep = choose(agent, state);
  assert.deepEqual(shallow.placements, greedy.placements);
  assert.equal(shallow.search.value, greedy.score);
  assert.deepEqual(greedy.words, ["START"]);
  assert.deepEqual(deep.words, ["CARS"]);
  assert.ok(deep.search.value > deep.score);
  assert.ok(Math.abs(deep.search.value - 23.16) < 1e-10);
  assert.deepEqual(choose(agent, state), deep);
  assert.equal(JSON.stringify(state), before);
  assert.equal(analyzeMove(state, deep.placements, dictionary).score, deep.score);
  state.players[0].rack.reverse();
  const reordered = choose(agent, state);
  assert.equal(reordered.search.value, deep.search.value);
  assert.deepEqual(reordered.words, deep.words);
  assert.equal(analyzeMove(state, reordered.placements, dictionary).score, reordered.score);
});

test("memoized DP reuses repeated board/rack/horizon states", () => {
  const dictionary = new Set(["AA"]);
  const state = createGameState();
  state.players[0].rack = [..."AAAAAAA"].map(createTile);
  const action = choose(createDPAgent(dictionary, { beamWidth: 12 }), state);
  assert.ok(action.search.cacheHits > 0);
  assert.ok(action.search.states > 0);
  assert.ok(Math.abs(action.search.value - 15.88) < 1e-10);
});

test("DP Bellman value matches exhaustive unpruned planning on a small rack", () => {
  const dictionary = new Set(["AA"]);
  const state = createGameState();
  state.players[0].rack = [..."AAAA"].map(createTile);

  function reference(board, rack, depth) {
    if (!depth || !rack.length) return 0;
    const snapshot = { board, players: [{ rack }, { rack: [] }], currentPlayer: 0,
      gameOver: false, bag: [], history: [] };
    const actions = new Map();
    for (const [dr, dc] of [[0, 1], [1, 0]]) {
      for (let r = 0; r + dr < 15; r += 1) {
        for (let c = 0; c + dc < 15; c += 1) {
          const placements = [];
          for (let i = 0; i < 2; i += 1) {
            const row = r + i * dr, col = c + i * dc;
            if (!board[row][col]) {
              placements.push({ row, col, rackIndex: placements.length, tile: createTile("A") });
            }
          }
          if (!placements.length || placements.length > rack.length) continue;
          let result;
          try {
            result = analyzeMove(snapshot, placements, dictionary);
          } catch (error) {
            assert.ok(error instanceof Error);
            continue;
          }
          const key = placements.map(p => `${p.row},${p.col}`).sort().join(";");
          actions.set(key, { placements, score: result.score });
        }
      }
    }
    let best = 0;
    for (const { placements, score } of actions.values()) {
      const next = board.map(row => row.slice());
      for (const p of placements) next[p.row][p.col] = p.tile;
      best = Math.max(best, score + 0.9 * reference(
        next, rack.slice(placements.length), depth - 1
      ));
    }
    return best;
  }
  const expected = reference(state.board, state.players[0].rack, 3);
  const actual = choose(createDPAgent(dictionary, { beamWidth: 12 }), state);
  assert.ok(Math.abs(actual.search.value - expected) < 1e-10);
});

test("DP uses blank-aware scores and falls back to exchange/pass when blocked", () => {
  const state = createGameState();
  state.players[0].rack = ["?", "A", "T"].map(createTile);
  const dictionary = new Set(["CAT"]);
  const agent = createDPAgent(dictionary);
  const action = choose(agent, state);
  assert.equal(action.score, 4);
  assert.equal(action.search.value, 4);
  assert.deepEqual(action.placements.find(p => p.tile.blank).tile, createAssignedBlank("C"));
  state.players[0].rack = [..."ZZZZZZZ"].map(createTile);
  assert.equal(choose(agent, state).kind, "exchange");
  state.bag = state.bag.slice(0, 6);
  assert.equal(choose(agent, state).kind, "pass");
});

test("DP rejects invalid search bounds", () => {
  const dictionary = new Set(["AT"]);
  for (const options of [{ depth: 0 }, { depth: 4 }, { depth: 1.5 },
    { beamWidth: 0 }, { beamWidth: 13 }, { beamWidth: NaN }]) {
    assert.throws(() => createDPAgent(dictionary, options), /DP/);
  }
});

test("DP forecasts do not inspect hidden information or invent draws", () => {
  const state = createGameState();
  const position = {
    board: state.board, rack: [..."CATS"].map(createTile), bagSize: 86
  };
  for (const key of ["opponentRack", "bag", "bagContents"]) {
    Object.defineProperty(position, key, {
      get() { throw new Error("Hidden information accessed"); }
    });
  }
  const agent = createDPAgent(new Set(["CAT", "CATS", "AT", "SAT"]));
  const action = agent(position);
  position.bagSize = 0;
  assert.deepEqual(agent(position), action);
  assert.ok(action.search.value >= action.score);
});

for (const [name, createAgent] of [["greedy", createGreedyAgent], ["dp", createDPAgent]]) {
test(`full ${name} self-play uses the bundled lexicon and conserves tiles through game over`, () => {
  const dictionary = new Set(readFileSync(new URL("../../docs/words.txt", import.meta.url), "utf8").split(/\s+/).filter(Boolean));
  const agent = createAgent(dictionary);
  const state = createGameState(["Human", "Agent"], () => 0.5);
  state.opponent = name;
  let turns = 0, placements = 0;
  while (!state.gameOver && turns < 200) {
    const action = choose(agent, state);
    if (action.kind === "place") {
      placements += 1;
      assert.equal(analyzeMove(state, action.placements, dictionary).score, action.score);
    }
    apply(state, action, dictionary);
    assert.equal(validateSavedState(state), true);
    turns += 1;
  }
  assert.ok(placements > 2);
  assert.equal(state.gameOver, true);
});
}
