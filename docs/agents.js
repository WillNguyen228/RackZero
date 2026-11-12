"use strict";

import {
  BOARD_SIZE, CENTER, RACK_SIZE, analyzeMove, createAssignedBlank
} from "./engine.js";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const inside = (row, col) => row >= 0 && col >= 0 && row < BOARD_SIZE && col < BOARD_SIZE;

/** Prefix search proposes legal plays; the existing engine supplies every score. */
function createMoveSearch(dictionary) {
  if (!(dictionary instanceof Set) || !dictionary.size) throw new Error("Dictionary is not ready.");
  const trie = new Map();
  for (const word of [...dictionary].sort()) {
    if (!/^[A-Z]{2,15}$/.test(word)) continue;
    let node = trie;
    for (const letter of word) {
      if (!node.has(letter)) node.set(letter, new Map());
      node = node.get(letter);
    }
    node.set("", null);
  }

  return function findPlacements({ board, rack }, limit) {
    const state = {
      board, players: [{ rack }, { rack: [] }], currentPlayer: 0,
      bag: [], history: [], gameOver: false
    };
    const available = new Map();
    rack.forEach((tile, index) => {
      if (!available.has(tile.letter)) available.set(tile.letter, []);
      available.get(tile.letter).push(index);
    });
    const empty = board.every(row => row.every(tile => tile === null));
    const anchors = new Set();
    for (let r = 0; r < BOARD_SIZE; r += 1) {
      for (let c = 0; c < BOARD_SIZE; c += 1) {
        if (board[r][c]) continue;
        if (empty ? r === CENTER && c === CENTER : [[0, 1], [0, -1], [1, 0], [-1, 0]]
          .some(([dr, dc]) => inside(r + dr, c + dc) && board[r + dr][c + dc])) {
          anchors.add(`${r},${c}`);
        }
      }
    }
    const best = [];
    const seen = new Set();
    for (const [dr, dc] of [[0, 1], [1, 0]]) {
      const allowed = new Map();
      for (let r = 0; r < BOARD_SIZE; r += 1) {
        for (let c = 0; c < BOARD_SIZE; c += 1) {
          if (board[r][c]) continue;
          let prefix = "", suffix = "";
          let ar = r - dc, ac = c - dr;
          while (inside(ar, ac) && board[ar][ac]) {
            prefix = board[ar][ac].letter + prefix;
            ar -= dc;
            ac -= dr;
          }
          ar = r + dc;
          ac = c + dr;
          while (inside(ar, ac) && board[ar][ac]) {
            suffix += board[ar][ac].letter;
            ar += dc;
            ac += dr;
          }
          allowed.set(`${r},${c}`, new Set([...LETTERS].filter(
            letter => !(prefix || suffix) || dictionary.has(prefix + letter + suffix)
          )));
        }
      }

      function walk(r, c, node, placed, touched) {
        if (node.has("") && placed.length && touched && (!inside(r, c) || !board[r][c])) {
          const key = placed.map(p => `${p.row},${p.col},${p.tile.letter},${p.tile.blank}`).join(";");
          if (!seen.has(key)) {
            const result = analyzeMove(state, placed, dictionary);
            if (best.length < limit || result.score > best[best.length - 1].score) {
              const candidate = { kind: "place", placements: placed.map(p => ({ ...p })), ...result };
              const index = best.findIndex(item => item.score < result.score);
              best.splice(index < 0 ? best.length : index, 0, candidate);
              if (best.length > limit) best.pop();
            }
            seen.add(key);
          }
        }
        if (!inside(r, c)) return;
        if (board[r][c]) {
          const child = node.get(board[r][c].letter);
          if (child) walk(r + dr, c + dc, child, placed, true);
          return;
        }
        if (placed.length === RACK_SIZE) return;
        for (const [letter, child] of node) {
          if (!letter || !allowed.get(`${r},${c}`).has(letter)) continue;
          for (const rackLetter of [letter, "?"]) {
            const indexes = available.get(rackLetter);
            if (!indexes?.length) continue;
            const rackIndex = indexes.pop();
            const tile = rackLetter === "?" ? createAssignedBlank(letter) : { ...rack[rackIndex] };
            placed.push({ row: r, col: c, rackIndex, tile });
            walk(r + dr, c + dc, child, placed, touched || anchors.has(`${r},${c}`));
            placed.pop();
            indexes.push(rackIndex);
          }
        }
      }

      for (let r = 0; r < BOARD_SIZE; r += 1) {
        for (let c = 0; c < BOARD_SIZE; c += 1) {
          if (inside(r - dr, c - dc) && board[r - dr][c - dc]) continue;
          let ar = r, ac = c, emptyCount = 0;
          while (inside(ar, ac)) {
            if (!board[ar][ac]) emptyCount += 1;
            if (emptyCount > rack.length) break;
            if (anchors.has(`${ar},${ac}`)) {
              walk(r, c, trie, [], false);
              break;
            }
            ar += dr;
            ac += dc;
          }
        }
      }
    }
    return best;
  };
}

function blockedAction({ rack, bagSize }) {
  if (bagSize >= RACK_SIZE && rack.length) {
    return { kind: "exchange", rackIndexes: rack.map((_, index) => index) };
  }
  return { kind: "pass" };
}

export function createGreedyAgent(dictionary) {
  const search = createMoveSearch(dictionary);
  return position => search(position, 1)[0] ?? blockedAction(position);
}

export function createCandidateGenerator(dictionary) {
  const search = createMoveSearch(dictionary);
  return position => {
    const actions = search(position, Infinity);
    if (position.bagSize >= RACK_SIZE) {
      const groups = new Map();
      position.rack.forEach((tile, index) => {
        if (!groups.has(tile.letter)) groups.set(tile.letter, []);
        groups.get(tile.letter).push(index);
      });
      const entries = [...groups].sort(([a], [b]) => a.localeCompare(b));
      function exchange(group, indexes) {
        if (group === entries.length) {
          if (indexes.length) actions.push({ kind: "exchange", rackIndexes: indexes });
          return;
        }
        const tiles = entries[group][1];
        for (let count = 0; count <= tiles.length; count += 1) {
          exchange(group + 1, indexes.concat(tiles.slice(0, count)));
        }
      }
      exchange(0, []);
    }
    actions.push({ kind: "pass" });
    return actions;
  };
}

function forecastPosition(position, action) {
  const board = position.board.map(row => row.slice());
  const used = new Set(action.placements.map(p => p.rackIndex));
  for (const p of action.placements) board[p.row][p.col] = { ...p.tile };
  return { board, rack: position.rack.filter((_, index) => !used.has(index)) };
}

function forecastKey({ board, rack }, depth) {
  const cells = board.map(row => row.map(
    tile => !tile ? "." : tile.blank ? tile.letter.toLowerCase() : tile.letter
  ).join("")).join("");
  return `${depth}:${cells}:${rack.map(tile => tile.letter).sort().join("")}`;
}

/**
 * Beam-limited Bellman recurrence over own-rack placements:
 * V(s,d) = max_a [score(s,a) + 0.9 V(next(s,a),d-1)].
 * Forecasts deliberately omit draws and opponent moves, never hidden information.
 */
export function createDPAgent(dictionary, { depth = 3, beamWidth = 6 } = {}) {
  if (!Number.isInteger(depth) || depth < 1 || depth > 3) {
    throw new Error("DP depth must be an integer from 1 to 3.");
  }
  if (!Number.isInteger(beamWidth) || beamWidth < 1 || beamWidth > 12) {
    throw new Error("DP beam width must be an integer from 1 to 12.");
  }
  const search = createMoveSearch(dictionary);
  return position => {
    const memo = new Map();
    let cacheHits = 0;
    function value(next, remaining) {
      const key = forecastKey(next, remaining);
      if (memo.has(key)) {
        cacheHits += 1;
        return memo.get(key);
      }
      let bestValue = 0;
      if (remaining > 0 && next.rack.length) {
        for (const action of search(next, beamWidth)) {
          bestValue = Math.max(bestValue,
            action.score + 0.9 * value(forecastPosition(next, action), remaining - 1));
        }
      }
      memo.set(key, bestValue);
      return bestValue;
    }

    const actions = search(position, beamWidth);
    if (!actions.length) return blockedAction(position);
    let best = actions[0], bestValue = -Infinity;
    for (const action of actions) {
      const candidateValue = action.score + 0.9 * value(
        forecastPosition(position, action), depth - 1
      );
      if (candidateValue > bestValue) {
        best = action;
        bestValue = candidateValue;
      }
    }
    return {
      ...best,
      search: { depth, beamWidth, value: bestValue, states: memo.size, cacheHits }
    };
  };
}
