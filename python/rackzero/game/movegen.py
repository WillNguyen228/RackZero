"""Fast simulator-only move generator, bit-for-bit equivalent to the reference.

``ScrabbleEnv.legal_moves`` (Java-scored) stays the rules authority for live
games. Inside search, ``Simulator`` calls this generator instead. It
reproduces the reference algorithm exactly: the same start squares, anchors,
cross-checks, trie traversal order, single-tile de-duplication across directions,
exchange enumeration and pass. So the returned tuple (moves, order and scores)
is identical. Two changes make it faster, without changing any result:

* Board-dependent data (anchors, perpendicular cross-check sets, cross-word
  sums) is computed once per board and cached. Many sampled worlds in one
  search share a board.
* Scores are accumulated incrementally while walking, using the same formula
  as ``score_placement`` / Java ``Board.preview``. There is no per-candidate
  re-scan, and the hot loop uses plain Python lists instead of NumPy indexing.

Parity with the authoritative generator is enforced by
``tests/python/test_rackzero.py::test_fast_movegen_matches_reference_generator``
and by ``scripts/verify_movegen.py`` on full-dictionary games.
"""

from __future__ import annotations

from collections import OrderedDict
from itertools import product

import numpy as np

from scrabble_gym.environment import ALPHABET, RACK_SIZE, SIZE, Move, ScoredMove


_LETTERS = ALPHABET[:-1]
_POINTS = (1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10)
_LETTER_POINTS = dict(zip(_LETTERS, _POINTS))


class BoardCache:
    """Per-board anchors, cross-checks and cross sums, keyed by board bytes."""

    def __init__(self, words: frozenset[str], letter_mult: np.ndarray, word_mult: np.ndarray,
                 capacity: int = 4096):
        self.words = words
        self.lm = [[int(x) for x in row] for row in letter_mult]
        self.wm = [[int(x) for x in row] for row in word_mult]
        self.capacity = capacity
        self._entries: OrderedDict[bytes, tuple] = OrderedDict()
        self.hits = self.misses = 0

    def get(self, board: np.ndarray, blanks: np.ndarray):
        key = board.tobytes() + blanks.tobytes()
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return entry
        self.misses += 1
        entry = self._build(board, blanks)
        self._entries[key] = entry
        if len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return entry

    def _build(self, board_array: np.ndarray, blanks_array: np.ndarray):
        board = board_array.tolist()
        blanks = blanks_array.tolist()
        words = self.words
        empty = not any(any(row) for row in board)
        anchors = [[False] * SIZE for _ in range(SIZE)]
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c]:
                    continue
                if empty:
                    anchors[r][c] = (r, c) == (7, 7)
                else:
                    anchors[r][c] = (
                        (c + 1 < SIZE and bool(board[r][c + 1]))
                        or (c > 0 and bool(board[r][c - 1]))
                        or (r + 1 < SIZE and bool(board[r + 1][c]))
                        or (r > 0 and bool(board[r - 1][c]))
                    )
        lines = []
        for direction in (0, 1):
            dr, dc = (0, 1) if direction == 0 else (1, 0)
            pr, pc = dc, dr
            per_line = []
            for line in range(SIZE):
                cells = []
                for i in range(SIZE):
                    r, c = (line, i) if direction == 0 else (i, line)
                    code = board[r][c]
                    if code:
                        letter = ALPHABET[code - 1]
                        cells.append((letter, 0 if blanks[r][c] else _LETTER_POINTS[letter],
                                      None, False, 0, False, 1, 1))
                        continue
                    prefix, suffix, cross_sum = "", "", 0
                    ar, ac = r - pr, c - pc
                    while 0 <= ar < SIZE and 0 <= ac < SIZE and board[ar][ac]:
                        prefix = ALPHABET[board[ar][ac] - 1] + prefix
                        cross_sum += 0 if blanks[ar][ac] else _POINTS[board[ar][ac] - 1]
                        ar, ac = ar - pr, ac - pc
                    ar, ac = r + pr, c + pc
                    while 0 <= ar < SIZE and 0 <= ac < SIZE and board[ar][ac]:
                        suffix += ALPHABET[board[ar][ac] - 1]
                        cross_sum += 0 if blanks[ar][ac] else _POINTS[board[ar][ac] - 1]
                        ar, ac = ar + pr, ac + pc
                    has_cross = bool(prefix or suffix)
                    allowed = (frozenset(ch for ch in _LETTERS if prefix + ch + suffix in words)
                               if has_cross else None)
                    cells.append((None, 0, allowed, anchors[r][c], cross_sum, has_cross,
                                  self.lm[r][c], self.wm[r][c]))
                per_line.append(cells)
            lines.append(per_line)
        return lines


def generate(board: np.ndarray, blanks: np.ndarray, rack_counts: np.ndarray, bag_size: int,
             trie: dict, cache: BoardCache) -> tuple[ScoredMove, ...]:
    """Same output as ``ScrabbleEnv.legal_moves`` for the given (hypothetical) state."""
    lines = cache.get(board, blanks)
    rack = dict(zip(ALPHABET, (int(x) for x in rack_counts)))
    rack_total = sum(rack.values())
    moves: list[ScoredMove] = []
    seen: set = set()

    for direction in (0, 1):
        def walk(cells, line_index, i, node, text, blank_offsets, placed, touched, start,
                 main, mult, cross):
            if "" in node and placed and touched and (i >= SIZE or cells[i][0] is None):
                key = tuple(placed)
                if key not in seen:
                    seen.add(key)
                    r, c = (line_index, start) if direction == 0 else (start, line_index)
                    score = main * mult + cross + (50 if len(placed) == RACK_SIZE else 0)
                    moves.append(ScoredMove(
                        Move("place", r, c, direction, text, blank_offsets), score))
            if i >= SIZE:
                return
            letter, points, allowed, anchor, cross_sum, has_cross, lm, wm = cells[i]
            if letter is not None:
                child = node.get(letter)
                if child is not None:
                    walk(cells, line_index, i + 1, child, text + letter, blank_offsets, placed,
                         True, start, main + points, mult, cross)
                return
            if len(placed) == RACK_SIZE:
                return
            if rack["?"] > 0:
                options = [ch for ch in node if ch]
            else:
                options = [ch for ch in _LETTERS if rack[ch] and ch in node]
            r, c = (line_index, i) if direction == 0 else (i, line_index)
            touched_next = touched or anchor
            for ch in options:
                if allowed is not None and ch not in allowed:
                    continue
                child = node[ch]
                for tile in (ch, "?"):
                    if not rack[tile]:
                        continue
                    rack[tile] -= 1
                    is_blank = tile == "?"
                    value = 0 if is_blank else _LETTER_POINTS[ch] * lm
                    walk(cells, line_index, i + 1, child, text + ch,
                         blank_offsets + (len(text),) if is_blank else blank_offsets,
                         placed + [(r, c, ch, is_blank)], touched_next, start,
                         main + value, mult * wm,
                         cross + ((cross_sum + value) * wm if has_cross else 0))
                    rack[tile] += 1

        # Reference order: start row outer, start column inner, for both directions.
        for r0 in range(SIZE):
            for c0 in range(SIZE):
                line_index, start = (r0, c0) if direction == 0 else (c0, r0)
                cells = lines[direction][line_index]
                if start > 0 and cells[start - 1][0] is not None:
                    continue
                empty_count = 0
                for j in range(start, SIZE):
                    empty_count += cells[j][0] is None
                    if empty_count > rack_total:
                        break
                    if cells[j][3]:
                        walk(cells, line_index, start, trie, "", (), [], False, start, 0, 1, 0)
                        break
    if bag_size >= RACK_SIZE:
        tiles = [t for t in ALPHABET if rack[t]]
        for amounts in product(*(range(rack[t] + 1) for t in tiles)):
            exchange = "".join(t * n for t, n in zip(tiles, amounts))
            if exchange:
                moves.append(ScoredMove(Move("exchange", exchange=exchange), 0))
    moves.append(ScoredMove(Move(), 0))
    return tuple(moves)
