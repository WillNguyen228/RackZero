"""Check that the fast simulator move generator equals the authoritative one.

Plays games with a mix of greedy and random moves. At every position it compares,
for a sampled hidden opponent rack:

* the live game's Java-scored legal moves (the rules authority),
* ``Simulator(movegen="reference")`` (the original Python walk), and
* ``Simulator(movegen="fast")``.

The fast tuple must equal the authority exactly: same moves, order and scores.
It exits nonzero on the first mismatch.

  python scripts/verify_movegen.py --dictionary WOW24v3.zip --games 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from rackzero.agents import choose  # noqa: E402
from rackzero.belief import RackBelief  # noqa: E402
from rackzero.game import Game, Simulator  # noqa: E402
from rackzero.training import lexicon_for_path  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dictionary", default="WOW24v3.zip", help="path, or 'tiny'")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    words = lexicon_for_path(None if args.dictionary == "tiny" else args.dictionary).words
    game = Game(words, particles=1)
    fast, reference = Simulator(words, movegen="fast"), Simulator(words, movegen="reference")
    rng = np.random.default_rng(args.seed)
    positions = moves_checked = 0
    started = time.perf_counter()
    warnings.simplefilter("ignore", RuntimeWarning)
    for index in range(args.games):
        info = game.reset(args.seed + index)
        while True:
            live = game.legal_moves()
            world = fast.determinize(info, RackBelief(info, 1, rng).sample())
            for name, simulator in (("fast", fast), ("reference", reference)):
                if simulator.legal_moves(world, rng) != live:
                    print(f"MISMATCH ({name}) in game {index} turn {info.observation['turn']}")
                    return 1
            positions += 1
            moves_checked += len(live)
            kind = "greedy" if rng.random() < 0.8 else "random"
            info, terminal, truncated = game.step(choose(kind, info, live, rng))
            if terminal or truncated:
                break
        print(f"game {index + 1}/{args.games}: {positions} positions, {moves_checked} moves "
              "identical so far", flush=True)
    print(f"OK: {positions} positions and {moves_checked} candidate moves identical "
          f"(fast == reference == Java authority) in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
