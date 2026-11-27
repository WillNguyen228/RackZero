"""Where does our agent lose to Quackle's championship player? Move gaps by game phase.

Positions come from self-play games of the agent (a random ply 0..--max-ply). At
each position the agent picks its move (Q agent, as in evaluation) and Quackle's
championship player (Smart Bogowin, --quackle-seconds per turn) picks its move.
Where they differ, both moves are valued by --samples rollouts on common sampled
worlds (opponent rack from the mover's belief, same bag order for both moves);
both sides of every rollout are played by Quackle's STATIC player, a neutral
yardstick that is neither of the two players being compared. With the bag empty
the world is exact, so one rollout suffices.

Reports per phase (by tiles in the bag): how often the picks differ, and on
those positions the mean advantage of Quackle's pick over ours in final spread
(points) and in rollout win rate, with standard errors; and the expected cost per
position of that phase (advantage x disagreement rate).

  python scripts/phase_gap.py --checkpoint PATH --positions 600 --samples 24 --workers 6
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

PHASES = [("opening (bag >= 60)", 60, 200), ("middle (bag 20-59)", 20, 59),
          ("late (bag 7-19)", 7, 19), ("pre-endgame (bag 1-6)", 1, 6), ("endgame (bag 0)", 0, 0)]
_STATE = {}


def phase_of(bag: int) -> str:
    return next(name for name, low, high in PHASES if low <= bag <= high)


def _init(checkpoint, seconds, prior_weight):
    import torch
    torch.set_num_threads(1)
    from rackzero.checkpoints import checkpoint_config, load_checkpoint, model_from_checkpoint
    from rackzero.game import Simulator
    from rackzero.models.networks import NetworkEvaluator
    from rackzero.quackle import QuackleEngine
    from rackzero.search import SearchConfig
    from rackzero.training import lexicon_for
    ck = load_checkpoint(checkpoint)
    cfg = checkpoint_config(ck)
    words = lexicon_for(cfg).words
    search = replace(SearchConfig(**asdict(cfg.search)), root_selection="q",
                     q_prior_weight=prior_weight, root_noise_fraction=0.0)
    _STATE.update(model=NetworkEvaluator(model_from_checkpoint(ck, "best_model")).eval(),
                  cfg=cfg, search=search, words=words, simulator=Simulator(words),
                  sim=QuackleEngine(words, sim_seconds=seconds), static=QuackleEngine(words))


def _rollout(world, root, move, seed, max_plies=80):
    sim, static = _STATE["simulator"], _STATE["static"]
    rng = np.random.default_rng(seed)
    world = sim.step(world, move, rng)
    for _ in range(max_plies):
        if world.terminal:
            break
        obs = world.information().observation
        moves = sim.legal_moves(world, rng)
        world = sim.step(world, moves[static.choose(obs, moves)].move, rng)
    scores = world.observation["scores"]
    return float(scores[root] - scores[1 - root])


def _position(task):
    from rackzero.game import Game
    from rackzero.search import Search
    seed, max_ply, samples = task
    rng = np.random.default_rng(seed)
    st = _STATE
    game = Game(st["words"], particles=st["search"].particles, rng=np.random.default_rng(seed + 1),
                movegen="fast")
    try:
        info = game.reset(seed=int(seed % 2**31))
        search = Search(st["simulator"], st["model"], st["search"], rng)
        for _ in range(int(rng.integers(0, max_ply + 1))):
            result = search.run(info, moves=game.legal_moves(), belief=game.belief)
            info, terminal, truncated = game.step(result.moves[int(result.selected)].move)
            if terminal or truncated:
                return None
        legal = game.legal_moves()
        if len(legal) < 2:
            return None
        obs = info.observation
        bag = int(obs["bag_size"])
        ours = search.run(info, moves=legal, belief=game.belief)
        our_move = ours.moves[int(ours.selected)].move
        theirs = legal[st["sim"].choose(obs, legal)].move
        row = {"seed": int(seed), "bag": bag, "phase": phase_of(bag), "differ": our_move != theirs}
        if not row["differ"]:
            return row
        root = int(obs["player"])
        n = 1 if bag == 0 else samples
        spreads = np.zeros((n, 2))
        for k in range(n):
            world = st["simulator"].determinize(info, game.belief.sample())
            s = int(rng.integers(0, 2**63 - 1))
            spreads[k] = (_rollout(world, root, theirs, s), _rollout(world, root, our_move, s))
        diff = spreads[:, 0] - spreads[:, 1]
        wins = (np.sign(spreads[:, 0]) - np.sign(spreads[:, 1])) / 2
        row.update(points=float(diff.mean()), wins=float(wins.mean()))
        return row
    finally:
        game.close()


def report(rows):
    def stat(xs):
        xs = np.asarray(xs, dtype=np.float64)
        if len(xs) < 2:
            return float("nan"), float("nan")
        return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs)))
    lines = [f"{'phase':<24}{'positions':>10}{'differ':>8}  {'Quackle pick better by (on differing)':<40}"
             f"{'cost per position':>18}"]
    for name, _, _ in PHASES:
        rs = [r for r in rows if r["phase"] == name]
        if not rs:
            continue
        differ = [r for r in rs if r["differ"]]
        rate = len(differ) / len(rs)
        p, ps = stat([r["points"] for r in differ])
        w, ws = stat([r["wins"] for r in differ])
        lines.append(f"{name:<24}{len(rs):>10}{rate:>8.2f}  {p:+6.2f} ± {ps:4.2f} pts, "
                     f"{100 * w:+5.1f} ± {100 * ws:4.1f}% wins   {p * rate:+6.2f} pts "
                     f"{100 * w * rate:+5.2f}% wins")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--quackle-seconds", type=int, default=20)
    parser.add_argument("--positions", type=int, default=600)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--max-ply", type=int, default=28)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", required=True, help="JSONL of per-position rows")
    args = parser.parse_args()
    tasks = [(args.seed + 7919 * i, args.max_ply, args.samples)
             for i in range(int(args.positions * 1.5) + 8)]
    rows, started = [], time.time()
    pool = get_context("spawn").Pool(args.workers, _init,
                                     (args.checkpoint, args.quackle_seconds, args.prior_weight))
    with open(args.output, "a", encoding="utf-8") as handle:
        for row in pool.imap_unordered(_position, tasks):
            if row is None:
                continue
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if len(rows) % 50 == 0:
                print(f"{len(rows)} positions ({time.time() - started:.0f}s)", flush=True)
                print(report(rows), flush=True)
            if len(rows) >= args.positions:
                break
    print(f"\nFINAL: {len(rows)} positions, {args.samples} rollouts each (Quackle static "
          f"plays both sides); positive = Quackle's championship pick is better than ours")
    print(report(rows), flush=True)
    for worker in pool._pool:   # workers hold a JVM and Quackle processes; do not join
        worker.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
