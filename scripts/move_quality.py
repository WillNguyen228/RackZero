"""Ground-truth move quality: do later checkpoints choose better moves?

Positions are mid-game positions (bag >= 7) from self-play games of a reference
agent. At each position every agent picks the move it would play in evaluation.
Every distinct pick, plus the rollout network's top prior moves up to
``--candidates``, is valued by ``--samples`` full-game rollouts on common sampled
worlds: the opponent rack is drawn from the mover's belief and every candidate
sees the same bag order (common random numbers). Both sides of every rollout are
played by one fixed policy (argmax prior of ``--rollout``), so all agents are
judged by the same yardstick. A move's value is its mean final spread for the
mover, in points.

Reports per agent, over positions, with standard errors:
  gain_vs_ref  mean value(agent pick) - value(--baseline agent pick)  (unbiased)
  regret       mean best pick value - value(agent pick) on contested positions (biased up by noise,
               equally for every agent)
  best_rate    fraction of contested positions where the agent has the best pick
and for Q agents, how far the Q head's predicted advantage of its pick over the
baseline's pick exceeds the measured advantage (overestimation).

  python scripts/move_quality.py --agent warm=PATH:raw --agent tdq100=PATH:q1 ...
      --reference tdq100 --baseline warm --rollout PATH --positions 400 --samples 32
Modes: raw (argmax policy prior), q1 (prior + Q, as trained), q0 (Q only).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from multiprocessing import get_context
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

_STATE = {}


def _init(agents, rollout_path):
    import torch
    torch.set_num_threads(1)
    from rackzero.checkpoints import checkpoint_config, load_checkpoint, model_from_checkpoint
    from rackzero.game import Simulator
    from rackzero.models.networks import NetworkEvaluator
    from rackzero.training import lexicon_for
    models = {}

    def evaluator(path):
        if path not in models:
            ck = load_checkpoint(path)
            models[path] = (NetworkEvaluator(model_from_checkpoint(ck, "best_model")).eval(),
                            checkpoint_config(ck))
        return models[path]

    rollout_eval, rollout_cfg = evaluator(rollout_path)
    lexicon = lexicon_for(rollout_cfg)
    _STATE.update(
        agents=[(name, *evaluator(path), mode) for name, path, mode in agents],
        rollout=rollout_eval, cfg=rollout_cfg, words=lexicon.words,
        simulator=Simulator(lexicon.words))


def _agent_pick(name, evaluator, cfg, mode, info, legal, moves, belief, rng):
    """Index into the common candidate list ``moves`` of the agent's move, and its
    predicted values in points per common index (None for the raw policy).

    ``legal`` are all legal moves; Q agents shortlist them exactly as in evaluation."""
    from rackzero.search import Search, SearchConfig
    if mode == "raw":
        probs, _, _ = evaluator.predict(info, moves, evaluator.belief_features(info, belief))
        return int(np.argmax(probs)), None
    search_cfg = replace(SearchConfig(**_plain(cfg.search)), root_selection="q",
                         q_prior_weight=1.0 if mode == "q1" else 0.0,
                         root_noise_fraction=0.0)
    search = Search(_STATE["simulator"], evaluator, search_cfg, rng)
    result = search.run(info, moves=legal, belief=belief)
    index = {m: i for i, m in enumerate(moves)}
    chosen = result.moves[int(result.selected)]
    if chosen not in index or set(result.moves) != set(moves):
        raise RuntimeError(f"{name}: shortlist differs from the common candidate list")
    q = {index[m]: 100 * float(v) for m, v in zip(result.moves, result.q)}
    return index[chosen], q


def _plain(dc):
    from dataclasses import asdict
    return asdict(dc)


def _rollout(world, root, move, seed, candidates, shortlist_kind, max_plies=80):
    from rackzero.search import shortlist
    sim, model = _STATE["simulator"], _STATE["rollout"]
    rng = np.random.default_rng(seed)
    world = sim.step(world, move, rng)
    for _ in range(max_plies):
        if world.terminal:
            break
        info = world.information()
        replies = shortlist(sim.legal_moves(world, rng), candidates, info.observation,
                            shortlist_kind)
        probs, _, _ = model.predict(info, replies, None)
        world = sim.step(world, replies[int(np.argmax(probs))].move, rng)
    scores = world.observation["scores"]
    return float(scores[root] - scores[1 - root]), bool(world.terminal)


def _position(task):
    """Play the reference agent to a random mid-game ply, then value every pick."""
    from rackzero.game import Game
    from rackzero.search import shortlist
    seed, reference, samples, k = task
    rng = np.random.default_rng(seed)
    cfg = _STATE["cfg"]
    cand, kind = cfg.search.candidates, cfg.search.shortlist
    agents = _STATE["agents"]
    ref = next(a for a in agents if a[0] == reference)
    game = Game(_STATE["words"], particles=cfg.search.particles, rng=np.random.default_rng(seed + 1),
                movegen="fast")
    try:
        info = game.reset(seed=int(seed % 2**31))
        stop = int(rng.integers(4, 31))
        for _ply in range(stop):
            legal = game.legal_moves()
            moves = shortlist(legal, cand, info.observation, kind)
            index, _ = _agent_pick(*ref, info, legal, moves, game.belief, rng)
            info, terminal, truncated = game.step(moves[index].move)
            if terminal or truncated or info.observation["bag_size"] < 7:
                return None
        legal = game.legal_moves()
        moves = shortlist(legal, cand, info.observation, kind)
        if len(moves) < 2:
            return None
        belief = game.belief
        picks, predicted = {}, {}
        for agent in agents:
            picks[agent[0]], predicted[agent[0]] = _agent_pick(*agent, info, legal, moves,
                                                               belief, rng)
        # Only the agents' distinct picks are rolled out (plus the rollout policy's
        # top prior moves when k > number of picks); if every agent agrees there is
        # nothing to compare and the position is recorded without rollouts.
        chosen = list(dict.fromkeys(picks.values()))
        if len(chosen) == 1:
            return {"seed": int(seed), "ply": stop, "values": {chosen[0]: 0.0}, "picks": picks,
                    "predicted": {}, "truncated_rollouts": 0, "agree": True}
        probs, _, _ = _STATE["rollout"].predict(info, moves, None)
        for i in np.argsort(-np.asarray(probs), kind="stable"):
            if len(chosen) >= k:
                break
            if int(i) not in chosen:
                chosen.append(int(i))
        root = int(info.observation["player"])
        totals = np.zeros(len(chosen))
        truncated = 0
        for _ in range(samples):
            world = _STATE["simulator"].determinize(info, belief.sample())
            s = int(rng.integers(0, 2**63 - 1))
            for j, i in enumerate(chosen):
                value, natural = _rollout(world, root, moves[i].move, s, cand, kind)
                totals[j] += value
                truncated += not natural
        values = {int(i): totals[j] / samples for j, i in enumerate(chosen)}
        return {
            "seed": int(seed), "ply": stop, "values": values, "picks": picks,
            "predicted": {name: (None if q is None else {int(i): q.get(int(i), float("nan"))
                                                         for i in chosen})
                          for name, q in predicted.items()},
            "truncated_rollouts": truncated, "agree": False,
        }
    finally:
        game.close()


def summarize(rows, agents, baseline):
    def stat(xs):
        xs = np.asarray(xs, dtype=np.float64)
        if len(xs) == 0:
            return float("nan"), float("nan")
        return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0

    report = {}
    for name in agents:
        gain, regret, best, over = [], [], [], []
        for r in rows:
            values = {int(k): v for k, v in r["values"].items()}
            pick, base = r["picks"][name], r["picks"][baseline]
            gain.append(values[pick] - values[base])
            if r.get("agree"):
                continue
            top = max(values.values())
            regret.append(top - values[pick])
            best.append(float(values[pick] >= top))
            pred = r["predicted"].get(name)
            if pred is not None:
                pred = {int(k): v for k, v in pred.items()}
                if np.isfinite(pred[pick]) and np.isfinite(pred.get(base, np.nan)):
                    over.append((pred[pick] - pred[base]) - (values[pick] - values[base]))
        g, gs = stat(gain)
        rg, rs = stat(regret)
        report[name] = {"gain_vs_baseline": g, "gain_se": gs, "regret": rg, "regret_se": rs,
                        "best_rate": float(np.mean(best)) if best else float("nan"),
                        "disagree_with_baseline": float(np.mean(
                            [r["picks"][name] != r["picks"][baseline] for r in rows])),
                        "q_overestimate": stat(over)[0] if over else None,
                        "q_overestimate_se": stat(over)[1] if over else None}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", action="append", required=True, metavar="NAME=PATH:MODE")
    parser.add_argument("--reference", required=True, help="agent that generates positions")
    parser.add_argument("--baseline", required=True, help="agent the gains are measured against")
    parser.add_argument("--rollout", required=True, help="checkpoint whose argmax prior plays rollouts")
    parser.add_argument("--positions", type=int, default=400)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=0,
                        help="roll out at least this many moves (agents' picks first)")
    parser.add_argument("--seed", type=int, default=11000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output", required=True, help="JSONL of per-position results")
    args = parser.parse_args()
    agents = []
    for spec in args.agent:
        name, rest = spec.split("=", 1)
        path, mode = rest.rsplit(":", 1)
        if mode not in ("raw", "q1", "q0"):
            parser.error(f"unknown mode {mode!r}")
        agents.append((name, path, mode))
    names = [a[0] for a in agents]
    if args.reference not in names or args.baseline not in names:
        parser.error("--reference and --baseline must name agents")
    tasks = [(args.seed + 7919 * i, args.reference, args.samples, args.candidates)
             for i in range(int(args.positions * 1.6) + 8)]   # some games end early
    rows, started = [], time.time()
    out = Path(args.output)
    # Workers hold a JVM, so pool teardown can hang: the summary is printed first
    # and the process then exits without joining them (see the end of main).
    pool = get_context("spawn").Pool(args.workers, _init, (agents, args.rollout))
    with out.open("a", encoding="utf-8") as handle:
        for row in pool.imap_unordered(_position, tasks):
            if row is None:
                continue
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if len(rows) % 25 == 0 or len(rows) == 1:
                print(f"{len(rows)} positions ({time.time() - started:.0f}s)", flush=True)
            if len(rows) >= args.positions:
                break
    report = summarize(rows, names, args.baseline)
    contested = sum(not r.get("agree") for r in rows)
    print(json.dumps({"positions": len(rows), "contested": contested, "samples": args.samples,
                      "truncated_rollouts": sum(r["truncated_rollouts"] for r in rows),
                      "agents": report}, indent=1))
    print(f"\nmove quality over {len(rows)} positions, {contested} where agents disagree "
          f"(points; gain vs {args.baseline} per position; regret/best among the agents' picks "
          f"on contested positions):")
    for name in names:
        r = report[name]
        over = ("" if r["q_overestimate"] is None
                else f"  Q overestimates its edge by {r['q_overestimate']:+.1f} ± {r['q_overestimate_se']:.1f}")
        print(f"  {name:>12}: gain {r['gain_vs_baseline']:+.2f} ± {r['gain_se']:.2f}  "
              f"regret {r['regret']:.2f} ± {r['regret_se']:.2f}  best {r['best_rate']:.2f}  "
              f"differs {r['disagree_with_baseline']:.2f}{over}")
    sys.stdout.flush()
    for worker in pool._pool:   # Pool.terminate() can hang on the workers' JVMs
        worker.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
