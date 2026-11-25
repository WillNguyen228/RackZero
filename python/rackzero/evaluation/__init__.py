"""Paired-seed, swapped-seat evaluation using only natural game outcomes.

Each seed is played twice, once with the evaluated agent in each seat. If either
game of a pair is truncated by ``max_turns``, the whole pair is discarded, so
exclusions cannot unbalance the seat comparison. Win rates are reported with
95% Wilson intervals (draws count half), and score margins with normal
intervals. Evaluation search never uses root exploration noise.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from rackzero.agents import choose
from rackzero.game import Game, Simulator
from rackzero.search import METHODS, Search, SearchConfig
from rackzero.stats import mean_interval, wilson_interval


OPPONENTS = ("random", "greedy", "heuristic", "neural", "search", "quackle")


def evaluation_search_config(search_config: SearchConfig | None, method: str | None = None):
    values = asdict(search_config or SearchConfig())
    if method is not None:
        values["method"] = method
    values["root_noise_fraction"] = 0.0
    values["gumbel_scale"] = 0.0      # Gumbel root: deterministic action choice
    return SearchConfig(**values)


def _belief_input(model, info, belief):
    if hasattr(model, "belief_features"):
        return model.belief_features(info, belief)
    return belief.expected()


def play_game(words, simulator, model, baseline, opponent, config, opponent_config,
              max_turns, game_seed, seat, game_factory=None, quackle=None):
    """One evaluation game with the evaluated agent in ``seat``.

    ``quackle`` is a ``rackzero.quackle.QuackleEngine`` (opponent "quackle").

    Returns the per-game result dict, or None if the game was truncated.
    """
    seat_rngs = [np.random.default_rng(np.random.SeedSequence([game_seed, player]))
                 for player in (0, 1)]
    searches = [Search(simulator, model if player == seat else baseline,
                       config if player == seat else opponent_config, seat_rngs[player])
                for player in (0, 1)]
    belief_rng = np.random.default_rng(np.random.SeedSequence([game_seed, 2]))
    if game_factory is None:
        game = Game(words, max_turns=max_turns, particles=config.particles, rng=belief_rng,
                    movegen=config.movegen)
    else:
        game = game_factory(config.particles, belief_rng)
    regret = 0.0
    decisions = turns = 0
    try:
        info = game.reset(seed=game_seed)
        terminated = truncated = False
        for _ in range(max_turns):
            player = int(info.observation["player"])
            moves = game.legal_moves()
            if player == seat or opponent == "search":
                result = searches[player].run(info, moves=moves, belief=game.belief)
                index = (result.selected if result.selected is not None
                         else int(np.argmax(result.policy)))
                move = result.moves[index].move
            elif opponent == "quackle":
                if quackle is None:
                    raise ValueError("The quackle opponent needs a QuackleEngine")
                move = moves[quackle.choose(info.observation, moves)].move
            elif opponent == "neural":
                policy = baseline.predict(info, moves,
                                          _belief_input(baseline, info, game.belief))[0]
                move = moves[int(np.argmax(policy))].move
            else:
                move = choose(opponent, info, moves, seat_rngs[player], model=baseline)
            if player == seat:
                score = next((item.score for item in moves if item.move == move), None)
                if score is None:
                    raise ValueError("Agent returned a move outside the legal set")
                regret += float(max(item.score for item in moves) - score)
                decisions += 1
            info, terminated, truncated = game.step(move)
            turns += 1
            if terminated or truncated:
                break
        if terminated and not truncated:
            scores = np.asarray(game.scores)
            return {"seat": seat, "margin": float(scores[seat] - scores[1 - seat]),
                    "score": float(scores[seat]), "regret": regret,
                    "decisions": decisions, "turns": turns}
        return None
    finally:
        game.close()


def pair_up(games: list) -> list:
    """[(seat 0 result, seat 1 result), ...] -> play_pair-style (pair or None, truncated)."""
    pairs = []
    for first, second in games:
        truncated = (first is None) + (second is None)
        pairs.append(([first, second] if not truncated else None, truncated))
    return pairs


def play_pair(words, simulator, model, baseline, opponent, config, opponent_config,
              max_turns, game_seed, game_factory=None, quackle=None):
    """Play one seed in both seats. Returns (per-game dicts or None, truncated count)."""
    games = [play_game(words, simulator, model, baseline, opponent, config, opponent_config,
                       max_turns, game_seed, seat, game_factory, quackle) for seat in (0, 1)]
    return pair_up([games])[0]


def summarize(pairs, *, method, opponent, seed, games, seconds) -> dict:
    """Aggregate ``play_pair`` outputs (in seed order) into a report."""
    results, discarded, truncated_games = [], 0, 0
    for pair, truncated in pairs:
        truncated_games += truncated
        if pair is None:
            discarded += 2
        else:
            results.extend(pair)
    if not results:
        raise RuntimeError("No naturally completed paired evaluation games; increase max_turns")
    margins = np.array([r["margin"] for r in results])
    decisions = sum(r["decisions"] for r in results)
    wins = int(np.count_nonzero(margins > 0))
    draws = int(np.count_nonzero(margins == 0))
    low, high = wilson_interval(wins + 0.5 * draws, len(results))
    mean, margin_low, margin_high = mean_interval(margins)
    per_seat = {}
    for seat in (0, 1):
        seat_margins = np.array([r["margin"] for r in results if r["seat"] == seat])
        seat_wins = np.count_nonzero(seat_margins > 0) + 0.5 * np.count_nonzero(seat_margins == 0)
        per_seat[f"seat{seat}_win_rate"] = float(seat_wins / len(seat_margins))
        per_seat[f"seat{seat}_mean_score_difference"] = float(seat_margins.mean())
    return {
        "method": method, "opponent": opponent, "seed": seed,
        "attempted_games": games, "games": len(results),
        "discarded_games": discarded, "truncated_games": truncated_games,
        "wins": wins, "draws": draws, "losses": int(np.count_nonzero(margins < 0)),
        "win_rate": (wins + 0.5 * draws) / len(results),
        "win_rate_ci95": [low, high],
        "mean_score_difference": float(mean),
        "score_difference_ci95": [margin_low, margin_high],
        "mean_score": float(np.mean([r["score"] for r in results])) if "score" in results[0] else None,
        "mean_game_length": float(np.mean([r.get("turns", 0) for r in results])),
        **per_seat,
        "seconds": seconds, "seconds_per_game": seconds / games,
        "immediate_score_regret_proxy": (
            sum(r["regret"] for r in results) / decisions if decisions else 0.0),
        "regret_decisions": decisions,
    }


def pair_seeds(seed: int, games: int) -> list[int]:
    if isinstance(games, bool) or not isinstance(games, int) or games < 2 or games % 2:
        raise ValueError("games must be a positive even number (at least 2) for paired seats")
    return [seed + pair for pair in range(games // 2)]


def evaluate_model(
    words, model, *, games=20, seed=42, opponent="greedy",
    opponent_model=None, method="ismcts", search_config=None, max_turns=200,
    progress=False, opponent_search_config=None,
):
    """Evaluate search with ``model`` against a baseline, in this process.

    ``model``/``opponent_model`` expose ``predict``/``predict_value``/``eval``
    (a ``NetworkEvaluator`` or the original ``PolicyValueNet``). For parallel
    evaluation use ``rackzero.workers.evaluate_parallel``.
    """
    seeds = pair_seeds(seed, games)
    if opponent not in OPPONENTS:
        raise ValueError(f"Unknown opponent: {opponent}")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    config = evaluation_search_config(search_config, method)
    opponent_config = (config if opponent_search_config is None
                       else evaluation_search_config(opponent_search_config))
    baseline = model if opponent_model is None else opponent_model
    model.eval()
    baseline.eval()
    simulator = Simulator(words, max_turns=max_turns, movegen=config.movegen)
    quackle = None
    if opponent == "quackle":
        from rackzero.quackle import QuackleEngine
        quackle = QuackleEngine(words)
    started = time.perf_counter()
    pairs = []
    try:
        for index, game_seed in enumerate(seeds):
            pairs.append(play_pair(words, simulator, model, baseline, opponent, config,
                                   opponent_config, max_turns, game_seed, quackle=quackle))
            if progress:
                kept = sum(2 for pair, _ in pairs if pair is not None)
                print(f"evaluation {2 * (index + 1)}/{games}: retained={kept}, "
                      f"discarded={2 * (index + 1) - kept}", flush=True)
    finally:
        simulator.close()
        if quackle is not None:
            quackle.close()
    return summarize(pairs, method=method, opponent=opponent, seed=seed, games=games,
                     seconds=time.perf_counter() - started)


def format_result(result: dict) -> str:
    low, high = result["win_rate_ci95"]
    m_low, m_high = result["score_difference_ci95"]
    return (f"{result['method']:>22} vs {result['opponent']:<9} "
            f"win {result['win_rate']:.3f} [{low:.3f}, {high:.3f}]  "
            f"margin {result['mean_score_difference']:+.1f} [{m_low:+.1f}, {m_high:+.1f}]  "
            f"n={result['games']} len={result['mean_game_length']:.1f} "
            f"({result['seconds_per_game']:.1f}s/game)")


def main(argv=None):
    from rackzero.checkpoints import checkpoint_config, load_checkpoint, model_from_checkpoint
    from rackzero.config import apply_overrides, to_plain
    from rackzero.training import configure_runtime, lexicon_for, resolve_device
    from rackzero.workers import Executor, WorkerSpec, evaluate_parallel

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="TRUSTED checkpoint file or run directory (uses checkpoints/latest.pt)")
    parser.add_argument("--use-current", action="store_true",
                        help="evaluate the latest trained weights instead of best_model")
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS,
                        help="one or more baselines (default: greedy)")
    parser.add_argument("--opponent", choices=OPPONENTS, help="single baseline (alias)")
    parser.add_argument("--opponent-checkpoint",
                        help="TRUSTED opponent checkpoint for neural/search baselines "
                             "(default: the evaluated model itself)")
    parser.add_argument("--opponent-set", action="append", default=[], metavar="KEY=VALUE",
                        help="override the opponent's search config, e.g. "
                             "--opponent-set simulations=8 (search opponent only)")
    parser.add_argument("--games", type=int, default=20,
                        help="even count per opponent; seeds repeated with swapped seats")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--compare-methods", nargs="+", choices=METHODS,
                        help="search methods under identical seeds/seats/config")
    parser.add_argument("--seed", type=int, help="first pair seed (default: evaluation.seed)")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        help="inference device for in-process evaluation")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel evaluation processes (0 = in-process)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override the checkpoint's config, e.g. --set search.simulations=64")
    parser.add_argument("--dictionary", help="override the lexicon path (fingerprint is checked)")
    parser.add_argument("--output", help="append one JSON line per result to this file "
                        "(default for a run directory: <run>/evaluations.jsonl)")
    args = parser.parse_args(argv)
    if args.opponent and args.opponents:
        parser.error("use --opponent or --opponents, not both")
    opponents = args.opponents or [args.opponent or "greedy"]
    if args.opponent_checkpoint and not set(opponents) & {"neural", "search"}:
        parser.error("--opponent-checkpoint requires a neural or search opponent")

    if args.output is None and Path(args.checkpoint).is_dir():
        args.output = str(Path(args.checkpoint) / "evaluations.jsonl")
    checkpoint = load_checkpoint(args.checkpoint)
    values = checkpoint_config(checkpoint).to_dict()
    if args.dictionary:
        values["dictionary"] = args.dictionary
    values = apply_overrides(values, args.set)
    from rackzero.config import Config
    config = Config.from_dict(values)
    if args.device:
        config.runtime.device = args.device
    if resolve_device(config.runtime.device) == "cpu":
        # runtime.precision is training autocast; evaluation inference is fp32
        # on the CPU, so a GPU-trained (bf16) checkpoint can be evaluated there.
        config.runtime.precision = "fp32"
    configure_runtime(config)
    device = resolve_device(config.runtime.device) if args.workers == 0 else "cpu"
    lexicon = lexicon_for(config)
    if checkpoint["dictionary_fingerprint"] != lexicon.fingerprint:
        raise ValueError("Checkpoint dictionary fingerprint does not match")
    if "quackle" in opponents:
        from rackzero.quackle import lexicon_dawg
        lexicon_dawg(lexicon.words)   # build once here, not concurrently in every worker
    which = "model" if args.use_current else "best_model"
    model = model_from_checkpoint(checkpoint, which)
    opponent_model = None
    if args.opponent_checkpoint:
        previous = load_checkpoint(args.opponent_checkpoint)
        if previous["dictionary_fingerprint"] != lexicon.fingerprint:
            raise ValueError("Opponent checkpoint dictionary fingerprint does not match")
        opponent_model = model_from_checkpoint(previous, "best_model")
    opponent_search = None
    if args.opponent_set:
        search_values = apply_overrides(to_plain(config.search), args.opponent_set)
        opponent_search = SearchConfig(**search_values)
    methods = args.compare_methods or [args.method or config.search.method]
    seed = config.evaluation.seed if args.seed is None else args.seed
    executor = None
    scratch = None
    try:
        if args.workers > 0:
            import tempfile
            from rackzero.models.networks import model_config_dict
            from rackzero.workers import write_model
            scratch = tempfile.TemporaryDirectory(prefix="rackzero-eval-")
            candidate_ref = write_model(Path(scratch.name), "candidate",
                                        model_config_dict(model), model.state_dict())
            opponent_ref = (write_model(Path(scratch.name), "opponent",
                                        model_config_dict(opponent_model),
                                        opponent_model.state_dict())
                            if opponent_model is not None else None)
            spec = WorkerSpec.from_config(config, lexicon.fingerprint)
            executor = Executor(spec, args.workers)
        for method in methods:
            for opponent in opponents:
                if executor is not None:
                    def report(done, total, opponent=opponent):
                        print(f"evaluation vs {opponent}: {done}/{total} games", flush=True)
                    result = evaluate_parallel(
                        executor, config, candidate_ref, opponent=opponent, games=args.games,
                        seed=seed, method=method, opponent_model=opponent_ref,
                        opponent_search=opponent_search, progress=report)
                else:
                    from rackzero.models.networks import NetworkEvaluator
                    evaluator = NetworkEvaluator(model.to(device), device)
                    baseline = (NetworkEvaluator(opponent_model.to(device), device)
                                if opponent_model is not None else None)
                    result = evaluate_model(
                        lexicon.words, evaluator, games=args.games, seed=seed,
                        opponent=opponent, method=method, opponent_model=baseline,
                        search_config=config.search, opponent_search_config=opponent_search,
                        max_turns=config.max_turns, progress=True)
                result["checkpoint"] = str(args.checkpoint)
                result["which"] = which
                result["iteration"] = checkpoint["iteration"]
                result["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                if args.output:
                    path = Path(args.output)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result, allow_nan=False) + "\n")
                print(json.dumps(result, allow_nan=False), flush=True)
                print(format_result(result), flush=True)
    except KeyboardInterrupt:
        print("Evaluation interrupted; incomplete results are not reported as success.", flush=True)
        return 130
    finally:
        if executor is not None:
            executor.close()
        if scratch is not None:
            scratch.cleanup()
    return 0
