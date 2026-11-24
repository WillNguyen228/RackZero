"""Train/evaluate an off-policy Double DQN against the fixed native Quackle policy."""

import argparse
from collections import deque
from collections.abc import Callable
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .build import ROOT
from .dqn import SCHEMA, Candidates, DoubleDQN, Transition, dictionary_hash, encode_state
from .opponents import QuackleMatch


def run_episode(match, agent, *, seed, learner, training, batch_size=32,
                warmup=128, epsilon: float | Callable[[int], float] = 0.0,
                train_every=1, policy="dqn", sink: list | None = None, focused=0.0):
    """Play one game; `training` learns inline, `sink` only collects transitions."""
    started = time.monotonic()

    def legal(observation, moves):
        actions = Candidates.from_moves(observation, moves)
        keep = actions.top_indexes(agent.top_candidates) if policy == "dqn" else None
        if keep is None:
            return actions, moves
        return actions.subset(keep), tuple(moves[i] for i in keep)

    observation, _ = match.reset(seed=seed, learner=learner)
    actions, moves = legal(observation, match.legal_moves())
    state = encode_state(observation)
    total, losses = 0.0, []
    exploring = training or sink is not None
    while True:
        exploration = epsilon(agent.decisions) if callable(epsilon) else epsilon
        if policy == "greedy":
            placements = np.flatnonzero(actions.kinds == 0)
            exchanges = np.flatnonzero(actions.kinds == 1)
            if len(placements):
                index = int(placements[actions.scores[placements].argmax()])
            elif len(exchanges):
                index = int(exchanges[actions.exchange[exchanges].sum(axis=1).argmax()])
            else:
                index = int(np.flatnonzero(actions.kinds == 2)[0])
        elif policy == "dqn":
            index = agent.choose(state, actions, exploration if exploring else 0,
                                 focused=focused)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        observation, reward, terminated, truncated, info = match.step(moves[index].move)
        next_state = encode_state(observation)
        next_actions, next_moves = (None, ()) if terminated else legal(
            observation, match.env.legal_moves())
        if exploring:
            transition = Transition(state, actions.take(index), reward, next_state,
                                    next_actions, terminated)
            if sink is not None:
                sink.append(transition)
            else:
                agent.replay.add(transition)
                agent.decisions += 1
                if agent.decisions % train_every == 0 and len(agent.replay) >= max(batch_size, warmup):
                    loss = agent.learn(batch_size)
                    if loss is not None:
                        losses.append(loss)
        total += reward
        if terminated or truncated:
            break
        state, actions, moves = next_state, next_actions, next_moves
    scores = info["scores"].tolist()
    return {
        "seed": seed, "learner": learner, "scores": scores,
        "score_difference": scores[learner] - scores[1 - learner],
        "win": scores[learner] > scores[1 - learner], "tie": scores[0] == scores[1],
        "return": total, "decisions": info["decisions"], "terminated": terminated,
        "truncated": truncated, "loss": sum(losses) / len(losses) if losses else None,
        "seconds": round(time.monotonic() - started, 3),
    }


def summarize(results):
    completed = [r for r in results if r["terminated"]]
    count = len(completed)
    wins = sum(r["win"] for r in completed)
    rate = wins / count if count else None
    interval = None
    if count:
        z = 1.96
        denominator = 1 + z * z / count
        center = (rate + z * z / (2 * count)) / denominator
        half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
        interval = [max(0, center - half), min(1, center + half)]
    differences = [r["score_difference"] for r in completed]
    return {
        "episodes": len(results), "completed_games": count,
        "truncated_games": len(results) - count, "wins": wins,
        "ties": sum(r["tie"] for r in completed), "win_rate": rate,
        "win_rate_95ci": interval,
        "mean_score_difference": statistics.mean(differences) if differences else None,
        "score_difference_standard_error": (
            statistics.stdev(differences) / math.sqrt(count) if count > 1 else None
        ),
        "by_seat": {
            str(seat): {
                "games": sum(r["learner"] == seat for r in completed),
                "wins": sum(r["win"] and r["learner"] == seat for r in completed),
            } for seat in (0, 1)
        },
    }


_ACTOR = {}


def _start_actor(max_decisions, word_list, seed):
    torch.set_num_threads(1)
    _ACTOR["match"] = QuackleMatch(max_decisions=max_decisions, word_list=word_list)
    _ACTOR["agent"] = DoubleDQN(seed=seed, replay_capacity=1, replay_mb=1)


def _act(task):
    agent, match = _ACTOR["agent"], _ACTOR["match"]
    if task["weights"] is not None:
        agent.online.load_state_dict(task["weights"])
    agent.rng.seed(task["rng_seed"])
    agent.top_candidates = task["top_candidates"]
    sink = [] if task["collect"] else None
    result = run_episode(match, agent, seed=task["seed"], learner=task["learner"],
                         training=False, epsilon=task["epsilon"], policy=task["policy"],
                         sink=sink, focused=task["focused"])
    return result, sink or []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=10000, help="Learner decisions")
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--target-interval", type=int, default=250)
    parser.add_argument("--replay-capacity", type=int, default=5000)
    parser.add_argument("--replay-mb", type=int, default=384)
    parser.add_argument("--train-every", type=int, default=2, help="Learner decisions per optimizer step")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Episodes between saves/exports")
    parser.add_argument("--time-limit-seconds", type=float,
                        help="Stop and save after a complete episode when this budget expires")
    parser.add_argument("--word-list", type=Path,
                        help="Local word-list text or single-text-file ZIP (e.g. WOW24/NWL2023)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel actor processes (each runs its own Java engine and Quackle)")
    parser.add_argument("--top-candidates", type=int, default=64,
                        help="Highest-scoring legal moves (plus pass) the DQN acts over and "
                             "bootstraps from; a checkpoint keeps its own value")
    parser.add_argument("--focused-exploration", type=float, default=0.8,
                        help="Share of exploratory moves sampled from the 10 highest-scoring moves")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", type=Path, help="Evaluate a checkpoint without learning")
    parser.add_argument("--baseline", choices=("untrained", "greedy"),
                        help="Evaluate a fresh untrained network or immediate-score Greedy")
    parser.add_argument("--baseline-network-seed", type=int, default=42,
                        help="Initial weights for --baseline untrained; match the training seed")
    args = parser.parse_args()
    if min(args.episodes, args.max_decisions, args.batch_size, args.warmup,
           args.epsilon_decay, args.target_interval, args.replay_capacity,
           args.replay_mb, args.threads, args.train_every, args.checkpoint_every,
           args.workers, args.top_candidates) < 1:
        parser.error("Counts and limits must be positive")
    if not 0 <= args.focused_exploration <= 1:
        parser.error("--focused-exploration must be from 0 to 1")
    if (not 0 <= args.epsilon_end <= args.epsilon_start <= 1
            or not 0 <= args.gamma <= 1 or not 0 < args.learning_rate < float("inf")):
        parser.error("Invalid epsilon, gamma or learning rate")
    if sum(bool(option) for option in (args.resume, args.evaluate, args.baseline)) > 1:
        parser.error("Choose --resume, --evaluate or --baseline, not more than one")
    if args.seed < 0 or args.baseline_network_seed < 0:
        parser.error("Seeds must be nonnegative")
    if args.time_limit_seconds is not None and (
            not math.isfinite(args.time_limit_seconds) or args.time_limit_seconds <= 0):
        parser.error("--time-limit-seconds must be finite and positive")
    if args.replay_capacity < max(args.batch_size, args.warmup):
        parser.error("--replay-capacity must hold at least --batch-size and --warmup")
    torch.set_num_threads(args.threads)
    digest = dictionary_hash(args.word_list)
    checkpoint = args.resume or args.evaluate
    args.output = args.output or (checkpoint.parent if checkpoint else ROOT / "target" / "dqn")
    if not checkpoint and not args.baseline and (args.output / "checkpoint.pt").exists():
        parser.error("Output already contains a checkpoint; use --resume or choose a new --output directory")
    options = dict(device=args.device, replay_capacity=args.replay_capacity, replay_mb=args.replay_mb)
    agent = (DoubleDQN.load(checkpoint, digest, **options) if checkpoint else DoubleDQN(
        seed=args.baseline_network_seed if args.baseline else args.seed,
        learning_rate=args.learning_rate, gamma=args.gamma,
        target_interval=args.target_interval, top_candidates=args.top_candidates, **options,
    ))
    training = not (args.evaluate or args.baseline)
    if args.evaluate and args.seed < agent.seed + agent.episodes and (
            args.seed + (args.episodes + 1) // 2 > agent.seed):
        parser.error("Evaluation seeds overlap training seeds; choose a held-out --seed")
    match = QuackleMatch(max_decisions=args.max_decisions, word_list=args.word_list)
    revision = match.bridge.manifest["revision"]
    if digest != match.bridge.manifest["dictionary_sha256"]:
        match.close()
        raise ValueError("Dictionary changed during initialization; restart the run")
    if agent.opponent_revision and revision != agent.opponent_revision:
        match.close()
        raise ValueError("Checkpoint was trained against another Quackle revision")
    agent.opponent_revision = revision
    log_name = "training" if training else args.baseline or "evaluation"
    log_path = args.output / f"{log_name}.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    if training:
        configuration = {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}
        configuration.update(schema=SCHEMA, dictionary_sha256=digest, opponent_revision=revision)
        with (args.output / "runs.jsonl").open("a", encoding="utf-8") as runs:
            runs.write(json.dumps(configuration, allow_nan=False) + "\n")
    results = []
    started = time.monotonic()
    policy = "greedy" if args.baseline == "greedy" else "dqn"

    def epsilon(decisions):
        return args.epsilon_end + (args.epsilon_start - args.epsilon_end) * max(
            0, 1 - decisions / args.epsilon_decay
        )

    def budget_reached():
        return (args.time_limit_seconds is not None
                and time.monotonic() - started >= args.time_limit_seconds)

    def seat_and_seed(number):
        return number % 2, agent.seed + number if training else args.seed + number // 2

    def inline_episodes():
        for episode in range(args.episodes):
            number = agent.episodes if training else episode
            learner, seed = seat_and_seed(number)
            initial_epsilon = epsilon(agent.decisions)
            result = run_episode(
                match, agent, seed=seed, learner=learner, training=training,
                batch_size=args.batch_size, warmup=args.warmup, epsilon=epsilon,
                train_every=args.train_every, policy=policy, focused=args.focused_exploration,
            )
            yield number, initial_epsilon, result
            if budget_reached():
                return

    def parallel_episodes(pool):
        # Actors play with the latest published weights while this process learns.
        pending = deque()
        submitted = 0
        first = agent.episodes if training else 0

        def submit():
            nonlocal submitted
            number = first + submitted
            learner, seed = seat_and_seed(number)
            initial_epsilon = epsilon(agent.decisions) if training else 0
            weights = None if policy == "greedy" else {
                name: value.detach().cpu().clone() for name, value in agent.online.state_dict().items()
            }
            pending.append((number, initial_epsilon, pool.apply_async(_act, ({
                "weights": weights, "seed": seed, "learner": learner,
                "epsilon": initial_epsilon, "policy": policy, "collect": training,
                "rng_seed": agent.seed * 1_000_003 + number,
                "top_candidates": agent.top_candidates, "focused": args.focused_exploration,
            },))))
            submitted += 1

        stopping = False
        while submitted < min(args.workers, args.episodes):
            submit()
        while pending:
            number, initial_epsilon, handle = pending.popleft()
            result, transitions = handle.get()
            stopping = stopping or budget_reached()
            if not stopping and submitted < args.episodes:
                submit()
            losses = []
            for transition in transitions:
                agent.replay.add(transition)
                agent.decisions += 1
                if (agent.decisions % args.train_every == 0
                        and len(agent.replay) >= max(args.batch_size, args.warmup)):
                    loss = agent.learn(args.batch_size)
                    if loss is not None:
                        losses.append(loss)
            if training:
                result["loss"] = sum(losses) / len(losses) if losses else None
            yield number, initial_epsilon, result

    pool = None
    if args.workers > 1:
        match.close()
        pool = multiprocessing.get_context("spawn").Pool(
            args.workers, _start_actor, (args.max_decisions, args.word_list, agent.seed))
        episodes = parallel_episodes(pool)
    else:
        episodes = inline_episodes()
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for number, initial_epsilon, result in episodes:
                if training:
                    agent.episodes += 1
                    if agent.episodes % args.checkpoint_every == 0:
                        agent.save(args.output / "checkpoint.pt", digest)
                        agent.export(args.output / "policy.json", digest)
                result.update(episode=number, epsilon=initial_epsilon if training else 0,
                              epsilon_after=epsilon(agent.decisions) if training else 0,
                              updates=agent.updates, replay_size=len(agent.replay),
                              policy=args.baseline or "dqn", dictionary_sha256=digest,
                              opponent_revision=revision, network_seed=agent.seed)
                line = json.dumps(result, allow_nan=False)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                results.append(result)
        if training and results and agent.episodes % args.checkpoint_every:
            agent.save(args.output / "checkpoint.pt", digest)
            agent.export(args.output / "policy.json", digest)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        else:
            match.close()
    summary = {
        "mode": "train" if training else "evaluate", "requested_episodes": args.episodes,
        "training_episodes": agent.episodes,
        "stop_reason": "time_budget" if len(results) < args.episodes else "episode_limit",
        **summarize(results), "policy": args.baseline or "dqn",
        "updates": agent.updates, "dictionary_sha256": digest, "opponent_revision": revision,
        "network_seed": agent.seed,
    }
    (args.output / f"{log_name}-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
