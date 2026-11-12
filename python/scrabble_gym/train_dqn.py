"""Train/evaluate an off-policy Double DQN against the fixed native Quackle policy."""

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import time

import torch

from .build import ROOT
from .dqn import Candidates, DoubleDQN, Transition, dictionary_hash, encode_state
from .opponents import QuackleMatch


def run_episode(match, agent, *, seed, learner, training, batch_size=32,
                warmup=128, epsilon: float | Callable[[int], float] = 0.0):
    started = time.monotonic()
    observation, _ = match.reset(seed=seed, learner=learner)
    moves = match.legal_moves()
    state, actions = encode_state(observation), Candidates.from_moves(observation, moves)
    total, losses = 0.0, []
    while True:
        exploration = epsilon(agent.decisions) if callable(epsilon) else epsilon
        index = agent.choose(state, actions, exploration if training else 0)
        observation, reward, terminated, truncated, info = match.step(moves[index].move)
        next_state = encode_state(observation)
        next_moves = () if terminated else match.env.legal_moves()
        next_actions = None if terminated else Candidates.from_moves(observation, next_moves)
        if training:
            agent.replay.add(Transition(state, actions.take(index), reward,
                                        next_state, next_actions, terminated))
            agent.decisions += 1
            if len(agent.replay) >= max(batch_size, warmup):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=10000, help="Learner decisions")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-interval", type=int, default=100)
    parser.add_argument("--replay-capacity", type=int, default=5000)
    parser.add_argument("--replay-mb", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", type=Path, help="Evaluate a checkpoint without learning")
    args = parser.parse_args()
    if min(args.episodes, args.max_decisions, args.batch_size, args.warmup,
           args.epsilon_decay, args.target_interval, args.replay_capacity,
           args.replay_mb, args.threads) < 1:
        parser.error("Counts and limits must be positive")
    if (not 0 <= args.epsilon_end <= args.epsilon_start <= 1
            or not 0 <= args.gamma <= 1 or not 0 < args.learning_rate < float("inf")):
        parser.error("Invalid epsilon, gamma or learning rate")
    if args.resume and args.evaluate:
        parser.error("Choose --resume or --evaluate, not both")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.replay_capacity < max(args.batch_size, args.warmup):
        parser.error("--replay-capacity must hold at least --batch-size and --warmup")
    torch.set_num_threads(args.threads)
    digest = dictionary_hash()
    checkpoint = args.resume or args.evaluate
    args.output = args.output or (checkpoint.parent if checkpoint else ROOT / "target" / "dqn")
    if not checkpoint and (args.output / "checkpoint.pt").exists():
        parser.error("Output already contains a checkpoint; use --resume or choose a new --output directory")
    options = dict(device=args.device, replay_capacity=args.replay_capacity, replay_mb=args.replay_mb)
    agent = (DoubleDQN.load(checkpoint, digest, **options) if checkpoint else DoubleDQN(
        seed=args.seed, learning_rate=args.learning_rate, gamma=args.gamma,
        target_interval=args.target_interval, **options,
    ))
    match = QuackleMatch(max_decisions=args.max_decisions)
    revision = match.bridge.manifest["revision"]
    if agent.opponent_revision and revision != agent.opponent_revision:
        match.close()
        raise ValueError("Checkpoint was trained against another Quackle revision")
    agent.opponent_revision = revision
    training = not args.evaluate
    log_path = args.output / ("training.jsonl" if training else "evaluation.jsonl")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for episode in range(args.episodes):
                number = agent.episodes if training else episode
                seed = (agent.seed if training else args.seed) + number
                def epsilon(decisions):
                    return args.epsilon_end + (args.epsilon_start - args.epsilon_end) * max(
                        0, 1 - decisions / args.epsilon_decay
                    )

                initial_epsilon = epsilon(agent.decisions)
                result = run_episode(
                    match, agent, seed=seed, learner=number % 2, training=training,
                    batch_size=args.batch_size, warmup=args.warmup, epsilon=epsilon,
                )
                if training:
                    agent.episodes += 1
                    agent.save(args.output / "checkpoint.pt", digest)
                    agent.export(args.output / "policy.json", digest)
                result.update(episode=number, epsilon=initial_epsilon if training else 0,
                              epsilon_after=epsilon(agent.decisions) if training else 0,
                              updates=agent.updates, replay_size=len(agent.replay))
                line = json.dumps(result, allow_nan=False)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                results.append(result)
    finally:
        match.close()
    completed = [r for r in results if r["terminated"]]
    print(json.dumps({
        "mode": "train" if training else "evaluate", "episodes": len(results),
        "completed_games": len(completed), "truncated_games": len(results) - len(completed),
        "win_rate": sum(r["win"] for r in completed) / len(completed) if completed else None,
        "mean_score_difference": sum(r["score_difference"] for r in results) / len(results),
        "updates": agent.updates,
    }), flush=True)


if __name__ == "__main__":
    main()
