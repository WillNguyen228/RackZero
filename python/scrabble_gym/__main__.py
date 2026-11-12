"""Run a reproducible greedy-versus-greedy smoke experiment."""

import argparse
import json

from .environment import ScrabbleEnv, greedy_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--word-list")
    parser.add_argument("--jar-path")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    env = ScrabbleEnv(
        word_list=args.word_list, jar_path=args.jar_path, max_turns=args.max_turns,
        render_mode="ansi" if args.render else None,
    )
    try:
        for episode in range(args.episodes):
            env.reset(seed=args.seed + episode)
            terminated = truncated = False
            turns = 0
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step(greedy_action(env))
                turns += 1
                if args.render:
                    print(env.render())
            print(json.dumps({
                "seed": args.seed + episode, "turns": turns,
                "scores": info["scores"].tolist(), "winner": info["winner"],
                "terminated": terminated, "truncated": truncated,
                "end_reason": info["end_reason"],
            }))
    finally:
        env.close()


if __name__ == "__main__":
    main()
