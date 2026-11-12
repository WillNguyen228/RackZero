"""Two-player Gymnasium interface to the repository's Java engine."""

from gymnasium.envs.registration import register

from .environment import Move, ScoredMove, ScrabbleEnv, greedy_action

register(id="Scrabble-v0", entry_point="scrabble_gym:ScrabbleEnv")

__all__ = ["Move", "ScoredMove", "ScrabbleEnv", "greedy_action"]
