import itertools
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

import scrabble_gym
from scrabble_gym import Move, ScrabbleEnv, greedy_action
from scrabble_gym.environment import ALPHABET


PAIRS = frozenset(a + b for a, b in itertools.product(ALPHABET[:-1], repeat=2))


class PassOnly(gym.ActionWrapper):
    """Make every sampled action legal when checking the Gymnasium lifecycle."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(1)

    def action(self, action):
        return Move().to_action()


class ScrabbleEnvironmentTest(unittest.TestCase):
    def make_env(self, **kwargs):
        env = ScrabbleEnv(words=PAIRS, **kwargs)
        self.addCleanup(env.close)
        return env

    def assert_observations_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            np.testing.assert_equal(first[key], second[key], err_msg=key)

    def assert_conservation(self, env, observation):
        self.assertEqual(
            np.count_nonzero(observation["board"])
            + int(observation["rack"].sum())
            + observation["opponent_rack_size"] + observation["bag_size"],
            100,
        )
        self.assertTrue(env.observation_space.contains(observation))

    def test_seeded_reset_and_hidden_information(self):
        env = self.make_env()
        first, info = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        self.assert_observations_equal(first, second)
        self.assertEqual(first["bag_size"], 86)
        self.assertEqual(first["rack"].sum(), 7)
        self.assertEqual(first["opponent_rack_size"], 7)
        self.assertNotIn("opponent_rack", first)
        self.assertNotIn("bag", first)
        self.assertEqual(info["dictionary_size"], len(PAIRS))
        self.assert_conservation(env, first)
        first["board"][0, 0] = 26
        first["rack"][:] = 0
        self.assert_observations_equal(second, env._observation())
        third, _ = env.reset(seed=43)
        self.assertFalse(np.array_equal(second["rack"], third["rack"]))

    def test_opening_scoring_and_java_ownership(self):
        env = self.make_env()
        before, _ = env.reset(seed=42)
        moves = env.legal_moves()
        self.assertIs(moves, env.legal_moves())
        places = [item for item in moves if item.move.kind == "place"]
        self.assertTrue(places)
        self.assertEqual(len([m for m in moves if m.move.kind == "pass"]), 1)
        for item in moves:
            self.assertTrue(env.action_space.contains(item.move.to_action()))
        chosen = max(places, key=lambda item: item.score)
        obs, reward, terminated, truncated, info = env.step(chosen.move)
        self.assertGreater(reward, 0)
        self.assertEqual(reward, chosen.score)
        self.assertEqual(info["move_score"], chosen.score)
        self.assertEqual(env._game.players().get(0).score(), chosen.score)
        self.assertEqual(env._game.board().tileCount(), 2)
        self.assertEqual(obs["scores"][0] - before["scores"][0], reward)
        self.assertEqual(obs["player"], 1)
        self.assertFalse(terminated or truncated)
        self.assertTrue(obs["board"][7, 7])
        self.assert_conservation(env, obs)

    def test_invalid_moves_are_atomic_including_rng(self):
        env = self.make_env()
        control = self.make_env()
        before, _ = env.reset(seed=42)
        control.reset(seed=42)
        chosen = next(m.move for m in env.legal_moves() if m.move.kind == "place")
        disconnected = Move("place", 0, 0, chosen.direction, chosen.word, chosen.blanks)
        with self.assertRaisesRegex(ValueError, "center"):
            env.step(disconnected)
        invalid_actions = [
            Move("place", word="NOTINLEXICON"),
            Move("place", row=14, col=14, word=chosen.word),
            Move("exchange"),
            Move("exchange", exchange="ZZZZZZZ"),
            Move("place", word="AA", blanks=(0, 0)),
            Move("place", row=-1, word="AA"),
            Move("nonsense"),
        ]
        malformed = chosen.to_action()
        malformed["word"][4] = 1
        invalid_actions.extend([malformed, {"kind": 9}])
        for action in invalid_actions:
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    env.step(action)
                self.assert_observations_equal(before, env._observation())
        exchange = max(
            (m.move for m in env.legal_moves() if m.move.kind == "exchange"),
            key=lambda move: len(move.exchange),
        )
        actual = env.step(exchange)
        expected = control.step(exchange)
        self.assert_observations_equal(actual[0], expected[0])
        self.assertEqual(actual[1:4], expected[1:4])

    def test_exchange_keeps_bag_size_and_switches_player(self):
        env = self.make_env()
        before, _ = env.reset(seed=5)
        action = max(
            (m.move for m in env.legal_moves() if m.move.kind == "exchange"),
            key=lambda move: len(move.exchange),
        )
        self.assertEqual(len(action.exchange), 7)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(obs["bag_size"], before["bag_size"])
        self.assertEqual(obs["scoreless_turns"], 1)
        self.assertEqual(info["acting_player"], 0)
        self.assertEqual(reward, 0)
        self.assertFalse(terminated or truncated)
        self.assert_conservation(env, obs)

    def test_six_passes_final_scoring_and_terminal_perspective(self):
        env = self.make_env(max_turns=6)
        env.reset(seed=42)
        deductions = np.array([
            -sum(t.points() for t in env._rack(p)) for p in range(2)
        ])
        for turn in range(6):
            obs, reward, terminated, truncated, info = env.step(Move())
            self.assertEqual(obs["player"], (turn + 1) % 2)
            self.assertEqual(terminated, turn == 5)
            self.assertFalse(truncated)
        np.testing.assert_array_equal(obs["scores"], deductions)
        np.testing.assert_array_equal(info["score_adjustments"], deductions)
        self.assertEqual(reward, deductions[1])
        self.assertEqual(info["end_reason"], "six_scoreless_turns")
        self.assert_conservation(env, obs)
        for operation in (lambda: env.step(Move()), env.legal_moves):
            with self.assertRaises(RuntimeError):
                operation()
        env.reset(seed=42)
        self.assertEqual(env.step(Move())[0]["turn"], 1)

    def test_truncation_does_not_apply_final_penalties(self):
        env = self.make_env(max_turns=1)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(Move())
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["end_reason"], "turn_limit")
        self.assertEqual(reward, 0)
        self.assertIsNone(info["winner"])
        np.testing.assert_array_equal(obs["scores"], [0, 0])
        np.testing.assert_array_equal(info["score_adjustments"], [0, 0])

    def test_blank_assignment_has_zero_value(self):
        env = self.make_env()
        for seed in range(100):
            obs, _ = env.reset(seed=seed)
            if obs["rack"][-1]:
                break
        else:
            self.fail("No blank in 100 seeded deals")
        chosen = next(
            m for m in env.legal_moves()
            if m.move.kind == "place" and m.move.blanks
        )
        points = jpype_tile_points()
        expected = 2 * sum(
            points[c] for i, c in enumerate(chosen.move.word) if i not in chosen.move.blanks
        )
        obs, reward, _, _, _ = env.step(chosen.move)
        self.assertEqual(reward, expected)
        self.assertEqual(obs["board_blanks"].sum(), len(chosen.move.blanks))
        self.assert_conservation(env, obs)

    def test_bingo_is_scored_by_java(self):
        probe = self.make_env()
        probe.reset(seed=42)
        letters = [str(t.letter()) for t in probe._rack(0)]
        word = "".join("A" if c == "?" else c for c in letters)
        blanks = tuple(i for i, c in enumerate(letters) if c == "?")
        env = ScrabbleEnv(words={word})
        self.addCleanup(env.close)
        env.reset(seed=42)
        move = Move("place", 7, 4, 0, word, blanks)
        preview = env._preview(move)
        obs, reward, _, _, info = env.step(move)
        self.assertEqual(reward, preview)
        self.assertGreaterEqual(reward, 50)
        self.assertEqual(info["move_score"], preview)
        self.assertEqual(np.count_nonzero(obs["board"]), 7)
        self.assert_conservation(env, obs)

    def test_seeded_self_play_and_legal_scores(self):
        env = self.make_env(max_turns=30)
        control = self.make_env(max_turns=30)
        env.reset(seed=7)
        control.reset(seed=7)
        placements = 0
        while True:
            action = greedy_action(env)
            move = env._decode(action)
            expected_score = env._preview(move) if move.kind == "place" else 0
            actual = env.step(action)
            expected = control.step(action)
            placements += move.kind == "place"
            self.assert_observations_equal(actual[0], expected[0])
            self.assertEqual(actual[1:4], expected[1:4])
            self.assertEqual(actual[4]["move_score"], expected_score)
            self.assert_conservation(env, actual[0])
            if actual[2] or actual[3]:
                break
        self.assertGreaterEqual(placements, 2)

    def test_move_generation_matches_exhaustive_java_previews(self):
        probe = self.make_env()
        probe.reset(seed=42)
        letters = {str(t.letter()) for p in range(2) for t in probe._rack(p)} - {"?"}
        words = {a + b for a, b in itertools.product(sorted(letters)[:4], repeat=2)}
        words.update({"AT", "TO", "IT", "AS"})
        env = ScrabbleEnv(words=words)
        self.addCleanup(env.close)
        env.reset(seed=42)

        def physical_key(move, board):
            dr, dc = ((0, 1), (1, 0))[move.direction]
            return tuple(sorted(
                (move.row + dr * i, move.col + dc * i, c, i in move.blanks)
                for i, c in enumerate(move.word)
                if not board[move.row + dr * i, move.col + dc * i]
            ))

        for _ in range(2):
            obs = env._observation()
            board = obs["board"]
            expected = {}
            for word, direction, row, col in itertools.product(
                sorted(words), range(2), range(15), range(15)
            ):
                dr, dc = ((0, 1), (1, 0))[direction]
                if row + dr >= 15 or col + dc >= 15:
                    continue
                cells = ((row, col), (row + dr, col + dc))
                if not np.any(board) and (7, 7) not in cells:
                    continue
                for mask in range(4):
                    move = Move("place", row, col, direction, word,
                                tuple(i for i in range(2) if mask & (1 << i)))
                    try:
                        score = env._preview(move)
                    except (ValueError, env._InvalidMove):
                        continue
                    expected[physical_key(move, board)] = score
            actual = {
                physical_key(item.move, board): item.score
                for item in env.legal_moves() if item.move.kind == "place"
            }
            self.assertTrue(expected)
            self.assertEqual(actual, expected)
            self.assertEqual(
                len(actual),
                sum(item.move.kind == "place" for item in env.legal_moves()),
            )
            env.step(greedy_action(env))

    def test_full_java_dictionary_game_rack_out_and_exchange_gate(self):
        env = ScrabbleEnv()
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=42)
        checked_exchange_gate = False
        while True:
            player = obs["player"]
            before_scores = obs["scores"]
            opponent_points = sum(t.points() for t in env._rack(1 - player))
            if obs["bag_size"] < 7 and obs["rack"].sum():
                letter = ALPHABET[int(np.flatnonzero(obs["rack"])[0])]
                with self.assertRaisesRegex(ValueError, "seven tiles"):
                    env.step(Move("exchange", exchange=letter))
                self.assert_observations_equal(obs, env._observation())
                checked_exchange_gate = True
            obs, reward, terminated, truncated, info = env.step(greedy_action(env))
            self.assert_conservation(env, obs)
            if terminated or truncated:
                break
        self.assertTrue(checked_exchange_gate)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["end_reason"], "rack_out")
        self.assertEqual(obs["bag_size"], 0)
        self.assertEqual(len(env._rack(player)), 0)
        self.assertEqual(obs["player"], 1 - player)
        self.assertEqual(reward, info["move_score"] + opponent_points)
        self.assertEqual(obs["scores"][player],
                         before_scores[player] + info["move_score"] + opponent_points)
        self.assertEqual(obs["scores"][1 - player],
                         before_scores[1 - player] - opponent_points)
        np.testing.assert_array_equal(
            info["score_adjustments"],
            [opponent_points if p == player else -opponent_points for p in range(2)],
        )

    def test_registration_and_gymnasium_contract(self):
        self.assertIsNotNone(scrabble_gym.ScrabbleEnv)
        env = gym.make("Scrabble-v0", words=PAIRS, render_mode="ansi")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertIn("Rack:", env.render())
        check_env(PassOnly(self.make_env()), skip_render_check=True)

    def test_lifecycle_and_multiple_environments(self):
        env = self.make_env()
        for operation in (env.legal_moves, lambda: env.step(Move())):
            with self.assertRaises(RuntimeError):
                operation()
        env.reset(seed=0)
        env.close()
        other = self.make_env()
        other.reset(seed=0)
        other.step(Move())
        env.reset(seed=0)
        self.assertEqual(env.step(Move())[0]["player"], 1)

    def test_configuration_and_dictionary_errors(self):
        for kwargs in ({"max_turns": 0}, {"max_turns": 1.5},
                       {"render_mode": "human"}, {"words": []},
                       {"words": {"lower"}}, {"words": {"A"}}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ScrabbleEnv(**kwargs)
        with self.assertRaises(FileNotFoundError):
            ScrabbleEnv(jar_path="missing-scrabble-engine.jar")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.txt"
            path.write_text("# teaching lexicon\nat CAT at\n", encoding="utf-8")
            env = ScrabbleEnv(word_list=path)
            self.addCleanup(env.close)
            self.assertEqual(env.words, {"AT", "CAT"})
            with self.assertRaises(ValueError):
                ScrabbleEnv(word_list=path, words={"AT"})
            for text in ("AT invalid-word", "AT \u00df"):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid word"):
                    ScrabbleEnv(word_list=path)
        env = self.make_env()
        with self.assertRaises(ValueError):
            env.reset(options={"unsupported": True})

    def test_bundled_dictionary_loads_from_jar(self):
        env = ScrabbleEnv()
        self.addCleanup(env.close)
        self.assertGreater(len(env.words), 1000)
        self.assertIn("CAT", env.words)
        obs, _ = env.reset(seed=42)
        self.assert_conservation(env, obs)


def jpype_tile_points():
    import jpype

    tile = jpype.JClass("com.masterchiefproject.scrabble.game.Tile")
    return {c: int(tile.pointsForLetter(c)) for c in ALPHABET[:-1]}


if __name__ == "__main__":
    unittest.main()
