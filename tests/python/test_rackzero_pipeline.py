"""Network, feature, hidden-information and parallel self-play regression tests."""

import importlib.util
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import warnings

import numpy as np

from rackzero.agents import choose
from rackzero.belief import RackBelief, sample_tiles, unseen_tiles
from rackzero.dictionary import TinyLexicon
from rackzero.game import Game, InformationState, Simulator, counts
from rackzero.search import Search, SearchConfig
from scrabble_gym.environment import Move


HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "yaml"))
if HAS_LEARNING:
    import torch
    from rackzero.config import apply_overrides, Config
    from rackzero.evaluation import evaluation_search_config
    from rackzero.features import (
        BELIEF_DIM, belief_summary, encode_record, stack_records, to_tensors,
    )
    from rackzero.models import encode
    from rackzero.models.networks import (
        LegacyNet, ModelConfig, NetworkEvaluator, build_model, premium_planes,
    )
    from rackzero.training import Trainer
    from rackzero.workers import SelfPlayTask, temperature_at


SMALL = {"channels": 8, "blocks": 2, "latent": 16, "rack_dim": 8, "belief_dim": 8,
         "global_dim": 4, "move_dim": 16, "policy_hidden": 16, "value_hidden": 16}


def quiet():
    context = warnings.catch_warnings()
    context.__enter__()
    warnings.simplefilter("ignore", RuntimeWarning)
    return context


def true_rack(game, player):
    return counts("".join(str(t.letter()) for t in game._env._rack(player)))


@unittest.skipUnless(HAS_LEARNING, "Install the learning extra to test RackZero networks")
class NetworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        cls.words = TinyLexicon().words

    def setUp(self):
        self.game = Game(self.words, particles=32)
        self.addCleanup(self.game.close)
        self.info = self.game.reset(5)
        self.rng = np.random.default_rng(0)
        self._quiet = quiet()
        self.addCleanup(self._quiet.__exit__, None, None, None)

    def positions(self, turns=10):
        info = self.info
        for _ in range(turns):
            yield info, self.game.legal_moves(), self.game.belief
            info, terminal, _ = self.game.step(choose("greedy", info, self.game.legal_moves(),
                                                      self.rng))
            if terminal:
                return

    def test_legacy_record_features_equal_original_encoder(self):
        for info, moves, belief in self.positions():
            moves = moves[:12]
            state, actions = encode(info, moves, belief_mean=belief.expected())
            record = encode_record(info, moves, belief_summary(info, belief), 16)
            batch = to_tensors(stack_records([record]), "cpu")
            np.testing.assert_allclose(LegacyNet.state_features(batch)[0].numpy(), state,
                                       atol=1e-6)
            np.testing.assert_allclose(
                LegacyNet.action_features(batch)[0, :len(moves)].numpy(), actions, atol=1e-6)

    def test_premium_layout_matches_standard_board_counts(self):
        planes = premium_planes()
        self.assertEqual(planes.shape, (4, 15, 15))
        # Double letter, triple letter, double word (incl. centre), triple word.
        self.assertEqual([int(p.sum()) for p in planes], [24, 12, 17, 8])
        np.testing.assert_array_equal(planes, planes[:, ::-1, :])
        np.testing.assert_array_equal(planes, planes.transpose(0, 2, 1))

    def test_shapes_masking_permutation_and_gradients(self):
        for arch, value_head in (("resnet", "scalar"), ("cnn", "wdl"), ("legacy_mlp", "scalar")):
            with self.subTest(arch=arch):
                model = build_model(dict(SMALL, arch=arch, value_head=value_head,
                                         legacy_hidden=16)).eval()
                info, moves = self.info, self.game.legal_moves()[:7]
                records = [encode_record(info, moves, belief_summary(info), 10),
                           encode_record(info, moves[::-1], belief_summary(info), 10)]
                batch = to_tensors(stack_records(records), "cpu")
                out = model(batch)
                self.assertEqual(out["logits"].shape, (2, 10))
                self.assertEqual(out["value"].shape, (2,))
                self.assertTrue(torch.all(torch.isinf(out["logits"][:, 7:])))
                probs = out["logits"].softmax(-1)
                self.assertTrue(torch.all(probs[:, 7:] == 0))
                torch.testing.assert_close(probs[0, :7], probs[1, :7].flip(0),
                                           atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(out["value"][0], out["value"][1])
                model.train()
                out = model(batch)
                loss = (out["logits"].log_softmax(-1)[:, :7].mean() + out["value"].sum()
                        + out["margin"].sum())
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.requires_grad))

    def test_attention_policy_layer_respects_padding(self):
        model = build_model(dict(SMALL, policy_attention_layers=1)).eval()
        moves = self.game.legal_moves()[:5]
        narrow = encode_record(self.info, moves, belief_summary(self.info), 5)
        wide = encode_record(self.info, moves, belief_summary(self.info), 9)
        a = model(to_tensors(stack_records([narrow]), "cpu"))["logits"][0]
        b = model(to_tensors(stack_records([wide]), "cpu"))["logits"][0, :5]
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_evaluator_scores_exactly_the_supplied_legal_moves(self):
        evaluator = NetworkEvaluator(build_model(SMALL))
        moves = self.game.legal_moves()
        probs, value, _ = evaluator.predict(self.info, moves)
        self.assertEqual(probs.shape, (len(moves),))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertTrue(-1 <= value <= 1)
        with self.assertRaises(ValueError):
            encode_record(self.info, (), belief_summary(self.info), 4)

    def test_model_config_validation(self):
        with self.assertRaises(ValueError):
            ModelConfig(arch="transformer")
        with self.assertRaises(ValueError):
            ModelConfig(policy_attention_layers=1, move_dim=10, attention_heads=4)


@unittest.skipUnless(HAS_LEARNING, "Install the learning extra to test hidden information")
class HiddenInformationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.words = TinyLexicon().words

    def setUp(self):
        self.game = Game(self.words, particles=32)
        self.simulator = Simulator(self.words)
        self.addCleanup(self.game.close)
        self.addCleanup(self.simulator.close)
        self.game.reset(7)
        self.rng = np.random.default_rng(1)
        self._quiet = quiet()
        self.addCleanup(self._quiet.__exit__, None, None, None)

    def advance(self, turns):
        for _ in range(turns):
            info = self.game.information()
            _, terminal, _ = self.game.step(choose("greedy", info, self.game.legal_moves(),
                                                   self.rng))
            if terminal:
                break

    def test_records_evaluator_and_search_ignore_the_true_opponent_rack(self):
        evaluator = NetworkEvaluator(build_model(SMALL))
        config = SearchConfig(method="belief", simulations=4, depth=2, candidates=6, particles=8)
        self.advance(4)

        def snapshot():
            info, moves = self.game.information(), self.game.legal_moves()
            belief = self.game.belief
            record = encode_record(info, moves[:6], evaluator.belief_features(info, belief), 6)
            probs = evaluator.predict(info, moves, evaluator.belief_features(info, belief))[0]
            result = Search(self.simulator, evaluator, config,
                            np.random.default_rng(5)).run(info, moves, belief=belief)
            return [*record.values(), probs, result.visits, result.policy]

        player = self.game.information().observation["player"]
        actual = true_rack(self.game, 1 - player)
        rng_state = self.game.belief.rng.bit_generator.state
        first = snapshot()
        info = self.game.information()
        while True:
            replacement = sample_tiles(unseen_tiles(info), int(actual.sum()), self.rng)
            if not np.array_equal(replacement, actual):
                break
        world = self.simulator.determinize(info, replacement)
        self.game._env._game = self.simulator._environment(world, self.rng)._game
        self.game._env._moves = None
        np.testing.assert_array_equal(true_rack(self.game, 1 - player), replacement)
        self.game.belief.rng.bit_generator.state = rng_state
        for a, b in zip(first, snapshot()):
            np.testing.assert_array_equal(a, b)

    def test_search_never_touches_the_live_engine(self):
        class Forbidden:
            def __getattr__(self, name):
                raise AssertionError(f"search accessed the live engine ({name})")

        self.advance(3)
        info, moves, belief = self.game.information(), self.game.legal_moves(), self.game.belief
        evaluator = NetworkEvaluator(build_model(SMALL))
        live, self.game._env = self.game._env, Forbidden()
        try:
            for method in ("ismcts", "belief", "pimc", "stochastic", "neural_determinization"):
                Search(self.simulator, evaluator,
                       SearchConfig(method=method, simulations=3, depth=2, candidates=6),
                       np.random.default_rng(0)).run(info, moves, belief=belief)
        finally:
            self.game._env = live

    def test_observation_has_only_public_fields_and_own_rack(self):
        info = self.game.information()
        self.assertEqual(set(info.observation),
                         {"board", "board_blanks", "rack", "scores", "player",
                          "opponent_rack_size", "bag_size", "scoreless_turns", "turn"})
        player = info.observation["player"]
        np.testing.assert_array_equal(info.observation["rack"], true_rack(self.game, player))
        with self.assertRaisesRegex(ValueError, "non-observable"):
            InformationState(dict(info.observation, opponent_rack=true_rack(self.game, 1)))

    def test_opponent_exchange_conserves_tiles_and_reveals_only_its_size(self):
        exchanger = self.game.information().observation["player"]
        observer = 1 - exchanger
        before = unseen_tiles(self.game._beliefs[observer].info)
        exchange = max((m.move for m in self.game.legal_moves() if m.move.kind == "exchange"),
                       key=lambda move: len(move.exchange))
        info, _, _ = self.game.step(exchange)
        belief = self.game._beliefs[observer]
        # The observer's unseen pool is unchanged; returned tiles stay unseen.
        np.testing.assert_array_equal(unseen_tiles(belief.info), before)
        self.assertEqual(info.history[-1].exchange_count, len(exchange.exchange))
        self.assertEqual(sum(info.history[-1].played), 0)
        self.assertIsNone(info.history[-1].move)
        racks = belief.racks
        self.assertTrue(np.all(racks <= before))
        np.testing.assert_array_equal(racks.sum(1), belief.info.observation["opponent_rack_size"])

    def test_empty_bag_belief_is_exact_in_real_games(self):
        for seed in range(30):
            info = self.game.reset(seed)
            while info.observation["bag_size"] > 0:
                info, terminal, truncated = self.game.step(
                    choose("greedy", info, self.game.legal_moves(), self.rng))
                if terminal or truncated:
                    break
            else:
                player = info.observation["player"]
                actual = true_rack(self.game, 1 - player)
                np.testing.assert_allclose(self.game.belief.expected(), actual)
                features = belief_summary(info, self.game.belief)
                np.testing.assert_allclose(features[:27], actual)
                np.testing.assert_allclose(features[27:], (actual > 0).astype(float))
                np.testing.assert_allclose(belief_summary(info)[27:], (actual > 0).astype(float))
                return
        self.skipTest("No sampled game reached an empty bag")

    def test_uniform_belief_summary_matches_monte_carlo(self):
        info = self.game.information()
        summary = belief_summary(info)
        self.assertEqual(summary.shape, (BELIEF_DIM,))
        pool = unseen_tiles(info)
        draws = np.stack([sample_tiles(pool, 7, self.rng) for _ in range(4000)])
        np.testing.assert_allclose(summary[:27], draws.mean(0), atol=0.06)
        np.testing.assert_allclose(summary[27:], (draws > 0).mean(0), atol=0.04)
        rich = belief_summary(info, RackBelief(info, 16, self.rng))
        np.testing.assert_allclose(rich, summary, atol=1e-9)


@unittest.skipUnless(HAS_LEARNING, "Install the learning extra to test self-play")
class SearchAndSelfPlayTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(".rackzero-test-" + uuid4().hex)
        self.addCleanup(lambda: shutil.rmtree(self.directory, ignore_errors=True))
        self._quiet = quiet()
        self.addCleanup(self._quiet.__exit__, None, None, None)

    def test_root_noise_is_selfplay_only_and_off_by_default(self):
        words = TinyLexicon().words
        game, simulator = Game(words), Simulator(words)
        self.addCleanup(game.close)
        self.addCleanup(simulator.close)
        info = game.reset(2)
        moves = game.legal_moves()
        evaluator = NetworkEvaluator(build_model(SMALL))

        def visits(**noise):
            config = SearchConfig(simulations=24, depth=1, candidates=8, **noise)
            return Search(simulator, evaluator, config, np.random.default_rng(4)).run(
                info, moves).visits

        np.testing.assert_array_equal(visits(), visits(root_noise_fraction=0.0))
        self.assertFalse(np.array_equal(
            visits(), visits(root_noise_alpha=0.3, root_noise_fraction=0.5)))
        noisy = SearchConfig(root_noise_alpha=0.3, root_noise_fraction=0.5)
        self.assertEqual(evaluation_search_config(noisy).root_noise_fraction, 0.0)
        with self.assertRaises(ValueError):
            SearchConfig(root_noise_fraction=0.5)

    def test_temperature_schedule(self):
        task = SelfPlayTask(0, None, {}, 6, 10, temperature=1.0, temperature_moves=4,
                            final_temperature=0.0)
        self.assertEqual([temperature_at(task, ply) for ply in (0, 3, 4, 9)], [1, 1, 0, 0])
        always = SelfPlayTask(0, None, {}, 6, 10)
        self.assertEqual(temperature_at(always, 150), 1.0)

    def test_parallel_workers_reproduce_inline_self_play_exactly(self):
        values = {
            "name": "w", "dictionary": None, "iterations": 1, "runtime": {"device": "cpu"},
            "model": SMALL,
            "search": {"simulations": 3, "depth": 1, "candidates": 6, "particles": 4},
            "selfplay": {"games_per_iteration": 3, "workers": 0},
            "replay": {"capacity": 1000},
            "logging": {"hardware_interval": 0, "progress": False},
        }
        replays = []
        for workers in (0, 2):
            config = Config.from_dict(apply_overrides(values, [f"selfplay.workers={workers}"]))
            trainer = Trainer(config, run_dir=self.directory / f"w{workers}")
            try:
                stats = trainer.self_play()
            finally:
                trainer.close()
            self.assertEqual(stats["games"], 3)
            replays.append(trainer.replay.ordered())
        self.assertGreater(len(replays[0]["policy"]), 10)
        for name, array in replays[0].items():
            np.testing.assert_array_equal(replays[1][name], array, err_msg=name)


if __name__ == "__main__":
    unittest.main()
