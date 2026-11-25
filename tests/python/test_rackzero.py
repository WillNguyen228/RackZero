import importlib.util
import unittest
import warnings

import numpy as np

from rackzero.agents import choose
from rackzero.belief import RackBelief, draw_probability, sample_tiles, unseen_tiles
from rackzero.dictionary import TinyLexicon, WordLexicon
from rackzero.game import (
    DISTRIBUTION, Game, InformationState, PublicEvent, Simulator, board_counts, counts,
)
from rackzero.search import METHODS, Search, SearchConfig, enumerate_draws
from scrabble_gym.environment import Move


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from rackzero.models import ACTION_DIM, STATE_DIM, PolicyValueNet, encode


@unittest.skipUnless(HAS_TORCH, "Install the learning extra to test RackZero networks/search")
class RackZeroTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.words = TinyLexicon().words

    def setUp(self):
        self.game = Game(self.words, particles=128)
        self.simulator = Simulator(self.words)
        self.addCleanup(self.game.close)
        self.addCleanup(self.simulator.close)
        self.info = self.game.reset(42)
        self.rng = np.random.default_rng(42)

    def assert_world_conserves(self, world):
        np.testing.assert_array_equal(
            board_counts(world.observation) + world.racks[0] + world.racks[1] + world.bag,
            DISTRIBUTION,
        )

    def test_simulator_python_scoring_matches_java_preview(self):
        # Search scores candidates in Python; the live game scores them via Java.
        info, checked = self.info, 0
        while True:
            live = self.game.legal_moves()
            world = self.simulator.determinize(info, RackBelief(info, 1, self.rng).sample())
            self.assertEqual(set(live), set(self.simulator.legal_moves(world, self.rng)))
            checked += sum(m.move.kind == "place" for m in live)
            info, terminal, truncated = self.game.step(max(live, key=lambda m: m.score).move)
            if terminal or truncated:
                break
        self.assertGreater(checked, 50)

    def test_fast_movegen_matches_reference_generator(self):
        # The fast simulator generator must return exactly the authoritative
        # Java-scored candidate tuple: same moves, same order, same scores.
        reference = Simulator(self.words, movegen="reference")
        self.addCleanup(reference.close)
        rng = np.random.default_rng(11)
        info, checked = self.info, 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for turn in range(60):
                live = self.game.legal_moves()
                world = self.simulator.determinize(info, RackBelief(info, 1, rng).sample())
                self.assertEqual(live, self.simulator.legal_moves(world, rng))
                self.assertEqual(live, reference.legal_moves(world, rng))
                checked += len(live)
                kind = "greedy" if turn % 3 else "random"
                info, terminal, truncated = self.game.step(choose(kind, info, live, rng))
                if terminal or truncated:
                    break
        self.assertGreater(checked, 200)

    def test_dictionary_fingerprint_and_validation(self):
        a = WordLexicon(frozenset(("AT", "CAT")))
        b = WordLexicon(frozenset(("CAT", "AT")))
        self.assertEqual(a.fingerprint, b.fingerprint)
        self.assertNotEqual(a.fingerprint, TinyLexicon().fingerprint)
        with self.assertRaises(ValueError):
            WordLexicon(frozenset(("a",)))

    def test_uniform_racks_have_no_replacement_and_correct_mean(self):
        pool = unseen_tiles(self.info)
        draws = np.stack([sample_tiles(pool, 7, self.rng) for _ in range(3000)])
        self.assertTrue(np.all(draws <= pool))
        np.testing.assert_array_equal(draws.sum(axis=1), 7)
        np.testing.assert_allclose(draws.mean(0), pool * 7 / pool.sum(), atol=0.07)
        with self.assertRaises(ValueError):
            sample_tiles(pool, 100, self.rng)

    def test_exact_empty_bag_inference(self):
        obs = dict(self.info.observation)
        own = obs["rack"]
        pool = DISTRIBUTION - own
        opponent = sample_tiles(pool, 7, self.rng)
        board_pool = pool - opponent
        board = np.zeros(225, dtype=np.int8)
        blanks = np.zeros(225, dtype=np.int8)
        offset = 0
        for i, count in enumerate(board_pool):
            board[offset:offset + count] = min(i + 1, 26)
            if i == 26:
                blanks[offset:offset + count] = 1
            offset += count
        obs.update(board=board.reshape(15, 15), board_blanks=blanks.reshape(15, 15), bag_size=0)
        info = InformationState(obs)
        belief = RackBelief(info, rng=self.rng)
        np.testing.assert_array_equal(belief.expected(), opponent)
        np.testing.assert_array_equal(belief.sample(), opponent)

    def test_bayesian_action_likelihood_reweights_particles(self):
        belief = RackBelief(self.info, particles=2, rng=self.rng,
                            likelihood=lambda info, event, rack: 0.9 if rack[0] else 0.1)
        without_a = np.zeros(27, dtype=np.int64)
        without_a[4] = 7
        with_a = without_a.copy()
        with_a[4] -= 1
        with_a[0] = 1
        belief.racks = np.stack((with_a, without_a))
        event = PublicEvent(1, "pass", (0,) * 27, 0, 0, 86, 7)
        belief.condition(event)
        np.testing.assert_allclose(belief.weights, [0.9, 0.1])
        self.assertAlmostEqual(belief.expected()[0], 0.9)
        belief.likelihood = lambda info, event, rack: -1
        with self.assertRaises(ValueError):
            belief.condition(event)

    def test_chance_probabilities_and_empirical_sampling(self):
        pool = np.zeros(27, dtype=np.int64)
        pool[:2] = [2, 1]
        outcomes = enumerate_draws(pool, 2)
        self.assertAlmostEqual(sum(p for _, p in outcomes), 1)
        self.assertEqual(len(outcomes), 2)
        target = np.zeros(27, dtype=np.int64)
        target[0] = 2
        self.assertAlmostEqual(draw_probability(pool, target), 1 / 3)
        frequency = np.mean([np.array_equal(sample_tiles(pool, 2, self.rng), target)
                             for _ in range(2000)])
        self.assertAlmostEqual(frequency, 1 / 3, delta=0.04)
        with self.assertRaisesRegex(ValueError, "support"):
            enumerate_draws(pool, 2, max_outcomes=1)

    def test_snapshot_matches_live_legal_moves_and_scoring(self):
        world = self.simulator.determinize(self.info, RackBelief(self.info, rng=self.rng).sample())
        actual = self.game.legal_moves()
        imagined = self.simulator.legal_moves(world, self.rng)
        self.assertEqual(actual, imagined)
        move = max(actual, key=lambda candidate: candidate.score).move
        after = self.simulator.step(world, move, self.rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            live, terminal, truncated = self.game.step(move)
        np.testing.assert_array_equal(after.observation["board"], live.observation["board"])
        np.testing.assert_array_equal(after.observation["scores"], live.observation["scores"])
        self.assertEqual(after.terminal, terminal)
        self.assertFalse(truncated)
        self.assert_world_conserves(world)
        self.assert_world_conserves(after)

    def test_exchanges_passes_and_terminal_penalties(self):
        world = self.simulator.determinize(self.info, RackBelief(self.info, rng=self.rng).sample())
        exchange = max((m for m in self.simulator.legal_moves(world, self.rng)
                        if m.move.kind == "exchange"), key=lambda m: len(m.move.exchange))
        world = self.simulator.step(world, exchange.move, self.rng)
        self.assertEqual(world.observation["bag_size"], 86)
        self.assert_world_conserves(world)
        for _ in range(5):
            world = self.simulator.step(world, Move(), self.rng)
            self.assert_world_conserves(world)
        self.assertTrue(world.terminal)
        self.assertTrue(np.all(world.observation["scores"] <= 0))
        with self.assertRaises(ValueError):
            self.simulator.step(world, Move(), self.rng)

    def test_policy_value_shapes_gradient_and_candidate_permutation(self):
        moves = self.game.legal_moves()[:8]
        model = PolicyValueNet(hidden=16)
        state, actions = encode(self.info, moves)
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(actions.shape, (len(moves), ACTION_DIM))
        logits, value, margin = model(torch.from_numpy(state), torch.from_numpy(actions))
        loss = logits.log_softmax(0).neg().mean() + value.square() + margin.square()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        p1, v1, m1 = model.predict(self.info, moves)
        p2, v2, m2 = model.predict(self.info, moves[::-1])
        np.testing.assert_allclose(p1, p2[::-1], atol=1e-7)
        self.assertEqual(v1, v2)
        self.assertEqual(m1, m2)
        self.assertAlmostEqual(float(p1.sum()), 1, delta=1e-6)

    def test_information_boundary_rejects_secrets_and_copies_arrays(self):
        with self.assertRaisesRegex(ValueError, "non-observable"):
            InformationState(dict(self.info.observation, opponent_rack=np.ones(27)))
        obs = self.game._env._observation()
        snapshot = InformationState(obs)
        obs["rack"][:] = 0
        self.assertEqual(snapshot.observation["rack"].sum(), 7)
        with self.assertRaises(ValueError):
            snapshot.observation["rack"][:] = 0

    def test_public_exchange_history_does_not_reveal_returned_tiles(self):
        exchange = next(m.move for m in self.game.legal_moves() if m.move.kind == "exchange")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            info, _, _ = self.game.step(exchange)
        event = info.history[-1]
        self.assertEqual(event.exchange_count, len(exchange.exchange))
        self.assertIsNone(event.move)
        self.assertEqual(sum(event.played), 0)

    def test_determinization_rejects_invalid_hidden_tile_counts(self):
        rack = RackBelief(self.info, rng=self.rng).sample().astype(np.float64)
        rack[0] += 0.5
        with self.assertRaisesRegex(ValueError, "integer"):
            self.simulator.determinize(self.info, rack)
        rack[0] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.simulator.determinize(self.info, rack)

    def test_search_does_not_depend_on_live_hidden_state(self):
        model = PolicyValueNet(hidden=16)
        moves = self.game.legal_moves()
        config = SearchConfig(simulations=4, depth=2, candidates=4, particles=8)
        first = Search(self.simulator, model, config, np.random.default_rng(8)).run(self.info, moves)
        # Replace the entire live engine with a different deal; the same information
        # supplied to an agent must produce exactly the same policy.
        self.game.reset(99)
        second = Search(self.simulator, model, config, np.random.default_rng(8)).run(self.info, moves)
        np.testing.assert_array_equal(first.visits, second.visits)
        np.testing.assert_array_equal(first.policy, second.policy)
        self.assertEqual(first.value, second.value)

    def test_observer_random_streams_are_isolated(self):
        expected = self.game._beliefs[0].sample()
        self.game.reset(42)
        self.game._beliefs[1].rng.random(100)
        np.testing.assert_array_equal(self.game._beliefs[0].sample(), expected)

    def test_all_search_methods_spend_exact_rollout_budget(self):
        model = PolicyValueNet(hidden=16)
        moves = self.game.legal_moves()
        for method in METHODS:
            with self.subTest(method=method):
                result = Search(self.simulator, model,
                                SearchConfig(method=method, simulations=5, depth=2, candidates=4),
                                np.random.default_rng(3)).run(self.info, moves)
                self.assertEqual(result.visits.sum(), 5)
                self.assertEqual(len(result.moves), 4)
                self.assertAlmostEqual(float(result.policy.sum()), 1)
                self.assertTrue(np.all(np.isfinite(result.policy)))
                self.assertTrue(any(m.move.kind == "pass" for m in result.moves))
                self.assertGreater(result.chance_outcomes, 0)

    def test_particle_transitions_conserve_tiles_for_both_observers(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(8):
                info = self.game.information()
                move = choose("greedy", info, self.game.legal_moves(), self.rng)
                _, terminal, _ = self.game.step(move)
                for belief in self.game._beliefs:
                    self.assertTrue(np.all(belief.racks <= unseen_tiles(belief.info)))
                    np.testing.assert_array_equal(
                        belief.racks.sum(1), belief.info.observation["opponent_rack_size"])
                if terminal:
                    break

    def assert_belief_conserves(self, belief):
        self.assertTrue(np.all(belief.racks <= unseen_tiles(belief.info)))
        np.testing.assert_array_equal(
            belief.racks.sum(1), belief.info.observation["opponent_rack_size"])

    def test_own_exchange_returns_tiles_to_bag_not_opponent_rack(self):
        exchange = max((m.move for m in self.game.legal_moves() if m.move.kind == "exchange"),
                       key=lambda m: len(m.exchange))
        returned = counts(exchange.exchange)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.game.step(exchange)
        exchanger, observer = self.game._beliefs
        self.assert_belief_conserves(exchanger)
        # Draw-before-return: the returned tiles are unseen again, but they were in the
        # bag when the opponent last drew, so no hypothesis may place them on its rack.
        allowed = unseen_tiles(exchanger.info) - returned
        self.assertTrue(np.all(exchanger.racks <= allowed))
        self.assertTrue(np.all(exchanger.expected() <= allowed + 1e-12))
        # The other observer only sees the exchange count and keeps its belief consistent.
        self.assert_belief_conserves(observer)

    def test_pass_leaves_rack_hypotheses_unchanged(self):
        def snapshot(belief):
            return ([[(t.copy(), m) for t, m in groups] for groups in belief.particles],
                    belief.weights.copy())

        old = [snapshot(b) for b in self.game._beliefs]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.game.step(Move())
        for (particles, weights), belief in zip(old, self.game._beliefs):
            self.assert_belief_conserves(belief)
            np.testing.assert_array_equal(weights, belief.weights)
            self.assertEqual(len(particles), len(belief.particles))
            for before, after in zip(particles, belief.particles):
                self.assertEqual(len(before), len(after))
                for (t0, m0), (t1, m1) in zip(before, after):
                    np.testing.assert_array_equal(t0, t1)
                    self.assertEqual(m0, m1)

    def test_opponent_placements_update_without_depletion(self):
        played = 0
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for _ in range(10):
                if self.game.information().observation["bag_size"] < 14:
                    break
                move = choose("greedy", self.game.information(), self.game.legal_moves(),
                              self.rng)
                _, terminal, _ = self.game.step(move)
                played = max(played, int(sum(self.game.history[-1].played)))
                for belief in self.game._beliefs:
                    self.assert_belief_conserves(belief)
                if terminal:
                    break
        self.assertGreaterEqual(played, 3)

    def test_network_inputs_invariant_to_true_opponent_rack(self):
        model = PolicyValueNet(hidden=16)
        config = SearchConfig(method="belief", simulations=4, depth=2, candidates=4,
                              particles=8)
        for _ in range(4):
            self.game.step(choose("greedy", self.game.information(), self.game.legal_moves(),
                                  self.rng))

        def snapshot():
            info, moves = self.game.information(), self.game.legal_moves()
            mean = self.game.belief.expected()
            state, actions = encode(info, moves, belief_mean=mean)
            search = Search(self.simulator, model, config,
                            np.random.default_rng(5)).run(info, moves, belief=self.game.belief)
            return state, actions, search.visits, search.policy

        player = self.game.information().observation["player"]
        true_opponent = counts("".join(str(t.letter()) for t in self.game._env._rack(1 - player)))
        rng_state = self.game.belief.rng.bit_generator.state
        first = snapshot()
        # Swap the live hidden opponent rack for another tile-conserving one, keeping
        # the acting player's information and belief state identical.
        info = self.game.information()
        pool = unseen_tiles(info)
        while True:
            replacement = sample_tiles(pool, int(true_opponent.sum()), self.rng)
            if not np.array_equal(replacement, true_opponent):
                break
        world = self.simulator.determinize(info, replacement)
        self.game._env._game = self.simulator._environment(world, self.rng)._game
        self.game._env._moves = None
        swapped_opponent = counts(
            "".join(str(t.letter()) for t in self.game._env._rack(1 - player)))
        np.testing.assert_array_equal(swapped_opponent, replacement)
        self.game.belief.rng.bit_generator.state = rng_state
        second = snapshot()
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
