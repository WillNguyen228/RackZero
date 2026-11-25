"""Exploration, playout cap randomization, Gumbel root search, q/z value targets,
Reanalyse, the mixed shortlist, and the Quackle warm start."""

import importlib.util
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import warnings

import numpy as np

from rackzero.agents import choose
from rackzero.dictionary import TinyLexicon
from rackzero.game import Game, Simulator
from rackzero.search import Search, SearchConfig, leave_values, shortlist
from scrabble_gym.environment import ScoredMove


HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "yaml"))
if HAS_LEARNING:
    import torch
    from rackzero.checkpoints import load_checkpoint, model_from_checkpoint
    from rackzero.config import Config, apply_overrides, classify_resume_change, config_differences
    from rackzero.features import (
        TARGET_FIELDS, belief_summary, encode_record, fill_target_defaults, record_information,
        record_moves, stack_records, to_tensors,
    )
    from rackzero.models.networks import NetworkEvaluator, build_model
    from rackzero.replay import ReplayBuffer
    from rackzero.training import Trainer, compute_losses

from rackzero.quackle import quackle_available


SMALL = {"channels": 8, "blocks": 1, "latent": 16, "rack_dim": 8, "belief_dim": 8,
         "global_dim": 4, "move_dim": 16, "policy_hidden": 16, "value_hidden": 16}
CANDIDATES = 6


def config_for(directory, **changes):
    values = {
        "name": "t", "dictionary": None, "iterations": 1, "output_root": str(directory),
        "runtime": {"device": "cpu"}, "model": SMALL,
        "search": {"simulations": 4, "depth": 1, "candidates": CANDIDATES, "particles": 4},
        "selfplay": {"games_per_iteration": 2, "workers": 0},
        "replay": {"capacity": 2000},
        "training": {"steps_per_iteration": 2, "batch_size": 8},
        "arena": {"mode": "off"},
        "evaluation": {"every": 0},
        "logging": {"hardware_interval": 0, "progress": False},
    }
    return Config.from_dict(apply_overrides(values, [f"{k}={v}" for k, v in changes.items()]))


class Quiet(unittest.TestCase):
    def setUp(self):
        self.directory = Path(".rackzero-test-" + uuid4().hex)
        self.addCleanup(lambda: shutil.rmtree(self.directory, ignore_errors=True))
        context = warnings.catch_warnings()
        context.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        self.addCleanup(context.__exit__, None, None, None)


def opening_position(seed=2):
    words = TinyLexicon().words
    game = Game(words)
    info = game.reset(seed)
    return game, info, game.legal_moves()


class LiveMoveGenerationTest(Quiet):
    def test_fast_live_moves_equal_the_java_preview_moves_for_a_whole_game(self):
        words = TinyLexicon().words
        game = Game(words, movegen="fast")
        self.addCleanup(game.close)
        rng = np.random.default_rng(0)
        for seed in (1, 4):
            info = game.reset(seed)
            for _ in range(60):
                fast = game.legal_moves()
                self.assertEqual(fast, game._env.legal_moves())
                info, terminal, truncated = game.step(choose("greedy", info, fast, rng))
                if terminal or truncated:
                    break
            self.assertTrue(terminal)
        with self.assertRaises(ValueError):
            Game(words, movegen="other")


class ShortlistTest(Quiet):
    def test_leave_values_match_the_heuristic_baseline(self):
        game, info, moves = opening_position()
        self.addCleanup(game.close)
        values = leave_values(info.observation, moves)
        rng = np.random.default_rng(0)
        best = moves[int(np.argmax([m.score + v + (0.001 * len(m.move.exchange)
                                                    if m.move.kind == "exchange" else 0)
                                    for m, v in zip(moves, values)]))].move
        self.assertEqual(best, choose("heuristic", info, moves, rng))

    def test_mixed_shortlist_keeps_top_scores_pass_and_largest_exchange(self):
        game, info, moves = opening_position()
        self.addCleanup(game.close)
        self.assertGreater(len(moves), 12)
        for limit in (6, 10):
            mixed = shortlist(moves, limit, info.observation, "mixed")
            self.assertEqual(len(mixed), limit)
            self.assertEqual(len(set(mixed)), limit)
            kinds = [m.move.kind for m in mixed]
            self.assertIn("pass", kinds)
            exchanges = [m for m in moves if m.move.kind == "exchange"]
            if exchanges:
                largest = max(len(m.move.exchange) for m in exchanges)
                self.assertIn(largest, [len(m.move.exchange) for m in mixed
                                        if m.move.kind == "exchange"])
            top = sorted(moves, key=lambda m: -m.score)[:limit // 2 - 2]
            for move in top:
                self.assertIn(move, mixed)
        self.assertEqual(shortlist(moves, 6), shortlist(moves, 6, info.observation, "score"))
        with self.assertRaises(ValueError):
            shortlist(moves, 6, None, "mixed")


@unittest.skipUnless(HAS_LEARNING, "Install the learning extra")
class GumbelTest(Quiet):
    def run_search(self, seed=4, **changes):
        game, info, moves = opening_position()
        simulator = Simulator(TinyLexicon().words)
        self.addCleanup(game.close)
        self.addCleanup(simulator.close)
        torch.manual_seed(0)
        evaluator = NetworkEvaluator(build_model(SMALL))
        config = SearchConfig(**{"simulations": 24, "depth": 1, "candidates": 12,
                                 "root_selection": "gumbel", "gumbel_actions": 8, **changes})
        return Search(simulator, evaluator, config, np.random.default_rng(seed)).run(info, moves)

    def test_budget_policy_and_selection(self):
        result = self.run_search()
        self.assertEqual(int(result.visits.sum()), 24)
        self.assertLessEqual(int(np.count_nonzero(result.visits)), 8)
        self.assertAlmostEqual(float(result.policy.sum()), 1.0, places=6)
        self.assertTrue(np.all(np.isfinite(result.policy)) and np.all(result.policy > 0))
        self.assertIsNotNone(result.selected)
        self.assertGreater(result.visits[result.selected], 0)
        # Sequential halving: the selected action survived every phase, so it has
        # at least as many visits as any other action.
        self.assertEqual(result.visits[result.selected], result.visits.max())

    def test_reproducible_and_noise_free_mode(self):
        a, b = self.run_search(seed=9), self.run_search(seed=9)
        np.testing.assert_array_equal(a.visits, b.visits)
        np.testing.assert_allclose(a.policy, b.policy)
        self.assertEqual(a.selected, b.selected)
        deterministic = self.run_search(gumbel_scale=0.0)
        self.assertAlmostEqual(float(deterministic.policy.sum()), 1.0, places=6)

    def test_puct_results_have_no_selected_action(self):
        result = self.run_search(root_selection="puct")
        self.assertIsNone(result.selected)
        np.testing.assert_allclose(result.policy, result.visits / result.visits.sum())


@unittest.skipUnless(HAS_LEARNING, "Install the learning extra")
class TargetsTest(Quiet):
    def batch(self, n=8, k=CANDIDATES):
        rng = np.random.default_rng(0)
        mask = np.ones((n, k), dtype=bool)
        policy = rng.dirichlet(np.ones(k), size=n).astype(np.float32)
        win = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        return {"cand_mask": torch.from_numpy(mask), "policy": torch.from_numpy(policy),
                "win": torch.from_numpy(win), "margin": torch.from_numpy(win * 30),
                "search_value": torch.from_numpy(rng.uniform(-1, 1, n).astype(np.float32))}

    def out(self, n=8, k=CANDIDATES):
        torch.manual_seed(1)
        return {"logits": torch.randn(n, k), "value": torch.tanh(torch.randn(n)),
                "margin": torch.randn(n), "wdl_logits": torch.randn(n, 3)}

    def test_policy_weight_masks_fast_search_positions(self):
        training = config_for(self.directory).training
        batch, out = self.batch(), self.out()
        weight = torch.tensor([1, 0, 1, 0, 1, 1, 0, 1], dtype=torch.float32)
        weighted = compute_losses(out, dict(batch, policy_weight=weight), training, "scalar")
        keep = weight.bool()
        subset = compute_losses({k: v[keep] for k, v in out.items()},
                                {k: v[keep] for k, v in batch.items()}, training, "scalar")
        self.assertAlmostEqual(float(weighted["policy"]), float(subset["policy"]), places=5)
        ones = compute_losses(out, dict(batch, policy_weight=torch.ones(8)), training, "scalar")
        plain = compute_losses(out, batch, training, "scalar")
        self.assertAlmostEqual(float(ones["policy"]), float(plain["policy"]), places=6)

    def test_value_target_mixes_outcome_and_search_value(self):
        batch, out = self.batch(), self.out()
        for head in ("scalar", "wdl"):
            zero = config_for(self.directory).training
            half = config_for(self.directory, **{"training.value_target_q_weight": 0.5}).training
            base = compute_losses(out, batch, zero, head)["value"]
            same = compute_losses(out, dict(batch, search_value=batch["win"]), half, head)["value"]
            self.assertAlmostEqual(float(base), float(same), places=5)
        half = config_for(self.directory, **{"training.value_target_q_weight": 0.5}).training
        mixed = compute_losses(out, batch, half, "scalar")["value"]
        target = 0.5 * batch["win"] + 0.5 * batch["search_value"]
        self.assertAlmostEqual(float(mixed), float((out["value"] - target).square().mean()),
                               places=5)

    def test_old_records_and_replay_states_get_neutral_defaults(self):
        old = {"policy": np.ones((3, CANDIDATES), np.float32) / CANDIDATES,
               "win": np.array([1, -1, 0], np.float32), "margin": np.zeros(3, np.float32)}
        filled = fill_target_defaults(old)
        np.testing.assert_array_equal(filled["search_value"], old["win"])
        np.testing.assert_array_equal(filled["policy_weight"], np.ones(3))
        buffer = ReplayBuffer(10, CANDIDATES)
        state = buffer.state_dict()
        records = {k: v for k, v in state["records"].items() if k not in
                   ("search_value", "policy_weight")}
        state["records"] = records
        ReplayBuffer(10, CANDIDATES).load_state_dict(state)
        self.assertEqual(set(TARGET_FIELDS), {"policy", "win", "margin", "search_value",
                                              "policy_weight"})


@unittest.skipUnless(HAS_LEARNING, "Install the learning extra")
class SelfPlayTest(Quiet):
    def self_play(self, **changes):
        trainer = Trainer(config_for(self.directory, **changes),
                          run_dir=self.directory / uuid4().hex)
        try:
            stats = trainer.self_play()
        finally:
            trainer.close()
        return stats, trainer.replay.ordered()

    def test_default_records_are_full_searches_with_search_values(self):
        stats, replay = self.self_play()
        self.assertEqual(stats["full_search_fraction"], 1.0)
        np.testing.assert_array_equal(replay["policy_weight"], 1.0)
        self.assertTrue(np.all(np.abs(replay["search_value"]) <= 1))

    def test_playout_cap_randomization_marks_fast_searches(self):
        stats, replay = self.self_play(**{"selfplay.full_search_probability": 0.5,
                                          "selfplay.fast_simulations": 1,
                                          "selfplay.full_simulations": 6})
        self.assertEqual(stats["simulations_per_move"], 6)
        weights = set(np.unique(replay["policy_weight"]).tolist())
        self.assertEqual(weights, {0.0, 1.0})
        self.assertAlmostEqual(stats["full_search_fraction"],
                               float(replay["policy_weight"].mean()), places=6)
        with self.assertRaises(ValueError):
            config_for(self.directory, **{"selfplay.full_search_probability": 0.5})

    def test_gumbel_self_play(self):
        stats, replay = self.self_play(**{"search.root_selection": "gumbel",
                                          "search.gumbel_actions": 4})
        self.assertGreater(stats["new_positions"], 5)
        np.testing.assert_allclose(replay["policy"].sum(1), 1.0, rtol=1e-5)


@unittest.skipUnless(HAS_LEARNING, "Install the learning extra")
class ReanalyseTest(Quiet):
    def records(self):
        game, info, _ = opening_position(5)
        self.addCleanup(game.close)
        records = []
        for _ in range(6):
            moves = shortlist(game.legal_moves(), CANDIDATES, info.observation, "mixed")
            record = encode_record(info, moves, belief_summary(info, game.belief), CANDIDATES)
            records.append((record, moves))
            info, terminal, _ = game.step(max(moves, key=lambda m: m.score).move)
            if terminal:
                break
        return records

    def test_records_decode_to_the_same_moves_and_inputs(self):
        for record, moves in self.records():
            self.assertEqual(record_moves(record), tuple(moves))
            info, decoded = record_information(record)
            again = encode_record(info, decoded, record["belief"], CANDIDATES)
            for name, array in record.items():
                np.testing.assert_array_equal(again[name], array, err_msg=name)

    def test_reanalyse_replaces_policy_targets(self):
        config = config_for(self.directory, **{"reanalyse.fraction": 0.5,
                                               "training.min_replay": 1})
        trainer = Trainer(config, run_dir=self.directory / "r")
        try:
            trainer.run()
        finally:
            trainer.close()
        record = trainer.history[-1]
        self.assertIn("reanalyse", record)
        self.assertGreater(record["reanalyse"]["positions"], 0)
        replay = trainer.replay.ordered()
        np.testing.assert_allclose(replay["policy"].sum(1), 1.0, rtol=1e-5)


@unittest.skipUnless(HAS_LEARNING, "Install the learning extra")
class ConfigTest(Quiet):
    def test_settings_added_later_do_not_block_resume(self):
        new = config_for(self.directory).to_dict()
        old = {k: (dict(v) if isinstance(v, dict) else v) for k, v in new.items()}
        for key in ("shortlist", "root_selection", "gumbel_actions", "gumbel_scale"):
            old["search"].pop(key)
        old["selfplay"].pop("full_search_probability")
        old["training"].pop("value_target_q_weight")
        old.pop("reanalyse")
        old.pop("init_from")
        self.assertEqual(config_differences(old, new), {})
        changed = config_for(self.directory, **{"search.root_selection": "gumbel"}).to_dict()
        self.assertIn("search.root_selection", config_differences(old, changed))
        self.assertEqual(classify_resume_change("init_from"), "neutral")

    def test_gpu_trained_checkpoint_evaluates_on_cpu(self):
        from rackzero.evaluation import main as evaluate_main

        trainer = Trainer(config_for(self.directory), run_dir=self.directory / "g")
        payload = trainer.checkpoint(include_replay=False)
        trainer.close()
        payload["config"]["runtime"].update(device="cuda", precision="bf16")
        path = self.directory / "gpu.pt"
        torch.save(payload, path)
        code = evaluate_main(["--checkpoint", str(path), "--device", "cpu", "--games", "2",
                              "--opponents", "greedy"])
        self.assertEqual(code, 0)

    def test_init_from_copies_weights(self):
        source = Trainer(config_for(self.directory), run_dir=self.directory / "a")
        path = self.directory / "a" / "source.pt"
        torch.save(source.checkpoint(include_replay=False), path)
        source.close()
        target = Trainer(config_for(self.directory, init_from=str(path).replace("\\", "/")),
                         run_dir=self.directory / "b")
        target.close()
        for key, value in source.model.state_dict().items():
            torch.testing.assert_close(target.model.state_dict()[key], value)
        bigger = dict(SMALL, channels=12)
        with self.assertRaisesRegex(ValueError, "architecture"):
            Trainer(config_for(self.directory, init_from=str(path).replace("\\", "/"),
                               **{"model.channels": bigger["channels"]}),
                    run_dir=self.directory / "c")


@unittest.skipUnless(HAS_LEARNING and quackle_available(), "Quackle is not built")
class QuackleTest(Quiet):
    def test_engine_ranks_legal_moves(self):
        from rackzero.quackle import QuackleEngine, move_key

        game, info, moves = opening_position(3)
        self.addCleanup(game.close)
        board = info.observation["board"]
        self.assertEqual(len({move_key(board, m.move) for m in moves}), len(moves))
        engine = QuackleEngine(TinyLexicon().words, top=5)
        self.addCleanup(engine.close)
        ranked = engine.rank_legal(info.observation, moves)
        self.assertGreaterEqual(len(ranked), 1)
        equities = [e for _, e in ranked]
        self.assertEqual(equities, sorted(equities, reverse=True))
        self.assertTrue(all(0 <= i < len(moves) for i, _ in ranked))
        for _ in range(3):
            self.assertEqual(engine.choose(info.observation, moves), ranked[0][0])

    def test_teacher_dataset_pretrain_and_evaluation(self):
        from rackzero.evaluation import evaluate_model
        from rackzero.warmstart import generate, load_dataset, pretrain

        config = config_for(self.directory, **{"search.shortlist": "mixed"})
        data_path = self.directory / "teacher.npz"
        meta = generate(config, games=2, workers=0, output=data_path, progress=False)
        self.assertEqual(meta["natural_games"], 2)
        data, _ = load_dataset([str(data_path)], TinyLexicon().fingerprint, CANDIDATES)
        np.testing.assert_allclose(data["policy"].sum(1), 1.0, rtol=1e-5)
        self.assertTrue(np.all(data["policy"][~data["cand_mask"]] == 0))
        np.testing.assert_array_equal(data["search_value"], data["win"])
        config.name = "t-pretrain"
        path = pretrain(config, [str(data_path)], epochs=1, batch_size=8, learning_rate=1e-3,
                        validation=0.5, run_dir=self.directory / "pre")
        model = model_from_checkpoint(load_checkpoint(path))
        result = evaluate_model(TinyLexicon().words, NetworkEvaluator(model), games=2,
                                opponent="quackle", search_config=config.search)
        self.assertEqual(result["attempted_games"], 2)
        trainer = Trainer(config_for(self.directory, init_from=str(path).replace("\\", "/")),
                          run_dir=self.directory / "warm")
        trainer.close()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(trainer.model.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
