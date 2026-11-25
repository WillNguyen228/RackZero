import copy
import importlib.util
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
import warnings

import numpy as np
from scrabble_gym.environment import Move, ScoredMove


HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "yaml"))
if HAS_LEARNING:
    import torch
    from rackzero.agents import choose
    from rackzero.config import (
        Config, apply_overrides, classify_resume_change, load_config, migrate_flat,
    )
    from rackzero.dictionary import TinyLexicon
    from rackzero.evaluation import evaluate_model
    from rackzero.features import belief_summary, encode_record, stack_records
    from rackzero.game import Game
    from rackzero.replay import ReplayBuffer
    from rackzero.stats import wilson_interval
    from rackzero.training import Trainer, atomic_checkpoint, load_checkpoint, lr_factor


CANDIDATES = 6


def tiny_config(directory, **changes):
    values = {
        "name": "test", "dictionary": None, "iterations": 1, "output_root": str(directory),
        "runtime": {"device": "cpu"},
        "model": {"channels": 8, "blocks": 1, "latent": 16, "rack_dim": 8, "belief_dim": 8,
                  "global_dim": 4, "move_dim": 16, "policy_hidden": 16, "value_hidden": 16},
        "search": {"simulations": 2, "depth": 1, "candidates": CANDIDATES, "particles": 4},
        "selfplay": {"games_per_iteration": 1, "workers": 0},
        "replay": {"capacity": 64},
        "training": {"steps_per_iteration": 2, "batch_size": 4},
        "arena": {"mode": "gate", "games": 2},
        "evaluation": {"every": 1, "games": 2, "opponents": ["greedy"]},
        "logging": {"hardware_interval": 0, "progress": False},
    }
    return Config.from_dict(apply_overrides(values, [f"{k}={v}" for k, v in changes.items()]))


_RECORDS = None


def real_records():
    """Encoded records from a short greedy TinyLexicon game (cached)."""
    global _RECORDS
    if _RECORDS is None:
        game = Game(TinyLexicon().words, particles=4)
        info = game.reset(3)
        rng = np.random.default_rng(0)
        records = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(12):
                moves = game.legal_moves()[:CANDIDATES]
                record = encode_record(info, moves, belief_summary(info, game.belief), CANDIDATES)
                record["policy"] = np.zeros(CANDIDATES, dtype=np.float32)
                record["policy"][:len(moves)] = 1 / len(moves)
                record["win"] = np.float32(1.0)
                record["margin"] = np.float32(30.0)
                records.append(record)
                info, terminal, _ = game.step(choose("greedy", info, game.legal_moves(), rng))
                if terminal:
                    break
        game.close()
        _RECORDS = records
    return [copy.deepcopy(r) for r in _RECORDS]


def fake_batch(rng, count=2):
    records = real_records()
    picks = [records[int(i)] for i in rng.integers(0, len(records), count)]
    for record in picks:
        record["win"] = np.float32(rng.choice((-1.0, 1.0)))
    return stack_records(picks)


@unittest.skipUnless(HAS_LEARNING, "Install the learning extra to test RackZero training")
class TrainingCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(".rackzero-test-" + uuid4().hex)
        self.addCleanup(lambda: shutil.rmtree(self.directory, ignore_errors=True))
        self.config = tiny_config(self.directory)

    def trainer(self, config=None, name="run", **kwargs):
        trainer = Trainer(config or self.config, run_dir=self.directory / name, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def test_round_trip_restores_model_optimizer_scheduler_replay_history_and_all_rngs(self):
        trainer = self.trainer()
        trainer.replay.add(fake_batch(trainer.rng, 8))
        trainer.optimize()
        trainer.iteration = 1
        trainer.history = [{"iteration": 1, "promoted": True}]
        trainer.counters["selfplay_games"] = 3
        trainer.best_model = trainer._cpu_state()
        path = self.directory / "checkpoint.pt"
        atomic_checkpoint(path, trainer.checkpoint())
        expected = (random.random(), np.random.random(), trainer.rng.random(), torch.rand(3))
        resumed = self.trainer(Config.from_dict(apply_overrides(self.config.to_dict(),
                                                                ["iterations=2"])),
                               name="resumed", resume=path)
        actual = (random.random(), np.random.random(), resumed.rng.random(), torch.rand(3))
        self.assertEqual(expected[:3], actual[:3])
        torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)
        self.assertEqual(resumed.iteration, 1)
        self.assertEqual(resumed.history, trainer.history)
        self.assertEqual(resumed.counters["selfplay_games"], 3)
        self.assertEqual(resumed.config.iterations, 2)
        self.assertEqual(len(resumed.replay), len(trainer.replay))
        for name, array in trainer.replay.ordered().items():
            np.testing.assert_array_equal(resumed.replay.ordered()[name], array)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(resumed.best_model[key], value.cpu(), rtol=0, atol=0)
        self.assertEqual(resumed.scheduler.state_dict()["last_epoch"],
                         trainer.scheduler.state_dict()["last_epoch"])
        expected_optimizer = trainer.optimizer.state_dict()
        actual_optimizer = resumed.optimizer.state_dict()
        self.assertEqual(expected_optimizer["param_groups"], actual_optimizer["param_groups"])
        for key, state in expected_optimizer["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(actual_optimizer["state"][key][name], value)

    def test_atomic_save_failure_keeps_previous_checkpoint(self):
        path = self.directory / "checkpoint.pt"
        trainer = self.trainer()
        atomic_checkpoint(path, trainer.checkpoint())
        original = path.read_bytes()
        with patch("rackzero.training.torch.save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                atomic_checkpoint(path, trainer.checkpoint())
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(path.with_name(path.name + ".partial").exists())

    def test_gate_rejection_restores_model_and_matching_optimizer(self):
        trainer = self.trainer()
        trainer.replay.add(fake_batch(trainer.rng, 8))
        trainer.optimize()
        before_model = copy.deepcopy(trainer.model.state_dict())
        before_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
        result = {"win_rate": 0.0, "wins": 0, "draws": 0, "games": 2,
                  "win_rate_ci95": [0, 1], "mean_score_difference": 0.0}
        with (
            patch.object(trainer, "self_play", return_value={
                "games": 1, "new_positions": 0, "games_per_hour": 1.0,
                "decision_seconds_mean": 0.0}),
            patch("rackzero.training.evaluate_parallel", return_value=result),
        ):
            trainer.run()
        saved = load_checkpoint(self.directory / "run")
        self.assertFalse(saved["history"][-1]["promoted"])
        for key, value in before_model.items():
            torch.testing.assert_close(saved["model"][key], value, rtol=0, atol=0)
        for key, state in before_optimizer["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(saved["optimizer"]["state"][key][name], value)

    def test_monitor_mode_always_accepts_the_candidate(self):
        config = tiny_config(self.directory, **{"arena.mode": "monitor"})
        trainer = self.trainer(config)
        trainer.replay.add(fake_batch(trainer.rng, 8))
        result = {"win_rate": 0.0, "wins": 0, "draws": 0, "games": 2,
                  "win_rate_ci95": [0, 1], "mean_score_difference": 0.0}
        with (
            patch.object(trainer, "self_play", return_value={
                "games": 1, "new_positions": 0, "games_per_hour": 1.0,
                "decision_seconds_mean": 0.0}),
            patch("rackzero.training.evaluate_parallel", return_value=result),
        ):
            trainer.run()
        self.assertTrue(trainer.history[-1]["promoted"])
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(trainer.best_model[key], value.cpu(), rtol=0, atol=0)

    def test_dictionary_mismatch_rejected(self):
        trainer = self.trainer()
        checkpoint = trainer.checkpoint()
        checkpoint["dictionary_fingerprint"] = "wrong"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            trainer.restore(checkpoint)

    def test_resumed_iteration_matches_uninterrupted_updates(self):
        def fake_self_play(trainer):
            trainer.replay.add(fake_batch(trainer.rng, 3))
            return {"games": 1, "new_positions": 3, "games_per_hour": 1.0,
                    "decision_seconds_mean": 0.0}

        result = {"win_rate": 1.0, "wins": 2, "draws": 0, "games": 2,
                  "win_rate_ci95": [0, 1], "mean_score_difference": 1.0}
        two = Config.from_dict(apply_overrides(self.config.to_dict(), ["iterations=2"]))
        with (
            patch.object(Trainer, "self_play", fake_self_play),
            patch("rackzero.training.evaluate_parallel", return_value=result),
        ):
            continuous = self.trainer(two, name="continuous")
            continuous.run()
            first = self.trainer(name="split")
            first.run()
            resumed = self.trainer(two, name="split", resume=self.directory / "split")
            resumed.run()
        for key, value in continuous.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual([item["loss"] for item in continuous.history],
                         [item["loss"] for item in resumed.history])
        self.assertEqual(continuous.rng.random(), resumed.rng.random())

    def test_resume_rejects_trajectory_changes_unless_allowed(self):
        trainer = self.trainer()
        trainer.replay.add(fake_batch(trainer.rng, 2))
        atomic_checkpoint(self.directory / "c.pt", trainer.checkpoint())
        changed = tiny_config(self.directory, **{"training.learning_rate": 0.01})
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            self.trainer(changed, name="b", resume=self.directory / "c.pt")
        resumed = self.trainer(changed, name="c", resume=self.directory / "c.pt",
                               allow_config_change=True)
        self.assertAlmostEqual(resumed.optimizer.param_groups[0]["lr"], 0.01)
        # Worker count changes are trajectory-neutral (per-game seeds).
        neutral = tiny_config(self.directory, **{"selfplay.workers": 3})
        self.trainer(neutral, name="d", resume=self.directory / "c.pt")
        self.assertEqual(classify_resume_change("selfplay.workers"), "neutral")
        self.assertEqual(classify_resume_change("search.simulations"), "trajectory")
        self.assertEqual(classify_resume_change("runtime.precision"), "numeric")

    def test_checkpoint_without_replay_cannot_resume(self):
        trainer = self.trainer()
        atomic_checkpoint(self.directory / "light.pt", dict(trainer.checkpoint(), replay=None))
        with self.assertRaisesRegex(ValueError, "replay"):
            self.trainer(name="x", resume=self.directory / "light.pt")

    def test_new_run_refuses_to_overwrite_existing_run(self):
        trainer = self.trainer()
        atomic_checkpoint(trainer.run_dir / "checkpoints" / "latest.pt", trainer.checkpoint())
        with self.assertRaisesRegex(ValueError, "already holds a run"):
            Trainer(self.config, run_dir=trainer.run_dir)

    def test_self_play_truncation_never_enters_replay(self):
        from rackzero.workers import ModelRef, SelfPlayTask, play_selfplay_game

        candidate = ScoredMove(Move(), 0)
        info = SimpleNamespace(observation={"player": 0}, history=())
        game = SimpleNamespace(
            reset=lambda seed, rng=None: info, legal_moves=lambda: (candidate,),
            step=lambda move: (info, True, True), close=lambda: None,
            scores=np.array([100, 0]), belief=None, _beliefs=[],
        )
        evaluator = SimpleNamespace(belief_features=lambda info, belief: np.zeros(54),
                                    stats=lambda: {})
        ctx = SimpleNamespace(game=game, simulator=SimpleNamespace(stats={}),
                              evaluator=lambda ref: evaluator)
        search = SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(
            moves=(candidate,), policy=np.ones(1)))
        task = SelfPlayTask(1, ModelRef("unused", "t"), {"candidates": CANDIDATES},
                            CANDIDATES, 200)
        with (
            patch("rackzero.workers.Search", return_value=search),
            patch("rackzero.workers.encode_record", return_value={}),
        ):
            result = play_selfplay_game(ctx, task)
        self.assertIsNone(result["records"])
        self.assertEqual(result["stats"]["truncated_games"], 1)
        self.assertEqual(result["stats"]["positions"], 0)
        trainer = self.trainer()
        executor = SimpleNamespace(map=lambda *args, **kwargs: [result])
        with patch.object(trainer, "_ensure_executor", return_value=executor):
            with self.assertRaisesRegex(RuntimeError, "naturally completed"):
                trainer.self_play()
        self.assertEqual(len(trainer.replay), 0)

    def test_evaluation_discards_whole_pair_if_one_game_truncates(self):
        trainer = self.trainer()
        candidate = ScoredMove(Move(), 0)
        info = SimpleNamespace(observation={"player": 0})
        instances = []

        def game_factory(*args, **kwargs):
            number = len(instances)
            game = SimpleNamespace(
                reset=lambda seed: info, legal_moves=lambda: (candidate,),
                step=lambda move: (info, True, number == 0),
                close=lambda: None, scores=np.array([1, 0]),
                belief=SimpleNamespace(expected=lambda: np.zeros(27)),
            )
            instances.append(game)
            return game

        search = SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(
            moves=(candidate,), policy=np.ones(1),
        ))
        with (
            patch("rackzero.evaluation.Game", side_effect=game_factory),
            patch("rackzero.evaluation.Simulator"),
            patch("rackzero.evaluation.Search", return_value=search),
        ):
            result = evaluate_model(frozenset(), trainer.model, games=4)
        self.assertEqual(result["games"], 2)
        self.assertEqual(result["discarded_games"], 2)
        self.assertEqual(result["truncated_games"], 1)
        self.assertEqual(result["win_rate"], 0.5)
        low, high = result["win_rate_ci95"]
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_unknown_configuration_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown configuration"):
            Config.from_dict({"iteratons": 3})
        with self.assertRaisesRegex(ValueError, "Unknown search"):
            Config.from_dict({"search": {"simulation": 2}})
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            Config.from_dict({"model": {"chanels": 2}})


@unittest.skipUnless(HAS_LEARNING, "Install the learning extra to test RackZero training")
class ConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(".rackzero-test-" + uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(lambda: shutil.rmtree(self.directory, ignore_errors=True))

    def test_extends_merges_and_cli_overrides_win(self):
        (self.directory / "base.yaml").write_text(
            "name: base\nmodel:\n  channels: 32\n  blocks: 2\narena:\n  mode: off\n",
            encoding="utf-8")
        (self.directory / "child.yaml").write_text(
            "extends: base.yaml\nname: child\nmodel:\n  blocks: 3\n", encoding="utf-8")
        config = load_config(self.directory / "child.yaml",
                             ["model.channels=48", "training.learning_rate=3e-4"])
        self.assertEqual((config.name, config.model.channels, config.model.blocks),
                         ("child", 48, 3))
        self.assertEqual(config.training.learning_rate, 3e-4)
        self.assertEqual(config.arena.mode, "off")  # YAML `off` is boolean false

    def test_flat_schema1_config_migrates_to_legacy_baseline(self):
        config = load_config(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")
        self.assertEqual(config.model.arch, "legacy_mlp")
        self.assertEqual(config.model.legacy_hidden, 128)
        self.assertEqual(config.training.steps_per_iteration, 64)
        self.assertEqual((config.arena.mode, config.arena.games, config.arena.threshold),
                         ("gate", 8, 0.55))
        self.assertEqual(migrate_flat({"hidden": 16})["model"]["legacy_hidden"], 16)

    def test_shipped_configs_validate(self):
        root = Path(__file__).resolve().parents[2] / "configs"
        for name in ("smoke", "small", "medium", "large", "cpu", "gpu", "default"):
            with self.subTest(config=name):
                load_config(root / f"{name}.yaml")

    def test_lr_schedules(self):
        config = tiny_config(self.directory, **{"training.lr_schedule": "cosine",
                                                "training.warmup_steps": 10,
                                                "training.lr_min_factor": 0.1})
        factor = lr_factor(config.training, 110)
        self.assertAlmostEqual(factor(0), 0.1)
        self.assertAlmostEqual(factor(10), 1.0)
        self.assertAlmostEqual(factor(110), 0.1)
        step = tiny_config(self.directory, **{"training.lr_schedule": "step",
                                              "training.lr_step_size": 5,
                                              "training.lr_gamma": 0.5})
        self.assertAlmostEqual(lr_factor(step.training, 100)(12), 0.25)

    def test_replay_ratio_sets_training_steps(self):
        config = tiny_config(self.directory, **{"training.samples_per_position": 8,
                                                "training.batch_size": 4})
        trainer = Trainer(config, run_dir=self.directory / "ratio")
        self.addCleanup(trainer.close)
        self.assertEqual(trainer.training_steps(10), 20)   # ceil(8 * 10 / 4)
        self.assertEqual(trainer.training_steps(0), 1)
        trainer.replay.add(fake_batch(trainer.rng, 4))
        stats = trainer.optimize(new_positions=3)          # ceil(8 * 3 / 4) = 6
        self.assertEqual(stats["steps"], 6)
        fixed = Trainer(tiny_config(self.directory), run_dir=self.directory / "fixed")
        self.addCleanup(fixed.close)
        self.assertEqual(fixed.training_steps(1000), fixed.config.training.steps_per_iteration)

    def test_replay_ring_wraps_and_round_trips(self):
        records = stack_records(real_records()[:5])
        replay = ReplayBuffer(3, CANDIDATES)
        replay.add({k: v[:2] for k, v in records.items()})
        replay.add({k: v[2:5] for k, v in records.items()})
        self.assertEqual((len(replay), replay.total_added), (3, 5))
        np.testing.assert_array_equal(replay.ordered()["board"], records["board"][2:5])
        clone = ReplayBuffer(3, CANDIDATES)
        clone.load_state_dict(replay.state_dict())
        for name, array in replay.ordered().items():
            np.testing.assert_array_equal(clone.ordered()[name], array)
        sample = replay.sample(np.random.default_rng(0), 7)
        self.assertEqual(sample["cand_mask"].shape, (7, CANDIDATES))
        with self.assertRaisesRegex(ValueError, "candidate width"):
            ReplayBuffer(3, CANDIDATES + 1).load_state_dict(replay.state_dict())

    def test_wilson_interval(self):
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
