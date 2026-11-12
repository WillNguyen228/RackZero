import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from scrabble_gym import Move, ScoredMove
from scrabble_gym.build import ROOT
from scrabble_gym.opponents import QuackleMatch, native_move, public_position
from scrabble_gym.quackle_build import TARGET


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from scrabble_gym.dqn import (
        ACTION_SIZE, STATE_SIZE, Candidates, DoubleDQN, Replay, Transition,
        dictionary_hash, encode_state,
    )
    from scrabble_gym.train_dqn import run_episode


def observation(player=0):
    rack = np.zeros(27, dtype=np.int64)
    rack[[0, 19, 26]] = 1
    return {
        "board": np.zeros((15, 15), dtype=np.int8),
        "board_blanks": np.zeros((15, 15), dtype=np.int8),
        "rack": rack, "scores": np.array([10, 20]), "player": player,
        "bag_size": 86, "opponent_rack_size": 7, "scoreless_turns": 0, "turn": 0,
    }


class OpponentTransitionTest(unittest.TestCase):
    def test_native_move_blank_and_existing_tile_conversion(self):
        obs = observation()
        obs["board"][7, 7] = 1
        state = public_position(obs)
        action = {"kind": "place", "placements": [
            {"row": 7, "col": 8, "tile": {"letter": "T", "blank": True}}
        ]}
        self.assertEqual(native_move(action, state), Move("place", 7, 7, 0, "AT", (1,)))
        self.assertEqual([tile["letter"] for tile in state["rack"]], ["A", "T", "?"])

    def match(self, steps):
        match = object.__new__(QuackleMatch)
        match.learner, match.decisions, match.max_decisions, match.done = 0, 0, 2, False
        match.env = Mock()
        match.env._scores.return_value = np.array([0, 0])
        match.env.step.side_effect = steps
        match._opponent_step = Mock(side_effect=steps)
        return match

    def test_full_round_reward_and_nonterminal_learner_perspective(self):
        first = (observation(1), 10, False, False, {"scores": np.array([10, 0])})
        second = (observation(0), 30, False, False, {"scores": np.array([10, 30])})
        match = self.match(iter([first, second]))
        obs, reward, terminated, truncated, _ = match.step(Move())
        self.assertEqual(reward, -0.2)
        self.assertEqual(obs["player"], 0)
        self.assertFalse(terminated or truncated)
        match._opponent_step.assert_called_once()

    def test_learner_terminal_skips_quackle_and_counts_both_final_adjustments(self):
        final = (observation(1), 0, True, False, {"scores": np.array([50, -10])})
        match = self.match(iter([final]))
        _, reward, terminated, _, _ = match.step(Move())
        self.assertEqual(reward, 0.6)
        self.assertTrue(terminated)
        match._opponent_step.assert_not_called()
        with self.assertRaises(RuntimeError):
            match.step(Move())

    def test_opponent_terminal_and_round_limit(self):
        first = (observation(1), 10, False, False, {"scores": np.array([10, 0])})
        for terminal in (False, True):
            final = (observation(0), 0, terminal, False, {"scores": np.array([-5, 50])})
            match = self.match(iter([first, final]))
            match.max_decisions = 1
            _, reward, terminated, truncated, _ = match.step(Move())
            self.assertEqual(reward, -0.55)
            self.assertEqual(terminated, terminal)
            self.assertEqual(truncated, not terminal)


@unittest.skipUnless(HAS_TORCH, "Install scrabble-gym[learning] for DQN tests")
class DoubleDQNTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.obs = observation()
        self.moves = (
            ScoredMove(Move("place", 7, 7, 0, "AT"), 4),
            ScoredMove(Move("exchange", exchange="?"), 0), ScoredMove(Move(), 0),
        )
        self.state = encode_state(self.obs)
        self.actions = Candidates.from_moves(self.obs, self.moves)
        self.agent = DoubleDQN(seed=1, gamma=0.5, target_interval=2)

    def transition(self, terminal=False):
        return Transition(self.state.copy(), self.actions.take(0), 0.2, self.state.copy(),
                          None if terminal else self.actions, terminal)

    def test_features_are_blank_aware_and_do_not_mutate_observations(self):
        before = copy.deepcopy(self.obs)
        features = self.actions.features(slice(None))
        self.assertEqual(features.shape, (3, ACTION_SIZE))
        self.assertEqual(self.state.shape, (STATE_SIZE,))
        self.assertEqual(features[0, 112 * 6], 1)
        self.assertEqual(features[1, 1350 + 26], 0)
        self.assertEqual(features[1, 1350 + 27 + 26], 1 / 7)
        for key in before:
            np.testing.assert_equal(self.obs[key], before[key])
        blank = Candidates.from_moves(self.obs, (ScoredMove(Move("place", 7, 7, 0, "AT", (0,)), 2),))
        self.assertEqual(blank.features(slice(None))[0, 112 * 6 + 5], 1)

    def test_double_dqn_selects_online_but_evaluates_frozen_target(self):
        # Online prefers score; target gives that action a negative value.
        with torch.no_grad():
            for network in (self.agent.online, self.agent.target):
                for parameter in network.parameters():
                    parameter.zero_()
                network.layers[0].weight[0, -1] = 1
                network.layers[2].weight[0, 0] = 1
            self.agent.online.layers[4].weight[0, 0] = 1
            self.agent.target.layers[4].weight[0, 0] = -1
        targets = self.agent.bellman_targets([self.transition(), self.transition(True)])
        np.testing.assert_allclose(targets.cpu(), [0.18, 0.2], atol=1e-6)
        self.assertTrue(all(not p.requires_grad for p in self.agent.target.parameters()))

    def test_replay_is_bounded_and_samples_off_policy(self):
        item = self.transition()
        replay = Replay(capacity=2, max_bytes=item.nbytes * 2)
        for _ in range(3):
            replay.add(item)
        self.assertEqual(len(replay), 2)
        self.assertEqual(replay.bytes, 2 * item.nbytes)
        self.assertEqual(len(replay.sample(2, self.agent.rng)), 2)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            Replay(2, item.nbytes - 1).add(item)
        before = [p.clone() for p in self.agent.online.parameters()]
        self.agent.replay = replay
        frozen = [p.clone() for p in self.agent.target.parameters()]
        self.assertTrue(np.isfinite(self.agent.learn(2)))
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(before, self.agent.online.parameters())))
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(frozen, self.agent.target.parameters())))
        self.agent.learn(2)
        self.assertTrue(all(torch.equal(p, q) for p, q in
                            zip(self.agent.online.parameters(), self.agent.target.parameters())))

    def test_checkpoint_roundtrip_optimizer_rng_and_dictionary_guard(self):
        self.agent.replay.add(self.transition())
        self.agent.learn(1)
        self.agent.episodes, self.agent.decisions = 2, 8
        digest = dictionary_hash()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            self.agent.save(checkpoint, digest)
            loaded = DoubleDQN.load(checkpoint, digest)
            np.testing.assert_equal(self.agent.values(self.state, self.actions),
                                    loaded.values(self.state, self.actions))
            self.assertEqual(loaded.updates, 1)
            self.assertEqual(loaded.episodes, 2)
            self.assertEqual(loaded.decisions, 8)
            self.assertEqual(loaded.rng.getstate(), self.agent.rng.getstate())
            self.assertEqual(len(loaded.replay), 1)
            self.assertEqual(loaded.replay.bytes, self.agent.replay.bytes)
            self.assertTrue(loaded.optimizer.state_dict()["state"])
            self.agent.learn(1)
            loaded.learn(1)
            np.testing.assert_equal(self.agent.values(self.state, self.actions),
                                    loaded.values(self.state, self.actions))
            with self.assertRaisesRegex(ValueError, "dictionary"):
                DoubleDQN.load(checkpoint, "different")

    @unittest.skipUnless(shutil.which("node"), "Node required for browser parity")
    def test_python_and_browser_features_and_q_values_match(self):
        from scrabble_gym.opponents import public_position
        state = public_position(self.obs)
        state.update(scores=[10, 20], opponentRackSize=7, scorelessTurns=0)
        action = {"kind": "place", "score": 4, "placements": [
            {"row": 7, "col": 7, "rackIndex": 0, "tile": state["rack"][0]},
            {"row": 7, "col": 8, "rackIndex": 1, "tile": state["rack"][1]},
        ]}
        self.agent.episodes, self.agent.updates = 1, 1
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "policy.json"
            self.agent.export(model_path, dictionary_hash())
            payload = {"model": json.loads(model_path.read_text()), "position": state, "action": action}
            result = subprocess.run(
                ["node", "--input-type=module", "-e",
                 "import {encodeState,encodeAction,qValue,validateModel} from './docs/dqn.js';"
                 "let text='';for await(const chunk of process.stdin)text+=chunk;"
                 "const p=JSON.parse(text),s=encodeState(p.position),a=encodeAction(p.position,p.action);"
                 "validateModel(p.model);console.log(JSON.stringify({s,a,q:qValue(p.model,s,a)}));"],
                input=json.dumps(payload), text=True, capture_output=True, check=True, cwd=ROOT,
            )
        browser = json.loads(result.stdout)
        np.testing.assert_allclose(browser["s"], self.state, atol=1e-7)
        np.testing.assert_allclose(browser["a"], self.actions.features(slice(0, 1))[0], atol=1e-7)
        self.assertAlmostEqual(browser["q"], float(self.agent.values(self.state, self.actions.take(0))[0]), places=5)


@unittest.skipUnless(HAS_TORCH and (TARGET / "manifest.json").is_file(), "Build native Quackle and install learning extra")
class DQNNativeTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node required for legal-candidate parity")
    def test_browser_and_java_enumerate_the_same_legal_physical_candidates(self):
        match = QuackleMatch()
        self.addCleanup(match.close)
        script = """
import {createCandidateGenerator} from './docs/agents.js';
import {readFileSync} from 'node:fs';
let text='';for await(const chunk of process.stdin)text+=chunk;
const p=JSON.parse(text), words=new Set(readFileSync('docs/words.txt','utf8').split(/\\s+/));
const letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ?';
const keys=createCandidateGenerator(words)(p).map(a=>{
  const cells=Array(225).fill(0),leave=Array(27).fill(0),exchange=Array(27).fill(0);
  for(const t of p.rack)leave[letters.indexOf(t.letter)]++;
  if(a.kind==='place')for(const q of a.placements){
    cells[q.row*15+q.col]=letters.indexOf(q.tile.letter)+1+26*Number(q.tile.blank);
    leave[letters.indexOf(q.tile.blank?'?':q.tile.letter)]--;
  }
  if(a.kind==='exchange')for(const i of a.rackIndexes){
    const c=letters.indexOf(p.rack[i].letter);exchange[c]++;leave[c]--;
  }
  return JSON.stringify([cells,leave,exchange,['place','exchange','pass'].indexOf(a.kind),a.score||0]);
});
console.log(JSON.stringify(keys.sort()));
"""
        for learner in (0, 1):
            obs, _ = match.reset(seed=42, learner=learner)
            candidates = Candidates.from_moves(obs, match.legal_moves())
            keys = sorted(json.dumps([
                candidates.cells[i].tolist(), candidates.leave[i].tolist(),
                candidates.exchange[i].tolist(), int(candidates.kinds[i]), int(candidates.scores[i]),
            ], separators=(",", ":")) for i in range(len(candidates)))
            result = subprocess.run(
                ["node", "--input-type=module", "-e", script],
                input=json.dumps(public_position(obs)), text=True, capture_output=True,
                check=True, cwd=ROOT,
            )
            self.assertEqual(keys, json.loads(result.stdout))

    def test_both_seats_training_evaluation_and_truncation_bootstrap(self):
        torch.set_num_threads(2)
        match = QuackleMatch(max_decisions=2)
        self.addCleanup(match.close)
        agent = DoubleDQN(seed=42)
        for learner in (0, 1):
            result = run_episode(match, agent, seed=42 + learner, learner=learner,
                                 training=True, batch_size=1, warmup=1, epsilon=0.5)
            self.assertTrue(result["terminated"] or result["truncated"])
            if result["truncated"]:
                self.assertFalse(agent.replay.items[-1].terminated)
                self.assertIsNotNone(agent.replay.items[-1].next_actions)
            self.assertGreater(agent.updates, 0)
        before = (agent.updates, agent.decisions, len(agent.replay))
        weights = [p.clone() for p in agent.online.parameters()]
        run_episode(match, agent, seed=7000, learner=0, training=False)
        self.assertEqual(before, (agent.updates, agent.decisions, len(agent.replay)))
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(weights, agent.online.parameters())))


if __name__ == "__main__":
    unittest.main()
