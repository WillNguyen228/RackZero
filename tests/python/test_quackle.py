import copy
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from scrabble_gym import Move, ScrabbleEnv
from scrabble_gym.quackle_build import TARGET, build_lexicon
from scrabble_gym.quackle_server import (
    DOCS, POINTS, QuackleBridge, QuackleServer, parse_action, validate_position,
)


def tile(c):
    return {"letter": c, "points": POINTS[c], "blank": c == "?"}


def position(rack="CAT"):
    return {"board": [[None] * 15 for _ in range(15)],
            "rack": [tile(c) for c in rack], "bagSize": 86}


class QuackleValidationTest(unittest.TestCase):
    def test_position_preserves_blank_flags(self):
        state = position("AT?")
        state["board"][7][7] = {"letter": "C", "points": 0, "blank": True}
        cells, rack, bag = validate_position(state)
        self.assertEqual(cells[112], "c")
        self.assertEqual(rack, "AT?")
        self.assertEqual(bag, 86)

    def test_rejects_bad_shapes_tiles_counts_and_hidden_fields(self):
        invalid = [
            {"board": []}, {"rack": []}, {"rack": [tile("Z")] * 7},
            {"bagSize": True}, {"bagSize": 101}, {"opponentRack": []},
            {"rack": [{"letter": "A", "points": 10, "blank": False}]},
            {"rack": [{"letter": "A", "points": 0, "blank": True}]},
            {"rack": [{"letter": "?", "points": 0, "blank": False}]},
            {"rack": [{"letter": "AA", "points": 1, "blank": False}]},
        ]
        for update in invalid:
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    validate_position({**position(), **update})
        state = position()
        state["board"][0][0] = tile("?")
        with self.assertRaises(ValueError):
            validate_position(state)

    def test_strict_native_action_parser(self):
        state = position("CAT?")
        action = parse_action("PLACE 4 2.5 2\n7 7 3 A 1\n7 8 2 T 0\n", state)
        self.assertTrue(action["placements"][0]["tile"]["blank"])
        self.assertEqual(action["score"], 4)
        self.assertEqual(parse_action("EXCHANGE 2 0 3", state)["rackIndexes"], [0, 3])
        self.assertEqual(parse_action("PASS", state), {"kind": "pass"})
        for output in ["", "UNKNOWN", "PLACE 2 nan 1\n7 7 0 C 0",
                       "PLACE 2 2 2\n7 7 0 C 0\n7 8 0 C 0",
                       "PLACE 2 2 1\n15 7 0 C 0", "EXCHANGE 2 0 0"]:
            with self.subTest(output=output):
                with self.assertRaises((RuntimeError, ValueError, IndexError)):
                    parse_action(output, state)

    def test_dictionary_writer_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "words.txt", Path(directory) / "demo.dawg"
            source.write_text("AT CAT CATS BAT A OVERLONGABCDEFGHIJK\n", encoding="utf-8")
            self.assertEqual(build_lexicon(source, output), 4)
            raw = output.read_bytes()
            words = set()

            def walk(index, prefix):
                while True:
                    record = raw[index * 7:(index + 1) * 7]
                    pointer, flags = int.from_bytes(record[:3], "big"), record[3]
                    text = prefix + chr(ord("A") + (flags & 31))
                    if flags & 32:
                        words.add(text)
                    if pointer:
                        walk(pointer, text)
                    if flags & 64:
                        break
                    index += 1

            walk(1, "")
            self.assertEqual(words, {"AT", "CAT", "CATS", "BAT"})


class StubBridge:
    manifest = {"revision": "test"}

    def check_dictionary(self):
        pass

    def choose(self, state):
        validate_position(state)
        return {"kind": "pass"}


class QuackleHTTPTest(unittest.TestCase):
    def setUp(self):
        self.bridge = StubBridge()
        self.server = QuackleServer(("127.0.0.1", 0), self.bridge)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        self.addCleanup(connection.close)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        return response.status, response.read()

    def test_capability_same_origin_move_and_static_page(self):
        status, data = self.request("GET", "/api/quackle")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(data)["available"])
        status, data = self.request("POST", "/api/quackle/move", json.dumps(position()), {
            "Content-Type": "application/json",
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
        })
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["action"], {"kind": "pass"})
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertEqual(self.request("GET", "/../pyproject.toml")[0], 404)

    def test_invalid_origin_host_and_requests(self):
        cases = [
            ("GET", "/api/quackle", None, {"Host": "attacker.example"}, 403),
            ("GET", "/api/quackle", None, {"Origin": "https://attacker.example"}, 403),
            ("POST", "/api/quackle/move", "{}", {"Content-Type": "text/plain"}, 415),
            ("POST", "/api/quackle/move", "{", {"Content-Type": "application/json"}, 400),
            ("POST", "/api/quackle/move", "{}", {"Content-Type": "application/json"}, 400),
            ("POST", "/api/quackle/move", " " * 33000, {"Content-Type": "application/json"}, 400),
            ("POST", "/api/unknown", "{}", {"Content-Type": "application/json"}, 404),
        ]
        for method, path, body, headers, expected in cases:
            with self.subTest(path=path, expected=expected):
                self.assertEqual(self.request(method, path, body, headers)[0], expected)

    def test_busy_and_failed_native_calls_surface_errors(self):
        for error, status in [(BlockingIOError("busy"), 429), (RuntimeError("failed"), 503)]:
            with patch.object(self.bridge, "choose", side_effect=error):
                actual, body = self.request("POST", "/api/quackle/move", json.dumps(position()),
                                            {"Content-Type": "application/json"})
                self.assertEqual(actual, status)
                self.assertEqual(json.loads(body)["error"], str(error))

    def test_dqn_model_endpoint_checks_checkpoint_and_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.json"
            with patch("scrabble_gym.quackle_server.POLICY", policy):
                self.assertEqual(self.request("GET", "/api/dqn/model")[0], 404)
                policy.write_text("{", encoding="utf-8")
                self.assertEqual(self.request("GET", "/api/dqn/model")[0], 503)
                model = {"schema": 1, "dictionary_sha256": hashlib.sha256((DOCS / "words.txt").read_bytes()).hexdigest(),
                         "episodes": 2, "updates": 5}
                policy.write_text(json.dumps(model), encoding="utf-8")
                status, body = self.request("GET", "/api/dqn/model")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["updates"], 5)
                for update in ({"updates": 0}, {"dictionary_sha256": "wrong"}):
                    policy.write_text(json.dumps({**model, **update}), encoding="utf-8")
                    self.assertEqual(self.request("GET", "/api/dqn/model")[0], 503)


@unittest.skipUnless((TARGET / "manifest.json").is_file(), "Build native Quackle to run integration tests")
class QuackleNativeTest(unittest.TestCase):
    def setUp(self):
        self.bridge = QuackleBridge()

    def test_opening_blank_bingo_and_determinism(self):
        state = position("CAT?ARS")
        before = copy.deepcopy(state)
        action = self.bridge.choose(state)
        self.assertEqual(action["kind"], "place")
        self.assertEqual(len(action["placements"]), 7)
        self.assertTrue(any(p["tile"]["blank"] for p in action["placements"]))
        self.assertTrue(any((p["row"], p["col"]) == (7, 7) for p in action["placements"]))
        self.assertEqual(action, self.bridge.choose(state))
        self.assertEqual(state, before)

    def test_blocked_rack_and_exchange_gate(self):
        self.assertEqual(self.bridge.choose(position("Q"))["kind"], "exchange")
        state = position("Q")
        state["bagSize"] = 0
        self.assertEqual(self.bridge.choose(state)["kind"], "pass")

    def test_native_error_and_timeout_are_not_success_shaped(self):
        state = position()
        with patch("scrabble_gym.quackle_server.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 1, "", "test error")):
            with self.assertRaisesRegex(RuntimeError, "test error"):
                self.bridge.choose(state)
        with patch("scrabble_gym.quackle_server.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("bridge", 20)):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.bridge.choose(state)
        self.assertIn(self.bridge.choose(state)["kind"], ("place", "exchange", "pass"))

    def test_full_games_native_moves_match_java_rules_and_scores(self):
        for seed in (42, 43):
            env = ScrabbleEnv()
            self.addCleanup(env.close)
            obs, _ = env.reset(seed=seed)
            moves = 0
            while True:
                state = position()
                state["board"] = [[None for _ in range(15)] for _ in range(15)]
                for r in range(15):
                    for c in range(15):
                        code = int(obs["board"][r, c])
                        if code:
                            letter = chr(ord("A") + code - 1)
                            blank = bool(obs["board_blanks"][r, c])
                            state["board"][r][c] = {
                                "letter": letter, "points": 0 if blank else POINTS[letter],
                                "blank": blank,
                            }
                state["rack"] = [tile(str(t.letter())) for t in env._rack(obs["player"])]
                state["bagSize"] = obs["bag_size"]
                action = self.bridge.choose(state)
                if action["kind"] == "place":
                    placed = {(p["row"], p["col"]): p["tile"] for p in action["placements"]}
                    first = next(iter(placed))
                    dr, dc = (0, 1)
                    if len({r for r, _ in placed}) > 1:
                        dr, dc = 1, 0
                    elif len(placed) == 1:
                        r, c = first
                        horizontal = any(0 <= c + d < 15 and state["board"][r][c + d]
                                         for d in (-1, 1))
                        if not horizontal:
                            dr, dc = 1, 0
                    r, c = min(placed)
                    while (0 <= r - dr < 15 and 0 <= c - dc < 15
                           and state["board"][r - dr][c - dc]):
                        r, c = r - dr, c - dc
                    start_r, start_c = r, c
                    text, blanks = "", []
                    while 0 <= r < 15 and 0 <= c < 15:
                        current = placed.get((r, c)) or state["board"][r][c]
                        if not current:
                            break
                        if (r, c) in placed and current["blank"]:
                            blanks.append(len(text))
                        text += current["letter"]
                        r, c = r + dr, c + dc
                    move = Move("place", start_r, start_c, int(dr == 1), text, tuple(blanks))
                elif action["kind"] == "exchange":
                    move = Move("exchange", exchange="".join(
                        state["rack"][i]["letter"] for i in action["rackIndexes"]
                    ))
                else:
                    move = Move()
                obs, _, terminated, truncated, info = env.step(move)
                if action["kind"] == "place":
                    self.assertEqual(action["score"], info["move_score"])
                    moves += 1
                if terminated or truncated:
                    self.assertTrue(terminated)
                    self.assertFalse(truncated)
                    break
            self.assertGreater(moves, 2)


if __name__ == "__main__":
    unittest.main()
