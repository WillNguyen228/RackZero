"""Loopback-only browser server for the native Quackle static policy."""

import argparse
from collections import Counter
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import subprocess
import threading
from urllib.parse import urlsplit

from .build import ROOT
from .quackle_build import TARGET


LOG = logging.getLogger(__name__)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ?"
POINTS = dict(zip(LETTERS, (1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3,
                           1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10, 0)))
COUNTS = dict(zip(LETTERS, (9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2,
                           6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2)))
DOCS = ROOT / "docs"
MAX_BODY = 32768
POLICY = ROOT / "target" / "dqn" / "policy.json"


def validate_position(position: object) -> tuple[str, str, int]:
    if not isinstance(position, dict) or set(position) != {"board", "rack", "bagSize"}:
        raise ValueError("Send only board, rack and bagSize")
    board, rack, bag_size = position["board"], position["rack"], position["bagSize"]
    if not isinstance(board, list) or len(board) != 15 or any(
        not isinstance(row, list) or len(row) != 15 for row in board
    ):
        raise ValueError("board must be a 15 by 15 array")
    if not isinstance(rack, list) or not 1 <= len(rack) <= 7:
        raise ValueError("rack must contain one to seven tiles")
    if type(bag_size) is not int or not 0 <= bag_size <= 100:
        raise ValueError("bagSize must be an integer from 0 to 100")
    seen: Counter[str] = Counter()

    def tile_code(tile: object, placed: bool) -> str:
        if not isinstance(tile, dict) or set(tile) != {"letter", "points", "blank"}:
            raise ValueError("Each tile needs letter, points and blank")
        letter, points, blank = tile["letter"], tile["points"], tile["blank"]
        if (not isinstance(letter, str) or letter not in LETTERS or len(letter) != 1
                or type(points) is not int or type(blank) is not bool):
            raise ValueError("Invalid tile fields")
        if blank:
            if points != 0 or (placed and letter == "?") or (not placed and letter != "?"):
                raise ValueError("Invalid blank assignment")
        elif letter == "?" or points != POINTS[letter]:
            raise ValueError("Invalid letter points")
        seen["?" if blank else letter] += 1
        return letter.lower() if placed and blank else letter

    cells = "".join("." if t is None else tile_code(t, True) for row in board for t in row)
    rack_text = "".join(tile_code(t, False) for t in rack)
    if any(n > COUNTS[c] for c, n in seen.items()) or sum(seen.values()) + bag_size > 100:
        raise ValueError("Position exceeds the standard tile distribution")
    return cells, rack_text, bag_size


def parse_action(output: str, position: dict) -> dict:
    lines = output.strip().splitlines()
    if not lines:
        raise RuntimeError("Quackle returned no action")
    header = lines[0].split()
    rack = position["rack"]
    if header == ["PASS"] and len(lines) == 1:
        return {"kind": "pass"}
    if header[0] == "EXCHANGE":
        indexes = [int(i) for i in header[2:]]
        if (len(lines) != 1 or len(indexes) != int(header[1]) or not indexes
                or len(set(indexes)) != len(indexes) or position["bagSize"] < 7
                or any(not 0 <= i < len(rack) for i in indexes)):
            raise RuntimeError("Quackle returned an invalid exchange")
        return {"kind": "exchange", "rackIndexes": indexes}
    if header[0] != "PLACE" or len(header) != 4:
        raise RuntimeError("Quackle returned an unsupported action")
    score, equity, count = int(header[1]), float(header[2]), int(header[3])
    if not math.isfinite(equity) or score < 0 or not 1 <= count <= 7 or len(lines) != count + 1:
        raise RuntimeError("Quackle returned invalid move metadata")
    placements = []
    used = set()
    squares = set()
    for line in lines[1:]:
        r, c, i, letter, blank = line.split()
        row, col, index = int(r), int(c), int(i)
        if (not 0 <= row < 15 or not 0 <= col < 15 or not 0 <= index < len(rack)
                or index in used or (row, col) in squares or position["board"][row][col]
                or letter not in POINTS or letter == "?" or blank not in ("0", "1")
                or rack[index]["letter"] != ("?" if blank == "1" else letter)):
            raise RuntimeError("Quackle returned invalid placement data")
        used.add(index)
        squares.add((row, col))
        placements.append({
            "row": row, "col": col, "rackIndex": index,
            "tile": {"letter": letter, "points": 0 if blank == "1" else POINTS[letter],
                     "blank": blank == "1"},
        })
    return {"kind": "place", "placements": placements, "score": score, "equity": equity}


class QuackleBridge:
    def __init__(self):
        manifest_path = TARGET / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Run python -m scrabble_gym.quackle_build first")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.executable = (ROOT / self.manifest["executable"]).resolve()
        if not self.executable.is_relative_to(TARGET.resolve()) or not self.executable.is_file():
            raise FileNotFoundError("Native bridge executable is missing or outside target/quackle")
        self._lock = threading.BoundedSemaphore(1)
        self.check_dictionary()

    def check_dictionary(self) -> None:
        digest = hashlib.sha256((DOCS / "words.txt").read_bytes()).hexdigest()
        if digest != self.manifest["dictionary_sha256"]:
            raise RuntimeError("Dictionary changed; rebuild Quackle before playing")

    def choose(self, position: dict) -> dict:
        cells, rack, bag_size = validate_position(position)
        self.check_dictionary()
        if not self._lock.acquire(blocking=False):
            raise BlockingIOError("Quackle is busy with another request; retry shortly")
        try:
            result = subprocess.run(
                [str(self.executable)], input=f"{cells}\n{rack}\n{bag_size}\n",
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=20, check=False,
            )
            if result.returncode:
                raise RuntimeError(f"Native Quackle failed ({result.returncode}): {result.stderr.strip()[:500]}")
            try:
                return parse_action(result.stdout, position)
            except (ValueError, IndexError) as error:
                raise RuntimeError("Native Quackle returned malformed output") from error
        finally:
            self._lock.release()


class QuackleServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], bridge: QuackleBridge):
        self.bridge = bridge
        super().__init__(address, Handler)


class Handler(SimpleHTTPRequestHandler):
    server: QuackleServer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS), **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(25)

    def _json(self, status: int, data: dict) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            LOG.info("Client disconnected before the response was delivered")

    def _local_request(self) -> bool:
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if host not in allowed or (origin is not None and origin != f"http://{host}"):
            self._json(403, {"error": "Only same-origin loopback requests are allowed"})
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._json(403, {"error": "Cross-site requests are not allowed"})
            return False
        return True

    def do_GET(self):
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/quackle":
            try:
                self.server.bridge.check_dictionary()
            except (OSError, RuntimeError) as error:
                LOG.error("Quackle availability: %s", error)
                self._json(503, {"available": False, "error": str(error)})
                return
            self._json(200, {"available": True, "policy": "Quackle static score-plus-leave",
                             "revision": self.server.bridge.manifest["revision"]})
        elif path == "/api/dqn/model":
            if not POLICY.is_file():
                self._json(404, {"error": "Train a DQN checkpoint first: python -m scrabble_gym.train_dqn"})
                return
            try:
                model = json.loads(POLICY.read_text(encoding="utf-8"))
                digest = hashlib.sha256((DOCS / "words.txt").read_bytes()).hexdigest()
                if (model.get("schema") != 1 or model.get("dictionary_sha256") != digest
                        or model.get("episodes", 0) < 1 or model.get("updates", 0) < 1):
                    raise ValueError("DQN checkpoint is incompatible, has no learning updates, or uses another dictionary")
                self._json(200, model)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                LOG.error("DQN availability: %s", error)
                self._json(503, {"error": str(error)})
        elif path.startswith("/api/"):
            self._json(404, {"error": "Unknown API route"})
        else:
            super().do_GET()

    def do_POST(self):
        if not self._local_request():
            return
        if urlsplit(self.path).path != "/api/quackle/move":
            self._json(404, {"error": "Unknown API route"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Use application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY or self.headers.get("Transfer-Encoding"):
                raise ValueError("Invalid request length (maximum 32768 bytes)")
            position = json.loads(self.rfile.read(length))
            validate_position(position)
            action = self.server.bridge.choose(position)
        except (ValueError, UnicodeDecodeError) as error:
            LOG.warning("Rejected Quackle request: %s", error)
            self._json(400, {"error": str(error)})
        except BlockingIOError as error:
            self._json(429, {"error": str(error)})
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            LOG.error("Quackle request failed: %s", error)
            self._json(503, {"error": str(error)})
        else:
            self._json(200, {"action": action})

    def translate_path(self, path: str) -> str:
        resolved = Path(super().translate_path(path)).resolve()
        if not resolved.is_relative_to(DOCS.resolve()):
            return str(DOCS / "__not_served__")
        return str(resolved)

    def list_directory(self, path):
        self.send_error(403, "Directory listings are disabled")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be from 1 to 65535")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    bridge = QuackleBridge()
    probe = {"board": [[None] * 15 for _ in range(15)],
             "rack": [{"letter": c, "points": POINTS[c], "blank": False} for c in "CAT"],
             "bagSize": 86}
    bridge.choose(probe)
    with QuackleServer(("127.0.0.1", args.port), bridge) as server:
        print(f"Scrabble with Quackle: http://127.0.0.1:{args.port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.", flush=True)


if __name__ == "__main__":
    main()
