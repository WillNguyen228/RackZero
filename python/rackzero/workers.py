"""Parallel self-play and evaluation workers, and the batched inference server.

Every self-play game and evaluation pair is a task with its own seed, so results
do not depend on how many workers run them or in which order they finish. A
worker process owns one JVM, one lexicon trie, one reusable ``Game`` and one
``Simulator``, and receives network weights as weights-only files tagged by
version. It never receives another player's hidden state: live ``Game``
objects stay inside the worker that plays them.

Inference modes:

* ``local``: each worker evaluates the network on its own CPU (default).
* ``server``: workers send observation records to a thread in the trainer
  process, which batches them across workers on the trainer's device (e.g. an
  A100). This helps only when network inference, not move generation,
  dominates search time (see docs/A100_TRAINING.md).
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import itertools
import multiprocessing
import queue
import threading
import time
from pathlib import Path

import numpy as np

from rackzero.features import encode_record, pad_record, stack_records, to_tensors
from rackzero.search import Search, SearchConfig


@dataclass(frozen=True)
class ModelRef:
    path: str
    tag: str


@dataclass(frozen=True)
class WorkerSpec:
    dictionary: str | None
    fingerprint: str
    max_turns: int
    particles: int
    movegen: str = "fast"
    threads: int = 1

    @classmethod
    def from_config(cls, config, fingerprint: str) -> "WorkerSpec":
        return cls(config.dictionary, fingerprint, config.max_turns, config.search.particles,
                   config.search.movegen, config.selfplay.worker_threads)


# --------------------------------------------------------------------------- #
# Worker context
# --------------------------------------------------------------------------- #

class WorkerContext:
    def __init__(self, spec: WorkerSpec, client=None):
        import torch

        from rackzero.game import Game, Simulator
        from rackzero.training import lexicon_for_path

        torch.set_num_threads(spec.threads)
        lexicon = lexicon_for_path(spec.dictionary)
        if lexicon.fingerprint != spec.fingerprint:
            raise ValueError("Worker lexicon fingerprint does not match the trainer's")
        self.spec = spec
        self.words = lexicon.words
        self.game = Game(self.words, max_turns=spec.max_turns, particles=spec.particles)
        self.simulator = Simulator(self.words, max_turns=spec.max_turns, movegen=spec.movegen)
        self.client = client
        self._models: dict[str, object] = {}

    def evaluator(self, ref: ModelRef | None):
        if ref is None:
            return None
        if ref.tag not in self._models:
            if self.client is not None:
                evaluator = RemoteEvaluator(self.client, ref.tag)
            else:
                from rackzero.checkpoints import load_model_file
                from rackzero.models.networks import NetworkEvaluator
                evaluator = NetworkEvaluator(load_model_file(ref.path), "cpu")
            if len(self._models) >= 3:
                self._models.pop(next(iter(self._models)))
            self._models[ref.tag] = evaluator
        return self._models[ref.tag]

    def game_factory(self, particles: int, rng: np.random.Generator):
        self.game._particles = particles
        self.game._rng, self.game._external_rng = rng, True
        return self.game


_CONTEXT: WorkerContext | None = None


def _initialize(spec: WorkerSpec, server_args=None) -> None:
    global _CONTEXT
    client = None
    if server_args is not None:
        requests, responses, counter = server_args
        with counter.get_lock():
            slot = counter.value
            counter.value += 1
        client = ServerClient(slot, requests, responses[slot])
    _CONTEXT = WorkerContext(spec, client)


def _run(kind: str, task):
    return TASKS[kind](_CONTEXT, task)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SelfPlayTask:
    seed: int
    model: ModelRef
    search: dict
    candidates: int
    max_turns: int
    temperature: float = 1.0
    temperature_moves: int | None = None
    final_temperature: float = 0.0


def selfplay_seed(base: int, iteration: int, game: int) -> int:
    return int(np.random.SeedSequence([base, iteration, game]).generate_state(1, np.uint64)[0])


def temperature_at(task: SelfPlayTask, ply: int) -> float:
    if task.temperature_moves is None or ply < task.temperature_moves:
        return task.temperature
    return task.final_temperature


def _counters(ctx: WorkerContext, evaluator) -> dict:
    stats = dict(ctx.simulator.stats)
    if evaluator is not None and hasattr(evaluator, "stats"):
        stats.update(evaluator.stats())
    return stats


def play_selfplay_game(ctx: WorkerContext, task: SelfPlayTask) -> dict:
    """One self-play game -> stacked replay records (None if truncated) and timings."""
    game_seq, search_seq, belief_seq = np.random.SeedSequence(task.seed).spawn(3)
    evaluator = ctx.evaluator(task.model)
    rng = np.random.default_rng(search_seq)
    search = Search(ctx.simulator, evaluator, SearchConfig(**task.search), rng)
    game = ctx.game
    before = _counters(ctx, evaluator)
    started = time.perf_counter()
    live_movegen = search_seconds = step_seconds = 0.0
    records, players = [], []
    info = game.reset(seed=int(game_seq.generate_state(1)[0] % 2**31),
                      rng=np.random.default_rng(belief_seq))
    terminated = truncated = False
    try:
        for ply in range(task.max_turns):
            belief = game.belief
            t0 = time.perf_counter()
            moves = game.legal_moves()
            t1 = time.perf_counter()
            result = search.run(info, moves=moves, belief=belief)
            t2 = time.perf_counter()
            live_movegen += t1 - t0
            search_seconds += t2 - t1
            policy = np.asarray(result.policy, dtype=np.float64)
            if not result.moves or policy.shape != (len(result.moves),):
                raise ValueError("Search returned an invalid policy")
            if not np.all(np.isfinite(policy)) or np.any(policy < 0) or policy.sum() <= 0:
                raise ValueError("Search returned nonfinite/negative/empty probabilities")
            policy = policy / policy.sum()
            record = encode_record(info, result.moves, evaluator.belief_features(info, belief),
                                   task.candidates)
            record["policy"] = np.zeros(task.candidates, dtype=np.float32)
            record["policy"][:len(policy)] = policy
            records.append(record)
            players.append(int(info.observation["player"]))
            tau = temperature_at(task, ply)
            if tau == 0:
                selected = int(np.argmax(policy))
            else:
                weights = np.log(np.maximum(policy, 1e-300))
                weights = np.exp((weights - weights.max()) / tau)
                selected = int(rng.choice(len(result.moves), p=weights / weights.sum()))
            t3 = time.perf_counter()
            info, terminated, truncated = game.step(result.moves[selected].move)
            step_seconds += time.perf_counter() - t3
            if terminated or truncated:
                break
        natural = terminated and not truncated
        scores = np.asarray(game.scores, dtype=np.float64)
        resets = sum(b.resets for b in game._beliefs)
    finally:
        game.close()
    after = _counters(ctx, evaluator)
    stats = {key: after[key] - before.get(key, 0) for key in after}
    stats.update(
        games=1, natural_games=int(natural), truncated_games=int(not natural),
        plies=len(records), seconds=time.perf_counter() - started,
        search_seconds=search_seconds, live_movegen_seconds=live_movegen,
        game_step_seconds=step_seconds, belief_resets=resets,
    )
    batch = None
    if natural:
        # A max-turn cutoff is never evidence for a win or a value target.
        for record, player in zip(records, players):
            margin = float(scores[player] - scores[1 - player])
            record["win"] = np.float32(np.sign(margin))
            record["margin"] = np.float32(margin)
        batch = stack_records(records)
        stats["positions"] = len(records)
    else:
        stats["positions"] = 0
    return {"records": batch, "stats": stats}


@dataclass(frozen=True)
class EvalGameTask:
    game_seed: int
    seat: int
    candidate: ModelRef
    opponent: str
    opponent_model: ModelRef | None
    search: dict
    opponent_search: dict
    max_turns: int


def play_evaluation_game(ctx: WorkerContext, task: EvalGameTask):
    from rackzero.evaluation import play_game

    model = ctx.evaluator(task.candidate)
    baseline = ctx.evaluator(task.opponent_model) or model
    return play_game(ctx.words, ctx.simulator, model, baseline, task.opponent,
                     SearchConfig(**task.search), SearchConfig(**task.opponent_search),
                     task.max_turns, task.game_seed, task.seat, game_factory=ctx.game_factory)


TASKS = {"selfplay": play_selfplay_game, "eval_game": play_evaluation_game}


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #

class Executor:
    """Runs tasks inline (``workers == 0``) or in persistent spawned processes."""

    def __init__(self, spec: WorkerSpec, workers: int, server: "InferenceServer | None" = None):
        self.spec = spec
        self.workers = int(workers)
        self.server = server
        self._context: WorkerContext | None = None
        self._pool = None
        if self.workers > 0:
            server_args = server.worker_args() if server is not None else None
            self._pool = ProcessPoolExecutor(
                self.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize, initargs=(spec, server_args))

    def context(self) -> WorkerContext:
        if self._context is None:
            self._context = WorkerContext(self.spec)
        return self._context

    def map(self, kind: str, tasks: list, progress=None) -> list:
        """Results in task order; ``progress(done, total)`` is called as tasks finish."""
        if self._pool is None:
            results = []
            for i, task in enumerate(tasks):
                results.append(TASKS[kind](self.context(), task))
                if progress:
                    progress(i + 1, len(tasks))
            return results
        if self.server is not None:
            self.server.start()
        try:
            futures = {self._pool.submit(_run, kind, task): i for i, task in enumerate(tasks)}
            results = [None] * len(tasks)
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    results[futures[future]] = future.result()
                if progress:
                    progress(len(tasks) - len(pending), len(tasks))
            return results
        finally:
            if self.server is not None:
                self.server.stop()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._context is not None:
            self._context.simulator.close()
            self._context = None


def evaluate_parallel(executor: Executor, config, candidate: ModelRef, *, opponent: str,
                      games: int, seed: int, method: str | None = None,
                      opponent_model: ModelRef | None = None,
                      opponent_search: SearchConfig | None = None, progress=None) -> dict:
    from rackzero.evaluation import evaluation_search_config, pair_seeds, pair_up, summarize

    search = evaluation_search_config(config.search, method)
    opponent_config = (search if opponent_search is None
                       else evaluation_search_config(opponent_search))
    # One task per game (not per pair) balances load across workers; pairs are
    # reassembled afterwards so a truncated game still discards its whole pair.
    tasks = [EvalGameTask(game_seed, seat, candidate, opponent, opponent_model, asdict(search),
                          asdict(opponent_config), config.max_turns)
             for game_seed in pair_seeds(seed, games) for seat in (0, 1)]
    started = time.perf_counter()
    results = executor.map("eval_game", tasks, progress)
    pairs = pair_up(list(zip(results[0::2], results[1::2])))
    return summarize(pairs, method=search.method, opponent=opponent, seed=seed, games=games,
                     seconds=time.perf_counter() - started)


# --------------------------------------------------------------------------- #
# Batched inference server (thread inside the trainer process)
# --------------------------------------------------------------------------- #

class ServerClient:
    def __init__(self, slot: int, requests, responses):
        self.slot, self.requests, self.responses = slot, requests, responses
        self._ids = itertools.count()

    def call(self, tag: str, record: dict, policy: bool) -> dict:
        request = next(self._ids)
        self.requests.put(("infer", self.slot, request, tag, policy, record))
        while True:
            reply_id, payload = self.responses.get()
            if reply_id == request:
                if isinstance(payload, Exception):
                    raise payload
                return payload


class RemoteEvaluator:
    """Worker-side stand-in for ``NetworkEvaluator`` that calls the server."""

    def __init__(self, client: ServerClient, tag: str):
        self.client, self.tag = client, tag
        self.calls = 0
        self.seconds = 0.0

    def eval(self):
        return self

    def belief_features(self, info, belief=None):
        from rackzero.features import belief_summary
        return belief_summary(info, belief)

    def _call(self, record, policy):
        start = time.perf_counter()
        out = self.client.call(self.tag, record, policy)
        self.calls += 1
        self.seconds += time.perf_counter() - start
        return out

    def predict(self, info, moves, belief_mean=None):
        from rackzero.features import belief_from_mean
        out = self._call(encode_record(info, moves, belief_from_mean(info, belief_mean),
                                       len(moves)), True)
        logits = out["logits"][:len(moves)]
        probs = np.exp(logits - logits.max())
        return probs / probs.sum(), float(out["value"]), float(out["margin"]) * 100

    def predict_value(self, info, belief_mean=None):
        from rackzero.features import belief_from_mean
        from scrabble_gym.environment import Move, ScoredMove
        out = self._call(encode_record(info, (ScoredMove(Move(), 0),),
                                       belief_from_mean(info, belief_mean), 1), False)
        return float(out["value"]), float(out["margin"]) * 100

    def stats(self) -> dict:
        return {"inference_calls": self.calls, "inference_positions": self.calls,
                "inference_seconds": self.seconds}


class InferenceServer:
    """Batches worker inference requests on the trainer's device."""

    def __init__(self, workers: int, device, precision: str = "fp32", batch_size: int = 32,
                 max_wait_ms: float = 2.0):
        context = multiprocessing.get_context("spawn")
        self.requests = context.Queue()
        self.responses = [context.Queue() for _ in range(workers)]
        self.counter = context.Value("i", 0)
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        self.max_wait = max_wait_ms / 1000
        self.models: dict[str, object] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.reset_stats()

    def reset_stats(self) -> None:
        self.batches = self.positions = 0
        self.forward_seconds = 0.0
        self.max_batch = 0

    def stats(self) -> dict:
        return {"server_batches": self.batches, "server_positions": self.positions,
                "server_mean_batch": self.positions / self.batches if self.batches else 0.0,
                "server_max_batch": self.max_batch,
                "server_forward_seconds": self.forward_seconds}

    def worker_args(self):
        return (self.requests, self.responses, self.counter)

    def load(self, ref: ModelRef) -> None:
        from rackzero.checkpoints import load_model_file
        if ref.tag not in self.models:
            if len(self.models) >= 3:
                self.models.pop(next(iter(self.models)))
            self.models[ref.tag] = load_model_file(ref.path, self.device)

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._serve, name="rackzero-inference",
                                            daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                first = self.requests.get(timeout=0.05)
            except queue.Empty:
                continue
            batch = [first]
            # Drain without timed waits: a timed Queue.get sleeps for at least one
            # OS timer tick (~15 ms on Windows), far longer than a forward pass.
            deadline = time.perf_counter() + self.max_wait
            while len(batch) < self.batch_size:
                try:
                    batch.append(self.requests.get_nowait())
                except queue.Empty:
                    if time.perf_counter() >= deadline:
                        break
                    time.sleep(0)
            groups: dict[tuple, list] = {}
            for message in batch:
                _, slot, request, tag, policy, record = message
                groups.setdefault((tag, policy), []).append((slot, request, record))
            for (tag, policy), items in groups.items():
                try:
                    outputs = self._forward(tag, policy, [r for _, _, r in items])
                except Exception as error:  # report to the waiting workers
                    outputs = [error] * len(items)
                for (slot, request, _), output in zip(items, outputs):
                    self.responses[slot].put((request, output))

    def _forward(self, tag: str, policy: bool, records: list[dict]) -> list:
        import torch

        if tag not in self.models:
            raise KeyError(f"Inference server has no model {tag!r}")
        start = time.perf_counter()
        width = max(len(r["cand_mask"]) for r in records)
        tensors = to_tensors(stack_records([pad_record(r, width) for r in records]), self.device)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.precision)
        with torch.inference_mode(), torch.autocast(
                torch.device(self.device).type, dtype=dtype or torch.float32,
                enabled=dtype is not None):
            out = self.models[tag](tensors, policy=policy)
        out = {k: v.float().cpu().numpy() for k, v in out.items()}
        self.forward_seconds += time.perf_counter() - start
        self.batches += 1
        self.positions += len(records)
        self.max_batch = max(self.max_batch, len(records))
        return [{k: v[i] for k, v in out.items()} for i in range(len(records))]


def write_model(directory: Path, tag: str, model_config: dict, state_dict: dict) -> ModelRef:
    from rackzero.checkpoints import save_model_file

    path = Path(directory) / f"{tag}.pt"
    save_model_file(path, model_config, state_dict)
    return ModelRef(str(path), tag)
