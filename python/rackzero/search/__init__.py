"""Sampled-chance PUCT, determinization, and single-observer information-set search."""

from dataclasses import dataclass, field
import math

import numpy as np

from rackzero.belief import RackBelief, draw_probability
from rackzero.game import InformationState, Simulator, World, used_tiles
from scrabble_gym.environment import ScoredMove


METHODS = ("stochastic", "pimc", "ismcts", "belief", "neural_determinization")


@dataclass(frozen=True)
class SearchConfig:
    method: str = "ismcts"
    simulations: int = 32
    depth: int = 3
    candidates: int = 16
    exploration: float = 1.5
    particles: int = 32
    # Self-play only: Dirichlet(alpha) noise mixed into root priors with weight
    # `root_noise_fraction` (AlphaZero exploration). 0 disables (the default).
    root_noise_alpha: float = 0.0
    root_noise_fraction: float = 0.0
    # Simulator move generator: "fast" (cached, bit-identical output) or "reference".
    movegen: str = "fast"

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown search method {self.method!r}; choose from {METHODS}")
        if any(isinstance(n, bool) or not isinstance(n, int)
               for n in (self.simulations, self.depth, self.particles, self.candidates)):
            raise ValueError("Search budgets must be integers")
        if min(self.simulations, self.depth, self.particles) < 1 or self.candidates < 3:
            raise ValueError("Positive budgets and at least three candidates are required")
        if not math.isfinite(self.exploration) or self.exploration <= 0:
            raise ValueError("exploration must be finite and positive")
        for name in ("root_noise_alpha", "root_noise_fraction"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.root_noise_fraction > 1:
            raise ValueError("root_noise_fraction must be at most 1")
        if self.root_noise_fraction > 0 and self.root_noise_alpha <= 0:
            raise ValueError("root noise requires root_noise_alpha > 0")
        if self.movegen not in ("fast", "reference"):
            raise ValueError("movegen must be fast or reference")


@dataclass
class SearchResult:
    moves: tuple[ScoredMove, ...]
    policy: np.ndarray
    visits: np.ndarray
    value: float
    chance_outcomes: int = 0


def shortlist(moves: tuple[ScoredMove, ...], limit: int) -> tuple[ScoredMove, ...]:
    if len(moves) <= limit:
        return moves
    keep = sorted(range(len(moves)), key=lambda i: (-moves[i].score, i))[:limit]
    mandatory = [i for i, move in enumerate(moves) if move.move.kind == "pass"]
    exchanges = [i for i, move in enumerate(moves) if move.move.kind == "exchange"]
    if exchanges:
        mandatory.append(max(exchanges, key=lambda i: len(moves[i].move.exchange)))
    for index in mandatory:
        if index not in keep:
            replace = next(i for i in reversed(range(len(keep))) if keep[i] not in mandatory)
            keep[replace] = index
    return tuple(moves[i] for i in sorted(keep))


def enumerate_draws(pool: np.ndarray, count: int, max_outcomes: int = 1024):
    """Exact unordered chance outcomes for small pools; fail rather than silently prune."""
    if count < 0 or count > int(pool.sum()) or max_outcomes < 1:
        raise ValueError("Invalid chance enumeration budget or draw")
    active = np.flatnonzero(pool)
    outcomes = []

    def visit(index, remaining, draw):
        if len(outcomes) > max_outcomes:
            raise ValueError("Chance support exceeds max_outcomes; use sampling")
        if index == len(active):
            if remaining == 0:
                outcomes.append((draw.copy(), draw_probability(pool, draw)))
            return
        tile = active[index]
        for n in range(min(int(pool[tile]), remaining) + 1):
            draw[tile] = n
            visit(index + 1, remaining - n, draw)
        draw[tile] = 0

    visit(0, count, np.zeros(27, dtype=np.int64))
    if len(outcomes) > max_outcomes:
        raise ValueError("Chance support exceeds max_outcomes; use sampling")
    return outcomes


@dataclass
class ChanceNode:
    # Probabilities are conditional on a sampled world's bag, not a known live bag.
    outcomes: dict[tuple, tuple[int, float]] = field(default_factory=dict)

    def record(self, before: World, after: World, candidate: ScoredMove):
        player = before.observation["player"]
        spent = used_tiles(before.information().observation, candidate.move)
        draw = after.racks[player] - before.racks[player] + spent
        key = (tuple(before.bag), tuple(draw))
        visits, _ = self.outcomes.get(key, (0, 0.0))
        self.outcomes[key] = (visits + 1, draw_probability(before.bag, draw))


@dataclass
class Node:
    moves: tuple[ScoredMove, ...]
    priors: np.ndarray
    visits: np.ndarray
    totals: np.ndarray
    availability: np.ndarray
    chance: dict = field(default_factory=dict)
    rack_sum: np.ndarray = field(default_factory=lambda: np.zeros(27))
    particles: int = 0
    noise: np.ndarray | None = None


def information_key(info: InformationState):
    obs = info.observation
    return (obs["board"].tobytes(), obs["board_blanks"].tobytes(),
            obs["rack"].tobytes(), tuple(obs["scores"]), obs["player"],
            obs["bag_size"], obs["opponent_rack_size"], obs["scoreless_turns"],
            obs["turn"], info.history)


class Search:
    """Bounded comparisons, not a claim to solve imperfect-information Scrabble.

    ISMCTS/belief optimize only root-player decisions; opponent actions follow
    an observation-only policy. Determinization variants deliberately retain
    perfect-information continuation bias as experimental baselines.
    """

    def __init__(self, simulator: Simulator, model, config: SearchConfig,
                 rng: np.random.Generator):
        self.simulator, self.model, self.config, self.rng = simulator, model, config, rng
        self.nodes: dict[tuple, Node] = {}
        self._root_noise: np.ndarray | None = None

    def _prediction(self, info, moves, mean=None):
        if self.model is not None and self.config.method != "pimc":
            return self.model.predict(info, moves, mean)[:2]
        scores = np.array([m.score for m in moves], dtype=np.float64) / 10
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        obs = info.observation
        player = obs["player"]
        value = float(np.tanh((obs["scores"][player] - obs["scores"][1 - player]) / 100))
        return probs, value

    def _node(self, key, info, moves, mean=None):
        if key not in self.nodes:
            priors, _ = self._prediction(info, moves, mean)
            priors = np.asarray(priors, dtype=np.float64)
            if (priors.shape != (len(moves),) or np.any(priors < 0)
                    or not np.all(np.isfinite(priors)) or priors.sum() <= 0):
                raise ValueError("Policy returned invalid candidate probabilities")
            self.nodes[key] = Node(moves, priors / priors.sum(), np.zeros(len(moves), dtype=np.int64),
                                   np.zeros(len(moves)), np.zeros(len(moves), dtype=np.int64))
            if key[0] == "root" and self._root_noise is not None:
                self.nodes[key].noise = self._root_noise
        return self.nodes[key]

    def _select(self, node: Node) -> int:
        node.availability += 1
        q = node.totals / np.maximum(1, node.visits)
        priors = node.priors
        if node.noise is not None:
            fraction = self.config.root_noise_fraction
            priors = (1 - fraction) * priors + fraction * node.noise
        bonus = (self.config.exploration * priors
                 * np.sqrt(node.availability) / (1 + node.visits))
        return int(np.argmax(q + bonus))

    def _terminal(self, world, root):
        scores = world.observation["scores"]
        return float(np.sign(scores[root] - scores[1 - root]))

    def _leaf_value(self, world, info):
        if self.model is not None and self.config.method != "pimc":
            if hasattr(self.model, "predict_value"):
                return self.model.predict_value(info)[0]
            moves = shortlist(self.simulator.legal_moves(world, self.rng), self.config.candidates)
            return self._prediction(info, moves)[1]
        obs = info.observation
        player = obs["player"]
        return float(np.tanh((obs["scores"][player] - obs["scores"][1 - player]) / 100))

    def _simulate(self, world, root, depth, root_moves, root_mean, tree_id=None):
        if world.terminal:
            return self._terminal(world, root)
        info = world.information()
        player = info.observation["player"]
        if depth >= self.config.depth:
            value = self._leaf_value(world, info)
            return value if player == root else -value
        moves = (root_moves if depth == 0 else
                 shortlist(self.simulator.legal_moves(world, self.rng), self.config.candidates))
        info_methods = self.config.method in ("ismcts", "belief")
        if info_methods and player != root:
            # Opponent cannot choose with knowledge of the root player's private rack.
            priors, _ = self._prediction(info, moves)
            index = int(self.rng.choice(len(moves), p=np.asarray(priors) / np.sum(priors)))
            after = self.simulator.step(world, moves[index].move, self.rng)
            return self._simulate(after, root, depth + 1, root_moves, root_mean, tree_id)
        key = ("root", tree_id) if depth == 0 else information_key(info)
        if not info_methods and depth > 0:
            key = (key, tuple(world.racks[1 - player]), tuple(world.bag), tree_id)
        node = self._node(key, info, moves, root_mean if depth == 0 else None)
        if self.config.method == "belief":
            node.rack_sum += world.racks[1 - player]
            node.particles += 1
            # Particle means are simulator estimates, never the actual live hidden rack.
            priors, _ = self._prediction(
                info, node.moves, root_mean if depth == 0 else node.rack_sum / node.particles)
            node.priors = np.asarray(priors, dtype=np.float64)
        index = self._select(node)
        candidate = node.moves[index]
        after = self.simulator.step(world, candidate.move, self.rng)
        node.chance.setdefault(index, ChanceNode()).record(world, after, candidate)
        value = self._simulate(after, root, depth + 1, root_moves, root_mean, tree_id)
        node.visits[index] += 1
        node.totals[index] += value if player == root else -value
        return value

    def _belief_input(self, info: InformationState, belief: RackBelief):
        """Root belief features: rich posterior summary when the model accepts it."""
        if self.model is not None and hasattr(self.model, "belief_features"):
            return self.model.belief_features(info, belief)
        return belief.expected()

    def run(self, info: InformationState, moves=None, belief: RackBelief | None = None):
        self.nodes = {}
        self._root_noise = None
        config = self.config
        use_belief = config.method == "belief"
        mean = self._belief_input(info, belief) if belief is not None else None
        if not use_belief or belief is None:
            belief = RackBelief(info, config.particles, self.rng)
        root = info.observation["player"]
        first_world = self.simulator.determinize(info, belief.sample())
        if moves is None:
            moves = self.simulator.legal_moves(first_world, self.rng)
        moves = shortlist(tuple(moves), config.candidates)
        if mean is None:
            mean = self._belief_input(info, belief)
        if config.root_noise_fraction > 0 and len(moves) > 1:
            self._root_noise = self.rng.dirichlet(np.full(len(moves), config.root_noise_alpha))
        determinization = config.method in ("pimc", "neural_determinization")
        # Equal fixed-world budgets; all methods use exactly config.simulations rollouts.
        determinizations = min(4, max(1, config.simulations // max(1, len(moves))))
        worlds = [self.simulator.determinize(info, belief.sample())
                  for _ in range(determinizations)] if determinization else []
        for simulation in range(config.simulations):
            tree_id = simulation % determinizations if determinization else None
            world = worlds[tree_id] if determinization else self.simulator.determinize(info, belief.sample())
            self._simulate(world, root, 0, moves, mean, tree_id)
        roots = [node for key, node in self.nodes.items()
                 if len(key) == 2 and key[0] == "root"]
        visits = sum((node.visits for node in roots), np.zeros(len(moves), dtype=np.int64))
        total = sum(float(node.totals.sum()) for node in roots)
        chance_outcomes = sum(len(chance.outcomes) for node in self.nodes.values()
                              for chance in node.chance.values())
        return SearchResult(moves, visits / visits.sum(), visits, total / visits.sum(), chance_outcomes)
