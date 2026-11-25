"""Sampled-chance PUCT, determinization, and single-observer information-set search."""

from dataclasses import dataclass, field
import math

import numpy as np

from rackzero.belief import RackBelief, draw_probability
from rackzero.game import POINTS, InformationState, Simulator, World, used_tiles
from scrabble_gym.environment import ScoredMove


METHODS = ("stochastic", "pimc", "ismcts", "belief", "neural_determinization")
ROOT_SELECTIONS = ("puct", "gumbel")
SHORTLISTS = ("score", "mixed")


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
    # Candidate shortlist: "score" keeps the top-scoring moves (original);
    # "mixed" keeps half by score and fills the rest by score + heuristic rack-leave
    # value, so leave-motivated plays and partial exchanges can be searched.
    shortlist: str = "score"
    # Root action selection: "puct" (AlphaZero visit counts; original) or
    # "gumbel" (Gumbel AlphaZero: Gumbel-top-k sampling + sequential halving,
    # policy target from completed Q-values; Danihelka et al., ICLR 2022).
    root_selection: str = "puct"
    gumbel_actions: int = 16          # m: actions considered at the root
    gumbel_c_visit: float = 50.0      # sigma(q) = (c_visit + max N) * c_scale * q01
    gumbel_c_scale: float = 1.0
    gumbel_scale: float = 1.0         # Gumbel noise scale; 0 = deterministic (evaluation)

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
        if self.shortlist not in SHORTLISTS:
            raise ValueError(f"shortlist must be one of {SHORTLISTS}")
        if self.root_selection not in ROOT_SELECTIONS:
            raise ValueError(f"root_selection must be one of {ROOT_SELECTIONS}")
        if (isinstance(self.gumbel_actions, bool) or not isinstance(self.gumbel_actions, int)
                or self.gumbel_actions < 2):
            raise ValueError("gumbel_actions must be an integer >= 2")
        for name in ("gumbel_c_visit", "gumbel_c_scale", "gumbel_scale"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class SearchResult:
    moves: tuple[ScoredMove, ...]
    policy: np.ndarray            # training target: visit distribution, or Gumbel improved policy
    visits: np.ndarray
    value: float                  # mean backed-up root value (root player's view, [-1, 1])
    chance_outcomes: int = 0
    selected: int | None = None   # Gumbel root: the action to play (None for PUCT)


def leave_values(observation: dict, moves: tuple[ScoredMove, ...]) -> np.ndarray:
    """The ``heuristic`` baseline's rack-leave term for every candidate (vectorized)."""
    rack = np.asarray(observation["rack"], dtype=np.int64)
    leaves = np.stack([rack - used_tiles(observation, m.move) for m in moves])
    vowels = leaves[:, [0, 4, 8, 14, 20]].sum(1)
    consonants = leaves[:, :26].sum(1) - vowels
    return (5 * leaves[:, 26] + 2 * leaves[:, 18] - 0.5 * (leaves @ POINTS)
            - np.abs(vowels - consonants) - np.maximum(leaves[:, :26] - 1, 0).sum(1))


def shortlist(moves: tuple[ScoredMove, ...], limit: int, observation: dict | None = None,
              mode: str = "score") -> tuple[ScoredMove, ...]:
    if len(moves) <= limit:
        return moves
    keep = sorted(range(len(moves)), key=lambda i: (-moves[i].score, i))[:limit]
    if mode == "mixed":
        if observation is None:
            raise ValueError("The mixed shortlist needs the acting player's observation")
        keep = keep[:limit // 2]
        equity = np.array([m.score for m in moves], dtype=np.float64) + leave_values(
            observation, moves)
        chosen = set(keep)
        for i in sorted(range(len(moves)), key=lambda i: (-equity[i], i)):
            if len(keep) >= limit:
                break
            if i not in chosen:
                keep.append(i)
                chosen.add(i)
    elif mode != "score":
        raise ValueError(f"Unknown shortlist mode {mode!r}")
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
            moves = self._shortlist(self.simulator.legal_moves(world, self.rng), info)
            return self._prediction(info, moves)[1]
        obs = info.observation
        player = obs["player"]
        return float(np.tanh((obs["scores"][player] - obs["scores"][1 - player]) / 100))

    def _shortlist(self, moves, info):
        return shortlist(moves, self.config.candidates, info.observation, self.config.shortlist)

    def _simulate(self, world, root, depth, root_moves, root_mean, tree_id=None, forced=None):
        if world.terminal:
            return self._terminal(world, root)
        info = world.information()
        player = info.observation["player"]
        if depth >= self.config.depth:
            value = self._leaf_value(world, info)
            return value if player == root else -value
        moves = (root_moves if depth == 0 else
                 self._shortlist(self.simulator.legal_moves(world, self.rng), info))
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
        index = forced if depth == 0 and forced is not None else self._select(node)
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

    def _root_stats(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        roots = [node for key, node in self.nodes.items()
                 if len(key) == 2 and key[0] == "root"]
        visits = sum((node.visits for node in roots), np.zeros(count, dtype=np.int64))
        totals = sum((node.totals for node in roots), np.zeros(count))
        return visits, totals

    def _sigma(self, q: np.ndarray, visits: np.ndarray) -> np.ndarray:
        """Gumbel AlphaZero's monotone transform of values in [-1, 1] (rescaled to [0, 1])."""
        c = self.config
        return (c.gumbel_c_visit + float(visits.max())) * c.gumbel_c_scale * (q + 1) / 2

    def _gumbel(self, moves, prior, prior_value, simulate):
        """Gumbel-top-k root sampling with sequential halving over ``simulations``.

        Returns (improved policy over all candidates, selected index).
        """
        c = self.config
        count = len(moves)
        logits = np.log(np.maximum(prior, 1e-12))
        gumbel = (self.rng.gumbel(size=count) * c.gumbel_scale if c.gumbel_scale > 0
                  else np.zeros(count))
        m = min(c.gumbel_actions, count, c.simulations)
        remaining = list(np.argsort(-(gumbel + logits), kind="stable")[:m])
        phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        used = 0

        def score(actions):
            visits, totals = self._root_stats(count)
            q = totals / np.maximum(1, visits)
            values = gumbel + logits + self._sigma(q, visits)
            return [values[a] for a in actions]

        for phase in range(phases):
            per_action = max(1, c.simulations // (phases * len(remaining)))
            for _ in range(per_action):
                for action in remaining:
                    if used < c.simulations:
                        simulate(used, int(action))
                        used += 1
            if len(remaining) > 1 and phase < phases - 1:
                order = np.argsort(-np.array(score(remaining)), kind="stable")
                remaining = [remaining[i] for i in order[:math.ceil(len(remaining) / 2)]]
        while used < c.simulations:   # rounding leftovers go to the surviving actions
            for action in remaining:
                if used < c.simulations:
                    simulate(used, int(action))
                    used += 1
        selected = int(remaining[int(np.argmax(score(remaining)))])
        visits, totals = self._root_stats(count)
        visited = visits > 0
        q = np.where(visited, totals / np.maximum(1, visits), 0.0)
        # Completed Q: unvisited actions get v_mix, the prior-weighted mix of the
        # network value and the visited actions' values.
        mass = float(prior[visited].sum())
        n = float(visits.sum())
        v_mix = prior_value if mass <= 0 else (
            (prior_value + n * float((prior[visited] * q[visited]).sum()) / mass) / (1 + n))
        completed = np.where(visited, q, v_mix)
        z = logits + self._sigma(completed, visits)
        improved = np.exp(z - z.max())
        return improved / improved.sum(), selected

    def run(self, info: InformationState, moves=None, belief: RackBelief | None = None,
            root_features: np.ndarray | None = None):
        """Search from ``info``. ``root_features`` overrides the root belief features
        (used by Reanalyse, which stores them but not the belief's particles)."""
        self.nodes = {}
        self._root_noise = None
        config = self.config
        use_belief = config.method == "belief"
        mean = self._belief_input(info, belief) if belief is not None else None
        if root_features is not None:
            mean = np.asarray(root_features)
        if not use_belief or belief is None:
            belief = RackBelief(info, config.particles, self.rng)
        root = info.observation["player"]
        first_world = self.simulator.determinize(info, belief.sample())
        if moves is None:
            moves = self.simulator.legal_moves(first_world, self.rng)
        moves = shortlist(tuple(moves), config.candidates, info.observation, config.shortlist)
        if mean is None:
            mean = self._belief_input(info, belief)
        gumbel = config.root_selection == "gumbel"
        if not gumbel and config.root_noise_fraction > 0 and len(moves) > 1:
            self._root_noise = self.rng.dirichlet(np.full(len(moves), config.root_noise_alpha))
        determinization = config.method in ("pimc", "neural_determinization")
        # Equal fixed-world budgets; all methods use exactly config.simulations rollouts.
        determinizations = min(4, max(1, config.simulations // max(1, len(moves))))
        worlds = [self.simulator.determinize(info, belief.sample())
                  for _ in range(determinizations)] if determinization else []

        def simulate(simulation, forced=None):
            tree_id = simulation % determinizations if determinization else None
            world = (worlds[tree_id] if determinization
                     else self.simulator.determinize(info, belief.sample()))
            self._simulate(world, root, 0, moves, mean, tree_id, forced)

        selected = None
        if gumbel and len(moves) > 1:
            prior, prior_value = self._prediction(info, moves, mean)
            policy, selected = self._gumbel(moves, np.asarray(prior, dtype=np.float64),
                                            float(prior_value), simulate)
        else:
            for simulation in range(config.simulations):
                simulate(simulation)
            policy = None
        visits, totals = self._root_stats(len(moves))
        if policy is None:
            policy = visits / visits.sum()
            if gumbel:
                selected = 0
        chance_outcomes = sum(len(chance.outcomes) for node in self.nodes.values()
                              for chance in node.chance.values())
        return SearchResult(moves, policy, visits, float(totals.sum()) / visits.sum(),
                            chance_outcomes, selected)
