"""Exact tile-conserving opponent-rack beliefs with optional action-likelihood filtering."""

from collections.abc import Callable
from itertools import product
from math import comb
import warnings

import numpy as np

from rackzero.game import DISTRIBUTION, InformationState, PublicEvent, board_counts, used_tiles
from scrabble_gym.environment import Move


def unseen_tiles(info: InformationState) -> np.ndarray:
    obs = info.observation
    result = DISTRIBUTION - board_counts(obs) - obs["rack"]
    if (np.any(result < 0)
            or int(result.sum()) != obs["bag_size"] + obs["opponent_rack_size"]):
        raise ValueError("Observation violates tile conservation")
    return result


def sample_tiles(pool: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count < 0 or count > int(pool.sum()) or np.any(pool < 0):
        raise ValueError("Cannot sample requested tiles without replacement")
    indexes = np.repeat(np.arange(27), pool)
    chosen = rng.choice(indexes, size=count, replace=False)
    return np.bincount(chosen, minlength=27).astype(np.int64)


def draw_probability(pool: np.ndarray, draw: np.ndarray) -> float:
    """Unordered draw probability under the multivariate hypergeometric law."""
    if np.any(draw < 0) or np.any(draw > pool):
        return 0.0
    n = int(draw.sum())
    denominator = comb(int(pool.sum()), n)
    return float(np.prod([comb(int(a), int(b)) for a, b in zip(pool, draw)],
                         dtype=object) / denominator)


Likelihood = Callable[[InformationState, PublicEvent, np.ndarray], float]
# A particle is a list of groups (tiles, m): the opponent holds a uniformly random
# m-subset of the physical unseen tiles ``tiles``; the rest of the group is in the bag.
Group = tuple[np.ndarray, int]


def _numerator(pool: np.ndarray, draw: np.ndarray) -> int:
    """prod_i C(pool_i, draw_i) as an exact integer (0 when infeasible)."""
    result = 1
    for have, take in zip(pool.tolist(), draw.tolist()):
        if take:
            if take < 0 or take > have:
                return 0
            result *= comb(have, take)
    return result


def _compositions(total: int, caps: list[int]):
    if len(caps) == 1:
        if total <= caps[0]:
            yield (total,)
        return
    for x in range(min(total, caps[0]) + 1):
        for rest in _compositions(total - x, caps[1:]):
            yield (x,) + rest


def _split_terms(target: np.ndarray, groups: list[Group], slots: list[int]):
    """Exact probabilities of every attribution of ``target`` letters to groups.

    A uniformly random n-subset is taken from positions of which ``slots[g]`` lie in
    group g; within a group those positions are a uniform subset of its tiles, so
    the letters taken from g are a uniform subset of ``tiles_g``.
    """
    n, total = int(target.sum()), sum(slots)
    if n > total:
        return [], []
    letters = np.flatnonzero(target)
    options = [list(_compositions(int(target[i]), [int(t[i]) for t, _ in groups]))
               for i in letters]
    denominator = comb(total, n)
    splits, terms = [], []
    for choice in product(*options):
        parts = np.zeros((len(groups), 27), dtype=np.int64)
        if len(letters):
            parts[:, letters] = np.array(choice, dtype=np.int64).T
        numerator, scale = 1, 1
        for (tiles, _), part, slot in zip(groups, parts, slots):
            taken = int(part.sum())
            numerator *= comb(slot, taken) * _numerator(tiles, part)
            scale *= comb(int(tiles.sum()), taken)
            if not numerator:
                break
        if numerator:
            splits.append(parts)
            terms.append(numerator / (denominator * scale))
    return splits, terms


def _normalize(groups: list[Group]) -> list[Group]:
    """Drop empty groups; merge groups wholly on the rack or wholly in the bag."""
    rack, bag, mixed = np.zeros(27, dtype=np.int64), np.zeros(27, dtype=np.int64), []
    for tiles, m in groups:
        size = int(tiles.sum())
        if not size:
            continue
        if m == size:
            rack += tiles
        elif m == 0:
            bag += tiles
        else:
            mixed.append((tiles, int(m)))
    if rack.any():
        mixed.append((rack, int(rack.sum())))
    if bag.any():
        mixed.append((bag, 0))
    return mixed


def _choose(rng: np.random.Generator, weights: list[float]) -> int:
    weights = np.asarray(weights, dtype=np.float64)
    return int(rng.choice(len(weights), p=weights / weights.sum()))


def _counts(rng: np.random.Generator, sizes: list[int], n: int) -> np.ndarray:
    """Multivariate hypergeometric split of n uniformly chosen positions."""
    if n < 0 or n > sum(sizes):
        raise ValueError("Cannot draw more tiles than available")
    if n == 0:
        return np.zeros(len(sizes), dtype=np.int64)
    return rng.multivariate_hypergeometric(np.asarray(sizes, dtype=np.int64), n)


class LeaveLikelihood:
    """Opponent model "players keep their best tiles": P(public action | full rack).

    Relative, not absolute: a placement that leaves ``k`` tiles is more likely
    from racks where the kept tiles are close to the best k tiles that rack could
    have kept, by the heuristic leave formula ``rackzero.search.rack_value``
    (points): P = exp(-beta * (best k-tile leave - actual leave)) <= 1. (Word
    constraints are ignored: any k-subset counts as keepable.) An exchange of n
    tiles is likelier from racks whose best (7-n)-tile keep is much better than
    keeping everything. Passes carry no information. beta is in 1/points.
    """

    def __init__(self, beta: float):
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        self.beta = float(beta)

    @staticmethod
    def best_keep(rack: np.ndarray, k: int) -> float:
        from itertools import combinations
        from rackzero.search import rack_value
        tiles = np.repeat(np.arange(27), np.asarray(rack, dtype=np.int64))
        if k <= 0:
            return rack_value(np.zeros(27, dtype=np.int64))
        if k >= len(tiles):
            return rack_value(rack)
        best = -np.inf
        for subset in set(combinations(tiles.tolist(), k)):
            best = max(best, rack_value(np.bincount(subset, minlength=27)))
        return float(best)

    def __call__(self, info: InformationState, event: PublicEvent, rack: np.ndarray) -> float:
        from rackzero.search import rack_value
        rack = np.asarray(rack, dtype=np.int64)
        if event.kind == "place":
            leave = rack - np.asarray(event.played, dtype=np.int64)
            gap = self.best_keep(rack, int(leave.sum())) - rack_value(leave)
        elif event.kind == "exchange":
            keep = int(rack.sum()) - int(event.exchange_count)
            # how much better than its whole rack this rack's best keep is
            gap = max(0.0, rack_value(rack) - self.best_keep(rack, keep) + 10.0)
        else:
            return 1.0
        return float(np.exp(-self.beta * max(0.0, gap)))


def leave_likelihood(beta: float) -> "LeaveLikelihood | None":
    """The ``search.belief_beta`` opponent model (None = uniform, the default)."""
    return LeaveLikelihood(beta) if beta > 0 else None


class RackBelief:
    """Rao-Blackwellized particle filter over the opponent's hidden rack.

    Each particle partitions the observer's unseen tiles into groups ``(tiles, m)``:
    the opponent holds a uniformly random m-subset of ``tiles`` and the rest are in
    the bag. The prior is one group with m = opponent rack size. Tiles an observer
    exchanges become an m = 0 group (the engine draws before returning, so they
    cannot be on the opponent's current rack); groups gain m as the opponent draws.

    Default opponent model: the tiles played or exchanged are a uniformly random
    physical subset of the rack of the observed size. This uniform-subset form is
    closed under every transition, so filtering is exact given each particle's
    sampled discrete choices (which group supplied which tiles); those are sampled
    from their exact posterior and the particle weight is the exact marginal
    likelihood of the observation. No letter of an unknown tile is ever sampled.

    ``likelihood(info, event, rack)`` optionally gives P(public action | full rack)
    (e.g. a learned or softmax opponent policy). Racks are then resolved with the
    default-model posterior as proposal and importance-weighted.
    """

    def __init__(self, info: InformationState, particles: int = 32,
                 rng: np.random.Generator | None = None,
                 likelihood: Likelihood | None = None):
        if particles < 1:
            raise ValueError("particles must be positive")
        self.info = info
        self.rng = rng if rng is not None else np.random.default_rng()
        self.likelihood = likelihood
        prior = _normalize([(unseen_tiles(info), int(info.observation["opponent_rack_size"]))])
        self.particles: list[list[Group]] = [list(prior) for _ in range(particles)]
        self.weights = np.full(particles, 1 / particles, dtype=np.float64)
        self.resets = 0
        self._check()

    @property
    def racks(self) -> np.ndarray:
        """One full rack sample per particle (unknown parts drawn uniformly)."""
        return np.stack([self._resolve_sample(groups) for groups in self.particles])

    @racks.setter
    def racks(self, racks: np.ndarray) -> None:
        unseen = unseen_tiles(self.info)
        self.particles = [_normalize([(np.asarray(r, dtype=np.int64).copy(), int(np.sum(r))),
                                      (unseen - r, 0)]) for r in racks]
        if len(self.particles) != len(self.weights):
            self.weights = np.full(len(self.particles), 1 / len(self.particles))
        self._check()

    def _resolve_sample(self, groups: list[Group]) -> np.ndarray:
        rack = np.zeros(27, dtype=np.int64)
        for tiles, m in groups:
            rack += sample_tiles(tiles, m, self.rng)
        return rack

    def _check(self) -> None:
        obs = self.info.observation
        unseen = unseen_tiles(self.info)
        for groups in self.particles:
            tiles = sum((t for t, _ in groups), np.zeros(27, dtype=np.int64))
            held = sum(m for _, m in groups)
            bag = sum(int(t.sum()) - m for t, m in groups)
            if (any(np.any(t < 0) or not 0 <= m <= t.sum() for t, m in groups)
                    or not np.array_equal(tiles, unseen)
                    or held != obs["opponent_rack_size"] or bag != obs["bag_size"]):
                raise RuntimeError("Belief state violated tile conservation")

    def sample(self) -> np.ndarray:
        i = int(self.rng.choice(len(self.weights), p=self.weights))
        return self._resolve_sample(self.particles[i])

    def expected(self) -> np.ndarray:
        means = [sum((t * (m / t.sum()) for t, m in groups), np.zeros(27))
                 for groups in self.particles]
        return np.average(np.array(means, dtype=np.float64), axis=0, weights=self.weights)

    def condition(self, event: PublicEvent) -> None:
        """Bayes' rule for an opponent action without its transition (no removal/draw)."""
        if event.player == self.info.observation["player"]:
            raise ValueError("Opponent likelihood cannot condition on own action")
        used = np.asarray(event.played, dtype=np.int64)
        factors = []
        for i, groups in enumerate(self.particles):
            if self.likelihood is None:
                factors.append(sum(_split_terms(used, groups, [m for _, m in groups])[1])
                               if event.kind == "place" else 1.0)
                continue
            rack = self._resolve_sample(groups)
            unseen = unseen_tiles(self.info)
            self.particles[i] = _normalize([(rack, int(rack.sum())), (unseen - rack, 0)])
            factors.append(self._likelihood(event, rack) if np.all(rack >= used) else 0.0)
        if not self._reweight(factors):
            raise ValueError("Observed action has zero particle likelihood; "
                             "increase particle budget")

    def update(self, event: PublicEvent, next_info: InformationState,
               own_move: Move | None = None) -> None:
        """Exact sequential Bayes update for one observer across a public turn."""
        before, after = self.info.observation, next_info.observation
        if before["player"] != after["player"]:
            raise ValueError("Belief updates must retain the observer's perspective")
        if event.player == before["player"]:
            if own_move is None:
                raise ValueError("Own move is required to infer the private draw")
            spent = used_tiles(before, own_move)
            draw = after["rack"] - before["rack"] + spent
            returned = spent if event.kind == "exchange" else None
            factors = [self._own(i, draw, returned) for i in range(len(self.particles))]
        else:
            drawn = int(after["opponent_rack_size"] - before["opponent_rack_size"]
                        + sum(event.played))
            factors = [self._opponent(i, event, drawn) for i in range(len(self.particles))]
        self.info = next_info
        if not self._reweight(factors):
            warnings.warn("Rack belief particle depletion: resetting to tile-conserving prior; "
                          "history information was lost. Increase particles.",
                          RuntimeWarning, stacklevel=2)
            resets = self.resets + 1
            self.__init__(next_info, len(self.particles), self.rng, self.likelihood)
            self.resets = resets
            return
        self._check()

    def _likelihood(self, event: PublicEvent, rack: np.ndarray) -> float:
        probability = self.likelihood(self.info, event, rack.copy())
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Opponent action likelihood must lie in [0,1]")
        return float(probability)

    def _reweight(self, factors: list[float]) -> bool:
        factors = np.asarray(factors, dtype=np.float64)
        if not np.all(np.isfinite(factors)) or np.any(factors < 0):
            raise ValueError("Invalid particle weight")
        weights = self.weights * factors
        if weights.sum() <= 0:
            return False
        self.weights = weights / weights.sum()
        # Zero-weight particles were not transitioned, so they must be resampled away.
        if (np.any(self.weights == 0)
                or 1 / np.square(self.weights).sum() < len(self.weights) / 2):
            keep = self.rng.choice(len(self.weights), len(self.weights), p=self.weights)
            self.particles = [list(self.particles[int(k)]) for k in keep]
            self.weights = np.full(len(keep), 1 / len(keep), dtype=np.float64)
        return True

    def _own(self, i: int, draw: np.ndarray, returned: np.ndarray | None) -> float:
        """Exact P(own private draw | particle); then remove the drawn tiles."""
        groups = self.particles[i]
        splits, terms = _split_terms(draw, groups, [int(t.sum()) - m for t, m in groups])
        if not terms:
            return 0.0
        parts = splits[_choose(self.rng, terms)]
        groups = [(t - part, m) for (t, m), part in zip(groups, parts)]
        if returned is not None:
            # The engine draws before returning: exchanged tiles are in the bag and
            # cannot be on the opponent's current rack.
            groups.append((returned.astype(np.int64), 0))
        self.particles[i] = _normalize(groups)
        return float(sum(terms))

    def _opponent(self, i: int, event: PublicEvent, drawn: int) -> float:
        groups = self.particles[i]
        factor = 1.0
        if event.kind == "place":
            used = np.asarray(event.played, dtype=np.int64)
            splits, terms = _split_terms(used, groups, [m for _, m in groups])
            if not terms:
                return 0.0
            parts = splits[_choose(self.rng, terms)]
            factor = float(sum(terms))
            groups = [(t - part, m - int(part.sum())) for (t, m), part in zip(groups, parts)]
            if self.likelihood is not None:
                groups, rack = self._resolve(groups)
                rack_before = rack + used
                factor *= (self._likelihood(event, rack_before)
                           / draw_probability(rack_before, used))
        elif self.likelihood is not None:
            groups, rack = self._resolve(groups)
            factor = self._likelihood(event, rack)
        if factor <= 0:
            return 0.0
        bag_sizes = [int(t.sum()) - m for t, m in groups]
        if event.kind == "exchange":
            returned = _counts(self.rng, [m for _, m in groups], event.exchange_count)
            # Draw from the pre-return bag; returned positions re-enter their group.
            gained = _counts(self.rng, bag_sizes, event.exchange_count) - returned
        else:
            gained = _counts(self.rng, bag_sizes, drawn)
        self.particles[i] = _normalize([(t, m + int(g)) for (t, m), g in zip(groups, gained)])
        return factor

    def _resolve(self, groups: list[Group]) -> tuple[list[Group], np.ndarray]:
        resolved, rest = [], []
        for tiles, m in groups:
            held = sample_tiles(tiles, m, self.rng)
            resolved.append(held)
            rest.append((tiles - held, 0))
        rack = sum(resolved, np.zeros(27, dtype=np.int64))
        return _normalize([(rack, int(rack.sum()))] + rest), rack


class SoftmaxScorePolicy:
    """Optional normalized opponent-action model; callback can instead be learned."""

    def __init__(self, simulator, temperature: float = 10):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.simulator, self.temperature = simulator, temperature

    def __call__(self, info: InformationState, event: PublicEvent, rack: np.ndarray) -> float:
        world = self.simulator.determinize(info, rack)
        obs = dict(world.observation)
        obs["player"] = event.player
        world.observation = obs
        moves = self.simulator.legal_moves(world, np.random.default_rng(0))
        scores = np.array([m.score for m in moves], dtype=np.float64) / self.temperature
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        opponent_info = world.information()
        result = 0.0
        for move, probability in zip(moves, probs):
            if move.move.kind != event.kind or move.score != event.score:
                continue
            if event.kind == "place":
                if event.move is not None and move.move != event.move:
                    continue
                if tuple(used_tiles(opponent_info.observation, move.move)) != event.played:
                    continue
            elif event.kind == "exchange" and len(move.move.exchange) != event.exchange_count:
                continue
            result += float(probability)
        return result
