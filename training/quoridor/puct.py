"""History-aware, cooperatively bounded PUCT for the new rules contract.

The evaluator returns original-coordinate logits and a current-player value.
This module imports no tensor or neural code. Root expansion is not a counted
simulation; each completed simulation backs up one root edge. Policy targets
are raw normalized visits, independently of the temperature used to pick an
action. An interrupted search returns diagnostics but never an actionable move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Protocol

from . import engine
from .episode import Episode, Outcome, advance_episode
from .project_runtime import BudgetExpired, Deadline


SEARCH_VERSION = "puct-v1"
POLICY_TARGET_VERSION = "root-visits-v1"
AdjudicationKey = tuple[tuple[str, ...], int, Outcome | None]


class SearchError(RuntimeError):
    """Search cannot provide a valid result; no fallback policy is invented."""


class InvalidEvaluationError(SearchError):
    """The evaluator violated the finite logits/current-player value contract."""


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 64
    c_puct: float = 1.5
    mode: str = "evaluation"
    root_noise: bool = False
    temperature: float = 0.0
    seed: int = 0
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25

    def __post_init__(self) -> None:
        if type(self.simulations) is not int or self.simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        if type(self.mode) is not str or self.mode not in {"evaluation", "selfplay"}:
            raise ValueError("mode must be evaluation or selfplay")
        if type(self.root_noise) is not bool:
            raise ValueError("root_noise must be a boolean")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("c_puct", "dirichlet_alpha"):
            if _finite_number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if _finite_number(self.temperature, "temperature") < 0:
            raise ValueError("temperature must be nonnegative")
        if not 0 <= _finite_number(self.dirichlet_fraction, "dirichlet_fraction") <= 1:
            raise ValueError("dirichlet_fraction must be between zero and one")
        if self.mode == "evaluation" and (self.root_noise or self.temperature != 0):
            raise ValueError("evaluation requires root_noise=false and temperature=0")


@dataclass(frozen=True, slots=True)
class Evaluation:
    """An evaluator response, validated when consumed by the search."""

    logits: tuple[float, ...]
    value: float


class Evaluator(Protocol):
    def __call__(self, episode: Episode, *, deadline: Deadline) -> Evaluation: ...


@dataclass(frozen=True, slots=True)
class NodeStats:
    adjudication_key: AdjudicationKey
    priors: tuple[float, ...]
    visits: tuple[int, ...]
    value_sums: tuple[float, ...]
    legal_actions: tuple[int, ...]
    initial_value: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    status: str
    action: int | None
    root_value: float | None
    visits: tuple[int, ...]
    policy: tuple[float, ...]
    priors: tuple[float, ...]
    value_sums: tuple[float, ...]
    nodes: tuple[NodeStats, ...]
    completed_simulations: int
    cycle_cutoffs: int
    terminal_leaves: int
    truncated_leaves: int
    evaluations: int
    elapsed_seconds: float
    budget_seconds: float
    overrun_seconds: float
    search_version: str = SEARCH_VERSION
    policy_target_version: str = POLICY_TARGET_VERSION


@dataclass(slots=True)
class _Node:
    episode: Episode
    legal_actions: tuple[int, ...]
    priors: tuple[float, ...]
    initial_value: float
    visits: list[int] = field(default_factory=lambda: [0] * engine.ACTION_SPACE_SIZE)
    value_sums: list[float] = field(default_factory=lambda: [0.0] * engine.ACTION_SPACE_SIZE)

    def snapshot(self) -> NodeStats:
        return NodeStats(
            self.episode.adjudication_key, self.priors, tuple(self.visits),
            tuple(self.value_sums), self.legal_actions, self.initial_value,
        )


class _BudgetedRandom(random.Random):
    """Check inside rejection samplers, which may need unbounded draws."""

    def __new__(cls, seed: int, deadline: Deadline) -> _BudgetedRandom:
        # The C Random allocator accepts at most one positional argument.
        return super().__new__(cls)

    def __init__(self, seed: int, deadline: Deadline) -> None:
        self._deadline = deadline
        super().__init__(seed)

    def random(self) -> float:
        self._deadline.check()
        return super().random()


def _validated_evaluation(result: object) -> tuple[tuple[float, ...], float]:
    if not isinstance(result, Evaluation):
        raise InvalidEvaluationError("evaluator must return Evaluation")
    if type(result.logits) is not tuple or len(result.logits) != engine.ACTION_SPACE_SIZE:
        raise InvalidEvaluationError("evaluation logits must be an immutable tuple of 209 numbers")
    try:
        logits = tuple(_finite_number(logit, "logit") for logit in result.logits)
        value = _finite_number(result.value, "value")
    except ValueError as error:
        raise InvalidEvaluationError(str(error)) from error
    if not -1 <= value <= 1:
        raise InvalidEvaluationError("evaluation value must be between -1 and +1")
    return logits, value


def _priors(logits: tuple[float, ...], legal: tuple[int, ...]) -> tuple[float, ...]:
    if not legal:
        raise SearchError("an unfinished episode has no legal actions")
    peak = max(logits[action] for action in legal)
    weights = [math.exp(logits[action] - peak) for action in legal]
    total = math.fsum(weights)
    probabilities = [0.0] * engine.ACTION_SPACE_SIZE
    for action, weight in zip(legal, weights):
        probabilities[action] = weight / total
    return tuple(probabilities)


def _root_noise(
    priors: tuple[float, ...], legal: tuple[int, ...], config: SearchConfig,
    rng: random.Random, deadline: Deadline,
) -> tuple[float, ...]:
    samples = []
    for _ in legal:
        deadline.check()
        samples.append(rng.gammavariate(config.dirichlet_alpha, 1.0))
        deadline.check()
    try:
        total = math.fsum(samples)
    except OverflowError as error:
        raise SearchError("Dirichlet sampling produced an invalid normalization") from error
    if not math.isfinite(total) or total <= 0:
        raise SearchError("Dirichlet sampling produced an invalid normalization")
    probabilities = [0.0] * engine.ACTION_SPACE_SIZE
    fraction = config.dirichlet_fraction
    for action, sample in zip(legal, samples):
        probabilities[action] = (1 - fraction) * priors[action] + fraction * sample / total
    return tuple(probabilities)


def _select(node: _Node, config: SearchConfig) -> int:
    scale = math.sqrt(max(1, sum(node.visits)))
    best_action = node.legal_actions[0]
    best_score = -math.inf
    for action in node.legal_actions:
        visits = node.visits[action]
        q = node.value_sums[action] / visits if visits else 0.0
        score = q + config.c_puct * node.priors[action] * scale / (1 + visits)
        if not math.isfinite(score):
            raise SearchError("PUCT score became nonfinite")
        # Engine actions are sorted: strict comparison retains the smallest ID.
        if score > best_score:
            best_action, best_score = action, score
    return best_action


def _choose_action(root: _Node, config: SearchConfig, rng: random.Random) -> int:
    if config.temperature == 0:
        return max(root.legal_actions, key=lambda action: (root.visits[action], -action))
    visited = [action for action in root.legal_actions if root.visits[action] > 0]
    log_peak = math.log(max(root.visits[action] for action in visited))
    weights = [
        math.exp((math.log(root.visits[action]) - log_peak) / config.temperature)
        for action in visited
    ]
    return rng.choices(visited, weights=weights, k=1)[0]


def run_search(
    episode: Episode, evaluator: Evaluator, config: SearchConfig, *, deadline: Deadline,
) -> SearchResult:
    """Search one exact episode, with a fresh tree and a required deadline.

    Already finished roots are reported without inference, even when the budget
    has expired. For live roots, expiration wins over further search work. A
    simulation-local repeated position contributes a zero *estimate* only; the
    immutable input episode and its replay eligibility never change. An actual
    goal/repetition outcome takes precedence over that approximation. A simulated
    unlabelled stop aborts search without backing up that incomplete simulation.

    Deadlines bracket operations, including inference, but cannot preempt Python
    or accelerator work. Overrun is reported. A backup is committed atomically
    between checks so partial statistics never masquerade as a simulation.
    """

    if not isinstance(episode, Episode):
        raise TypeError("search requires a quoridor-v1 Episode")
    if not isinstance(config, SearchConfig):
        raise TypeError("search requires SearchConfig")
    if not isinstance(deadline, Deadline):
        raise TypeError("search requires project_runtime.Deadline")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")

    nodes: dict[AdjudicationKey, _Node] = {}
    root: _Node | None = None
    root_value: float | None = None
    completed = cycle_cutoffs = terminal_leaves = truncated_leaves = evaluations = 0
    rng = _BudgetedRandom(config.seed, deadline)

    def finish(status: str, action: int | None = None) -> SearchResult:
        visits = tuple(root.visits) if root is not None else (0,) * engine.ACTION_SPACE_SIZE
        priors = root.priors if root is not None else (0.0,) * engine.ACTION_SPACE_SIZE
        sums = tuple(root.value_sums) if root is not None else (0.0,) * engine.ACTION_SPACE_SIZE
        total = sum(visits)
        policy = tuple(count / total for count in visits) if total else (0.0,) * engine.ACTION_SPACE_SIZE
        snapshots = tuple(node.snapshot() for node in nodes.values())
        elapsed = deadline.elapsed_seconds
        # Include selection/snapshot time in the result's budget accounting.
        if status in {"completed", "simulation_truncated"} and elapsed >= deadline.seconds:
            status, action = "budget_exhausted", None
        return SearchResult(
            status, action, root_value, visits, policy, priors, sums, snapshots,
            completed, cycle_cutoffs, terminal_leaves, truncated_leaves,
            evaluations, elapsed, deadline.seconds, max(0.0, elapsed - deadline.seconds),
        )

    def expand(current: Episode, *, is_root: bool = False) -> _Node:
        nonlocal evaluations
        deadline.check()
        legal = engine.legal_actions(current.state)
        deadline.check()
        evaluations += 1
        try:
            response = evaluator(current, deadline=deadline)
        except (BudgetExpired, SearchError):
            raise
        except Exception as error:
            raise SearchError(f"evaluator failed: {error}") from error
        deadline.check()
        logits, value = _validated_evaluation(response)
        priors = _priors(logits, legal)
        deadline.check()
        if is_root and config.root_noise:
            priors = _root_noise(priors, legal, config, rng, deadline)
        deadline.check()
        node = _Node(current, legal, priors, value)
        nodes[current.adjudication_key] = node
        return node

    if episode.outcome is not None:
        if episode.outcome.terminated:
            root_value = episode.outcome.value_for(episode.state.to_play)
            return finish("terminal")
        return finish("root_stopped")

    try:
        deadline.check()
        root = expand(episode, is_root=True)
        root_value = root.initial_value
        for _ in range(config.simulations):
            deadline.check()
            node = root
            path: list[tuple[_Node, int]] = []
            ancestors = {episode.history[-1]}
            while True:
                deadline.check()
                action = _select(node, config)
                deadline.check()
                # The search deadline belongs to search, not the actual episode.
                child = advance_episode(node.episode, action)
                deadline.check()
                path.append((node, action))
                if child.outcome is not None:
                    if child.outcome.terminated:
                        terminal_leaves += 1
                        value = child.outcome.value_for(child.state.to_play)
                        break
                    if child.outcome.truncated:
                        truncated_leaves += 1
                        return finish("simulation_truncated")
                    raise SearchError(f"legal search transition failed: {child.outcome.reason}")
                if child.history[-1] in ancestors:
                    cycle_cutoffs += 1
                    value = 0.0
                    break
                ancestors.add(child.history[-1])
                child_node = nodes.get(child.adjudication_key)
                if child_node is None:
                    child_node = expand(child)
                    value = child_node.initial_value
                    break
                node = child_node
            deadline.check()
            for parent, action in reversed(path):
                value = -value
                parent.visits[action] += 1
                parent.value_sums[action] += value
            completed += 1
            deadline.check()
        deadline.check()
        action = _choose_action(root, config, rng)
        deadline.check()
        return finish("completed", action)
    except BudgetExpired:
        return finish("budget_exhausted")


__all__ = [
    "SEARCH_VERSION", "POLICY_TARGET_VERSION", "SearchConfig", "Evaluation", "Evaluator",
    "NodeStats", "SearchResult", "SearchError", "InvalidEvaluationError", "run_search",
]
