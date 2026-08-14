"""AlphaZero PUCT search over the complete legal Quoridor action set."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .codec import canonicalize_state, to_canonical_action_index
from .game import (
    State,
    apply_action_index,
    is_terminal,
    legal_action_indices,
    winner_value,
)


ACTION_SIZE = 209


class Evaluator(Protocol):
    """Network adapter used by search.

    The state passed to ``evaluate`` is canonicalized to the player-to-move
    perspective. The returned value must use that same player's perspective.
    """

    def evaluate(self, state: State) -> tuple[Sequence[float], float]: ...


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 800
    c_base: float = 19_652.0
    c_init: float = 1.25
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float | None = None
    temperature: float = 1.0
    add_root_noise: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.simulations, int)
            or isinstance(self.simulations, bool)
            or self.simulations <= 0
        ):
            raise ValueError("simulations must be a positive integer")
        if not math.isfinite(self.c_base) or self.c_base <= 0:
            raise ValueError("c_base must be finite and positive")
        if not math.isfinite(self.c_init) or self.c_init < 0:
            raise ValueError("c_init must be finite and non-negative")
        if (
            not math.isfinite(self.dirichlet_epsilon)
            or not 0 <= self.dirichlet_epsilon <= 1
        ):
            raise ValueError("dirichlet_epsilon must be between 0 and 1")
        if self.dirichlet_alpha is not None and (
            not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0
        ):
            raise ValueError("dirichlet_alpha must be finite and positive or None")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if self.seed is not None and (
            type(self.seed) is not int or not 0 <= self.seed < 2**63
        ):
            raise ValueError("seed must be None or an integer in [0, 2^63)")


@dataclass
class EdgeStats:
    """Statistics for an action, always stored from its parent player's view."""

    prior: float
    N: int = 0
    W: float = 0.0
    child: _Node | None = field(default=None, repr=False)

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0

    @property
    def visit_count(self) -> int:
        return self.N

    @property
    def value_sum(self) -> float:
        return self.W

    @property
    def q(self) -> float:
        return self.Q


@dataclass
class _Node:
    state: State
    edges: dict[int, EdgeStats] = field(default_factory=dict)
    expanded: bool = False
    network_value: float | None = None

    @property
    def visit_count(self) -> int:
        return sum(edge.N for edge in self.edges.values())


@dataclass(frozen=True)
class SearchResult:
    action_index: int | None
    policy: tuple[float, ...]
    visit_counts: tuple[int, ...]
    root_value: float
    root_edges: Mapping[int, EdgeStats]


def _as_float(value: object, name: str) -> float:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain numeric scalars") from error
    return result


def _softmax_legal(logits: Sequence[float]) -> list[float]:
    positive_infinities = [index for index, value in enumerate(logits) if value == math.inf]
    if positive_infinities:
        probability = 1.0 / len(positive_infinities)
        return [probability if index in positive_infinities else 0.0 for index in range(len(logits))]

    finite_logits = [value if math.isfinite(value) else -math.inf for value in logits]
    maximum = max(finite_logits)
    if maximum == -math.inf:
        return [1.0 / len(finite_logits)] * len(finite_logits)

    weights = [math.exp(value - maximum) if value != -math.inf else 0.0 for value in finite_logits]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        return [1.0 / len(finite_logits)] * len(finite_logits)
    return [weight / total for weight in weights]


class MCTS:
    def __init__(self, evaluator: Evaluator, config: MCTSConfig | None = None) -> None:
        if not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("evaluator must define evaluate(state)")
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self._rng = random.Random(self.config.seed)

    def run(
        self,
        state: State,
        *,
        temperature: float | None = None,
        add_root_noise: bool | None = None,
    ) -> SearchResult:
        selected_temperature = self.config.temperature if temperature is None else temperature
        if not math.isfinite(selected_temperature) or selected_temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        use_noise = self.config.add_root_noise if add_root_noise is None else add_root_noise

        if is_terminal(state):
            terminal_value = winner_value(state)
            if terminal_value is None:
                raise RuntimeError("terminal state has no winner value")
            return SearchResult(
                action_index=None,
                policy=(0.0,) * ACTION_SIZE,
                visit_counts=(0,) * ACTION_SIZE,
                root_value=float(terminal_value),
                root_edges={},
            )

        root = _Node(state)
        self._expand(root)
        if use_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.config.simulations):
            self._simulate(root)

        visit_counts = [0] * ACTION_SIZE
        for action_index, edge in root.edges.items():
            visit_counts[action_index] = edge.N
        policy = self._visit_policy(root, selected_temperature)
        action_index = self._sample_action(root, policy, selected_temperature)
        root_visits = root.visit_count
        root_value = (
            sum(edge.W for edge in root.edges.values()) / root_visits
            if root_visits
            else float(root.network_value or 0.0)
        )
        return SearchResult(
            action_index=action_index,
            policy=tuple(policy),
            visit_counts=tuple(visit_counts),
            root_value=root_value,
            root_edges=dict(root.edges),
        )

    def _expand(self, node: _Node) -> float:
        legal_actions = tuple(legal_action_indices(node.state))
        if not legal_actions:
            raise RuntimeError("non-terminal state has no legal actions")
        if len(set(legal_actions)) != len(legal_actions):
            raise ValueError("legal_action_indices returned duplicate actions")
        if any(action < 0 or action >= ACTION_SIZE for action in legal_actions):
            raise ValueError("legal action index is outside the 209-action space")

        original_player = node.state.current_player
        canonical_state = canonicalize_state(node.state)
        policy_logits, value = self.evaluator.evaluate(canonical_state)
        try:
            policy_size = len(policy_logits)
        except TypeError as error:
            raise TypeError("policy logits must be a sequence of length 209") from error
        if policy_size != ACTION_SIZE:
            raise ValueError(f"policy logits must have length {ACTION_SIZE}")

        canonical_actions = [
            to_canonical_action_index(action, original_player)
            for action in legal_actions
        ]
        legal_logits = [
            _as_float(policy_logits[action], "policy logits")
            for action in canonical_actions
        ]
        priors = _softmax_legal(legal_logits)
        node.edges = {
            action: EdgeStats(prior=prior)
            for action, prior in zip(legal_actions, priors)
        }
        node.expanded = True
        node.network_value = _as_float(value, "value")
        if not math.isfinite(node.network_value):
            raise ValueError("value must be finite")
        node.network_value = max(-1.0, min(1.0, node.network_value))
        return node.network_value

    def _add_dirichlet_noise(self, root: _Node) -> None:
        if not root.edges or self.config.dirichlet_epsilon == 0:
            return
        actions = tuple(root.edges)
        alpha = self.config.dirichlet_alpha
        if alpha is None:
            alpha = min(0.30, max(0.03, 10.0 / len(actions)))
        noise = [
            self._rng.gammavariate(alpha, 1.0)
            for _ in actions
        ]
        total = sum(noise)
        if total <= 0 or not math.isfinite(total):
            noise = [1.0 / len(actions)] * len(actions)
        else:
            noise = [value / total for value in noise]

        epsilon = self.config.dirichlet_epsilon
        for action, sample in zip(actions, noise):
            edge = root.edges[action]
            edge.prior = (1.0 - epsilon) * edge.prior + epsilon * sample

    def _simulate(self, root: _Node) -> None:
        node = root
        path: list[EdgeStats] = []

        while node.expanded:
            action, edge = self._select_edge(node)
            path.append(edge)
            if edge.child is None:
                edge.child = _Node(apply_action_index(node.state, action))
            node = edge.child
            if is_terminal(node.state):
                break

        if is_terminal(node.state):
            value = winner_value(node.state)
            if value is None:
                raise RuntimeError("terminal state has no winner value")
            backed_value = float(value)
        else:
            backed_value = self._expand(node)

        for edge in reversed(path):
            backed_value = -backed_value
            edge.N += 1
            edge.W += backed_value

    def _select_edge(self, node: _Node) -> tuple[int, EdgeStats]:
        parent_visits = node.visit_count
        dynamic_c = (
            math.log((parent_visits + self.config.c_base + 1.0) / self.config.c_base)
            + self.config.c_init
        )
        sqrt_parent = math.sqrt(parent_visits + 1.0)

        best_action = -1
        best_edge: EdgeStats | None = None
        best_score = -math.inf
        for action, edge in node.edges.items():
            exploration = (
                dynamic_c * edge.prior * sqrt_parent / (1.0 + edge.N)
            )
            score = edge.Q + exploration
            if score > best_score or (score == best_score and action < best_action):
                best_action = action
                best_edge = edge
                best_score = score
        if best_edge is None:
            raise RuntimeError("cannot select from an unexpanded node")
        return best_action, best_edge

    @staticmethod
    def _visit_policy(root: _Node, temperature: float) -> list[float]:
        policy = [0.0] * ACTION_SIZE
        if temperature == 0:
            best_action = max(
                root.edges,
                key=lambda action: (
                    root.edges[action].N,
                    root.edges[action].prior,
                    -action,
                ),
            )
            policy[best_action] = 1.0
            return policy

        visited = [
            (action, edge.N)
            for action, edge in root.edges.items()
            if edge.N > 0
        ]
        if not visited:
            for action, edge in root.edges.items():
                policy[action] = edge.prior
            return policy

        scaled_logs = [math.log(visits) / temperature for _, visits in visited]
        maximum = max(scaled_logs)
        weights = [math.exp(value - maximum) for value in scaled_logs]
        total = sum(weights)
        for (action, _), weight in zip(visited, weights):
            policy[action] = weight / total
        return policy

    def _sample_action(
        self,
        root: _Node,
        policy: Sequence[float],
        temperature: float,
    ) -> int:
        if temperature == 0:
            return max(
                root.edges,
                key=lambda action: (policy[action], root.edges[action].prior, -action),
            )

        threshold = self._rng.random()
        cumulative = 0.0
        last_action = next(iter(root.edges))
        for action in root.edges:
            probability = policy[action]
            if probability <= 0:
                continue
            last_action = action
            cumulative += probability
            if threshold < cumulative:
                return action
        return last_action


def run(
    state: State,
    evaluator: Evaluator,
    config: MCTSConfig | None = None,
    *,
    temperature: float | None = None,
    add_root_noise: bool | None = None,
) -> SearchResult:
    """One-shot convenience wrapper suitable for simple self-play workers."""

    return MCTS(evaluator, config).run(
        state,
        temperature=temperature,
        add_root_noise=add_root_noise,
    )


__all__ = [
    "ACTION_SIZE",
    "EdgeStats",
    "Evaluator",
    "MCTSConfig",
    "SearchResult",
    "MCTS",
    "run",
]
