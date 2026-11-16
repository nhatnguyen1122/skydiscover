"""Per-island lineage tree: nodes, attempt records, rewards, backup, selection.

Implements Sections 2-6 of the LineageEvolve method specification.  This
module is pure search-state logic: it never calls an LLM or the evaluator,
and it holds program *ids* rather than programs, so it can be unit-tested
and (de)serialised on its own.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

STATUS_ADMITTED = "admitted"
STATUS_CRASHED = "crashed"
STATUS_DUPLICATE = "duplicate"

UNSPECIFIED = "unspecified"

P_FLOOR = 1e-12
_BETA_FLOOR = 1e-9


def numerical_tolerance(epsilon: float, score: float) -> float:
    """Evaluator noise level plus the numerical tolerance of Section 3.2."""
    return epsilon + 1e-12 * max(1.0, abs(score))


@dataclass
class AttemptRecord:
    """One expansion attempt at a node (Section 2.2)."""

    id: str
    parent: str
    island: int
    iteration: int
    status: str
    child: Optional[str] = None
    intent: Dict[str, Any] = field(default_factory=dict)
    delta: Optional[float] = None
    y_parent: int = 0
    q: float = 1.0
    shares: Dict[str, float] = field(default_factory=dict)  # strategy id -> c_{e,s}
    ext_shares: Dict[str, float] = field(default_factory=dict)  # source id -> a_{e,j}
    prompt_sources: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    parent_score: Optional[float] = None
    error: Optional[str] = None
    strategy_id: Optional[str] = None
    strategy_name: str = UNSPECIFIED
    # False for a failed try that was retried within its iteration and is
    # kept only for the attempt log (no backup, no memory evidence).
    evidence: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttemptRecord":
        return cls(**data)


@dataclass
class LineageNode:
    """One admitted program in an island's lineage tree (Section 2.1)."""

    id: str  # program id
    score: float
    island: int
    parent_node: Optional[str] = None
    depth: int = 0
    iteration: int = 0
    incoming_edit: Optional[str] = None  # attempt id
    alpha: float = 1.0
    beta: float = 1.0
    mu0: float = 0.5
    provenance: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    attempts: List[AttemptRecord] = field(default_factory=list)

    @property
    def p_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def is_root(self) -> bool:
        return self.parent_node is None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["attempts"] = [a.to_dict() for a in self.attempts]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageNode":
        data = dict(data)
        attempts = [AttemptRecord.from_dict(a) for a in data.pop("attempts", [])]
        node = cls(**data)
        node.attempts = attempts
        return node


@dataclass
class SelectionInfo:
    """Quantities logged for the selected node (Section 13)."""

    node_id: str
    island: int
    R: float
    alpha: float
    beta: float
    p_mean: float
    p_sample: float
    index: float
    theta: Optional[float]
    best_score: float
    tau: float
    num_nodes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LineageTree:
    """Lineage tree and island-level scalars of one island."""

    def __init__(
        self,
        island: int,
        gamma: float = 0.9,
        kappa: float = 2.0,
        rho: float = 0.9,
        epsilon: float = 0.0,
    ) -> None:
        self.island = island
        self.gamma = gamma
        self.kappa = kappa
        self.rho = rho
        self.epsilon = epsilon

        self.nodes: Dict[str, LineageNode] = {}
        self.roots: List[str] = []

        # Island-level scalars (Section 2.3)
        self.best_score: float = float("-inf")
        self.theta: Optional[float] = None
        self.attempts_since_progress: int = 0
        self.attempts_since_meta: int = 10**9  # "long ago"
        self.num_attempts: int = 0
        self.num_progress: int = 0
        self.pending_meta_ideas: List[str] = []

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def get(self, node_id: str) -> Optional[LineageNode]:
        return self.nodes.get(node_id)

    def add_root(
        self,
        node_id: str,
        score: float,
        mu0: float = 0.5,
        iteration: int = 0,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        """Add a program with no local parent (initial, imported or spawned)."""
        mu0 = min(max(mu0, _BETA_FLOOR), 1.0 - _BETA_FLOOR)
        node = LineageNode(
            id=node_id,
            score=score,
            island=self.island,
            parent_node=None,
            depth=0,
            iteration=iteration,
            alpha=self.kappa * mu0,
            beta=self.kappa * (1.0 - mu0),
            mu0=mu0,
            provenance=provenance or {"type": "initial"},
        )
        self.nodes[node_id] = node
        self.roots.append(node_id)
        if score > self.best_score:
            self.best_score = score
        return node

    def add_child(
        self,
        parent: LineageNode,
        node_id: str,
        score: float,
        record: AttemptRecord,
        iteration: int,
        inherit_prior: bool = True,
    ) -> LineageNode:
        """Create the child of an admitted edit.

        MUST be called after ``backup`` for the same record, so that the
        child's prior includes the outcome of its own creating edit through
        its parent (Section 4.4 / ordering constraint of Section 11).
        """
        mu0 = parent.p_mean if inherit_prior else 0.5
        mu0 = min(max(mu0, _BETA_FLOOR), 1.0 - _BETA_FLOOR)
        node = LineageNode(
            id=node_id,
            score=score,
            island=self.island,
            parent_node=parent.id,
            depth=parent.depth + 1,
            iteration=iteration,
            incoming_edit=record.id,
            alpha=self.kappa * mu0,
            beta=self.kappa * (1.0 - mu0),
            mu0=mu0,
            provenance={"type": "edit"},
        )
        self.nodes[node_id] = node
        parent.children.append(node_id)
        return node

    def path_from_root(self, node: LineageNode) -> List[LineageNode]:
        """Nodes on the path root -> ... -> node (inclusive)."""
        path = [node]
        while path[-1].parent_node is not None:
            path.append(self.nodes[path[-1].parent_node])
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Individual Reward (Section 3)
    # ------------------------------------------------------------------

    def individual_reward(self, node: LineageNode) -> float:
        if self.theta is None or self.theta <= 0.0:
            return 0.0
        return (node.score - self.best_score) / self.theta

    def is_progress(self, parent_score: float, child_score: float) -> bool:
        return child_score > parent_score + numerical_tolerance(self.epsilon, parent_score)

    # ------------------------------------------------------------------
    # Lineage Reward: backup (Section 4.5)
    # ------------------------------------------------------------------

    def backup(
        self,
        node: LineageNode,
        admitted: bool,
        child_score: Optional[float],
        q: float,
        to_ancestors: bool = True,
    ) -> int:
        """Back an attempt made at ``node`` up into ``node`` and its ancestors.

        Returns the parent-relative outcome ``y_e(node)``.
        """
        q = min(max(q, 0.0), 1.0)
        m = float("-inf")
        y_parent = 0
        a: Optional[LineageNode] = node
        while a is not None:
            m = max(m, a.score)
            y = (
                1
                if admitted
                and child_score is not None
                and child_score > m + numerical_tolerance(self.epsilon, m)
                else 0
            )
            w = (self.gamma ** (node.depth - a.depth)) * q
            a.alpha += w * y
            a.beta += w * (1 - y)
            if a is node:
                y_parent = y
                if not to_ancestors:
                    break
            a = self.nodes.get(a.parent_node) if a.parent_node is not None else None
        return y_parent

    # ------------------------------------------------------------------
    # Island scalars (Section 11, step 11)
    # ------------------------------------------------------------------

    def update_scalars(self, record: AttemptRecord) -> bool:
        """Update theta, the stagnation counter and J*. Returns True on progress."""
        self.num_attempts += 1
        self.attempts_since_meta += 1
        progress = (
            record.status == STATUS_ADMITTED
            and record.delta is not None
            and record.parent_score is not None
            and record.delta > numerical_tolerance(self.epsilon, record.parent_score)
        )
        if progress:
            if self.theta is None:
                self.theta = record.delta
            else:
                self.theta = self.rho * self.theta + (1.0 - self.rho) * record.delta
            self.attempts_since_progress = 0
            self.num_progress += 1
        else:
            self.attempts_since_progress += 1
        if record.status == STATUS_ADMITTED and record.score is not None:
            if record.score > self.best_score:
                self.best_score = record.score
        return progress

    # ------------------------------------------------------------------
    # Selection (Section 6)
    # ------------------------------------------------------------------

    def select(
        self,
        rng: random.Random,
        tau: float = 1.0,
        use_lineage_reward: bool = True,
    ) -> Tuple[LineageNode, SelectionInfo]:
        """Thompson sampling on ``R(u) + log p~_u`` over all nodes of the island."""
        if not self.nodes:
            raise ValueError(f"Cannot select: lineage tree of island {self.island} is empty")

        best_index = float("-inf")
        best: List[Tuple[LineageNode, float, float]] = []
        for node in self.nodes.values():
            if use_lineage_reward:
                a = max(tau * node.alpha, _BETA_FLOOR)
                b = max(tau * node.beta, _BETA_FLOOR)
            else:
                a, b = 1.0, 1.0
            p = rng.betavariate(a, b)
            p = min(max(p, P_FLOOR), 1.0)
            r = self.individual_reward(node)
            index = r + math.log(p)
            if index > best_index:
                best_index = index
                best = [(node, p, r)]
            elif index == best_index:
                best.append((node, p, r))

        node, p, r = best[0] if len(best) == 1 else rng.choice(best)
        info = SelectionInfo(
            node_id=node.id,
            island=self.island,
            R=r,
            alpha=node.alpha,
            beta=node.beta,
            p_mean=node.p_mean,
            p_sample=p,
            index=best_index,
            theta=self.theta,
            best_score=self.best_score,
            tau=tau,
            num_nodes=len(self.nodes),
        )
        return node, info

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "island": self.island,
            "gamma": self.gamma,
            "kappa": self.kappa,
            "rho": self.rho,
            "epsilon": self.epsilon,
            "roots": list(self.roots),
            "best_score": self.best_score,
            "theta": self.theta,
            "attempts_since_progress": self.attempts_since_progress,
            "attempts_since_meta": self.attempts_since_meta,
            "num_attempts": self.num_attempts,
            "num_progress": self.num_progress,
            "pending_meta_ideas": list(self.pending_meta_ideas),
            "nodes": [n.to_dict() for n in self.nodes.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageTree":
        tree = cls(
            island=data["island"],
            gamma=data.get("gamma", 0.9),
            kappa=data.get("kappa", 2.0),
            rho=data.get("rho", 0.9),
            epsilon=data.get("epsilon", 0.0),
        )
        tree.roots = list(data.get("roots", []))
        best = data.get("best_score")
        tree.best_score = float("-inf") if best is None else best
        tree.theta = data.get("theta")
        tree.attempts_since_progress = data.get("attempts_since_progress", 0)
        tree.attempts_since_meta = data.get("attempts_since_meta", 10**9)
        tree.num_attempts = data.get("num_attempts", 0)
        tree.num_progress = data.get("num_progress", 0)
        tree.pending_meta_ideas = list(data.get("pending_meta_ideas", []))
        for nd in data.get("nodes", []):
            node = LineageNode.from_dict(nd)
            tree.nodes[node.id] = node
        return tree
