"""Per-island lineage tree: nodes, attempt records, rewards, backup, selection.

Implements the search state of both LineageEvolve variants:

* Thompson variant (LineageEvolve_method_spec.md, Sections 2-6): Thompson
  sampling on ``R(u) + log p~_u`` over all nodes (``LineageTree.select``).
* PUCT variant (LineageEvolve_method_spec_puct.md, Sections 2-7): MCTS
  descent from a virtual island root with an explicit "expand here" action
  (``LineageTree.puct_select``), using visit counts, local evidence and the
  max-backed-up subtree best.

The quantities only the PUCT variant needs (visits, local evidence, subtree
best) are maintained for both variants; the Thompson variant ignores them.  This
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
    selection_path: List[str] = field(default_factory=list)  # node ids, root first

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
    alpha: float = 1.0  # subtree evidence (Beta)
    beta: float = 1.0
    mu0: float = 0.5
    alpha_loc: float = 1.0  # local evidence: attempts made directly at this node
    beta_loc: float = 1.0
    best_sub: float = float("-inf")  # best score in the subtree (max-backup)
    visits: int = 0  # N(w): simulations whose selection path passed through w
    expand_visits: int = 0  # N(+w): expansion attempts made at w
    provenance: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    attempts: List[AttemptRecord] = field(default_factory=list)

    @property
    def p_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def p_local(self) -> float:
        return self.alpha_loc / (self.alpha_loc + self.beta_loc)

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
        if data.get("best_sub") is None:
            data["best_sub"] = data.get("score", float("-inf"))
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
        theta_mode: str = "window",
        theta_window: int = 5,
        stagnation_rel: float = 0.0,
        theta_min_rel: float = 0.0,
    ) -> None:
        self.island = island
        self.gamma = gamma
        self.kappa = kappa
        self.rho = rho
        self.epsilon = epsilon
        # Step scale: "ema" (spec: EMA of progress gains, factor rho) or
        # "window" (median of the last ``theta_window`` progress gains).  The
        # EMA is dominated for a long time by the first, typically huge, jump
        # from the initial program, which flattens the Individual Reward.
        self.theta_mode = theta_mode
        self.theta_window = max(1, int(theta_window))
        # Minimum relative improvement of the island best that resets the
        # stagnation counter (tiny gains on a plateau do not count).
        self.stagnation_rel = max(0.0, float(stagnation_rel))
        # Floor on the step scale relative to |J*_k|: on a plateau the recent
        # gains become tiny and, without a floor, any node slightly behind the
        # best gets an Individual Reward of -50 or worse and is never revisited.
        self.theta_min_rel = max(0.0, float(theta_min_rel))

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
        self.root_visits: int = 0  # N(r_k) of the virtual island root (PUCT)
        self.progress_gains: List[float] = []  # recent progress gains (theta window)
        self.attempts_since_best: int = 0  # attempts since J*_k last improved

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
            alpha_loc=self.kappa * mu0,
            beta_loc=self.kappa * (1.0 - mu0),
            best_sub=score,
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
            alpha_loc=self.kappa * mu0,
            beta_loc=self.kappa * (1.0 - mu0),
            best_sub=score,
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
        return self.reward_of(node.score)

    def effective_theta(self) -> Optional[float]:
        if self.theta is None or self.theta <= 0.0:
            return None
        floor = self.theta_min_rel * abs(self.best_score) if math.isfinite(self.best_score) else 0.0
        return max(self.theta, floor)

    def reward_of(self, score: float) -> float:
        """``(J - J*_k) / theta_k``, or 0 while theta_k is undefined."""
        theta = self.effective_theta()
        if theta is None:
            return 0.0
        return (score - self.best_score) / theta

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
                # Local evidence: attempts made directly at this node
                # (PUCT spec, Section 7 step 3).
                a.alpha_loc += q * y
                a.beta_loc += q * (1 - y)
                if not to_ancestors:
                    break
            a = self.nodes.get(a.parent_node) if a.parent_node is not None else None
        return y_parent

    def backup_max(self, node: LineageNode, child_score: float) -> None:
        """Max-backup of the subtree best into ``node`` and its ancestors."""
        a: Optional[LineageNode] = node
        while a is not None:
            if child_score > a.best_sub:
                a.best_sub = child_score
            a = self.nodes.get(a.parent_node) if a.parent_node is not None else None

    def record_visits(self, path: List[str]) -> None:
        """One visit along a selection path; the last node was expanded."""
        if not path:
            return
        self.root_visits += 1
        for node_id in path:
            node = self.nodes.get(node_id)
            if node is not None:
                node.visits += 1
        last = self.nodes.get(path[-1])
        if last is not None:
            last.expand_visits += 1

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
            self.progress_gains = (self.progress_gains + [record.delta])[-self.theta_window :]
            if self.theta_mode == "window":
                gains = sorted(self.progress_gains)
                mid = len(gains) // 2
                self.theta = gains[mid] if len(gains) % 2 else 0.5 * (gains[mid - 1] + gains[mid])
            elif self.theta is None:
                self.theta = record.delta
            else:
                self.theta = self.rho * self.theta + (1.0 - self.rho) * record.delta
            self.attempts_since_progress = 0
            self.num_progress += 1
        else:
            self.attempts_since_progress += 1
        margin = numerical_tolerance(self.epsilon, self.best_score)
        if math.isfinite(self.best_score):
            margin = max(margin, self.stagnation_rel * abs(self.best_score))
        improved_best = (
            record.status == STATUS_ADMITTED
            and record.score is not None
            and record.score > self.best_score + margin
        )
        if record.status == STATUS_ADMITTED and record.score is not None:
            if record.score > self.best_score:
                self.best_score = record.score
        self.attempts_since_best = 0 if improved_best else self.attempts_since_best + 1
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
    # PUCT tree policy (PUCT spec, Section 6)
    # ------------------------------------------------------------------

    def _action_values(
        self, node: LineageNode, expand: bool, sw: Dict[str, bool]
    ) -> Tuple[float, float]:
        """``(Q, pi0)`` of descending into ``node`` or (``expand``) expanding it."""
        use_lin = sw.get("lineage_reward", True)
        use_ind = sw.get("individual_reward", True)
        if expand:
            if sw.get("local_expand_value", True):
                p_bar = node.p_local if use_lin else 0.5
            else:
                p_bar = node.p_mean if use_lin else 0.5
            r_q = self.reward_of(node.score) if use_ind else 0.0
        else:
            p_bar = node.p_mean if use_lin else 0.5
            target = node.best_sub if sw.get("max_backup", True) else node.score
            r_q = self.reward_of(target) if use_ind else 0.0
        r_p = self.reward_of(node.score) if use_ind else 0.0
        mu0 = node.mu0 if use_lin else 0.5
        return p_bar * math.exp(r_q), mu0 * math.exp(r_p)

    def puct_select(
        self,
        rng: random.Random,
        c_t: float,
        eta: float = 0.25,
        switches: Optional[Dict[str, bool]] = None,
    ) -> Tuple[LineageNode, Dict[str, Any]]:
        """Descend from the virtual root until an expand action is chosen."""
        if not self.roots:
            raise ValueError(f"Cannot select: lineage tree of island {self.island} is empty")
        sw = switches or {}
        use_prior = sw.get("puct_prior", True)
        use_uct = sw.get("uct", False)

        path: List[str] = []
        steps: List[Dict[str, Any]] = []
        current: Optional[LineageNode] = None  # None = virtual root
        while True:
            # (label, target node, is_expand, N(a))
            actions: List[Tuple[str, LineageNode, bool, int]] = []
            if current is None:
                parent_visits = self.root_visits
                for rid in self.roots:
                    n = self.nodes[rid]
                    actions.append((rid, n, False, n.visits))
            else:
                parent_visits = current.visits
                for cid in current.children:
                    n = self.nodes[cid]
                    actions.append((cid, n, False, n.visits))
                actions.append(("expand", current, True, current.expand_visits))

            qs, pis = [], []
            for _, target, is_expand, _ in actions:
                q, pi0 = self._action_values(target, is_expand, sw)
                qs.append(q)
                pis.append(pi0)
            k = len(actions)
            total = sum(pis)
            if use_prior and total > 0.0:
                priors = [(1.0 - eta) * pi / total + eta / k for pi in pis]
            else:
                priors = [1.0 / k] * k

            scores = []
            for (_, _, _, n_a), q, prior in zip(actions, qs, priors):
                if use_uct:
                    bonus = c_t * math.sqrt(math.log(parent_visits + 1) / (1 + n_a))
                else:
                    bonus = c_t * prior * math.sqrt(max(parent_visits, 1)) / (1 + n_a)
                scores.append(q + bonus)

            best_score = max(scores)
            best_idx = [i for i, sc in enumerate(scores) if sc == best_score]
            chosen = best_idx[0] if len(best_idx) == 1 else rng.choice(best_idx)
            order = sorted(range(k), key=lambda i: scores[i], reverse=True)
            alt = next((i for i in order if i != chosen), None)

            def _desc(i: int) -> Dict[str, Any]:
                return {
                    "action": actions[i][0],
                    "Q": qs[i],
                    "P": priors[i],
                    "N": actions[i][3],
                    "score": scores[i],
                }

            steps.append(
                {
                    "at": "root" if current is None else current.id,
                    "N_parent": parent_visits,
                    "num_actions": k,
                    "chosen": _desc(chosen),
                    "best_alternative": _desc(alt) if alt is not None else None,
                }
            )
            label, target, is_expand, _ = actions[chosen]
            if is_expand:
                break
            current = target
            path.append(target.id)

        node = current
        info = {
            "policy": "puct",
            "node_id": node.id,
            "island": self.island,
            "path": list(path),
            "steps": steps,
            "R": self.reward_of(node.score),
            "R_sub": self.reward_of(node.best_sub),
            "alpha": node.alpha,
            "beta": node.beta,
            "alpha_loc": node.alpha_loc,
            "beta_loc": node.beta_loc,
            "mu0": node.mu0,
            "p_mean": node.p_mean,
            "p_local": node.p_local,
            "theta": self.theta,
            "best_score": self.best_score,
            "c_t": c_t,
            "num_nodes": len(self.nodes),
        }
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
            "root_visits": self.root_visits,
            "theta_mode": self.theta_mode,
            "theta_window": self.theta_window,
            "progress_gains": list(self.progress_gains),
            "attempts_since_best": self.attempts_since_best,
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
        tree.root_visits = data.get("root_visits", 0)
        tree.theta_mode = data.get("theta_mode", "ema")
        tree.theta_window = data.get("theta_window", 5)
        tree.progress_gains = list(data.get("progress_gains", []))
        tree.attempts_since_best = data.get("attempts_since_best", 0)
        for nd in data.get("nodes", []):
            node = LineageNode.from_dict(nd)
            tree.nodes[node.id] = node
        return tree
