"""LineageEvolve database: AdaEvolve's islands with a lineage tree per island.

What is inherited unchanged from ``AdaEvolveDatabase``:
  island scheduler (decayed UCB), migration, dynamic island spawning, the
  per-island archive used to sample context programs, best-program tracking
  and checkpointing of all of the above.

What is replaced / added (LineageEvolve specification):
  * within-island parent selection -> Thompson sampling on R(u) + L(u) over
    the island's lineage tree (Sections 3-6);
  * attempt records, backup, inherited priors, island scalars (Sections 2, 4);
  * Strategy Memory with attribution-weighted evidence (Sections 5, 8.3-8.4);
  * meta-guidance trigger on stagnation (Section 8.5);
  * run-wide hash index for pre-evaluation deduplication (Section 10);
  * imports become root nodes carrying a prior (Section 9).

All state updates belonging to one attempt are applied by ``commit_attempt``
in one synchronous call, i.e. atomically with respect to the asyncio loop.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.base_database import Program
from skydiscover.search.lineageevolve.attribution import code_hash, token_set
from skydiscover.search.lineageevolve.memory import ORIGIN_EDIT, ORIGIN_META, StrategyMemory
from skydiscover.search.lineageevolve.tree import (
    STATUS_ADMITTED,
    UNSPECIFIED,
    AttemptRecord,
    LineageNode,
    LineageTree,
    SelectionInfo,
)

logger = logging.getLogger(__name__)

STATE_FILE = "lineageevolve_state.json"


class LineageEvolveDatabase(AdaEvolveDatabase):
    """AdaEvolve islands + per-island lineage trees + shared Strategy Memory."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)

        # AdaEvolve's paradigm tracker (threshold on accumulated improvement
        # signal) is replaced by the stagnation-window trigger of Section 8.5.
        self.use_meta_guidance = bool(
            getattr(config, "use_meta_guidance", getattr(config, "use_paradigm_breakthrough", True))
        )
        self.use_paradigm_breakthrough = False
        self.paradigm_tracker = None

        # Hyperparameters (Section 12)
        self.gamma = float(getattr(config, "gamma", 0.9))
        self.kappa = float(getattr(config, "kappa", 2.0))
        self.theta_decay = float(getattr(config, "theta_decay", 0.9))
        self.epsilon = float(getattr(config, "epsilon", 0.0))
        self.tau = float(getattr(config, "tau", 1.0))
        self.tau_schedule = str(getattr(config, "tau_schedule", "constant"))
        self.k_plus = int(getattr(config, "k_plus", 2))
        self.k_minus = int(getattr(config, "k_minus", 1))
        self.path_recent = int(getattr(config, "path_recent", 5))
        self.stagnation_window = int(getattr(config, "stagnation_window", 10))
        cooldown = getattr(config, "meta_cooldown", None)
        self.meta_cooldown = int(cooldown) if cooldown is not None else self.stagnation_window
        self.meta_num_ideas = int(getattr(config, "meta_num_ideas", 3))
        self.strategy_list_cap = int(getattr(config, "strategy_list_cap", 30))

        # Component switches (Section 12)
        self.use_lineage_reward = bool(getattr(config, "use_lineage_reward", True))
        self.use_ancestor_backup = bool(getattr(config, "use_ancestor_backup", True))
        self.use_inherited_prior = bool(getattr(config, "use_inherited_prior", True))
        self.use_source_attribution = bool(getattr(config, "use_source_attribution", True))
        self.use_lineage_context = bool(getattr(config, "use_lineage_context", True))
        self.use_strategy_memory = bool(getattr(config, "use_strategy_memory", True))
        self.use_dedup = bool(getattr(config, "use_dedup", True))
        # In-iteration retries of failed candidates: "final" backs up only the
        # last outcome of an iteration, "all" backs up every failed try.
        self.retry_failure_evidence = str(getattr(config, "retry_failure_evidence", "final"))

        # Reproducible RNG, seeded from the run seed (Section 6).
        seed = os.environ.get("SKYDISCOVER_SEED")
        self.random_seed = int(seed) if seed not in (None, "") else getattr(config, "random_seed", 42)
        self.rng = random.Random(self.random_seed)
        if self.random_seed is not None:
            # AdaEvolve's own sampling (context programs, UCB tie-breaks,
            # spawn presets) uses the module-level generator.
            random.seed(self.random_seed)

        # Search state (Section 2)
        self.trees: List[LineageTree] = [self._new_tree(i) for i in range(self.num_islands)]
        self.node_island: Dict[str, int] = {}
        self.memory = StrategyMemory()
        self.hash_index: Set[str] = set()
        self._attempt_counter = 0
        self._token_cache: Dict[str, frozenset] = {}
        self.total_iterations: Optional[int] = None  # H, set by the controller
        self.last_selection: Optional[SelectionInfo] = None
        self._root_events: List[Dict[str, Any]] = []  # roots created since last drained

        logger.info(
            "LineageEvolveDatabase initialized: "
            f"gamma={self.gamma}, kappa={self.kappa}, rho={self.theta_decay}, "
            f"epsilon={self.epsilon}, tau={self.tau} ({self.tau_schedule}), "
            f"K+={self.k_plus}, K-={self.k_minus}, h={self.path_recent}, "
            f"W={self.stagnation_window}, seed={self.random_seed}; switches: "
            f"lineage_reward={self.use_lineage_reward}, backup={self.use_ancestor_backup}, "
            f"inherited_prior={self.use_inherited_prior}, "
            f"attribution={self.use_source_attribution}, "
            f"lineage_context={self.use_lineage_context}, memory={self.use_strategy_memory}, "
            f"dedup={self.use_dedup}, meta_guidance={self.use_meta_guidance}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_tree(self, island: int) -> LineageTree:
        return LineageTree(
            island=island,
            gamma=self.gamma,
            kappa=self.kappa,
            rho=self.theta_decay,
            epsilon=self.epsilon,
        )

    def _ensure_tree(self, island: int) -> LineageTree:
        while len(self.trees) <= island:
            self.trees.append(self._new_tree(len(self.trees)))
        return self.trees[island]

    def get_node(self, program_id: str) -> Optional[LineageNode]:
        island = self.node_island.get(program_id)
        if island is None:
            return None
        return self.trees[island].nodes.get(program_id)

    def drain_root_events(self) -> List[Dict[str, Any]]:
        """Root nodes created since the last call (for the per-iteration log)."""
        events, self._root_events = self._root_events, []
        return events

    def tokens_of(self, program_id: str) -> Optional[frozenset]:
        """Token set of a program's normalised source (cached)."""
        cached = self._token_cache.get(program_id)
        if cached is not None:
            return cached
        program = self.programs.get(program_id)
        if program is None:
            return None
        tokens = token_set(program.solution)
        self._token_cache[program_id] = tokens
        return tokens

    def _in_island(self, island: int, program_id: str) -> bool:
        if self.use_unified_archive and self.archives:
            return self.archives[island].get(program_id) is not None
        return any(p.id == program_id for p in self.islands[island])

    # ------------------------------------------------------------------
    # Adding programs
    # ------------------------------------------------------------------

    def add(
        self,
        program: Program,
        iteration: Optional[int] = None,
        parent_id: Optional[str] = None,
        target_island: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Add a program to an island.

        Programs added through this method without ``lineage_child=True`` have
        no local parent (the initial program, its seeded copies, migrants):
        they become root nodes of the destination tree (Sections 2.1, 9).
        Children of admitted edits are added by ``commit_attempt``.
        """
        lineage_child = bool(kwargs.pop("lineage_child", False))
        island_idx = target_island if target_island is not None else self.current_island
        self._ensure_tree(island_idx)

        result = super().add(
            program, iteration=iteration, parent_id=parent_id, target_island=target_island
        )
        accepted = self._in_island(island_idx, program.id)

        if lineage_child:
            # Every node of the tree stays expandable, even if the bounded
            # archive (used only for context sampling/migration) rejected it.
            self.programs[program.id] = program
            if not accepted:
                self.adapter.record_evaluation(island_idx, self._get_fitness(program))
                self._invalidate_global_pareto_cache()
                self._update_best_program(program)
                if self.config.db_path:
                    self._save_program(program)
        elif accepted:
            self._register_root(program, island_idx, iteration)
        return result

    def _register_root(
        self,
        program: Program,
        island_idx: int,
        iteration: Optional[int] = None,
        kind: str = "import",
    ) -> Optional[LineageNode]:
        tree = self._ensure_tree(island_idx)
        if program.id in tree.nodes:
            return None

        src_id = program.parent_id
        src_island = self.node_island.get(src_id) if src_id else None
        if src_island is not None and src_island != island_idx:
            src_node = self.trees[src_island].nodes[src_id]
            mu0 = src_node.p_mean if self.use_inherited_prior else 0.5
            provenance = {
                "type": kind,
                "source_island": src_island,
                "source_program": src_id,
                "source_p_mean": src_node.p_mean,
            }
        else:
            mu0 = 0.5
            provenance = {"type": "initial"}

        node = tree.add_root(
            program.id,
            self._get_fitness(program),
            mu0=mu0,
            iteration=iteration if iteration is not None else self._iteration_count,
            provenance=provenance,
        )
        self.node_island[program.id] = island_idx
        self.hash_index.add(code_hash(program.solution))
        self._root_events.append(
            {
                "node_id": program.id,
                "island": island_idx,
                "score": node.score,
                "mu0": node.mu0,
                "provenance": provenance,
            }
        )
        logger.debug(
            f"Lineage root {program.id[:8]} on island {island_idx} "
            f"({provenance['type']}, mu0={mu0:.3f})"
        )
        return node

    def _seed_new_island(self, island_idx: int) -> None:
        """Spawned island: seeds become roots; theta is inherited (Section 3.2)."""
        self._ensure_tree(island_idx)
        super()._seed_new_island(island_idx)

        tree = self.trees[island_idx]
        source_islands: Set[int] = set()
        for program in self.archives[island_idx].get_all():
            src_island = self.node_island.get(program.parent_id) if program.parent_id else None
            if src_island is not None:
                source_islands.add(src_island)
            self._register_root(program, island_idx, self._iteration_count, kind="spawn")
        thetas = [
            self.trees[i].theta for i in sorted(source_islands) if self.trees[i].theta is not None
        ]
        if thetas:
            tree.theta = sum(thetas) / len(thetas)

    # ------------------------------------------------------------------
    # Selection (Section 6)
    # ------------------------------------------------------------------

    def current_tau(self) -> float:
        if self.tau_schedule == "budget" and self.total_iterations:
            remaining = max(1, self.total_iterations - self._iteration_count)
            return self.total_iterations / remaining
        return self.tau

    def sample(
        self,
        num_context_programs: Optional[int] = 4,
        force_exploration: bool = False,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        """Select the parent by lineage-tree Thompson sampling; sample context
        programs exactly as AdaEvolve does (the parent itself excluded)."""
        island_idx = self.current_island
        tree = self._ensure_tree(island_idx)
        if not tree.nodes:
            for program in self.get_island_population(island_idx):
                self._register_root(program, island_idx, self._iteration_count)
        if not tree.nodes:
            raise ValueError(f"Cannot sample: island {island_idx} is empty")

        node, info = tree.select(
            self.rng, tau=self.current_tau(), use_lineage_reward=self.use_lineage_reward
        )
        parent = self.programs[node.id]

        num = num_context_programs or 4
        if self.use_unified_archive and self.archives:
            local_count = max(1, int(num * self.local_context_program_ratio))
            global_count = num - local_count
            context = self.archives[island_idx].sample_other_context_programs(parent, local_count)
            context = context + self._sample_global_top(parent.id, global_count)
        else:
            context = self._sample_global_top(parent.id, num)

        seen: Set[str] = {parent.id}
        unique: List[Program] = []
        for p in context:
            if p.id not in seen:
                seen.add(p.id)
                unique.append(p)

        self.last_selection = info
        self._last_sampling_mode = "lineage"
        logger.debug(
            f"Selected node {node.id[:8]} on island {island_idx}: R={info.R:.3f}, "
            f"p~={info.p_sample:.3f}, alpha={info.alpha:.2f}, beta={info.beta:.2f}"
        )
        return {"": parent}, {"": unique}

    # ------------------------------------------------------------------
    # Strategy Memory access
    # ------------------------------------------------------------------

    def peek_strategy(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the declared strategy without modifying the memory."""
        raw = (intent or {}).get("strategy") or UNSPECIFIED
        out = {"id": None, "name": UNSPECIFIED, "is_new": False, "label": UNSPECIFIED}
        if not self.use_strategy_memory or raw == UNSPECIFIED:
            return out
        if raw.startswith("new:") or raw.startswith("name:"):
            name = raw.split(":", 1)[1]
            existing = self.memory.find_by_name(name)
            if existing is not None:
                out.update(id=existing.id, name=existing.name, label=f"{existing.id} ({name})")
            elif name:
                out.update(name=name, is_new=True, label=f"new: {name}")
            return out
        strategy = self.memory.get(raw)
        if strategy is not None:
            out.update(
                id=strategy.id, name=strategy.name, label=f"{strategy.id} ({strategy.name})"
            )
        return out

    def retrieve_strategies(
        self, node: LineageNode, forced: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Lessons to try and warnings for the selected parent (Section 8.4)."""
        if not self.use_strategy_memory or len(self.memory) == 0:
            return [], []
        tree = self.trees[node.island]

        excluded: Set[str] = set()
        path = tree.path_from_root(node)
        for parent, child in zip(path[:-1], path[1:]):
            for record in parent.attempts:
                if record.id == child.incoming_edit:
                    if record.y_parent and record.strategy_id:
                        excluded.add(record.strategy_id)  # already embodied in u
                    break
        for record in node.attempts:
            if record.strategy_id:
                excluded.add(record.strategy_id)  # already shown in the attempt log
        if forced:
            excluded.discard(forced)

        parent_tokens = self.tokens_of(node.id) or frozenset()
        lessons, warnings = self.memory.retrieve(
            parent_tokens,
            self.tokens_of,
            excluded,
            self.rng,
            k_plus=self.k_plus,
            k_minus=self.k_minus,
            forced=forced,
        )
        shown = {item["strategy"].id for item in lessons + warnings}
        for item in lessons + warnings:
            item["strategy"].times_shown += 1
        for t in self.trees:
            if t.pending_meta_ideas:
                t.pending_meta_ideas = [sid for sid in t.pending_meta_ideas if sid not in shown]
        return lessons, warnings

    # ------------------------------------------------------------------
    # Meta-guidance (Section 8.5)
    # ------------------------------------------------------------------

    def should_trigger_meta(self, island: int) -> bool:
        if not self.use_meta_guidance or not self.use_strategy_memory:
            return False
        tree = self.trees[island]
        return (
            tree.attempts_since_progress >= self.stagnation_window
            and not tree.pending_meta_ideas
            and tree.attempts_since_meta >= self.meta_cooldown
        )

    def register_meta_ideas(
        self, island: int, ideas: List[Dict[str, Any]], iteration: int
    ) -> List[str]:
        """Insert ideas as memory items with no evidence, i.e. utility Beta(1,1)."""
        tree = self.trees[island]
        tree.attempts_since_meta = 0
        new_ids: List[str] = []
        for idea in ideas:
            if self.memory.find_by_name(idea["name"]) is not None:
                continue
            strategy = self.memory.create(
                idea["name"],
                summary=idea.get("summary", ""),
                mechanism=idea.get("mechanism", ""),
                conditions=idea.get("conditions") or [],
                origin=ORIGIN_META,
                iteration=iteration,
                island=island,
            )
            new_ids.append(strategy.id)
        tree.pending_meta_ideas = list(new_ids)
        return new_ids

    # ------------------------------------------------------------------
    # Commit of one attempt (Section 11, steps 9-13)
    # ------------------------------------------------------------------

    def commit_attempt(
        self,
        *,
        parent_id: str,
        island: int,
        iteration: int,
        status: str,
        intent: Dict[str, Any],
        prompt_sources: Dict[str, Any],
        child_program: Optional[Program] = None,
        error: Optional[str] = None,
        candidate_hash: Optional[str] = None,
        interpretation: Optional[Dict[str, Any]] = None,
        evidence: bool = True,
        count_visit: bool = True,
    ) -> AttemptRecord:
        """Apply all updates of one attempt record atomically.

        Order: backup (9) -> child prior (10) -> island scalars (11) ->
        memory (12) -> island scheduler (13).

        ``evidence=False`` records a failed try that is being retried inside
        the same iteration: it is stored on the parent (so the attempt log
        shows it) and its hash is indexed, but it is not backed up, does not
        touch the island scalars and adds no strategy evidence.
        ``count_visit=False`` keeps it invisible to the island scheduler, so
        one iteration is one visit.
        """
        tree = self.trees[island]
        parent = tree.nodes[parent_id]
        admitted = status == STATUS_ADMITTED
        if admitted and child_program is None:
            raise ValueError("An admitted attempt requires a child program")

        interpretation = interpretation or {}
        q = 1.0
        ext_shares: Dict[str, float] = {}
        if admitted and self.use_source_attribution and interpretation.get("valid"):
            q = float(interpretation.get("q", 1.0))
            ext_shares = dict(interpretation.get("ext_shares") or {})

        self._attempt_counter += 1
        child_score = self._get_fitness(child_program) if admitted else None
        record = AttemptRecord(
            id=f"A{self._attempt_counter}",
            parent=parent_id,
            island=island,
            iteration=iteration,
            status=status,
            child=child_program.id if admitted else None,
            intent=dict(intent or {}),
            delta=(child_score - parent.score) if admitted else None,
            q=q,
            ext_shares=ext_shares,
            prompt_sources=dict(prompt_sources or {}),
            score=child_score,
            parent_score=parent.score,
            error=error,
            evidence=bool(evidence) or admitted,
        )

        # Step 9: backup into the expanded node and its ancestors.
        if record.evidence:
            record.y_parent = tree.backup(
                parent, admitted, child_score, q, to_ancestors=self.use_ancestor_backup
            )

        # Step 10: child node with the inherited prior (after backup).
        if admitted:
            tree.add_child(
                parent,
                child_program.id,
                child_score,
                record,
                iteration,
                inherit_prior=self.use_inherited_prior,
            )
            self.node_island[child_program.id] = island
            self.hash_index.add(candidate_hash or code_hash(child_program.solution))
        elif candidate_hash and status != "duplicate":
            self.hash_index.add(candidate_hash)

        # Step 11: island scalars.
        if record.evidence:
            tree.update_scalars(record)

        # Step 12: Strategy Memory.
        self._update_memory(record, parent_id, interpretation, admitted, iteration, island)

        parent.attempts.append(record)

        # Step 13: island scheduler.  Admitted edits are a regular evaluation;
        # crashed and duplicate attempts are a visit with zero reward (Section 9).
        if admitted:
            self.add(
                child_program,
                iteration=iteration,
                parent_id=parent_id,
                target_island=island,
                lineage_child=True,
            )
        else:
            self.last_iteration = max(self.last_iteration, iteration)
            if count_visit:
                self.adapter.record_evaluation(island, float("-inf"))
        return record

    def _update_memory(
        self,
        record: AttemptRecord,
        parent_id: str,
        interpretation: Dict[str, Any],
        admitted: bool,
        iteration: int,
        island: int,
    ) -> None:
        if not self.use_strategy_memory:
            return
        declared = self.peek_strategy(record.intent)
        sid: Optional[str] = declared["id"]
        description = (record.intent or {}).get("description") or ""

        if not record.evidence:  # retried failure: name it for the attempt log only
            record.strategy_id = sid
            record.strategy_name = declared["name"]
            return

        if declared["is_new"]:
            lesson = interpretation.get("lesson") if admitted else None
            lesson = lesson if isinstance(lesson, dict) else {}
            conditions = lesson.get("conditions") or []
            if isinstance(conditions, str):
                conditions = [conditions]
            strategy = self.memory.create(
                declared["name"],
                summary=str(lesson.get("summary") or description)[:600],
                mechanism=str(lesson.get("mechanism") or "")[:600],
                conditions=[str(c)[:300] for c in conditions][:3],
                origin=ORIGIN_EDIT,
                iteration=iteration,
                island=island,
            )
            sid = strategy.id
        elif sid is not None and admitted:
            new_condition = interpretation.get("new_condition")
            if isinstance(new_condition, str) and new_condition.strip():
                self.memory.add_condition(sid, new_condition)  # at most one per attempt

        if sid is not None:
            record.strategy_id = sid
            record.strategy_name = self.memory.strategies[sid].name

        shares: Dict[str, float] = {}
        if sid is not None:
            shares[sid] = record.q
        for src, share in record.ext_shares.items():
            if src in self.memory.strategies:  # context-program shares credit no strategy
                shares[src] = shares.get(src, 0.0) + share
        for strategy_id, share in shares.items():
            self.memory.add_evidence(strategy_id, record.id, parent_id, share, record.y_parent)
        record.shares = shares

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def lineage_stats(self) -> Dict[str, Any]:
        return {
            "islands": [
                {
                    "island": t.island,
                    "nodes": len(t.nodes),
                    "roots": len(t.roots),
                    "max_depth": max((n.depth for n in t.nodes.values()), default=0),
                    "theta": t.theta,
                    "best_score": None if t.best_score == float("-inf") else t.best_score,
                    "attempts": t.num_attempts,
                    "progress_events": t.num_progress,
                    "attempts_since_progress": t.attempts_since_progress,
                    "pending_meta_ideas": list(t.pending_meta_ideas),
                }
                for t in self.trees
            ],
            "strategies": len(self.memory),
            "hash_index_size": len(self.hash_index),
        }

    # ------------------------------------------------------------------
    # Checkpointing (Section 2.4: all state MUST be saved and restored)
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        node_programs = {
            pid: self.programs[pid] for pid in self.node_island if pid in self.programs
        }
        super().save(path, iteration)  # rebuilds self.programs from the archives
        save_path = path or self.config.db_path
        if not save_path:
            return

        for pid, program in node_programs.items():
            if pid not in self.programs:
                self.programs[pid] = program
                self._save_program(program, save_path)

        rng_state = self.rng.getstate()
        state = {
            "version": 1,
            "random_seed": self.random_seed,
            "rng_state": [rng_state[0], list(rng_state[1]), rng_state[2]],
            "attempt_counter": self._attempt_counter,
            "trees": [t.to_dict() for t in self.trees],
            "memory": self.memory.to_dict(),
            "hash_index": sorted(self.hash_index),
            "hyperparameters": {
                "gamma": self.gamma,
                "kappa": self.kappa,
                "theta_decay": self.theta_decay,
                "epsilon": self.epsilon,
                "tau": self.tau,
                "tau_schedule": self.tau_schedule,
                "k_plus": self.k_plus,
                "k_minus": self.k_minus,
                "path_recent": self.path_recent,
                "stagnation_window": self.stagnation_window,
            },
        }
        with open(os.path.join(save_path, STATE_FILE), "w") as f:
            json.dump(state, f)
        logger.info(f"Saved LineageEvolve state to {save_path}")

    def load(self, path: str) -> None:
        super().load(path)
        self.use_paradigm_breakthrough = False
        self.paradigm_tracker = None

        state_path = os.path.join(path, STATE_FILE)
        if not os.path.exists(state_path):
            logger.warning(f"No LineageEvolve state at {path}; rebuilding trees from islands")
            self.trees = [self._new_tree(i) for i in range(self.num_islands)]
            self.node_island = {}
            for i in range(self.num_islands):
                for program in self.get_island_population(i):
                    self._register_root(program, i, self.last_iteration)
            return

        with open(state_path, "r") as f:
            state = json.load(f)

        self.trees = [LineageTree.from_dict(td) for td in state.get("trees", [])]
        for tree in self.trees:  # current config's hyperparameters take precedence
            tree.gamma, tree.kappa = self.gamma, self.kappa
            tree.rho, tree.epsilon = self.theta_decay, self.epsilon
        while len(self.trees) < self.num_islands:
            self.trees.append(self._new_tree(len(self.trees)))
        self.node_island = {nid: t.island for t in self.trees for nid in t.nodes}
        self.memory = StrategyMemory.from_dict(state.get("memory", {}))
        self.hash_index = set(state.get("hash_index", []))
        self._attempt_counter = state.get("attempt_counter", 0)
        self._token_cache = {}
        rng_state = state.get("rng_state")
        if rng_state:
            self.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        missing = [nid for nid in self.node_island if nid not in self.programs]
        if missing:
            logger.warning(f"{len(missing)} lineage nodes have no stored program; dropping them")
            for nid in missing:
                tree = self.trees[self.node_island.pop(nid)]
                node = tree.nodes.pop(nid)
                if nid in tree.roots:
                    tree.roots.remove(nid)
                for child_id in node.children:
                    child = tree.nodes.get(child_id)
                    if child is not None:
                        child.parent_node = None
                        tree.roots.append(child_id)
        logger.info(
            f"Loaded LineageEvolve state: {sum(len(t) for t in self.trees)} nodes, "
            f"{len(self.memory)} strategies, {len(self.hash_index)} hashes"
        )
