"""LineageEvolve database: AdaEvolve's islands with a lineage tree per island.

One class serves both variants; ``selection_policy`` picks the tree policy:
``"thompson"`` (LineageEvolve_method_spec.md, Section 6) or ``"puct"``
(LineageEvolve_method_spec_puct.md, Section 6).

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
from skydiscover.search.lineageevolve.attribution import code_hash, snake_case, token_set
from skydiscover.search.lineageevolve.memory import ORIGIN_EDIT, ORIGIN_META, StrategyMemory
from skydiscover.search.lineageevolve.tree import (
    numerical_tolerance,
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

        # Meta-guidance mode:
        # - "paradigm" (default): AdaEvolve's paradigm breakthrough, unchanged
        #   (global stagnation tracker, ideas generated from the global best,
        #   each idea used max_uses times on the global best with a mandatory
        #   "implement this" block).  Ideas are also entered into the Strategy
        #   Memory so the intent header can cite them and attribution credits them.
        # - "lineage": the spec's per-island trigger; ideas compete as lessons.
        # - "off": no meta-guidance.
        self.meta_guidance_mode = str(getattr(config, "meta_guidance_mode", "paradigm")).lower()
        if self.meta_guidance_mode not in ("paradigm", "lineage", "off"):
            raise ValueError(f"Unknown meta_guidance_mode: {self.meta_guidance_mode}")
        self.use_meta_guidance = self.meta_guidance_mode == "lineage" and bool(
            getattr(config, "use_meta_guidance", True)
        )
        if self.meta_guidance_mode != "paradigm":
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
        self.stagnation_window = int(getattr(config, "stagnation_window", 8))
        # "island_best": attempts since the island best last improved (default);
        # "progress": attempts since the last progress event (spec).  With a
        # deterministic evaluator tiny gains count as progress, so the spec
        # signal almost never fires.
        self.stagnation_signal = str(getattr(config, "stagnation_signal", "island_best"))
        self.theta_mode = str(getattr(config, "theta_mode", "window"))
        self.theta_window = int(getattr(config, "theta_window", 5))
        self.stagnation_rel = float(getattr(config, "stagnation_min_rel_improvement", 1e-3))
        self.theta_min_rel = float(getattr(config, "theta_min_rel", 2e-3))
        # A child with exactly its parent's score adds no new expandable node
        # (it is still recorded as a no-progress attempt and kept as a program).
        self.keep_score_clones = bool(getattr(config, "keep_score_clones", False))
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

        # Tree policy: "thompson" or "puct" (+ PUCT hyperparameters/switches).
        self.selection_policy = str(getattr(config, "selection_policy", "thompson")).lower()
        if self.selection_policy not in ("thompson", "puct"):
            raise ValueError(f"Unknown selection_policy: {self.selection_policy}")
        self.c0 = float(getattr(config, "c0", 1.0))
        # Lower bound of the decayed exploration strength, as a fraction of c0
        # (0 = spec: c_t decays linearly to 0 over the budget).
        self.c_min_frac = float(getattr(config, "c_min_frac", 0.0))
        # Where an active paradigm is applied: "global_best" (AdaEvolve) or
        # "tree" (the parent chosen by the tree policy).
        self.paradigm_parent = str(getattr(config, "paradigm_parent", "global_best")).lower()
        # Migration imports become expandable lineage roots (spec) or are
        # kept as context-only programs (False).
        self.import_as_roots = bool(getattr(config, "import_as_roots", True))
        # Edit Interpreter input: "full" (context-program code in the prompt)
        # or "matched" (only the deterministic per-unit copy fractions).
        self.interpreter_context = str(getattr(config, "interpreter_context", "full")).lower()
        # Probability that a try uses the active paradigm; the other tries are
        # ordinary tree-policy tries (1.0 = AdaEvolve: every try uses it).
        self.paradigm_prob = min(max(float(getattr(config, "paradigm_prob", 1.0)), 0.0), 1.0)
        if self.paradigm_parent not in ("global_best", "tree"):
            raise ValueError(f"Unknown paradigm_parent: {self.paradigm_parent}")
        self.eta = float(getattr(config, "eta", 0.25))
        self.puct_switches = {
            "lineage_reward": self.use_lineage_reward,
            "individual_reward": bool(getattr(config, "use_individual_reward", True)),
            "puct_prior": bool(getattr(config, "use_puct_prior", True)),
            "uct": bool(getattr(config, "use_uct", False)),
            "max_backup": bool(getattr(config, "use_max_backup", True)),
            "local_expand_value": bool(getattr(config, "use_local_expand_value", True)),
        }
        retrieval = getattr(config, "retrieval_mode", None)
        self.retrieval_mode = (
            str(retrieval).lower()
            if retrieval
            else ("ucb" if self.selection_policy == "puct" else "thompson")
        )

        # Reproducible RNG for the method's own random choices (tree policy,
        # retrieval), seeded from the run seed (Section 6).  The module-level
        # generator that AdaEvolve's inherited code uses (context sampling,
        # UCB tie-breaks, spawn presets) is left unseeded, exactly as in
        # AdaEvolve.
        seed = os.environ.get("SKYDISCOVER_SEED")
        self.random_seed = int(seed) if seed not in (None, "") else getattr(config, "random_seed", 42)
        self.rng = random.Random(self.random_seed)
        # Separate stream for the paradigm coin, so that paradigm_prob=1.0
        # leaves the tree policy's random stream unchanged.
        self.paradigm_rng = random.Random(
            None if self.random_seed is None else int(self.random_seed) + 7919
        )

        # Search state (Section 2)
        self.trees: List[LineageTree] = [self._new_tree(i) for i in range(self.num_islands)]
        self.node_island: Dict[str, int] = {}
        self.memory = StrategyMemory()
        self.hash_index: Set[str] = set()
        self._attempt_counter = 0
        self._token_cache: Dict[str, frozenset] = {}
        self.total_iterations: Optional[int] = None  # H, set by the controller
        self.last_selection: Optional[Dict[str, Any]] = None
        self.current_iteration: int = 0  # set by the controller each iteration
        self._root_events: List[Dict[str, Any]] = []  # roots created since last drained

        logger.info(
            "LineageEvolveDatabase initialized: "
            f"gamma={self.gamma}, kappa={self.kappa}, rho={self.theta_decay}, "
            f"epsilon={self.epsilon}, tau={self.tau} ({self.tau_schedule}), "
            f"K+={self.k_plus}, K-={self.k_minus}, h={self.path_recent}, "
            f"W={self.stagnation_window}, seed={self.random_seed}, "
            f"policy={self.selection_policy} (c0={self.c0}, eta={self.eta}, "
            f"retrieval={self.retrieval_mode}, puct_switches={self.puct_switches}); switches: "
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
            theta_mode=self.theta_mode,
            theta_window=self.theta_window,
            stagnation_rel=self.stagnation_rel,
            theta_min_rel=self.theta_min_rel,
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
            is_migrant = "migrated_from" in (program.metadata or {})
            if is_migrant and not self.import_as_roots:
                # Context-only import: the migrant stays in the island's
                # archive (so it can be shown as a context program) but does
                # not become an expandable lineage root.
                self.hash_index.add(code_hash(program.solution))
            else:
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
            tree.progress_gains = [tree.theta]

    # ------------------------------------------------------------------
    # Selection (Section 6)
    # ------------------------------------------------------------------

    def current_tau(self) -> float:
        if self.tau_schedule == "budget" and self.total_iterations:
            remaining = max(1, self.total_iterations - self._iteration_count)
            return self.total_iterations / remaining
        return self.tau

    def current_c(self) -> float:
        """PUCT exploration strength ``c_t = c0 (1 - t/H)`` (PUCT spec, 6.4)."""
        if not self.total_iterations:
            return self.c0
        frac = min(max(self.current_iteration / self.total_iterations, 0.0), 1.0)
        return self.c0 * max(1.0 - frac, self.c_min_frac)

    def sample(
        self,
        num_context_programs: Optional[int] = 4,
        force_exploration: bool = False,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        """Select the parent with the lineage tree policy (Thompson or PUCT);
        sample context programs exactly as AdaEvolve does (parent excluded)."""
        island_idx = self.current_island
        tree = self._ensure_tree(island_idx)
        if not tree.nodes:
            for program in self.get_island_population(island_idx):
                self._register_root(program, island_idx, self._iteration_count)
        if not tree.nodes:
            raise ValueError(f"Cannot sample: island {island_idx} is empty")

        if self.selection_policy == "puct":
            node, info = tree.puct_select(
                self.rng, c_t=self.current_c(), eta=self.eta, switches=self.puct_switches
            )
        else:
            node, sel = tree.select(
                self.rng, tau=self.current_tau(), use_lineage_reward=self.use_lineage_reward
            )
            info = {"policy": "thompson", **sel.to_dict()}
            info["path"] = [n.id for n in tree.path_from_root(node)]
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
            f"Selected node {node.id[:8]} on island {island_idx} ({info['policy']}): "
            f"R={info['R']:.3f}, alpha={info['alpha']:.2f}, beta={info['beta']:.2f}"
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
            mode=self.retrieval_mode,
            c0=self.c0,
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

    # ------------------------------------------------------------------
    # AdaEvolve paradigm, linked to the lineage tree and Strategy Memory
    # ------------------------------------------------------------------

    def set_paradigms(self, paradigms: List[Dict[str, Any]]) -> None:
        """AdaEvolve's set_paradigms, plus one Strategy Memory item per idea."""
        super().set_paradigms(paradigms)
        if not self.use_strategy_memory:
            return
        for paradigm in paradigms:
            name = snake_case(str(paradigm.get("idea") or ""))[:60] or "paradigm"
            cautions = paradigm.get("cautions")
            strategy = self.memory.create(
                name,
                summary=str(paradigm.get("idea") or "")[:600],
                mechanism=str(paradigm.get("description") or "")[:600],
                conditions=[str(cautions)[:300]] if cautions else [],
                origin=ORIGIN_META,
                iteration=self.current_iteration,
                island=None,
            )
            paradigm["strategy_id"] = strategy.id

    def use_paradigm_this_try(self) -> bool:
        """Coin flip of ``paradigm_prob`` (no random draw when it is 1 or 0)."""
        if self.paradigm_prob >= 1.0:
            return True
        if self.paradigm_prob <= 0.0:
            return False
        return self.paradigm_rng.random() < self.paradigm_prob

    def paradigm_parent_node(self) -> Optional[LineageNode]:
        """Node of the global best program (AdaEvolve applies a paradigm to it).

        Falls back to the best node of the current island if the global best
        is not a node (e.g. an archive-only program)."""
        best = self.get_best_program()
        if best is not None:
            node = self.get_node(best.id)
            if node is not None:
                return node
        tree = self.trees[self.current_island]
        if not tree.nodes:
            return None
        return max(tree.nodes.values(), key=lambda n: n.score)

    def select_for_paradigm(self, node: LineageNode) -> Dict[str, Any]:
        """Make ``node`` the selected parent: the attempt belongs to its island,
        and the selection path is its lineage path."""
        self.current_island = node.island
        tree = self.trees[node.island]
        self.last_selection = {
            "policy": "paradigm_best",
            "node_id": node.id,
            "island": node.island,
            "path": [n.id for n in tree.path_from_root(node)],
            "R": tree.reward_of(node.score),
            "alpha": node.alpha,
            "beta": node.beta,
            "theta": tree.theta,
            "best_score": tree.best_score,
        }
        return self.last_selection

    def should_trigger_meta(self, island: int) -> bool:
        if not self.use_meta_guidance or not self.use_strategy_memory:
            return False
        tree = self.trees[island]
        stalled = (
            tree.attempts_since_best
            if self.stagnation_signal == "island_best"
            else tree.attempts_since_progress
        )
        return (
            stalled >= self.stagnation_window
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
        selection_path: Optional[List[str]] = None,
    ) -> AttemptRecord:
        """Apply all updates of one attempt record atomically.

        Order: visits + backup (9) -> child prior (10) -> island scalars (11)
        -> memory (12) -> island scheduler (13, admitted edits only; failed
        iterations are reported with ``record_failed_visit``).
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
            selection_path=list(selection_path or []),
        )

        # Step 9: visits along the selection path (PUCT), backup of the
        # evidence into the expanded node and its ancestors, max-backup.
        tree.record_visits(record.selection_path or [parent_id])
        record.y_parent = tree.backup(
            parent, admitted, child_score, q, to_ancestors=self.use_ancestor_backup
        )
        if admitted:
            tree.backup_max(parent, child_score)

        # Step 10: child node with the inherited prior (after backup).
        is_clone = (
            admitted
            and not self.keep_score_clones
            and abs(child_score - parent.score) <= numerical_tolerance(self.epsilon, parent.score)
        )
        if is_clone:
            record.error = "same score as parent (no new node)"
            self.hash_index.add(candidate_hash or code_hash(child_program.solution))
        elif admitted:
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
        tree.update_scalars(record)

        # Step 12: Strategy Memory.
        self._update_memory(record, parent_id, interpretation, admitted, iteration, island)

        parent.attempts.append(record)

        # Step 13: island scheduler.  An admitted edit is a regular
        # evaluation, exactly as AdaEvolve adds a child.  Failed attempts are
        # reported once per iteration by the controller (record_failed_visit).
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
        return record

    def record_failed_visit(self, island: int) -> None:
        """An iteration that ended with a crashed/duplicate attempt is one visit
        of the island with zero reward (spec Section 9.2 / PUCT Section 10.2).

        Only the UCB statistics are touched.  AdaEvolve's per-island adaptive
        state (``total_evaluations``, which feeds the global productivity of
        the island-spawning rule) is left alone, because AdaEvolve itself does
        not count failed iterations there.
        """
        a = self.adapter
        if 0 <= island < len(a.dimension_visits):
            a.dimension_visits[island] += 1
            a.decayed_visits[island] = a.decay * a.decayed_visits[island] + 1.0
            a.dimension_rewards[island] = a.decay * a.dimension_rewards[island]

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
        if self.meta_guidance_mode != "paradigm":
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
            tree.theta_mode, tree.theta_window = self.theta_mode, self.theta_window
            tree.stagnation_rel = self.stagnation_rel
            tree.theta_min_rel = self.theta_min_rel
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
