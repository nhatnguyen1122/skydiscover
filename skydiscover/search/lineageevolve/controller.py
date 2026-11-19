"""
LineageEvolve Controller - AdaEvolve's controller with lineage-tree search.

This file is a copy of ``skydiscover/search/adaevolve/controller.py`` and
keeps its protocol unchanged: the same iteration loop, the same retry loop
(``_run_normal_step``: up to 1 + max_error_retries tries per iteration, each
try re-samples the parent and sees the previous error), the same response
parsing, the same evaluation-failure rule, the same prompt templates and the
same logging/checkpoint cadence.  ``diff`` against the AdaEvolve controller
shows only the method changes:

- parent selection by the lineage tree policy (Thompson or PUCT, chosen by
  ``search.database.selection_policy``) instead of explore/exploit sampling;
- AdaEvolve's paradigm breakthrough is kept unchanged (default
  ``meta_guidance_mode: paradigm``); the paradigm's parent (global best) is
  located in the lineage tree and each idea is also a Strategy Memory item
  that the intent header can cite.  ``meta_guidance_mode: lineage`` uses the
  spec's per-island meta-guidance instead;
- lineage context, lessons/warnings, strategy list and the intent header in
  the prompt instead of sibling context;
- pre-evaluation deduplication, attempt records for duplicate / crashed /
  admitted candidates, the Edit Interpreter, and the atomic commit of all
  search-state updates (``LineageEvolveDatabase.commit_attempt``).
"""

import json
import logging
import math
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from skydiscover.context_builder.adaevolve import AdaEvolveContextBuilder
from skydiscover.context_builder.lineageevolve import LineageEvolveContextBuilder
from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.llm.usage import USAGE, usage_purpose
from skydiscover.search.adaevolve.paradigm import ParadigmGenerator
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.lineageevolve import prompt_blocks
from skydiscover.search.lineageevolve.attribution import (
    build_inventory,
    context_overlap,
    code_hash,
    compute_shares,
    parse_intent,
    validate_labels,
)
from skydiscover.search.lineageevolve.interpreter import EditInterpreter, MetaGuide
from skydiscover.search.lineageevolve.tree import (
    STATUS_ADMITTED,
    STATUS_CRASHED,
    STATUS_DUPLICATE,
)
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


class LineageEvolveController(DiscoveryController):
    """
    AdaEvolve's controller with LineageEvolve's parent selection, prompt
    context, attempt records and search-state updates (see module docstring).
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        # Configuration
        db_config = self.config.search.database
        self.enable_retry = getattr(db_config, "enable_error_retry", True)
        self.max_retries = getattr(db_config, "max_error_retries", 2)
        self.num_context_programs = self.config.search.num_context_programs

        # Components
        self.llms = LLMPool(self.config.llm.models)
        self.context_builder = LineageEvolveContextBuilder(self.config)

        # Paradigm generator (if paradigm breakthrough is enabled)
        # Note: We check database.use_paradigm_breakthrough at runtime, not this init-time flag
        # This ensures correct behavior after checkpoint load
        if self.database.use_paradigm_breakthrough:
            model_names = ", ".join(m.name for m in self.guide_llms.models_cfg)
            logger.info(f"Paradigm LLM: using guide_models [{model_names}]")

            self.paradigm_generator = ParadigmGenerator(
                llm_pool=self.guide_llms,
                system_message=self.config.context_builder.system_message or "",
                evaluator_code=self._load_evaluator_code(),
                num_paradigms=self.database.get_paradigm_num_to_generate(),
                eval_timeout=self.config.evaluator.timeout,
                language=self.config.language or "python",
                objective_names=getattr(db_config, "pareto_objectives", []),
                higher_is_better=getattr(db_config, "higher_is_better", {}),
                fitness_key=getattr(db_config, "fitness_key", None),
            )
        else:
            self.paradigm_generator = None

        # LineageEvolve: Edit Interpreter and meta-guidance (guide models).
        self.interpreter = EditInterpreter(
            self.guide_llms,
            temperature=float(getattr(db_config, "interpreter_temperature", 0.0)),
            max_tokens=int(getattr(db_config, "interpreter_max_tokens", 4000)),
        )
        self.meta_guide: Optional[MetaGuide] = None
        if self.database.use_meta_guidance and self.database.use_strategy_memory:
            self.meta_guide = MetaGuide(
                self.guide_llms,
                task_description=self.context_builder._get_system_message() or "",
                evaluator_code=self._load_evaluator_code(),
                num_ideas=self.database.meta_num_ideas,
            )
        self._lineage_log_path: Optional[str] = None
        self._iter_entry: Dict[str, Any] = {}

        # JSON logging for comprehensive AdaEvolve stats
        self._iteration_stats_log_path: Optional[str] = None
        self._iteration_stats_file = None
        self._last_sampling_mode: Optional[str] = None
        self._last_sampling_intensity: Optional[float] = None

        logger.info(
            f"LineageEvolveController initialized "
            f"(language={self.config.language}, "
            f"policy={self.database.selection_policy}, "
            f"meta_guidance={self.meta_guide is not None})"
        )

    def _load_evaluator_code(self) -> str:
        """Load evaluator source code for paradigm generation context."""
        from skydiscover.search.utils.discovery_utils import load_evaluator_code

        # Ablation switch: hide the evaluator source from meta-guidance.
        if not getattr(self.config.search.database, "meta_guidance_include_evaluator", True):
            return ""
        return load_evaluator_code(self.evaluation_file)

    # =========================================================================
    # JSON Logging for AdaEvolve Stats
    # =========================================================================

    def _setup_iteration_stats_logging(self, output_dir: Optional[str] = None) -> None:
        """
        Set up JSON logging for comprehensive iteration statistics.

        Creates a JSONL file that records all AdaEvolve signals at each iteration.
        This enables detailed post-hoc analysis of the discovery process.

        Args:
            output_dir: Directory to write the log file. If None, uses database.config.db_path
        """
        # Determine output directory
        if output_dir is None:
            output_dir = self.output_dir
        if output_dir is None:
            output_dir = getattr(self.database.config, "db_path", None)
        if output_dir is None:
            output_dir = "."

        os.makedirs(output_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._iteration_stats_log_path = os.path.join(
            output_dir, f"adaevolve_iteration_stats_{timestamp}.jsonl"
        )

        logger.info(
            f"AdaEvolve iteration stats will be logged to: {self._iteration_stats_log_path}"
        )

    def _log_iteration_stats(
        self,
        iteration: int,
        sampling_mode: Optional[str] = None,
        sampling_intensity: Optional[float] = None,
        child_program: Optional[Dict] = None,
        iteration_time: Optional[float] = None,
        llm_generation_time: Optional[float] = None,
        eval_time: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log comprehensive iteration statistics to JSON file.

        This method collects all AdaEvolve signals and writes them as a single
        JSON line to the log file for easy post-processing.

        Args:
            iteration: Current iteration number
            sampling_mode: The mode used for sampling (exploration/exploitation/balanced)
            sampling_intensity: The search intensity value used
            child_program: The child program dict if successfully generated
            iteration_time: Time taken for this iteration
            error: Error message if iteration failed
        """
        if self._iteration_stats_log_path is None:
            return

        try:
            # Get comprehensive stats from database
            stats = self.database.get_comprehensive_iteration_stats(
                iteration=iteration,
                sampling_mode=(
                    sampling_mode if sampling_mode is not None else self._last_sampling_mode
                ),
                sampling_intensity=(
                    sampling_intensity
                    if sampling_intensity is not None
                    else self._last_sampling_intensity
                ),
            )

            # Add timestamp
            stats["timestamp"] = datetime.now().isoformat()

            # Add iteration-specific info
            stats["iteration_result"] = {
                "success": error is None,
                "error": error,
                "iteration_time_seconds": iteration_time,
                "llm_generation_time_seconds": llm_generation_time,
                "eval_time_seconds": eval_time,
            }

            # Token usage: this iteration (all LLM calls, incl. retries and
            # paradigm generation) and cumulative for the whole run.
            stats["iteration_result"]["token_usage"] = getattr(
                self, "_last_iteration_usage", None
            )
            stats["cumulative_token_usage"] = USAGE.snapshot()["total"]
            run_start = getattr(self, "_run_start_time", None)
            if run_start is not None:
                stats["elapsed_seconds"] = time.time() - run_start

            # Add child program info if available
            if child_program:
                stats["iteration_result"]["child_program"] = {
                    "id": child_program.get("id"),
                    "metrics": child_program.get("metrics"),
                    "generation": child_program.get("generation"),
                    "parent_id": child_program.get("parent_id"),
                }

            # Write to JSONL file
            with open(self._iteration_stats_log_path, "a") as f:
                f.write(json.dumps(stats, default=str) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log iteration stats: {e}")

    def get_iteration_stats_log_path(self) -> Optional[str]:
        """Get the path to the iteration stats log file."""
        return self._iteration_stats_log_path

    # =========================================================================
    # Main Evolution Loop
    # =========================================================================

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run evolution with adaptive search intensity and island rotation."""
        total = start_iteration + max_iterations
        logger.info(
            f"AdaEvolve: Running {max_iterations} iterations "
            f"across {self.database.num_islands} islands"
        )

        # Set up comprehensive JSON logging for iteration stats
        self._setup_iteration_stats_logging()
        self._run_start_time = time.time()

        # LineageEvolve: budget H for the PUCT exploration decay, and the
        # per-iteration lineage log (Section 13 / PUCT Section 14).
        self.database.total_iterations = total
        log_dir = os.path.dirname(self._iteration_stats_log_path or "") or "."
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._lineage_log_path = os.path.join(log_dir, f"lineageevolve_iterations_{stamp}.jsonl")

        # Ensure all islands are seeded
        self._ensure_all_islands_seeded()

        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested")
                break

            try:
                await self._run_iteration(iteration, checkpoint_callback)
            except Exception as e:
                logger.exception(f"Iteration {iteration} failed: {e}")
            finally:
                # CRITICAL: Tell database iteration is complete
                # This handles island rotation (UCB) and migration
                self.database.end_iteration(iteration)

        logger.info("LineageEvolve completed")
        logger.info(f"LineageEvolve final state: {json.dumps(self.database.lineage_stats())}")
        self.database.log_status()

        # Log final summary and stats file location
        if self._iteration_stats_log_path:
            logger.info(f"AdaEvolve iteration stats saved to: {self._iteration_stats_log_path}")

        return self.database.get_best_program()

    def _ensure_all_islands_seeded(self) -> None:
        """Ensure all islands have at least one program."""
        # Find a seed program
        seed_program = None
        for i in range(self.database.num_islands):
            size = self.database.get_island_size(i)
            if size > 0 and seed_program is None:
                population = self.database.get_island_population(i)
                if population:
                    seed_program = population[0]
                    break

        if seed_program is None:
            logger.warning("No seed program found")
            return

        # Seed empty islands
        for i in range(self.database.num_islands):
            if self.database.get_island_size(i) == 0:
                copy = Program(
                    id=str(uuid.uuid4()),
                    solution=seed_program.solution,
                    language=seed_program.language,
                    metrics=seed_program.metrics.copy() if seed_program.metrics else {},
                    iteration_found=seed_program.iteration_found,
                    parent_id=None,
                    generation=0,
                    metadata={"seeded_to_island": i},
                )
                self.database.add(copy, iteration=0, target_island=i)
                logger.info(f"Seeded island {i}")

    async def _run_iteration(self, iteration: int, checkpoint_callback) -> None:
        """Execute one evolution iteration."""
        iteration_start_time = time.time()
        usage_before = USAGE.snapshot()
        self.database.current_iteration = iteration
        island = self.database.current_island
        self._iter_entry = {"iteration": iteration, "island": island, "tries": []}

        # Check for global paradigm stagnation
        # Use database flag directly to stay in sync after checkpoint load
        if self.database.use_paradigm_breakthrough and self.database.is_paradigm_stagnating():
            with usage_purpose("meta_guidance"):
                await self._generate_paradigms_if_needed()

        with usage_purpose("mutation"):
            result = await self._run_normal_step(iteration)

        iteration_time = time.time() - iteration_start_time
        self._last_iteration_usage = USAGE.delta(usage_before)

        # LineageEvolve: an iteration that ends with a crashed/duplicate
        # attempt is one zero-reward visit of the island (Section 9.2).
        tries = self._iter_entry.get("tries", [])
        if result.error and any(t.get("status") in (STATUS_CRASHED, STATUS_DUPLICATE) for t in tries):
            self.database.record_failed_visit(island)
        self._write_lineage_log(result, iteration_time)

        if result.error:
            logger.warning(f"Iteration {iteration}: {result.error}")
            # Log failed iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=None,
                iteration_time=iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=result.error,
            )
        else:
            self._process_result(result, iteration, checkpoint_callback)
            # Log successful iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=result.child_program_dict,
                iteration_time=result.iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=None,
            )

    async def _generate_paradigms_if_needed(self) -> None:
        """Generate new paradigms if stagnating and none active."""
        if self.paradigm_generator is None:
            return

        if self.database.has_active_paradigm():
            return  # Already have paradigms to use

        logger.info("Global paradigm stagnation detected, generating breakthrough ideas...")

        # Get current best program for context
        best_program = self.database.get_best_program()
        best_solution = best_program.solution if best_program else ""
        best_score = self.database.get_program_proxy_score(best_program)

        # Extract evaluator feedback from the best program's artifacts
        evaluator_feedback = None
        if best_program and best_program.artifacts:
            fb = best_program.artifacts.get("feedback")
            if fb and isinstance(fb, str):
                evaluator_feedback = fb

        # Get previously tried ideas for feedback
        previously_tried = self.database.get_previously_tried_ideas()

        # Generate new paradigms
        paradigms = await self.paradigm_generator.generate(
            current_program_solution=best_solution,
            current_best_score=best_score,
            previously_tried_ideas=previously_tried,
            evaluator_feedback=evaluator_feedback,
        )

        if paradigms:
            self.database.set_paradigms(paradigms)
            logger.info(f"Generated {len(paradigms)} breakthrough paradigms")
        else:
            logger.warning("Failed to generate paradigms")

    async def _run_normal_step(self, iteration: int) -> SerializableResult:
        """Run a normal iteration with optional retry."""
        last_error = None
        attempts = 1 + (self.max_retries if self.enable_retry else 0)

        for attempt in range(attempts):
            result = await self._generate_child(iteration, error_context=last_error)
            if not result.error:
                return result
            last_error = result.error
            logger.debug(f"Attempt {attempt + 1}/{attempts} failed: {last_error}")

        return SerializableResult(
            error=f"All {attempts} attempts failed: {last_error}",
            iteration=iteration,
        )

    def _process_result(
        self,
        result: SerializableResult,
        iteration: int,
        checkpoint_callback,
    ) -> None:
        """Process a successful result by adding to database."""
        child = Program(**result.child_program_dict)

        # LineageEvolve: the child was already added to its island by
        # LineageEvolveDatabase.commit_attempt (with the backup, the child
        # prior and the memory update), inside _execute_generation.

        # Fire monitor callback (live dashboard)
        if self.monitor_callback:
            try:
                self.monitor_callback(child, iteration)
            except Exception:
                logger.debug("Monitor callback error", exc_info=True)

        # Log prompt
        if result.prompt:
            self.database.log_prompt(
                template_key=(
                    "full_rewrite_user_message"
                    if not self.config.diff_based_generation
                    else "diff_user_message"
                ),
                program_id=child.id,
                prompt=result.prompt,
                responses=[result.llm_response] if result.llm_response else [],
            )

        # Log progress
        logger.info(
            f"Iteration {iteration}: Program {child.id[:8]} "
            f"(parent: {result.parent_id[:8] if result.parent_id else 'None'}) "
            f"completed in {result.iteration_time:.2f}s"
            f" (llm: {result.llm_generation_time:.2f}s,"
            f" eval: {result.eval_time:.2f}s)"
        )

        # Log metrics
        if child.metrics:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in child.metrics.items()
            )
            logger.info(f"Metrics: {metrics_str}")

        # Check for new best
        if self.database.is_multiobjective_enabled():
            pareto_front_ids = {program.id for program in self.database.get_pareto_front()}
            if child.id in pareto_front_ids:
                logger.info(f"Program entered the global Pareto front at iteration {iteration}")
            if self.database.best_program_id == child.id:
                logger.info(f"New representative Pareto solution found at iteration {iteration}")
        elif self.database.best_program_id == child.id:
            logger.info(f"New best solution found at iteration {iteration}")

        # Checkpoint callback
        if iteration > 0 and iteration % self.config.checkpoint_interval == 0:
            logger.info(f"Checkpoint interval reached at iteration {iteration}")
            self.database.log_status()
            if checkpoint_callback:
                checkpoint_callback(iteration)

    # =========================================================================
    # Child Generation
    # =========================================================================

    async def _generate_child(
        self,
        iteration: int,
        error_context: Optional[str] = None,
        force_exploration: bool = False,
    ) -> SerializableResult:
        """Generate and evaluate a single child program."""
        try:
            if not self.database.programs:
                return await self._run_from_scratch_iteration(iteration)

            # Ensure all islands are seeded (needed after from-scratch bootstrap)
            self._ensure_all_islands_seeded()

            # Sample parent and context programs (database returns standard framework dicts)
            parent_dict, context_programs_dict = self.database.sample(
                self.num_context_programs,
                force_exploration=force_exploration,
            )

            # Unpack parent dict (standard framework pattern)
            if not parent_dict:
                logger.error("sample() returned empty parent dict")
                return SerializableResult(
                    error="Empty parent dict from sample()", iteration=iteration
                )
            parent_label = list(parent_dict.keys())[0]
            parent = list(parent_dict.values())[0]

            # When paradigm is active, use best program as parent
            # This ensures paradigm (designed from best) is applied to best, not random parent
            paradigm = (
                self.database.get_current_paradigm()
                if self.database.use_paradigm_breakthrough
                else None
            )
            # LineageEvolve: with paradigm_prob < 1 only a fraction of the tries
            # use the active paradigm; the others are ordinary tries.
            self._paradigm_skipped = False
            if paradigm and not self.database.use_paradigm_this_try():
                paradigm = None
                self._paradigm_skipped = True
            if paradigm and self.database.paradigm_parent == "global_best":
                best_node = self.database.paradigm_parent_node()
                if best_node is not None:
                    best_program = self.database.programs[best_node.id]
                    parent_dict = {parent_label: best_program}
                    parent = best_program
                    # LineageEvolve: the attempt is made at the best node.
                    self.database.select_for_paradigm(best_node)
                    # Keep context_programs_dict from sampling for diversity

            # LineageEvolve: the parent was chosen by the lineage tree policy.
            db = self.database
            self._last_sampling_mode = getattr(db, "_last_sampling_mode", None) or "lineage"
            self._last_sampling_intensity = None
            island = db.current_island
            tree = db.trees[island]
            node = tree.nodes[parent.id]
            selection = dict(db.last_selection or {})
            context_programs = [p for ps in context_programs_dict.values() for p in ps]
            context_ids = {f"C{i}": p for i, p in enumerate(context_programs, start=1)}

            path_text = prompt_blocks.path_trace(tree, node, recent=db.path_recent)
            attempts_text = prompt_blocks.attempt_log(node, epsilon=db.epsilon)

            # Meta-guidance on stagnation (replaces AdaEvolve's paradigm).
            forced: Optional[str] = None
            meta_ids: List[str] = []
            if self.meta_guide is not None and db.should_trigger_meta(island):
                with usage_purpose("meta_guidance"):
                    ideas = await self.meta_guide.generate(
                        self.meta_guide.build_prompt(
                            parent.solution,
                            node.score,
                            path_text,
                            attempts_text,
                            [st.name for st in db.memory.strategies.values()],
                            language=self.config.language or "python",
                        )
                    )
                meta_ids = db.register_meta_ideas(island, ideas, iteration)
                if meta_ids:
                    forced = db.rng.choice(meta_ids)
                logger.info(
                    f"Iteration {iteration}: island {island} stagnating -> "
                    f"{len(meta_ids)} meta-guidance idea(s): "
                    f"{[db.memory.strategies[i].name for i in meta_ids]}"
                )

            # Lessons, warnings and the strategy list (Strategy Memory).
            lessons, warnings = db.retrieve_strategies(node, forced=forced)
            strategies = (
                db.memory.strategy_list(db.strategy_list_cap) if db.use_strategy_memory else []
            )
            listed = {st.id for st in strategies}
            for it in lessons:  # lessons shown must be citable in the intent header
                if it["strategy"].id not in listed:
                    strategies.append(it["strategy"])
                    listed.add(it["strategy"].id)
            paradigm_strategy = db.memory.get((paradigm or {}).get("strategy_id"))
            if paradigm_strategy is not None and paradigm_strategy.id not in listed:
                strategies.append(paradigm_strategy)
                listed.add(paradigm_strategy.id)
            # External strategies the edit may draw on (for attribution).
            source_strategies = [
                {"strategy": it["strategy"]} for it in lessons
            ] + ([{"strategy": paradigm_strategy}] if paradigm_strategy is not None else [])

            lineage = {
                "island": island,
                "node": node,
                "selection": selection,
                "lessons": source_strategies,
                "context_ids": context_ids,
                "meta": {
                    "triggered": bool(meta_ids) or forced is not None,
                    "ideas": meta_ids,
                    "paradigm": (paradigm or {}).get("idea"),
                    "paradigm_skipped": self._paradigm_skipped,
                    "paradigm_strategy": paradigm_strategy.id if paradigm_strategy else None,
                },
                "prompt_sources": {
                    "lessons": [it["strategy"].id for it in lessons],
                    "warnings": [it["strategy"].id for it in warnings],
                    "context_programs": {cid: p.id for cid, p in context_ids.items()},
                    "meta_idea": forced or (paradigm_strategy.id if paradigm_strategy else None),
                },
                "retrieved": {
                    "lessons": [
                        {"id": it["strategy"].id, "a": it["a"], "b": it["b"], "rank": it["draw"]}
                        for it in lessons
                    ],
                    "warnings": [
                        {"id": it["strategy"].id, "a": it["a"], "b": it["b"]} for it in warnings
                    ],
                },
            }

            # Build context for prompt generation
            # Only database-derived data — config values are read by the
            # context builder from self.config directly.
            context = {
                "program_metrics": parent.metrics,
                "other_context_programs": context_programs_dict,
                # LineageEvolve-specific keys (consumed by LineageEvolveContextBuilder)
                "lineage_context": prompt_blocks.lineage_context(
                    path_text,
                    attempts_text,
                    prompt_blocks.lessons_block(lessons),
                    prompt_blocks.warnings_block(warnings),
                    show_lineage=db.use_lineage_context,
                    show_memory=db.use_strategy_memory,
                ),
                "strategy_list": prompt_blocks.strategy_list_block(strategies),
                "intent_instructions": prompt_blocks.intent_instructions(
                    use_memory=db.use_strategy_memory, has_strategies=bool(strategies)
                ),
                "error_context": error_context,
                # AdaEvolve's paradigm (rendered exactly as in AdaEvolve)
                "paradigm": paradigm,
                "paradigm_strategy_id": paradigm_strategy.id if paradigm_strategy else None,
            }
            # Include any extra prompt context
            for k, v in self._prompt_context.items():
                if k not in context:
                    context[k] = v

            # Build prompt (LineageEvolveContextBuilder formats the lineage blocks)
            prompt = self.context_builder.build_prompt(parent_dict, context)

            # Mark paradigm as used after prompt is built
            if paradigm:
                self.database.use_paradigm()

            # Build tracking info for child program
            parent_info = (parent_label, parent.id)
            context_info = [
                (label, p.id) for label, programs in context_programs_dict.items() for p in programs
            ]
            context_program_ids = [
                p.id for programs in context_programs_dict.values() for p in programs
            ]

            # Apply human feedback (append or replace mode)
            if self.feedback_reader:
                self.feedback_reader.set_current_prompt(prompt["system"])
                feedback = self.feedback_reader.read()
                if feedback:
                    prompt = self.feedback_reader.apply_feedback(prompt)
                    self.feedback_reader.log_usage(iteration, feedback, self.feedback_reader.mode)

            # Generate and evaluate
            return await self._execute_generation(
                parent,
                prompt,
                iteration,
                parent_info=parent_info,
                context_info=context_info,
                context_program_ids=context_program_ids,
                other_context_programs=context_programs_dict,
                lineage=lineage,
            )

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return SerializableResult(error=str(e), iteration=iteration)

    # =========================================================================
    # LLM Generation
    # =========================================================================

    async def _execute_generation(
        self,
        parent: Program,
        prompt: Dict[str, str],
        iteration: int,
        parent_info: Optional[tuple] = None,
        context_info: Optional[List[tuple]] = None,
        context_program_ids: Optional[List[str]] = None,
        other_context_programs: Optional[Dict] = None,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> SerializableResult:
        """Execute LLM generation and evaluation."""
        start_time = time.time()

        image_path = None
        child_id = str(uuid.uuid4())

        # Generate
        llm_generation_time = 0.0
        try:
            llm_start = time.time()
            if self.config.language == "image":
                from skydiscover.search.utils.discovery_utils import build_image_content

                user_content = build_image_content(
                    prompt["user"], parent, other_context_programs or {}
                )
                result = await self._call_llm(
                    prompt["system"],
                    user_content,
                    image_output=True,
                    output_dir=self._get_image_output_dir(),
                    program_id=child_id,
                )
                response = result.text or ""
                image_path = result.image_path
                if not image_path:
                    return SerializableResult(
                        error="VLM did not generate an image", iteration=iteration
                    )
            else:
                result = await self._call_llm(prompt["system"], prompt["user"])
                response = result.text
            llm_generation_time = time.time() - llm_start
        except Exception as e:
            return SerializableResult(error=f"LLM error: {e}", iteration=iteration)

        if not response and self.config.language != "image":
            return SerializableResult(error="Empty LLM response", iteration=iteration)

        # Parse code from response
        if self.config.language == "image":
            child_solution = response or "(image generated)"
            changes = "Image generation"
        elif self.config.diff_based_generation:
            diffs = extract_diffs(response)
            if diffs:
                child_solution = apply_diff(parent.solution, response)
                changes = format_diff_summary(diffs)
            else:
                # No diffs found, try full rewrite
                child_solution = parse_full_rewrite(response, self.config.language)
                changes = "Full rewrite"
        else:
            child_solution = parse_full_rewrite(response, self.config.language)
            changes = "Full rewrite"

        if not child_solution:
            return SerializableResult(error="No valid solution in response", iteration=iteration)

        # LineageEvolve: declared intent and pre-evaluation deduplication
        # (Section 10).  A duplicate is recorded as a failed attempt and, like
        # any other failure, retried by _run_normal_step.
        intent = parse_intent(response or "")
        candidate_hash = code_hash(child_solution)
        if lineage and self.database.use_dedup and candidate_hash in self.database.hash_index:
            self._commit(
                lineage, iteration, STATUS_DUPLICATE, intent,
                error="duplicate", candidate_hash=candidate_hash,
            )
            return SerializableResult(
                error=(
                    "Duplicate: the generated program is identical to an existing program "
                    "(not evaluated). Produce a materially different program."
                ),
                iteration=iteration,
            )

        # Evaluate
        try:
            eval_input = image_path if self.config.language == "image" else child_solution
            eval_start = time.time()
            eval_result = await self.evaluator.evaluate_program(eval_input, child_id)
            eval_time = time.time() - eval_start
        except Exception as e:
            return SerializableResult(error=f"Evaluation error: {e}", iteration=iteration)

        metrics = eval_result.metrics
        artifacts = eval_result.artifacts

        # Check for eval failure (e.g., constraint violation, malformed solution)
        if (
            metrics.get("validity") in (0, -1)
            or (metrics.get("timeout") is True and metrics.get("validity") is None)
            or (
                metrics.get("combined_score") == 0
                and (metrics.get("error") is not None or "error" in (artifacts or {}))
            )
        ):
            error_msg = (
                (metrics.get("error") if isinstance(metrics.get("error"), str) else None)
                or (artifacts or {}).get("error")
                or metrics.get("error_message")
                or "Evaluation failed"
            )
            if lineage:
                self._commit(
                    lineage, iteration, STATUS_CRASHED, intent,
                    error=self._failure_class(error_msg), candidate_hash=candidate_hash,
                )
            return SerializableResult(error=f"Eval failure: {error_msg}", iteration=iteration)

        # Extract image_path from evaluator metrics (non-image mode fallback)
        if not image_path:
            image_path = (
                metrics.pop("image_path", None)
                if isinstance(metrics.get("image_path"), str)
                else None
            )

        # Build child program with full tracking info
        child_metadata = {"changes": changes, "parent_metrics": parent.metrics, "intent": intent}
        if image_path:
            child_metadata["image_path"] = image_path
        child = Program(
            id=child_id,
            solution=child_solution,
            language=self.config.language,
            metrics=metrics,
            iteration_found=iteration,
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            parent_info=parent_info,
            context_info=context_info,
            generation=parent.generation + 1,
            metadata=child_metadata,
            artifacts=artifacts,
        )

        if lineage:
            # A finite score is required for admission (Section 2.2).
            if not math.isfinite(self.database._get_fitness(child)):
                self._commit(
                    lineage, iteration, STATUS_CRASHED, intent,
                    error="non-finite score", candidate_hash=candidate_hash,
                )
                return SerializableResult(
                    error="Eval failure: non-finite score", iteration=iteration
                )
            # Interpret the edit (Section 5) and commit all updates (9-13).
            interpretation = await self._interpret(lineage, parent, child, intent)
            self._commit(
                lineage, iteration, STATUS_ADMITTED, intent,
                child=child, candidate_hash=candidate_hash, interpretation=interpretation,
            )

        iteration_time = time.time() - start_time

        return SerializableResult(
            child_program_dict=child.to_dict(),
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            iteration_time=iteration_time,
            llm_generation_time=llm_generation_time,
            eval_time=eval_time,
            prompt=prompt,
            llm_response=response,
            iteration=iteration,
        )

    # =========================================================================
    # LineageEvolve helpers
    # =========================================================================

    @staticmethod
    def _failure_class(error_msg: Any) -> str:
        """Short failure description for the attempt log."""
        text = str(error_msg or "evaluation failed").strip()
        first = text.splitlines()[0] if text else "evaluation failed"
        return first[:160]

    def _commit(
        self,
        lineage: Dict[str, Any],
        iteration: int,
        status: str,
        intent: Dict[str, Any],
        *,
        child: Optional[Program] = None,
        error: Optional[str] = None,
        candidate_hash: Optional[str] = None,
        interpretation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one attempt and apply all its search-state updates."""
        db = self.database
        selection = lineage.get("selection") or {}
        record = db.commit_attempt(
            parent_id=lineage["node"].id,
            island=lineage["island"],
            iteration=iteration,
            status=status,
            intent=intent,
            prompt_sources=lineage["prompt_sources"],
            child_program=child,
            error=error,
            candidate_hash=candidate_hash,
            interpretation=interpretation,
            selection_path=selection.get("path"),
        )
        tree = db.trees[lineage["island"]]
        interp = interpretation or {}
        self._iter_entry.setdefault("tries", []).append(
            {
                "status": status,
                "attempt_id": record.id,
                "parent_id": record.parent,
                "parent_score": record.parent_score,
                "selection": selection,
                "meta_guidance": lineage.get("meta"),
                "retrieved": lineage.get("retrieved"),
                "intent": intent,
                "child_id": record.child,
                "child_score": record.score,
                "delta": record.delta,
                "y_parent": record.y_parent,
                "q": record.q,
                "strategy_id": record.strategy_id,
                "strategy_name": record.strategy_name,
                "strategy_shares": record.shares,
                "ext_shares": record.ext_shares,
                "interpretation": {k: interp.get(k) for k in ("valid", "q", "labels", "units")},
                "error": error,
                "theta_after": tree.theta,
                "island_best_after": tree.best_score,
                "attempts_since_progress": tree.attempts_since_progress,
            }
        )
        if status == STATUS_ADMITTED:
            logger.info(
                f"Iteration {iteration}: island {lineage['island']} parent "
                f"{record.parent[:8]} -> child {record.child[:8]} "
                f"score={record.score:.6f} delta={record.delta:+.6f} y={record.y_parent} "
                f"q={record.q:.2f} strategy={record.strategy_name} "
                f"({selection.get('policy')}, depth {lineage['node'].depth})"
            )

    async def _interpret(
        self,
        lineage: Dict[str, Any],
        parent: Program,
        child: Program,
        intent: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Edit Interpreter call and deterministic label validation (Section 5)."""
        db = self.database
        units = build_inventory(parent.solution, child.solution, self.config.language)
        declared = db.peek_strategy(intent)
        want_attribution = db.use_source_attribution
        want_lesson = db.use_strategy_memory and (declared["is_new"] or declared["id"])
        out: Dict[str, Any] = {"valid": False, "units": [u.to_dict() for u in units]}
        if not units or not (want_attribution or want_lesson):
            return out

        node = lineage["node"]
        tree = db.trees[lineage["island"]]
        parent_score = node.score
        child_score = db._get_fitness(child)
        if tree.is_progress(parent_score, child_score):
            outcome = "improved (progress)"
        elif child_score < parent_score:
            outcome = "regressed"
        else:
            outcome = "no change"
        lessons = lineage.get("lessons") or []
        context_code = {cid: p.solution for cid, p in lineage["context_ids"].items()}
        lean_interpreter = db.interpreter_context == "matched"
        prompt = self.interpreter.build_prompt(
            units=units,
            intent=intent,
            strategy_sources=[
                {
                    "id": it["strategy"].id,
                    "name": it["strategy"].name,
                    "summary": it["strategy"].summary,
                    "mechanism": it["strategy"].mechanism,
                    "origin": it["strategy"].origin,
                }
                for it in lessons
            ],
            context_programs=(
                context_code if (want_attribution and not lean_interpreter) else {}
            ),
            context_matches=(
                context_overlap(units, context_code, parent.solution)
                if (want_attribution and lean_interpreter)
                else None
            ),
            context_ids=list(context_code) if lean_interpreter else None,
            parent_score=parent_score,
            child_score=child_score,
            outcome=outcome,
            declared_strategy=declared,
            want_attribution=want_attribution,
        )
        with usage_purpose("interpreter"):
            data = await self.interpreter.interpret(prompt)
        if not isinstance(data, dict):
            return out  # abstained / invalid output -> q_e = 1
        out["lesson"] = data.get("lesson")
        out["new_condition"] = data.get("new_condition")
        if not want_attribution:
            return out
        labels = validate_labels(
            data.get("units"),
            units,
            strategy_source_ids={it["strategy"].id for it in lessons},
            declared_sources=set(intent.get("sources") or []),
            context_programs=context_code,
            parent_code=parent.solution,
        )
        if labels is None:
            return out
        q, ext_shares = compute_shares(units, labels)
        out.update(valid=True, q=q, ext_shares=ext_shares, labels=labels)
        return out

    def _write_lineage_log(self, result: SerializableResult, iteration_time: float) -> None:
        """One JSON line per iteration with every try's attempt record."""
        if not self._lineage_log_path:
            return
        db = self.database
        entry = dict(self._iter_entry)
        tries = entry.get("tries", [])
        entry.update(
            # Final outcome of the iteration: the status of the last recorded
            # attempt, or an infrastructure failure if none was recorded.
            status=(
                tries[-1]["status"]
                if tries and (result.error is None or tries[-1]["status"] != STATUS_ADMITTED)
                else ("infrastructure_failure" if result.error else STATUS_ADMITTED)
            ),
            error=result.error,
            timestamp=datetime.now().isoformat(),
            iteration_time_seconds=iteration_time,
            llm_generation_time_seconds=result.llm_generation_time,
            eval_time_seconds=result.eval_time,
            token_usage=getattr(self, "_last_iteration_usage", None),
            cumulative_token_usage=USAGE.snapshot()["total"],
            elapsed_seconds=time.time() - getattr(self, "_run_start_time", time.time()),
            global_best_score=(
                None if math.isinf(db._global_best_score) else db._global_best_score
            ),
            new_roots=db.drain_root_events(),
        )
        try:
            with open(self._lineage_log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write lineage log: {e}")
