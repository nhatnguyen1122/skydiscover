"""LineageEvolve controller: one iteration of Section 11 of the method spec.

Built on ``AdaEvolveController``.  The outer loop (island scheduler,
migration, spawning, checkpoint cadence, AdaEvolve-format iteration log) is
inherited; the body of an iteration is replaced:

  1. island       - AdaEvolve's scheduler (already chosen: database.current_island)
  2. select       - Thompson sampling on R(u) + log p~_u; context as AdaEvolve
  3. stagnation   - meta-guidance ideas -> Strategy Memory
  4. prompt       - path trace, attempt log, lessons, warnings, context, strategy list
  5. expand       - one LLM call -> intent header + candidate
  6. dedup        - normalised hash against the run-wide index
  7. evaluate     - admitted / crashed / infrastructure failure
  8. interpret    - changed-unit inventory + Edit Interpreter -> q_e, shares
  9-13. commit    - backup, child prior, island scalars, memory, scheduler
                    (``LineageEvolveDatabase.commit_attempt``, atomic)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.context_builder.lineageevolve import LineageEvolveContextBuilder
from skydiscover.llm.usage import USAGE, usage_purpose
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput
from skydiscover.search.lineageevolve import prompt_blocks
from skydiscover.search.lineageevolve.attribution import (
    build_inventory,
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
from skydiscover.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)

STATUS_INFRA = "infrastructure_failure"


class LineageEvolveController(AdaEvolveController):
    """Evolution controller implementing the LineageEvolve iteration."""

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        db = self.database
        db_config = self.config.search.database

        self.context_builder = LineageEvolveContextBuilder(self.config)

        self.interpreter = EditInterpreter(
            self.guide_llms,
            temperature=float(getattr(db_config, "interpreter_temperature", 0.0)),
            max_tokens=int(getattr(db_config, "interpreter_max_tokens", 4000)),
        )
        self.meta_guide: Optional[MetaGuide] = None
        if db.use_meta_guidance and db.use_strategy_memory:
            self.meta_guide = MetaGuide(
                self.guide_llms,
                task_description=self.context_builder._get_system_message() or "",
                evaluator_code=self._load_evaluator_code(),
                num_ideas=db.meta_num_ideas,
            )

        # Infrastructure failures (LLM/API error, no candidate parsed) create
        # no attempt record; like AdaEvolve they may be retried inside the
        # same iteration.
        self.infra_retries = self.max_retries if self.enable_retry else 0
        # Candidate failures (crashed / duplicate) are retried on the same
        # parent, as AdaEvolve does, unless retry_candidate_failures is off.
        self.candidate_retries = (
            self.max_retries
            if self.enable_retry and getattr(db_config, "retry_candidate_failures", True)
            else 0
        )

        self._lineage_log_path: Optional[str] = None
        logger.info(
            f"LineageEvolveController initialized (infra_retries={self.infra_retries}, "
            f"candidate_retries={self.candidate_retries}, "
            f"meta_guidance={self.meta_guide is not None})"
        )

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        self.database.total_iterations = start_iteration + max_iterations
        out_dir = self.output_dir or getattr(self.database.config, "db_path", None) or "."
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._lineage_log_path = os.path.join(out_dir, f"lineageevolve_iterations_{stamp}.jsonl")
        logger.info(f"LineageEvolve iteration log: {self._lineage_log_path}")
        best = await super().run_discovery(start_iteration, max_iterations, checkpoint_callback)
        logger.info(f"LineageEvolve final state: {json.dumps(self.database.lineage_stats())}")
        return best

    def _write_lineage_log(self, entry: Dict[str, Any]) -> None:
        if not self._lineage_log_path:
            return
        try:
            with open(self._lineage_log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write lineage log: {e}")

    # ------------------------------------------------------------------
    # Candidate parsing / evaluation outcome
    # ------------------------------------------------------------------

    def _parse_candidate(self, response: str, parent: Program) -> Tuple[Optional[str], str]:
        """Extract the candidate program from the LLM response.

        Returns ``(solution, changes_summary)``; ``solution`` is ``None`` when
        no candidate could be parsed (an infrastructure failure).
        """
        if self.config.diff_based_generation:
            diffs = extract_diffs(response)
            if diffs:
                solution = apply_diff(parent.solution, response)
                if solution == parent.solution:
                    return None, "SEARCH blocks did not match the parent"
                return solution, format_diff_summary(diffs)
        if "```" not in response:
            return None, "no code block in response"
        solution = parse_full_rewrite(response, self.config.language)
        if not solution or not solution.strip():
            return None, "empty code block"
        return solution, "Full rewrite"

    @staticmethod
    def _candidate_failure(metrics: Dict[str, Any], artifacts: Dict[str, Any]) -> Optional[str]:
        """Return a short failure class if the candidate failed because of itself."""
        metrics = metrics or {}
        artifacts = artifacts or {}
        validity = metrics.get("validity")
        score = metrics.get("combined_score")
        has_error = metrics.get("error") is not None or "error" in artifacts
        failed = (
            validity in (0, -1)
            or (metrics.get("timeout") is True and validity is None)
            or (score == 0 and has_error)
            or (score is None and has_error)
        )
        if not failed and score is not None:
            try:
                failed = not math.isfinite(float(score))
            except (TypeError, ValueError):
                failed = True
        if not failed:
            return None

        if metrics.get("timeout") is True:
            return "timeout"
        for msg in (
            metrics.get("error") if isinstance(metrics.get("error"), str) else None,
            artifacts.get("error") if isinstance(artifacts.get("error"), str) else None,
            metrics.get("error_message"),
        ):
            if msg:
                first = str(msg).strip().splitlines()[0] if str(msg).strip() else ""
                if first:
                    return first[:160]
        if validity in (0, -1):
            return "invalid output (failed the evaluator's validity check)"
        return "non-finite or missing score"

    # ------------------------------------------------------------------
    # One iteration
    # ------------------------------------------------------------------

    async def _run_iteration(self, iteration: int, checkpoint_callback) -> None:
        start = time.time()
        usage_before = USAGE.snapshot()
        db = self.database
        entry: Dict[str, Any] = {"iteration": iteration, "status": STATUS_INFRA}
        timings = {"llm": 0.0, "eval": 0.0, "interpreter": 0.0, "meta_guidance": 0.0}
        child: Optional[Program] = None
        prompt: Optional[Dict[str, str]] = None
        response: Optional[str] = None
        error: Optional[str] = None

        try:
            if not db.programs:
                # Bootstrap without an initial program (not part of the method).
                result = await self._run_from_scratch_iteration(iteration)
                if result.error:
                    raise RuntimeError(result.error)
                db.add(Program(**result.child_program_dict), iteration=iteration)
                entry["status"] = "bootstrap"
                return

            self._ensure_all_islands_seeded()

            # ---- 1-2. island, parent, context -------------------------------
            island = db.current_island
            parent_dict, context_dict = db.sample(self.num_context_programs)
            parent = list(parent_dict.values())[0]
            selection = db.last_selection
            tree = db.trees[island]
            node = tree.nodes[parent.id]
            context_programs: List[Program] = [p for ps in context_dict.values() for p in ps]
            context_ids = {f"C{i}": p for i, p in enumerate(context_programs, start=1)}
            entry.update(
                island=island,
                parent_id=parent.id,
                parent_score=node.score,
                parent_depth=node.depth,
                selection=selection.to_dict() if selection else None,
                context_programs={cid: p.id for cid, p in context_ids.items()},
            )

            path_text = prompt_blocks.path_trace(tree, node, recent=db.path_recent)
            attempts_text = prompt_blocks.attempt_log(node, epsilon=db.epsilon)

            # ---- 3. stagnation check ---------------------------------------
            forced: Optional[str] = None
            meta_ids: List[str] = []
            meta_triggered = False
            if self.meta_guide is not None and db.should_trigger_meta(island):
                meta_triggered = True
                t = time.time()
                with usage_purpose("meta_guidance"):
                    meta_prompt = self.meta_guide.build_prompt(
                        parent.solution,
                        node.score,
                        path_text,
                        attempts_text,
                        [s.name for s in db.memory.strategies.values()],
                        language=self.config.language or "python",
                    )
                    ideas = await self.meta_guide.generate(meta_prompt)
                timings["meta_guidance"] = time.time() - t
                meta_ids = db.register_meta_ideas(island, ideas, iteration)
                if meta_ids:
                    forced = db.rng.choice(meta_ids)
                logger.info(
                    f"Iteration {iteration}: island {island} stagnating "
                    f"({tree.attempts_since_progress} attempts without progress) -> "
                    f"{len(meta_ids)} meta-guidance idea(s): "
                    f"{[db.memory.strategies[i].name for i in meta_ids]}"
                )
            entry["meta_guidance"] = {"triggered": meta_triggered, "ideas": meta_ids, "forced": forced}

            # ---- 4. prompt --------------------------------------------------
            lessons, warnings = db.retrieve_strategies(node, forced=forced)
            lesson_ids = [it["strategy"].id for it in lessons]
            warning_ids = [it["strategy"].id for it in warnings]
            strategies = (
                db.memory.strategy_list(db.strategy_list_cap) if db.use_strategy_memory else []
            )
            listed = {s.id for s in strategies}
            for it in lessons:  # lessons shown must be citable in the intent header
                if it["strategy"].id not in listed:
                    strategies.append(it["strategy"])
                    listed.add(it["strategy"].id)
            entry["retrieved"] = {
                "lessons": [
                    {"id": it["strategy"].id, "a": it["a"], "b": it["b"], "draw": it["draw"]}
                    for it in lessons
                ],
                "warnings": [
                    {"id": it["strategy"].id, "a": it["a"], "b": it["b"]} for it in warnings
                ],
            }
            prompt_sources = {
                "lessons": lesson_ids,
                "warnings": warning_ids,
                "context_programs": {cid: p.id for cid, p in context_ids.items()},
                "meta_idea": forced,
            }

            base_context = {
                "program_metrics": parent.metrics,
                "other_context_programs": context_dict,
                "strategy_list": prompt_blocks.strategy_list_block(strategies),
                "intent_instructions": prompt_blocks.intent_instructions(
                    use_memory=db.use_strategy_memory, has_strategies=bool(strategies)
                ),
            }
            for k, v in self._prompt_context.items():
                if k not in base_context:
                    base_context[k] = v

            # ---- 5-7. expand, dedup, evaluate (with in-iteration retries) ----
            # Like AdaEvolve, a candidate that fails because of itself (crashed
            # or duplicate) may be retried inside the same iteration.  Every
            # failed try is still recorded on the parent, so the retry prompt
            # shows it in "Already tried on this program" together with the
            # error.  With retry_failure_evidence="final" only the last outcome
            # of the iteration is backed up as evidence.
            max_tries = 1 + self.candidate_retries
            entry["tries"] = []
            llm_calls = 0
            error_context: Optional[str] = None
            child_solution: Optional[str] = None
            changes = ""
            intent: Dict[str, Any] = {}
            declared: Dict[str, Any] = {}
            candidate_hash = ""
            for try_idx in range(max_tries):
                final_try = try_idx == max_tries - 1
                as_evidence = final_try or db.retry_failure_evidence == "all"

                context = dict(base_context)
                context["lineage_context"] = prompt_blocks.lineage_context(
                    path_text,
                    prompt_blocks.attempt_log(node, epsilon=db.epsilon),
                    prompt_blocks.lessons_block(lessons),
                    prompt_blocks.warnings_block(warnings),
                    show_lineage=db.use_lineage_context,
                    show_memory=db.use_strategy_memory,
                )
                context["error_context"] = error_context
                prompt = self.context_builder.build_prompt(parent_dict, context)

                if self.feedback_reader:
                    self.feedback_reader.set_current_prompt(prompt["system"])
                    feedback = self.feedback_reader.read()
                    if feedback:
                        prompt = self.feedback_reader.apply_feedback(prompt)
                        self.feedback_reader.log_usage(
                            iteration, feedback, self.feedback_reader.mode
                        )

                # ---- 5. expand ----------------------------------------------
                child_solution = None
                for _ in range(1 + self.infra_retries):
                    llm_calls += 1
                    t = time.time()
                    try:
                        with usage_purpose("mutation"):
                            result = await self._call_llm(prompt["system"], prompt["user"])
                        response = result.text
                    except Exception as e:
                        error = f"LLM error: {e}"
                        response = None
                    finally:
                        timings["llm"] += time.time() - t
                    if not response:
                        error = error or "Empty LLM response"
                        continue
                    child_solution, changes = self._parse_candidate(response, parent)
                    if child_solution:
                        error = None
                        break
                    error = f"No candidate parsed ({changes})"
                entry["llm_calls"] = llm_calls
                if not child_solution:
                    return  # infrastructure failure: no record, no update

                intent = parse_intent(response)
                entry["intent"] = intent
                declared = db.peek_strategy(intent)
                candidate_hash = code_hash(child_solution)

                failure_status: Optional[str] = None
                failure: Optional[str] = None
                artifacts: Dict[str, Any] = {}

                # ---- 6. dedup -----------------------------------------------
                if db.use_dedup and candidate_hash in db.hash_index:
                    failure_status, failure = STATUS_DUPLICATE, "duplicate"
                    error = "duplicate of an existing program (not evaluated)"
                else:
                    # ---- 7. evaluate ----------------------------------------
                    child_id = str(uuid.uuid4())
                    t = time.time()
                    try:
                        eval_result = await self.evaluator.evaluate_program(
                            child_solution, child_id
                        )
                    except Exception as e:
                        error = f"Evaluation error: {e}"
                        return  # infrastructure failure
                    finally:
                        timings["eval"] += time.time() - t
                    metrics = eval_result.metrics or {}
                    artifacts = eval_result.artifacts or {}

                    failure = self._candidate_failure(metrics, artifacts)
                    if failure is None:
                        child = Program(
                            id=child_id,
                            solution=child_solution,
                            language=self.config.language,
                            metrics=metrics,
                            iteration_found=iteration,
                            parent_id=parent.id,
                            other_context_ids=[p.id for p in context_programs],
                            parent_info=("", parent.id),
                            context_info=[("", p.id) for p in context_programs],
                            generation=parent.generation + 1,
                            metadata={
                                "changes": changes,
                                "parent_metrics": parent.metrics,
                                "intent": intent,
                                "island": island,
                                "tries": try_idx + 1,
                            },
                            artifacts=artifacts,
                        )
                        if not math.isfinite(db._get_fitness(child)):
                            failure = "non-finite score"
                            child = None
                    if failure is not None:
                        failure_status = STATUS_CRASHED
                        error = failure

                if failure_status is None:
                    entry["tries"].append({"try": try_idx + 1, "status": STATUS_ADMITTED})
                    break  # admitted -> interpret and commit below

                record = db.commit_attempt(
                    parent_id=parent.id,
                    island=island,
                    iteration=iteration,
                    status=failure_status,
                    intent=intent,
                    prompt_sources=prompt_sources,
                    error=failure,
                    candidate_hash=candidate_hash,
                    evidence=as_evidence,
                    count_visit=final_try,
                )
                entry["tries"].append(
                    {
                        "try": try_idx + 1,
                        "status": failure_status,
                        "attempt_id": record.id,
                        "error": failure,
                        "evidence": as_evidence,
                        "strategy": record.strategy_name,
                    }
                )
                if final_try:
                    entry["status"] = failure_status
                    self._fill_record(entry, record, tree)
                    return
                error_context = self._retry_context(failure_status, failure, artifacts)
                logger.info(
                    f"Iteration {iteration}: try {try_idx + 1}/{max_tries} {failure_status} "
                    f"({failure}); retrying on the same parent"
                )
            error = None

            # ---- 8. interpret -----------------------------------------------
            child_score = db._get_fitness(child)
            interpretation: Dict[str, Any] = {"valid": False}
            units = build_inventory(parent.solution, child_solution, self.config.language)
            want_attribution = db.use_source_attribution
            want_lesson = db.use_strategy_memory and (declared["is_new"] or declared["id"])
            if units and (want_attribution or want_lesson):
                t = time.time()
                interpretation = await self._interpret(
                    units=units,
                    intent=intent,
                    declared=declared,
                    lessons=lessons,
                    lesson_ids=lesson_ids,
                    context_ids=context_ids,
                    parent=parent,
                    parent_score=node.score,
                    child_score=child_score,
                    progress=tree.is_progress(node.score, child_score),
                    want_attribution=want_attribution,
                )
                timings["interpreter"] = time.time() - t
            entry["interpretation"] = {
                k: interpretation.get(k) for k in ("valid", "q", "ext_shares", "labels", "units")
            }

            # ---- 9-13. commit -----------------------------------------------
            record = db.commit_attempt(
                parent_id=parent.id,
                island=island,
                iteration=iteration,
                status=STATUS_ADMITTED,
                intent=intent,
                prompt_sources=prompt_sources,
                child_program=child,
                candidate_hash=candidate_hash,
                interpretation=interpretation,
            )
            entry["status"] = STATUS_ADMITTED
            self._fill_record(entry, record, tree)

        except Exception as e:
            logger.exception(f"Iteration {iteration} failed: {e}")
            error = error or f"Unhandled error: {e}"
        finally:
            iteration_time = time.time() - start
            self._last_iteration_usage = USAGE.delta(usage_before)
            entry.update(
                error=error,
                timestamp=datetime.now().isoformat(),
                iteration_time_seconds=iteration_time,
                timings_seconds=timings,
                token_usage=self._last_iteration_usage,
                cumulative_token_usage=USAGE.snapshot()["total"],
                elapsed_seconds=time.time() - getattr(self, "_run_start_time", start),
                global_best_score=(
                    None if math.isinf(db._global_best_score) else db._global_best_score
                ),
                new_roots=db.drain_root_events(),
            )
            self._write_lineage_log(entry)
            self._finish_iteration(
                iteration, entry, child, prompt, response, error, iteration_time, timings,
                checkpoint_callback,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _interpret(
        self,
        *,
        units,
        intent: Dict[str, Any],
        declared: Dict[str, Any],
        lessons: List[Dict[str, Any]],
        lesson_ids: List[str],
        context_ids: Dict[str, Program],
        parent: Program,
        parent_score: float,
        child_score: float,
        progress: bool,
        want_attribution: bool,
    ) -> Dict[str, Any]:
        """Call the Edit Interpreter and validate its labels deterministically."""
        if child_score > parent_score and progress:
            outcome = "improved (progress)"
        elif child_score < parent_score:
            outcome = "regressed"
        else:
            outcome = "no change"
        strategy_sources = [
            {
                "id": it["strategy"].id,
                "name": it["strategy"].name,
                "summary": it["strategy"].summary,
                "mechanism": it["strategy"].mechanism,
                "origin": it["strategy"].origin,
            }
            for it in lessons
        ]
        context_code = {cid: p.solution for cid, p in context_ids.items()}
        prompt = self.interpreter.build_prompt(
            units=units,
            intent=intent,
            strategy_sources=strategy_sources,
            context_programs=context_code if want_attribution else {},
            parent_score=parent_score,
            child_score=child_score,
            outcome=outcome,
            declared_strategy=declared,
            want_attribution=want_attribution,
        )
        with usage_purpose("interpreter"):
            data = await self.interpreter.interpret(prompt)

        out: Dict[str, Any] = {"valid": False, "units": [u.to_dict() for u in units]}
        if not isinstance(data, dict):
            return out  # abstained / invalid output -> q_e = 1
        out["lesson"] = data.get("lesson")
        out["new_condition"] = data.get("new_condition")
        if not want_attribution:
            return out
        labels = validate_labels(
            data.get("units"),
            units,
            strategy_source_ids=set(lesson_ids),
            declared_sources=set(intent.get("sources") or []),
            context_programs=context_code,
            parent_code=parent.solution,
        )
        if labels is None:
            return out
        q, ext_shares = compute_shares(units, labels)
        out.update(valid=True, q=q, ext_shares=ext_shares, labels=labels)
        return out

    @staticmethod
    def _retry_context(status: str, failure: Optional[str], artifacts: Dict[str, Any]) -> str:
        """Error text shown to the model when a failed candidate is retried."""
        if status == STATUS_DUPLICATE:
            return (
                "Your previous attempt reproduced a program that already exists in the search "
                "(identical code after normalisation), so it was not evaluated. Produce a "
                "materially different program."
            )
        parts = [f"Your previous attempt failed evaluation: {failure}"]
        for key in ("stderr", "traceback", "error", "failure_stage"):
            value = (artifacts or {}).get(key)
            if isinstance(value, str) and value.strip() and value.strip() != (failure or ""):
                parts.append(f"{key}: {value.strip()[-1500:]}")
        parts.append("Fix this problem; the result must be a valid, runnable program.")
        return "\n".join(parts)

    @staticmethod
    def _fill_record(entry: Dict[str, Any], record, tree) -> None:
        entry.update(
            attempt_id=record.id,
            child_id=record.child,
            child_score=record.score,
            delta=record.delta,
            y_parent=record.y_parent,
            q=record.q,
            strategy_shares=record.shares,
            ext_shares=record.ext_shares,
            strategy_id=record.strategy_id,
            strategy_name=record.strategy_name,
            theta_after=tree.theta,
            island_best_after=tree.best_score,
            attempts_since_progress=tree.attempts_since_progress,
        )

    def _finish_iteration(
        self,
        iteration: int,
        entry: Dict[str, Any],
        child: Optional[Program],
        prompt: Optional[Dict[str, str]],
        response: Optional[str],
        error: Optional[str],
        iteration_time: float,
        timings: Dict[str, float],
        checkpoint_callback,
    ) -> None:
        """Logging, monitor callback and checkpoint cadence (as AdaEvolve)."""
        db = self.database
        status = entry.get("status")
        admitted = status == STATUS_ADMITTED and child is not None

        if admitted:
            if self.monitor_callback:
                try:
                    self.monitor_callback(child, iteration)
                except Exception:
                    logger.debug("Monitor callback error", exc_info=True)
            if prompt:
                db.log_prompt(
                    template_key=(
                        "diff_user_message"
                        if self.config.diff_based_generation
                        else "full_rewrite_user_message"
                    ),
                    program_id=child.id,
                    prompt=prompt,
                    responses=[response] if response else [],
                )
            sel = entry.get("selection") or {}
            logger.info(
                f"Iteration {iteration}: island {entry.get('island')} "
                f"parent {str(entry.get('parent_id'))[:8]} (depth {entry.get('parent_depth')}, "
                f"R={sel.get('R', 0.0):.2f}, p~={sel.get('p_sample', 0.0):.2f}) -> "
                f"child {child.id[:8]} score={entry.get('child_score'):.6f} "
                f"delta={entry.get('delta'):+.6f} y={entry.get('y_parent')} q={entry.get('q'):.2f} "
                f"strategy={entry.get('strategy_name')} "
                f"[{iteration_time:.1f}s: llm {timings['llm']:.1f}s, eval {timings['eval']:.1f}s]"
            )
            if db.best_program_id == child.id:
                logger.info(f"New best solution found at iteration {iteration}")
        else:
            logger.warning(f"Iteration {iteration}: {status}: {error}")

        # AdaEvolve-format stats line (consumed by scripts/export_convergence.py).
        self._log_iteration_stats(
            iteration=iteration,
            sampling_mode="lineage",
            sampling_intensity=None,
            child_program=child.to_dict() if admitted else None,
            iteration_time=iteration_time,
            llm_generation_time=timings["llm"],
            eval_time=timings["eval"],
            error=None if admitted else f"{status}: {error}",
        )

        if iteration > 0 and iteration % self.config.checkpoint_interval == 0:
            logger.info(f"Checkpoint interval reached at iteration {iteration}")
            db.log_status()
            if checkpoint_callback:
                checkpoint_callback(iteration)
