"""LLM helpers of LineageEvolve: the Edit Interpreter and meta-guidance.

* ``EditInterpreter`` (Section 5.3): one low-temperature call per admitted
  edit that labels every changed unit with its source and writes the lesson
  text of the declared strategy.
* ``MetaGuide`` (Section 8.5): proposes strategy-level ideas for a stagnating
  island, conditioned on the selected parent's lineage context.

Both return plain dicts; all validation and all state updates happen in the
caller, deterministically.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from skydiscover.llm.llm_pool import LLMPool
from skydiscover.search.lineageevolve.attribution import ChangedUnit, snake_case

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Optional[Any]:
    """Best-effort extraction of a JSON object/array from an LLM response."""
    if not text:
        return None
    candidates: List[str] = []
    for m in re.finditer(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL):
        candidates.append(m.group(1))
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j > i:
            candidates.append(text[i : j + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except (ValueError, TypeError):
            continue
    return None


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


INTERPRETER_SYSTEM = (
    "You are the Edit Interpreter of an evolutionary program-search system. "
    "You analyse ONE edit that turned a parent program into a child program and "
    "attribute each changed unit to its source. You are precise and conservative: "
    "you only credit an external source when the changed code clearly implements it. "
    "Respond with a single JSON object and nothing else."
)


class EditInterpreter:
    """Labels changed units with their source and writes lesson text."""

    def __init__(self, llm_pool: LLMPool, temperature: float = 0.0, max_tokens: int = 4000):
        self.llms = llm_pool
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_prompt(
        self,
        units: List[ChangedUnit],
        intent: Dict[str, Any],
        strategy_sources: List[Dict[str, Any]],
        context_programs: Dict[str, str],
        parent_score: float,
        child_score: float,
        outcome: str,
        declared_strategy: Dict[str, Any],
        want_attribution: bool = True,
        context_matches: Optional[Dict[str, Dict[str, float]]] = None,
        context_ids: Optional[List[str]] = None,
    ) -> str:
        """Interpreter prompt.  With ``context_matches`` (lean mode) the code of
        the context programs is not included; instead each unit lists the
        fraction of its added lines found in each context program (computed
        deterministically), which is what label validation uses anyway."""
        lean = context_matches is not None
        parts: List[str] = []
        parts.append("# Edit under analysis")
        parts.append(f"- Parent score: {parent_score:.6f}")
        parts.append(f"- Child score: {child_score:.6f}")
        parts.append(f"- Outcome: {outcome}")
        parts.append("")
        parts.append("# Intent declared by the author BEFORE evaluation")
        parts.append(f"- strategy: {declared_strategy.get('label', 'unspecified')}")
        parts.append(f"- sources: {intent.get('sources') or []}")
        parts.append(f"- description: {intent.get('description') or '(none)'}")
        parts.append("")

        parts.append("# External sources that were in the author's prompt")
        if strategy_sources:
            for s in strategy_sources:
                kind = "meta-guidance idea" if s.get("origin") == "meta_guide" else "lesson"
                parts.append(f"## {s['id']} ({kind}): {s['name']}")
                if s.get("summary"):
                    parts.append(f"Summary: {s['summary']}")
                if s.get("mechanism"):
                    parts.append(f"Mechanism: {s['mechanism']}")
        if lean and context_ids:
            parts.append(
                f"## Context programs {', '.join(context_ids)} (code not shown; for each changed "
                "unit below, the share of its added lines copied from each context program is given)"
            )
        elif context_programs:
            for cid, code in context_programs.items():
                parts.append(f"## {cid} (context program)")
                parts.append("```\n" + _clip(code, 6000) + "\n```")
        if not strategy_sources and not context_programs and not (lean and context_ids):
            parts.append("(none)")
        parts.append("")

        parts.append("# Changed units (parent -> child)")
        for u in units:
            parts.append(
                f"## {u.id} | enclosing symbol: {u.symbol or '(module level)'} | "
                f"size: {u.size} lines ({len(u.removed)} removed, {len(u.added)} added)"
            )
            parts.append("Removed from parent:")
            parts.append("```\n" + (u.excerpt(u.removed) or "(nothing)") + "\n```")
            parts.append("Added in child:")
            parts.append("```\n" + (u.excerpt(u.added) or "(nothing)") + "\n```")
            if lean:
                m = (context_matches or {}).get(u.id) or {}
                copied = ", ".join(f"{cid} {frac:.0%}" for cid, frac in sorted(m.items()))
                parts.append(f"Added lines copied from context programs: {copied or 'none'}")
        parts.append("")

        parts.append("# Your task")
        if want_attribution:
            parts.append(
                "1. For EVERY unit id above give exactly one source label:\n"
                '   - "lineage": continues, refines or tunes a mechanism already present in the parent;\n'
                '   - "novel": introduces an idea not traceable to the parent\'s existing mechanisms '
                "or to any external source above;\n"
                '   - "ext:<source id>": implements a specific lesson / meta-guidance idea (e.g. "ext:S3"), '
                'or borrows code from a specific context program (e.g. "ext:C1").\n'
                "   If you cannot judge, abstain by returning \"units\": null."
            )
        else:
            parts.append('1. Return "units": null (attribution is not requested).')

        if declared_strategy.get("is_new"):
            parts.append(
                "2. The author declared a NEW strategy named "
                f"`{declared_strategy.get('name')}`. Write its lesson card, generalised beyond this "
                "one program:\n"
                '   - "summary": one or two sentences: what the strategy does;\n'
                '   - "mechanism": why it is expected to help;\n'
                '   - "conditions": list of 1-3 short statements on when it applies / does not apply, '
                "informed by this outcome."
            )
        elif declared_strategy.get("id"):
            parts.append(
                "2. The author used the existing strategy "
                f"{declared_strategy['id']} (`{declared_strategy.get('name')}`). If this outcome teaches a "
                'NEW condition on when the strategy applies or fails, give it as "new_condition" '
                "(one short sentence); otherwise null."
            )
        else:
            parts.append('2. Return "lesson": null and "new_condition": null.')

        parts.append("")
        parts.append("Return exactly this JSON shape:")
        parts.append(
            "```json\n"
            "{\n"
            '  "units": {"U1": "lineage", "U2": "ext:S3"},\n'
            '  "lesson": {"summary": "...", "mechanism": "...", "conditions": ["..."]},\n'
            '  "new_condition": null\n'
            "}\n"
            "```"
        )
        return "\n".join(parts)

    async def interpret(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call the interpreter; ``None`` on any failure (caller falls back to q=1)."""
        try:
            response = await self.llms.generate(
                INTERPRETER_SYSTEM,
                [{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.warning(f"Edit Interpreter call failed: {e}")
            return None
        data = extract_json(response.text or "")
        if not isinstance(data, dict):
            logger.warning("Edit Interpreter returned no valid JSON object")
            return None
        return data


META_SYSTEM = (
    "You are a research strategist for an evolutionary program-search system. "
    "The search on one lineage has stagnated. You propose a small set of distinct, "
    "concrete, strategy-level ideas that could unlock further progress. "
    "Respond with a single JSON array and nothing else."
)


class MetaGuide:
    """Generates strategy-level ideas when an island stagnates."""

    def __init__(
        self,
        llm_pool: LLMPool,
        task_description: str,
        evaluator_code: str = "",
        num_ideas: int = 3,
        max_tokens: int = 6000,
    ):
        self.llms = llm_pool
        self.task_description = task_description or ""
        self.evaluator_code = evaluator_code or ""
        self.num_ideas = num_ideas
        self.max_tokens = max_tokens

    def build_prompt(
        self,
        parent_code: str,
        parent_score: float,
        path_trace: str,
        attempt_log: str,
        strategy_names: List[str],
        language: str = "python",
    ) -> str:
        parts = ["# Task", _clip(self.task_description, 8000), ""]
        if self.evaluator_code:
            parts += ["# Evaluator (how programs are scored)", "```", _clip(self.evaluator_code, 6000), "```", ""]
        parts += [
            f"# Selected program (score {parent_score:.6f})",
            f"```{language}",
            _clip(parent_code, 20000),
            "```",
            "",
            "# Lineage: how this program was reached",
            path_trace or "(no edits yet)",
            "",
            "# Already tried on this program",
            attempt_log or "(nothing yet)",
            "",
            "# Strategies already in the shared memory (do NOT repeat these)",
            ", ".join(strategy_names) if strategy_names else "(none)",
            "",
            "# Your task",
            f"Propose {self.num_ideas} distinct strategy-level ideas for improving the score of this "
            "program. Each idea must be implementable in a single edit, materially different from "
            "the strategies listed above and from what was already tried, and specific to this task.",
            "",
            "Return a JSON array; each element:",
            "```json",
            '{"name": "short_snake_case_name", "summary": "what the strategy does (1-2 sentences)", '
            '"mechanism": "why it is expected to help", "conditions": ["when it applies / does not apply"]}',
            "```",
        ]
        return "\n".join(parts)

    async def generate(self, prompt: str) -> List[Dict[str, Any]]:
        try:
            response = await self.llms.generate(
                META_SYSTEM,
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.warning(f"Meta-guidance call failed: {e}")
            return []
        data = extract_json(response.text or "")
        if isinstance(data, dict):
            data = data.get("ideas") or data.get("strategies") or [data]
        if not isinstance(data, list):
            logger.warning("Meta-guidance returned no valid JSON array")
            return []
        ideas: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = snake_case(str(item.get("name") or ""))
            summary = str(item.get("summary") or "").strip()
            if not name or not summary:
                continue
            conditions = item.get("conditions") or []
            if isinstance(conditions, str):
                conditions = [conditions]
            ideas.append(
                {
                    "name": name,
                    "summary": summary[:600],
                    "mechanism": str(item.get("mechanism") or "").strip()[:600],
                    "conditions": [str(c).strip()[:300] for c in conditions if str(c).strip()][:3],
                }
            )
        return ideas[: self.num_ideas]
