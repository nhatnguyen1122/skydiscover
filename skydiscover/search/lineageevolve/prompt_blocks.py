"""Text blocks of the lineage-aware mutation prompt (Section 8).

Everything here is built deterministically from the search state; no LLM
call is involved.  Numeric search statistics (alpha, beta, p, R) are never
rendered.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from skydiscover.search.lineageevolve.memory import ORIGIN_META, Strategy
from skydiscover.search.lineageevolve.tree import (
    STATUS_ADMITTED,
    STATUS_CRASHED,
    STATUS_DUPLICATE,
    AttemptRecord,
    LineageNode,
    LineageTree,
    numerical_tolerance,
)


def fmt_score(score: Optional[float]) -> str:
    return "n/a" if score is None else f"{score:.6f}"


def _desc(record: AttemptRecord) -> str:
    desc = (record.intent or {}).get("description") or ""
    return desc.strip() or "(no description declared)"


def _strategy_label(record: AttemptRecord) -> str:
    return record.strategy_name or "unspecified"


# ----------------------------------------------------------------------
# 8.1 Path trace
# ----------------------------------------------------------------------


def path_trace(tree: LineageTree, node: LineageNode, recent: int = 5, budget: int = 12) -> str:
    """How ``node`` was reached: incoming edits from the lineage root, oldest first."""
    path = tree.path_from_root(node)
    root = path[0]
    prov = root.provenance or {}
    if prov.get("type") == "import":
        first = (
            f"- Origin: imported from island {prov.get('source_island')} "
            f"(score {fmt_score(root.score)})."
        )
    elif prov.get("type") == "spawn":
        first = (
            f"- Origin: seeded from island {prov.get('source_island')} when this island was "
            f"spawned (score {fmt_score(root.score)})."
        )
    else:
        first = f"- Origin: the initial program (score {fmt_score(root.score)})."

    edits = []  # (record, parent_node, child_node)
    for parent, child in zip(path[:-1], path[1:]):
        record = next((a for a in parent.attempts if a.id == child.incoming_edit), None)
        edits.append((record, parent, child))

    lines = [first]
    if not edits:
        lines.append("- No edits yet: this program is the root of its lineage.")
        return "\n".join(lines)

    n = len(edits)
    keep = [True] * n
    if n > budget:
        for i, (record, _, _) in enumerate(edits):
            is_progress = bool(record and record.y_parent)
            keep[i] = is_progress or i >= n - recent

    omitted = 0
    step = 0
    for i, (record, parent, child) in enumerate(edits):
        step += 1
        if not keep[i]:
            omitted += 1
            continue
        if omitted:
            lines.append(f"- ... ({omitted} non-progress edit(s) omitted) ...")
            omitted = 0
        tag = "progress" if (record and record.y_parent) else "no progress"
        desc = _desc(record) if record else "(edit record unavailable)"
        strat = _strategy_label(record) if record else "unspecified"
        lines.append(
            f"- Edit {step} [{tag}]: {desc} (strategy: {strat}) — "
            f"score {fmt_score(parent.score)} → {fmt_score(child.score)}"
        )
    if omitted:
        lines.append(f"- ... ({omitted} non-progress edit(s) omitted) ...")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 8.2 Attempt log
# ----------------------------------------------------------------------


def outcome_text(record: AttemptRecord, epsilon: float = 0.0) -> str:
    if record.status == STATUS_DUPLICATE:
        return "duplicate (produced a program that already exists; not evaluated)"
    if record.status == STATUS_CRASHED:
        return f"crashed: {record.error or 'evaluation failed'}"
    delta = record.delta or 0.0
    base = record.parent_score if record.parent_score is not None else 0.0
    tol = numerical_tolerance(epsilon, base)
    if delta > tol:
        return f"improved to {fmt_score(record.score)}"
    if delta < -tol:
        return f"regressed to {fmt_score(record.score)}"
    return "no change"


def _outcome_kind(record: AttemptRecord, epsilon: float) -> str:
    text = outcome_text(record, epsilon)
    for kind in ("duplicate", "crashed", "improved", "regressed", "no change"):
        if text.startswith(kind):
            return kind
    return "other"


def attempt_log(node: LineageNode, epsilon: float = 0.0, budget: int = 12) -> str:
    """Everything already tried at ``node``, most recent first."""
    records = list(reversed(node.attempts))
    if not records:
        return ""
    lines = []
    for record in records[:budget]:
        lines.append(
            f"- {_desc(record)} (strategy: {_strategy_label(record)}) → "
            f"{outcome_text(record, epsilon)}"
        )
    rest = records[budget:]
    if rest:
        counts: Dict[str, int] = {}
        for record in rest:
            kind = _outcome_kind(record, epsilon)
            counts[kind] = counts.get(kind, 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        lines.append(f"- ... and {len(rest)} earlier attempt(s): {summary}.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 8.4 Lessons and warnings, strategy list
# ----------------------------------------------------------------------


def _strategy_card(strategy: Strategy) -> List[str]:
    kind = " (idea, not yet tried)" if strategy.origin == ORIGIN_META and not strategy.evidence else ""
    lines = [f"### {strategy.id}: {strategy.name}{kind}"]
    if strategy.summary:
        lines.append(f"- What: {strategy.summary}")
    if strategy.mechanism:
        lines.append(f"- Why it may help: {strategy.mechanism}")
    if strategy.conditions:
        lines.append("- Conditions: " + " | ".join(strategy.conditions[-4:]))
    return lines


def lessons_block(lessons: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in lessons:
        lines.extend(_strategy_card(item["strategy"]))
    return "\n".join(lines)


def warnings_block(warnings: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in warnings:
        s = item["strategy"]
        lines.append(f"### {s.id}: {s.name} — failed on similar programs")
        if s.summary:
            lines.append(f"- What: {s.summary}")
        if s.conditions:
            lines.append("- Conditions: " + " | ".join(s.conditions[-4:]))
    return "\n".join(lines)


def strategy_list_block(strategies: List[Strategy]) -> str:
    return "\n".join(f"- {s.id}: {s.name}" for s in strategies)


def lineage_context(
    path: str,
    attempts: str,
    lessons: str,
    warnings: str,
    show_lineage: bool = True,
    show_memory: bool = True,
) -> str:
    """Blocks 3-6 of Section 8.6, in order."""
    sections: List[str] = []
    if show_lineage:
        sections.append("# Lineage: how this program was reached\n" + (path or "(no edits yet)"))
        if attempts:
            sections.append(
                "# Already tried on this program\n"
                "These edits were already attempted on exactly this program. Do NOT repeat an "
                "approach that crashed, was a duplicate, or did not improve the score; try "
                "something materially different.\n" + attempts
            )
        else:
            sections.append(
                "# Already tried on this program\nNothing has been tried on this program yet."
            )
    if show_memory:
        if lessons:
            sections.append(
                "# Lessons from other lineages\n"
                "Strategies learned elsewhere in the search that may transfer to this program. "
                "They are suggestions to consider, not mandatory instructions.\n" + lessons
            )
        if warnings:
            sections.append("# Known dead ends on similar programs\n" + warnings)
    return "\n\n".join(sections)


def intent_instructions(use_memory: bool = True, has_strategies: bool = True) -> str:
    """Instruction for the machine-readable intent header (Section 7.1)."""
    if use_memory:
        if has_strategies:
            strategy_line = (
                "strategy: <the id of ONE strategy from the Strategy list that best names your "
                "approach, e.g. S3 — or `new: short_snake_case_name` if none fits>"
            )
        else:
            strategy_line = "strategy: new: <short_snake_case_name for your approach>"
    else:
        strategy_line = "strategy: unspecified"
    return (
        "FIRST, before any code or diff, output this intent header exactly once "
        "(it is recorded before the score is known):\n\n"
        "<intent>\n"
        f"{strategy_line}\n"
        "sources: [<ids of the lessons/ideas (S#) and context programs (C#) your edit draws on; "
        "[] if none>]\n"
        "description: <one sentence describing the change>\n"
        "</intent>"
    )
