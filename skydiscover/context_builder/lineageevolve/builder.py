"""Context builder for LineageEvolve.

Uses AdaEvolve's prompt templates unchanged.  Only the content of the
``{search_guidance}`` section (rendered right after "# Current Solution")
differs: AdaEvolve puts evaluator feedback, paradigm guidance, sibling
context and the retry error there; LineageEvolve puts evaluator feedback,
AdaEvolve's paradigm guidance (same text), the lineage context (path trace,
attempt log, lessons, warnings), the strategy list, the intent-header
instruction and the retry error.

Context programs are labelled ``C1..Cn`` instead of ``Program 1..n`` so the
intent header can cite them as sources.  Numeric search statistics are
never shown to the mutation model.
"""

from typing import Any, Dict, List, Union

from skydiscover.context_builder.adaevolve import AdaEvolveContextBuilder
from skydiscover.search.base_database import Program


class LineageEvolveContextBuilder(AdaEvolveContextBuilder):
    def _build_search_guidance(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any],
    ) -> str:
        parent_program = (
            list(current_program.values())[0]
            if isinstance(current_program, dict)
            else current_program
        )
        sections: List[str] = []

        # 1. Evaluator feedback from parent artifacts (as AdaEvolve)
        feedback_section = self._format_evaluator_feedback(parent_program)
        if feedback_section:
            sections.append(feedback_section)

        # 2. Paradigm breakthrough guidance (exactly as AdaEvolve)
        paradigm = context.get("paradigm")
        if paradigm:
            section = self._format_paradigm_guidance(paradigm, self.config.language or "python")
            sid = context.get("paradigm_strategy_id")
            if sid:
                section += (
                    f"\n\nIn the intent header, declare this breakthrough idea as strategy {sid} "
                    f"and list {sid} in `sources`."
                )
            sections.append(section)

        # 3. Lineage context, lessons and warnings
        lineage_context = context.get("lineage_context")
        if lineage_context:
            sections.append(lineage_context)

        # 4. Strategy list and intent header instruction
        strategy_list = context.get("strategy_list")
        if strategy_list:
            sections.append(
                "# Strategy list\n"
                "Existing strategies (id: name), for the `strategy` field of the intent header:\n"
                + strategy_list
            )
        intent_instructions = context.get("intent_instructions")
        if intent_instructions:
            sections.append("# Required output header\n" + intent_instructions)

        # 5. Error retry context (as AdaEvolve)
        error_context = context.get("error_context")
        if error_context:
            sections.append(self._format_error_context(error_context))

        return "\n\n".join(sections)

    def _format_single_context_program(
        self, program: Program, index: int, language: str, lines: list
    ) -> None:
        """Label context programs C1..Cn so the intent header can cite them."""
        start = len(lines)
        super()._format_single_context_program(program, index, language, lines)
        if len(lines) > start and lines[start].startswith(f"### Program {index}"):
            lines[start] = lines[start].replace(
                f"### Program {index}", f"### Context program C{index}", 1
            )
