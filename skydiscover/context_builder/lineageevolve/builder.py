"""Context builder for LineageEvolve (Section 8.6 of the method spec).

Prompt blocks, in order:
  1. task/system message                     (unchanged)
  2. selected parent: code, score, metrics, evaluator feedback (unchanged)
  3. lineage: how this program was reached   (path trace)
  4. already tried on this program           (attempt log)
  5. lessons from other lineages
  6. known dead ends on similar programs
  7. context programs                        (unchanged, labelled C1..Cn)
  8. strategy list
  9. output instructions: intent header, then program or diff

The controller passes the pre-rendered text blocks through ``context``:
``lineage_context`` (blocks 3-6), ``strategy_list`` (block 8) and
``intent_instructions`` (block 9).  Numeric search statistics are never
shown to the mutation model.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

from skydiscover.config import Config
from skydiscover.context_builder.adaevolve import AdaEvolveContextBuilder
from skydiscover.context_builder.utils import TemplateManager
from skydiscover.search.base_database import Program


class LineageEvolveContextBuilder(AdaEvolveContextBuilder):
    def __init__(self, config: Config):
        super().__init__(config)
        default_templates = str(Path(__file__).parent.parent / "default" / "templates")
        adaevolve_templates = str(Path(__file__).parent.parent / "adaevolve" / "templates")
        lineage_templates = str(Path(__file__).parent / "templates")
        self.template_manager = TemplateManager(
            default_templates,
            adaevolve_templates,
            lineage_templates,
            self.context_config.template_dir,
        )

    def build_prompt(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        context = context or {}
        strategy_list = context.get("strategy_list") or ""
        if strategy_list:
            strategy_list = (
                "# Strategy list\n"
                "Existing strategies (id: name), for the `strategy` field of the intent header:\n"
                + strategy_list
            )
        for key in ("lineage_context", "strategy_list", "intent_instructions"):
            kwargs.pop(key, None)
        return super().build_prompt(
            current_program,
            context,
            lineage_context=context.get("lineage_context") or "",
            strategy_list=strategy_list,
            intent_instructions=context.get("intent_instructions") or "",
            **kwargs,
        )

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
