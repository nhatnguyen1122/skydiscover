"""Strategy Memory: a run-wide store of strategies shared by all islands.

Implements Sections 8.3 and 8.4 of the LineageEvolve specification.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from skydiscover.search.lineageevolve.attribution import jaccard

ORIGIN_EDIT = "edit"
ORIGIN_META = "meta_guide"


@dataclass
class Strategy:
    id: str
    name: str
    summary: str = ""
    mechanism: str = ""
    conditions: List[str] = field(default_factory=list)
    # evidence entries: {"attempt", "program", "share", "y"}
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    origin: str = ORIGIN_EDIT
    created_iteration: int = 0
    island: Optional[int] = None
    times_shown: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Strategy":
        return cls(**data)


class StrategyMemory:
    """Strategies with similarity-weighted Beta utilities."""

    def __init__(self) -> None:
        self.strategies: Dict[str, Strategy] = {}
        self._by_name: Dict[str, str] = {}
        self._next_id = 1

    def __len__(self) -> int:
        return len(self.strategies)

    def get(self, sid: Optional[str]) -> Optional[Strategy]:
        return self.strategies.get(sid) if sid else None

    def find_by_name(self, name: str) -> Optional[Strategy]:
        sid = self._by_name.get(name)
        return self.strategies.get(sid) if sid else None

    def create(
        self,
        name: str,
        summary: str = "",
        mechanism: str = "",
        conditions: Optional[List[str]] = None,
        origin: str = ORIGIN_EDIT,
        iteration: int = 0,
        island: Optional[int] = None,
    ) -> Strategy:
        """Create a new item; if the name already exists, return the existing one."""
        existing = self.find_by_name(name)
        if existing is not None:
            return existing
        sid = f"S{self._next_id}"
        self._next_id += 1
        strategy = Strategy(
            id=sid,
            name=name,
            summary=(summary or "").strip(),
            mechanism=(mechanism or "").strip(),
            conditions=[c.strip() for c in (conditions or []) if c and c.strip()],
            origin=origin,
            created_iteration=iteration,
            island=island,
        )
        self.strategies[sid] = strategy
        self._by_name[name] = sid
        return strategy

    def add_evidence(self, sid: str, attempt_id: str, program_id: str, share: float, y: int) -> None:
        strategy = self.strategies.get(sid)
        if strategy is None or share <= 0.0:
            return
        strategy.evidence.append(
            {"attempt": attempt_id, "program": program_id, "share": float(share), "y": int(y)}
        )

    def add_condition(self, sid: str, condition: str) -> None:
        strategy = self.strategies.get(sid)
        condition = (condition or "").strip()
        if strategy is None or not condition or condition in strategy.conditions:
            return
        strategy.conditions.append(condition[:300])

    # ------------------------------------------------------------------
    # Utility and retrieval (Section 8.4)
    # ------------------------------------------------------------------

    def utility(
        self,
        strategy: Strategy,
        parent_tokens: frozenset,
        tokens_of: Callable[[str], Optional[frozenset]],
    ) -> Tuple[float, float]:
        """Beta posterior ``(a_s(u), b_s(u))`` of a strategy for parent ``u``."""
        a, b = 1.0, 1.0
        for ev in strategy.evidence:
            tokens = tokens_of(ev["program"])
            k = jaccard(parent_tokens, tokens) if tokens is not None else 0.0
            w = k * ev["share"]
            a += w * ev["y"]
            b += w * (1 - ev["y"])
        return a, b

    def retrieve(
        self,
        parent_tokens: frozenset,
        tokens_of: Callable[[str], Optional[frozenset]],
        excluded: Set[str],
        rng: random.Random,
        k_plus: int = 2,
        k_minus: int = 1,
        forced: Optional[str] = None,
        mode: str = "thompson",
        c0: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (lessons to try, warnings), each a list of
        ``{"strategy": Strategy, "a": float, "b": float, "draw": float|None}``.

        ``forced`` is a strategy id that takes one of the ``k_plus`` slots
        (the meta-guidance idea generated in this iteration).

        ``mode="thompson"`` ranks lessons by a Beta draw (Thompson variant,
        Section 8.4); ``mode="ucb"`` ranks them by the posterior mean plus
        ``c0`` posterior standard deviations (PUCT variant, Section 9.4).
        """
        eligible = [s for sid, s in self.strategies.items() if sid not in excluded]
        scored = []
        for s in eligible:
            a, b = self.utility(s, parent_tokens, tokens_of)
            if mode == "ucb":
                mean = a / (a + b)
                draw = mean + c0 * math.sqrt(mean * (1.0 - mean) / (a + b + 1.0))
            else:
                draw = rng.betavariate(a, b)
            scored.append({"strategy": s, "a": a, "b": b, "draw": draw})

        lessons: List[Dict[str, Any]] = []
        if forced is not None and k_plus > 0:
            for item in scored:
                if item["strategy"].id == forced:
                    lessons.append(item)
                    break
            else:
                s = self.strategies.get(forced)
                if s is not None:
                    a, b = self.utility(s, parent_tokens, tokens_of)
                    lessons.append({"strategy": s, "a": a, "b": b, "draw": None})
        chosen = {item["strategy"].id for item in lessons}
        rest = sorted(
            (it for it in scored if it["strategy"].id not in chosen),
            key=lambda it: it["draw"],
            reverse=True,
        )
        lessons.extend(rest[: max(0, k_plus - len(lessons))])
        chosen = {item["strategy"].id for item in lessons}

        warn_pool = [
            it for it in scored if it["strategy"].id not in chosen and it["b"] - 1.0 > 0.0
        ]
        warn_pool.sort(key=lambda it: it["b"], reverse=True)
        warnings = warn_pool[: max(0, k_minus)]
        return lessons, warnings

    def strategy_list(self, cap: int = 30) -> List[Strategy]:
        """Existing strategies for the intent declaration, most-evidenced first."""
        items = sorted(
            self.strategies.values(),
            key=lambda s: (len(s.evidence), -int(s.id[1:])),
            reverse=True,
        )
        return items[:cap]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_id": self._next_id,
            "strategies": [s.to_dict() for s in self.strategies.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyMemory":
        mem = cls()
        for sd in data.get("strategies", []):
            s = Strategy.from_dict(sd)
            mem.strategies[s.id] = s
            mem._by_name[s.name] = s.id
        mem._next_id = data.get("next_id", len(mem.strategies) + 1)
        return mem
