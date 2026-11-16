"""Process-wide LLM usage accounting (tokens, calls, wall-clock time).

Every LLM backend reports each completed API call to the global
``USAGE`` tracker.  Callers can tag calls with a *purpose* (e.g.
``"mutation"``, ``"interpreter"``, ``"meta_guidance"``) through the
``usage_purpose`` context manager; the tag is stored in a ``ContextVar`` so
it follows the asyncio task that makes the call.

Typical use::

    from skydiscover.llm.usage import USAGE, usage_purpose

    before = USAGE.snapshot()
    with usage_purpose("mutation"):
        await llm.generate(...)
    delta = USAGE.delta(before)   # tokens spent by this block
"""

from __future__ import annotations

import contextvars
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_PURPOSE: contextvars.ContextVar[str] = contextvars.ContextVar("llm_usage_purpose", default="other")

_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")


@contextmanager
def usage_purpose(purpose: str) -> Iterator[None]:
    """Tag all LLM calls made inside the block with ``purpose``."""
    token = _PURPOSE.set(purpose)
    try:
        yield
    finally:
        _PURPOSE.reset(token)


def current_purpose() -> str:
    return _PURPOSE.get()


def _empty() -> Dict[str, float]:
    d: Dict[str, float] = {k: 0 for k in _FIELDS}
    d["llm_seconds"] = 0.0
    return d


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class UsageTracker:
    """Thread-safe accumulator of token usage, split by purpose and by model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._total = _empty()
            self._by_purpose: Dict[str, Dict[str, float]] = {}
            self._by_model: Dict[str, Dict[str, float]] = {}
            self._calls_without_usage = 0
            self._started_at = time.time()

    def record(
        self,
        model: Optional[str],
        usage: Any,
        seconds: float = 0.0,
        purpose: Optional[str] = None,
    ) -> None:
        """Record one completed API call.

        ``usage`` is the provider's usage object (OpenAI ``CompletionUsage``,
        Responses-API usage, or a plain dict); missing fields count as zero.
        """
        purpose = purpose or _PURPOSE.get()
        prompt = _get(usage, "prompt_tokens")
        if prompt is None:
            prompt = _get(usage, "input_tokens")
        completion = _get(usage, "completion_tokens")
        if completion is None:
            completion = _get(usage, "output_tokens")
        total = _get(usage, "total_tokens")
        details = _get(usage, "completion_tokens_details") or _get(usage, "output_tokens_details")
        reasoning = _get(details, "reasoning_tokens")

        prompt = int(prompt or 0)
        completion = int(completion or 0)
        reasoning = int(reasoning or 0)
        total = int(total) if total is not None else prompt + completion

        row = {
            "calls": 1,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "total_tokens": total,
            "llm_seconds": float(seconds or 0.0),
        }
        with self._lock:
            if usage is None:
                self._calls_without_usage += 1
            for bucket in (
                self._total,
                self._by_purpose.setdefault(purpose, _empty()),
                self._by_model.setdefault(model or "unknown", _empty()),
            ):
                for k, v in row.items():
                    bucket[k] += v

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the current totals."""
        with self._lock:
            return {
                "total": dict(self._total),
                "by_purpose": {k: dict(v) for k, v in self._by_purpose.items()},
                "by_model": {k: dict(v) for k, v in self._by_model.items()},
                "calls_without_usage": self._calls_without_usage,
            }

    def delta(self, before: Dict[str, Any]) -> Dict[str, Any]:
        """Usage accumulated since ``before`` (a previous ``snapshot()``)."""
        now = self.snapshot()

        def sub(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
            out = {k: a.get(k, 0) - b.get(k, 0) for k in a}
            out["llm_seconds"] = round(out.get("llm_seconds", 0.0), 3)
            return out

        by_purpose = {}
        for name, row in now["by_purpose"].items():
            d = sub(row, before.get("by_purpose", {}).get(name, {}))
            if d.get("calls"):
                by_purpose[name] = d
        return {"total": sub(now["total"], before.get("total", {})), "by_purpose": by_purpose}


USAGE = UsageTracker()
