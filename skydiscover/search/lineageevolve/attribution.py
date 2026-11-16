"""Deterministic parts of source attribution and deduplication.

Implements Section 5 (changed-unit inventory, label validation, shares),
Section 7.1 (declared-intent parsing) and Section 10 (normalised hash).
The LLM call of the Edit Interpreter lives in ``interpreter.py``.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from skydiscover.search.lineageevolve.tree import UNSPECIFIED

LABEL_LINEAGE = "lineage"
LABEL_NOVEL = "novel"
EXT_PREFIX = "ext:"

_EVOLVE_START = "EVOLVE-BLOCK-START"
_EVOLVE_END = "EVOLVE-BLOCK-END"

# ----------------------------------------------------------------------
# Normalisation / hashing (Section 10)
# ----------------------------------------------------------------------


def _evolve_region(code: str) -> str:
    """Return the text inside EVOLVE-BLOCK markers if present, else all of it."""
    if _EVOLVE_START not in code:
        return code
    parts: List[str] = []
    inside = False
    for line in code.split("\n"):
        if _EVOLVE_START in line:
            inside = True
            continue
        if _EVOLVE_END in line:
            inside = False
            continue
        if inside:
            parts.append(line)
    return "\n".join(parts) if parts else code


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def normalize_code(code: str) -> str:
    """Blank lines removed, lines trimmed, internal whitespace collapsed."""
    lines = [normalize_line(ln) for ln in _evolve_region(code or "").split("\n")]
    return "\n".join(ln for ln in lines if ln)


def code_hash(code: str) -> str:
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()


_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\sA-Za-z_0-9]")


def token_set(code: str) -> frozenset:
    """Token set of the normalised source (for the Jaccard kernel, Section 8.4)."""
    return frozenset(_TOKEN_RE.findall(normalize_code(code)))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


# ----------------------------------------------------------------------
# Declared intent (Section 7.1)
# ----------------------------------------------------------------------

_INTENT_BLOCK_RE = re.compile(r"<intent>(.*?)</intent>", re.DOTALL | re.IGNORECASE)
_FIELD_RE = {
    "strategy": re.compile(r"^\s*[-*]?\s*\**strategy\**\s*[:=]\s*(.+?)\s*$", re.I | re.M),
    "sources": re.compile(r"^\s*[-*]?\s*\**sources\**\s*[:=]\s*(.*?)\s*$", re.I | re.M),
    "description": re.compile(r"^\s*[-*]?\s*\**description\**\s*[:=]\s*(.+?)\s*$", re.I | re.M),
}
_SOURCE_ID_RE = re.compile(r"\b([SC]\d+)\b")


def snake_case(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    return name[:60]


def parse_intent(response: str) -> Dict[str, Any]:
    """Parse the machine-readable intent header that precedes the code.

    Returns ``{"strategy": ..., "sources": [...], "description": ..., "parsed": bool}``.
    ``strategy`` is an existing strategy id (``"S3"``), ``"new:<name>"`` or
    ``"unspecified"``.  A missing/unparsable header yields the unspecified
    intent; the candidate is still used.
    """
    intent = {"strategy": UNSPECIFIED, "sources": [], "description": "", "parsed": False}
    if not response:
        return intent
    m = _INTENT_BLOCK_RE.search(response)
    if m:
        block = m.group(1)
    else:
        # Tolerate a header without tags, but only before the first code/diff.
        cut = len(response)
        for marker in ("```", "<<<<<<< SEARCH"):
            idx = response.find(marker)
            if idx != -1:
                cut = min(cut, idx)
        block = response[:cut]

    ms = _FIELD_RE["strategy"].search(block)
    if not ms:
        return intent
    raw = ms.group(1).strip().strip("`'\"")
    low = raw.lower()
    if low.startswith("new"):
        name = snake_case(re.sub(r"^new\s*[:=\-]?\s*", "", raw, flags=re.I))
        intent["strategy"] = f"new:{name}" if name else UNSPECIFIED
    else:
        mid = re.match(r"^(S\d+)\b", raw, re.I)
        if mid:
            intent["strategy"] = mid.group(1).upper()
        elif low in ("", "none", UNSPECIFIED, "n/a"):
            intent["strategy"] = UNSPECIFIED
        else:
            # A bare name: resolved against the memory by the database.
            name = snake_case(raw)
            intent["strategy"] = f"name:{name}" if name else UNSPECIFIED

    msrc = _FIELD_RE["sources"].search(block)
    if msrc:
        seen: List[str] = []
        for sid in _SOURCE_ID_RE.findall(msrc.group(1).upper()):
            if sid not in seen:
                seen.append(sid)
        intent["sources"] = seen

    md = _FIELD_RE["description"].search(block)
    if md:
        intent["description"] = md.group(1).strip().strip("`'\"")[:400]

    intent["parsed"] = True
    return intent


# ----------------------------------------------------------------------
# Changed-unit inventory (Section 5.2)
# ----------------------------------------------------------------------

_DEF_RE = re.compile(r"^(\s*)(?:async\s+def|def|class)\s+([A-Za-z_]\w*)")


@dataclass
class ChangedUnit:
    id: str
    symbol: Optional[str]
    removed: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.added) + len(self.removed)

    def excerpt(self, lines: List[str], max_lines: int = 30) -> str:
        if len(lines) <= max_lines:
            return "\n".join(lines)
        head = max_lines // 2
        tail = max_lines - head
        omitted = len(lines) - max_lines
        return "\n".join(lines[:head] + [f"... ({omitted} lines omitted) ..."] + lines[-tail:])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "size": self.size,
            "removed": len(self.removed),
            "added": len(self.added),
        }


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _enclosing_symbol(lines: List[str], idx: int) -> Optional[str]:
    """Qualified name of the innermost def/class enclosing line ``idx`` (Python)."""
    if not lines:
        return None
    idx = max(0, min(idx, len(lines) - 1))
    # Use the first non-blank line at or after idx to define the indentation.
    j = idx
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines):
        j = idx
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j < 0:
            return None
    names: List[str] = []
    m = _DEF_RE.match(lines[j])
    if m:
        cur_indent = len(m.group(1))
        names.append(m.group(2))
    else:
        cur_indent = _indent(lines[j])
    k = j - 1
    while k >= 0 and cur_indent > 0:
        line = lines[k]
        if line.strip():
            m = _DEF_RE.match(line)
            if m and len(m.group(1)) < cur_indent:
                names.append(m.group(2))
                cur_indent = len(m.group(1))
        k -= 1
    if not names:
        return None
    return ".".join(reversed(names))


def build_inventory(parent_code: str, child_code: str, language: str = "python") -> List[ChangedUnit]:
    """Deterministic list of changed units between parent and child.

    A unit is a contiguous diff hunk, merged with other hunks inside the same
    enclosing function/class when that can be determined (Python).  Computed
    identically for diff-mode and full-rewrite outputs because it only looks
    at the two program texts.
    """
    a = (parent_code or "").split("\n")
    b = (child_code or "").split("\n")
    is_python = (language or "python").lower() in ("python", "py")

    groups: List[Tuple[Optional[str], List[str], List[str]]] = []
    by_symbol: Dict[str, int] = {}
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        removed = [ln for ln in a[i1:i2] if ln.strip()]
        added = [ln for ln in b[j1:j2] if ln.strip()]
        if not removed and not added:
            continue  # whitespace-only hunk
        symbol = None
        if is_python:
            symbol = _enclosing_symbol(b, j1) if j2 > j1 else _enclosing_symbol(a, i1)
        if symbol is not None and symbol in by_symbol:
            g = groups[by_symbol[symbol]]
            g[1].extend(removed)
            g[2].extend(added)
        else:
            if symbol is not None:
                by_symbol[symbol] = len(groups)
            groups.append((symbol, removed, added))

    return [
        ChangedUnit(id=f"U{n}", symbol=sym, removed=rem, added=add)
        for n, (sym, rem, add) in enumerate(groups, start=1)
    ]


# ----------------------------------------------------------------------
# Label validation and shares (Sections 5.3, 5.4)
# ----------------------------------------------------------------------


def validate_labels(
    labels: Dict[str, str],
    units: List[ChangedUnit],
    strategy_source_ids: Set[str],
    declared_sources: Set[str],
    context_programs: Dict[str, str],
    parent_code: str,
) -> Optional[Dict[str, str]]:
    """Validate interpreter labels deterministically.

    Returns ``None`` when the interpretation as a whole is invalid (unknown
    unit ids cited, or not a mapping), which makes the caller fall back to
    ``q_e = 1``.  Otherwise returns one label per unit; a label failing its
    validity rule is replaced by ``lineage``, and units the interpreter did
    not label are ``lineage`` as well.
    """
    if not isinstance(labels, dict):
        return None
    unit_ids = {u.id for u in units}
    if any(str(k) not in unit_ids for k in labels):
        return None

    parent_lines = {normalize_line(ln) for ln in (parent_code or "").split("\n") if ln.strip()}
    ctx_lines: Dict[str, Set[str]] = {
        cid: {normalize_line(ln) for ln in (code or "").split("\n") if ln.strip()}
        for cid, code in context_programs.items()
    }

    out: Dict[str, str] = {}
    for unit in units:
        raw = str(labels.get(unit.id, LABEL_LINEAGE)).strip()
        low = raw.lower()
        label = LABEL_LINEAGE
        if low == LABEL_NOVEL:
            label = LABEL_NOVEL
        elif low.startswith(EXT_PREFIX):
            src = raw[len(EXT_PREFIX) :].strip().upper()
            if src in strategy_source_ids:
                if src in declared_sources:
                    label = EXT_PREFIX + src
            elif src in ctx_lines:
                added = [normalize_line(ln) for ln in unit.added]
                if added:
                    hits = sum(
                        1 for ln in added if ln in ctx_lines[src] and ln not in parent_lines
                    )
                    if 2 * hits >= len(added):
                        label = EXT_PREFIX + src
        out[unit.id] = label
    return out


def compute_shares(
    units: List[ChangedUnit], labels: Dict[str, str]
) -> Tuple[float, Dict[str, float]]:
    """Lineage share ``q_e`` and external shares ``a_{e,j}`` (sum to 1)."""
    total = sum(u.size for u in units)
    if total <= 0:
        return 1.0, {}
    lineage = 0
    ext: Dict[str, float] = {}
    for u in units:
        label = labels.get(u.id, LABEL_LINEAGE)
        if label.startswith(EXT_PREFIX):
            src = label[len(EXT_PREFIX) :]
            ext[src] = ext.get(src, 0.0) + u.size
        else:
            lineage += u.size
    return lineage / total, {k: v / total for k, v in ext.items()}
