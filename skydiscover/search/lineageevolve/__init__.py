"""
LineageEvolve - lineage-tree search built on AdaEvolve.

Keeps AdaEvolve's islands (decayed-UCB scheduling, migration, dynamic
spawning) and replaces within-island parent selection by Thompson sampling
on ``R(u) + L(u)`` over a per-island lineage tree, with source attribution,
a shared Strategy Memory and a lineage-aware mutation prompt.

See ``LineageEvolve_method_spec.md`` for the method specification.
"""

from skydiscover.search.lineageevolve.memory import Strategy, StrategyMemory
from skydiscover.search.lineageevolve.tree import AttemptRecord, LineageNode, LineageTree

__all__ = ["AttemptRecord", "LineageNode", "LineageTree", "Strategy", "StrategyMemory"]
