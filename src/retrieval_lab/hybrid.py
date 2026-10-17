"""Reciprocal rank fusion over independent retrievers.

Each component ranks the corpus to ``depth``; a document's fused score is the sum
of ``1 / (rrf_k + rank)`` over the rankings it appears in. The sum is taken in
exact rational arithmetic, so two documents with the same multiset of ranks tie
exactly, whatever the component order, and the tie goes to the smaller doc id as
everywhere else in the lab.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import TYPE_CHECKING

from retrieval_lab.config import HybridConfig

if TYPE_CHECKING:
    from retrieval_lab.retrievers import Retriever

Ranking = list[tuple[str, float]]


class FusionIndex:
    def __init__(self, components: Sequence[Retriever], config: HybridConfig) -> None:
        if len(components) != len(config.retrievers):
            raise ValueError(
                f"{len(components)} components for {len(config.retrievers)} retriever configs"
            )
        self.config = config
        self._components = list(components)

    def describe(self) -> dict[str, int]:
        """Each component's statistics, keyed by position and kind."""
        return {
            f"{position}.{part.kind}.{key}": value
            for position, (part, component) in enumerate(
                zip(self.config.retrievers, self._components, strict=True)
            )
            for key, value in component.describe().items()
        }

    def search(self, query: str, k: int) -> Ranking:
        """Return up to ``k`` (doc id, fused score) pairs, best first."""
        fused: dict[str, Fraction] = {}
        for component in self._components:
            ranking = component.search(query, self.config.depth)
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                fused[doc_id] = fused.get(doc_id, Fraction(0)) + Fraction(
                    1, self.config.rrf_k + rank
                )
        best = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(doc_id, float(score)) for doc_id, score in best]
