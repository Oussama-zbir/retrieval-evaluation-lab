"""Two-stage retrieval: a first-stage ranking rescored by a cross-encoder.

A cross-encoder scores a (query, document) pair jointly, attending across both
texts, so it judges relevance better than comparing two independent embeddings,
but it cannot index anything: every candidate costs a forward pass at query time.
Hence two stages. The first stage ranks the whole corpus cheaply; the reranker
reorders its top ``depth``. Recall within the first ``depth`` cannot change, only
order, and query latency grows with ``depth``. Both effects land in the run record.

Reranker scores are raw logits. A sigmoid on top would not change the order, but
in float32 it saturates to exactly 1.0 for confident pairs and turns them into ties.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol

from retrieval_lab.config import RerankConfig
from retrieval_lab.datasets import Document

if TYPE_CHECKING:
    from retrieval_lab.retrievers import Retriever

Ranking = list[tuple[str, float]]


class Scorer(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per document, higher is more relevant."""
        ...


class RerankIndex:
    def __init__(
        self,
        first_stage: Retriever,
        documents: Iterable[Document],
        config: RerankConfig,
        scorer: Scorer,
    ) -> None:
        self.config = config
        self._first_stage = first_stage
        self._scorer = scorer
        # Same document text as the first stage: title and body.
        self._texts = {doc.doc_id: f"{doc.title} {doc.text}".strip() for doc in documents}

    def describe(self) -> dict[str, int]:
        return {f"first_stage.{key}": value for key, value in self._first_stage.describe().items()}

    def search(self, query: str, k: int) -> Ranking:
        """Return up to ``k`` (doc id, score) pairs, best first.

        The reranked head carries cross-encoder scores; a tail beyond ``depth``
        carries first-stage scores, which are on another scale and only order the tail.
        """
        candidates = self._first_stage.search(query, max(k, self.config.depth))
        head, tail = candidates[: self.config.depth], candidates[self.config.depth :]
        if not head:
            return []
        scores = self._scorer.score(query, [self._texts[doc_id] for doc_id, _ in head])
        if len(scores) != len(head):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(head)} documents")
        reranked = sorted(
            zip((doc_id for doc_id, _ in head), scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        return (reranked + tail)[:k]


class CrossEncoderScorer:
    """A sentence-transformers cross-encoder, pinned to a revision, on CPU (see ``dense``)."""

    def __init__(self, config: RerankConfig) -> None:
        # Imported here so that a BM25-only install never needs torch.
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            from retrieval_lab.retrievers import MissingDependencyError

            raise MissingDependencyError(
                "reranking needs the optional dependencies: pip install -e '.[dense]'"
            ) from exc
        self._model = CrossEncoder(
            config.model, revision=config.revision, device="cpu", max_length=config.max_length
        )
        self._identity = torch.nn.Identity()
        self._batch_size = config.batch_size

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        logits = self._model.predict(
            [(query, document) for document in documents],
            batch_size=self._batch_size,
            activation_fn=self._identity,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in logits]


def load_scorer(config: RerankConfig) -> Scorer:
    return CrossEncoderScorer(config)
