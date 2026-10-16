"""Dense retrieval: a bi-encoder with exact (brute-force) cosine search.

Exact search first, on purpose: an approximate index (HNSW, IVF) trades recall for
latency, and that trade can only be measured against an exact baseline. At BEIR
scale (thousands of documents, 384 dimensions) one matrix-vector product per query
is a few milliseconds anyway.

Embeddings are L2-normalised, so the dot product is the cosine similarity. Ties are
broken by document id, as in BM25, so a ranking is a function of its inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from retrieval_lab.config import DenseConfig
from retrieval_lab.datasets import Document
from retrieval_lab.retrievers import MissingDependencyError

Ranking = list[tuple[str, float]]
Embeddings = NDArray[np.float32]


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Embeddings:
        """Return one L2-normalised row per text."""
        ...


class DenseIndex:
    def __init__(
        self, documents: Iterable[Document], config: DenseConfig, encoder: Encoder
    ) -> None:
        self.config = config
        self._encoder = encoder
        docs = list(documents)
        self._doc_ids = [doc.doc_id for doc in docs]
        # Same document text as BM25: title and body.
        self._matrix = encoder.encode(
            [f"{config.document_prefix}{doc.title} {doc.text}".strip() for doc in docs]
        )
        if self._matrix.shape[0] != len(docs):
            raise ValueError(f"encoder returned {self._matrix.shape[0]} rows for {len(docs)} docs")
        # Rank of each document id in sorted order: the tie-break key.
        self._id_rank = np.argsort(np.argsort(np.array(self._doc_ids, dtype=object)))

    def __len__(self) -> int:
        return len(self._doc_ids)

    def describe(self) -> dict[str, int]:
        return {"documents": len(self), "dimensions": int(self._matrix.shape[1])}

    def search(self, query: str, k: int) -> Ranking:
        """Return up to ``k`` (doc id, score) pairs, best first. Every document scores."""
        k = min(k, len(self._doc_ids))
        if k == 0:
            return []
        [query_vector] = self._encoder.encode([f"{self.config.query_prefix}{query}"])
        scores = self._matrix @ query_vector
        # Every document scoring at least the k-th best score, so ties at the cut-off
        # are resolved by doc id instead of by argpartition's internal order.
        threshold = np.partition(scores, -k)[-k]
        candidates = np.flatnonzero(scores >= threshold)
        order = np.lexsort((self._id_rank[candidates], -scores[candidates]))[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in candidates[order]]


class SentenceTransformerEncoder:
    """A Hugging Face sentence-transformers model, pinned to a revision, on CPU.

    CPU only: GPU and MPS kernels are not bit-stable from run to run, and CI, where
    records are reproduced, has neither.
    """

    def __init__(self, config: DenseConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise MissingDependencyError(
                "dense retrieval needs the optional dependencies: pip install -e '.[dense]'"
            ) from exc
        self._model = SentenceTransformer(config.model, revision=config.revision, device="cpu")
        self._batch_size = config.batch_size

    def encode(self, texts: Sequence[str]) -> Embeddings:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def load_encoder(config: DenseConfig) -> Encoder:
    return SentenceTransformerEncoder(config)
