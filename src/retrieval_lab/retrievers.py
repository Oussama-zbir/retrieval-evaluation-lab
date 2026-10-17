"""One interface over every retriever, so a run does not care which kind it executes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from retrieval_lab.bm25 import BM25Index
from retrieval_lab.config import BM25Config, HybridConfig, RetrieverConfig
from retrieval_lab.datasets import Document
from retrieval_lab.hybrid import FusionIndex


class MissingDependencyError(RuntimeError):
    pass


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...

    def describe(self) -> dict[str, int]:
        """Index statistics stored in the run record."""
        ...


def build(config: RetrieverConfig, documents: Iterable[Document]) -> Retriever:
    if isinstance(config, HybridConfig):
        corpus = list(documents)
        return FusionIndex([build(part, corpus) for part in config.retrievers], config)
    if isinstance(config, BM25Config):
        return BM25Index(documents, config)
    # Imported here so that a BM25-only install never needs numpy.
    try:
        from retrieval_lab import dense
    except ImportError as exc:
        raise MissingDependencyError(
            "dense retrieval needs the optional dependencies: pip install -e '.[dense]'"
        ) from exc
    return dense.DenseIndex(documents, config, dense.load_encoder(config))
