"""Pipeline configuration: the part of a run that is a choice, not an input.

Configs are frozen and hashed canonically, so two runs with the same hash ran
the same pipeline, and a config file can be checked in next to its results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnalyzerConfig(_Frozen):
    """How text becomes terms. Applied identically to documents and queries."""

    lowercase: bool = True
    # Lucene's default English stop set (33 words), as used by Elasticsearch's
    # `english` analyzer that BEIR's published BM25 numbers were produced with.
    stopwords: Literal["none", "lucene-english"] = "lucene-english"
    stemmer: Literal["none", "porter"] = "porter"


class BM25Config(_Frozen):
    kind: Literal["bm25"] = "bm25"
    # Lucene / Elasticsearch defaults.
    k1: float = Field(default=1.2, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    analyzer: AnalyzerConfig = AnalyzerConfig()


class DenseConfig(_Frozen):
    """A bi-encoder: documents and queries embedded once, ranked by cosine similarity."""

    kind: Literal["dense"] = "dense"
    # A Hugging Face model id pinned to a commit, so the weights cannot change under a record.
    model: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    # Instruction prefixes for models trained with them (E5, BGE); empty for the rest.
    query_prefix: str = ""
    document_prefix: str = ""
    # Part of the config because padding within a batch can move the last float bits.
    batch_size: int = Field(default=64, gt=0)


# A retriever that ranks the corpus on its own; the parts a hybrid can fuse.
BaseRetrieverConfig = Annotated[BM25Config | DenseConfig, Field(discriminator="kind")]


class HybridConfig(_Frozen):
    """Reciprocal rank fusion (Cormack et al., 2009) of independent retrievers.

    A document scores sum(1 / (rrf_k + rank)) over the component rankings it appears
    in, rank starting at 1. Only ranks are fused, so BM25 and cosine scores, which
    live on unrelated scales, need no normalisation.
    """

    kind: Literal["hybrid"] = "hybrid"
    retrievers: tuple[BaseRetrieverConfig, ...] = Field(min_length=2)
    # 60 is the value from the paper and the default in Elasticsearch and OpenSearch.
    rrf_k: int = Field(default=60, ge=0)
    # Ranking depth taken from each component. A document below it contributes
    # nothing from that component, so depth bounds what fusion can recover.
    depth: int = Field(default=100, gt=0)


RetrieverConfig = Annotated[BM25Config | DenseConfig | HybridConfig, Field(discriminator="kind")]


def is_exact(config: RetrieverConfig) -> bool:
    """True when the retriever is pure integer and fixed-order float arithmetic.

    Such a pipeline must reproduce bit for bit; anything with embeddings only
    within a tolerance.
    """
    if isinstance(config, HybridConfig):
        return all(is_exact(part) for part in config.retrievers)
    return isinstance(config, BM25Config)


class PipelineConfig(_Frozen):
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    retriever: RetrieverConfig
    # Depth of the ranking that is retrieved and evaluated.
    top_k: int = Field(default=100, gt=0)

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def from_file(cls, path: Path) -> PipelineConfig:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
