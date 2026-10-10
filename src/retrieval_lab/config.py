"""Pipeline configuration: the part of a run that is a choice, not an input.

Configs are frozen and hashed canonically, so two runs with the same hash ran
the same pipeline, and a config file can be checked in next to its results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

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


class PipelineConfig(_Frozen):
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    retriever: BM25Config
    # Depth of the ranking that is retrieved and evaluated.
    top_k: int = Field(default=100, gt=0)

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def from_file(cls, path: Path) -> PipelineConfig:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
