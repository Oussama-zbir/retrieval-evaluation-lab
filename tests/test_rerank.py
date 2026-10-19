from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from retrieval_lab import datasets
from retrieval_lab.cli import main
from retrieval_lab.config import BM25Config, HybridConfig, PipelineConfig, RerankConfig, is_exact
from retrieval_lab.datasets import Document
from retrieval_lab.rerank import RerankIndex
from retrieval_lab.runs import execute
from tests.test_hybrid import Fixed

NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=UTC)
BM25 = BM25Config()


def config(depth: int = 100, first_stage: BM25Config | HybridConfig = BM25) -> RerankConfig:
    return RerankConfig(
        first_stage=first_stage, model="fake/overlap", revision="0" * 40, depth=depth
    )


class Overlap:
    """Deterministic stand-in for a cross-encoder: query words found in the document."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        words = query.lower().split()
        return [float(sum(word in document.lower() for word in words)) for document in documents]


def corpus(**texts: str) -> list[Document]:
    return [Document(doc_id=doc_id, title="", text=text) for doc_id, text in texts.items()]


DOCS = corpus(a="bread", b="cats", c="cats and dogs", d="dogs", e="cats")


def test_head_is_reordered_by_the_scorer_and_ties_go_to_the_smaller_id() -> None:
    scorer = Overlap()
    index = RerankIndex(Fixed("a", "e", "b", "c"), DOCS, config(), scorer)
    ranking = index.search("cats dogs", 10)
    assert ranking == [("c", 2.0), ("b", 1.0), ("e", 1.0), ("a", 0.0)]
    # The scorer reads the document text, not the id, and the query unchanged.
    assert scorer.calls == [("cats dogs", ["bread", "cats", "cats", "cats and dogs"])]


def test_only_depth_documents_are_scored_and_the_tail_keeps_first_stage_order() -> None:
    scorer, first_stage = Overlap(), Fixed("a", "b", "c", "d", "e")
    index = RerankIndex(first_stage, DOCS, config(depth=2), scorer)
    ranking = index.search("cats", 4)
    assert [doc_id for doc_id, _ in ranking] == ["b", "a", "c", "d"]
    # The first stage is asked for k when that exceeds depth; only two pairs are scored.
    assert first_stage.depths == [4]
    assert [len(documents) for _, documents in scorer.calls] == [2]


def test_first_stage_is_searched_to_depth_when_k_is_smaller() -> None:
    first_stage = Fixed("a", "b", "c", "d")
    index = RerankIndex(first_stage, DOCS, config(depth=3), Overlap())
    assert index.search("dogs", 1) == [("c", 1.0)]
    assert first_stage.depths == [3]


def test_empty_first_stage_scores_nothing() -> None:
    scorer = Overlap()
    assert RerankIndex(Fixed(), DOCS, config(), scorer).search("cats", 10) == []
    assert scorer.calls == []


def test_scorer_returning_the_wrong_count_is_an_error() -> None:
    class Short(Overlap):
        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            return super().score(query, documents)[1:]

    with pytest.raises(ValueError, match="1 scores for 2 documents"):
        RerankIndex(Fixed("a", "b"), DOCS, config(), Short()).search("cats", 2)


def test_rerankers_do_not_nest_and_are_never_exact() -> None:
    nested = {"kind": "rerank", "first_stage": config().model_dump()}
    with pytest.raises(ValidationError):
        RerankConfig.model_validate({**config().model_dump(), "first_stage": nested})
    # Even over pure BM25: the cross-encoder is floating point on whatever CPU runs it.
    assert is_exact(BM25)
    assert not is_exact(config())


def install(monkeypatch: pytest.MonkeyPatch, scorer: Overlap) -> None:
    monkeypatch.setattr("retrieval_lab.retrievers.load_scorer", lambda config: scorer)


def test_rerank_run_record_and_reproduce(
    beir_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch, Overlap())
    first_stage = HybridConfig(retrievers=(BM25, BM25.model_copy(update={"b": 0.4})))
    pipeline = PipelineConfig(name="rerank", retriever=config(first_stage=first_stage))
    record = execute(datasets.load(beir_dir), pipeline, now=NOW)
    assert record.index["first_stage.0.bm25.documents"] == 4
    assert record.metrics["mrr@10"] == 1.0
    assert record.pipeline == PipelineConfig.model_validate_json(pipeline.model_dump_json())

    path = record.write(tmp_path / "runs")
    assert main(["reproduce", str(path), "--data-dir", str(beir_dir.parent)]) == 0
    assert "reproduced within" in capsys.readouterr().out
