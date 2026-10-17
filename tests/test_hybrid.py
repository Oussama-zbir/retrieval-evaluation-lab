from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from retrieval_lab import datasets
from retrieval_lab.cli import main
from retrieval_lab.config import BM25Config, DenseConfig, HybridConfig, PipelineConfig, is_exact
from retrieval_lab.hybrid import FusionIndex
from retrieval_lab.runs import execute
from tests.test_dense import CONFIG as DENSE
from tests.test_dense import BagOfWords, install

NOW = datetime(2026, 10, 17, 12, 0, 0, tzinfo=UTC)
BM25 = BM25Config()


class Fixed:
    """A component that returns a fixed ranking and remembers the depth asked for."""

    def __init__(self, *doc_ids: str) -> None:
        self._ranking = [(doc_id, 100.0 - rank) for rank, doc_id in enumerate(doc_ids)]
        self.depths: list[int] = []

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        self.depths.append(k)
        return self._ranking[:k]

    def describe(self) -> dict[str, int]:
        return {"documents": len(self._ranking)}


def fusion(*components: Fixed, rrf_k: int = 60, depth: int = 100) -> FusionIndex:
    config = HybridConfig(retrievers=(BM25,) * len(components), rrf_k=rrf_k, depth=depth)
    return FusionIndex(components, config)


def test_scores_are_summed_reciprocal_ranks() -> None:
    index = fusion(Fixed("a", "b", "c"), Fixed("c", "a", "d"))
    ranking = index.search("q", 10)
    assert [doc_id for doc_id, _ in ranking] == ["a", "c", "b", "d"]
    assert ranking[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert ranking[1][1] == pytest.approx(1 / 63 + 1 / 61)
    assert ranking[3][1] == pytest.approx(1 / 63)


def ranked(positions: dict[str, int], length: int = 100) -> Fixed:
    """A component with the given documents at the given ranks and filler elsewhere."""
    at_rank = {rank: doc_id for doc_id, rank in positions.items()}
    return Fixed(*(at_rank.get(rank, f"filler-{rank}") for rank in range(1, length + 1)))


def test_equal_fused_scores_tie_exactly_and_go_to_the_smaller_id() -> None:
    # 1/63 + 1/140 == 1/84 + 1/90, but as float sums the first is one ulp smaller.
    index = fusion(ranked({"a": 3, "b": 24}), ranked({"a": 80, "b": 30}))
    ranking = index.search("q", 200)
    ids = [doc_id for doc_id, _ in ranking]
    assert ids.index("a") == ids.index("b") - 1
    assert ranking[ids.index("a")][1] == ranking[ids.index("b")][1]

    # The same holds for one rank multiset summed in different orders.
    index = fusion(Fixed("z", "y", "x"), Fixed("y", "x", "z"), Fixed("x", "z", "y"), rrf_k=0)
    assert [doc_id for doc_id, _ in index.search("q", 3)] == ["x", "y", "z"]


def test_components_are_searched_to_depth_not_to_k() -> None:
    first, second = Fixed("a", "b", "c"), Fixed("c", "b", "a")
    index = fusion(first, second, depth=2)
    ranking = index.search("q", 1)
    assert first.depths == second.depths == [2]
    # "b" is second in both depth-2 lists; "a" and "c" each appear once, at rank 1.
    [(doc_id, score)] = ranking
    assert doc_id == "b"
    assert score == pytest.approx(2 / 62)


def test_describe_keys_each_component_by_position_and_kind() -> None:
    config = HybridConfig(retrievers=(BM25, DENSE))
    index = FusionIndex([Fixed("a"), Fixed("a", "b")], config)
    assert index.describe() == {"0.bm25.documents": 1, "1.dense.documents": 2}
    with pytest.raises(ValueError, match="1 components for 2"):
        FusionIndex([Fixed("a")], config)


def test_config_needs_two_flat_retrievers() -> None:
    with pytest.raises(ValidationError):
        HybridConfig(retrievers=(BM25,))
    nested = {"kind": "hybrid", "retrievers": [BM25.model_dump(), BM25.model_dump()]}
    with pytest.raises(ValidationError):
        HybridConfig.model_validate({"retrievers": [nested, BM25.model_dump()]})


def test_only_pure_bm25_pipelines_are_exact() -> None:
    assert is_exact(BM25)
    assert not is_exact(DENSE)
    assert is_exact(HybridConfig(retrievers=(BM25, BM25.model_copy(update={"k1": 0.9}))))
    assert not is_exact(HybridConfig(retrievers=(BM25, DENSE)))


def test_hybrid_run_record_fuses_real_indexes(
    beir_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, BagOfWords())
    pipeline = PipelineConfig(name="hybrid", retriever=HybridConfig(retrievers=(BM25, DENSE)))
    record = execute(datasets.load(beir_dir), pipeline, now=NOW)
    assert record.index["0.bm25.documents"] == record.index["1.dense.documents"] == 4
    assert record.metrics["mrr@10"] == 1.0
    assert record.pipeline == PipelineConfig.model_validate_json(pipeline.model_dump_json())


@pytest.mark.parametrize(
    ("second", "expected"),
    [(BM25.model_copy(update={"b": 0.4}), "metrics identical"), (DENSE, "reproduced within")],
)
def test_reproduce_is_exact_only_without_embeddings(
    second: BM25Config | DenseConfig,
    expected: str,
    beir_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch, BagOfWords())
    pipeline = PipelineConfig(name="hybrid", retriever=HybridConfig(retrievers=(BM25, second)))
    path = execute(datasets.load(beir_dir), pipeline, now=NOW).write(tmp_path / "runs")
    assert main(["reproduce", str(path), "--data-dir", str(beir_dir.parent)]) == 0
    assert expected in capsys.readouterr().out
