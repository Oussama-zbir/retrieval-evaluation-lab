from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from retrieval_lab import datasets, dense, retrievers
from retrieval_lab.cli import main
from retrieval_lab.config import ChunkingConfig, DenseConfig, PipelineConfig
from retrieval_lab.datasets import Document
from retrieval_lab.dense import DenseIndex, Embeddings, chunk
from retrieval_lab.runs import RunRecord, execute

NOW = datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC)
CONFIG = DenseConfig(model="fake/bag-of-words", revision="0" * 40)
VOCABULARY = ("cats", "dogs", "felines", "lions", "bread", "domesticated", "wild")


class BagOfWords:
    """Deterministic stand-in for a sentence model: normalised keyword counts."""

    def __init__(self, noise: float = 0.0) -> None:
        self.calls: list[list[str]] = []
        self._noise = noise

    def encode(self, texts: Sequence[str]) -> Embeddings:
        self.calls.append(list(texts))
        rows = np.array(
            [[text.lower().count(word) for word in VOCABULARY] for text in texts], dtype=np.float32
        ).reshape(len(texts), len(VOCABULARY))
        rows += self._noise
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return (rows / np.where(norms == 0, 1, norms)).astype(np.float32)


def docs(*texts: str) -> list[Document]:
    return [Document(doc_id=f"d{i}", title="", text=text) for i, text in enumerate(texts)]


def test_ranks_by_cosine_similarity() -> None:
    index = DenseIndex(docs("bread", "cats cats", "cats dogs"), CONFIG, BagOfWords())
    ranking = index.search("cats", 3)
    assert [doc_id for doc_id, _ in ranking] == ["d1", "d2", "d0"]
    assert ranking[0][1] == pytest.approx(1.0)
    assert ranking[1][1] == pytest.approx(2**-0.5)
    assert index.describe() == {"documents": 3, "dimensions": len(VOCABULARY)}


def test_ties_at_the_cutoff_are_broken_by_doc_id() -> None:
    # Ten identical documents: whichever two argpartition picks, ids decide.
    documents = [Document(doc_id=f"d{i:02}", title="", text="cats") for i in reversed(range(10))]
    index = DenseIndex(documents, CONFIG, BagOfWords())
    assert [doc_id for doc_id, _ in index.search("cats", 2)] == ["d00", "d01"]


def test_k_larger_than_the_corpus_returns_every_document() -> None:
    index = DenseIndex(docs("cats", "dogs"), CONFIG, BagOfWords())
    assert len(index.search("cats", 100)) == 2
    assert DenseIndex([], CONFIG, BagOfWords()).search("cats", 10) == []


def test_prefixes_are_applied_to_their_side_only() -> None:
    encoder = BagOfWords()
    config = CONFIG.model_copy(update={"query_prefix": "query: ", "document_prefix": "passage: "})
    index = DenseIndex([Document(doc_id="d", title="Cats", text="purr")], config, encoder)
    index.search("dogs", 1)
    assert encoder.calls == [["passage: Cats purr"], ["query: dogs"]]


def test_encoder_row_count_is_checked() -> None:
    class Short(BagOfWords):
        def encode(self, texts: Sequence[str]) -> Embeddings:
            return super().encode(texts)[:1]

    with pytest.raises(ValueError, match="1 rows for 2 texts"):
        DenseIndex(docs("cats", "dogs"), CONFIG, Short())


def test_revision_must_be_a_commit_sha() -> None:
    with pytest.raises(ValueError, match="revision"):
        DenseConfig(model="m", revision="main")


def test_chunks_cover_every_word_within_the_window_size() -> None:
    words = [f"w{i}" for i in range(10)]
    windows = chunk(" ".join(words), ChunkingConfig(max_words=4, stride=3))
    assert windows == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    # Uneven tail: the last window is pulled back to end at the last word.
    windows = chunk(" ".join(words[:9]), ChunkingConfig(max_words=4, stride=3))
    assert windows == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w5 w6 w7 w8"]


def test_short_and_empty_texts_keep_one_window() -> None:
    config = ChunkingConfig(max_words=4, stride=2)
    assert chunk("a  b\nc", config) == ["a b c"]
    assert chunk("", config) == [""]
    assert chunk("a  b", None) == ["a  b"]


def test_stride_cannot_skip_words() -> None:
    with pytest.raises(ValueError, match="skip words"):
        ChunkingConfig(max_words=4, stride=5)


def test_chunked_document_scores_as_its_best_window() -> None:
    chunked = CONFIG.model_copy(update={"chunking": ChunkingConfig(max_words=2, stride=2)})
    # "cats" only appears past the first window, where a truncating encoder stops.
    # Windows score 0, 0.71 and 1: max gives 1, where a sum or mean would not.
    late = Document(doc_id="late", title="Pets", text="bread bread cats dogs cats cats")
    mixed = Document(doc_id="mixed", title="", text="cats dogs")
    encoder = BagOfWords()
    index = DenseIndex([late, mixed], chunked, encoder)
    assert encoder.calls[0] == ["Pets bread bread", "Pets cats dogs", "Pets cats cats", "cats dogs"]
    ranking = index.search("cats", 2)
    assert [doc_id for doc_id, _ in ranking] == ["late", "mixed"]
    assert [score for _, score in ranking] == pytest.approx([1.0, 2**-0.5])
    assert index.describe() == {"documents": 2, "dimensions": len(VOCABULARY), "chunks": 4}


def test_unchunked_config_hashes_are_unchanged() -> None:
    # Records written before chunking existed keep their run ids and CI globs.
    root = Path(__file__).resolve().parents[1]
    records = sorted(root.glob("runs/*/*.json"))
    assert records
    for path in records:
        record = RunRecord.read(path)
        assert record.pipeline.config_hash() == record.config_hash, path.name


def install(monkeypatch: pytest.MonkeyPatch, encoder: BagOfWords) -> None:
    monkeypatch.setattr(dense, "load_encoder", lambda config: encoder)


def test_dense_run_record(beir_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, BagOfWords())
    pipeline = PipelineConfig(name="dense", retriever=CONFIG)
    record = execute(datasets.load(beir_dir), pipeline, now=NOW)
    assert record.index == {"documents": 4, "dimensions": len(VOCABULARY)}
    assert record.metrics["mrr@10"] == 1.0
    assert record.pipeline.retriever == CONFIG


def test_reproduce_tolerates_float_noise_but_not_metric_drift(
    beir_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch, BagOfWords())
    pipeline = PipelineConfig(name="dense", retriever=CONFIG)
    record = execute(datasets.load(beir_dir), pipeline, now=NOW)
    path = record.write(tmp_path / "runs")
    data = ["--data-dir", str(beir_dir.parent)]

    # Embeddings that differ slightly, as on another CPU, leave the rankings intact.
    install(monkeypatch, BagOfWords(noise=1e-6))
    assert main(["reproduce", str(path), *data]) == 0
    assert "reproduced within 0.001" in capsys.readouterr().out

    drifted = record.model_copy(update={"metrics": {**record.metrics, "ndcg@10": 0.9}})
    path.write_text(drifted.model_dump_json())
    assert main(["reproduce", str(path), *data]) == 1
    assert "NOT reproduced" in capsys.readouterr().out


def test_missing_optional_dependency_is_a_clean_error(
    beir_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable(config: DenseConfig) -> dense.Encoder:
        raise retrievers.MissingDependencyError("dense retrieval needs the optional dependencies")

    monkeypatch.setattr(dense, "load_encoder", unavailable)
    config = Path(__file__).resolve().parents[1] / "configs" / "dense-minilm.json"
    args = ["run", "--dataset", "toy", "--config", str(config), "--data-dir", str(beir_dir.parent)]
    assert main([*args, "--runs-dir", str(tmp_path)]) == 1
    assert "optional dependencies" in capsys.readouterr().err


def test_bm25_records_still_parse() -> None:
    [record] = sorted(Path(__file__).resolve().parents[1].glob("runs/scifact/*-bm25-780acdbb.json"))
    assert RunRecord.read(record).pipeline.retriever.kind == "bm25"


def test_bm25_install_does_not_need_numpy() -> None:
    # numpy is optional: the CLI and BM25 runs must import and work without it.
    script = (
        "import sys; sys.modules['numpy'] = None\n"
        "from retrieval_lab import cli, retrievers\n"
        "from retrieval_lab.config import DenseConfig\n"
        "config = DenseConfig(model='m', revision='0' * 40)\n"
        "try:\n"
        "    retrievers.build(config, [])\n"
        "except retrievers.MissingDependencyError:\n"
        "    print('clean error')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "clean error"
