from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from retrieval_lab import datasets
from retrieval_lab.datasets import DatasetError, DatasetSource
from tests.conftest import write_beir


def test_load_keeps_only_judged_queries(beir_dir: Path) -> None:
    dataset = datasets.load(beir_dir)
    assert dataset.name == "toy"
    assert list(dataset.queries) == ["q1", "q2"]
    assert dataset.qrels["q1"] == {"d1": 2, "d2": 1}
    assert len(dataset.corpus) == 4
    assert dataset.corpus["d1"].title == "Cats"


def test_fingerprint_tracks_file_contents(beir_dir: Path) -> None:
    before = datasets.load(beir_dir).fingerprint
    expected = hashlib.sha256((beir_dir / "qrels" / "test.tsv").read_bytes()).hexdigest()
    assert before["qrels/test.tsv"] == expected

    (beir_dir / "qrels" / "test.tsv").write_text("q1\td1\t1\n")
    after = datasets.load(beir_dir).fingerprint
    assert after["qrels/test.tsv"] != before["qrels/test.tsv"]
    assert after["corpus.jsonl"] == before["corpus.jsonl"]


@pytest.mark.parametrize(
    ("qrels", "message"),
    [
        ("q1\tmissing-doc\t1\n", "unknown"),
        ("q9\td1\t1\n", "not in queries"),
        ("q1\td1\n", "3 tab-separated"),
        ("q1\td1\thigh\n", "non-integer"),
    ],
)
def test_load_rejects_inconsistent_qrels(tmp_path: Path, qrels: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        datasets.load(write_beir(tmp_path / "bad", qrels=qrels))


def test_load_rejects_duplicate_ids_and_bad_json(beir_dir: Path) -> None:
    corpus = beir_dir / "corpus.jsonl"
    original = corpus.read_text()
    corpus.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(DatasetError, match="duplicate"):
        datasets.load(beir_dir)
    corpus.write_text(original + "{not json\n")
    with pytest.raises(DatasetError, match="invalid JSON"):
        datasets.load(beir_dir)


def test_load_reports_missing_split(beir_dir: Path) -> None:
    with pytest.raises(DatasetError, match=r"qrels/dev\.tsv"):
        datasets.load(beir_dir, split="dev")


def _zip_dataset(tmp_path: Path) -> Path:
    source = write_beir(tmp_path / "src" / "toy")
    archive = tmp_path / "toy.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in source.rglob("*"):
            zf.write(file, file.relative_to(source.parent).as_posix())
    return archive


def _register(monkeypatch: pytest.MonkeyPatch, archive: Path, md5: str) -> None:
    source = DatasetSource("toy", archive.as_uri(), md5)
    monkeypatch.setattr(datasets, "SOURCES", {"toy": source})


def test_fetch_verifies_checksum_then_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _zip_dataset(tmp_path)
    _register(monkeypatch, archive, hashlib.md5(archive.read_bytes()).hexdigest())

    path = datasets.fetch("toy", tmp_path / "data")
    assert datasets.load(path).qrels["q2"] == {"d3": 1}
    # Second call reuses the extracted copy.
    archive.unlink()
    assert datasets.fetch("toy", tmp_path / "data") == path


def test_fetch_refuses_checksum_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _register(monkeypatch, _zip_dataset(tmp_path), "0" * 32)
    with pytest.raises(DatasetError, match="does not match"):
        datasets.fetch("toy", tmp_path / "data")
    assert not (tmp_path / "data" / "toy").exists()
    assert not (tmp_path / "data" / "toy.zip").exists()


def test_fetch_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="unknown dataset"):
        datasets.fetch("nope", tmp_path)


def test_extract_refuses_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "x")
    with pytest.raises(DatasetError, match="unsafe"):
        datasets.extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()
