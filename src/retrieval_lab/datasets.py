"""Public retrieval benchmarks in the BEIR layout: fetch, verify, load.

A BEIR dataset is a directory holding ``corpus.jsonl``, ``queries.jsonl`` and
``qrels/<split>.tsv``. Downloads are pinned to a published MD5 and refused on
mismatch, and every loaded dataset carries a SHA-256 fingerprint of the exact
files it was read from, so a run record can say which bytes it measured.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


@dataclass(frozen=True, slots=True)
class DatasetSource:
    name: str
    url: str
    md5: str


# MD5s as published by the BEIR project alongside each archive.
SOURCES: Mapping[str, DatasetSource] = {
    "scifact": DatasetSource(
        name="scifact",
        url=f"{BEIR_BASE_URL}/scifact.zip",
        md5="5f7d1de60b170fc8027bb7898e2efca1",
    ),
}


class DatasetError(Exception):
    """A dataset could not be fetched, verified or parsed."""


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    split: str
    corpus: Mapping[str, Document]
    # Only queries that have judgements in this split; the rest cannot be scored.
    queries: Mapping[str, str]
    # query id -> doc id -> graded relevance (> 0 means relevant).
    qrels: Mapping[str, Mapping[str, int]]
    # File name -> SHA-256 of the files this dataset was loaded from.
    fingerprint: Mapping[str, str]


def fetch(name: str, data_dir: Path) -> Path:
    """Download and extract a dataset into ``data_dir/<name>``; return that path.

    Idempotent: an existing extracted directory is reused. The archive is checked
    against the pinned MD5 before anything is extracted.
    """
    source = SOURCES.get(name)
    if source is None:
        raise DatasetError(f"unknown dataset {name!r}; known: {sorted(SOURCES)}")
    target = data_dir / name
    if (target / "corpus.jsonl").is_file():
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{name}.zip"
    if not archive.is_file():
        partial = archive.with_suffix(".zip.part")
        with urllib.request.urlopen(source.url, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.rename(archive)

    actual = _digest(archive, "md5")
    if actual != source.md5:
        archive.unlink()
        raise DatasetError(f"{archive.name}: md5 {actual} does not match pinned {source.md5}")
    extract_zip(archive, data_dir)
    if not (target / "corpus.jsonl").is_file():
        raise DatasetError(f"{archive.name} did not contain {name}/corpus.jsonl")
    return target


def extract_zip(archive: Path, destination: Path) -> None:
    """Extract ``archive`` into ``destination``, refusing members that would escape it."""
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            path = PurePosixPath(member)
            if path.is_absolute() or ".." in path.parts:
                raise DatasetError(f"{archive.name}: unsafe member path {member!r}")
        zf.extractall(destination)


def load(path: Path, split: str = "test") -> Dataset:
    """Load a BEIR-layout directory, keeping only queries judged in ``split``."""
    files = {
        "corpus.jsonl": path / "corpus.jsonl",
        "queries.jsonl": path / "queries.jsonl",
        f"qrels/{split}.tsv": path / "qrels" / f"{split}.tsv",
    }
    for relative, file in files.items():
        if not file.is_file():
            raise DatasetError(f"{path.name}: missing {relative}")

    corpus: dict[str, Document] = {}
    for row in _jsonl(files["corpus.jsonl"]):
        doc_id = _str_field(row, "_id", "corpus.jsonl")
        if doc_id in corpus:
            raise DatasetError(f"corpus.jsonl: duplicate document id {doc_id!r}")
        corpus[doc_id] = Document(doc_id, str(row.get("title") or ""), str(row.get("text") or ""))

    all_queries: dict[str, str] = {}
    for row in _jsonl(files["queries.jsonl"]):
        all_queries[_str_field(row, "_id", "queries.jsonl")] = _str_field(
            row, "text", "queries.jsonl"
        )

    qrels = _read_qrels(files[f"qrels/{split}.tsv"])
    for query_id, judged in qrels.items():
        if query_id not in all_queries:
            raise DatasetError(f"qrels/{split}.tsv: query {query_id!r} not in queries.jsonl")
        unknown = [doc_id for doc_id in judged if doc_id not in corpus]
        if unknown:
            raise DatasetError(f"qrels/{split}.tsv: query {query_id!r} judges unknown {unknown}")

    return Dataset(
        name=path.name,
        split=split,
        corpus=corpus,
        queries={query_id: all_queries[query_id] for query_id in sorted(qrels)},
        qrels=qrels,
        fingerprint={relative: _digest(file, "sha256") for relative, file in files.items()},
    )


def _read_qrels(file: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with file.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if line_no == 1 and fields[0] == "query-id":
                continue
            if len(fields) != 3:
                raise DatasetError(f"{file.name}:{line_no}: expected 3 tab-separated fields")
            query_id, doc_id, grade = fields
            try:
                qrels.setdefault(query_id, {})[doc_id] = int(grade)
            except ValueError:
                raise DatasetError(f"{file.name}:{line_no}: non-integer grade {grade!r}") from None
    return qrels


def _jsonl(file: Path) -> Iterator[dict[str, object]]:
    with file.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{file.name}:{line_no}: invalid JSON ({exc.msg})") from None
            if not isinstance(row, dict):
                raise DatasetError(f"{file.name}:{line_no}: expected a JSON object")
            yield row


def _str_field(row: Mapping[str, object], key: str, source: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{source}: row without a string {key!r}")
    return value


def _digest(file: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
