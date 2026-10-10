from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS = [
    {"_id": "d1", "title": "Cats", "text": "Cats are small domesticated felines."},
    {"_id": "d2", "title": "Dogs", "text": "Dogs are loyal domesticated animals."},
    {"_id": "d3", "title": "Felines", "text": "Wild felines include lions and tigers."},
    {"_id": "d4", "title": "", "text": "Bread is baked from flour and water."},
]
QUERIES = [
    {"_id": "q1", "text": "domesticated cats"},
    {"_id": "q2", "text": "wild lions"},
    {"_id": "q3", "text": "an unjudged query"},
]
QRELS = "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t1\nq2\td3\t1\n"


def write_beir(directory: Path, qrels: str = QRELS) -> Path:
    (directory / "qrels").mkdir(parents=True)
    (directory / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in CORPUS))
    (directory / "queries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in QUERIES))
    (directory / "qrels" / "test.tsv").write_text(qrels)
    return directory


@pytest.fixture
def beir_dir(tmp_path: Path) -> Path:
    return write_beir(tmp_path / "toy")
