from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from retrieval_lab import datasets
from retrieval_lab.cli import main
from retrieval_lab.config import BM25Config, PipelineConfig
from retrieval_lab.runs import RunRecord, execute

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NOW = datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)


def pipeline(k1: float = 1.2) -> PipelineConfig:
    return PipelineConfig(name="bm25", retriever=BM25Config(k1=k1))


def test_record_describes_the_run(beir_dir: Path) -> None:
    record = execute(datasets.load(beir_dir), pipeline(), now=NOW)

    assert record.run_id == f"20261010T120000Z-bm25-{record.config_hash[:8]}"
    assert record.dataset.queries == 2
    assert record.dataset.judgements == 3
    assert set(record.dataset.fingerprint) == {"corpus.jsonl", "queries.jsonl", "qrels/test.tsv"}
    assert set(record.per_query) == {"q1", "q2"}
    assert set(record.metrics) == {
        f"{name}@{k}" for name in ("ndcg", "recall", "mrr") for k in (10, 100)
    }
    # Both relevant documents for q1 and the one for q2 are retrieved and ranked first.
    assert record.metrics["recall@10"] == 1.0
    assert record.metrics["mrr@10"] == 1.0
    assert len(record.code.source_digest) == 64


def test_metrics_are_deterministic(beir_dir: Path) -> None:
    dataset = datasets.load(beir_dir)
    first, second = execute(dataset, pipeline(), now=NOW), execute(dataset, pipeline(), now=NOW)
    assert first.metrics == second.metrics
    assert first.per_query == second.per_query


def test_config_hash_is_canonical_and_sensitive() -> None:
    assert pipeline().config_hash() == pipeline().config_hash()
    assert pipeline().config_hash() != pipeline(k1=0.9).config_hash()


def test_record_round_trips_and_is_never_overwritten(beir_dir: Path, tmp_path: Path) -> None:
    record = execute(datasets.load(beir_dir), pipeline(), now=NOW)
    path = record.write(tmp_path / "runs")
    assert path == tmp_path / "runs" / "toy" / f"{record.run_id}.json"
    assert RunRecord.read(path) == record
    with pytest.raises(FileExistsError):
        record.write(tmp_path / "runs")


def test_checked_in_configs_are_valid_and_named_after_their_file() -> None:
    files = sorted(CONFIGS.glob("*.json"))
    assert files
    for file in files:
        assert PipelineConfig.from_file(file).name == file.stem


def test_unknown_config_fields_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "typo.json"
    config.write_text(json.dumps({"name": "typo", "retriever": {"kind": "bm25", "k_1": 2.0}}))
    assert main(["run", "--dataset", "toy", "--config", str(config)]) == 1


def test_cli_run_writes_a_record(
    beir_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["run", "--dataset", "toy", "--config", str(CONFIGS / "bm25.json")]
    paths = ["--data-dir", str(beir_dir.parent), "--runs-dir", str(tmp_path / "runs")]
    assert main([*args, *paths]) == 0

    [written] = (tmp_path / "runs" / "toy").glob("*.json")
    record = RunRecord.read(written)
    assert record.pipeline.name == "bm25"
    assert "ndcg@10" in capsys.readouterr().out


def test_cli_reports_missing_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", "--dataset", "absent", "--config", str(CONFIGS / "bm25.json")]
    assert main([*args, "--data-dir", str(tmp_path)]) == 1
    assert "missing corpus.jsonl" in capsys.readouterr().err


def test_reproduce_accepts_an_unchanged_run_and_rejects_drift(
    beir_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = execute(datasets.load(beir_dir), pipeline(), now=NOW)
    path = record.write(tmp_path / "runs")
    data = ["--data-dir", str(beir_dir.parent)]
    assert main(["reproduce", str(path), *data]) == 0

    tampered = record.model_copy(update={"metrics": {**record.metrics, "ndcg@10": 0.5}})
    path.write_text(tampered.model_dump_json())
    assert main(["reproduce", str(path), *data]) == 1
    assert "NOT reproduced" in capsys.readouterr().out


def test_reproduce_refuses_changed_dataset(beir_dir: Path, tmp_path: Path) -> None:
    path = execute(datasets.load(beir_dir), pipeline(), now=NOW).write(tmp_path / "runs")
    (beir_dir / "qrels" / "test.tsv").write_text("q1\td2\t1\n")
    assert main(["reproduce", str(path), "--data-dir", str(beir_dir.parent)]) == 1
