"""Execute a pipeline against a dataset and record the run.

A run record is the unit of evidence: enough to say what was measured (dataset
fingerprint and split), how (the full pipeline config and its hash), with which
code (git SHA, whether the tree was dirty, interpreter and library versions) and
what came out (mean and per-query metrics, latency). Records are written once
and never overwritten; comparing pipelines means comparing records.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from retrieval_lab.bm25 import BM25Index
from retrieval_lab.config import PipelineConfig
from retrieval_lab.datasets import Dataset
from retrieval_lab.metrics import evaluate

SCHEMA_VERSION = 1
CUTOFFS = (10, 100)


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetInfo(_Record):
    name: str
    split: str
    documents: int
    queries: int
    judgements: int
    fingerprint: dict[str, str]


class CodeInfo(_Record):
    # SHA-256 over this package's source files. Unlike the git SHA it identifies the
    # code even when the record is committed together with the change that made it.
    source_digest: str
    git_sha: str | None
    git_dirty: bool | None
    package_version: str
    python: str
    dependencies: dict[str, str]


class Timing(_Record):
    index_seconds: float
    query_ms_p50: float
    query_ms_p95: float
    query_ms_max: float


class RunRecord(_Record):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    created_at: datetime
    dataset: DatasetInfo
    pipeline: PipelineConfig
    config_hash: str
    code: CodeInfo
    index: dict[str, int]
    metrics: dict[str, float]
    timing: Timing
    per_query: dict[str, dict[str, float]]

    def write(self, runs_dir: Path) -> Path:
        directory = runs_dir / self.dataset.name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        # "x": a record is evidence; never silently replace one.
        with path.open("x", encoding="utf-8") as handle:
            handle.write(self.model_dump_json(indent=2) + "\n")
        return path

    @classmethod
    def read(cls, path: Path) -> RunRecord:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def execute(
    dataset: Dataset, pipeline: PipelineConfig, *, now: datetime | None = None
) -> RunRecord:
    started = time.perf_counter()
    index = BM25Index(dataset.corpus.values(), pipeline.retriever)
    index_seconds = time.perf_counter() - started

    rankings: dict[str, list[str]] = {}
    latencies_ms: list[float] = []
    for query_id, text in dataset.queries.items():
        started = time.perf_counter()
        ranking = index.search(text, pipeline.top_k)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        rankings[query_id] = [doc_id for doc_id, _ in ranking]

    means, per_query = evaluate(rankings, dataset.qrels, CUTOFFS)
    created_at = (now or datetime.now(UTC)).replace(microsecond=0)
    config_hash = pipeline.config_hash()
    return RunRecord(
        run_id=f"{created_at:%Y%m%dT%H%M%SZ}-{pipeline.name}-{config_hash[:8]}",
        created_at=created_at,
        dataset=DatasetInfo(
            name=dataset.name,
            split=dataset.split,
            documents=len(dataset.corpus),
            queries=len(dataset.queries),
            judgements=sum(len(judged) for judged in dataset.qrels.values()),
            fingerprint=dict(dataset.fingerprint),
        ),
        pipeline=pipeline,
        config_hash=config_hash,
        code=code_info(),
        index={"documents": len(index), "vocabulary": index.vocabulary_size},
        metrics={name: round(value, 6) for name, value in means.items()},
        timing=Timing(
            index_seconds=round(index_seconds, 3),
            query_ms_p50=round(_percentile(latencies_ms, 50), 3),
            query_ms_p95=round(_percentile(latencies_ms, 95), 3),
            query_ms_max=round(max(latencies_ms), 3),
        ),
        per_query={
            query_id: {name: round(value, 6) for name, value in scores.items()}
            for query_id, scores in per_query.items()
        },
    )


def code_info() -> CodeInfo:
    repo = Path(__file__).resolve().parents[2]
    sha = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return CodeInfo(
        source_digest=source_digest(),
        git_sha=sha,
        git_dirty=None if sha is None else bool(status),
        package_version=version("retrieval-evaluation-lab"),
        python=platform.python_version(),
        dependencies={name: version(name) for name in ("pydantic", "PyStemmer")},
    )


def source_digest() -> str:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for file in sorted(package.rglob("*.py")):
        digest.update(file.relative_to(package).as_posix().encode() + b"\0")
        digest.update(file.read_bytes() + b"\0")
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _percentile(values: list[float], pct: int) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    rank = max(1, -(-pct * len(ordered) // 100))
    return ordered[rank - 1]
