"""Command line: ``retrieval-lab fetch``, ``run``, ``reproduce`` and ``compare``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from retrieval_lab import compare, datasets
from retrieval_lab.config import PipelineConfig, is_exact
from retrieval_lab.retrievers import MissingDependencyError
from retrieval_lab.runs import RunRecord, execute

# Largest change in a mean metric accepted when reproducing a record that involves
# embeddings (dense, or a hybrid with a dense part). BM25 is integer arithmetic plus
# deterministic float sums and must reproduce exactly;
# embeddings differ in the last bits across CPUs and BLAS builds, which can swap
# near-tied documents. Per-query changes are reported, not failed on.
DENSE_TOLERANCE = 1e-3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval-lab")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download and verify a public dataset")
    fetch.add_argument("dataset", choices=sorted(datasets.SOURCES))
    fetch.add_argument("--data-dir", type=Path, default=Path("data"))

    run = commands.add_parser("run", help="evaluate a pipeline and write a run record")
    run.add_argument("--dataset", required=True)
    run.add_argument("--config", type=Path, required=True, help="pipeline config JSON")
    run.add_argument("--split", default="test")
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))

    reproduce = commands.add_parser(
        "reproduce", help="re-execute a run record and check its metrics still hold"
    )
    reproduce.add_argument("record", type=Path)
    reproduce.add_argument("--data-dir", type=Path, default=Path("data"))

    comparison = commands.add_parser(
        "compare", help="paired significance test of candidate records against a baseline"
    )
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidates", type=Path, nargs="+")
    comparison.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="metric to compare (repeatable; default: ndcg@10 and recall@100)",
    )
    comparison.add_argument("--trials", type=int, default=10_000)
    comparison.add_argument("--seed", type=int, default=0)
    comparison.add_argument("--confidence", type=float, default=0.95)

    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            path = datasets.fetch(args.dataset, args.data_dir)
            print(f"{args.dataset}: verified and extracted to {path}")
            return 0
        if args.command == "reproduce":
            return _reproduce(args)
        if args.command == "compare":
            return _compare(args)
        return _run(args)
    except (
        datasets.DatasetError,
        ValidationError,
        OSError,
        compare.ComparisonError,
        MissingDependencyError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    pipeline = PipelineConfig.from_file(args.config)
    dataset = datasets.load(args.data_dir / args.dataset, args.split)
    record = execute(dataset, pipeline)
    path = record.write(args.runs_dir)

    if record.code.git_dirty:
        print("warning: working tree has uncommitted changes; see code.source_digest")
    print(f"{record.run_id}  ({record.dataset.queries} queries, {record.dataset.documents} docs)")
    for name, value in record.metrics.items():
        print(f"  {name:<11}{value:.4f}")
    print(
        f"  index {record.timing.index_seconds:.2f}s, "
        f"query p50 {record.timing.query_ms_p50:.2f} ms / p95 {record.timing.query_ms_p95:.2f} ms"
    )
    print(f"written to {path}")
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    recorded = RunRecord.read(args.record)
    dataset = datasets.load(args.data_dir / recorded.dataset.name, recorded.dataset.split)
    if dict(dataset.fingerprint) != recorded.dataset.fingerprint:
        print("error: dataset files differ from the ones the record was measured on")
        return 1
    rerun = execute(dataset, recorded.pipeline)
    exact = is_exact(recorded.pipeline.retriever)
    tolerance = 0.0 if exact else DENSE_TOLERANCE
    drift = {
        name: (value, rerun.metrics.get(name))
        for name, value in recorded.metrics.items()
        if (now := rerun.metrics.get(name)) is None or abs(now - value) > tolerance
    }
    changed = sum(
        rerun.per_query.get(query_id) != scores for query_id, scores in recorded.per_query.items()
    )
    if exact and rerun.per_query != recorded.per_query:
        drift.setdefault("per_query", (0.0, None))
    if drift:
        for name, (expected, actual) in drift.items():
            print(f"  {name}: recorded {expected}, now {actual}")
        print(f"{recorded.run_id}: NOT reproduced")
        return 1
    queries = len(recorded.per_query)
    if exact:
        print(f"{recorded.run_id}: reproduced ({queries} queries, metrics identical)")
    else:
        largest = max(abs(rerun.metrics[name] - value) for name, value in recorded.metrics.items())
        print(
            f"{recorded.run_id}: reproduced within {tolerance:g} "
            f"(largest mean change {largest:.6f}; {changed} of {queries} queries changed)"
        )
    return 0


def _compare(args: argparse.Namespace) -> int:
    baseline = RunRecord.read(args.baseline)
    candidates = [RunRecord.read(path) for path in args.candidates]
    metrics = args.metrics or ["ndcg@10", "recall@100"]
    options = {"trials": args.trials, "seed": args.seed, "confidence": args.confidence}
    rows = compare.compare(baseline, candidates, metrics, **options)
    print(compare.render_markdown(baseline, candidates, rows, **options), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
