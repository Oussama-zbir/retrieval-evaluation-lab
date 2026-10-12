from __future__ import annotations

import itertools
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

import pytest

from retrieval_lab import datasets
from retrieval_lab.cli import main
from retrieval_lab.compare import (
    ComparisonError,
    bootstrap_interval,
    compare,
    holm,
    randomisation_test,
)
from retrieval_lab.config import BM25Config, PipelineConfig
from retrieval_lab.runs import RunRecord, execute
from tests.conftest import write_beir

NOW = datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)


def exact_p_value(differences: list[float]) -> float:
    """Two-sided sign-flip p-value by enumerating all 2^n flips."""
    observed = abs(sum(differences))
    flips = list(itertools.product((1, -1), repeat=len(differences)))
    extreme = sum(
        abs(sum(s * d for s, d in zip(signs, differences, strict=True))) >= observed - 1e-9
        for signs in flips
    )
    return extreme / len(flips)


def record(beir_dir: Path, name: str = "bm25", k1: float = 1.2) -> RunRecord:
    config = PipelineConfig(name=name, retriever=BM25Config(k1=k1))
    return execute(datasets.load(beir_dir), config, now=NOW)


def with_scores(base: RunRecord, run_id: str, scores: dict[str, float]) -> RunRecord:
    per_query = {q: {"ndcg@10": s} for q, s in scores.items()}
    return base.model_copy(
        update={
            "run_id": run_id,
            "per_query": per_query,
            "metrics": {"ndcg@10": fmean(scores.values())},
        }
    )


@pytest.mark.parametrize(
    "differences",
    [
        [0.3, 0.2, 0.25, 0.1, 0.4, 0.15, 0.05, 0.35],  # consistent gain
        [0.3, -0.2, 0.1, -0.1, 0.05, -0.25, 0.2, 0.0, 0.15, -0.05],  # mixed
        [0.5, -0.5, 0.5, -0.5],  # mean exactly zero
    ],
)
def test_monte_carlo_p_value_matches_exact_enumeration(differences: list[float]) -> None:
    exact = exact_p_value([d for d in differences if d != 0.0])
    estimate = randomisation_test(differences, trials=20_000, seed=1)
    # Binomial standard error at 20k trials is at most 0.0036.
    assert estimate == pytest.approx(exact, abs=0.015)


def test_p_value_is_never_zero_and_is_one_without_differences() -> None:
    assert randomisation_test([1.0] * 40, trials=500, seed=0) == pytest.approx(1 / 501)
    assert randomisation_test([0.0, 0.0, 0.0], trials=100, seed=0) == 1.0
    # One nonzero difference: both sign flips are as extreme as the observation.
    assert randomisation_test([0.0, 0.2, 0.0], trials=100, seed=0) == 1.0


def test_tests_are_deterministic_for_a_seed() -> None:
    differences = [0.1, -0.05, 0.2, 0.0, -0.1, 0.15]
    assert randomisation_test(differences, trials=1_000, seed=7) == randomisation_test(
        differences, trials=1_000, seed=7
    )
    assert bootstrap_interval(
        differences, resamples=1_000, seed=7, confidence=0.9
    ) == bootstrap_interval(differences, resamples=1_000, seed=7, confidence=0.9)


def test_bootstrap_interval_brackets_the_mean() -> None:
    differences = [0.02 * (i % 7) - 0.03 for i in range(200)]
    low, high = bootstrap_interval(differences, resamples=2_000, seed=0, confidence=0.95)
    assert low < fmean(differences) < high
    narrow = bootstrap_interval(differences, resamples=2_000, seed=0, confidence=0.5)
    assert low < narrow[0] < narrow[1] < high


def test_holm_matches_hand_computed_values() -> None:
    # Sorted: 0.01*4=0.04, 0.02*3=0.06, 0.03*2=0.06 (monotone), 0.5*1=0.5.
    assert holm([0.03, 0.01, 0.5, 0.02]) == pytest.approx([0.06, 0.04, 0.5, 0.06])
    assert holm([0.6, 0.9]) == [1.0, 1.0]


def test_compare_reports_paired_deltas(beir_dir: Path) -> None:
    base = record(beir_dir)
    baseline = with_scores(base, "base", {"q1": 0.5, "q2": 0.4})
    candidate = with_scores(base, "cand", {"q1": 0.7, "q2": 0.4})

    [row] = compare(baseline, [candidate], ["ndcg@10"], trials=200)

    assert row.delta == pytest.approx(0.1)
    assert (row.wins, row.ties, row.losses) == (1, 1, 0)
    assert row.ci_low <= row.delta <= row.ci_high
    assert row.p_value == row.p_holm == 1.0  # one changed query cannot be significant


def test_compare_refuses_records_from_different_data(beir_dir: Path, tmp_path: Path) -> None:
    baseline = record(beir_dir)
    # Same dataset name and queries, one judgement changed.
    other_qrels = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t1\nq2\td3\t1\n"
    other = record(write_beir(tmp_path / "copy" / "toy", other_qrels))
    with pytest.raises(ComparisonError, match="fingerprint"):
        compare(baseline, [other], ["ndcg@10"])


def test_compare_refuses_different_queries_and_unknown_metrics(beir_dir: Path) -> None:
    base = record(beir_dir)
    baseline = with_scores(base, "base", {"q1": 0.5, "q2": 0.4})
    fewer = with_scores(base, "fewer", {"q1": 0.5})
    with pytest.raises(ComparisonError, match="different queries"):
        compare(baseline, [fewer], ["ndcg@10"])
    with pytest.raises(ComparisonError, match="no 'map'"):
        compare(baseline, [baseline], ["map"])
    with pytest.raises(ComparisonError, match="candidate"):
        compare(baseline, [], ["ndcg@10"])


def test_cli_compare_prints_a_markdown_table(
    beir_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    baseline = record(beir_dir).write(runs)
    candidate = record(beir_dir, name="bm25-k1-2", k1=2.0).write(runs)

    assert main(["compare", str(baseline), str(candidate), "--trials", "200"]) == 0

    out = capsys.readouterr().out
    assert "| `bm25-k1-2` | ndcg@10 |" in out
    assert "| `bm25-k1-2` | recall@100 |" in out
    assert "Holm-adjusted over 2 tests" in out


def test_cli_compare_reports_incomparable_records(
    beir_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = record(beir_dir).write(tmp_path / "runs")
    assert main(["compare", str(path), str(path), "--metric", "map"]) == 1
    assert "error:" in capsys.readouterr().err
