from __future__ import annotations

import math
import random

import pytest
import pytrec_eval

from retrieval_lab.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k

JUDGED = {"a": 2, "b": 1, "c": 0}


def test_ndcg_hand_computed() -> None:
    # DCG = 1/log2(2) + 2/log2(3); IDCG = 2/log2(2) + 1/log2(3).
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a", "x"], JUDGED, 10) == pytest.approx(expected)


def test_ndcg_perfect_and_empty() -> None:
    assert ndcg_at_k(["a", "b"], JUDGED, 10) == pytest.approx(1.0)
    assert ndcg_at_k([], JUDGED, 10) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}, 10) == 0.0


def test_ndcg_cutoff_applies_to_ideal_ranking() -> None:
    # At k=1 the ideal is a single grade-2 document.
    assert ndcg_at_k(["b"], JUDGED, 1) == pytest.approx(0.5)


def test_recall_ignores_zero_grades() -> None:
    assert recall_at_k(["c", "a"], JUDGED, 10) == pytest.approx(0.5)
    assert recall_at_k(["c", "a", "b"], JUDGED, 2) == pytest.approx(0.5)


def test_mrr_first_relevant_within_cutoff() -> None:
    assert mrr_at_k(["x", "c", "b"], JUDGED, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "c", "b"], JUDGED, 2) == 0.0


def test_evaluate_counts_missing_rankings_as_zero() -> None:
    means, per_query = evaluate({"q1": ["a"]}, {"q1": {"a": 1}, "q2": {"b": 1}}, [10])
    assert per_query["q2"] == {"ndcg@10": 0.0, "recall@10": 0.0, "mrr@10": 0.0}
    assert means["ndcg@10"] == pytest.approx(0.5)


def test_evaluate_rejects_empty_qrels() -> None:
    with pytest.raises(ValueError):
        evaluate({}, {}, [10])


def test_matches_trec_eval_on_random_rankings() -> None:
    rng = random.Random(7)
    docs = [f"d{i}" for i in range(60)]
    qrels: dict[str, dict[str, int]] = {}
    rankings: dict[str, list[str]] = {}
    for n in range(40):
        q = f"q{n}"
        judged = rng.sample(docs, rng.randint(1, 12))
        qrels[q] = {doc: rng.choice([0, 1, 1, 2, 3]) for doc in judged}
        if not any(grade > 0 for grade in qrels[q].values()):
            qrels[q][judged[0]] = 1
        rankings[q] = rng.sample(docs, rng.randint(0, 40))

    means, per_query = evaluate(rankings, qrels, [5, 10])

    def trec(measures: set[str], depth: int | None = None) -> dict[str, dict[str, float]]:
        # Strictly decreasing scores, so trec_eval's tie order cannot matter.
        run = {
            q: {doc: float(len(r) - i) for i, doc in enumerate(r[:depth])}
            for q, r in rankings.items()
        }
        result: dict[str, dict[str, float]] = pytrec_eval.RelevanceEvaluator(
            qrels, measures
        ).evaluate(run)
        return result

    reference = trec({"ndcg_cut.5,10", "recall.5,10"})
    for k in (5, 10):
        reference_rr = trec({"recip_rank"}, depth=k)
        for query_id in qrels:
            ref = reference.get(query_id, {})
            assert per_query[query_id][f"ndcg@{k}"] == pytest.approx(ref.get(f"ndcg_cut_{k}", 0.0))
            assert per_query[query_id][f"recall@{k}"] == pytest.approx(ref.get(f"recall_{k}", 0.0))
            rr = reference_rr.get(query_id, {}).get("recip_rank", 0.0)
            assert per_query[query_id][f"mrr@{k}"] == pytest.approx(rr)
    assert 0 < means["ndcg@10"] < 1
