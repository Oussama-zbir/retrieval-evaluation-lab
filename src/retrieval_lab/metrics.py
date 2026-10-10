"""Rank-based IR metrics, matching trec_eval / pytrec_eval conventions as used by BEIR.

- nDCG@k: linear gain (the judged grade), discount log2(rank + 1), ideal ranking
  built from every positively judged document for the query.
- Recall@k: relevant documents in the top k over all relevant documents.
- MRR@k: reciprocal rank of the first relevant document in the top k, else 0.

A document is relevant when its grade is > 0. Unjudged documents count as not
relevant, which is the standard assumption and a known bias against systems that
surface relevant documents the original assessors never saw.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import fmean

Qrels = Mapping[str, Mapping[str, int]]


def ndcg_at_k(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    dcg = sum(
        max(judged.get(doc_id, 0), 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
    )
    ideal_gains = sorted((grade for grade in judged.values() if grade > 0), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, grade in judged.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], judged: Mapping[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if judged.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


METRICS = {"ndcg": ndcg_at_k, "recall": recall_at_k, "mrr": mrr_at_k}


def evaluate(
    rankings: Mapping[str, Sequence[str]], qrels: Qrels, cutoffs: Sequence[int]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Score every judged query; return (mean per metric, per-query metrics).

    Means are taken over all judged queries: a query the system returned nothing
    for scores 0 rather than being dropped from the denominator.
    """
    if not qrels:
        raise ValueError("no judged queries to evaluate")
    per_query: dict[str, dict[str, float]] = {}
    for query_id, judged in sorted(qrels.items()):
        ranked = rankings.get(query_id, ())
        per_query[query_id] = {
            f"{name}@{k}": fn(ranked, judged, k) for name, fn in METRICS.items() for k in cutoffs
        }
    names = [f"{name}@{k}" for name in METRICS for k in cutoffs]
    means = {name: fmean(scores[name] for scores in per_query.values()) for name in names}
    return means, per_query
