"""Okapi BM25 over an in-memory inverted index.

Scoring follows Lucene's ``BM25Similarity``: idf = ln(1 + (N - df + 0.5) / (df + 0.5)),
which stays positive for terms in more than half the corpus, unlike the classic
Robertson idf. The (k1 + 1) numerator factor Lucene dropped in 8.0 is kept; it
scales every score equally and cannot change a ranking. Ties are broken by
document id so a ranking is a pure function of corpus, query and config.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Iterable

from retrieval_lab.config import BM25Config
from retrieval_lab.datasets import Document
from retrieval_lab.text import build_analyzer

Ranking = list[tuple[str, float]]


class BM25Index:
    def __init__(self, documents: Iterable[Document], config: BM25Config) -> None:
        self.config = config
        self._analyze = build_analyzer(config.analyzer)
        self._doc_ids: list[str] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        lengths: list[int] = []
        for doc in documents:
            index = len(self._doc_ids)
            self._doc_ids.append(doc.doc_id)
            # BEIR convention: the title is part of the searchable text.
            terms = self._analyze(f"{doc.title} {doc.text}")
            lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                self._postings.setdefault(term, []).append((index, tf))

        n_docs = len(self._doc_ids)
        avg_length = sum(lengths) / n_docs if n_docs else 0.0
        b = config.b
        # Per-document length normalisation, precomputed once.
        self._norm = [
            config.k1 * (1 - b + b * length / avg_length) if avg_length else config.k1
            for length in lengths
        ]
        self._idf = {
            term: math.log(1 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self._doc_ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def describe(self) -> dict[str, int]:
        return {"documents": len(self), "vocabulary": self.vocabulary_size}

    def search(self, query: str, k: int) -> Ranking:
        """Return up to ``k`` (doc id, score) pairs, best first. Only matching docs score."""
        k1_plus_1 = self.config.k1 + 1
        scores: dict[int, float] = {}
        # A repeated query term weighs proportionally more, as in Lucene.
        for term, query_tf in Counter(self._analyze(query)).items():
            postings = self._postings.get(term)
            if postings is None:
                continue
            weight = self._idf[term] * query_tf
            for doc, tf in postings:
                scores[doc] = scores.get(doc, 0.0) + weight * tf * k1_plus_1 / (
                    tf + self._norm[doc]
                )
        best = heapq.nsmallest(
            k, scores.items(), key=lambda item: (-item[1], self._doc_ids[item[0]])
        )
        return [(self._doc_ids[doc], score) for doc, score in best]
