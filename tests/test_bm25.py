from __future__ import annotations

import math

import pytest

from retrieval_lab.bm25 import BM25Index
from retrieval_lab.config import AnalyzerConfig, BM25Config
from retrieval_lab.datasets import Document
from retrieval_lab.text import build_analyzer

PLAIN = AnalyzerConfig(stopwords="none", stemmer="none")


def doc(doc_id: str, text: str) -> Document:
    return Document(doc_id, "", text)


def test_score_matches_bm25_formula() -> None:
    docs = [doc("a", "apple apple pie"), doc("b", "banana"), doc("c", "cherry pie")]
    index = BM25Index(docs, BM25Config(k1=1.2, b=0.75, analyzer=PLAIN))

    # Title is empty, so doc "a" has 3 terms; avgdl = (3 + 1 + 2) / 3 = 2.
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / 2)
    expected = idf * 2 * 2.2 / (2 + norm)

    [(doc_id, score)] = index.search("apple", 10)
    assert doc_id == "a"
    assert score == pytest.approx(expected)


def test_idf_stays_positive_for_common_terms() -> None:
    docs = [doc("a", "the cat"), doc("b", "the dog"), doc("c", "the end")]
    index = BM25Index(docs, BM25Config(analyzer=PLAIN))
    assert all(score > 0 for _, score in index.search("the", 10))


def test_ties_are_broken_by_document_id() -> None:
    docs = [doc("z", "same text"), doc("m", "same text"), doc("a", "same text")]
    index = BM25Index(docs, BM25Config(analyzer=PLAIN))
    assert [doc_id for doc_id, _ in index.search("same", 10)] == ["a", "m", "z"]


def test_top_k_and_non_matching_documents() -> None:
    docs = [doc(str(i), f"term{i} shared") for i in range(5)]
    index = BM25Index(docs, BM25Config(analyzer=PLAIN))
    assert len(index.search("shared", 3)) == 3
    assert index.search("absent", 10) == []
    assert index.search("", 10) == []


def test_title_is_searchable() -> None:
    index = BM25Index([Document("a", "Felines", "body text")], BM25Config(analyzer=PLAIN))
    assert index.search("felines", 1)[0][0] == "a"


def test_analyzer_stopwords_and_porter_stemming() -> None:
    analyze = build_analyzer(AnalyzerConfig())
    assert analyze("The Cats are Running to the trees") == ["cat", "run", "tree"]
    assert build_analyzer(PLAIN)("The Cats") == ["the", "cats"]


def test_stemming_matches_inflected_forms() -> None:
    docs = [doc("a", "running shoes"), doc("b", "a quiet library")]
    stemmed = BM25Index(docs, BM25Config())
    plain = BM25Index(docs, BM25Config(analyzer=PLAIN))
    assert stemmed.search("runs", 10)[0][0] == "a"
    assert plain.search("runs", 10) == []
