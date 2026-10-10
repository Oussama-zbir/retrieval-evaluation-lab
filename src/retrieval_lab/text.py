"""Text analysis shared by indexing and querying."""

from __future__ import annotations

import re
from collections.abc import Callable

import Stemmer

from retrieval_lab.config import AnalyzerConfig

# org.apache.lucene.analysis.en.EnglishAnalyzer.ENGLISH_STOP_WORDS_SET
# fmt: off
LUCENE_ENGLISH_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
    "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with",
})
# fmt: on

_TOKEN = re.compile(r"\w+")

Analyzer = Callable[[str], list[str]]


def build_analyzer(config: AnalyzerConfig) -> Analyzer:
    """Return a function mapping text to its index terms under ``config``."""
    stopwords = LUCENE_ENGLISH_STOPWORDS if config.stopwords == "lucene-english" else frozenset()
    stemmer = Stemmer.Stemmer("porter") if config.stemmer == "porter" else None

    def analyze(text: str) -> list[str]:
        if config.lowercase:
            text = text.lower()
        tokens = [token for token in _TOKEN.findall(text) if token not in stopwords]
        if stemmer is None:
            return tokens
        stemmed: list[str] = stemmer.stemWords(tokens)
        return stemmed

    return analyze
