# Retrieval Evaluation Lab

[![CI](https://github.com/Oussama-zbir/retrieval-evaluation-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Oussama-zbir/retrieval-evaluation-lab/actions/workflows/ci.yml)

Retrieval pipelines for RAG, measured instead of demoed. Each pipeline runs
against a public corpus with human relevance judgements, and every result is a
versioned run record that CI re-executes from the raw data and requires to match
exactly.

The question this repository answers is the one every RAG system eventually
gets asked: *how do you know the retriever is any good, and how do you know a
change made it better?*

## Status

This is the foundation: data handling, metrics, run records and a BM25
baseline. Dense retrieval, hybrid fusion, reranking and an LLM-judge metric come
next (see [Roadmap](#roadmap)); each will be added as a measured delta against
the baseline.

| Area | What exists |
|---|---|
| Data | BEIR-format loader; downloads pinned to the published MD5; zip-slip-safe extraction; qrels cross-checked against the corpus and queries |
| Metrics | nDCG@k, Recall@k, MRR@k, tested for equivalence with `trec_eval` (via `pytrec_eval`) on randomised rankings |
| Retrieval | Okapi BM25 with Lucene's idf over an in-memory inverted index; analyzer (stopwords, Porter stemming) as configuration |
| Run records | Dataset SHA-256 fingerprint, full pipeline config and hash, source digest, git SHA, library versions, mean and per-query metrics, latency |
| Reproducibility | `retrieval-lab reproduce` re-runs a record and fails on any metric or per-query difference; CI does this for every checked-in record |

## Results: SciFact (BEIR), test split

5,183 abstracts, 300 test claims, 339 relevance judgements.

| Pipeline | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 query |
|---|---|---|---|---|---|
| `bm25` (k1 1.2, b 0.75, stopwords + Porter) | **0.683** | **0.817** | 0.928 | **0.646** | 0.95 ms |
| `bm25-k1-09-b-04` (Anserini's defaults) | 0.677 | 0.803 | **0.929** | 0.642 | 0.96 ms |
| `bm25-no-stemming` (lowercase only) | 0.663 | 0.788 | 0.883 | 0.631 | 2.15 ms |

Records: [`runs/scifact/`](runs/scifact/). Configs: [`configs/`](configs/).

Reading the table:

- **The baseline is in the expected range.** The BEIR paper reports 0.665
  nDCG@10 for its Elasticsearch BM25 on SciFact. Elasticsearch scores title and
  body as separate fields; this index concatenates them, which is one plausible
  source of the gap. A baseline far from published numbers would point to a
  tokenisation or metric bug, so this check comes first.
- **Text analysis is worth more than parameter tuning here.** Dropping
  stopwords and stemming costs 0.02 nDCG@10 and 4.5 points of Recall@100, and
  doubles query latency, because stopwords have long posting lists. Changing
  k1/b moves nDCG@10 by 0.006.
- **Only the stemming effect on recall survives a significance test.**
  `retrieval-lab compare` runs a paired randomisation test per metric and
  adjusts for the four comparisons (Holm):

  | Candidate vs `bm25` | Metric | Delta | 95% CI | W / T / L | p (Holm) |
  |---|---|---|---|---|---|
  | `bm25-k1-09-b-04` | nDCG@10 | -0.0059 | [-0.0154, +0.0038] | 15 / 253 / 32 | 0.45 |
  | `bm25-k1-09-b-04` | Recall@100 | +0.0011 | [+0.0000, +0.0033] | 1 / 299 / 0 | 1.00 |
  | `bm25-no-stemming` | nDCG@10 | -0.0193 | [-0.0383, -0.0019] | 33 / 227 / 40 | 0.11 |
  | `bm25-no-stemming` | Recall@100 | -0.0450 | [-0.0717, -0.0217] | 1 / 284 / 15 | 0.003 |

  The k1/b change is noise. Dropping stemming loses Recall@100 on 15 queries
  and gains it on one. Its nDCG@10 loss has p = 0.036 before adjustment and a
  bootstrap interval that excludes zero, but after correcting for four tests
  it is not significant at 0.05. Most queries are ties: on 300 queries, small
  pipeline changes alter few rankings.
- Latency is single-threaded pure Python on a laptop, so only the relative
  numbers mean anything.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

retrieval-lab fetch scifact                     # download, verify MD5, extract to data/
retrieval-lab run --dataset scifact --config configs/bm25.json
retrieval-lab reproduce runs/scifact/<run-id>.json
retrieval-lab compare runs/scifact/<baseline>.json runs/scifact/<candidate>.json ...
```

A pipeline is a JSON file validated by Pydantic. Unknown fields are rejected, so
a typo cannot silently fall back to a default:

```json
{
  "name": "bm25",
  "retriever": {
    "kind": "bm25", "k1": 1.2, "b": 0.75,
    "analyzer": {"lowercase": true, "stopwords": "lucene-english", "stemmer": "porter"}
  },
  "top_k": 100
}
```

## Design decisions

**Run records are written once.** A record names its dataset by content hash
rather than by name, its pipeline by full config plus canonical hash, and its
code by a SHA-256 over the package source as well as the git SHA. The git SHA
alone is not enough, because a record committed with the change that produced
it cannot contain that commit's SHA. Records are opened with exclusive create,
so a rerun produces a new record and never overwrites an old one.

**Metrics follow `trec_eval`, and a test checks that.** IR metrics are easy to
get subtly wrong: a truncated ideal ranking, the wrong log base, or queries with
no results dropped from the mean. The test suite compares nDCG, Recall and MRR
with `pytrec_eval` on 40 random queries with graded judgements. Queries with no
results count as zero, so a pipeline cannot improve its average by failing on
hard queries.

**BM25 is implemented, not imported.** It is under 80 lines, and owning it makes
the analyzer, the idf variant and tie-breaking explicit choices that appear in
the config and the record. Ties are broken by document id, so a ranking is a
deterministic function of its inputs. Exact reproduction depends on that.

**Only judged queries are evaluated.** The 809 SciFact queries without test
judgements are dropped at load time, and the judgement count is recorded. A
document nobody judged counts as not relevant; this is the standard assumption,
and it is biased against systems that find relevant documents the assessors
never saw. That bias matters more once dense retrieval is added.

**A pipeline is not "better" until a paired test says so.** Both pipelines are
scored on the same queries, so `compare` works on per-query differences: a
two-sided sign-flip randomisation test for the p-value (recommended for IR by
Smucker, Allan and Carterette, 2007; estimated as (hits + 1) / (trials + 1), so
it is never 0) and a paired percentile bootstrap for the interval. One baseline
against several candidates on several metrics is a family of tests, so p-values
are Holm-adjusted across the whole report. Records with different dataset
fingerprints or query sets are refused rather than compared. The test is
checked against exact enumeration of all 2^n sign flips on small inputs. Both
procedures are seeded, so a report is deterministic.

## Layout

```
src/retrieval_lab/
  datasets.py   fetch (pinned MD5, safe extraction), load + validate, fingerprint
  text.py       analyzer: tokenise, stopwords, Porter stemming
  bm25.py       inverted index and Lucene-idf BM25
  metrics.py    nDCG / Recall / MRR at k, trec_eval conventions
  config.py     frozen, hashable pipeline configs
  runs.py       execute a pipeline, build and persist the run record
  compare.py    paired randomisation test, bootstrap CI, Holm adjustment
  cli.py        fetch / run / reproduce / compare
configs/        checked-in pipeline configs
runs/           checked-in run records (evidence for the Results table)
```

## Testing

```bash
pytest          # 49 tests, offline; no dataset download needed
ruff check . && ruff format --check .
mypy            # strict
```

CI runs these on Python 3.12 and 3.13. A separate job fetches SciFact and runs
`reproduce` on every record in `runs/`.

## Limitations

- One dataset so far. SciFact is small (5k documents) and domain-specific;
  conclusions about stemming or k1/b may not transfer.
- Single in-memory index in pure Python: fine for BEIR-scale corpora, not a
  serving design.
- Significance is per dataset. With 300 queries and mostly tied rankings,
  the tests can only detect fairly large effects; a non-significant delta is
  not evidence of no difference.
- Recorded latency is wall-clock on the machine that ran it and is excluded
  from the reproduction check.

## Roadmap

1. Dense retrieval with a sentence-embedding model, then reciprocal-rank-fusion
   hybrid, each as a measured delta.
2. Cross-encoder reranking, with the latency cost recorded next to the quality gain.
3. Chunking and ingestion strategy as a measured variable.
4. An LLM-judge answer-faithfulness metric, validated against human labels
   before it is trusted.

## Data

SciFact (Wadden et al., 2020) as distributed by BEIR (Thakur et al., 2021). The
dataset is downloaded at run time and not redistributed here.
