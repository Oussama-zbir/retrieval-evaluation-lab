# Retrieval Evaluation Lab

[![CI](https://github.com/Oussama-zbir/retrieval-evaluation-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Oussama-zbir/retrieval-evaluation-lab/actions/workflows/ci.yml)

Retrieval pipelines for RAG, measured instead of demoed. Each pipeline runs
against a public corpus with human relevance judgements, and every result is a
versioned run record that CI re-executes from the raw data and requires to match
exactly (or, for float embeddings, within a stated tolerance).

The question this repository answers is the one every RAG system eventually
gets asked: *how do you know the retriever is any good, and how do you know a
change made it better?*

## Status

Data handling, metrics, run records, a BM25 baseline, a first dense retriever,
a reciprocal-rank-fusion hybrid of the two, and a cross-encoder reranker over
the hybrid, on two BEIR datasets. Chunking and an LLM-judge metric come next
(see [Roadmap](#roadmap)); each will be added as a measured delta.

| Area | What exists |
|---|---|
| Data | SciFact and NFCorpus (BEIR); loader downloads pinned to the published MD5; zip-slip-safe extraction; qrels cross-checked against the corpus and queries |
| Metrics | nDCG@k, Recall@k, MRR@k, tested for equivalence with `trec_eval` (via `pytrec_eval`) on randomised rankings |
| Retrieval | Okapi BM25 with Lucene's idf over an in-memory inverted index; analyzer (stopwords, Porter stemming) as configuration. Dense bi-encoder (sentence-transformers, model pinned to a commit) with exact cosine search in numpy. Reciprocal rank fusion of any two or more of these. Cross-encoder reranking of any of them to a configured depth |
| Run records | Dataset SHA-256 fingerprint, full pipeline config and hash, source digest, git SHA, library versions, mean and per-query metrics, latency |
| Reproducibility | `retrieval-lab reproduce` re-runs a record; BM25 (and BM25-only fusion) must match exactly; anything with a neural model within 0.001 per mean metric. CI does this for every checked-in record |

## Results

The same six pipelines on two datasets, so a conclusion is only stated if it
holds on more than one corpus.

### SciFact (BEIR), test split

5,183 abstracts, 300 test claims, 339 relevance judgements.

| Pipeline | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 query |
|---|---|---|---|---|---|
| `bm25` (k1 1.2, b 0.75, stopwords + Porter) | 0.683 | 0.817 | 0.928 | 0.646 | 0.95 ms |
| `bm25-k1-09-b-04` (Anserini's defaults) | 0.677 | 0.803 | 0.929 | 0.642 | 0.96 ms |
| `bm25-no-stemming` (lowercase only) | 0.663 | 0.788 | 0.883 | 0.631 | 2.15 ms |
| `dense-minilm` (all-MiniLM-L6-v2, exact cosine) | 0.645 | 0.783 | 0.925 | 0.605 | 5.1 ms |
| `hybrid-bm25-minilm` (RRF of `bm25` + `dense-minilm`) | **0.718** | **0.851** | **0.955** | **0.684** | 7.3 ms |
| `rerank-hybrid-minilm` (ms-marco-MiniLM-L6-v2 cross-encoder over the hybrid's top 100) | 0.687 | 0.809 | **0.955** | 0.657 | 1,329 ms |

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

### NFCorpus (BEIR), test split

3,633 medical documents (PubMed abstracts), 323 test queries, 12,334 graded
judgements (11,758 at grade 1, 576 at grade 2). Queries are short lay-language
titles from NutritionFacts.org; documents are technical.

| Pipeline | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 query |
|---|---|---|---|---|---|
| `bm25` (k1 1.2, b 0.75, stopwords + Porter) | 0.322 | 0.153 | 0.246 | 0.527 | 0.11 ms |
| `bm25-k1-09-b-04` (Anserini's defaults) | 0.321 | 0.152 | 0.246 | 0.523 | 0.12 ms |
| `bm25-no-stemming` (lowercase only) | 0.306 | 0.149 | 0.236 | 0.514 | 0.19 ms |
| `dense-minilm` (all-MiniLM-L6-v2, exact cosine) | 0.317 | 0.155 | 0.311 | 0.508 | 4.7 ms |
| `hybrid-bm25-minilm` (RRF of `bm25` + `dense-minilm`) | **0.353** | **0.173** | **0.321** | 0.554 | 5.3 ms |
| `rerank-hybrid-minilm` (ms-marco-MiniLM-L6-v2 cross-encoder over the hybrid's top 100) | **0.353** | 0.165 | **0.321** | **0.584** | 1,360 ms |

Records: [`runs/nfcorpus/`](runs/nfcorpus/).

| Candidate vs `bm25` | Metric | Delta | 95% CI | W / T / L | p (Holm) |
|---|---|---|---|---|---|
| `bm25-k1-09-b-04` | nDCG@10 | -0.0018 | [-0.0056, +0.0020] | 51 / 214 / 58 | 0.70 |
| `bm25-k1-09-b-04` | Recall@100 | -0.0001 | [-0.0025, +0.0020] | 21 / 282 / 20 | 0.93 |
| `bm25-no-stemming` | nDCG@10 | -0.0161 | [-0.0267, -0.0056] | 67 / 177 / 79 | 0.016 |
| `bm25-no-stemming` | Recall@100 | -0.0104 | [-0.0181, -0.0010] | 25 / 224 / 74 | 0.020 |

- **The baseline matches the published number.** BEIR reports 0.325 nDCG@10
  for Elasticsearch BM25 on NFCorpus; this index gives 0.322.
- **Stemming now matters for ranking quality too.** Without it, both nDCG@10
  and Recall@100 drop significantly after Holm adjustment. On SciFact only
  recall did. The nDCG@10 loss is smaller here (-0.016 against -0.019), but
  its bootstrap interval is about half as wide, so it is detected. The
  direction agrees across both datasets; the strength of the evidence does
  not, which is the reason for running more than one.
- **k1/b is noise on both datasets.** Neither delta comes close to
  significance on either corpus, so further BM25 parameter search is not worth
  a session.
- **Low recall is partly the benchmark, not the retriever.** NFCorpus queries
  average 38 relevant documents (median 16, maximum 475); 22 queries have more
  than 100, so Recall@100 cannot reach 1 for them. Absolute values are only
  comparable between pipelines on the same dataset.

### Dense retrieval against BM25

`dense-minilm` is a small general-purpose bi-encoder (6 layers, 384
dimensions), not trained on either dataset. Its scores match the MTEB
leaderboard values for this model (0.645 on SciFact, 0.317 on NFCorpus), which
checks the encoding and search path the same way the BEIR numbers check BM25.
It is tested against `bm25` as its own family of two tests per dataset:

| Dataset | Metric | Delta | 95% CI | W / T / L | p (Holm) |
|---|---|---|---|---|---|
| SciFact | nDCG@10 | -0.0377 | [-0.0731, -0.0025] | 64 / 164 / 72 | 0.067 |
| SciFact | Recall@100 | -0.0026 | [-0.0347, +0.0303] | 15 / 272 / 13 | 0.90 |
| NFCorpus | nDCG@10 | -0.0056 | [-0.0278, +0.0160] | 111 / 110 / 102 | 0.62 |
| NFCorpus | Recall@100 | +0.0651 | [+0.0433, +0.0876] | 147 / 111 / 65 | 0.0002 |

- **Dense does not beat a tuned BM25 at the top of the ranking.** On SciFact
  it is 0.038 nDCG@10 behind (p = 0.033 raw, 0.067 after Holm); on NFCorpus
  the two are indistinguishable. SciFact claims share precise technical terms
  with their evidence, which is the case lexical matching handles well.
- **Dense finds documents BM25 misses on the vocabulary-mismatch dataset.**
  NFCorpus pairs lay queries with technical abstracts; dense raises
  Recall@100 by 6.5 points (26% relative), winning on 147 queries and losing
  on 65. On SciFact recall is unchanged.
- **The two disagree on many queries.** Fewer than half the NFCorpus queries
  are ties, and on SciFact each method wins nDCG@10 on roughly 70. Similar
  means with different per-query wins is the precondition for fusion, tested
  next.
- **The cost is in indexing.** Encoding the corpus takes 25-39 s on a laptop
  CPU against well under a second for BM25; a query (encoding plus one
  matrix-vector product) is about 5 ms.

### Hybrid: reciprocal rank fusion of BM25 and dense

`hybrid-bm25-minilm` fuses the two rankings above with RRF (k = 60, each
searched to depth 100). Neither parameter was tuned. It is tested against each
of its components as a separate family of two tests per dataset:

| Dataset | Baseline | Metric | Delta | 95% CI | W / T / L | p (Holm) |
|---|---|---|---|---|---|---|
| SciFact | `bm25` | nDCG@10 | +0.0353 | [+0.0121, +0.0580] | 74 / 196 / 30 | 0.004 |
| SciFact | `bm25` | Recall@100 | +0.0274 | [+0.0067, +0.0507] | 13 / 285 / 2 | 0.013 |
| SciFact | `dense-minilm` | nDCG@10 | +0.0730 | [+0.0494, +0.0972] | 83 / 190 / 27 | 0.0002 |
| SciFact | `dense-minilm` | Recall@100 | +0.0300 | [+0.0067, +0.0533] | 11 / 287 / 2 | 0.023 |
| NFCorpus | `bm25` | nDCG@10 | +0.0305 | [+0.0171, +0.0443] | 123 / 133 / 67 | 0.0002 |
| NFCorpus | `bm25` | Recall@100 | +0.0750 | [+0.0576, +0.0936] | 154 / 141 / 28 | 0.0002 |
| NFCorpus | `dense-minilm` | nDCG@10 | +0.0362 | [+0.0211, +0.0519] | 127 / 122 / 74 | 0.0002 |
| NFCorpus | `dense-minilm` | Recall@100 | +0.0099 | [-0.0007, +0.0213] | 91 / 178 / 54 | 0.075 |

- **Fusion beats both components at the top of the ranking, on both
  datasets.** The expectation from the dense results was that fusion would
  keep BM25's nDCG@10 and add dense recall. It does better: nDCG@10 rises
  3-4 points over the stronger component on each dataset, including SciFact,
  where dense alone was worse than BM25. Documents that both retrievers rank
  highly are more often relevant than those only one of them ranks highly.
- **Recall@100 improves on BM25 on both datasets.** Against dense on NFCorpus
  the gain (+0.010) is not significant: fusion keeps nearly all of what dense
  found, and only depth-100 candidates from either side can be recovered.
- **Two indexes for one query.** A hybrid query runs both retrievers in
  sequence (5-7 ms, dominated by query encoding) and the index costs the
  dense encoding time. For a lab of this size that is free; in a service the
  two searches would run concurrently.

### Cross-encoder reranking over the hybrid

`rerank-hybrid-minilm` rescores the hybrid's top 100 with
`cross-encoder/ms-marco-MiniLM-L6-v2` (pinned commit, 512 tokens per pair) and
is tested against the hybrid it reorders. Recall@100 is identical by
construction (depth = 100), so only the nDCG@10 test carries information:

| Dataset | Metric | Delta | 95% CI | W / T / L | p | p (Holm) |
|---|---|---|---|---|---|---|
| SciFact | nDCG@10 | -0.0306 | [-0.0600, -0.0008] | 48 / 186 / 66 | 0.043 | 0.086 |
| NFCorpus | nDCG@10 | -0.0001 | [-0.0130, +0.0129] | 103 / 125 / 95 | 0.98 | 1.00 |

- **The reranker does not improve on the hybrid.** On NFCorpus it reorders
  most rankings (198 of 323 queries change nDCG@10) with no net effect; MRR@10
  rises from 0.554 to 0.584, so it is slightly better at the first relevant
  document and slightly worse below it. On SciFact it loses more queries than
  it wins; the loss is not significant after Holm but the interval excludes
  zero. It still beats `bm25` on NFCorpus (+0.030 nDCG@10, Holm p 0.0002) and
  ties it on SciFact (+0.005, p 0.76), in line with the BEIR paper's BM25 +
  cross-encoder numbers (0.688 and 0.350).
- **Why, plausibly:** the model is trained on MS MARCO web queries and
  passages. SciFact claims and NFCorpus lay titles against PubMed abstracts
  are out of its domain, and abstracts often exceed what a 6-layer model reads
  well. The first stage already has 0.72 nDCG@10 on SciFact; a reranker has to
  be better than that ordering, not better than BM25.
- **It costs about 185x the query latency.** p50 is 1.3 s per query on a
  laptop CPU against 5-7 ms for the hybrid: 100 forward passes per query. In a
  service this is a GPU, a smaller `depth`, or both; the next measurement is
  `depth` as a quality / latency curve.
- **The decision this supports:** do not add a general-purpose reranker to
  this pipeline on the strength of its reputation. It would need a
  domain-matched or larger model, and a run record showing the gain, first.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Dense pipelines also need (CPU-only torch first, to skip the CUDA wheels):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dense]"

retrieval-lab fetch scifact                     # or nfcorpus; download, verify MD5, extract
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

**Dense search is exact before it is approximate.** Every query is scored
against every document with one matrix-vector product. An ANN index (HNSW,
IVF) trades recall for latency, and that trade can only be measured against an
exact baseline; at a few thousand documents exact search costs 5 ms anyway.
Ties are broken by document id, as in BM25.

**Dense records reproduce within a tolerance, not exactly.** Embeddings differ
in the last float bits across CPUs and BLAS builds, which can swap near-tied
documents. `reproduce` therefore requires every mean metric within 0.001 and
reports how many queries changed, instead of requiring identical per-query
scores. BM25 stays on exact equality. The model is pinned to a Hugging Face
commit, encoding is forced onto the CPU, and the batch size is part of the
config, so the remaining variation is numerical, not a different model. The
alternative, checking in the embeddings, would make the record exact but
would stop it from testing that the model still produces them.

**Fusion uses ranks, not scores.** BM25 scores are unbounded and
query-dependent; cosine similarities sit in a narrow band near the top. A
weighted score sum needs a normalisation and a weight, both tuned on the same
queries it is then evaluated on. Reciprocal rank fusion (Cormack et al., 2009)
needs neither: a document scores the sum of 1 / (60 + rank) over the rankings
it appears in. A hybrid is a config holding its component configs, so the run
record and hash cover both retrievers, and the components are searched to a
`depth` (100) independent of `top_k`. The sum is taken in exact rational
arithmetic: different rank pairs can have equal true sums (1/63 + 1/140 =
1/84 + 1/90) whose float sums differ by one ulp, and that ulp, not the doc-id
tie-break, would then decide the order.

**Reranking is a stage wrapping a first-stage config, and only reorders.**
A `rerank` config holds its first stage (BM25, dense or a hybrid; rerankers do
not nest), a cross-encoder pinned to a commit, a `depth` and a `max_length`.
The cross-encoder rescores the first stage's top `depth`; anything below keeps
its first-stage order after them. So Recall@`depth` is fixed by the first
stage by construction, and `depth` is the knob that trades quality for
latency: every candidate is one forward pass at query time, which is why the
cross-encoder is not the first stage. Scores are raw logits: the sigmoid most
cross-encoder wrappers apply does not change the order but saturates to
exactly 1.0 in float32 for confident pairs, turning them into doc-id ties.

**Dense dependencies are optional.** torch and sentence-transformers are
several hundred megabytes. They live in the `dense` extra and are imported only
when a dense pipeline is built, so the BM25 path and the metric tests run
without them; the unit tests use a deterministic fake encoder. A test runs the
CLI with numpy blocked to keep it that way.

## Layout

```
src/retrieval_lab/
  datasets.py   fetch (pinned MD5, safe extraction), load + validate, fingerprint
  text.py       analyzer: tokenise, stopwords, Porter stemming
  bm25.py       inverted index and Lucene-idf BM25
  dense.py      bi-encoder, exact cosine search (optional dependencies)
  hybrid.py     reciprocal rank fusion over component retrievers
  rerank.py     cross-encoder reranking of a first-stage ranking
  retrievers.py one interface over every retriever kind
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
pytest          # 76 tests, offline; no dataset download needed
ruff check . && ruff format --check .
mypy            # strict
```

CI runs these on Python 3.12 and 3.13. A separate job fetches both datasets,
runs `reproduce` on every record in `runs/`, and prints both comparison tables.

## Limitations

- Two small scientific/medical datasets (3.6k and 5.2k documents). Conclusions
  about the analyzer agree across them, but neither covers general-domain web
  text or a corpus large enough to stress the index.
- Single in-memory index in pure Python: fine for BEIR-scale corpora, not a
  serving design.
- Significance is per dataset. With 300 queries and mostly tied rankings,
  the tests can only detect fairly large effects; a non-significant delta is
  not evidence of no difference.
- One dense model, and a small one. A larger or instruction-tuned encoder
  (E5, BGE; the config already carries their prefixes) would likely beat
  BM25 on SciFact; this is a baseline for fusion, not a model comparison.
- Fusion parameters are the published defaults (k = 60, depth 100), not
  tuned; with two datasets and no held-out split, tuning them would be
  fitting the test queries. A hybrid query costs both retrievers in sequence.
- One cross-encoder, small and trained on web search; the negative reranking
  result is about this model on these domains, not about reranking in general.
- Recorded latency is wall-clock on the machine that ran it and is excluded
  from the reproduction check.

## Roadmap

1. Chunking and ingestion strategy as a measured variable.
2. Rerank depth as a measured quality / latency curve (one config per depth).
3. An LLM-judge answer-faithfulness metric, validated against human labels
   before it is trusted.

## Data

SciFact (Wadden et al., 2020) and NFCorpus (Boteva et al., 2016) as
distributed by BEIR (Thakur et al., 2021). Datasets are downloaded at run time
and not redistributed here.
