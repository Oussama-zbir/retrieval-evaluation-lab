"""Paired comparison of run records: is a difference in mean metric real?

Two pipelines evaluated on the same queries give paired per-query scores, so the
question is about the per-query differences d_i = candidate_i - baseline_i.

- p-value: two-sided paired randomisation (sign-flip) test on the mean of d.
  Under the null hypothesis that the two pipelines are exchangeable, each d_i
  is as likely to have the opposite sign, so the observed mean is compared
  with the means of randomly sign-flipped differences (Smucker, Allan and
  Carterette, 2007, recommend this test for IR). The Monte Carlo estimate is
  (hits + 1) / (trials + 1), which never reports p = 0 (Phipson and Smyth, 2010).
- Confidence interval: percentile bootstrap of the mean difference, resampling
  queries with replacement and keeping each query's pair together.
- Multiple comparisons: Holm's step-down adjustment over every test in a report,
  because one baseline against several candidates on several metrics is a
  family of tests and some will reach p < 0.05 by chance.

Only records measured on the same dataset files, split and queries can be
compared. Everything is seeded, so a report is a deterministic function of its
inputs.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import fmean

from retrieval_lab.runs import RunRecord

# Per-query scores are stored rounded to 6 decimals, so sums of differences carry
# floating-point noise far below this. A permuted statistic within it of the
# observed one counts as "at least as extreme".
_TOLERANCE = 1e-9


class ComparisonError(ValueError):
    """The records cannot be compared (different data, queries or metrics)."""


@dataclass(frozen=True)
class Comparison:
    baseline: str
    candidate: str
    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    p_holm: float
    wins: int
    ties: int
    losses: int


def randomisation_test(differences: Sequence[float], *, trials: int, seed: int) -> float:
    """Two-sided paired sign-flip test of mean(differences) == 0."""
    if trials < 1:
        raise ValueError("trials must be positive")
    # Zero differences are unchanged by a sign flip and only add work.
    nonzero = [d for d in differences if d != 0.0]
    total = sum(nonzero)
    observed = abs(total)
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        bits = rng.getrandbits(len(nonzero)) if nonzero else 0
        flipped = sum(d for i, d in enumerate(nonzero) if bits >> i & 1)
        if abs(total - 2 * flipped) >= observed - _TOLERANCE:
            hits += 1
    return (hits + 1) / (trials + 1)


def bootstrap_interval(
    differences: Sequence[float], *, resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    """Percentile bootstrap interval for mean(differences)."""
    if not differences:
        raise ValueError("no differences to resample")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(fmean(rng.choices(differences, k=n)) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return _quantile(means, tail), _quantile(means, 1 - tail)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def compare(
    baseline: RunRecord,
    candidates: Sequence[RunRecord],
    metrics: Sequence[str],
    *,
    trials: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> list[Comparison]:
    """Compare each candidate with the baseline on each metric."""
    if not candidates:
        raise ComparisonError("at least one candidate record is required")
    if not metrics:
        raise ComparisonError("at least one metric is required")
    if trials < 1:
        raise ComparisonError("trials must be positive")
    if not 0 < confidence < 1:
        raise ComparisonError("confidence must be in (0, 1)")
    for candidate in candidates:
        _check_comparable(baseline, candidate, metrics)

    queries = sorted(baseline.per_query)
    rows: list[Comparison] = []
    for candidate in candidates:
        for metric in metrics:
            base = [baseline.per_query[q][metric] for q in queries]
            cand = [candidate.per_query[q][metric] for q in queries]
            differences = [c - b for b, c in zip(base, cand, strict=True)]
            low, high = bootstrap_interval(
                differences, resamples=trials, seed=seed, confidence=confidence
            )
            rows.append(
                Comparison(
                    baseline=baseline.run_id,
                    candidate=candidate.run_id,
                    metric=metric,
                    baseline_mean=fmean(base),
                    candidate_mean=fmean(cand),
                    delta=fmean(differences),
                    ci_low=low,
                    ci_high=high,
                    p_value=randomisation_test(differences, trials=trials, seed=seed),
                    p_holm=1.0,
                    wins=sum(d > 0 for d in differences),
                    ties=sum(d == 0 for d in differences),
                    losses=sum(d < 0 for d in differences),
                )
            )
    adjusted = holm([row.p_value for row in rows])
    return [replace(row, p_holm=p) for row, p in zip(rows, adjusted, strict=True)]


def render_markdown(
    baseline: RunRecord,
    candidates: Sequence[RunRecord],
    rows: Sequence[Comparison],
    *,
    trials: int,
    seed: int,
    confidence: float,
    alpha: float = 0.05,
) -> str:
    names = {record.run_id: record.pipeline.name for record in (baseline, *candidates)}
    dataset = baseline.dataset
    lines = [
        f"Baseline: `{baseline.pipeline.name}` ({baseline.run_id})",
        f"Dataset: {dataset.name} / {dataset.split}, {dataset.queries} queries",
        f"Test: paired randomisation, two-sided, {trials:,} sign flips; "
        f"{confidence:.0%} paired bootstrap CI, {trials:,} resamples; seed {seed}; "
        f"Holm-adjusted over {len(rows)} tests; significant at {alpha} marked *",
        "",
        "| Candidate | Metric | Baseline | Candidate | Delta | "
        f"{confidence:.0%} CI | W / T / L | p | p (Holm) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        mark = " *" if row.p_holm < alpha else ""
        lines.append(
            f"| `{names[row.candidate]}` | {row.metric} | {row.baseline_mean:.4f} | "
            f"{row.candidate_mean:.4f} | {row.delta:+.4f} | "
            f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}] | "
            f"{row.wins} / {row.ties} / {row.losses} | {row.p_value:.4f} | "
            f"{row.p_holm:.4f}{mark} |"
        )
    return "\n".join(lines) + "\n"


def _check_comparable(baseline: RunRecord, candidate: RunRecord, metrics: Sequence[str]) -> None:
    label = f"{candidate.run_id} vs {baseline.run_id}"
    a, b = baseline.dataset, candidate.dataset
    if (a.name, a.split) != (b.name, b.split):
        raise ComparisonError(f"{label}: measured on {b.name}/{b.split}, not {a.name}/{a.split}")
    if a.fingerprint != b.fingerprint:
        raise ComparisonError(f"{label}: dataset files differ (fingerprint mismatch)")
    if set(baseline.per_query) != set(candidate.per_query):
        raise ComparisonError(f"{label}: evaluated on different queries")
    for metric in metrics:
        for record in (baseline, candidate):
            if metric not in record.metrics:
                known = ", ".join(sorted(record.metrics))
                raise ComparisonError(f"{record.run_id} has no {metric!r} (has: {known})")


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of sorted values (numpy's default method)."""
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
