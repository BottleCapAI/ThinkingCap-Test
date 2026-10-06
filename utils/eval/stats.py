"""Confidence intervals for one run, and the paired difference between two.

Scores are bootstrapped over whole prompts, never over individual generations:
nine samples of one question are not nine questions, and treating them as such
makes every interval look about three times tighter than it is.

    from utils.eval.stats import confidence, summarize

    records = [{"id": "q1", "correct": True, "output_tokens": 120, "truncated": False}, ...]
    summarize(records)["accuracy"]     # 0.58
    confidence(records)["accuracy"]    # [0.546, 0.613]

Everything here takes graded generations -- what evaluate.py calls records, one
per question per sample. `id` is the question they came from, and grouping on
it is the whole point: nine samples of one question are one question's worth of
evidence, not nine.
"""

import math
import random

RESAMPLES = 2000
SEED = 42


def summarize(rows):
    """One benchmark's graded generations -> its descriptive numbers.

    Rows whose grader failed carry correct=None; they are left out of every
    statistic and counted separately, so a grader problem shrinks the
    denominator visibly instead of silently.
    """
    graded = [r for r in rows if r["correct"] is not None]
    if not graded:
        raise ValueError("No graded rows")
    count = len(graded)
    return {
        "n": count,
        "prompts": len({r["id"] for r in graded}),
        "ungraded": len(rows) - count,
        "accuracy": sum(r["correct"] for r in graded) / count,
        "truncation_rate": sum(r["truncated"] for r in graded) / count,
        "parse_failure_rate": sum(bool(r.get("parse_error")) for r in graded) / count,
        "mean_output_tokens": sum(r["output_tokens"] for r in graded) / count,
        "mean_reasoning_tokens": sum(r.get("reasoning_tokens", 0) for r in graded) / count,
        "total_output_tokens": sum(r["output_tokens"] for r in graded),
    }


def confidence(rows, resamples=RESAMPLES, seed=SEED):
    """One run -> 95% intervals for its accuracy and mean token count.

    This is the error bar a leaderboard row carries: how much the score would
    move on a different sample of prompts from the same distribution.
    """
    prompts = group_by_prompt(rows, lambda r: (int(r["correct"]), r["output_tokens"], 1))
    result = {"method": "prompt_cluster_percentile", "resamples": resamples, "seed": seed,
              "prompts": len(prompts), "accuracy": None, "mean_output_tokens": None}
    if len(prompts) < 2:
        return result  # one prompt says nothing about variation across prompts
    rng = random.Random(seed)
    accuracies, token_counts = [], []
    for totals in resample(prompts, rng, resamples, width=3):
        correct, tokens, count = totals
        accuracies.append(correct / count)
        token_counts.append(tokens / count)
    result["accuracy"] = interval(accuracies)
    result["mean_output_tokens"] = interval(token_counts)
    return result


def compare_runs(base, candidate, resamples=RESAMPLES, seed=SEED):
    """Two runs of the same cohort -> intervals for the difference between them.

    Pairing cancels question difficulty, so this is far tighter than the gap
    between two separate confidence() intervals. Rows must already be aligned.
    """
    if len(base) != len(candidate):
        raise ValueError("Runs must contain the same number of rows")
    pairs = []
    for left, right in zip(base, candidate):
        if left["id"] != right["id"] or left.get("sample_index") != right.get("sample_index"):
            raise ValueError("Runs must be in identical order")
        if left["correct"] is None or right["correct"] is None:
            continue
        pairs.append((left, right))
    prompts = group_by_pair(pairs)
    result = {"method": "paired_prompt_cluster_percentile", "resamples": resamples,
              "seed": seed, "prompts": len(prompts),
              "accuracy_delta": None, "token_reduction": None}
    if len(prompts) < 2:
        return result
    rng = random.Random(seed)
    deltas, savings = [], []
    for totals in resample(prompts, rng, resamples, width=5):
        base_correct, new_correct, base_tokens, new_tokens, count = totals
        deltas.append((new_correct - base_correct) / count)
        if base_tokens:
            savings.append(1 - new_tokens / base_tokens)
    result["accuracy_delta"] = interval(deltas)
    if len(savings) == resamples:
        result["token_reduction"] = interval(savings)
    return result


def compare_macro(groups, resamples=RESAMPLES, seed=SEED):
    """Unweighted mean of per-group accuracy changes and geometric mean of per-group token ratios -> intervals.

    `groups` is a list of (base_rows, candidate_rows), each aligned as for
    compare_runs. Every resample redraws prompts inside each group separately and
    averages the group results, so a small benchmark counts as much as a large one.
    """
    pooled = []
    for base, candidate in groups:
        pairs = [(l, r) for l, r in zip(base, candidate)
                 if l["correct"] is not None and r["correct"] is not None]
        pooled.append(group_by_pair(pairs))
    result = {"method": "macro_paired_prompt_cluster_percentile", "resamples": resamples,
              "seed": seed, "groups": len(groups), "accuracy_delta": None, "token_reduction": None}
    if not pooled or any(len(prompts) < 2 for prompts in pooled):
        return result
    rng = random.Random(seed)
    deltas, savings = [], []
    for draws in zip(*(resample(prompts, rng, resamples, width=5) for prompts in pooled)):
        group_deltas = [(new - old) / count for old, new, _, _, count in draws]
        ratios = [new / old for _, _, old, new, _ in draws if old and new]
        deltas.append(sum(group_deltas) / len(group_deltas))
        if len(ratios) == len(draws):
            savings.append(1 - math.exp(sum(math.log(r) for r in ratios) / len(ratios)))
    result["accuracy_delta"] = interval(deltas)
    if len(savings) == resamples:
        result["token_reduction"] = interval(savings)
    return result


def floor(base, candidate):
    """Smallest accuracy half-width more samples could ever reach, in percentage points.

    Variance splits into a part that varies between questions and a part that
    varies between samples of one question:

        Var(mean) = between / P  +  within / (P x k)

    Only the second term shrinks with more samples k, so the first is a floor.
    Getting under it needs more questions, not more samples. The variance of the
    per-question means still contains within / k, so that share is estimated from
    the spread inside each question and subtracted (the ANOVA variance-components
    estimate). Returns None when there is nothing to estimate from.
    """
    per_prompt = {}
    for left, right in zip(base, candidate):
        if left["correct"] is None or right["correct"] is None:
            continue
        per_prompt.setdefault(left["id"], []).append(int(right["correct"]) - int(left["correct"]))
    groups = [v for v in per_prompt.values() if v]
    repeated = [v for v in groups if len(v) > 1]
    if len(groups) < 2 or not repeated:
        return None
    means = [sum(v) / len(v) for v in groups]
    average = sum(means) / len(means)
    spread = sum((m - average) ** 2 for m in means) / (len(means) - 1)
    within = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)
                 for v in repeated) / len(repeated)
    shrinkable = within * sum(1 / len(v) for v in groups) / len(groups)
    between = max(0.0, spread - shrinkable)
    return {"prompts": len(groups), "between_variance": between, "within_variance": within,
            "half_width_pp": 1.96 * math.sqrt(between / len(groups)) * 100}


def group_by_pair(pairs):
    """One running total per question, over both runs at once."""
    totals = {}
    for left, right in pairs:
        running = totals.setdefault(left["id"], [0] * 5)
        for i, value in enumerate((int(left["correct"]), int(right["correct"]),
                                   left["output_tokens"], right["output_tokens"], 1)):
            running[i] += value
    return list(totals.values())


def group_by_prompt(rows, measure):
    """Collapse rows into one running total per prompt, so resampling draws prompts."""
    totals = {}
    for row in rows:
        values = measure(row)
        running = totals.setdefault(row["id"], [0] * len(values))
        for i, value in enumerate(values):
            running[i] += value
    return list(totals.values())


def resample(prompts, rng, resamples, width):
    """Yield `resamples` bootstrap totals, each drawing len(prompts) prompts with replacement."""
    for _ in range(resamples):
        totals = [0] * width
        for _ in prompts:
            drawn = rng.choice(prompts)
            for i, value in enumerate(drawn):
                totals[i] += value
        yield totals


def interval(values):
    """Sorted 2.5th and 97.5th percentiles of a bootstrap distribution."""
    values.sort()

    def quantile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return [quantile(0.025), quantile(0.975)]
