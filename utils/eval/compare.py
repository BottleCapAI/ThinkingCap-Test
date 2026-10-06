"""Compares two checkpoints that evaluate.py scored separately.

    from utils.eval.compare import compare, report
    print(report(compare("runs/base", "runs/mine")))

Both runs drew the same generations for the same questions, because the seed
comes from the cohort rather than from the run. That lets the rows be paired
afterwards, which cancels question difficulty and gives a much tighter interval
than subtracting two separate scores.

Pairing only holds if the runs really did ask the same thing, so that is
checked rather than assumed: same cohort, same sample count, same seed, same
token budget.
"""

import json
import math
from pathlib import Path
import re

from utils.eval.recipe import SAMPLES, SAMPLING, SEED, token_cap
from utils.eval.stats import compare_macro, compare_runs, floor, summarize

# owner/name, the only shape a Hub id takes. Anything else is a path.
HUB_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")

# What has to match, and why a reader should care that it does not. A run is
# only a controlled comparison if the other run asked the same questions in the
# same way; each of these changes what the model was asked or how far it got.
MUST_MATCH = {
    "selection": "a different list of questions is a different benchmark",
    "cohort_sha256": "the questions were edited between these runs",
    "seed": "different seeds draw different generations, so the rows cannot be paired",
    "samples": "fewer samples is a wider interval, not a worse score",
    "max_tokens": "a different budget changes what the model generates",
}


def compare(base_dir, candidate_dir):
    base_meta, base_rows = load(base_dir)
    new_meta, new_rows = load(candidate_dir)
    shared = check(base_meta, new_meta)
    results, pooled, ungraded = {}, {}, {}
    for dataset in sorted({row["dataset"] for row in base_rows}):
        left, right = pair([r for r in base_rows if r["dataset"] == dataset],
                           [r for r in new_rows if r["dataset"] == dataset])
        if not left:
            continue
        # evaluate.py deliberately survives a grader that could not run and still writes its
        # other benchmarks; this has to survive the same run. Dropping the benchmark keeps the
        # rest of a comparison that cost hours of generation.
        missing = [side for side, rows in (("base", left), ("candidate", right))
                   if not any(row["correct"] is not None for row in rows)]
        if missing:
            ungraded[dataset] = {"rows": len(left), "runs": missing}
            continue
        group = pooled.setdefault(average_group(dataset), ([], []))
        group[0].extend(left)
        group[1].extend(right)
        before, after = summarize(left), summarize(right)
        results[dataset] = {
            "paired_rows": len(left), "samples": base_meta["samples"],
            "base": before, "candidate": after,
            "accuracy_delta": after["accuracy"] - before["accuracy"],
            "token_reduction": token_reduction(before, after),
            "intervals": compare_runs(left, right), "floor": floor(left, right),
        }
    if not results:
        raise ValueError(
            "No benchmark could be compared; every one of them is ungraded in at least one run:\n"
            + "\n".join(f"  {name}   no graded rows in the {' and '.join(detail['runs'])} run"
                        for name, detail in sorted(ungraded.items()))
            + "\n\nRe-grade the stored generations with:\n"
              "  python evaluate.py --out <run directory> --regrade")
    return {"base": describe(base_dir, base_meta), "candidate": describe(candidate_dir, new_meta),
            "benchmarks": shared, "cohort_sha256": base_meta["cohort_sha256"],
            "samples": base_meta["samples"], "seed": base_meta["seed"], "results": results,
            "ungraded_benchmarks": ungraded, "average": average(pooled),
            "full_evaluation": {"base": full_evaluation_problems(base_meta, base_rows),
                                "candidate": full_evaluation_problems(new_meta, new_rows)}}


def full_evaluation_problems(meta, rows):
    """Why a run is not the full evaluation, as a list; empty when it is.

    Full means the `full` question selection with every question of every
    benchmark scored, under the settings in utils/eval/recipe.py. Anything else --
    the smoke subset, --benchmarks, a benchmark that failed to grade, another
    sample count or token cap -- is a different measurement and does not belong
    next to leaderboard rows.
    """
    problems = []
    if meta.get("selection") != "full":
        problems.append(f"it used the '{meta.get('selection')}' question selection, not 'full'")
    scored = len({(r["dataset"], r["id"]) for r in rows})
    total = meta.get("cohort_total")
    if total is not None and scored != total:
        problems.append(f"it scored {scored} of the {total} questions "
                        f"(benchmarks run: {', '.join(meta['benchmarks'])})")
    if meta.get("failed_benchmarks"):
        problems.append("these benchmarks failed to grade: "
                        + ", ".join(sorted(meta["failed_benchmarks"])))
    return problems + recipe_problems(meta)


def recipe_problems(meta):
    """Each setting that differs from utils/eval/recipe.py, said as what was used and what is required."""
    problems = []
    for field, wanted in (("samples", SAMPLES), ("seed", SEED)):
        if meta.get(field) != wanted:
            problems.append(f"it used {field} {meta.get(field)}, the recipe is {wanted}")
    caps = meta.get("max_tokens")
    if not isinstance(caps, dict):
        problems.append("it does not record per-benchmark token caps")
    else:
        off = {d: c for d, c in caps.items() if c != token_cap(d)}
        if off:
            problems.append("its token caps differ from the recipe: " + ", ".join(
                f"{d} {c} (recipe {token_cap(d)})" for d, c in sorted(off.items())))
    if meta.get("sampling") != SAMPLING:
        problems.append(f"its sampling was {meta.get('sampling') or 'not recorded'}, "
                        f"the recipe is {SAMPLING}")
    return problems


def average_group(dataset):
    """The BFCL subsets are one benchmark for averaging, so it is not weighted three times."""
    return "bfcl" if dataset.startswith("bfcl_") else dataset


def average(pooled):
    """Average over benchmarks, or None for one.

    Accuracy change is the arithmetic mean. Tokens saved is 1 minus the geometric mean of the
    per-benchmark token ratios (candidate / baseline), so the result does not depend on
    which of the two runs is put in the numerator.
    """
    if len(pooled) < 2:
        return None
    ratios, deltas = [], []
    for left, right in pooled.values():
        before, after = summarize(left), summarize(right)
        saved = token_reduction(before, after)
        ratios.append(None if saved is None else 1 - saved)
        deltas.append(after["accuracy"] - before["accuracy"])
    return {"benchmarks": sorted(pooled), "token_reduction": geometric_saving(ratios),
            "accuracy_delta": sum(deltas) / len(deltas),
            "intervals": compare_macro(list(pooled.values()))}


def geometric_saving(ratios):
    """1 minus the geometric mean of token ratios; None if any ratio is missing or not positive."""
    if any(r is None or r <= 0 for r in ratios):
        return None
    return 1 - math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def load(directory):
    directory = Path(directory)
    meta = json.loads((directory / "results.json").read_text())
    rows = [json.loads(line) for line in
            (directory / "records.jsonl").read_text().splitlines() if line.strip()]
    return meta, rows


def check(base, candidate):
    """Raise unless the two runs are comparable, naming everything that is not.

    Every mismatch is reported at once. Stopping at the first one makes the
    user fix, rerun, and discover the next -- and reruns here are hours.
    """
    problems = [describe_mismatch(f, why, base.get(f), candidate.get(f))
                for f, why in MUST_MATCH.items() if base.get(f) != candidate.get(f)]
    if base["model"] == candidate["model"] and base["adapter"] == candidate["adapter"]:
        problems.append("  checkpoint   both runs scored the same model and adapter\n"
                        "               there is nothing to compare")
    shared = sorted(set(base["benchmarks"]) & set(candidate["benchmarks"]))
    if not shared and not problems:
        problems.append(f"  benchmarks   {base['benchmarks']} vs {candidate['benchmarks']}\n"
                        "               the runs have no benchmark in common")
    if problems:
        raise ValueError("These runs are not comparable:\n\n" + "\n".join(problems)
                         + "\n" + what_did_match(base, candidate))
    return shared


def describe_mismatch(field, why, before, after):
    """One mismatch, as the values that differ and the consequence of differing."""
    if isinstance(before, dict) and isinstance(after, dict):
        changed = sorted(set(before) | set(after))
        detail = ", ".join(f"{k} {before.get(k)} vs {after.get(k)}"
                           for k in changed if before.get(k) != after.get(k))
    else:
        detail = f"{before} vs {after}"
    return f"  {field:<12} {detail}\n               {why}"


def what_did_match(base, candidate):
    """Say what is still shared, so the user knows how much has to be redone."""
    same = [f for f in MUST_MATCH if base.get(f) == candidate.get(f)]
    if not same:
        return "\nNothing about the two setups matches."
    if "cohort_sha256" in same and "selection" in same:
        return (f"\nBoth were scored on selection {base['selection']} "
                f"({base['cohort_sha256'][:8]}), so the questions do match.")
    return "\nStill shared: " + ", ".join(same) + "."


def pair(base_rows, candidate_rows):
    """Line rows up on (id, sample_index), keeping only what both runs scored."""
    indexed = {(r["id"], r["sample_index"]): r for r in candidate_rows}
    left, right = [], []
    for row in base_rows:
        match = indexed.get((row["id"], row["sample_index"]))
        if match is not None:
            left.append(row)
            right.append(match)
    return left, right


def token_reduction(before, after):
    if not before["total_output_tokens"]:
        return None
    return 1 - after["total_output_tokens"] / before["total_output_tokens"]


def describe(directory, meta):
    return {"run": str(Path(directory).resolve()), "model": meta["model"],
            "adapter": meta["adapter"]}


def loading_note(comparison):
    """Warn when the two checkpoints were referenced differently.

    A Hub id and a local path to the same weights do not generate the same
    text -- measured at 46 tokens on the SVAMP mean, about 6% -- and nothing
    in the artifacts can tell that apart from a real difference.
    """
    def kind(side):
        # A Hub id is owner/name and nothing else. Counting slashes called "./my-qwen3" a Hub
        # id, which is the very spelling the README hands out for a local checkpoint.
        return "a Hub id" if HUB_ID.fullmatch(side["model"]) else "a local path"
    if kind(comparison["base"]) != kind(comparison["candidate"]):
        return ("\nThese runs referenced their weights differently, one by Hub id and one\n"
                "by local path. That alone shifts generation, so some of the difference\n"
                "below is loading rather than training. Score both the same way.")
    return ""


def report(comparison):
    lines = [f"base       {name_of(comparison['base'])}",
             f"candidate  {name_of(comparison['candidate'])}"]
    for name, result in sorted(comparison["results"].items()):
        lines.append(f"\n{name}  ({result['intervals']['prompts']} prompts "
                     f"x {result['samples']} samples)")
        lines += token_line(result) + accuracy_lines(result)
    lines += average_lines(comparison["average"])
    lines.append(ungraded_note(comparison.get("ungraded_benchmarks")))
    lines.append(leaderboard_note(comparison["full_evaluation"]))
    lines.append(loading_note(comparison))
    return "\n".join(lines).rstrip()


def ungraded_note(ungraded):
    """Name each benchmark left out because a grader could not run, or "" when none was."""
    if not ungraded:
        return ""
    lines = [f"    - {name}: no graded rows in the {' and '.join(detail['runs'])} run "
             f"({detail['rows']} rows)" for name, detail in sorted(ungraded.items())]
    return ("\nNOT COMPARED: a grader could not run on these benchmarks, so they are left out\n"
            "of the table and of the average above.\n" + "\n".join(lines)
            + "\n  Re-grade the stored generations with: "
              "python evaluate.py --out <run directory> --regrade")


def leaderboard_note(problems):
    """A warning naming each run that is not the full evaluation, or "" when both are."""
    lines = [f"  {side}\n" + "\n".join(f"    - {reason}" for reason in why)
             for side, why in problems.items() if why]
    if not lines:
        return ""
    return ("\nNOT SUITABLE FOR THE LEADERBOARD: this is not the full evaluation.\n"
            + "\n".join(lines))


def average_lines(mean):
    if not mean:
        return []
    names = ", ".join(mean["benchmarks"])
    note = "equal weights; tokens saved is a geometric mean" + (
        "; BFCL subsets pooled as one" if "bfcl" in mean["benchmarks"] else "")
    return [f"\naverage over {len(mean['benchmarks'])} benchmarks  ({names})", f"  {note}"
            ] + token_line(mean) + accuracy_lines(mean)


def name_of(side):
    return side["model"] + (f" + {side['adapter']}" if side["adapter"] else "")


def token_line(result):
    bounds = result["intervals"]["token_reduction"]
    if not bounds:
        return []
    return [f"  {'tokens saved':<17}{result['token_reduction']:+7.1%}    "
            f"CI [{bounds[0]:+.1%}, {bounds[1]:+.1%}]"]


def accuracy_lines(result):
    bounds = result["intervals"]["accuracy_delta"]
    if not bounds:
        return []
    return [f"  {'accuracy change':<17}{result['accuracy_delta'] * 100:+6.2f} pp  "
            f"CI [{bounds[0] * 100:+.2f} pp, {bounds[1] * 100:+.2f} pp]"]


