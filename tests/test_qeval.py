"""Everything here runs on a CPU in under a second.

    python -m unittest discover -s tests -t .
"""

import json
from pathlib import Path
import tempfile
import unittest

from evaluate import (BENCHMARKS, DEFAULT_MAX_TOKENS, MAX_TOKENS, caps_in_use, collect,
                      load_suite, report, select, token_cap)
from utils.eval import benchmarks, sandbox
from utils.eval.compare import average, average_group, check, full_evaluation_problems, pair
from utils.eval.recipe import SAMPLES, SAMPLING, SEED
from utils.eval.graders import (decode_calls, extract_code, load_sanitize, mbpp_code, final_answer, grade_numeric,
                           grade_call, importable, split_thinking)
from utils.eval.stats import confidence, floor, summarize


def rows(pattern, tokens):
    """Two samples per prompt, so ids repeat and clusters are real."""
    return [{"id": f"p{i // 2}", "dataset": "d", "sample_index": i % 2, "correct": bool(c),
             "truncated": False, "output_tokens": tokens, "reasoning_tokens": 0}
            for i, c in enumerate(pattern)]


class Thinking(unittest.TestCase):
    def test_balanced_pair_splits_the_final_answer(self):
        self.assertEqual(split_thinking([1, 7, 8, 2, 9], 1, 2, {9}), ([], 4, None))

    def test_close_without_open_is_a_boundary_error(self):
        # Qwen3.5 injects the opening tag, so replies arrive with only </think>.
        final, _, error = split_thinking([7, 2, 8], 1, 2, set())
        self.assertEqual((final, error), ([], "unbalanced_thinking"))

    def test_a_reply_with_no_markers_is_graded_whole(self):
        self.assertEqual(split_thinking([5, 6], 1, 2, set()), ([5, 6], 0, None))

    def test_repeated_blocks_are_rejected(self):
        self.assertEqual(split_thinking([1, 2, 1, 2], 1, 2, set())[2], "unbalanced_thinking")


class Answers(unittest.TestCase):
    def test_exact_number_matches(self):
        self.assertEqual(grade_numeric("<think>w</think>\nFinal answer: 42", "42", False),
                         (True, None))

    def test_truncated_is_never_correct(self):
        self.assertFalse(grade_numeric("<think>w</think>\nFinal answer: 42", "42", True)[0])

    def test_irrelevance_counts_any_reply_that_is_not_a_call_as_correct(self):
        row = {"category": "irrelevance"}
        for reply in ("No function fits this request.", "[]", "[f(x=y)]"):
            self.assertEqual(grade_call(row, reply, None, None), (True, None), reply)

    def test_irrelevance_with_a_parseable_call_is_wrong(self):
        self.assertFalse(grade_call({"category": "irrelevance"}, '[f(x=1)]', None, None)[0])

    def test_truncated_irrelevance_is_never_correct(self):
        row = {"category": "irrelevance"}
        self.assertFalse(grade_call(row, "No function fits", None, None, truncated=True)[0])

    def test_units_are_not_stripped(self):
        self.assertEqual(grade_numeric("<think>w</think>\nFinal answer: 42 apples", "42", False),
                         (False, "invalid_numeric_answer"))

    def test_a_wrong_number_is_not_a_parse_failure(self):
        # It was read correctly and is simply wrong; calling that unreadable
        # made SVAMP look far more broken than it is.
        self.assertEqual(grade_numeric("<think>w</think>\nFinal answer: 7", "42", False),
                         (False, None))

    def test_trailing_prose_is_rejected(self):
        self.assertEqual(final_answer("<think>w</think>\nFinal answer: 42\nThanks!")[1],
                         "trailing_content")

    def test_a_repeated_final_answer_line_counts_the_last_one(self):
        self.assertEqual(final_answer("Final answer: 4\nso 6 - 2 = 4.\nFinal answer: 4"), ("4", None))
        self.assertEqual(final_answer("Final answer: 5\nwait.\nFinal answer: 4"), ("4", None))

    def test_a_repeated_final_answer_still_has_to_be_the_last_line(self):
        self.assertEqual(final_answer("Final answer: 4\nso.\nFinal answer: 4\nThanks!")[1],
                         "trailing_content")

    def test_one_fenced_block_is_the_code(self):
        self.assertEqual(extract_code("text\n```python\nx = 1\n```"), ("x = 1\n", None))

    def test_two_blocks_are_ambiguous(self):
        self.assertEqual(extract_code("```\na\n```\n```\nb\n```")[1], "ambiguous_code_fences")


class Parsing(unittest.TestCase):
    """Reading calls must never run them, and must not leave sys.path dirty."""

    def test_calls_decode_to_names_and_literals(self):
        self.assertEqual(decode_calls('[fn(city="Paris", count=2)]'),
                         [{"fn": {"city": "Paris", "count": 2}}])

    def test_a_call_inside_an_argument_is_refused(self):
        # literal_eval accepts literals only, so this cannot execute.
        with self.assertRaises(ValueError):
            decode_calls('[fn(path=os.system("rm -rf /"))]')

    def test_a_bare_statement_is_refused(self):
        with self.assertRaises((ValueError, SyntaxError)):
            decode_calls('import os; os.system("echo hi")')

    def test_positional_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            decode_calls('[fn("Paris")]')

    def test_the_import_path_is_handed_back(self):
        import sys
        before = list(sys.path)
        with importable("/tmp/some-grader"):
            self.assertIn("/tmp/some-grader", sys.path)
        self.assertEqual(sys.path, before)

    def test_the_path_is_handed_back_after_a_failed_import(self):
        import sys
        before = list(sys.path)
        with self.assertRaises(RuntimeError):
            with importable("/tmp/some-grader"):
                raise RuntimeError("import blew up")
        self.assertEqual(sys.path, before)


class Selection(unittest.TestCase):
    COHORT = [{"dataset": d} for d in
              ("mbpp", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance", "svamp", "ifeval")]

    def names(self, wanted):
        return [row["dataset"] for row in select(self.COHORT, wanted)]

    def test_every_benchmark_selects_something(self):
        # A benchmark that matches nothing drops out of the default run with no
        # error at all, which is how SVAMP once went missing.
        for name in BENCHMARKS:
            self.assertTrue(self.names([name]), name)

    def test_svamp_is_selectable_by_its_own_name(self):
        self.assertEqual(self.names(["svamp"]), ["svamp"])

    def test_bfcl_selects_all_three_datasets(self):
        self.assertEqual(len(self.names(["bfcl"])), 3)

    def test_the_default_covers_the_whole_cohort(self):
        self.assertEqual(len(self.names(list(BENCHMARKS))), len(self.COHORT))

    def test_an_unknown_benchmark_is_rejected(self):
        with self.assertRaises(ValueError):
            select(self.COHORT, ["numeric"])


class Intervals(unittest.TestCase):
    def test_an_interval_brackets_its_own_estimate(self):
        measured = rows([1, 1, 0, 1, 0, 0, 1, 1], 100)
        bounds = confidence(measured, resamples=500)["accuracy"]
        self.assertLessEqual(bounds[0], summarize(measured)["accuracy"])
        self.assertGreaterEqual(bounds[1], summarize(measured)["accuracy"])

    def test_more_prompts_narrow_the_interval(self):
        def width(repeats):
            bounds = confidence(rows([1, 1, 0, 0] * repeats, 100), resamples=500)["accuracy"]
            return bounds[1] - bounds[0]
        self.assertLess(width(40), width(5))

    def test_identical_prompts_leave_nothing_to_resample(self):
        bounds = confidence(rows([1, 0] * 20, 100), resamples=500)["accuracy"]
        self.assertEqual(bounds[0], bounds[1])

    def test_clusters_are_prompts_not_rows(self):
        self.assertEqual(confidence(rows([1] * 8, 100), resamples=200)["prompts"], 4)

    def test_ungraded_rows_are_excluded_but_counted(self):
        measured = rows([1, 1, 0, 0], 100)
        measured[0]["correct"] = None
        self.assertEqual(summarize(measured)["ungraded"], 1)

class Floor(unittest.TestCase):
    @staticmethod
    def paired(per_question):
        """Rows for runs whose per-sample difference (candidate - base) is given per question."""
        base, cand = [], []
        for i, diffs in enumerate(per_question):
            for d in diffs:
                base.append({"id": i, "correct": d < 0})
                cand.append({"id": i, "correct": d > 0})
        return base, cand

    def test_sampling_noise_inside_questions_is_not_a_floor(self):
        self.assertEqual(floor(*self.paired([[1, 0], [0, 0], [1, 0], [0, 0]]))["half_width_pp"], 0)

    def test_differences_between_questions_are_the_floor(self):
        result = floor(*self.paired([[1, 1], [-1, -1], [1, 1], [-1, -1]]))
        self.assertAlmostEqual(result["half_width_pp"], 1.96 * (4 / 3 / 4) ** 0.5 * 100)

    def test_one_sample_per_question_cannot_separate_the_two(self):
        self.assertIsNone(floor(*self.paired([[1], [0], [-1]])))


class BatchGrading(unittest.TestCase):
    def items(self, count, tasks=3):
        return [(i, f"Mbpp/{i % tasks}", "def f(): pass") for i in range(count)]

    def grade(self, fake, count=12, workers=4):
        original = sandbox.run_batch
        sandbox.run_batch = fake
        try:
            return sandbox.grade_programs("rlimit", "/e", "/m", self.items(count), workers)
        finally:
            sandbox.run_batch = original

    def passing(self, mode, evalplus, mbpp, items, timeout):
        return {key: {"id": task, "base": "pass", "plus": "pass"} for key, task, _ in items}

    def failing(self, *args, **kwargs):
        raise RuntimeError("worker died")

    def test_every_item_gets_a_verdict(self):
        verdicts = self.grade(self.passing)
        self.assertEqual(len(verdicts), 12)
        self.assertTrue(all(v["base"] == "pass" for v in verdicts.values()))

    def test_a_dead_worker_does_not_lose_items(self):
        # Its rows come back marked ungradeable rather than vanishing, so a
        # crash cannot quietly shrink the denominator.
        verdicts = self.grade(self.failing)
        self.assertEqual(len(verdicts), 12)
        self.assertTrue(all(v["error"] for v in verdicts.values()))

    def test_empty_input_needs_no_workers(self):
        self.assertEqual(sandbox.grade_programs("rlimit", "/e", "/m", []), {})

    def test_samples_of_one_task_share_a_worker(self):
        # Keeping a task together is what lets its oracle run once, not nine times.
        chunks = sandbox.split_by_task(self.items(30, tasks=5), workers=4)
        for task in {item[1] for item in self.items(30, tasks=5)}:
            holding = [c for c in chunks if any(i[1] == task for i in c)]
            self.assertEqual(len(holding), 1, task)


class Sandbox(unittest.TestCase):
    def test_requesting_bwrap_where_it_cannot_run_raises(self):
        original = sandbox.have_bwrap
        sandbox.have_bwrap = lambda: False
        try:
            with self.assertRaises(RuntimeError):
                sandbox.choose_mode("bwrap")
            self.assertEqual(sandbox.choose_mode(None), ("rlimit", "auto_downgrade"))
        finally:
            sandbox.have_bwrap = original

    def test_explicit_modes_are_honoured(self):
        self.assertEqual(sandbox.choose_mode("off")[0], "off")
        self.assertEqual(sandbox.choose_mode("rlimit")[0], "rlimit")

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            sandbox.choose_mode("yolo")


class Guards(unittest.TestCase):
    """compare.py refuses runs that are not a controlled comparison."""

    META = {"selection": "full", "cohort_sha256": "abc123def456", "seed": 9001,
            "samples": 8, "max_tokens": {"mbpp": 8192, "svamp": 4096},
            "model": "Qwen/Qwen3-0.6B", "adapter": None, "benchmarks": ["svamp"]}

    def meta(self, **changes):
        return dict(self.META, **changes)

    def refusal(self, **changes):
        with self.assertRaises(ValueError) as caught:
            check(self.meta(), self.meta(**changes))
        return str(caught.exception)

    def test_compatible_runs_pass(self):
        self.assertEqual(check(self.meta(), self.meta(adapter="/my-lora")), ["svamp"])

    def test_the_same_checkpoint_twice_is_refused(self):
        self.assertIn("nothing to compare", self.refusal())

    def test_every_mismatch_is_reported_at_once(self):
        # Reporting one at a time means fix, rerun for hours, find the next.
        message = self.refusal(seed=42, samples=3, adapter="/a")
        self.assertIn("seed", message)
        self.assertIn("samples", message)

    def test_a_mismatch_says_why_it_matters(self):
        self.assertIn("draw different generations", self.refusal(seed=42, adapter="/a"))

    def test_a_differing_budget_names_the_benchmark(self):
        # Printing two dicts would leave the reader to diff them by eye.
        message = self.refusal(max_tokens={"mbpp": 4096, "svamp": 4096}, adapter="/a")
        self.assertIn("mbpp 8192 vs 4096", message)
        self.assertNotIn("svamp 4096 vs 4096", message)

    def test_it_says_what_still_matched(self):
        self.assertIn("the questions do match", self.refusal(samples=3, adapter="/a"))


class TokenCaps(unittest.TestCase):
    QUESTIONS = [{"dataset": d} for d in
                 ("mbpp", "svamp", "bfcl_simple_python", "bfcl_multiple", "ifeval")]

    def arguments(self, max_tokens=None):
        import argparse
        return argparse.Namespace(max_tokens=max_tokens)

    def test_mbpp_gets_more_room_than_the_rest(self):
        # A truncated program never passes, so this one needs the headroom.
        caps = caps_in_use(self.QUESTIONS, self.arguments())
        self.assertEqual(caps["mbpp"], MAX_TOKENS["mbpp"])
        self.assertGreater(caps["mbpp"], caps["svamp"])
        self.assertTrue(all(c == DEFAULT_MAX_TOKENS for d, c in caps.items() if d != "mbpp"))

    def test_the_flag_overrides_every_benchmark(self):
        caps = caps_in_use(self.QUESTIONS, self.arguments(1024))
        self.assertEqual(set(caps.values()), {1024})

    def test_an_unlisted_benchmark_takes_the_default(self):
        self.assertEqual(token_cap("something_new", self.arguments()), DEFAULT_MAX_TOKENS)


class ManifestCheck(unittest.TestCase):
    """An edited question must stop the run, not score quietly."""

    def built(self, text="{\"id\": \"q1\"}\n"):
        from utils.eval.benchmarks import sha256
        root = Path(tempfile.mkdtemp())
        (root / "questions.jsonl").write_text(text)
        (root / "smoke.jsonl").write_text(text)
        (root / "manifest.json").write_text(json.dumps({
            "total": 1, "smoke_total": 1,
            "cohort_sha256": sha256(root / "questions.jsonl"),
            "smoke_sha256": sha256(root / "smoke.jsonl")}))
        return root

    def test_an_untouched_selection_loads(self):
        self.assertEqual(load_suite(self.built())["selection"], "full")

    def test_smoke_reads_the_other_list(self):
        self.assertEqual(load_suite(self.built(), smoke=True)["selection"], "smoke")

    def test_a_smoke_run_checks_the_smoke_list(self):
        # Each list is verified against its own hash, so editing one cannot be
        # hidden by running the other.
        root = self.built()
        (root / "smoke.jsonl").write_text("{\"id\": \"edited\"}\n")
        load_suite(root)                       # the full list is untouched
        with self.assertRaises(SystemExit):
            load_suite(root, smoke=True)

    def test_an_edited_question_is_refused(self):
        root = self.built()
        (root / "questions.jsonl").write_text("{\"id\": \"q1\", \"edited\": true}\n")
        with self.assertRaises(SystemExit) as caught:
            load_suite(root)
        self.assertIn("does not match the manifest", str(caught.exception))


class Cohort(unittest.TestCase):
    RESULTS = {"svamp": {"n": 15, "accuracy": 0.5, "mean_output_tokens": 100.0,
                         "truncation_rate": 0, "parse_failure_rate": 0, "ungraded": 0,
                         "confidence": {"accuracy": None, "mean_output_tokens": None}}}

    def test_a_smoke_run_reports_no_scores(self):
        text = report(self.RESULTS, {"cohort_sha256": "abc123def456", "total": 30,
                                     "smoke": True, "selection": "smoke"})
        self.assertIn("Smoke run", text)
        self.assertNotIn("50.00%", text)

    def test_a_real_run_names_and_hashes_its_selection(self):
        # The name says which set was meant, the hash which was actually used.
        text = report(self.RESULTS, {"cohort_sha256": "abc123def456", "total": 2759,
                                     "smoke": False, "selection": "full"})
        self.assertIn("50.00%", text)
        self.assertIn("full", text)
        self.assertIn("abc123de", text)
        self.assertIn("2759 prompts", text)

    def test_every_source_is_pinned(self):
        for name, revision in benchmarks.REVISIONS.items():
            self.assertNotIn(revision, ("main", "master", "HEAD"), name)

    def test_release_hashes_are_full_sha256(self):
        for digest in (benchmarks.MBPP_SHA, benchmarks.SVAMP_SHA):
            self.assertEqual(len(digest), 64)

    def test_grader_paths_follow_the_root_they_are_asked_about(self):
        # Recording absolute paths once meant a moved cohort pointed at /tmp.
        here = benchmarks.paths("/a/cohort")
        there = benchmarks.paths("/b/cohort")
        self.assertEqual(str(here["bfcl"]), "/a/cohort/vendor/bfcl")
        self.assertEqual(str(there["bfcl"]), "/b/cohort/vendor/bfcl")
        self.assertEqual(set(here), set(benchmarks.LAYOUT))

    def test_the_manifest_records_no_paths(self):
        # suite.json describes the questions, not the machine that built them.
        self.assertFalse(set(benchmarks.LAYOUT) & {"revisions", "counts", "cohort_sha256"})

    def test_every_cohort_path_is_relative(self):
        # An absolute entry here would survive into a copied cohort and point home.
        for name, relative in benchmarks.LAYOUT.items():
            self.assertFalse(Path(relative).is_absolute(), name)

    def test_whole_number_answers_lose_their_float_tail(self):
        self.assertEqual(benchmarks.whole_number(42.0), "42")
        self.assertEqual(benchmarks.whole_number(42.5), "42.5")

    def test_trailing_zeros_survive(self):
        # An earlier version used rstrip(".0"), which strips every trailing dot
        # and zero: 90.0 became "9" and 350.0 became "35". That corrupted 102
        # of 1,000 SVAMP gold answers and understated the score by ~5 points.
        self.assertEqual(benchmarks.whole_number(90.0), "90")
        self.assertEqual(benchmarks.whole_number(350.0), "350")
        self.assertEqual(benchmarks.whole_number(100.0), "100")


class Average(unittest.TestCase):
    """The headline mean counts BFCL's three subsets as one benchmark."""

    @staticmethod
    def rows(prefix, count, correct, tokens):
        return [{"id": f"{prefix}{i}", "sample_index": 0, "correct": correct(i),
                 "output_tokens": tokens, "truncated": False} for i in range(count)]

    def test_bfcl_subsets_share_one_group(self):
        self.assertEqual({average_group(n) for n in ("bfcl_simple_python", "bfcl_multiple")}, {"bfcl"})
        self.assertEqual(average_group("svamp"), "svamp")

    def test_mean_is_unweighted_across_groups(self):
        big = (self.rows("a", 40, lambda i: True, 100), self.rows("a", 40, lambda i: True, 50))
        small = (self.rows("b", 4, lambda i: True, 100), self.rows("b", 4, lambda i: i < 2, 100))
        mean = average({"big": big, "small": small})
        self.assertAlmostEqual(mean["token_reduction"], 1 - 0.5 ** 0.5)
        self.assertAlmostEqual(mean["accuracy_delta"], -0.25)

    def test_token_ratio_direction_does_not_matter(self):
        a = (self.rows("a", 4, lambda i: True, 100), self.rows("a", 4, lambda i: True, 50))
        b = (self.rows("b", 4, lambda i: True, 100), self.rows("b", 4, lambda i: True, 80))
        forward = average({"a": a, "b": b})["token_reduction"]
        backward = average({"a": a[::-1], "b": b[::-1]})["token_reduction"]
        self.assertAlmostEqual(1 - backward, 1 / (1 - forward))

    def test_a_benchmark_with_no_tokens_has_no_token_average(self):
        a = (self.rows("a", 4, lambda i: True, 100), self.rows("a", 4, lambda i: True, 0))
        b = (self.rows("b", 4, lambda i: True, 100), self.rows("b", 4, lambda i: True, 80))
        self.assertIsNone(average({"a": a, "b": b})["token_reduction"])

    def test_a_single_benchmark_has_no_average(self):
        rows = self.rows("a", 4, lambda i: True, 10)
        self.assertIsNone(average({"only": (rows, rows)}))


class FullEvaluation(unittest.TestCase):
    """A run is leaderboard material only if it covered every question of every benchmark."""

    ROWS = [{"dataset": "svamp", "id": "a"}, {"dataset": "mbpp", "id": "a"}]

    def meta(self, **changes):
        return dict({"selection": "full", "cohort_total": 2, "benchmarks": ["svamp", "mbpp"],
                     "failed_benchmarks": {}, "samples": SAMPLES, "seed": SEED,
                     "sampling": SAMPLING, "max_tokens": {"svamp": 4096, "mbpp": 8192}}, **changes)

    def test_a_complete_run_has_no_problems(self):
        self.assertEqual(full_evaluation_problems(self.meta(), self.ROWS), [])

    def test_the_smoke_selection_is_flagged(self):
        self.assertTrue(full_evaluation_problems(self.meta(selection="smoke"), self.ROWS))

    def test_a_narrowed_benchmark_list_is_flagged(self):
        problems = full_evaluation_problems(self.meta(benchmarks=["svamp"]), self.ROWS[:1])
        self.assertIn("1 of the 2", problems[0])

    def test_a_setting_off_the_recipe_is_flagged(self):
        for change in ({"samples": 3}, {"seed": 1}, {"max_tokens": {"svamp": 2048, "mbpp": 8192}},
                       {"sampling": {"temperature": 1.0}}, {"sampling": None}):
            with self.subTest(change=change):
                self.assertTrue(full_evaluation_problems(self.meta(**change), self.ROWS))

    def test_a_benchmark_that_failed_to_grade_is_flagged(self):
        meta = self.meta(failed_benchmarks={"mbpp": {"rows": 3}})
        self.assertIn("mbpp", full_evaluation_problems(meta, self.ROWS)[0])


class PartialGrading(unittest.TestCase):
    def test_one_ungraded_row_fails_the_benchmark_and_spares_the_others(self):
        rows = lambda dataset, correct: [
            {"dataset": dataset, "id": str(i), "sample_index": 0, "correct": c,
             "output_tokens": 10, "truncated": False, "parse_error": None}
            for i, c in enumerate(correct)]
        questions = [{"dataset": "mbpp"}, {"dataset": "svamp"}]
        results, failed = collect(questions, rows("mbpp", [True, None]) + rows("svamp", [True, False]), 1)
        self.assertEqual(set(failed), {"mbpp"})
        self.assertEqual(set(results), {"svamp"})


class MbppCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vendor = Path(__file__).resolve().parent.parent / "questions/vendor/evalplus"
        if not vendor.exists():
            raise unittest.SkipTest("EvalPlus not downloaded")
        cls.sanitize = staticmethod(load_sanitize(vendor))

    def read(self, reply, truncated=False):
        return mbpp_code(reply, "add", self.sanitize, truncated)

    def test_prose_and_a_second_block_do_not_fail_a_correct_answer(self):
        reply = "Here:\n```python\ndef add(a, b):\n    return a + b\n```\nCheck:\n```python\nprint(add(1, 2))\n```"
        code, error = self.read(reply)
        self.assertIsNone(error)
        self.assertIn("def add", code)

    def test_truncated_reply_is_wrong(self):
        self.assertEqual(self.read("```python\ndef add(a, b):\n    return a + b\n```", True)[1], "truncated")

    def test_empty_reply_is_empty_code(self):
        self.assertEqual(self.read("")[1], "empty_code")


class Prompts(unittest.TestCase):
    def test_mbpp_prompt_names_the_entry_point_the_tests_call(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mbpp.jsonl"
            source.write_text(json.dumps({"task_id": 7, "entry_point": "word_len",
                                          "prompt": '"""\nCheck length.\nassert word_len("a")\n"""\n'}) + "\n")
            (row,) = benchmarks.mbpp(source)
        content = row["messages"][0]["content"]
        self.assertIn("The tests call `word_len` and nothing else", content)
        self.assertTrue(content.endswith("Define `word_len`, spelled exactly like that."))

    def test_numeric_prompt_asks_for_the_line_the_grader_reads(self):
        self.assertIn("'Final answer:'", benchmarks.NUMERIC_INSTRUCTION)

    def test_bfcl_system_prompt_is_upstreams_default(self):
        root = Path(__file__).resolve().parent.parent / "questions/vendor/bfcl"
        if not (root / "bfcl_eval/constants/default_prompts.py").exists():
            self.skipTest("BFCL not downloaded")
        prompt = benchmarks.bfcl_system_prompt(root, [{"name": "f"}])
        self.assertTrue(prompt.startswith("You are an expert in composing functions."))
        self.assertIn("You SHOULD NOT include any other text in the response.", prompt)
        self.assertTrue(prompt.endswith(json.dumps([{"name": "f"}], indent=4) + "\n"))


if __name__ == "__main__":
    unittest.main()
