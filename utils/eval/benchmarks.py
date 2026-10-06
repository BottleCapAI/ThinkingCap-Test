"""Downloads the four benchmarks and freezes them into one list of questions.

    python utils/eval/benchmarks.py

Every line of questions/questions.jsonl looks like:

    {"id": "svamp-1", "dataset": "svamp", "grader": "numeric", "answer": "42",
     "messages": [{"role": "user", "content": "...question..."}], "seed_index": 7}

`messages` is ready for tokenizer.apply_chat_template, `grader` picks the
checker in utils/eval/graders.py, and `seed_index` fixes the sampling seed so two
runs over the same selection draw the same generations and can be compared.

Every source is pinned to an exact revision and hash-checked. Nothing follows a
moving branch, so the selection you build today is the one you built last year.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import urllib.parse
import urllib.request

# Exactly which version of every external source. A benchmark that changes
# under you moves every score you have already recorded, and moves it silently.
REVISIONS = {
    "bfcl": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
    "evalplus": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
    "svamp": "78e727689e1c1bebfc4be39c446898e8e10b0518",
    "ifeval": "4700efb9afa54286b0e04473ba80a13e8461e25f",
    "mbpp": "v0.2.0",
}
# Every URL carries a commit, so GitHub serves those files content-addressed.
# The MBPP+ release is a tag, which a maintainer can re-point at a different
# asset, so that one needs a hash of its own. SVAMP is pinned by commit too;
# its hash is belt and braces after that source was once left on main.
BFCL_CATEGORIES = ("simple_python", "multiple", "irrelevance")
MBPP_SHA = "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"
SVAMP_SHA = "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"
# The files each grader actually loads, found by importing it and reading
# sys.modules. Cloning the repos instead would pull 225 MB to use under 1 MB,
# and a missing file here fails loudly at import rather than skewing a score.
# Everything the questions directory holds, relative to it. One table, so a path
# is written and read from the same string: when download() and the grader each
# spelled out "raw/mbpp.jsonl" separately, either could move without the other.
# Nothing here is recorded in the manifest, so the directory can be copied.
LAYOUT = {
    "questions": "questions.jsonl",
    "smoke": "smoke.jsonl",
    "manifest": "manifest.json",
    "evalplus": "vendor/evalplus",
    "bfcl": "vendor/bfcl",
    "ifeval": "vendor/instruction_following_eval",
    "mbpp": "raw/mbpp.jsonl",
    "svamp": "raw/svamp.json",
}

IFEVAL_FILES = ("instructions.py", "instructions_registry.py", "instructions_util.py",
                "evaluation_lib.py", "data/input_data.jsonl")
EVALPLUS_FILES = (
    "evalplus/__init__.py", "evalplus/codegen.py", "evalplus/config.py",
    "evalplus/data/__init__.py", "evalplus/data/humaneval.py", "evalplus/data/mbpp.py",
    "evalplus/data/utils.py", "evalplus/eval/__init__.py", "evalplus/eval/_special_oracle.py",
    "evalplus/eval/utils.py", "evalplus/evaluate.py", "evalplus/gen/__init__.py",
    "evalplus/gen/util/__init__.py", "evalplus/provider/__init__.py",
    "evalplus/provider/base.py", "evalplus/provider/utility.py", "evalplus/sanitize.py",
    "evalplus/syncheck.py", "evalplus/utils.py",
)
BFCL_FILES = (
    "bfcl_eval/__init__.py", "bfcl_eval/constants/__init__.py", "bfcl_eval/constants/enums.py",
    "bfcl_eval/constants/type_mappings.py", "bfcl_eval/constants/default_prompts.py",
    "bfcl_eval/eval_checker/__init__.py",
    "bfcl_eval/eval_checker/ast_eval/__init__.py",
    "bfcl_eval/eval_checker/ast_eval/ast_checker.py",
    "bfcl_eval/eval_checker/ast_eval/type_convertor/__init__.py",
    "bfcl_eval/eval_checker/ast_eval/type_convertor/java_type_converter.py",
    "bfcl_eval/eval_checker/ast_eval/type_convertor/js_type_converter.py",
) + tuple(f"bfcl_eval/data/BFCL_v4_{c}.json" for c in BFCL_CATEGORIES) \
  + tuple(f"bfcl_eval/data/possible_answer/BFCL_v4_{c}.json"
          for c in BFCL_CATEGORIES if c != "irrelevance")

# How each benchmark's question is worded, and how small a --smoke selection is.
MBPP_INSTRUCTION = "Write a complete Python solution to the following task. Return the solution in one Python code block."
# MBPP+ names the function only inside the task's assert, and the model likes
# to "improve" such names; a name it changes fails every test.
MBPP_NAME_REMINDER = ("The tests call `{name}` and nothing else, so any other function name "
                      "fails. Define `{name}`, spelled exactly like that.")
# These strings are part of the benchmark: changing a word changes the
# selection hash and makes every recorded score incomparable. Reword only with
# a fresh evaluation run, never while tidying.
NUMERIC_INSTRUCTION = ("\n\nThe last line of your reply must start with 'Final answer:' "
                       "followed by the number alone.")
SMOKE_PROMPTS = 5
# A selection is named so people can talk about it and hashed so they can check
# it. The name says which set was meant; the hash says which was actually used,
# and only the hash notices when a prompt is reworded or an answer corrected.
FULL, SMOKE = "full", "smoke"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="questions",
                        help="Directory to download the benchmarks into and write the "
                             "question lists in (default: ./questions)")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    root = Path(arguments.questions).resolve()
    download(root)
    suite = build(root, arguments.seed)
    print(json.dumps(suite["counts"], indent=2))
    print("total", suite["total"], "   smoke", suite["smoke_total"])
    print(f"\nWrote {root}. Evaluate with: python evaluate.py --out runs/base")


def download(root):
    """Fetch every pinned source. Safe to re-run; it verifies whatever already exists."""
    at = paths(root)
    for name in ("evalplus", "mbpp"):
        at[name].parent.mkdir(parents=True, exist_ok=True)
    collect(f"https://raw.githubusercontent.com/evalplus/evalplus/{REVISIONS['evalplus']}/",
            at["evalplus"], EVALPLUS_FILES)
    collect("https://raw.githubusercontent.com/ShishirPatil/gorilla/"
            f"{REVISIONS['bfcl']}/berkeley-function-call-leaderboard/",
            at["bfcl"], BFCL_FILES)
    fetch("https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
          at["mbpp"], MBPP_SHA)
    fetch(f"https://raw.githubusercontent.com/arkilpatel/SVAMP/{REVISIONS['svamp']}/SVAMP.json",
          at["svamp"], SVAMP_SHA)
    fetch_ifeval(at["ifeval"])


def collect(base_url, root, names):
    """Download `names` under `root`, skipping what is already there.

    The URL carries a commit, so GitHub serves these content-addressed: asking
    for the same path twice at the same commit cannot give two different files.
    """
    for name in names:
        target = root / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(base_url + name, timeout=120) as response:
            target.write_bytes(response.read())


def fetch(url, path, expected_sha256):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            body = response.read()
        path.write_bytes(gzip.decompress(body) if url.endswith(".gz") else body)
    if sha256(path) != expected_sha256:
        raise ValueError(f"Hash mismatch for {path}; the pinned source changed")


def fetch_ifeval(official):
    """The verifiers import themselves as instruction_following_eval, so keep that name."""
    for name in IFEVAL_FILES:
        target = official / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            url = ("https://raw.githubusercontent.com/google-research/google-research/"
                   f"{REVISIONS['ifeval']}/instruction_following_eval/{name}")
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
    (official / "__init__.py").touch()
    # Some verifiers tokenize sentences, so punkt is a dependency of the grader,
    # not an optional extra. It goes to nltk's own directory because nltk refuses
    # to write anywhere group-writable, which a shared checkout usually is.
    import nltk
    for package in ("punkt_tab", "punkt"):
        if not nltk.download(package, quiet=True):
            raise RuntimeError(f"Could not download NLTK {package}; IFEval cannot be graded")


def build(root, seed=42):
    """Write both selections -- the full benchmark, and a smoke subset of it.

    Smoke is the first few of each benchmark's shuffle, so it is a subset of
    the full list rather than a different draw, and producing it costs nothing.
    Writing both means one download serves both, and asking for the quick one
    cannot overwrite the real one.
    """
    at = paths(root)
    rng = random.Random(seed)
    chosen, smoke = [], []

    def add(questions):
        questions = list(questions)
        rng.shuffle(questions)
        chosen.extend(questions)
        smoke.extend(questions[:SMOKE_PROMPTS])

    add(mbpp(at["mbpp"]))
    for category in BFCL_CATEGORIES:
        add(bfcl(at["bfcl"], category))
    add(svamp(at["svamp"]))
    add(ifeval(at["ifeval"]))

    # seed_index fixes which generations a question draws, so it is assigned
    # from the full list and smoke inherits it: a question answered in a smoke
    # run gets exactly the generations it would get in a real one.
    for index, question in enumerate(chosen):
        question["seed_index"] = index
    at["questions"].write_text("".join(json.dumps(q) + "\n" for q in chosen))
    at["smoke"].write_text("".join(json.dumps(q) + "\n" for q in smoke))
    counts = {}
    for question in chosen:
        counts[question["dataset"]] = counts.get(question["dataset"], 0) + 1
    suite = {"revisions": REVISIONS, "seed": seed, "counts": counts,
             "total": len(chosen), "smoke_total": len(smoke),
             "smoke_sha256": sha256(at["smoke"]),
             # Kept under the old name so results.json files already written
             # stay readable; it is the hash of questions.jsonl.
             "cohort_sha256": sha256(at["questions"])}
    at["manifest"].write_text(json.dumps(suite, indent=2) + "\n")
    return suite


def mbpp(source):
    for task in read_jsonl(source):
        reminder = MBPP_NAME_REMINDER.format(name=task["entry_point"])
        yield {"id": str(task["task_id"]), "dataset": "mbpp", "grader": "mbpp",
               "entry_point": task["entry_point"],
               "messages": [{"role": "user",
                             "content": f"{MBPP_INSTRUCTION}\n\n{task['prompt']}\n{reminder}"}]}


def bfcl_system_prompt(root, functions):
    """BFCL's own default system prompt for a question's functions.

    Built from the vendored default_prompts.py using the format string upstream
    names as its default, so the wording is theirs and not ours.
    """
    spec = importlib.util.spec_from_file_location(
        "bfcl_default_prompts", root / "bfcl_eval/constants/default_prompts.py")
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)
    options = dict(urllib.parse.parse_qsl(prompts.DEFAULT_SYSTEM_PROMPT_FORMAT))
    if options["tool_call_tag"] != "False" or options["func_doc_fmt"] != "json":
        raise ValueError(f"BFCL's default prompt format changed: {options}")
    style = prompts.PROMPT_STYLE_TEMPLATES[options["style"]]
    call_format = style["tool_call_no_tag"].format(
        output_format=prompts.OUTPUT_FORMAT_MAPPING[options["ret_fmt"]],
        param_types=prompts.PARAM_TYPE_MAPPING[options["ret_fmt"]])
    return prompts.PROMPT_TEMPLATE_MAPPING[options["prompt_fmt"]].format(
        persona=style["persona"], task=style["task"], tool_call_format=call_format,
        multiturn_behavior=style["multiturn_behavior"],
        available_tools=style["available_tools"].format(
            format="json", functions=json.dumps(functions, indent=4)))


def bfcl(root, category):
    data = root / "bfcl_eval/data"
    references = {}
    if category != "irrelevance":
        references = {r["id"]: r["ground_truth"]
                      for r in read_jsonl(data / f"possible_answer/BFCL_v4_{category}.json")}
    for task in read_jsonl(data / f"BFCL_v4_{category}.json"):
        if len(task["question"]) != 1:
            raise ValueError(f"Expected a single-turn BFCL question: {task['id']}")
        system = bfcl_system_prompt(root, task["function"])
        yield {"id": task["id"], "dataset": f"bfcl_{category}", "grader": "bfcl",
               "category": category, "function": task["function"],
               "reference": references.get(task["id"]),
               "messages": [{"role": "system", "content": system}] + task["question"][0]}


def svamp(source):
    for task in json.loads(source.read_text()):
        question = f"{task['Body'].strip()} {task['Question'].strip()}"
        yield {"id": f"svamp-{task['ID']}", "dataset": "svamp", "grader": "numeric",
               "answer": whole_number(task["Answer"]),
               "messages": [{"role": "user", "content": question + NUMERIC_INSTRUCTION}]}


def whole_number(value):
    """SVAMP stores whole answers as floats; '42.0' and '42' must grade the same."""
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def ifeval(official):
    for task in read_jsonl(official / "data/input_data.jsonl"):
        yield {"id": str(task["key"]), "dataset": "ifeval", "grader": "ifeval",
               "instruction_id_list": task["instruction_id_list"], "kwargs": task["kwargs"],
               "original_prompt": task["prompt"],
               "messages": [{"role": "user", "content": task["prompt"]}]}


def paths(root):
    """Every file in the questions directory, resolved against one of them."""
    return {name: Path(root) / relative for name, relative in LAYOUT.items()}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
