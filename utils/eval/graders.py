"""Scores a model's reply against each benchmark's own checker.

Nothing here judges quality; every benchmark has an objective answer. SVAMP
compares numbers, BFCL runs the leaderboard's AST checker, IFEval runs Google's
verifiers, and MBPP+ runs EvalPlus's `sanitize` then executes the code (see utils/eval/sandbox.py).

    from utils.eval.graders import grade_numeric, split_thinking

    final, reasoning, error = split_thinking(token_ids, open_id, close_id, eos_ids)
    grade_numeric("<think>...</think>\nFinal answer: 42", "42", truncated=False)  # (True, None)

Each grader returns (correct, error). `error` names why extraction failed and is
None when the reply was read successfully but simply got the wrong answer.

utils/train/rewards.py grades the *training* data and is deliberately more forgiving
than this file: it accepts \boxed{42}, $42, 84/2 and <answer>42</answer>, which
grade_numeric rejects. That is not an oversight in either -- the two prompts ask
for different things, and each grader matches its own prompt. It does mean a
reward of 1.0 during training is not a guarantee of a point here, and on the
measured baseline 47% of SVAMP replies fall in that gap.
"""

import ast
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace

FINAL_ANSWER = re.compile(r"^Final answer:[ \t]*([^\n]+)[ \t]*$", re.MULTILINE)
FINAL_ANSWER_LINE = re.compile(r"Final answer:[ \t]*[^\n]+")
PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?")
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)
OPEN_CODE_BLOCK = re.compile(r"^```(?:python|py)?[ \t]*\r?\n(.*)\Z", re.S | re.I | re.M)
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def split_thinking(token_ids, open_id, close_id, eos_ids):
    """Token ids -> (final answer ids, reasoning token count, error).

    Qwen3's chat template stops at the assistant header without opening a
    <think> tag, so the model emits it. Anything other than one balanced pair at
    the start means the reply has no well-defined final answer.
    """
    body = [token for token in token_ids if token not in eos_ids]
    opens = [i for i, token in enumerate(body) if token == open_id]
    closes = [i for i, token in enumerate(body) if token == close_id]
    if not opens and not closes:
        return body, 0, None
    if len(opens) != 1 or len(closes) != 1 or opens[0] != 0 or closes[0] < opens[0]:
        return [], len(body), "unbalanced_thinking"
    return body[closes[0] + 1:], closes[0] + 1, None


def final_answer(reply):
    """Text after the thinking block -> the last 'Final answer:' value, or (None, error).

    Repeating the line is allowed so long as it is also the last line.
    """
    reply = reply.strip()
    opens, closes = reply.count(THINK_OPEN), reply.count(THINK_CLOSE)
    if opens or closes:
        if opens != 1 or closes != 1 or not reply.startswith(THINK_OPEN):
            return None, "malformed_thinking_block"
        reply = reply.split(THINK_CLOSE, 1)[1]
    reply = reply.strip()
    matches = FINAL_ANSWER.findall(reply)
    if not matches:
        return None, "missing_final_answer"
    if not FINAL_ANSWER_LINE.fullmatch(reply.splitlines()[-1]):
        return None, "trailing_content"
    return matches[-1].strip(), None


def grade_numeric(reply, answer, truncated):
    """SVAMP: exact decimal equality. No unit stripping and no hunting for a number."""
    if truncated:
        return False, None
    predicted, error = final_answer(reply)
    if error:
        return False, error
    gold = to_number(answer)
    if gold is None:
        raise ValueError(f"Gold answer is not a plain number: {answer!r}")
    value = to_number(predicted)
    if value is None:
        return False, "invalid_numeric_answer"
    return value == gold, None


def to_number(text):
    """'1,234.5' -> Decimal; anything with units, LaTeX or prose -> None."""
    text = text.strip()
    if not PLAIN_NUMBER.fullmatch(text):
        return None
    return Decimal(text.replace(",", ""))


def extract_code(reply, allow_unclosed=False):
    """The single fenced Python block, or the whole reply when unfenced."""
    reply = reply.strip()
    if "```" not in reply:
        return reply, None if reply else "empty_code"
    if allow_unclosed and reply.count("```") == 1:
        open_block = OPEN_CODE_BLOCK.search(reply)
        if open_block:
            return open_block.group(1), None
    blocks = CODE_BLOCK.findall(reply)
    if len(blocks) != 1 or reply.count("```") != 2:
        return "", "ambiguous_code_fences"
    return blocks[0], None


def load_sanitize(evalplus):
    """Import EvalPlus's own `sanitize` from the vendored checkout."""
    with importable(Path(evalplus)):
        from evalplus.sanitize import sanitize
    return sanitize


def mbpp_code(reply, entry_point, sanitize, truncated):
    """MBPP+: reply -> (code, error), reading the reply as EvalPlus does.

    `sanitize` keeps the longest valid Python span and only what the entry
    point depends on, so prose, several fences or leftover tests do not fail a
    correct answer. It cannot repair a renamed function. A reply cut off by the
    token cap is always wrong.
    """
    if truncated:
        return "", "truncated"
    code = sanitize(reply, entry_point)
    return (code, None) if code.strip() else ("", "empty_code")


def decode_calls(reply):
    """BFCL: '[fn(city="Paris")]' -> [{'fn': {'city': 'Paris'}}]. Raises on anything else.

    This parses the reply, it never runs it. ast.parse builds a tree without
    evaluating, and literal_eval accepts only literals, so a reply containing
    os.system(...) fails to decode rather than executing. MBPP+ is the opposite
    case -- its answers only mean anything once run -- which is why that one
    needs utils/eval/sandbox.py and this one does not.
    """
    reply = reply.strip()
    if reply.startswith("```"):
        reply, error = extract_code(reply)
        if error:
            raise ValueError(error)
    body = ast.parse(reply, mode="eval").body
    calls = body.elts if isinstance(body, ast.List) else [body]
    decoded = []
    for call in calls:
        if not isinstance(call, ast.Call) or call.args:
            raise ValueError("Expected calls with named literal arguments")
        decoded.append({dotted_name(call.func): literal_arguments(call)})
    return decoded


def dotted_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise ValueError("Invalid function name")
    parts.append(node.id)
    return ".".join(reversed(parts))


def literal_arguments(call):
    arguments = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in arguments:
            raise ValueError("Expanded or duplicate argument")
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return arguments


def grade_call(row, reply, checker, language, truncated=False):
    """BFCL: the leaderboard's own AST checker.

    Irrelevance follows BFCL's rule: any reply that is not a parseable function
    call (prose, an empty list, or a malformed call) counts as no call, so it is
    correct; only a parseable call is wrong. A reply cut off by the token cap is always wrong.
    """
    if truncated:
        return False, "truncated"
    try:
        calls = decode_calls(reply)
    except (ValueError, SyntaxError, TypeError, RecursionError) as error:
        if row["category"] == "irrelevance":
            return True, None
        return False, f"{type(error).__name__}: {error}"
    if row["category"] == "irrelevance":
        return calls == [], None
    checked = checker(row["function"], calls, row["reference"], language, row["category"], "local")
    return bool(checked["valid"]), None


@contextmanager
def importable(directory):
    """Put `directory` on sys.path for one import, then take it back off.

    Both vendored graders import themselves by name, so their parent has to be
    importable while they load. Leaving it there afterwards would let any later
    import in the process resolve against a benchmark checkout, which is a
    surprise nobody debugging an eval should have to find.
    """
    directory = str(directory)
    sys.path.insert(0, directory)
    try:
        yield
    finally:
        try:
            sys.path.remove(directory)
        except ValueError:  # something else already took it off
            pass


def load_bfcl(root):
    """Import the upstream AST checker, dropping only its provider-config import.

    That import pulls in every API client. Our calls keep their dotted names, so
    no provider-specific dot-to-underscore conversion applies.
    """
    root = Path(root)
    source = root / "bfcl_eval/eval_checker/ast_eval/ast_checker.py"
    tree = ast.parse(source.read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "bfcl_eval.constants.model_config"]
    if len(imports) != 1 or [(a.name, a.asname) for a in imports[0].names] != [("MODEL_CONFIG_MAPPING", None)]:
        raise ValueError("Unexpected BFCL import contract")
    tree.body.remove(imports[0])
    for node in tree.body:
        uses_config = any(isinstance(n, ast.Name) and n.id == "MODEL_CONFIG_MAPPING"
                          for n in ast.walk(node))
        if uses_config and not (isinstance(node, ast.FunctionDef) and node.name == "convert_func_name"):
            raise ValueError("BFCL configuration used outside name conversion")
    namespace = {"MODEL_CONFIG_MAPPING": {"local": SimpleNamespace(underscore_to_dot=False)}}
    with importable(root):
        exec(compile(tree, str(source), "exec"), namespace)
        from bfcl_eval.constants.enums import Language
    return namespace["ast_checker"], Language.PYTHON


def load_ifeval(official):
    """Return a grade(row, reply) using Google's strict verifiers on the final answer only.

    The upstream package imports itself by name, so its directory has to keep
    the name instruction_following_eval and its parent goes on the path.
    """
    # langdetect samples internally and gives different answers run to run unless
    # it is seeded. Several verifiers detect language, so without this the same
    # stored generations grade to a different score every time.
    import langdetect
    langdetect.DetectorFactory.seed = 0
    with importable(Path(official).parent):
        from instruction_following_eval import instructions_registry

    def grade(row, reply):
        if not reply.strip():
            return False, None
        for index, instruction_id in enumerate(row["instruction_id_list"]):
            # build_description draws a random value for any argument the dataset
            # leaves unset, so the same reply grades differently run to run unless
            # the draw is pinned. Seeding per question keeps it stable and keeps
            # questions independent of each other's order.
            random.seed(f"{row['id']}:{instruction_id}")
            instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
            arguments = {k: v for k, v in (row["kwargs"][index] or {}).items() if v is not None}
            instruction.build_description(**arguments)
            wants_prompt = (instruction.get_instruction_args() or {}) if hasattr(
                instruction, "get_instruction_args") else {}
            if "prompt" in wants_prompt:
                instruction.build_description(prompt=row["original_prompt"])
            if not instruction.check_following(reply):
                return False, None
        return True, None

    return grade
