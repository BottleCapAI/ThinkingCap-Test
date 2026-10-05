"""Checks if the model answered correctly on the suggested datasets.

The model has to end its reply with a line "Final answer: <answer>"; its <think> block is ignored.
Pass the model's replies, the correct answers and the task of each question ("math" for
GSM8K / Skywork, "multiple_choice" for CommonsenseQA / ARC Easy), all as lists of the same length:

    from utils.train.rewards import correctness

    replies = ["<think>...</think>\n\nFinal answer: 42", "<think>...</think>\n\nFinal answer: C"]
    correctness(replies, answer=["42", "B"], task=["math", "multiple_choice"])  # [1.0, 0.0]

For a single reply, is_task_correct(reply, "42", "math") returns True or False.

This is more forgiving than utils/eval/graders.py, which scores the benchmarks: it
accepts \boxed{42} and friends where the evaluation wants a bare number. Each
matches the prompt it was written for, so a high reward here does not by itself
predict a high benchmark score -- check how your model formats its answers
before reading too much into either number.
"""

import re
from fractions import Fraction

FINAL_ANSWER = re.compile(r"^Final answer:[ \t]*([^\n]+?)[ \t]*$", re.MULTILINE)
CHOICE_LABEL = re.compile(r"\(?([A-Za-z0-9])\b\)?")
ANSWER_TAG = re.compile(r"</?answer>")
LATEX_WRAPPER = re.compile(r"\\(?:boxed|text)\{(.*)\}")
DEGREES = re.compile(r"(?:°|\^\\circ|\^\{\\circ\})$")


def correctness(completions, answer, task, **kwargs):
    """Batch of completions -> list of 1.0 (correct) / 0.0 (wrong); usable as a reward function.

    A completion is a string or, as trainers pass it for chat prompts, a list of messages.
    """
    return [float(is_task_correct(text_of(completion), gold, kind))
            for completion, gold, kind in zip(completions, answer, task)]


def text_of(completion):
    return completion if isinstance(completion, str) else completion[-1]["content"]


def is_task_correct(completion, answer, task):
    """One completion -> True if its final answer matches; task is "math" or "multiple_choice"."""
    response = completion.split("</think>")[-1]
    predicted = final_answer(response.replace("**", ""))
    if predicted is None:
        return False
    predicted = clean_answer(predicted)
    if task == "math":
        return same_number(predicted, answer)
    if task == "multiple_choice":
        return same_choice(predicted, answer)
    raise ValueError(f"Unknown task: {task!r}")


def final_answer(response):
    response = response.strip()
    if "<think>" in response or "</think>" in response:
        return None
    matches = FINAL_ANSWER.findall(response)
    if len(matches) != 1 or not response.splitlines()[-1].startswith("Final answer:"):
        return None
    return matches[0].strip()


def clean_answer(text):
    text = ANSWER_TAG.sub("", text)
    previous = None
    while text != previous:
        previous = text
        latex = LATEX_WRAPPER.fullmatch(text)
        text = latex.group(1) if latex else text
        text = text.strip(" *$<>").rstrip(".")
    return text


def same_number(predicted, answer):
    predicted_value, answer_value = to_number(predicted), to_number(answer)
    if predicted_value is None or answer_value is None:
        return normalize(predicted) == normalize(answer)
    return predicted_value == answer_value


def to_number(text):
    text = DEGREES.sub("", text.replace(",", "").replace("$", "").strip())
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def same_choice(predicted, answer):
    label = CHOICE_LABEL.match(predicted)
    return label is not None and label.group(1).upper() == answer.strip().upper()


def normalize(text):
    return " ".join(text.casefold().split())
