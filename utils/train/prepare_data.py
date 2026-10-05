"""Download the four suggested datasets and write them in one format to data/.

    python utils/train/prepare_data.py

Every row of data/train.jsonl and data/val.jsonl looks like:

    {"prompt": [{"role": "user", "content": "...question... Put the final answer on the last line, like: Final answer: 42"}],
     "answer": "42", "task": "math", "source": "gsm8k"}

`prompt` is a chat message list ready for tokenizer.apply_chat_template, `answer` and `task`
go straight to utils.train.rewards.correctness, and `source` tells the datasets apart.
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

MATH_INSTRUCTION = "Put the final answer on the last line, like: Final answer: 42"
CHOICE_INSTRUCTION = "Answer with the label of the correct option. Put it on the last line, like: Final answer: B"
VAL_PER_SOURCE = 200


def row(question, instruction, answer, task, source):
    return {"prompt": [{"role": "user", "content": f"{question}\n\n{instruction}"}],
            "answer": answer, "task": task, "source": source}


def with_options(question, choices):
    options = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
    return f"{question}\n\n{options}"


def gsm8k():
    for r in load_dataset("openai/gsm8k", "main", split="train"):
        answer = r["answer"].split("####")[-1].strip().replace(",", "")
        yield row(r["question"], MATH_INSTRUCTION, answer, "math", "gsm8k")


def skywork():
    # The easy band with whole-number answers: DeepSeek-R1-Distill-Qwen-1.5B failed at most 8 of 16 tries.
    for r in load_dataset("Skywork/Skywork-OR1-RL-Data", split="math"):
        gold = json.loads(r["reward_model"]["ground_truth"])
        if len(gold) == 1 and re.fullmatch(r"-?\d+", gold[0].strip()) \
                and r["extra_info"]["model_difficulty"]["DeepSeek-R1-Distill-Qwen-1.5B"] <= 8:
            yield row(r["prompt"][-1]["content"], MATH_INSTRUCTION, gold[0].strip(), "math", "skywork")


def commonsense_qa():
    for r in load_dataset("tau/commonsense_qa", split="train"):
        yield row(with_options(r["question"], r["choices"]), CHOICE_INSTRUCTION, r["answerKey"],
                  "multiple_choice", "commonsense_qa")


def arc_easy():
    for r in load_dataset("allenai/ai2_arc", "ARC-Easy", split="train"):
        yield row(with_options(r["question"], r["choices"]), CHOICE_INSTRUCTION, r["answerKey"],
                  "multiple_choice", "arc_easy")


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{path}: {len(rows)} rows")


def main():
    rng = random.Random(0)
    train, val = [], []
    for load in (gsm8k, skywork, commonsense_qa, arc_easy):
        rows = list(load())
        rng.shuffle(rows)
        val += rows[:VAL_PER_SOURCE]
        train += rows[VAL_PER_SOURCE:]
    rng.shuffle(train)
    out = Path("data")
    out.mkdir(exist_ok=True)
    write(out / "train.jsonl", train)
    write(out / "val.jsonl", val)


if __name__ == "__main__":
    main()
