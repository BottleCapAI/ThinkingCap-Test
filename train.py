"""A starting point: loads Qwen3-0.6B and the prepared data, generates a reply and scores it.

This only shows how to use our helpers (utils/train/prepare_data.py, utils/train/rewards.py); you don't have
to stick to them. Build your own method from here. Good luck!
"""

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.train.rewards import correctness

TRAIN_DATA = Path("data/train.jsonl")


def load(model_id):
    """Return (model, tokenizer), in bf16 on a GPU and fp32 on CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    return model, tokenizer


def rollout(model, tokenizer, row, max_new_tokens=2048):
    """Generate one reply to a prepared row; return (reply text, number of generated tokens)."""
    inputs = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                            temperature=0.6, top_p=0.95, top_k=20)
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)


def main():
    if not TRAIN_DATA.exists():
        raise SystemExit(f"{TRAIN_DATA} does not exist yet. Build the suggested datasets with:\n"
                         "    python utils/train/prepare_data.py")
    model, tokenizer = load("Qwen/Qwen3-0.6B")
    data = load_dataset("json", data_files=str(TRAIN_DATA), split="train")

    row = data[0]
    reply, length = rollout(model, tokenizer, row)
    reward = correctness([reply], [row["answer"]], [row["task"]])[0]
    print(f"{row['source']}: reward {reward}, {length} tokens")


if __name__ == "__main__":
    main()
