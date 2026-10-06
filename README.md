# ThinkingCap Test

![logo](img/logo.png)
Hey there! You may have heard of our [ThinkingCap](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B) models, which shorten the reasoning of LLMs. Now we're looking for people to help us build what comes next.

This is our second open test for people interested in joining [BottleCapAI](https://www.bottlecapai.com). This time it focuses on post-training.

Are you interested in LLMs? Do you like experimenting with neural networks, implementing different ideas and testing them out? Would you like to do that for a living? Then you're in the right place!

## 📌 About BottleCapAI

At **BottleCapAI**, we’re making large language models **radically more efficient** — aiming for **100× improvements** over today’s approaches. 🚀  

### 👥 Founders
- Tomas Mikolov – creator of *word2vec*, pioneer of neural language models.  
- Jaroslav Beck – co-founder of *Beat Games* (*Beat Saber*, 10M+ copies sold, acquired by Meta).  
- David Herel – creator of Thinking Tokens, co-founder of an AI trading startup, and Amazon Alexa Prize finalist.

### 🌍 Our vision
Training and inference for frontier LLMs costs **tens of millions** today. Our new algorithms already cut that by **~50%**. We’re building a European hub to push AI forward through **algorithms, not brute force**.  
 
📧 **hey(at)bottlecapai.com** · 🌐 [bottlecapai.com](https://www.bottlecapai.com)  

---


## Objective

Make **[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)** think **shorter** without losing accuracy.

Reasoning models are good because they think before they answer, and expensive for the same reason: every token of that thinking has to be generated. Qwen3-0.6B happily spends hundreds of tokens on questions it could answer in a few dozen. Your job is to teach it to get to the point.

**Goal:** generate **fewer tokens** (thinking included) than the original Qwen3-0.6B while keeping the **same accuracy**.

---

## What's the point?

Shortening is easy. Truncate the thinking, penalize every token, and the outputs get short — and usually wrong. The hard part is cutting the reasoning that doesn't matter while keeping the reasoning that does.

We're not here to find the magic sentence that makes the model think less. We're here to explore **algorithmic ideas that can scale**: things that change how the model reasons, rather than how it is prompted. A prompt can be part of your method, for example to generate training data, as long as the shortening ends up in the model itself.

This benchmark is meant for:
- People with limited hardware
- People with ideas and curiosity

You're free to try almost anything, as long as the model ends up thinking shorter (see [Rules](#rules)). For example:
- Fine-tune it with your own objective
- Fine-tune it on data of your choice (not on the evaluation benchmarks; see [Rules](#rules))
- Modify the model itself
- Something nobody tried yet!

You're **not** expected to:
- Rely on prompt engineering
- Turn thinking off

We're interested in your own ideas, not how well you can copy other's. These ideas should be general and work on different setups and not be hardcoded to a very specific one.

---

## Getting started

[`train.py`](train.py) loads the model from Hugging Face and hands it over to you; everything after that is up to you.
```bash
git clone https://github.com/BottleCapAI/ThinkingCap-Test && cd ThinkingCap-Test
curl -LsSf https://astral.sh/uv/install.sh | sh    # skip if you already have uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

python utils/train/prepare_data.py    # writes data/train.jsonl, which train.py reads
python train.py
```

### Suggested datasets

You can train on whatever you like (see the [Rules](#rules)). To make it easier for you to begin, we suggest these four: Qwen3-0.6B solves a good part of them, but not all, so there is signal to learn from.

| Dataset | Domain | Train size |
| --- | --- | --- |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Simple maths | 7,473 |
| [Skywork-OR1](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data), easy band | Maths with headroom | ~28k |
| [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) | Question answering | 9,741 |
| [ARC Easy](https://huggingface.co/datasets/allenai/ai2_arc) | Question answering | 2,251 |

The [`utils/train/`](utils/train/) folder has two helpers for them. Using them is not compulsory, but they save you the boring part.

**[`utils/train/prepare_data.py`](utils/train/prepare_data.py)** downloads all four and writes them in one format to `data/train.jsonl` and `data/val.jsonl` (200 held-out questions per dataset):
```bash
python utils/train/prepare_data.py
```
Every row looks like this:
```json
{"prompt": [{"role": "user", "content": "<question> ... Put the final answer on the last line, like: Final answer: 42"}],
 "answer": "42", "task": "math", "source": "gsm8k"}
```
`prompt` is a chat conversation you can pass straight to the tokenizer's chat template, `answer` and `task` are what the grader needs, and `source` tells you which dataset a row came from, e.g. if you want to balance them. A list of chat messages in a `prompt` column plus extra columns is also what most training libraries expect, so the same file works whichever method you pick.

**[`utils/train/rewards.py`](utils/train/rewards.py)** checks whether an answer is right. Give it the model's replies, the correct answers and the task types, and it returns 1.0 for each right answer and 0.0 for each wrong one. It expects the reply to end with a line `Final answer: <answer>`, which the prepared prompts ask for. It follows TRL's reward-function interface, so it should work with most TRL trainers.

[`train.py`](train.py) shows how they fit together: it loads a prepared question, generates a reply with `rollout()`, and prints the reply's reward and its length in tokens, the two numbers this task is about.

---

## Benchmark evaluation

To evaluate your checkpoint, we prepared four benchmarks, each testing a different ability: math (SVAMP), function calling (BFCL), coding (MBPP+) and instruction following (IFEval).

The evaluation has **2,759 questions** across the four benchmarks. Each question is answered 8 times, so one run produces 22,072 responses. Other sample counts work for quick tests, but only 8 counts as the full evaluation.

| Benchmark | Questions | What the questions ask |
|---|---:|---|
| MBPP+ | 378 | Write a Python function from a one-sentence description, keeping the required name. |
| BFCL | 840 | Answer a request with function calls, from the functions offered. Three subsets: |
| &nbsp;&nbsp;simple_python | 400 | One function is offered, and the request needs exactly one call to it. |
| &nbsp;&nbsp;multiple | 200 | Two to four functions are offered; pick the right one and fill in its arguments. |
| &nbsp;&nbsp;irrelevance | 240 | The offered function does not fit the request, so the correct answer is no call at all. |
| SVAMP | 1,000 | Grade-school arithmetic word problems with a single numeric answer. |
| IFEval | 541 | Write a response that obeys one to three checkable constraints: 25 kinds, in 9 groups such as keywords, length, formatting, casing, language. |
| **Total** | **2,759** | |

The average over benchmarks counts each of the four equally, with the three BFCL
subsets pooled as one. Accuracy change is averaged with the arithmetic mean, tokens saved
with the geometric mean of the per-benchmark token ratios.

### Running the baseline

With the repository cloned and the requirements installed
([Getting started](#getting-started)), scoring the unmodified Qwen3-0.6B takes
two commands:

```bash
python utils/eval/benchmarks.py        # download and freeze the questions
python evaluate.py --out runs/base     # the full baseline
```

The first command downloads the four benchmarks at pinned revisions and stores
them in `questions/questions.jsonl`. The second prints one row per benchmark (the
table above) and writes `runs/base/` with the per-response `records.jsonl` and a
`results.json` that records everything needed to interpret the scores. Score it
once and keep it: every later run is compared against this directory with
`compare.py`.

Before committing to a full run, check the plumbing with `--smoke`. It answers a
30-question subset, grades it exactly as a real run would, reports no scores, and
takes about a minute:

```bash
python evaluate.py --out runs/check --smoke
```

It names any benchmark whose grader could not run. Finding that here costs a
minute; finding it after a full run costs the run.

### Evaluation of your checkpoint

Evaluate your checkpoint on the same questions as the baseline. What you
trained decides which flags to pass:

```bash
# a LoRA on top of the base model
python evaluate.py --out runs/mine --adapter ./my-lora

# your own weights, with or without an adapter of their own
python evaluate.py --out runs/mine --model ./my-qwen3
python evaluate.py --out runs/mine --model ./my-qwen3 --adapter ./my-lora
```

Then compare it with the baseline. `compare.py` pairs each of your responses with
the baseline's response to the same question and reports the token saving and the
accuracy change, each with a confidence interval:

```bash
python compare.py runs/base runs/mine
```

For example, the output for a LoRA adapter (an illustrative run, not a target):

```
base       Qwen/Qwen3-0.6B
candidate  Qwen/Qwen3-0.6B + /home/.../adapter

bfcl_irrelevance  (240 prompts x 8 samples)
  tokens saved       +6.3%    CI [+2.3%, +10.0%]
  accuracy change   +0.00 pp  CI [-0.94 pp, +0.94 pp]

bfcl_multiple  (200 prompts x 8 samples)
  tokens saved       +8.5%    CI [+5.8%, +11.5%]
  accuracy change   +0.06 pp  CI [-1.50 pp, +1.69 pp]

bfcl_simple_python  (400 prompts x 8 samples)
  tokens saved       +6.9%    CI [+5.0%, +8.8%]
  accuracy change   +0.03 pp  CI [-0.78 pp, +0.84 pp]

ifeval  (541 prompts x 8 samples)
  tokens saved       +4.8%    CI [+1.8%, +7.7%]
  accuracy change   -0.46 pp  CI [-1.64 pp, +0.74 pp]

mbpp  (378 prompts x 8 samples)
  tokens saved       +7.8%    CI [+4.6%, +10.8%]
  accuracy change   -1.09 pp  CI [-2.45 pp, +0.40 pp]

svamp  (1000 prompts x 8 samples)
  tokens saved      +19.2%    CI [+17.4%, +20.9%]
  accuracy change   -0.14 pp  CI [-0.85 pp, +0.63 pp]

average over 4 benchmarks  (bfcl, ifeval, mbpp, svamp)
  equal weights; tokens saved is a geometric mean; BFCL subsets pooled as one
  tokens saved       +9.9%    CI [+8.6%, +11.1%]
  accuracy change   -0.42 pp  CI [-0.92 pp, +0.08 pp]
```

`accuracy change` is your accuracy minus the baseline's, in percentage points,
where accuracy is the percentage of answers that are correct. Answers cut off by
the token cap count as wrong. `tokens saved` is the reduction in total tokens
(thinking and answer) relative to the baseline, as a percentage. Each value comes with its
95% confidence interval. The aim is to maximize `tokens saved` while keeping
`accuracy change` close to zero. Exact definitions are in the
[note on the target metric](#comment-on-the-target-metric).

The last entry of the output averages the accuracy change (arithmetic mean) and the tokens
saved (geometric mean of the token ratios) over the benchmarks. It weights the four
benchmarks equally, treating the three BFCL subsets as a single benchmark.

### Running less of it

If you do not have a fast GPU, we recommend not running the full evaluation while you
develop — fewer benchmarks and fewer samples keep the loop short. For your final numbers,
try to run all of it. If that is not possible, submit anyway and let us know what you did
not run; we re-evaluate all submissions ourselves. Whatever you run, score the base model
exactly as you score your checkpoint, or the comparison means nothing, and read the
confidence interval.

### Sources and licenses

| Benchmark | Source | License |
|---|---|---|
| MBPP+ | [EvalPlus MBPP+ release](https://github.com/evalplus/mbppplus_release), built on [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) (Austin et al., Google) | Apache-2.0 (MBPP+), CC-BY-4.0 (MBPP) |
| BFCL | [Gorilla](https://github.com/ShishirPatil/gorilla), Berkeley Function Calling Leaderboard v4 | Apache-2.0 |
| SVAMP | [arkilpatel/SVAMP](https://github.com/arkilpatel/SVAMP) | MIT |
| IFEval | [google-research](https://github.com/google-research/google-research), `instruction_following_eval` | Apache-2.0 |

The benchmarks belong to their authors. `utils/eval/benchmarks.py` downloads them from
the sources above and does not redistribute them. If you
publish results, credit the original benchmarks and follow their terms.

---

## Rules

1. Start from **Qwen3-0.6B**. Full fine-tuning, LoRA, minor architectural changes[^1] and similar are all fine, as long as the result is still a roughly 0.6B-sized model that thinks before it answers.
2. Do **not** train on the evaluation benchmarks.
3. Measure with the provided evaluation and keep its settings (prompt, sampling, token cap) unchanged. Fewer benchmarks, or fp16 where your GPU has no bf16, is fine if your baseline run matches.
4. Document your idea in `IDEA.md` (motivation, method, results). Negative results are welcome—share what you learned!

[^1]: The modified model must load with the standard Hugging Face Transformers and vLLM releases, since `evaluate.py` runs it through both and does not execute custom model code.

---

## Submission

Send us your work. Don't worry if you lost some accuracy, we want to see it anyway.

Please include:

- your code, as a git bundle: `git bundle create <first name>-<last name>.bundle --all`
- the `results.json` that `evaluate.py` wrote to your `--out` directory
- your trained model (or LoRA adapter) in `.safetensors` format, so we can evaluate it ourselves
- a short `IDEA.md` with what you tried and why, what worked, what didn't, and the hardware you used

Does accuracy start to drop as you shorten further? Send us several checkpoints along the way. We're interested in how your algorithm trades length for accuracy, and a few points on that curve tell us more than one.

**How to send it**

If you can, use the [submission page](https://bottlecapai.com/careers/thinkingcap-test-submission/): put everything above in one `.zip` of at most 500 MB, named `<first name>-<last name>.zip`, and upload it.

If it doesn't fit, upload it to [Hugging Face](https://huggingface.co) or [Google Drive](https://drive.google.com) and email the link to hey(at)bottlecapai.com with the subject `<first name>-<last name> <token reduction> <accuracy change>` (use your best checkpoint's numbers).

At this moment, we are interested mainly in candidates willing to relocate to Prague. (If you’re an exceptional fit, we’re happy to discuss possible support options.)

---
## Technical Notes

While this project is designed to run on **1 GPU**, there are a few things to keep in mind:

- GPU memory and Google Colab:
  Qwen3-0.6B is small enough to both generate and train on a free [Google Colab](https://colab.research.google.com/) T4 (16 GB), as long as you are a bit careful with memory. The usual culprit is not the model (~1.2 GB in fp16) but the **logits**: with a 151,936-token vocabulary, one 4,096-token sequence produces ~2.3 GB of fp32 logits, and a naive loss keeps several copies of them alive during the backward pass. A few things that help:
  - **Use LoRA** instead of full fine-tuning. Gradients and optimizer state then exist only for the small adapter, not for all 600M parameters.
  - **Don't keep a second copy of the model.** If you need the original model next to the one you train (e.g. as a reference), switch the LoRA adapter off instead of loading it twice.
  - **Turn on gradient checkpointing** (`model.gradient_checkpointing_enable()`): slower, but activations stop dominating memory.
  - **Keep the loss cheap.** Compute it only on the generated tokens, never on the prompt, and use a chunked or fused cross-entropy (e.g. [Liger Kernel](https://github.com/linkedin/Liger-Kernel)) so the full logits are never materialized at once.
  - **Use small micro-batches** (even 1 sequence) and gradient accumulation for a larger effective batch. Shorter sequences help too: halving the length halves the logits.
  - **Mind the T4's precision:** it has no bf16 support, so use fp16 with the LoRA weights kept in fp32 (mixed precision), or fp32 if fp16 gives you NaNs.

  On Colab, speed and session time run out before memory does: long generations on a T4 are slow and free sessions disconnect, so save checkpoints regularly.

- Evaluating on a T4:
  vLLM will not start there at all: it takes bf16 from the model config and the T4 has
  no bf16. Add `dtype="half"` to the `LLM(...)` call in `load_engine`, and use the same
  setting for your baseline run. The full evaluation also takes about five hours on a T4,
  so if you can, rent an hour on any Ampere-or-newer card — an RTX 3090/4090, A5000, L4
  or RTX PRO 6000 all run bf16 unchanged and finish in one to two hours.

- Sampling:
  Qwen recommends temperature 0.6, top-p 0.95 and top-k 20 in thinking mode. Greedy decoding makes Qwen3 repeat itself, which looks like long reasoning but isn't.

---

### Comment on the target metric

Every question is answered 8 times, which the full evaluation requires, and every response is scored on its own.

- **Accuracy** on a benchmark is the percentage of its responses that are correct. An answer cut off by the token cap is wrong.
- **Averaged accuracy** is the mean of the accuracies of the four benchmarks (the three BFCL subsets count as one).
- **Accuracy change** is your averaged accuracy minus the baseline's, in percentage points.
- **Tokens saved** on a benchmark is `1 - tokens_yours / tokens_baseline`, where `tokens_yours` and `tokens_baseline` are the total numbers of generated tokens (thinking included) over all of the benchmark's responses, wrong and cut-off ones too.
- **Average tokens saved** is `1 - (r_1 × r_2 × r_3 × r_4)^(1/4)`, where `r_i = tokens_yours / tokens_baseline` on benchmark `i`: the geometric mean of the four token ratios.

Both numbers come with a 95% confidence interval, from resampling whole questions
with your run and the baseline's paired question by question. Read the interval,
not only the point estimate: an accuracy change whose interval covers zero is a
change this evaluation cannot distinguish from no change at all.
