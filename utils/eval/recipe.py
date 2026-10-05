"""The settings a run must use for its scores to count on the leaderboard.

evaluate.py takes these as its defaults and compare.py checks runs against
them, so there is one place to read what "the full evaluation" means. Changing
a value here changes what the model generates: scores recorded under the old
recipe are no longer comparable with new ones.

    from utils.eval.recipe import SAMPLES, token_cap
    token_cap("mbpp")      # 8192
"""

SAMPLES = 8
SEED = 9001
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

# A truncated answer always scores as wrong, so a cap set too low subtracts
# accuracy and, by clipping long generations, flatters any method whose point is
# being shorter. MBPP+ writes whole programs and needs the room; BFCL answers in
# a few hundred tokens and never reaches the default.
MAX_TOKENS = {"mbpp": 8192}
DEFAULT_MAX_TOKENS = 4096


def token_cap(dataset):
    return MAX_TOKENS.get(dataset, DEFAULT_MAX_TOKENS)
