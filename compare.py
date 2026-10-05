#!/usr/bin/env python3
"""Compares two checkpoints that evaluate.py scored separately.

    python evaluate.py --out runs/base
    python evaluate.py --out runs/mine --adapter ./my-lora
    python compare.py runs/base runs/mine

Both runs drew the same generations for the same questions, because the seed
comes from the question list rather than the run. That lets the rows be paired
afterwards, which cancels question difficulty and gives a much tighter interval
than comparing two scores side by side.

Pairing only holds if the runs really did ask the same thing, so that is
checked rather than assumed, and every mismatch is reported at once.
"""

import argparse
import json
from pathlib import Path

from utils.eval.compare import compare, report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", help="Run directory for the model you started from")
    parser.add_argument("candidate", help="Run directory for your finetune")
    parser.add_argument("--out", help="Also write the comparison as JSON here")
    arguments = parser.parse_args()
    try:
        comparison = compare(arguments.base, arguments.candidate)
    except ValueError as error:
        raise SystemExit(str(error))
    if arguments.out:
        Path(arguments.out).write_text(json.dumps(comparison, indent=2) + "\n")
    print(report(comparison))


if __name__ == "__main__":
    main()
