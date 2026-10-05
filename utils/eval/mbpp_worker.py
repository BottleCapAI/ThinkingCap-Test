"""Grade a stream of MBPP+ solutions in one process.

Reads one JSON object per line on stdin and writes one verdict per line, in
order. Running many items per process is what makes this affordable: the
evalplus import costs about half a second, and the trusted execution of a
problem's canonical solution is identical for every sample of that problem,
so both are paid once here instead of once per generation.

Executes untrusted model-written code. The caller is responsible for the
sandbox; see utils/eval/sandbox.py.
"""
import contextlib
import io
import json
import os
import sys


def main():
    vendor, data = sys.argv[1:3]
    sys.path.insert(0, vendor)
    os.environ['MBPP_OVERRIDE_PATH'] = data
    from evalplus.data import get_mbpp_plus
    from evalplus.evaluate import check_correctness
    from evalplus.gen.util import trusted_exec
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

    problems = get_mbpp_plus()
    oracles = {}

    def oracle_for(task):
        if task not in oracles:
            problem = problems[task]
            values = {}
            with contextlib.redirect_stdout(io.StringIO()):
                for name in ('base', 'plus'):
                    values[name], values[name + '_time'] = trusted_exec(
                        problem['prompt'] + problem['canonical_solution'],
                        problem[name + '_input'], problem['entry_point'],
                        record_time=True,
                        output_not_none=problem['entry_point'] in MBPP_OUTPUT_NOT_NONE_TASKS)
            oracles[task] = values
        return oracles[task]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        try:
            problem = problems[item['id']]
            with contextlib.redirect_stdout(io.StringIO()):
                result = check_correctness('mbpp', 0, problem, item['code'],
                                           oracle_for(item['id']), fast_check=True,
                                           min_time_limit=4.0, gt_time_limit_factor=4.0)
            verdict = dict(id=item['id'], base=result['base'][0], plus=result['plus'][0])
        except Exception as error:
            # One ungradeable solution must not take the rest of the batch with it.
            verdict = dict(id=item['id'], base=None, plus=None,
                           error=f'{type(error).__name__}: {error}')
        print(json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
