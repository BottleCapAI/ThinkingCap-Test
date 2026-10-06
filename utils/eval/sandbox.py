"""Runs MBPP+ solutions, which means running code the model wrote.

By default that happens inside [bubblewrap](https://github.com/containers/bubblewrap):
no network, read-only filesystem, 4 GB and 120 seconds. Bubblewrap needs
unprivileged user namespaces, which containers often disable, so on Colab it
usually is not available. There the fallback is resource limits alone — fine on
a disposable VM you own, wrong on a machine you share, so it is always
announced and never silent.

    from utils.eval.sandbox import choose_mode, grade_programs

    mode, why = choose_mode(None)                     # ("bwrap", "auto")
    items = [(0, "Mbpp/2", "def f(x): return x")]     # (your key, task id, code)
    grade_programs(mode, evalplus_dir, mbpp_file, items)
    # {0: {"id": "Mbpp/2", "base": "fail", "plus": "fail"}}
"""

from concurrent.futures import as_completed, ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

MODES = ("bwrap", "rlimit", "off")
MEMORY_BYTES = 4 * 1024 ** 3
CPU_SECONDS = 120
WORKER = Path(__file__).with_name("mbpp_worker.py")
DOWNGRADE_NOTICE = (
    "Bubblewrap unavailable, so model-written code will run with resource limits "
    "only — no filesystem or network isolation. Use --sandbox bwrap to require isolation, "
    "or leave mbpp out of --benchmarks to execute nothing."
)


def have_bwrap():
    """True only if a namespace can actually be created; the binary existing proves nothing."""
    try:
        probe = subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/", "/", "--", "true"],
                               capture_output=True, timeout=30)
        return probe.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def choose_mode(requested):
    """None (pick the best available) or a mode name -> (mode, why). Prints any downgrade."""
    if requested is not None and requested not in MODES:
        raise ValueError(f"Mode must be one of {MODES}")
    if requested in ("off", "rlimit"):
        return requested, "requested"
    available = have_bwrap()
    if requested == "bwrap":
        if not available:
            raise RuntimeError("bubblewrap was requested but cannot create a namespace here")
        return "bwrap", "requested"
    if available:
        return "bwrap", "auto"
    print(DOWNGRADE_NOTICE, file=sys.stderr)
    return "rlimit", "auto_downgrade"


def grade_programs(mode, evalplus, mbpp, items, workers=None, timeout=1800):
    """(key, task id, code) triples -> {key: verdict}, run in parallel batches.

    Samples of one task stay in the same worker, so its trusted solution is
    executed once instead of once per sample. A worker that dies marks its own
    items ungradeable rather than quietly dropping them.
    """
    if not items:
        return {}
    workers = workers or min(8, os.cpu_count() or 1)
    chunks = split_by_task(items, workers)
    verdicts = {}
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        running = {pool.submit(run_batch, mode, evalplus, mbpp, chunk, timeout): chunk
                   for chunk in chunks}
        for future in as_completed(running):
            chunk = running[future]
            try:
                verdicts.update(future.result())
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                for key, task, _ in chunk:
                    verdicts[key] = {"id": task, "base": None, "plus": None, "error": str(error)}
    return verdicts


def split_by_task(items, workers):
    """Deal whole tasks round-robin into `workers` chunks, biggest first."""
    by_task = {}
    for item in items:
        by_task.setdefault(item[1], []).append(item)
    chunks = [[] for _ in range(workers)]
    for index, group in enumerate(sorted(by_task.values(), key=len, reverse=True)):
        chunks[index % workers].extend(group)
    return [chunk for chunk in chunks if chunk]


def run_batch(mode, evalplus, mbpp, items, timeout):
    """Feed one chunk to a single worker process and read back one verdict per item."""
    command, before_exec = worker_command(mode, evalplus, mbpp)
    payload = "".join(json.dumps({"id": task, "code": code}) + "\n" for _, task, code in items)
    finished = subprocess.run(command, input=payload, text=True, capture_output=True,
                              timeout=timeout, preexec_fn=before_exec)
    if finished.returncode:
        raise RuntimeError(f"Sandbox worker failed: {finished.stderr[-2000:]}")
    verdicts = [json.loads(line) for line in finished.stdout.splitlines() if line.strip()]
    if len(verdicts) != len(items):
        raise RuntimeError(f"Worker returned {len(verdicts)} verdicts for {len(items)} items")
    return {key: verdict for (key, _, _), verdict in zip(items, verdicts)}


def worker_command(mode, evalplus, mbpp):
    """-> (argv, preexec_fn). The plain command is the same one bwrap wraps."""
    if mode == "bwrap":
        return bwrap_command(evalplus, mbpp), None
    plain = [sys.executable, str(WORKER), str(evalplus), str(mbpp)]
    return plain, apply_limits if mode == "rlimit" else None


def apply_limits():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))


def bwrap_command(evalplus, mbpp):
    """Read-only mounts of what the worker needs, nothing else, and no network."""
    environment = Path(sys.prefix).resolve()
    runtime = Path(sys.base_prefix).resolve()
    mounts = [Path("/usr"), Path("/lib"), Path("/lib64"), environment, runtime,
              Path(evalplus), Path(mbpp), WORKER.resolve()]
    # bwrap applies these in order, so /tmp has to be mounted before the binds: a tmpfs laid
    # down afterwards hides everything bound beneath it, and a checkout under /tmp then has no
    # venv, no vendored evalplus and no worker script -- every MBPP+ row fails to grade.
    command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for path in dict.fromkeys(path for path in mounts if path.exists()):
        command += ["--ro-bind", str(path), str(path)]
    command += runtime_alias(environment, runtime)
    command += ["--chdir", "/tmp",
                "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "OMP_NUM_THREADS", "1",
                "--setenv", "PYTHONHASHSEED", "0",
                "/usr/bin/prlimit", f"--as={MEMORY_BYTES}", f"--cpu={CPU_SECONDS}", "--",
                str(environment / "bin/python"), str(WORKER), str(evalplus), str(mbpp)]
    return command


def runtime_alias(environment, runtime):
    """uv points the venv launcher through a version-agnostic symlink; mount that path too."""
    launcher = environment / "bin/python"
    if not launcher.is_symlink():
        return []
    target = Path(os.readlink(launcher))
    if not target.is_absolute():
        target = launcher.parent / target
    alias = target.parent.parent
    return ["--ro-bind", str(runtime), str(alias)] if alias != runtime else []
