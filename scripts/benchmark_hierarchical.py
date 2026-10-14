"""Run correctness-gated, serial CPU benchmarks in fresh worker processes."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

from benchmarks.hierarchical_benchmark import BenchmarkCase, RepeatPolicy, run_case


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        msg = "must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+", choices=["numpy", "jax", "torch"], default=["numpy"]
    )
    parser.add_argument("--groups", nargs="+", type=_positive, default=[4, 16])
    parser.add_argument("--stages", nargs="+", type=_positive, default=[2, 4])
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--coupling", choices=["independent", "coupled", "both"], default="both"
    )
    parser.add_argument("--repeats", type=_positive, default=5)
    parser.add_argument("--inner-loops", type=_positive, default=100)
    parser.add_argument("--warmups", type=_positive, default=3)
    parser.add_argument(
        "--check-only", action="store_true", help="qualify without collecting timings"
    )
    parser.add_argument("--output", type=Path, help="JSON file; defaults to stdout")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _worker(
    case: BenchmarkCase, policy: RepeatPolicy, *, check_only: bool
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark_hierarchical",
        "--worker",
        "--backends",
        case.backend,
        "--groups",
        str(case.groups),
        "--stages",
        str(case.stages),
        "--dtype",
        case.dtype,
        "--coupling",
        "coupled" if case.coupled else "independent",
        "--repeats",
        str(policy.repeats),
        "--inner-loops",
        str(policy.inner_loops),
        "--warmups",
        str(policy.warmups),
    ]
    if check_only:
        command.append("--check-only")
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    process = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        command,
        cwd=Path(__file__).resolve().parent.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, object] = json.loads(process.stdout)
    return result


def _revision() -> dict[str, object]:
    directory = Path(__file__).resolve().parent.parent
    executable = shutil.which("git")
    if executable is None:
        return {"git_commit": None, "working_tree_dirty": None}
    revision = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [executable, "rev-parse", "HEAD"],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [executable, "status", "--porcelain"],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(status.stdout) if status.returncode == 0 else None,
    }


def main() -> int:
    """Run every requested case, aborting instead of reporting failed agreement.

    Returns:
        Zero after all cases qualify and the strict JSON report is written.

    Raises:
        ValueError: If an internal worker is given more than one case.
    """
    args = _parser().parse_args()
    policy = RepeatPolicy(args.repeats, args.inner_loops, args.warmups)
    coupled = [False, True] if args.coupling == "both" else [args.coupling == "coupled"]
    cases = [
        BenchmarkCase(backend, groups, stages, args.dtype, coupling)
        for backend, groups, stages, coupling in itertools.product(
            args.backends, args.groups, args.stages, coupled
        )
    ]
    if args.worker:
        if len(cases) != 1:
            msg = "a worker accepts exactly one case"
            raise ValueError(msg)
        print(
            json.dumps(
                run_case(cases[0], policy, measure=not args.check_only), allow_nan=False
            )
        )
        return 0
    results = []
    for case in cases:
        print(f"Qualifying {case}", file=sys.stderr, flush=True)
        try:
            results.append(_worker(case, policy, check_only=args.check_only))
        except subprocess.CalledProcessError as error:
            print(error.stderr, file=sys.stderr)
            return 1
    report = {
        "schema_version": 1,
        "revision": _revision(),
        "arguments": sys.argv[1:],
        "mode": "correctness-only" if args.check_only else "timing",
        "execution": "serial isolated CPU processes; one thread requested per worker",
        "cases": results,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
