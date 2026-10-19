"""Measure named-axis NumPy evaluation against a direct matrix reference.

Run ``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m
scripts.benchmark_named_kernels --sizes 4 16 128 256 512 1024``. Large compiler
normalizations can take time and memory; the sweep is optional, outside CI.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from examples.named_kernel_products import inputs, model_spec, reference
from op_system import compile_spec

if TYPE_CHECKING:
    from collections.abc import Callable


def _median_seconds(function: Callable[[], object], repeats: int) -> float:
    """Measure independent warmed calls without a timing acceptance threshold.

    Returns:
        Median call duration in seconds.
    """
    for _ in range(5):
        function()
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - start)
    return statistics.median(measurements)


def main() -> None:
    """Check numerical equivalence before reporting each timed comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 16, 128])
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes) or args.repeats < 1:
        parser.error("Sizes and repeat counts must be positive")
    for size in args.sizes:
        state, params = inputs(size, size)
        start = time.perf_counter()
        compiled = compile_spec(model_spec(size, size))
        compile_seconds = time.perf_counter() - start
        expected = reference(state, params)
        actual = compiled.eval_fn(0.0, state, **params)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
        print(
            json.dumps({
                "size": size,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "compile_seconds": compile_seconds,
                "compiled_median_seconds": _median_seconds(
                    partial(compiled.eval_fn, 0.0, state, **params), args.repeats
                ),
                "reference_median_seconds": _median_seconds(
                    partial(reference, state, params), args.repeats
                ),
                "max_absolute_difference": float(np.max(np.abs(actual - expected))),
                "repeats": args.repeats,
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
                },
            }),
            flush=True,
        )


if __name__ == "__main__":
    main()
