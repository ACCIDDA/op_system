"""Fresh-process NumPy/Numba CPU measurements with packing and cold costs.

Run serial workers with fixed affinity and thread limits. Each worker qualifies
against the public handwritten oracle before reporting warm samples. Baseline
workers never import Numba; candidate workers alternate warm methods and include
an unchanged NumPy control. This measures RHS calls, not solver or training speed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import resource
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from benchmarks.hierarchical_benchmark import (
    _agreement,
    _environment,
    _reference_gain_derivative,
)
from benchmarks.hierarchical_model import TEMPLATES, fixture, model_spec, reference_rhs
from op_system import array_namespace, compile_spec
from op_system.numba import compile_numba

if TYPE_CHECKING:
    from collections.abc import Callable


def _peak() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1 if sys.platform == "darwin" else 1024))


def run_case(  # ruff: ignore[too-many-locals, too-many-statements, too-many-arguments, complex-structure, too-many-branches]
    groups: int,
    stages: int,
    *,
    coupled: bool,
    dtype: Literal["float32", "float64"],
    native: bool,
    repeats: int = 15,
    inner_loops: int = 200,
) -> dict[str, Any]:
    """Qualify, then time one fresh worker with rotating warm method order.

    Returns:
        Raw samples, phase times, numerical checks, source and environment metadata.

    Raises:
        ValueError: Dimensions or sampling budgets are invalid.
    """
    if any(type(v) is not int or v < 1 for v in (groups, stages, repeats, inner_loops)):
        msg = "Dimensions and sample budgets must be positive integers"
        raise ValueError(msg)
    started = time.perf_counter()
    compiled = compile_spec(model_spec(groups, stages, coupled=coupled))
    compiler_seconds = time.perf_counter() - started
    state, params = fixture(groups, stages, dtype=dtype, coupled=coupled)
    y = np.concatenate([state[name].ravel() for name in TEMPLATES])
    structured = compiled.pytree_eval_fn
    if structured is None:
        msg = "Benchmark fixture requires structured evaluation"
        raise ValueError(msg)
    methods: dict[str, Callable[[], object]] = {
        "numpy_flat": lambda: compiled.eval_fn(0.0, y, **params),
        "numpy_structured": lambda: structured(0.0, state, **params),
        "numpy_control": lambda: compiled.eval_fn(0.0, y, **params),
    }
    first = {}
    for name, function in methods.items():
        started = time.perf_counter()
        function()
        first[name] = time.perf_counter() - started
    base_peak = _peak()
    phase = None
    native_source = None
    if native:
        artifact = compile_numba(compiled, dtype=dtype)
        phase = asdict(artifact.compilation_seconds)
        native_source = hashlib.sha256(artifact.source.encode()).hexdigest()
        started = time.perf_counter()
        p = artifact.pack_parameters(params)
        packing_seconds = time.perf_counter() - started
        started = time.perf_counter()
        artifact.evaluate(0.0, y, p)
        first["numba_evaluate"] = time.perf_counter() - started

        def structured_native() -> object:
            result = artifact.evaluate(
                0.0, artifact.pack_state(state), artifact.pack_parameters(params)
            )
            return {
                buffer.name: result[
                    buffer.offset : buffer.offset + buffer.size
                ].reshape(buffer.shape)
                for buffer in artifact.state_layout
            }

        methods.update({
            "numba_kernel": lambda: artifact.kernel(0.0, y, p),
            "numba_evaluate": lambda: artifact.evaluate(0.0, y, p),
            "numba_pack_parameters": lambda: artifact.evaluate(
                0.0, y, artifact.pack_parameters(params)
            ),
            "numba_pack_all": structured_native,
        })
    else:
        packing_seconds = None
    expected = reference_rhs(state, params, coupled=coupled)
    checks = {}
    for name, function in methods.items():

        def shaped(
            values: object,
            parameters: object,
            operation: Callable[[], object] = function,
        ) -> object:
            del values, parameters
            result = operation()
            if isinstance(result, dict):
                return result
            flat = np.asarray(result).reshape(3, groups, stages)
            return dict(zip(TEMPLATES, flat, strict=True))

        checks[name] = _agreement(
            shaped,
            state,
            params,
            expected,
            np,
            array_namespace(y),
            dtype,
            _reference_gain_derivative(
                state,
                params,
                coupled=coupled,
                step=2e-3 if dtype == "float32" else 1e-5,
            ),
        )
    # An independent finite-difference gain direction checks reusable packing.
    gain_check = None
    if native:
        step = 2e-3 if dtype == "float32" else 1e-5
        objectives = []
        for direction in (-1, 1):
            changed = {
                **params,
                "gain": np.asarray(params["gain"] + direction * step, dtype=dtype),
            }
            values = artifact.evaluate(0.0, y, artifact.pack_parameters(changed))
            objectives.append(float(np.sum(values * values)))
        derivative = (objectives[1] - objectives[0]) / (2 * step)
        oracle = _reference_gain_derivative(state, params, coupled=coupled, step=step)
        np.testing.assert_allclose(
            derivative, oracle, rtol=3e-3 if dtype == "float32" else 1e-7, atol=1e-7
        )
        gain_check = {"native": derivative, "oracle": oracle, "step": step}
    samples: dict[str, list[float]] = {name: [] for name in methods}
    names = list(methods)
    orders = []
    for function in methods.values():
        for _ in range(5):
            function()
    for repeat in range(repeats):
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        if repeat % 2:
            order.reverse()
        orders.append(order)
        for name in order:
            function = methods[name]
            started = time.perf_counter()
            for _ in range(inner_loops):
                function()
            samples[name].append((time.perf_counter() - started) / inner_loops)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    break_even = {}
    if phase is not None:
        for name in methods:
            if name.startswith("numba_"):
                baseline = (
                    "numpy_structured" if name == "numba_pack_all" else "numpy_flat"
                )
                saving = medians[baseline] - medians[name]
                break_even[name] = (
                    math.ceil(sum(phase.values()) / saving) if saving > 0 else None
                )
    root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "status": "qualified",
        "case": {
            "groups": groups,
            "stages": stages,
            "coupled": coupled,
            "dtype": dtype,
            "native": native,
        },
        "repeats": repeats,
        "inner_loops": inner_loops,
        "compile_spec_seconds": compiler_seconds,
        "native_phases_seconds": phase,
        "first_call_seconds": first,
        "first_parameter_packing_seconds": packing_seconds,
        "max_errors": checks,
        "gain_direction": gain_check,
        "samples_seconds": samples,
        "sample_orders": orders,
        "median_seconds": medians,
        "cold_break_even_rhs_calls": break_even,
        "process_peak_before_optional_bytes": base_peak,
        "process_peak_rss_bytes": _peak(),
        "source_sha256": {
            str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in (
                "src/op_system/numba.py",
                "src/op_system/_vectorize.py",
                "benchmarks/hierarchical_model.py",
                "benchmarks/numba_rhs.py",
            )
        },
        "generated_source_sha256": native_source,
        "environment": {
            **_environment(np),
            "numba": importlib.metadata.version("numba"),
            "llvmlite": importlib.metadata.version("llvmlite"),
        },
    }


def main() -> None:
    """Run exactly one worker; failures produce no qualified receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--coupled", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_case(
        args.groups,
        args.stages,
        coupled=args.coupled,
        dtype=args.dtype,
        native=args.native,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
