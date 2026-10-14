"""Isolated hierarchical compiler measurements; correctness is a prerequisite."""

from __future__ import annotations

import importlib
import json
import os
import platform
import resource
import statistics
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from benchmarks.hierarchical_model import (
    TEMPLATES,
    fixture,
    handwritten_rhs,
    model_spec,
    reference_rhs,
)
from op_system import CompiledRhs, array_namespace, compile_rhs, normalize_rhs

if TYPE_CHECKING:
    from collections.abc import Callable


class BenchmarkCase(NamedTuple):
    """One backend, shape, precision, and group-coupling workload."""

    backend: str
    groups: int
    stages: int
    dtype: str = "float64"
    coupled: bool = False


class RepeatPolicy(NamedTuple):
    """Serial measurement policy; initialization and warm samples are separate."""

    repeats: int = 5
    inner_loops: int = 100
    warmups: int = 3


def _host(value: Any) -> np.ndarray:  # ruff: ignore[any-type]
    return np.asarray(
        value.detach().cpu().numpy() if hasattr(value, "detach") else value
    )


def _synchronize(values: dict[str, Any]) -> None:
    for value in values.values():
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "samples_seconds": samples,
    }


def _layouts(
    compiled: CompiledRhs,
    module: Any,  # ruff: ignore[any-type]
    xp: Any,  # ruff: ignore[any-type]
    *,
    coupled: bool,
) -> dict[str, Callable[..., Any]]:
    shapes = compiled.template_shapes
    if shapes is None or compiled.pytree_eval_fn is None:
        msg = "hierarchical benchmark requires shaped structured evaluation"
        raise ValueError(msg)

    def flat(state: Any, params: Any) -> Any:  # ruff: ignore[any-type]
        y = xp.concat([xp.reshape(state[name], (-1,)) for name in TEMPLATES])
        derivative = compiled.eval_fn(0.0, y, **params)
        width = int(np.prod(shapes["A"]))
        return {
            name: xp.reshape(derivative[i * width : (i + 1) * width], shapes[name])
            for i, name in enumerate(TEMPLATES)
        }

    structured = compiled.pytree_eval_fn
    functions = {
        "handwritten": lambda y, p: handwritten_rhs(y, p, xp, coupled=coupled),
        "flat": flat,
        "structured": lambda y, p: structured(0.0, y, **p),
    }
    if compiled.block_pytree_eval_fn is None:
        return functions
    block = compiled.block_pytree_eval_fn

    def blocks(state: Any, params: Any) -> Any:  # ruff: ignore[any-type]
        def one(y: Any, local: Any) -> Any:  # ruff: ignore[any-type]
            return block(0.0, y, **{**params, "local": local})

        if xp.__name__.startswith("jax"):
            return module.vmap(one)(state, params["local"])
        if "torch" in xp.__name__:
            return module.func.vmap(one)(state, params["local"])
        rows = [
            one(
                {name: value[index] for name, value in state.items()},
                params["local"][index],
            )
            for index in range(state["A"].shape[0])
        ]
        return {name: xp.stack([row[name] for row in rows]) for name in TEMPLATES}

    functions["block"] = blocks
    return functions


def _gain_derivative(
    function: Callable[..., Any],
    state: Any,  # ruff: ignore[any-type]
    params: Any,  # ruff: ignore[any-type]
    module: Any,  # ruff: ignore[any-type]
    xp: Any,  # ruff: ignore[any-type]
) -> float | None:
    def objective(gain: Any) -> Any:  # ruff: ignore[any-type]
        output = function(state, {**params, "gain": gain})
        return sum(xp.sum(value * value) for value in output.values())

    if xp.__name__.startswith("jax"):
        return float(module.grad(objective)(params["gain"]))
    if "torch" in xp.__name__:
        gain = params["gain"].detach().clone().requires_grad_()
        return float(module.autograd.grad(objective(gain), gain)[0].detach())
    return None


def _reference_gain_derivative(
    state: Any,  # ruff: ignore[any-type]
    params: Any,  # ruff: ignore[any-type]
    *,
    coupled: bool,
    step: float,
) -> float:
    objectives = []
    for sign in [1, -1]:
        result = reference_rhs(
            state, {**params, "gain": params["gain"] + sign * step}, coupled=coupled
        )
        objectives.append(
            sum(
                float(np.sum(value.astype(np.float64) ** 2))
                for value in result.values()
            )
        )
    return (objectives[0] - objectives[1]) / (2 * step)


def _measure(  # ruff: ignore[too-many-arguments]
    function: Callable[..., Any],
    state: Any,  # ruff: ignore[any-type]
    params: Any,  # ruff: ignore[any-type]
    module: Any,  # ruff: ignore[any-type]
    policy: RepeatPolicy,
    *,
    jit: bool,
) -> dict[str, object]:
    cold_jit = None
    operation = function
    if jit:
        start = time.perf_counter()
        operation = module.jit(function).lower(state, params).compile()
        cold_jit = time.perf_counter() - start
    start = time.perf_counter()
    _synchronize(operation(state, params))
    first = time.perf_counter() - start
    for _ in range(policy.warmups):
        _synchronize(operation(state, params))
    samples = []
    for _ in range(policy.repeats):
        start = time.perf_counter()
        for _ in range(policy.inner_loops):
            # Synchronize each call: report end-to-end RHS latency, not
            # asynchronous enqueue speed or host-to-device transfer cost.
            _synchronize(operation(state, params))
        samples.append((time.perf_counter() - start) / policy.inner_loops)
    return {
        "cold_jit_seconds": cold_jit,
        "first_evaluation_seconds": first,
        "warm": _summary(samples),
        "warm_calls_per_second": 1 / statistics.median(samples),
    }


def _compile_case(
    case: BenchmarkCase, repeats: int
) -> tuple[CompiledRhs, dict[str, object], dict[str, object], int]:
    specification = model_spec(case.groups, case.stages, coupled=case.coupled)
    normalize_samples, compile_samples = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        normalized = normalize_rhs(specification)
        normalize_samples.append(time.perf_counter() - start)
        start = time.perf_counter()
        compiled = compile_rhs(normalized)
        compile_samples.append(time.perf_counter() - start)
    size = len(json.dumps(specification, separators=(",", ":")).encode())
    return compiled, _summary(normalize_samples), _summary(compile_samples), size


def _backend_arrays(
    case: BenchmarkCase,
    state: Any,  # ruff: ignore[any-type]
    params: Any,  # ruff: ignore[any-type]
) -> tuple[Any, Any, Any, Any]:
    module = np if case.backend == "numpy" else importlib.import_module(case.backend)
    namespace = (
        importlib.import_module("jax.numpy") if case.backend == "jax" else module
    )
    state = {
        name: namespace.asarray(value, dtype=getattr(namespace, case.dtype))
        for name, value in state.items()
    }
    params = {
        name: namespace.asarray(value, dtype=getattr(namespace, case.dtype))
        for name, value in params.items()
    }
    _synchronize(state)
    return module, array_namespace(state["A"]), state, params


def _agreement(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    function: Callable[..., Any],
    state: Any,  # ruff: ignore[any-type]
    params: Any,  # ruff: ignore[any-type]
    reference: Any,  # ruff: ignore[any-type]
    module: Any,  # ruff: ignore[any-type]
    xp: Any,  # ruff: ignore[any-type]
    dtype: str,
    reference_derivative: float,
) -> dict[str, object]:
    tolerance = 2e-5 if dtype == "float32" else 1e-10
    output = function(state, params)
    _synchronize(output)
    maximum = 0.0
    for template in TEMPLATES:
        actual = _host(output[template])
        if (
            output[template].dtype != state[template].dtype
            or array_namespace(output[template]) is not xp
        ):
            msg = f"{template} changed dtype or namespace"
            raise AssertionError(msg)
        np.testing.assert_allclose(
            actual, reference[template], rtol=tolerance, atol=tolerance
        )
        maximum = max(maximum, float(np.max(np.abs(actual - reference[template]))))
    derivative = _gain_derivative(function, state, params, module, xp)
    derivative_rtol = 3e-3 if dtype == "float32" else 1e-7
    if derivative is not None:
        np.testing.assert_allclose(
            derivative, reference_derivative, rtol=derivative_rtol, atol=1e-8
        )
    return {
        "value_rtol": tolerance,
        "value_atol": tolerance,
        "max_abs_value_error": maximum,
        "gain_derivative": derivative,
        "reference_gain_derivative": reference_derivative,
        "gain_derivative_rtol": derivative_rtol,
        "gain_derivative_atol": 1e-8,
        "gain_derivative_abs_error": abs(derivative - reference_derivative)
        if derivative is not None
        else None,
    }


def _environment(module: Any) -> dict[str, object]:  # ruff: ignore[any-type]
    processor = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        processor = next(
            (
                line.partition(":")[2].strip()
                for line in cpuinfo.read_text(encoding="utf-8").splitlines()
                if line.startswith("model name")
            ),
            processor,
        )
    return {
        "python": platform.python_version(),
        "op_system": version("op_system"),
        "numpy": np.__version__,
        "backend_version": module.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "cpu_count": os.cpu_count(),
        "thread_settings": {
            name: os.environ.get(name)
            for name in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
            ]
        },
    }


def run_case(  # ruff: ignore[too-many-locals]
    case: BenchmarkCase, policy: RepeatPolicy | None = None, *, measure: bool = True
) -> dict[str, object]:
    """Qualify a case and optionally measure serial compiler/evaluator costs.

    Agreement failures raise AssertionError before timings are reported.

    Returns:
        Strict-JSON-compatible dimensions, provenance, agreement, and timings.

    Raises:
        ValueError: If a dimension, backend, precision, or repeat control is invalid.
    """
    policy = policy or RepeatPolicy()
    if case.backend not in {"numpy", "jax", "torch"} or case.dtype not in {
        "float32",
        "float64",
    }:
        msg = "choose numpy/jax/torch and float32/float64"
        raise ValueError(msg)
    if (
        case.groups < 1
        or case.stages < 1
        or policy.repeats < 1
        or policy.inner_loops < 1
        or policy.warmups < 0
    ):
        msg = (
            "dimensions, repeats, and inner loops must be positive; "
            "warmups non-negative"
        )
        raise ValueError(msg)
    compiled, normalization, compilation, size = _compile_case(
        case, policy.repeats if measure else 1
    )
    raw_state, raw_params = fixture(
        case.groups, case.stages, dtype=case.dtype, coupled=case.coupled
    )
    reference = reference_rhs(raw_state, raw_params, coupled=case.coupled)
    start = time.perf_counter()
    module, xp, state, params = _backend_arrays(case, raw_state, raw_params)
    setup = time.perf_counter() - start
    fd_step = 2e-3 if case.dtype == "float32" else 1e-5
    reference_derivative = _reference_gain_derivative(
        raw_state, raw_params, coupled=case.coupled, step=fd_step
    )
    evaluators = _layouts(compiled, module, xp, coupled=case.coupled)
    results = {}
    for name, function in evaluators.items():
        agreement = _agreement(
            function,
            state,
            params,
            reference,
            module,
            xp,
            case.dtype,
            reference_derivative,
        )
        results[name] = {
            **agreement,
            "gain_derivative_step": fd_step,
            "timings": _measure(
                function, state, params, module, policy, jit=case.backend == "jax"
            )
            if measure
            else None,
        }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "case": case._asdict(),
        "status": "qualified",
        "policy": policy._asdict(),
        "fixture": "benchmarks.hierarchical_model.fixture; deterministic, no seed",
        "specification_json_bytes": size,
        "state_elements": 3 * case.groups * case.stages,
        "local_parameter_elements": case.groups,
        "shared_kernel_elements": case.stages**2,
        "fixture_bytes": sum(
            value.nbytes for value in [*raw_state.values(), *raw_params.values()]
        ),
        "normalize": normalization if measure else None,
        "compile_rhs": compilation if measure else None,
        "backend_setup_seconds": setup if measure else None,
        "evaluators": results,
        "block_unavailable_reason": "group reduction couples groups"
        if case.coupled
        else None,
        "process_peak_rss_bytes": int(
            peak if platform.system() == "Darwin" else peak * 1024
        ),
        "environment": _environment(module),
    }
