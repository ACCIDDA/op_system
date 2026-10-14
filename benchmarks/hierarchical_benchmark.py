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


