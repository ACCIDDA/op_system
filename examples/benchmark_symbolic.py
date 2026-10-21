"""Measure bounded dense derivative construction and native evaluation.

Run each backend/size in a fresh process. This synthetic linear ring has an
independent analytic reference; it is not a whole-application performance claim.
JAX first evaluation includes compilation. Warm results synchronize native work.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import resource
import statistics
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from op_system import compile_spec, symbolic_derivatives


def _ready(values: tuple[object, ...]) -> None:
    """Wait for asynchronous JAX outputs before stopping a timer."""
    for value in values:
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()


def _host(value: Any) -> Any:  # ruff: ignore[any-type]
    """Copy only at the independent assertion boundary.

    Returns:
        A host array after measured native execution.
    """
    return value.detach().numpy() if hasattr(value, "detach") else np.asarray(value)


def _measure(backend: str, size: int) -> dict[str, object]:
    xp: Any = importlib.import_module(
        {"numpy": "numpy", "jax": "jax.numpy", "torch": "torch"}[backend]
    )
    names = [f"x{i}" for i in range(size)]
    started = time.perf_counter()
    compiled = compile_spec({
        "kind": "expr",
        "state": names,
        "equations": {
            name: f"-a*{name}+b*{names[(i + 1) % size]}" for i, name in enumerate(names)
        },
        "observables": {"measurement": {"expr": f"gain*x0+{names[-1]}"}},
    })
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    artifact = symbolic_derivatives(compiled)
    construction_seconds = time.perf_counter() - started
    state = xp.asarray(np.linspace(0.5, 1.5, size), dtype=xp.float64)
    parameters = {
        name: xp.asarray(value, dtype=xp.float64)
        for name, value in zip(artifact.parameter_names, (0.3, 0.2, 2.0), strict=True)
    }
    evaluate = artifact.evaluate
    if backend == "jax":
        evaluate = importlib.import_module("jax").jit(evaluate)
    started = time.perf_counter()
    values = evaluate(xp.asarray(0.5, dtype=xp.float64), state, **parameters)
    _ready(values)
    first_seconds = time.perf_counter() - started
    elapsed = []
    for _ in range(7):
        started = time.perf_counter()
        values = evaluate(xp.asarray(0.5, dtype=xp.float64), state, **parameters)
        _ready(values)
        elapsed.append(time.perf_counter() - started)
    _validate(values, state, size)
    return {
        "backend": backend,
        "states": size,
        "parameters": artifact.parameter_names,
        "dense_entries": (size + 1) * (size + 3),
        "model": artifact.model.digest,
        "dtype": "float64",
        "device": "cpu",
        "source_compile_seconds": compile_seconds,
        "derivative_construction_seconds": construction_seconds,
        "first_evaluation_seconds": first_seconds,
        "warm_evaluation_seconds": statistics.median(elapsed),
        "warm_repetitions": len(elapsed),
        "whole_worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "analytic_reference": "passed",
    }


def _validate(values: tuple[object, ...], state: object, size: int) -> None:
    """Compare four derivative matrices with an independently constructed ring."""
    reference = -0.3 * np.eye(size) + 0.2 * np.roll(np.eye(size), 1, axis=1)
    parameter_reference = np.column_stack((
        -_host(state),
        np.roll(_host(state), -1),
        np.zeros(size),
    ))
    observation_reference = np.zeros((1, size))
    observation_reference[0, 0] += 2.0
    observation_reference[0, -1] += 1.0
    for value, expected in zip(
        values,
        (
            reference,
            parameter_reference,
            observation_reference,
            np.array([[0.0, 0.0, 0.5]]),
        ),
        strict=True,
    ):
        np.testing.assert_allclose(_host(value), expected, rtol=1e-12, atol=1e-12)


def main() -> None:
    """Measure one native case, retaining versions and synthetic source identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "jax", "torch"), required=True)
    parser.add_argument("--states", type=int, choices=(2, 16, 32), required=True)
    arguments = parser.parse_args()
    context = nullcontext()
    if arguments.backend == "jax":
        jax = importlib.import_module("jax")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        context = enable()
    with context:
        result = _measure(arguments.backend, arguments.states)
    result["versions"] = {name: version(name) for name in ("numpy", arguments.backend)}
    result["fixture_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(result, sort_keys=True))  # ruff: ignore[print]


if __name__ == "__main__":
    main()
