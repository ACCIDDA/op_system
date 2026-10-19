"""Compiler contracts for independent-clock stochastic consumers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from op_system import compile_spec


def _spec(*, explicit_time: bool = False) -> dict[str, Any]:
    return {
        "kind": "transitions",
        "axes": [
            {"name": "rep", "coords": ["a", "b"]},
            {"name": "time", "type": "continuous", "coords": [0.0, 2.0]},
        ],
        "factorize_axes": ["rep"],
        "time_interpolation": "previous",
        "state": ["X[rep]"],
        "transitions": [
            {
                "name": "birth",
                "to": "X[rep]",
                "rate": "beta * (1 + t)" if explicit_time else "beta[time,rep]",
            }
        ],
    }


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_resolved_callback_retains_native_shapes_and_independence(backend: str) -> None:
    """An engine can resolve each table at its own replicate's clock once."""
    xp = (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )
    compiled = compile_spec(_spec())
    reaction = compiled.reactions[0]
    assert reaction.independent_axes == ("rep",)
    assert reaction.time_varying_params == (("beta", ("time", "rep")),)
    assert not reaction.uses_time
    resolved = reaction.resolved_propensity_fn
    assert resolved is not None
    state = {"X": xp.asarray([0.0, 0.0])}
    table = xp.asarray([[0.2, 0.4], [0.6, 0.8]])
    np.testing.assert_allclose(
        np.asarray(reaction.propensity_fn(2.0, state, beta=table)),
        [0.6, 0.8],
        rtol=1e-6,
    )
    got = resolved(xp.asarray([0.0, 2.0]), state, beta=xp.asarray([0.2, 0.8]))
    np.testing.assert_allclose(np.asarray(got), [0.2, 0.8], rtol=1e-6)
    if backend == "jax":
        jax = pytest.importorskip("jax")
        got = jax.jit(lambda b: resolved(0.0, state, beta=b))(xp.asarray([0.3, 0.7]))
        np.testing.assert_allclose(got, [0.3, 0.7], rtol=1e-6)


def test_continuous_time_dependency_is_explicit() -> None:
    """A batch axis does not make an explicitly time-dependent hazard exact SSA."""
    reaction = compile_spec(_spec(explicit_time=True)).reactions[0]
    assert reaction.uses_time
    assert reaction.time_varying_params == ()


def test_undeclared_axis_is_not_an_independence_certificate() -> None:
    """Identical shapes alone cannot establish independent stochastic replicates."""
    spec = _spec()
    spec.pop("factorize_axes")
    assert compile_spec(spec).reactions[0].independent_axes == ()
