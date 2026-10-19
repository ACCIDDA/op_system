"""Reaction shape and dependency contracts, independent of an engine."""

from __future__ import annotations

import importlib
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from op_system import compile_spec


def _spec(*, batched: bool = False, timed: bool = False) -> dict[str, Any]:
    suffix = "[rep]" if batched else ""
    beta = "beta[time,rep]" if batched and timed else "beta[time]" if timed else "beta"
    result: dict[str, Any] = {
        "kind": "transitions",
        "state": [name + suffix for name in ("S", "I", "R")],
        "aliases": {
            "N" + suffix: f"S{suffix} + I{suffix} + R{suffix}",
            "foi" + suffix: f"{beta} * I{suffix} / N{suffix}",
        },
        "transitions": [
            {
                "name": "infect",
                "from": "S" + suffix,
                "to": "I" + suffix,
                "rate": "foi" + suffix,
            },
            {
                "name": "recover",
                "from": "I" + suffix,
                "to": "R" + suffix,
                "rate": "gamma",
            },
        ],
    }
    axes: list[dict[str, Any]] = (
        [{"name": "rep", "coords": ["a", "b"]}] if batched else []
    )
    if timed:
        axes.append({"name": "time", "type": "continuous", "coords": [0.0, 2.0]})
    if axes:
        result["axes"] = axes
    return result


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_declared_shapes_and_scalar_batch_errors(backend: str) -> None:
    """Aliases and source multiplication obey the same declared shape contract."""
    xp = (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )
    scalar = compile_spec(_spec())
    state = {"S": xp.asarray(90.0), "I": xp.asarray(10.0), "R": xp.asarray(0.0)}
    params = {"beta": 0.3, "gamma": 0.1}
    for reaction, expected in zip(scalar.reactions, (2.7, 1.0), strict=True):
        assert np.asarray(
            reaction.propensity_fn(0.0, state, **params)
        ).item() == pytest.approx(expected)
        with pytest.raises(
            ValueError, match=r"state 'S' expected shape \(\), received \(2,\)"
        ):
            reaction.propensity_fn(
                0.0, {k: xp.stack((v, v)) for k, v in state.items()}, **params
            )
    compiled = compile_spec(_spec(batched=True))
    batched = {k: xp.stack((v, v)) for k, v in state.items()}
    for reaction, expected in zip(compiled.reactions, (2.7, 1.0), strict=True):
        got = np.asarray(reaction.propensity_fn(0.0, batched, **params))
        assert got.shape == (2,)
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-6)
    if backend == "jax":
        jax = pytest.importorskip("jax")
        got = jax.jit(lambda y: compiled.reactions[0].propensity_fn(0.0, y, **params))(
            batched
        )
        np.testing.assert_allclose(np.asarray(got), [2.7, 2.7], rtol=1e-6)
    if backend == "torch":
        batched["S"].requires_grad_()
        cast(
            "Any", compiled.reactions[0].propensity_fn(0.0, batched, **params)
        ).sum().backward()
        np.testing.assert_allclose(batched["S"].grad, [0.003, 0.003], rtol=1e-5)


@pytest.mark.parametrize("policy", ["linear", "previous"])
@pytest.mark.parametrize("batched", [False, True])
def test_only_transitively_required_tables_are_interpolated(
    policy: str, *, batched: bool
) -> None:
    """A nested alias consumes its table; recovery ignores even malformed tables."""
    spec = _spec(batched=batched, timed=True)
    spec["time_interpolation"] = policy
    compiled = compile_spec(spec)
    state = {
        k: np.full((2,) if batched else (), v)
        for k, v in {"S": 90.0, "I": 10.0, "R": 0.0}.items()
    }
    table = np.array([[0.2, 0.4], [0.6, 0.8]]) if batched else np.array([0.2, 0.6])
    module = importlib.import_module("op_system.compile")
    with patch.object(
        module,
        "_interp_along_axis",
        wraps=module._interp_along_axis,  # ruff: ignore[private-member-access]
    ) as spy:
        for time in [-1.0, 0.0, 1.0, 2.0, 3.0]:
            got = compiled.reactions[0].propensity_fn(
                time, state, beta=table, gamma=0.1
            )
            weight = (
                np.clip(time / 2, 0.0, 1.0)
                if policy == "linear"
                else float(time >= 2.0)
            )
            beta = (1 - weight) * table[0] + weight * table[1]
            np.testing.assert_allclose(np.asarray(got), 9.0 * beta)
            compiled.reactions[1].propensity_fn(
                time, state, beta=np.zeros((7, 9)), gamma=0.1
            )
        assert spy.call_count == 5
    table[...] = 0.5
    np.testing.assert_allclose(
        np.asarray(compiled.reactions[0].propensity_fn(1.0, state, beta=table)), 4.5
    )
