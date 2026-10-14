"""Required compiler qualification against independent hierarchical equations."""

from __future__ import annotations

from contextlib import contextmanager
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarks.hierarchical_model import (
    TEMPLATES,
    fixture,
    handwritten_rhs,
    model_spec,
    reference_rhs,
)
from op_system import array_namespace, compile_spec
from op_system._errors import UnsupportedFeatureError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from op_system import CompiledRhs

CASES = [
    (False, "flat"),
    (False, "structured"),
    (False, "block"),
    (True, "flat"),
    (True, "structured"),
]
BACKENDS = ("numpy", "jax", "torch")


@contextmanager
def _precision(backend: str) -> Iterator[None]:
    if backend != "jax":
        yield
        return
    jax = pytest.importorskip("jax")
    previous = jax.config.read("jax_enable_x64")
    enable_x64 = True
    jax.config.update("jax_enable_x64", enable_x64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _backend_arrays(
    backend: str, dtype: str, *, coupled: bool
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    xp = (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )
    state, params = fixture(dtype=dtype, coupled=coupled)
    # Native Torch tensors need the Array API compatibility namespace.
    convert = xp.asarray
    state = {
        name: convert(value, dtype=getattr(xp, dtype)) for name, value in state.items()
    }
    params = {
        name: convert(value, dtype=getattr(xp, dtype)) for name, value in params.items()
    }
    return array_namespace(state["A"]), state, params


def _evaluate(
    compiled: CompiledRhs, evaluator: str, state: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    xp = array_namespace(state["A"])
    if evaluator == "flat":
        values = xp.concat([xp.reshape(state[name], (-1,)) for name in TEMPLATES])
        derivative = compiled.eval_fn(0.0, values, **params)
        size = prod(state["A"].shape)
        return {
            name: xp.reshape(
                derivative[index * size : (index + 1) * size], state[name].shape
            )
            for index, name in enumerate(TEMPLATES)
        }
    if evaluator == "structured":
        assert compiled.pytree_eval_fn is not None
        return compiled.pytree_eval_fn(0.0, state, **params)
    assert compiled.block_pytree_eval_fn is not None
    blocks = [
        compiled.block_pytree_eval_fn(
            0.0,
            {name: value[index] for name, value in state.items()},
            **{**params, "local": params["local"][index]},
        )
        for index in range(state["A"].shape[0])
    ]
    return {name: xp.stack([block[name] for block in blocks]) for name in TEMPLATES}


def _host(value: Any) -> np.ndarray:  # ruff: ignore[any-type]
    return np.asarray(
        value.detach().cpu().numpy() if hasattr(value, "detach") else value
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(("coupled", "evaluator"), CASES)
def test_hierarchical_value_contract(
    backend: str, dtype: str, evaluator: str, *, coupled: bool
) -> None:
    """Every qualified layout preserves shape/dtype/namespace and matches the oracle."""
    with _precision(backend):
        xp, state, params = _backend_arrays(backend, dtype, coupled=coupled)
        compiled = compile_spec(model_spec(3, 2, coupled=coupled))
        expected = reference_rhs(
            {k: _host(v) for k, v in state.items()},
            {k: _host(v) for k, v in params.items()},
            coupled=coupled,
        )
        result = _evaluate(compiled, evaluator, state, params)
        handwritten = handwritten_rhs(state, params, xp, coupled=coupled)
        tolerance = 2e-5 if dtype == "float32" else 1e-12
        for name in TEMPLATES:
            assert array_namespace(result[name]) is xp
            assert result[name].shape == state[name].shape
            assert result[name].dtype == state[name].dtype
            assert np.isfinite(_host(result[name])).all()
            np.testing.assert_allclose(
                _host(result[name]), expected[name], rtol=tolerance, atol=tolerance
            )
            np.testing.assert_allclose(
                _host(handwritten[name]), expected[name], rtol=tolerance, atol=tolerance
            )
        if coupled:
            assert compiled.block_pytree_eval_fn is None
            changed = {
                **state,
                "B": state["B"]
                + xp.asarray(
                    [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=state["B"].dtype
                ),
            }
            perturbed = _evaluate(compiled, evaluator, changed, params)
            assert not np.allclose(_host(perturbed["A"])[0], _host(result["A"])[0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_coupled_groups_reject_block_factorization(backend: str) -> None:
    """A declared independence optimization cannot silently remove coupling."""
    with _precision(backend):
        _backend_arrays(backend, "float64", coupled=True)
    spec = model_spec(3, 2, coupled=True)
    spec["factorize_axes"] = ["group"]
    with pytest.raises(UnsupportedFeatureError, match="group"):
        compile_spec(spec)


