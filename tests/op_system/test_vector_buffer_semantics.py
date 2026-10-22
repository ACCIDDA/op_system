"""Broadcast optimizations retain value, alias, mutability and native AD behavior."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from op_system import array_namespace, compile_spec


def _spec() -> dict[str, object]:
    return {
        "kind": "expr",
        "axes": [
            {"name": "group", "coords": ["a", "b", "c"]},
            {"name": "stage", "coords": ["u", "v"]},
        ],
        "state": ["x[group,stage]", "z[group,stage]"],
        "aliases": {
            "identity[group,stage]": "x[group,stage]",
            "constant[group,stage]": "level",
        },
        "equations": {
            "x[group,stage]": "identity[group,stage]",
            "z[group,stage]": "constant[group,stage]",
        },
    }


class _ArraySubclass(np.ndarray[tuple[int, ...], np.dtype[np.generic]]):
    """Exercise normal broadcast coercion for an ndarray subclass."""


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("subclass", [False, True])
def test_numpy_readonly_views_preserve_noncontiguous_inputs(
    dtype: str, *, subclass: bool
) -> None:
    """An identity derivative shares storage without changing input mutability."""
    compiled = compile_spec(_spec())
    operation = compiled.pytree_eval_fn
    assert operation is not None
    values = np.arange(12, dtype=dtype).reshape(3, 4)[:, ::2]
    if subclass:
        values = values.view(_ArraySubclass)
    result: Any = operation(
        0.0,
        {"x": values, "z": np.zeros_like(values)},
        level=np.asarray(2.0, dtype=dtype),
    )
    np.testing.assert_array_equal(result["x"], values)
    np.testing.assert_array_equal(result["z"], np.full((3, 2), 2.0))
    assert type(result["x"]) is np.ndarray
    assert np.shares_memory(result["x"], values)
    assert values.flags.writeable
    assert not result["x"].flags.writeable
    assert not result["z"].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result["x"][0, 0] = -1
    values[0, 0] = 19
    assert result["x"][0, 0] == 19


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_broadcast_values_and_gradients(backend: str, dtype: str) -> None:
    """Identity and scalar broadcasting retain native dtypes and exact derivatives."""
    module = pytest.importorskip(backend)
    precision: Any = nullcontext()
    xp = module
    if backend == "jax":
        enable = getattr(module, "enable_x64", None)
        if enable is None:
            enable = module.experimental.enable_x64
        precision = enable()
        xp = pytest.importorskip("jax.numpy")
    compiled = compile_spec(_spec())
    operation = compiled.pytree_eval_fn
    assert operation is not None
    with precision:
        values = xp.asarray(np.arange(6).reshape(3, 2), dtype=getattr(xp, dtype))
        level = xp.asarray(2.0, dtype=getattr(xp, dtype))
        result: Any = operation(
            0.0, {"x": values, "z": xp.zeros_like(values)}, level=level
        )
        for name in ("x", "z"):
            assert result[name].shape == (3, 2)
            assert result[name].dtype == values.dtype
            assert array_namespace(result[name]) is array_namespace(values)
        if backend == "numpy":
            np.testing.assert_array_equal(result["x"], values)
            np.testing.assert_array_equal(result["z"], np.full((3, 2), 2.0))
        elif backend == "jax":

            def objective(y: Any, a: Any) -> Any:  # ruff: ignore[any-type]
                out: Any = operation(0.0, {"x": y, "z": xp.zeros_like(y)}, level=a)
                return xp.sum(out["x"] + out["z"])

            value, (dy, da) = module.jit(
                module.value_and_grad(objective, argnums=(0, 1))
            )(values, level)
            np.testing.assert_allclose(value, 27.0)
            np.testing.assert_array_equal(dy, np.ones((3, 2)))
            np.testing.assert_allclose(da, 6.0)
        else:
            values.requires_grad_()
            level.requires_grad_()
            out: Any = operation(
                0.0, {"x": values, "z": xp.zeros_like(values)}, level=level
            )
            total = xp.sum(out["x"] + out["z"])
            dy, da = module.autograd.grad(total, (values, level))
            np.testing.assert_allclose(total.detach().numpy(), 27.0)
            np.testing.assert_array_equal(dy.numpy(), np.ones((3, 2)))
            np.testing.assert_allclose(da.numpy(), 6.0)


def test_jax_foreign_numpy_buffer_still_converts_namespace() -> None:
    """A matching NumPy parameter cannot bypass a native broadcast conversion."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    compiled = compile_spec({
        "kind": "expr",
        "axes": [{"name": "g", "coords": ["a", "b"]}],
        "state": ["x[g]"],
        "aliases": {"same[g]": "fixed"},
        "equations": {"x[g]": "same[g]"},
    })
    operation = compiled.pytree_eval_fn
    assert operation is not None
    values = jnp.zeros(2, dtype=jnp.float32)
    fixed = np.asarray([2.0, 3.0], dtype=np.float32)
    result: Any = jax.jit(lambda y: operation(0.0, {"x": y}, fixed=fixed)["x"])(values)
    assert array_namespace(result) is array_namespace(values)
    np.testing.assert_array_equal(result, fixed)
