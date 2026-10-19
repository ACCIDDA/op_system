"""Rectangular named-axis products and cached expression evaluation contracts."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from examples.named_kernel_products import inputs, model_spec, reference
from op_system import array_namespace, compile_spec, normalize_rhs
from op_system._vectorize import build_vector_plan

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


def _host(value: object) -> NDArray[np.float64]:
    """Copy a test result to NumPy after any backend differentiation.

    Returns:
        A host array for independent numerical comparisons.
    """
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


@contextmanager
def _backend(name: str) -> Iterator[Any]:
    """Require the selected backend, with scoped double precision for JAX.

    Yields:
        The selected array namespace.
    """
    module = pytest.importorskip(name)
    if name == "jax":
        enable = getattr(module, "enable_x64", None)
        if enable is None:
            enable = module.experimental.enable_x64
        with enable():
            yield import_module("jax.numpy")
    else:
        yield module


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize(("resources", "consumers"), [(3, 2), (2, 4)])
def test_rectangular_products_match_independent_equations(
    backend: str, resources: int, consumers: int
) -> None:
    """Axis order is observable with nonsquare kernels and unequal groups."""
    state, params = inputs(resources, consumers)
    compiled = compile_spec(model_spec(resources, consumers))
    with _backend(backend) as xp:
        native_state = xp.asarray(state, dtype=xp.float64)
        native_params = {
            name: xp.asarray(value, dtype=xp.float64) for name, value in params.items()
        }
        actual = compiled.eval_fn(0.0, native_state, **native_params)
        assert array_namespace(actual) is array_namespace(native_state)
        np.testing.assert_allclose(
            _host(actual), reference(state, params), rtol=1e-13, atol=1e-14
        )
        assert compiled.pytree_eval_fn is not None
        structured = compiled.pytree_eval_fn(
            0.0,
            {"P": native_state[:resources], "Z": native_state[resources:]},
            **native_params,
        )
        np.testing.assert_allclose(
            np.concatenate([_host(structured["P"]), _host(structured["Z"])]),
            reference(state, params),
            rtol=1e-13,
            atol=1e-14,
        )


def test_plan_uses_two_matvecs_and_a_vector_conditional() -> None:
    """Guard the contraction lowering without timing thresholds."""
    plan = build_vector_plan(normalize_rhs(model_spec()))
    assert plan is not None
    aliases = {base: expression for base, expression, _shape in plan.alias_codes}
    for name in ("exposure", "removal"):
        assert "matmul" in aliases[name].code.co_names
        assert "sum" not in aliases[name].code.co_names
    assert "where" in {
        name
        for group in plan.eq_groups
        for expression in group.codes
        for name in expression.code.co_names
    }


def test_cached_functions_bind_fresh_parameters_without_runtime_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flat and PyTree calls keep parameters dynamic without invoking eval."""
    compiled = compile_spec(model_spec())
    state, params = inputs()
    changed = {**params, "K": 2 * params["K"]}

    def reject_eval(*_args: object, **_kwargs: object) -> None:
        msg = "Runtime eval must not be needed by a vector plan"
        raise AssertionError(msg)

    with monkeypatch.context() as patch:
        patch.setattr("builtins.eval", reject_eval)
        first = compiled.eval_fn(0.0, state, **params)
        second = compiled.eval_fn(0.0, state, **changed)
        assert compiled.pytree_eval_fn is not None
        structured = compiled.pytree_eval_fn(
            0.0, {"P": state[:3], "Z": state[3:]}, **changed
        )
    np.testing.assert_allclose(first, reference(state, params))
    np.testing.assert_allclose(second, reference(state, changed))
    np.testing.assert_allclose(
        np.concatenate([_host(structured["P"]), _host(structured["Z"])]), second
    )


def test_jax_jit_and_kernel_derivative() -> None:
    """JIT state and parameter derivatives agree with direct matrix equations."""
    jax = pytest.importorskip("jax")
    state, params = inputs()
    compiled = compile_spec(model_spec())
    with _backend("jax") as xp:
        native = xp.asarray(state)
        native_params = {name: xp.asarray(value) for name, value in params.items()}

        def objective(kernel: Any, values: Any) -> Any:  # ruff: ignore[any-type]
            return xp.sum(
                compiled.eval_fn(0.0, values, **{**native_params, "K": kernel}) ** 2
            )

        def direct(kernel: Any, values: Any) -> Any:  # ruff: ignore[any-type]
            p, z = values[:3], values[3:]
            dp = -p * (kernel.T @ (native_params["g"] * z))
            dz = native_params["efficiency"] * z * (kernel @ p) - xp.where(
                native_params["linear"] > 0,
                native_params["loss"] * z,
                native_params["loss"] * z**2,
            )
            return xp.sum(dp**2) + xp.sum(dz**2)

        actual = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
            native_params["K"], native
        )
        expected = jax.value_and_grad(direct, argnums=(0, 1))(
            native_params["K"], native
        )
        for got, wanted in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            np.testing.assert_allclose(got, wanted, rtol=1e-12, atol=1e-13)


def test_torch_kernel_derivative() -> None:
    """The rectangular contraction keeps CPU autograd connected."""
    torch = pytest.importorskip("torch")
    state, params = inputs()
    compiled = compile_spec(model_spec())
    native = torch.tensor(state, dtype=torch.float64, requires_grad=True)
    kernel = torch.tensor(params["K"], dtype=torch.float64, requires_grad=True)
    native_params = {
        name: torch.asarray(value, dtype=torch.float64)
        for name, value in params.items()
    }
    actual = torch.sum(
        compiled.eval_fn(0.0, native, **{**native_params, "K": kernel}) ** 2
    )
    actual.backward()
    assert kernel.grad is not None
    assert native.grad is not None
    # Independent central differences of the NumPy reference equations.
    epsilon = 1e-6
    for index in np.ndindex(params["K"].shape):
        plus, minus = params["K"].copy(), params["K"].copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        expected = (
            np.sum(reference(state, {**params, "K": plus}) ** 2)
            - np.sum(reference(state, {**params, "K": minus}) ** 2)
        ) / (2 * epsilon)
        np.testing.assert_allclose(
            kernel.grad[index].item(), expected, rtol=1e-8, atol=1e-9
        )
