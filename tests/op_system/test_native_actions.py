"""Native actions checked against independent dense derivatives and graph contracts."""

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest

from op_system import (
    OperatorDescriptor,
    compile_spec,
    state_actions,
    symbolic_derivatives,
)
from op_system import actions as action_module

if TYPE_CHECKING:
    from types import ModuleType


def _arrays(backend: str) -> tuple[Any, Any]:
    """Select native arrays and a call-local float64 setting.

    Returns:
        Optional namespace and local precision context.
    """
    if backend == "torch":
        return pytest.importorskip("torch"), nullcontext()
    jax = pytest.importorskip("jax")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    return pytest.importorskip("jax.numpy"), enable()


def _host(value: Any) -> Any:  # ruff: ignore[any-type]
    """Transfer only at the independent assertion boundary.

    Returns:
        Host values after native action and any outer differentiation.
    """
    return value.detach().numpy() if hasattr(value, "detach") else np.asarray(value)


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("observable", [None, "measurement"])
def test_nonlinear_actions_match_independent_dense_values_and_transpose_duality(
    backend: Literal["jax", "torch"], dtype: str, observable: str | None
) -> None:
    """Analytic off-diagonal derivatives catch orientation and measurement errors."""
    compiled = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "aliases": {"growth": "np.exp(a*x)"},
        "equations": {"x": "growth+b*z*z", "z": "x*z"},
        "observables": {"measurement": {"expr": "gain*x+z*z"}},
    })
    xp, precision = _arrays(backend)
    action = state_actions(compiled, backend=backend, observable=observable)
    with precision:
        y = xp.asarray([0.7, 1.2], dtype=getattr(xp, dtype))
        v = xp.asarray([0.2, -0.3], dtype=y.dtype)
        params = {
            name: xp.asarray(value, dtype=y.dtype)
            for name, value in (("a", 0.4), ("b", 0.6), ("gain", 1.5))
        }
        native_params = {name: params[name] for name in action.parameter_names}
        matrix = (
            np.array([[0.4 * np.exp(0.28), 1.44], [1.2, 0.7]])
            if observable is None
            else np.array([1.5, 2.4])
        )
        w = xp.asarray([-0.4, 0.6] if observable is None else 0.6, dtype=y.dtype)
        jvp, vjp = action.jvp, action.vjp
        if backend == "jax":
            jvp = pytest.importorskip("jax").jit(jvp)
            vjp = pytest.importorskip("jax").jit(vjp)
        forward = jvp(0.5, y, v, **native_params)
        reverse = vjp(0.5, y, w, **native_params)
        np.testing.assert_allclose(
            _host(forward), matrix @ np.array([0.2, -0.3]), rtol=2e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            _host(reverse),
            matrix.T @ _host(w) if observable is None else matrix * _host(w),
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.sum(_host(w) * _host(forward)),
            np.sum(_host(v) * _host(reverse)),
            rtol=2e-6,
            atol=1e-7,
        )
        dense = symbolic_derivatives(compiled).evaluate(0.5, y, **params)
        np.testing.assert_allclose(
            _host(dense.state)
            if observable is None
            else _host(dense.observable_state)[0],
            matrix,
            rtol=2e-6,
            atol=1e-7,
        )
        assert (
            getattr(forward, "dtype", None)
            == getattr(reverse, "dtype", None)
            == y.dtype
        )


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("observable", [None, "cells", "total", "constant"])
def test_shaped_parameters_and_array_or_reduced_observables(
    backend: Literal["jax", "torch"], observable: str | None
) -> None:
    """Coupled reductions and their transpose use complete declared output shapes."""
    compiled = compile_spec({
        "kind": "expr",
        "state": ["x[loc]"],
        "axes": [{"name": "loc", "coords": ["a", "b", "c"]}],
        "equations": {"x[loc]": "-rate[loc]*x[loc]+mix*sum_over(x[loc:l],loc=l)"},
        "observables": {
            "cells": {"expr": "gain*x[loc]", "axes": ["loc"]},
            "total": {"expr": "x[loc]", "reduction": {"kind": "sum", "axes": ["loc"]}},
            "constant": {"expr": "2"},
        },
    })
    xp, precision = _arrays(backend)
    action = state_actions(compiled, backend=backend, observable=observable)
    with precision:
        y = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)
        v = xp.asarray([0.2, -0.3, 0.5], dtype=y.dtype)
        parameters = {
            "rate": xp.asarray([0.3, 0.5, 0.7], dtype=y.dtype),
            "mix": xp.asarray(0.2, dtype=y.dtype),
            "gain": xp.asarray(2.0, dtype=y.dtype),
        }
        params = {name: parameters[name] for name in action.parameter_names}
        matrix = {
            None: -np.diag([0.3, 0.5, 0.7]) + 0.2 * np.ones((3, 3)),
            "cells": 2 * np.eye(3),
            "total": np.ones(3),
            "constant": np.zeros(3),
        }[observable]
        w = xp.asarray([0.4, -0.2, 0.6] if action.output_shape else 0.4, dtype=y.dtype)
        forward = action.jvp(0.0, y, v, **params)
        reverse = action.vjp(0.0, y, w, **params)
        np.testing.assert_allclose(
            _host(forward), matrix @ _host(v), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            _host(reverse),
            matrix.T @ _host(w) if action.output_shape else matrix * _host(w),
            rtol=1e-12,
            atol=1e-12,
        )
        assert getattr(forward, "shape", None) == action.output_shape


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_actions_retain_outer_state_and_parameter_gradients_and_rebind_values(
    backend: Literal["jax", "torch"], mode: str
) -> None:
    """Independent mixed derivatives detect stale bindings and detached AD graphs."""
    xp, precision = _arrays(backend)
    action = state_actions(
        compile_spec({
            "kind": "expr",
            "state": ["x"],
            "equations": {"x": "np.exp(a*x)"},
        }),
        backend=backend,
    )
    with precision:
        y = xp.asarray([0.7], dtype=xp.float64)
        a = xp.asarray(0.4, dtype=y.dtype)
        vector = xp.asarray([0.3], dtype=y.dtype)
        method = getattr(action, mode)
        if backend == "jax":
            jax = pytest.importorskip("jax")
            dy, da = jax.jit(
                jax.grad(
                    lambda yy, aa: xp.sum(method(0.0, yy, vector, a=aa)), argnums=(0, 1)
                )
            )(y, a)
        else:
            y.requires_grad_()
            a.requires_grad_()
            dy, da = xp.autograd.grad(method(0.0, y, vector, a=a).sum(), (y, a))
        np.testing.assert_allclose(_host(dy), [0.3 * 0.4**2 * np.exp(0.28)], rtol=1e-12)
        np.testing.assert_allclose(
            _host(da), 0.3 * np.exp(0.28) * (1 + 0.4 * 0.7), rtol=1e-12
        )
        changed = method(0.0, y, vector, a=xp.asarray(0.8, dtype=y.dtype))
        np.testing.assert_allclose(
            _host(changed), [0.3 * 0.8 * np.exp(0.56)], rtol=1e-12
        )


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_vmap_and_nonfinite_behavior(backend: Literal["jax", "torch"]) -> None:
    """Batching retains independent native actions and nonfinite domain failures."""
    xp, precision = _arrays(backend)
    action = state_actions(
        compile_spec({
            "kind": "expr",
            "state": ["x"],
            "equations": {"x": "np.exp(a*x)"},
        }),
        backend=backend,
    )
    with precision:
        states = xp.asarray([[0.2], [0.7]], dtype=xp.float64)
        vector = xp.asarray([0.3], dtype=xp.float64)
        a = xp.asarray(0.4, dtype=xp.float64)
        transform = pytest.importorskip("jax").vmap if backend == "jax" else xp.vmap
        actual = transform(lambda y: action.jvp(0.0, y, vector, a=a))(states)
        np.testing.assert_allclose(
            _host(actual),
            0.3 * 0.4 * np.exp(0.4 * np.array([[0.2], [0.7]])),
            rtol=1e-12,
        )
        invalid = action.jvp(
            0.0, xp.asarray([float("nan")], dtype=xp.float64), vector, a=a
        )
        assert np.isnan(_host(invalid)).all()


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize(("policy", "rate"), [("linear", 0.3), ("previous", 0.2)])
def test_time_table_bindings_keep_the_declared_interpolation(
    backend: Literal["jax", "torch"], policy: str, rate: float
) -> None:
    """State derivatives freeze time and reuse the existing forcing evaluation."""
    xp, precision = _arrays(backend)
    action = state_actions(
        compile_spec({
            "kind": "expr",
            "time_axis": "time",
            "axes": [{"name": "time", "type": "continuous", "coords": [0.0, 1.0, 2.0]}],
            "time_interpolation": policy,
            "state": ["x"],
            "equations": {"x": "-rate[time]*x"},
        }),
        backend=backend,
    )
    assert action.parameter_names == ("rate",)
    with precision:
        y = xp.asarray([0.7], dtype=xp.float64)
        table = xp.asarray([0.2, 0.6, 0.9], dtype=y.dtype)
        forward = action.jvp(0.25, y, xp.asarray([0.5], dtype=y.dtype), rate=table)
        reverse = action.vjp(0.25, y, xp.asarray([0.8], dtype=y.dtype), rate=table)
        np.testing.assert_allclose(_host(forward), [-rate * 0.5], rtol=1e-12)
        np.testing.assert_allclose(_host(reverse), [-rate * 0.8], rtol=1e-12)


def test_jax_plain_real_time_keeps_eager_and_jit_state_actions_equal() -> None:
    """Common integer initial times retain weak real-scalar promotion under jit."""
    xp, precision = _arrays("jax")
    action = state_actions(
        compile_spec({"kind": "expr", "state": ["x"], "equations": {"x": "t*a*x"}}),
        backend="jax",
    )
    with precision:
        y = xp.asarray([0.7], dtype=xp.float32)
        v = xp.asarray([0.3], dtype=y.dtype)
        a = xp.asarray(0.4, dtype=y.dtype)
        for method in (action.jvp, action.vjp):
            compiled_method = pytest.importorskip("jax").jit(method)
            for time in (0, 2, 0.0, 2.0):
                expected = [time * 0.4 * 0.3]
                np.testing.assert_allclose(
                    _host(method(time, y, v, a=a)), expected, rtol=2e-6
                )
                np.testing.assert_allclose(
                    _host(compiled_method(time, y, v, a=a)), expected, rtol=2e-6
                )
            with pytest.raises(ValueError, match="native"):
                compiled_method(xp.asarray(0, dtype=xp.int32), y, v, a=a)


def test_jax_published_action_workflow() -> None:
    """Execute the published example and check independent product values."""
    _arrays("jax")
    namespace: dict[str, Any] = {}
    guide = Path(__file__).resolve().parents[2] / "docs/guides/native-actions.md"
    for block in re.findall(r"```python\n(.*?)\n```", guide.read_text(), re.DOTALL):
        exec(block, namespace)  # ruff: ignore[exec-builtin] - trusted guide
    for name, expected in (
        ("forward", [-0.26, 0.26]),
        ("reverse", [0.4, -0.6]),
        ("measured", 0.0),
        ("backprojected", [0.9, 0.6]),
    ):
        np.testing.assert_allclose(
            _host(namespace[name]), expected, rtol=2e-6, atol=1e-7
        )


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_layout_namespace_precision_and_parameter_failures(
    backend: Literal["jax", "torch"],
) -> None:
    """Invalid native requests fail instead of silently selecting a dense fallback."""
    xp, precision = _arrays(backend)
    action = state_actions(
        compile_spec({"kind": "expr", "state": ["x"], "equations": {"x": "a*x"}}),
        backend=backend,
    )
    with precision:
        y = xp.asarray([1.0], dtype=xp.float64)
        v = xp.asarray([0.3], dtype=y.dtype)
        a = xp.asarray(0.4, dtype=y.dtype)
        for method in (action.jvp, action.vjp):
            with pytest.raises(ValueError, match="parameter names"):
                method(0.0, y, v)
            with pytest.raises(ValueError, match="precision"):
                method(0.0, y, v, a=xp.asarray(0.4, dtype=xp.float32))
            with pytest.raises(ValueError, match="native"):
                method(0.0, np.ones(1), v, a=a)
            with pytest.raises(ValueError, match="shape"):
                method(0.0, y, xp.asarray([0.1, 0.2], dtype=y.dtype), a=a)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_missing_optional_backend_is_explicit(
    backend: Literal["jax", "torch"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Factory failure precedes numerical execution when an extra is absent."""

    def missing(name: str) -> ModuleType:
        raise ImportError(name)

    monkeypatch.setattr(action_module, "import_module", missing)
    compiled = compile_spec({"kind": "expr", "state": ["x"], "equations": {"x": "-x"}})
    with pytest.raises(ImportError, match="optional"):
        state_actions(compiled, backend=backend)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_action_graphs_avoid_dense_state_matrices(
    backend: Literal["jax", "torch"],
) -> None:
    """A nonlinear local ring retains scalar/vector graph intermediates.

    Graph inspection covers this fixture, not a general backend memory bound.
    Torch FX inspection is a test-only version-scoped probe.
    """
    xp, precision = _arrays(backend)
    size = 16
    compiled = compile_spec({
        "kind": "expr",
        "state": [f"x{i}" for i in range(size)],
        "equations": {
            f"x{i}": f"np.exp(a*x{i})+b*x{(i + 1) % size}" for i in range(size)
        },
    })
    action = state_actions(compiled, backend=backend)

    def evaluate(y: object, v: object, a: object, b: object) -> tuple[object, object]:
        return action.jvp(0.0, y, v, a=a, b=b), action.vjp(0.0, y, v, a=a, b=b)

    with precision:
        inputs = (
            xp.asarray(np.linspace(0.2, 0.8, size), dtype=xp.float64),
            xp.asarray(np.linspace(0.1, 0.5, size), dtype=xp.float64),
            xp.asarray(0.4, dtype=xp.float64),
            xp.asarray(0.3, dtype=xp.float64),
        )
        if backend == "jax":
            graph = pytest.importorskip("jax").make_jaxpr(evaluate)(*inputs)
            shapes = {
                tuple(getattr(getattr(var, "aval", None), "shape", ()))
                for equation in graph.jaxpr.eqns
                for var in equation.outvars
            }
        else:
            graph = pytest.importorskip("torch.fx.experimental.proxy_tensor").make_fx(
                evaluate
            )(*inputs)
            shapes = {
                tuple(node.meta["val"].shape)
                for node in graph.graph.nodes
                if hasattr(node.meta.get("val"), "shape")
            }
        assert (size,) in shapes
        assert shapes <= {(), (1,), (size,)}


def test_factory_rejects_unsupported_scope_and_targets() -> None:
    """Source, coordinate and backend capabilities are separate from native AD."""
    compiled = compile_spec({"kind": "expr", "state": ["x"], "equations": {"x": "-x"}})
    with pytest.raises(ValueError, match="explicit"):
        state_actions(compiled, backend="numpy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="observable"):
        state_actions(compiled, backend="jax", observable="absent")
    with pytest.raises(ValueError, match="source"):
        state_actions(replace(compiled, _rhs=None), backend="jax")
    with pytest.raises(ValueError, match="layouts"):
        state_actions(replace(compiled, state_names=("other",)), backend="jax")
    with pytest.raises(NotImplementedError, match="exclude"):
        state_actions(
            replace(compiled, history_requirements=({"synthetic": "history"},)),
            backend="jax",
        )
    with pytest.raises(NotImplementedError, match="exclude"):
        state_actions(
            replace(compiled, operators=(OperatorDescriptor(axis="loc"),)),
            backend="jax",
        )
