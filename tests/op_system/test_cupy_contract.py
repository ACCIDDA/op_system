"""Optional CuPy GPU contracts; CPU skips are explicitly not device qualification.

Use ``scripts/qualify_backend.py cupy`` on a suitable CUDA device. That required
path rejects missing/broken CUDA before collection and rejects every skipped case.
Default hosted CPU runners do not execute these device checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.hierarchical_model import TEMPLATES, fixture, model_spec, reference_rhs
from op_system import array_namespace, compile_spec, symbolic_derivatives

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture(scope="module")
def cupy_backend() -> ModuleType:
    """Return CuPy on device zero, or explicitly skip absent optional hardware.

    Returns:
        Imported CUDA array module with an available device.
    """
    module = cast("ModuleType", pytest.importorskip("cupy"))
    try:
        count = module.cuda.runtime.getDeviceCount()
    except (OSError, module.cuda.runtime.CUDARuntimeError) as exc:
        pytest.skip(f"CuPy CUDA hardware unavailable; not qualified: {exc}")
    if count < 1:
        pytest.skip("CuPy needs a CUDA device; CPU skip is not qualification")
    module.cuda.Device(0).use()
    return module


def _on_device(module: ModuleType, value: object, dtype: str) -> None:
    """Assert native GPU storage and floating precision before host comparisons."""
    assert isinstance(value, module.ndarray)
    assert value.device.id == 0
    assert value.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("coupled", [False, True])
def test_cupy_hierarchical_flat_and_structured(  # ruff: ignore[too-many-locals]
    cupy_backend: ModuleType, dtype: str, *, coupled: bool
) -> None:
    """One compiled artifact preserves shaped inputs, reducers and flat state order."""
    cp = cupy_backend
    model = compile_spec(model_spec(3, 2, coupled=coupled))
    states, params = fixture(3, 2, dtype=dtype, coupled=coupled)
    expected = reference_rhs(states, params, coupled=coupled)
    flat = np.concatenate([states[name].ravel() for name in TEMPLATES])
    cpu_result = model.eval_fn(0.0, flat, **params)
    native_states = {name: cp.asarray(value) for name, value in states.items()}
    native_params = {name: cp.asarray(value) for name, value in params.items()}
    y = cp.asarray(flat)
    before = y.copy()
    actual = model.eval_fn(0.0, y, **native_params)
    _on_device(cp, actual, dtype)
    assert array_namespace(actual) is array_namespace(y)
    tolerance = 2e-5 if dtype == "float32" else 1e-12
    reference = np.concatenate([expected[name].ravel() for name in TEMPLATES])
    np.testing.assert_allclose(
        cp.asnumpy(actual), reference, rtol=tolerance, atol=tolerance / 10
    )
    np.testing.assert_allclose(
        cp.asnumpy(actual), cpu_result, rtol=tolerance, atol=tolerance / 10
    )
    assert model.pytree_eval_fn is not None
    structured = model.pytree_eval_fn(0.0, native_states, **native_params)
    assert tuple(structured) == TEMPLATES
    for name in TEMPLATES:
        _on_device(cp, structured[name], dtype)
        assert structured[name].shape == states[name].shape
        np.testing.assert_allclose(
            cp.asnumpy(structured[name]),
            expected[name],
            rtol=tolerance,
            atol=tolerance / 10,
        )
        np.testing.assert_array_equal(cp.asnumpy(native_states[name]), states[name])
    np.testing.assert_array_equal(cp.asnumpy(y), cp.asnumpy(before))
    for name in params:
        np.testing.assert_array_equal(cp.asnumpy(native_params[name]), params[name])
    # A strided GPU state remains native and follows the same declared order.
    storage = cp.empty((flat.size * 2,), dtype=dtype)
    strided = storage[::2]
    strided[...] = y
    again = model.eval_fn(0.0, strided, **native_params)
    _on_device(cp, again, dtype)
    np.testing.assert_allclose(
        cp.asnumpy(again), reference, rtol=tolerance, atol=tolerance / 10
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cupy_scalar_time_and_direct_literals(
    cupy_backend: ModuleType, dtype: str
) -> None:
    """Scalar time and direct constant equations retain the state's GPU and dtype."""
    cp = cupy_backend
    model = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "equations": {"x": "0", "z": "2.0*t + 1.5"},
    })
    y = cp.asarray([2.0, 4.0], dtype=dtype)
    t = cp.asarray(0.5, dtype=dtype)
    actual = model.eval_fn(t, y)
    _on_device(cp, actual, dtype)
    np.testing.assert_array_equal(cp.asnumpy(actual), [0.0, 2.5])
    assert model.pytree_eval_fn is not None
    result = model.pytree_eval_fn(t, {"x": y[0], "z": y[1]})
    for name, expected in zip(("x", "z"), (0.0, 2.5), strict=True):
        _on_device(cp, result[name], dtype)
        assert result[name].shape == ()
        np.testing.assert_array_equal(cp.asnumpy(result[name]), expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cupy_shape_errors_match_numpy(cupy_backend: ModuleType, dtype: str) -> None:
    """Malformed flat/structured layouts fail before producing a derivative."""
    model = compile_spec({
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["X[loc]"],
        "equations": {"X[loc]": "-X[loc]"},
    })
    assert model.pytree_eval_fn is not None
    for module in (np, cupy_backend):
        with pytest.raises(ValueError, match=r"shape|length|1-D|one-dimensional"):
            model.eval_fn(0.0, module.ones((3,), dtype=dtype))
        with pytest.raises(ValueError, match=r"shape|size|broadcast"):
            model.pytree_eval_fn(0.0, {"X": module.ones((3,), dtype=dtype)})


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cupy_bounded_derivative_values(cupy_backend: ModuleType, dtype: str) -> None:
    """Symbolic matrices remain GPU arrays; this does not claim CuPy autodiff."""
    cp = cupy_backend
    model = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "equations": {"x": "-a*x+b*z", "z": "a*x-b*z"},
        "observables": {"measurement": {"expr": "gain*x+z"}},
    })
    artifact = symbolic_derivatives(model)
    y = cp.asarray([2.0, 4.0], dtype=dtype)
    params = {
        name: cp.asarray(value, dtype=dtype)
        for name, value in {"a": 0.3, "b": 0.7, "gain": 3.0}.items()
    }
    matrices = artifact.evaluate(cp.asarray(0.5, dtype=dtype), y, **params)
    for value in matrices:
        _on_device(cp, value, dtype)
    expected = (
        [[-0.3, 0.7], [0.3, -0.7]],
        [[-2.0, 4.0, 0.0], [2.0, -4.0, 0.0]],
        [[3.0, 1.0]],
        [[0.0, 0.0, 2.0]],
    )
    for actual, oracle in zip(matrices, expected, strict=True):
        np.testing.assert_allclose(
            cp.asnumpy(actual),
            oracle,
            rtol=2e-6 if dtype == "float32" else 1e-13,
            atol=1e-7,
        )
