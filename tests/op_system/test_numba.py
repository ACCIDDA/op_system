"""Required optional native compilation checks against independent public models."""

from __future__ import annotations

import importlib
from typing import Literal

import numpy as np
import pytest

from benchmarks.hierarchical_model import fixture, model_spec, reference_rhs
from op_system import compile_spec
from op_system.numba import compile_numba


def _require_numba() -> None:
    pytest.importorskip("numba")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("coupled", [False, True])
def test_native_hierarchy_matches_independent_balance_equations(
    dtype: Literal["float32", "float64"], *, coupled: bool
) -> None:
    """Both precisions preserve shaped parameters, reducers, aliases and flat order."""
    _require_numba()
    compiled = compile_spec(model_spec(3, 2, coupled=coupled))
    artifact = compile_numba(compiled, dtype=dtype)
    state, params = fixture(3, 2, coupled=coupled, dtype=dtype)
    y, p = artifact.pack_state(state), artifact.pack_parameters(params)
    before = y.copy(), p.copy()
    result = artifact.evaluate(0.25, y, p)
    expected = np.concatenate([
        value.ravel()
        for value in reference_rhs(state, params, coupled=coupled).values()
    ])
    tolerance = 3e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance / 10)
    np.testing.assert_allclose(
        result, compiled.eval_fn(0.25, y, **params), rtol=tolerance, atol=tolerance / 10
    )
    assert result.dtype == y.dtype == p.dtype == np.dtype(dtype)
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, y)
    assert not y.flags.writeable
    assert not p.flags.writeable
    np.testing.assert_array_equal(y, before[0])
    np.testing.assert_array_equal(p, before[1])
    assert len(artifact.kernel.nopython_signatures) == 1
    changed = dict(params)
    changed["gain"] = np.asarray(0.7, dtype=dtype)
    other = artifact.evaluate(0.25, y, artifact.pack_parameters(changed))
    assert not np.allclose(other, result)
    np.testing.assert_array_equal(artifact.evaluate(0.25, y, p), result)
    # The compiled signature accepts writable inputs without specializing again.
    np.testing.assert_array_equal(artifact.kernel(0.25, y.copy(), p.copy()), result)
    assert len(artifact.kernel.nopython_signatures) == 1


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_scalar_time_cse_and_internal_name_collision(
    dtype: Literal["float32", "float64"],
) -> None:
    """User names cannot collide with generated variables or output bindings."""
    _require_numba()
    compiled = compile_spec({
        "kind": "expr",
        "state": ["X", "Y"],
        "aliases": {"u": "np.exp(-k*t)"},
        "equations": {"X": "-k*X + 1.5*u", "Y": "output_X*Y + 2*u"},
    })
    native = compile_numba(compiled, dtype=dtype)
    state = {"X": np.asarray(2, dtype=dtype), "Y": np.asarray(3, dtype=dtype)}
    params = {"k": 0.5, "output_X": 0.3}
    y, p = native.pack_state(state), native.pack_parameters(params)
    t = np.asarray(0.2, dtype=dtype)
    result = native.evaluate(t, y, p)
    expected = [-1 + 1.5 * np.exp(-0.1), 0.9 + 2 * np.exp(-0.1)]
    np.testing.assert_allclose(
        result, expected, rtol=2e-6 if dtype == "float32" else 1e-13
    )


def test_native_float32_arithmetic_does_not_hide_float64_intermediates() -> None:
    """Float32 overflow/cancellation agrees with NumPy rather than double arithmetic."""
    _require_numba()
    compiled = compile_spec({
        "kind": "expr",
        "state": ["X"],
        "equations": {"X": "X * 2.0 - X"},
    })
    native = compile_numba(compiled, dtype="float32")
    y = native.pack_state({"X": np.asarray(2e38, dtype="float32")})
    with np.errstate(over="ignore", invalid="ignore"):
        expected = compiled.eval_fn(0.0, y)
        actual = native.evaluate(0.0, y, native.pack_parameters({}))
    assert np.isposinf(expected).all()
    np.testing.assert_array_equal(actual, expected)


def test_native_input_contract_and_bounds() -> None:
    """Shape, precision, contiguity and namespace errors cannot silently specialize."""
    _require_numba()
    native = compile_numba(
        compile_spec({
            "kind": "expr",
            "state": ["X", "Y"],
            "equations": {"X": "-k*X", "Y": "k*X"},
        })
    )
    y = native.pack_state({"X": 1.0, "Y": 2.0})
    p = native.pack_parameters({"k": 0.5})
    with pytest.raises(ValueError, match="names"):
        native.pack_parameters({"other": 0.5})
    with pytest.raises(ValueError, match="shape"):
        native.pack_parameters({"k": np.ones(2)})
    with pytest.raises(TypeError, match="dtype"):
        native.evaluate(0.0, y.astype("float32"), p)
    with pytest.raises(TypeError, match="contiguous"):
        native.evaluate(0.0, np.ones(4)[::2], p)
    with pytest.raises(ValueError, match="shape"):
        native.evaluate(np.ones(2), y, p)
    with pytest.raises(ValueError, match="lengths"):
        native.kernel(0.0, y[:1], p)
    with pytest.raises(ValueError, match="lengths"):
        native.kernel(0.0, y, p[:0])

    class Foreign:
        def __array__(self) -> object:  # ruff: ignore[bad-dunder-method-name]
            pytest.fail("Foreign host conversion must not be attempted")

    with pytest.raises(TypeError, match="plain NumPy"):
        native.evaluate(0.0, Foreign(), p)
    with pytest.raises(TypeError, match="dtype"):
        native.pack_parameters({"k": np.float32(0.5)})


def test_native_unsupported_call_fails_instead_of_falling_back() -> None:
    """The optional compiler cannot label an unsupported Python evaluator native."""
    compiled = compile_spec({
        "kind": "expr",
        "state": ["X"],
        "equations": {"X": "np.clip(X, 0, 1)"},
    })
    with pytest.raises(NotImplementedError, match="documented NumPy"):
        compile_numba(compiled)


def test_native_optional_dependency_error_and_disabled_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent or disabled acceleration is an explicit error on the required path."""
    compiled = compile_spec({"kind": "expr", "state": ["X"], "equations": {"X": "-X"}})
    original = importlib.import_module

    def unavailable(name: str) -> object:
        if name == "numba":
            raise ImportError(name)
        return original(name)

    with monkeypatch.context() as context:
        context.setattr(importlib, "import_module", unavailable)
        with pytest.raises(ImportError, match=r"op_system\[numba\]"):
            compile_numba(compiled)
    _require_numba()
    numba = importlib.import_module("numba")
    monkeypatch.setattr(numba.config, "DISABLE_JIT", True)
    with pytest.raises(ValueError, match="DISABLE_JIT"):
        compile_numba(compiled)


def test_native_piecewise_and_numpy_zero_division() -> None:
    """Native branches and singular arithmetic retain NumPy floating behavior."""
    _require_numba()
    compiled = compile_spec({
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {"x": "np.where(x > 0, np.sin(x), np.cos(x))", "y": "x/y"},
    })
    native = compile_numba(compiled)
    for x in (-1.0, 1.0):
        y = native.pack_state({"x": x, "y": 0.0})
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = compiled.eval_fn(0.0, y)
        np.testing.assert_allclose(
            native.evaluate(0.0, y, native.pack_parameters({})), expected, rtol=1e-13
        )


def test_native_benchmark_receipt_accounting() -> None:
    """The public harness records packing, cold costs, raw orders and oracle checks."""
    _require_numba()
    from benchmarks.numba_rhs import run_case  # ruff: ignore[import-outside-top-level]

    receipt = run_case(
        2, 2, coupled=True, dtype="float32", native=True, repeats=2, inner_loops=2
    )
    assert receipt["status"] == "qualified"
    assert set(receipt["native_phases_seconds"]) == {
        "generation",
        "optional_import",
        "native_compilation",
    }
    assert len(receipt["sample_orders"]) == 2
    assert all(len(samples) == 2 for samples in receipt["samples_seconds"].values())
    assert set(receipt["cold_break_even_rhs_calls"]) == {
        "numba_kernel",
        "numba_evaluate",
        "numba_pack_parameters",
        "numba_pack_all",
    }
    assert receipt["gain_direction"] is not None
    assert (
        receipt["process_peak_rss_bytes"]
        >= receipt["process_peak_before_optional_bytes"]
    )
