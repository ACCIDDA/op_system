"""Acceptance tests for issue #101.

Verifies that ``compile_rhs(rhs).eval_fn`` is genuinely **JAX-native** when
called with JAX arrays — i.e. trace-pure, ``jax.jit``-able and ``jax.vmap``-
able without any wrapping for correctness — and that NumPy and JAX call
paths agree numerically.

Mirrors the contract documented in :mod:`op_system._typing` and
:func:`op_system.compile.compile_rhs`: the array namespace is inferred from
the input ``y`` at call time via ``y.__array_namespace__()``; no compile-
time backend selection is required.
"""

from __future__ import annotations

import numpy as np
import pytest

import op_system.compile as _op_system_compile
from op_system import Array, compile_spec

# ---------------------------------------------------------------------------
# Symbol removal: assert _is_numpy_backend is gone (issue #101 step 4).
# ---------------------------------------------------------------------------


def test_is_numpy_backend_is_removed() -> None:
    """The dual-path discriminator should no longer exist."""
    assert not hasattr(_op_system_compile, "_is_numpy_backend"), (
        "_is_numpy_backend should be removed: namespace selection is now "
        "per-call from the input via __array_namespace__()."
    )
    assert not hasattr(_op_system_compile, "_BackendNamespace"), (
        "_BackendNamespace Protocol should be removed: the runtime contract "
        "is op_system.Array (a structural Array-API protocol)."
    )


# ---------------------------------------------------------------------------
# Array protocol export.
# ---------------------------------------------------------------------------


def test_array_protocol_is_exported_and_runtime_checkable() -> None:
    """``op_system.Array`` is a runtime-checkable Protocol."""
    arr = np.asarray([1.0, 2.0])
    assert isinstance(arr, Array)


# ---------------------------------------------------------------------------
# NumPy path (no jax dependency).
# ---------------------------------------------------------------------------


def _expr_spec() -> dict[str, object]:
    return {
        "kind": "expr",
        "state": ["S", "I", "R"],
        "equations": {
            "S": "-beta * S * I",
            "I": "beta * S * I - gamma * I",
            "R": "gamma * I",
        },
    }


def test_numpy_call_no_xp_kwarg() -> None:
    """Compile with no xp; eval with a NumPy array; get a NumPy array out."""
    compiled = compile_spec(_expr_spec())
    y0 = np.asarray([0.99, 0.01, 0.0])
    out = compiled.eval_fn(0.0, y0, beta=1.5, gamma=0.5)
    assert out.__array_namespace__() is np
    expected = np.asarray([
        -1.5 * 0.99 * 0.01,
        1.5 * 0.99 * 0.01 - 0.5 * 0.01,
        0.5 * 0.01,
    ])
    assert np.allclose(np.asarray(out), expected, atol=1e-15)


def test_non_array_y_raises_typeerror() -> None:
    """Passing a Python list (no __array_namespace__) raises a clear error."""
    compiled = compile_spec(_expr_spec())
    with pytest.raises(TypeError, match="__array_namespace__"):
        compiled.eval_fn(0.0, [0.99, 0.01, 0.0], beta=1.5, gamma=0.5)


def test_non_numeric_dtype_raises_typeerror() -> None:
    """An object-dtype array fails the numeric-dtype gate."""
    compiled = compile_spec(_expr_spec())
    bad = np.asarray(["a", "b", "c"], dtype=object)
    with pytest.raises(TypeError, match="numeric"):
        compiled.eval_fn(0.0, bad, beta=1.5, gamma=0.5)


# ---------------------------------------------------------------------------
# JAX path: trace-purity + numerical agreement (skipped if jax unavailable).
# ---------------------------------------------------------------------------


def test_jax_eval_fn_is_jaxpr_pure() -> None:
    """``jax.make_jaxpr(eval_fn)`` succeeds with no concretization warnings.

    This is the trace-purity acceptance criterion from #101: the compiled
    function must be a pure JAX-native callable with no implicit numpy
    coercion that would break tracing.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    compiled = compile_spec(_expr_spec())
    y0 = jnp.asarray([0.99, 0.01, 0.0])

    def f(t: object, y: object, beta: object, gamma: object) -> object:
        return compiled.eval_fn(t, y, beta=beta, gamma=gamma)

    jaxpr = jax.make_jaxpr(f)(0.0, y0, 1.5, 0.5)
    # If we got here the trace closed without TracerArrayConversionError.
    text = str(jaxpr)
    assert "Tracer" not in text  # tracers should not leak as str-reprs
    assert text  # non-empty jaxpr proves the body was actually traced


def test_jax_eval_fn_is_jit_callable() -> None:
    """``jax.jit(eval_fn)`` wraps for performance only; correctness is native."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    compiled = compile_spec(_expr_spec())
    y0 = jnp.asarray([0.99, 0.01, 0.0])
    jit_eval = jax.jit(
        lambda t, y, beta, gamma: compiled.eval_fn(t, y, beta=beta, gamma=gamma)
    )
    out = jit_eval(0.0, y0, 1.5, 0.5)
    assert out.__array_namespace__() is jnp
    expected = np.asarray([
        -1.5 * 0.99 * 0.01,
        1.5 * 0.99 * 0.01 - 0.5 * 0.01,
        0.5 * 0.01,
    ])
    assert np.allclose(np.asarray(out), expected, atol=1e-12)


def test_jax_eval_fn_is_vmap_compatible() -> None:
    """``jax.vmap(eval_fn, in_axes=(None, 0))`` runs over batched state."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    compiled = compile_spec(_expr_spec())
    batch = jnp.asarray([
        [0.99, 0.01, 0.0],
        [0.80, 0.10, 0.10],
        [0.50, 0.30, 0.20],
    ])
    vmapped = jax.vmap(
        lambda y: compiled.eval_fn(0.0, y, beta=1.5, gamma=0.5),
        in_axes=0,
    )
    out = vmapped(batch)
    assert out.shape == (3, 3)
    # Spot-check the first row matches the scalar call.
    scalar = compiled.eval_fn(0.0, batch[0], beta=1.5, gamma=0.5)
    assert np.allclose(np.asarray(out[0]), np.asarray(scalar), atol=1e-12)


def test_numpy_and_jax_paths_agree_numerically() -> None:
    """Same eval_fn, same inputs → same outputs across NumPy and JAX."""
    jnp = pytest.importorskip("jax.numpy")

    compiled = compile_spec(_expr_spec())
    y0_np = np.asarray([0.7, 0.2, 0.1])
    y0_jx = jnp.asarray(y0_np)
    out_np = compiled.eval_fn(0.0, y0_np, beta=0.4, gamma=0.1)
    out_jx = compiled.eval_fn(0.0, y0_jx, beta=0.4, gamma=0.1)
    assert np.allclose(np.asarray(out_np), np.asarray(out_jx), rtol=1e-12)


def test_vectorized_path_is_jax_native_too() -> None:
    """The vectorized eval path also infers namespace from input."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec: dict[str, object] = {
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["y", "o"]}],
        "state": ["S[age]", "I[age]", "R[age]"],
        "equations": {
            "S[age]": "-beta * S[age] * I[age]",
            "I[age]": "beta * S[age] * I[age] - gamma * I[age]",
            "R[age]": "gamma * I[age]",
        },
    }
    compiled = compile_spec(spec)
    y0 = jnp.asarray([0.49, 0.49, 0.01, 0.01, 0.0, 0.0])
    out = jax.jit(lambda y: compiled.eval_fn(0.0, y, beta=1.0, gamma=0.2))(y0)
    assert out.__array_namespace__() is jnp
    assert out.shape == (6,)
