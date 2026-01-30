"""Unit tests for op_system.compile (pytest).

These tests cover:
- compile_rhs produces an eval_fn callable
- eval_fn evaluates expr correctly
- CompiledRhs.bind returns a 2-arg RHS with parameters fixed
- bind returns a strict 2-arg callable (no runtime kwargs)
- shape validation rejects non-1D and wrong-length states
- missing symbols raise parameter errors
- invalid syntax is rejected during normalization (specs) in v1
- AST whitelist rejects disallowed calls/attribute access
- alias dependency resolution and cycles
- transitions RHS compiles and evaluates with expected flow semantics

Note:
- errors.py was removed, so we only assert built-in exception types and messages.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from op_system.compile import CompiledRhs, compile_rhs
from op_system.specs import NormalizedRhs, normalize_rhs


@pytest.fixture
def expr_spec_xy() -> dict[str, object]:
    """Base 2-state expression RHS spec used by multiple tests.

    Returns:
        Base RHS spec dictionary.
    """
    return {
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {"x": "a * x", "y": "x + y + b"},
    }


@pytest.fixture
def rhs_xy(expr_spec_xy: dict[str, object]) -> NormalizedRhs:
    """Normalized RHS for the base 2-state spec.

    Args:
        expr_spec_xy: Base RHS spec dictionary.

    Returns:
        NormalizedRhs instance.
    """
    return normalize_rhs(expr_spec_xy)


@pytest.fixture
def compiled_xy(rhs_xy: NormalizedRhs) -> CompiledRhs:
    """Compiled RHS for the base 2-state spec.

    Args:
        rhs_xy: NormalizedRhs instance.

    Returns:
        CompiledRhs instance.
    """
    return compile_rhs(rhs_xy)


def test_compile_rhs_expr_happy_path_and_eval(compiled_xy: CompiledRhs) -> None:
    """compile_rhs produces eval_fn that evaluates expressions correctly."""
    out = compiled_xy.eval_fn(
        np.float64(0.0),
        np.array([2.0, 3.0], dtype=np.float64),
        a=2.0,
        b=1.0,
    )
    assert out.dtype == np.float64
    assert out.shape == (2,)
    assert np.allclose(out, np.array([4.0, 6.0], dtype=np.float64))


def test_compiledrhs_bind_binds_params(compiled_xy: CompiledRhs) -> None:
    """CompiledRhs.bind returns a 2-arg callable with parameters fixed."""
    bound = compiled_xy.bind({"a": 2.0, "b": 1.0})
    assert callable(bound)

    out = bound(np.float64(0.0), np.array([2.0, 3.0], dtype=np.float64))
    assert out.dtype == np.float64
    assert out.shape == (2,)
    assert np.allclose(out, np.array([4.0, 6.0], dtype=np.float64))


def test_bind_is_strict_two_arg_callable(compiled_xy: CompiledRhs) -> None:
    """bind() returns a strict (t, y) callable and rejects runtime kwargs."""
    bound = compiled_xy.bind({"a": 2.0, "b": 1.0})
    with pytest.raises(TypeError, match=r"unexpected keyword argument"):
        bound(
            np.float64(0.0),
            np.array([2.0, 3.0], dtype=np.float64),
            a=3.0,
        )


def test_eval_rejects_non_1d_state_shape() -> None:
    """eval_fn rejects non-1D state arrays with standardized message."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "0.0"}}
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

    y_bad = np.zeros((1, 1), dtype=np.float64)
    msg = "state has an invalid shape/value. Expected 1D array."
    with pytest.raises(ValueError, match=re.escape(msg)):
        compiled.eval_fn(np.float64(0.0), y_bad)


def test_eval_rejects_wrong_state_length() -> None:
    """eval_fn rejects state arrays of incorrect length with standardized message."""
    spec = {
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {"x": "0.0", "y": "0.0"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

    msg = "state has an invalid shape/value. Expected (n_state=2,)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        compiled.eval_fn(np.float64(0.0), np.zeros((3,), dtype=np.float64))


def test_eval_unknown_symbol_raises_parameter_error() -> None:
    """eval_fn raises parameter error for unknown symbols in expressions."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "beta * x"}}
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

    with pytest.raises(TypeError, match=r"Invalid parameters for op_system"):
        compiled.eval_fn(np.float64(0.0), np.array([1.0], dtype=np.float64))


def test_invalid_expression_syntax_rejected_during_normalize() -> None:
    """In v1, expression syntax is validated during specs normalization."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "beta **"}}
    with pytest.raises(ValueError, match=r"Invalid op_system expression"):
        normalize_rhs(spec)


def test_compile_rejects_disallowed_function_calls() -> None:
    """Disallowed function calls (e.g., np.sin) should be rejected."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "np.sin(x)"}}
    rhs = normalize_rhs(spec)
    with pytest.raises(ValueError, match=r"disallowed function call"):
        compile_rhs(rhs)


def test_compile_rejects_disallowed_attribute_access() -> None:
    """Disallowed attribute access should be rejected."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "x.real"}}
    rhs = normalize_rhs(spec)
    with pytest.raises(ValueError, match=r"disallowed attribute access"):
        compile_rhs(rhs)


def test_eval_allows_whitelisted_np_calls() -> None:
    """A small set of np.* calls is whitelisted and should work."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "np.maximum(x, 0.0)"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)
    out = compiled.eval_fn(np.float64(0.0), np.array([-2.0], dtype=np.float64))
    assert np.allclose(out, np.array([0.0], dtype=np.float64))


def test_alias_dependency_resolution() -> None:
    """Aliases may depend on earlier aliases and should resolve."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "aliases": {"a": "x + 1", "b": "a + 2"},
        "equations": {"x": "b"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)
    out = compiled.eval_fn(np.float64(0.0), np.array([3.0], dtype=np.float64))
    assert np.allclose(out, np.array([6.0], dtype=np.float64))


def test_alias_cycle_is_rejected() -> None:
    """Alias cycles should be rejected during evaluation (cannot resolve)."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "aliases": {"a": "b + 1", "b": "a + 1"},
        "equations": {"x": "a"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)
    with pytest.raises(ValueError, match=r"could not resolve alias dependencies"):
        compiled.eval_fn(np.float64(0.0), np.array([1.0], dtype=np.float64))


def test_transitions_rhs_compiles_and_evaluates() -> None:
    """Transitions RHS should compile and evaluate with expected flow semantics."""
    spec = {
        "kind": "transitions",
        "state": ["S", "I", "R"],
        "aliases": {"N": "S + I + R"},
        "transitions": [
            {"from": "S", "to": "I", "rate": "beta * I / N"},
            {"from": "I", "to": "R", "rate": "gamma"},
        ],
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

    y = np.array([999.0, 1.0, 0.0], dtype=np.float64)
    out = compiled.eval_fn(np.float64(0.0), y, beta=0.3, gamma=0.1)

    n_total = float(np.sum(y))
    inf = 0.3 * float(y[1]) / n_total * float(y[0])
    rec = 0.1 * float(y[1])
    expected = np.array([-inf, inf - rec, rec], dtype=np.float64)
    assert np.allclose(out, expected)
