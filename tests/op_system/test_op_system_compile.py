"""Unit tests for op_system.compile (pytest).

These tests cover:
- compile_rhs produces an eval_fn callable
- eval_fn evaluates expr correctly
- CompiledRhs.bind returns a 2-arg RHS with parameters fixed
- shape validation uses op_system.errors messages
- missing symbols raise parameter errors
- invalid syntax is rejected during normalization (specs) in v1
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from op_system.compile import compile_rhs
from op_system.errors import ErrorCode, OpSystemError
from op_system.specs import normalize_rhs


def _cause_code(exc: BaseException) -> ErrorCode | None:
    """
    Extract the ErrorCode from an exception's cause chain.

    Args:
        exc: The exception to inspect.

    Returns:
        The ErrorCode if found, otherwise None.
    """
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, OpSystemError):
        return cause.code
    return None


def test_compile_rhs_expr_happy_path_and_eval() -> None:
    """compile_rhs produces eval_fn that evaluates expressions correctly."""
    spec = {
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {"x": "a * x", "y": "x + y + b"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

    out = compiled.eval_fn(
        np.float64(0.0),
        np.array([2.0, 3.0], dtype=np.float64),
        a=2.0,
        b=1.0,
    )
    assert out.dtype == np.float64
    assert out.shape == (2,)
    assert np.allclose(out, np.array([4.0, 6.0], dtype=np.float64))


def test_compiledrhs_bind_binds_params() -> None:
    """CompiledRhs.bind returns a 2-arg callable with parameters fixed."""
    spec = {
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {"x": "a * x", "y": "x + y + b"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

    bound = compiled.bind({"a": 2.0, "b": 1.0})
    out = bound(np.float64(0.0), np.array([2.0, 3.0], dtype=np.float64))

    assert out.dtype == np.float64
    assert out.shape == (2,)
    assert np.allclose(out, np.array([4.0, 6.0], dtype=np.float64))


def test_eval_rejects_non_1d_state_shape() -> None:
    """eval_fn rejects non-1D state arrays with standardized message."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "0.0"}}
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

    y_bad = np.zeros((1, 1), dtype=np.float64)
    msg = "state has an invalid shape/value. Expected 1D array."
    with pytest.raises(ValueError, match=re.escape(msg)) as exc:
        compiled.eval_fn(np.float64(0.0), y_bad)

    assert _cause_code(exc.value) == ErrorCode.INVALID_STATE_SHAPE


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
    with pytest.raises(ValueError, match=re.escape(msg)) as exc:
        compiled.eval_fn(np.float64(0.0), np.zeros((3,), dtype=np.float64))

    assert _cause_code(exc.value) == ErrorCode.INVALID_STATE_SHAPE


def test_eval_unknown_symbol_raises_parameter_error() -> None:
    """eval_fn raises parameter error for unknown symbols in expressions."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "beta * x"}}
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

    with pytest.raises(TypeError, match=r"Invalid parameters for op_system") as exc:
        compiled.eval_fn(np.float64(0.0), np.array([1.0], dtype=np.float64))

    assert _cause_code(exc.value) == ErrorCode.INVALID_PARAMETERS


def test_invalid_expression_syntax_rejected_during_normalize() -> None:
    """In v1, expression syntax is validated during specs normalization."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "beta **"}}

    with pytest.raises(ValueError, match=r"Invalid op_system expression") as exc:
        normalize_rhs(spec)

    assert _cause_code(exc.value) == ErrorCode.INVALID_EXPRESSION
