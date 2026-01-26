"""
Unit tests for op_system.errors.

These tests validate:
- ErrorCode enum values
- Base OpSystemError behavior
- Helper raiser functions produce correct built-in exceptions
- Exception chaining preserves OpSystemError as __cause__
"""

from __future__ import annotations

import re

import pytest

from op_system.errors import (
    ErrorCode,
    OpSystemError,
    raise_compilation_error,
    raise_invalid_expression,
    raise_invalid_rhs_spec,
    raise_parameter_error,
    raise_state_shape_error,
    raise_unsupported_feature,
)


def test_base_error_carries_code() -> None:
    """Test base error carries code."""
    err = OpSystemError("test", code=ErrorCode.INVALID_PARAMETERS)
    assert err.code == ErrorCode.INVALID_PARAMETERS
    assert "test" in str(err)


def test_raise_invalid_rhs_spec_basic() -> None:
    """Test invalid RHS spec raises ValueError and chains OpSystemError."""
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid op_system RHS specification"),
    ) as exc:
        raise_invalid_rhs_spec(detail="unknown operator")

    e = exc.value
    assert "unknown operator" in str(e)
    assert isinstance(e.__cause__, OpSystemError)
    assert e.__cause__.code == ErrorCode.INVALID_RHS_SPEC


def test_raise_invalid_rhs_spec_missing_fields_sorted_unique() -> None:
    """Test missing fields are sorted and de-duplicated in message."""
    with pytest.raises(
        ValueError,
        match=re.escape("Missing required field(s): ['a', 'b']."),
    ):
        raise_invalid_rhs_spec(missing=["b", "a", "a"])


def test_raise_invalid_expression() -> None:
    """Test invalid expression raises ValueError and chains OpSystemError."""
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid op_system expression"),
    ) as exc:
        raise_invalid_expression(detail="bad token")

    e = exc.value
    assert "bad token" in str(e)
    assert isinstance(e.__cause__, OpSystemError)
    assert e.__cause__.code == ErrorCode.INVALID_EXPRESSION


def test_raise_compilation_error() -> None:
    """Test compilation error raises RuntimeError and chains OpSystemError."""
    with pytest.raises(
        RuntimeError,
        match=re.escape("op_system compilation failed"),
    ) as exc:
        raise_compilation_error(detail="JIT failed")

    e = exc.value
    assert "JIT failed" in str(e)
    assert isinstance(e.__cause__, OpSystemError)
    assert e.__cause__.code == ErrorCode.COMPILATION_FAILED


def test_raise_state_shape_error() -> None:
    """Test invalid state shape raises ValueError and chains OpSystemError."""
    with pytest.raises(
        ValueError,
        match=re.escape("state has an invalid shape/value"),
    ) as exc:
        raise_state_shape_error(name="state", expected="(3,)", got=(2,))

    e = exc.value
    msg = str(e)
    assert "Expected (3,)" in msg
    assert "Got: (2,)" in msg
    assert isinstance(e.__cause__, OpSystemError)
    assert e.__cause__.code == ErrorCode.INVALID_STATE_SHAPE


def test_raise_parameter_error() -> None:
    """Test invalid parameter raises TypeError and chains OpSystemError."""
    with pytest.raises(
        TypeError,
        match=re.escape("Invalid parameters for op_system"),
    ) as exc:
        raise_parameter_error(detail="beta must be float")

    e = exc.value
    assert "beta must be float" in str(e)
    assert isinstance(e.__cause__, OpSystemError)
    assert e.__cause__.code == ErrorCode.INVALID_PARAMETERS


def test_raise_unsupported_feature() -> None:
    """Test unsupported feature raises NotImplementedError and chains OpSystemError."""
    with pytest.raises(
        NotImplementedError,
        match=re.escape("Unsupported op_system feature"),
    ) as exc:
        raise_unsupported_feature(feature="imex", detail="not wired")

    e = exc.value
    msg = str(e)
    assert "Feature 'imex' is not supported" in msg
    assert "not wired" in msg
    assert isinstance(e.__cause__, OpSystemError)
    assert e.__cause__.code == ErrorCode.UNSUPPORTED_FEATURE
