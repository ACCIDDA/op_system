"""
Unit tests for op_system.flepimop2.errors.

These tests are written for pytest.

Coverage:
- Error classes carry codes.
- Standardized raisers produce expected exception types and messages.
- Dependency guard raises OptionalDependencyMissingError when flepimop2 is absent.

Note:
- If flepimop2 is installed in the test environment, the dependency-guard test is
  skipped automatically.
"""

from __future__ import annotations

import pytest

from flepimop2.system.op_system.errors import (
    ErrorCode,
    OpSystemFlepimop2Error,
    OptionalDependencyMissingError,
    RhsSpecError,
    SystemConfigError,
    raise_invalid_rhs_spec,
    raise_invalid_system_config,
    raise_parameter_error,
    raise_state_shape_error,
    raise_unsupported_feature,
    require_flepimop2,
)


def test_base_error_carries_code() -> None:
    """Test that OpSystemFlepimop2Error carries an error code correctly."""
    err = OpSystemFlepimop2Error("x", code=ErrorCode.INVALID_PARAMETERS)
    assert err.code == ErrorCode.INVALID_PARAMETERS


def test_raise_invalid_system_config_minimal() -> None:
    """Test raise_invalid_system_config with minimal args."""
    with pytest.raises(SystemConfigError) as ei:
        raise_invalid_system_config()
    e = ei.value
    assert isinstance(e, OpSystemFlepimop2Error)
    assert e.code == ErrorCode.INVALID_SYSTEM_CONFIG
    assert "Invalid op_system.flepimop2 system configuration" in str(e)


def test_raise_invalid_system_config_missing_fields_sorted_unique() -> None:
    """Test raise_invalid_system_config with missing fields.

    Ensures sorting and uniqueness.
    """
    with pytest.raises(SystemConfigError) as ei:
        raise_invalid_system_config(missing=["b", "a", "a"])
    msg = str(ei.value)
    assert "['a', 'b']" in msg
    assert ei.value.code == ErrorCode.INVALID_SYSTEM_CONFIG


def test_raise_invalid_system_config_detail_included() -> None:
    """Test raise_invalid_system_config with detail message included."""
    with pytest.raises(SystemConfigError) as ei:
        raise_invalid_system_config(detail="bad field")
    msg = str(ei.value)
    assert "Detail: bad field" in msg
    assert ei.value.code == ErrorCode.INVALID_SYSTEM_CONFIG


def test_raise_invalid_rhs_spec() -> None:
    """Test raise_invalid_rhs_spec with detail message included."""
    with pytest.raises(RhsSpecError) as ei:
        raise_invalid_rhs_spec(detail="unsupported token")
    e = ei.value
    assert isinstance(e, OpSystemFlepimop2Error)
    assert e.code == ErrorCode.INVALID_RHS_SPEC
    assert "unsupported token" in str(e)


def test_raise_parameter_error_is_typeerror_and_chained() -> None:
    """Test raise_parameter_error with detail message included."""
    with pytest.raises(TypeError) as ei:
        raise_parameter_error(detail="beta must be float")
    e = ei.value
    assert "beta must be float" in str(e)
    assert isinstance(e.__cause__, OpSystemFlepimop2Error)
    assert e.__cause__.code == ErrorCode.INVALID_PARAMETERS


def test_raise_state_shape_error_is_valueerror_and_chained() -> None:
    """Test raise_state_shape_error with detail message included."""
    with pytest.raises(ValueError, match=r"state has an invalid shape/value") as ei:
        raise_state_shape_error(name="state", expected="(3,)", got=(2,))
    e = ei.value
    assert "state has an invalid shape/value" in str(e)
    assert "Expected (3,)" in str(e)
    assert "Got: (2,)" in str(e)
    assert isinstance(e.__cause__, OpSystemFlepimop2Error)
    assert e.__cause__.code == ErrorCode.INVALID_STATE_SHAPE


def test_raise_unsupported_feature_is_notimplemented_and_chained() -> None:
    """Test raise_unsupported_feature with detail message included."""
    with pytest.raises(NotImplementedError) as ei:
        raise_unsupported_feature(feature="imex", detail="operators not wired")
    e = ei.value
    assert "Feature 'imex' is not supported" in str(e)
    assert "operators not wired" in str(e)
    assert isinstance(e.__cause__, OpSystemFlepimop2Error)
    assert e.__cause__.code == ErrorCode.UNSUPPORTED_FEATURE


def test_require_flepimop2_guard_behavior() -> None:
    """Test require_flepimop2 raises when flepimop2 is absent.

    If flepimop2 is installed, skip (missing-dependency branch not testable).
    """
    try:
        require_flepimop2()
    except OptionalDependencyMissingError:
        with pytest.raises(OptionalDependencyMissingError) as ei:
            require_flepimop2()
        e = ei.value
        assert e.code == ErrorCode.OPTIONAL_DEPENDENCY_MISSING
        assert "requires flepimop2" in str(e).lower()
    else:
        pytest.skip(
            "flepimop2 is installed; missing-dependency branch not testable here."
        )
