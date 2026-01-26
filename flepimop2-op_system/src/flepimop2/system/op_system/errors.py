"""
Error types and dependency-guard utilities for op_system.flepimop2.

Design intent:
- op_system can be installed without flepimop2
- op_system.flepimop2 fails fast with clear, actionable messages if used without
  the optional extra dependencies.

This module mirrors the style and ergonomics of op_engine.flepimop2.errors.
"""

from __future__ import annotations

from enum import StrEnum
from importlib.util import find_spec
from typing import Final

# =============================================================================
# Install guidance
# =============================================================================

_FLEPIMOP2_EXTRA_INSTALL_MSG: Final[str] = (
    "Install the optional dependency group with:\n"
    "  pip install 'op_system[flepimop2]'\n"
    "or, if you are using uv:\n"
    "  uv pip install '.[flepimop2]'"
)


# =============================================================================
# Error codes
# =============================================================================


class ErrorCode(StrEnum):
    """Machine-readable classification for op_system.flepimop2 failures.

    Use these codes to support consistent logging and (optional) programmatic
    recovery without requiring many custom exception subclasses.
    """

    OPTIONAL_DEPENDENCY_MISSING = "optional_dependency_missing"
    INVALID_SYSTEM_CONFIG = "invalid_system_config"
    INVALID_RHS_SPEC = "invalid_rhs_spec"
    INVALID_STATE_SHAPE = "invalid_state_shape"
    INVALID_PARAMETERS = "invalid_parameters"
    UNSUPPORTED_FEATURE = "unsupported_feature"


# =============================================================================
# Base exceptions
# =============================================================================


class OpSystemFlepimop2Error(Exception):
    """Base exception for op_system.flepimop2 integration errors.

    This exists so callers can catch integration-layer failures explicitly
    without depending on a wide taxonomy of custom exception subclasses.
    """

    def __init__(self, message: str, *, code: ErrorCode | None = None) -> None:
        """
        Initialize an OpSystemFlepimop2Error.

        Args:
            message: Human-readable error message.
            code: Optional machine-readable error code.
        """
        super().__init__(message)
        self.code: ErrorCode | None = code


class OptionalDependencyMissingError(OpSystemFlepimop2Error, ImportError):
    """Raised when an optional dependency is required but missing."""


class SystemConfigError(OpSystemFlepimop2Error, ValueError):
    """Raised when a flepimop2 system config is invalid or incomplete."""


class RhsSpecError(OpSystemFlepimop2Error, ValueError):
    """Raised when an RHS specification is invalid or unsupported."""


# =============================================================================
# Dependency guards
# =============================================================================


def require_flepimop2() -> None:
    """Require that flepimop2 is importable.

    Raises:
        OptionalDependencyMissingError: If flepimop2 cannot be imported.
    """
    if find_spec("flepimop2") is not None:
        return

    msg = (
        "The op_system.flepimop2 integration requires flepimop2, but it is not "
        "available in this environment.\n\n"
        "Import detail: Module spec not found\n\n"
        f"{_FLEPIMOP2_EXTRA_INSTALL_MSG}"
    )
    raise OptionalDependencyMissingError(
        msg, code=ErrorCode.OPTIONAL_DEPENDENCY_MISSING
    )


# =============================================================================
# Standardized raisers (small helper API)
# =============================================================================


def raise_invalid_system_config(
    *,
    missing: list[str] | None = None,
    detail: str | None = None,
) -> None:
    """Raise a standardized system configuration error.

    Args:
        missing: List of missing required fields, if any.
        detail: Additional detail about the configuration issue.

    Raises:
        SystemConfigError: Always.
    """
    parts: list[str] = ["Invalid op_system.flepimop2 system configuration."]
    if missing:
        parts.append(f"Missing required field(s): {sorted(set(missing))}.")
    if detail:
        parts.append(f"Detail: {detail}")
    msg = " ".join(parts)
    raise SystemConfigError(msg, code=ErrorCode.INVALID_SYSTEM_CONFIG)


def raise_invalid_rhs_spec(*, detail: str) -> None:
    """Raise a standardized RHS specification error.

    Args:
        detail: Detail text describing the RHS-spec issue.

    Raises:
        RhsSpecError: Always.
    """
    msg = f"Invalid RHS specification for op_system.flepimop2 adapter: {detail}"
    raise RhsSpecError(msg, code=ErrorCode.INVALID_RHS_SPEC)


def raise_unsupported_feature(*, feature: str, detail: str | None = None) -> None:
    """Raise a standardized unsupported-feature error.

    Args:
        feature: Short feature name (e.g., 'imex', 'multi_axis_state').
        detail: Optional explanatory detail.

    Raises:
        NotImplementedError: Always.
    """
    base = (
        f"Feature '{feature}' is not supported under the current "
        "op_system.flepimop2 system adapter."
    )
    msg = f"{base} Detail: {detail}" if detail else base
    raise NotImplementedError(msg) from OpSystemFlepimop2Error(
        msg, code=ErrorCode.UNSUPPORTED_FEATURE
    )


def raise_state_shape_error(*, name: str, expected: str, got: object) -> None:
    """Raise a standardized state/time array shape error.

    Args:
        name: Name of the array (for error messages).
        expected: Description of the expected shape/value.
        got: Actual value received.

    Raises:
        ValueError: Always.
    """
    msg = f"{name} has an invalid shape/value. Expected {expected}. Got: {got!r}."
    raise ValueError(msg) from OpSystemFlepimop2Error(
        msg, code=ErrorCode.INVALID_STATE_SHAPE
    )


def raise_parameter_error(*, detail: str) -> None:
    """Raise a standardized parameter/type error.

    Args:
        detail: Detail text describing the parameter issue.

    Raises:
        TypeError: Always.
    """
    msg = f"Invalid parameters for op_system.flepimop2 adapter: {detail}"
    raise TypeError(msg) from OpSystemFlepimop2Error(
        msg, code=ErrorCode.INVALID_PARAMETERS
    )
