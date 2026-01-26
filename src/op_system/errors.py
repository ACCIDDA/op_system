"""
Core error types and helpers for op_system.

Design intent:
- Domain-agnostic: no imports from flepimop2 or other optional adapters.
- Lean on built-in exception classes for ergonomics (ValueError/TypeError/etc.).
- Provide machine-readable error codes via a single lightweight base error that
  can be used as an exception cause for structured handling.

Contract:
- Public raiser helpers raise built-in exceptions and chain an OpSystemError as
  the cause, carrying an ErrorCode.
- Callers that want structured handling can catch built-ins and inspect
  `exc.__cause__` for an OpSystemError (and its `code`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class ErrorCode(StrEnum):
    """Machine-readable classification for op_system failures."""

    INVALID_RHS_SPEC = "invalid_rhs_spec"
    INVALID_EXPRESSION = "invalid_expression"
    COMPILATION_FAILED = "compilation_failed"
    INVALID_STATE_SHAPE = "invalid_state_shape"
    INVALID_PARAMETERS = "invalid_parameters"
    UNSUPPORTED_FEATURE = "unsupported_feature"


class OpSystemError(Exception):
    """Lightweight, structured error carrying an ErrorCode.

    This is intentionally not raised directly by most core APIs. Instead, core
    helpers raise built-in exceptions (ValueError/TypeError/etc.) and set an
    OpSystemError as the exception cause (`raise X from OpSystemError(...)`).
    """

    def __init__(self, message: str, *, code: ErrorCode | None = None) -> None:
        """
        Initialize OpSystemError.

        Args:
            message: Human-readable error message.
            code: Optional ErrorCode classifying the error.
        """
        super().__init__(message)
        self.code: ErrorCode | None = code


# -----------------------------------------------------------------------------
# Standardized message prefixes
# -----------------------------------------------------------------------------

_INVALID_SPEC_PREFIX: Final[str] = "Invalid op_system RHS specification."
_INVALID_EXPR_PREFIX: Final[str] = "Invalid op_system expression."
_COMPILATION_PREFIX: Final[str] = "op_system compilation failed."
_UNSUPPORTED_PREFIX: Final[str] = "Unsupported op_system feature."
_INVALID_PARAMS_PREFIX: Final[str] = "Invalid parameters for op_system."


# -----------------------------------------------------------------------------
# Raiser helpers (raise built-ins; chain OpSystemError with code)
# -----------------------------------------------------------------------------


def raise_invalid_rhs_spec(
    *,
    missing: list[str] | None = None,
    detail: str | None = None,
) -> None:
    """Raise a standardized RHS specification error.

    Raises:
        ValueError: Always, chained from OpSystemError(code=INVALID_RHS_SPEC).
    """
    parts: list[str] = [_INVALID_SPEC_PREFIX]
    if missing:
        parts.append(f"Missing required field(s): {sorted(set(missing))}.")
    if detail:
        parts.append(f"Detail: {detail}")
    msg = " ".join(parts)

    raise ValueError(msg) from OpSystemError(msg, code=ErrorCode.INVALID_RHS_SPEC)


def raise_invalid_expression(*, detail: str) -> None:
    """Raise a standardized expression error.

    Raises:
        ValueError: Always, chained from OpSystemError(code=INVALID_EXPRESSION).
    """
    msg = f"{_INVALID_EXPR_PREFIX} Detail: {detail}"
    raise ValueError(msg) from OpSystemError(msg, code=ErrorCode.INVALID_EXPRESSION)


def raise_compilation_error(*, detail: str) -> None:
    """Raise a standardized compilation error.

    Raises:
        RuntimeError: Always, chained from OpSystemError(code=COMPILATION_FAILED).
    """
    msg = f"{_COMPILATION_PREFIX} Detail: {detail}"
    raise RuntimeError(msg) from OpSystemError(msg, code=ErrorCode.COMPILATION_FAILED)


def raise_state_shape_error(*, name: str, expected: str, got: object) -> None:
    """Raise a standardized state shape/value error.

    Raises:
        ValueError: Always, chained from OpSystemError(code=INVALID_STATE_SHAPE).
    """
    msg = f"{name} has an invalid shape/value. Expected {expected}. Got: {got!r}."
    raise ValueError(msg) from OpSystemError(msg, code=ErrorCode.INVALID_STATE_SHAPE)


def raise_parameter_error(*, detail: str) -> None:
    """Raise a standardized parameter/type error.

    Raises:
        TypeError: Always, chained from OpSystemError(code=INVALID_PARAMETERS).
    """
    msg = f"{_INVALID_PARAMS_PREFIX} {detail}"
    raise TypeError(msg) from OpSystemError(msg, code=ErrorCode.INVALID_PARAMETERS)


def raise_unsupported_feature(*, feature: str, detail: str | None = None) -> None:
    """Raise a standardized unsupported feature error.

    Raises:
        NotImplementedError: Chained from OpSystemError(code=UNSUPPORTED_FEATURE).
    """
    msg = f"{_UNSUPPORTED_PREFIX} Feature '{feature}' is not supported."
    if detail:
        msg = f"{msg} Detail: {detail}"
    raise NotImplementedError(msg) from OpSystemError(
        msg, code=ErrorCode.UNSUPPORTED_FEATURE
    )
