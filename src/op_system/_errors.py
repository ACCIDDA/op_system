"""op_system._errors.

Shared error-raising helpers used across all op_system modules.
"""

from __future__ import annotations

from typing import NoReturn

_INVALID_RHS_SPEC_PREFIX = "Invalid op_system RHS specification."
_INVALID_EXPRESSION_PREFIX = "Invalid op_system expression."
_UNSUPPORTED_FEATURE_PREFIX = "Unsupported op_system feature."


def _raise_invalid_rhs_spec(
    *, missing: list[str] | None = None, detail: str | None = None
) -> NoReturn:
    """Raise a standardized RHS specification error.

    Args:
        missing: Optional list of missing field names.
        detail: Optional additional detail string.

    Raises:
        ValueError: Always.
    """
    parts: list[str] = [_INVALID_RHS_SPEC_PREFIX]
    if missing:
        parts.append(f"Missing required field(s): {sorted(set(missing))}.")
    if detail:
        parts.append(f"Detail: {detail}")
    raise ValueError(" ".join(parts))


def _raise_invalid_expression(*, detail: str) -> NoReturn:
    """Raise a standardized expression error.

    Args:
        detail: Error detail.

    Raises:
        ValueError: Always.
    """
    msg = f"{_INVALID_EXPRESSION_PREFIX} Detail: {detail}"
    raise ValueError(msg)


def _raise_unsupported_feature(*, feature: str, detail: str | None = None) -> NoReturn:
    """Raise a standardized unsupported feature error.

    Args:
        feature: Feature identifier.
        detail: Optional additional detail.

    Raises:
        NotImplementedError: Always.
    """
    msg = f"{_UNSUPPORTED_FEATURE_PREFIX} Feature '{feature}' is not supported."
    if detail:
        msg = f"{msg} Detail: {detail}"
    raise NotImplementedError(msg)
