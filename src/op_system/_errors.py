"""op_system._errors.

Shared exception types raised by op_system normalization and parsing modules.

Each exception subclasses a built-in (``ValueError`` / ``NotImplementedError``)
so existing ``except ValueError`` / ``except NotImplementedError`` sites in
downstream code continue to work, while new code can catch the more specific
subclasses when finer-grained handling is desired.
"""

from __future__ import annotations

_INVALID_RHS_SPEC_PREFIX = "Invalid op_system RHS specification."
_INVALID_EXPRESSION_PREFIX = "Invalid op_system expression."
_UNSUPPORTED_FEATURE_PREFIX = "Unsupported op_system feature."


class InvalidRhsSpecError(ValueError):
    """Raised when an op_system RHS spec is structurally invalid.

    Attributes:
        missing: Optional list of missing required field names.
        detail: Optional human-readable detail describing the violation.
    """

    def __init__(
        self,
        *,
        missing: list[str] | None = None,
        detail: str | None = None,
    ) -> None:
        self.missing = list(missing) if missing else None
        self.detail = detail
        parts: list[str] = [_INVALID_RHS_SPEC_PREFIX]
        if missing:
            parts.append(f"Missing required field(s): {sorted(set(missing))}.")
        if detail:
            parts.append(f"Detail: {detail}")
        super().__init__(" ".join(parts))


class InvalidExpressionError(ValueError):
    """Raised when an op_system expression cannot be parsed/validated.

    Attributes:
        detail: Human-readable detail describing the parse/validation failure.
    """

    def __init__(self, *, detail: str) -> None:
        self.detail = detail
        super().__init__(f"{_INVALID_EXPRESSION_PREFIX} Detail: {detail}")


class UnsupportedFeatureError(NotImplementedError):
    """Raised when a spec references an op_system feature that is not yet supported.

    Attributes:
        feature: Identifier for the unsupported feature.
        detail: Optional additional detail.
    """

    def __init__(self, *, feature: str, detail: str | None = None) -> None:
        self.feature = feature
        self.detail = detail
        msg = f"{_UNSUPPORTED_FEATURE_PREFIX} Feature {feature!r} is not supported."
        if detail:
            msg = f"{msg} Detail: {detail}"
        super().__init__(msg)
