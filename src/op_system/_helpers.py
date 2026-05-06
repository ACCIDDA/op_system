"""op_system._helpers.

Shared type-validation and coercion utilities used across op_system modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

from op_system._errors import _raise_invalid_rhs_spec


def _ensure_str_list(x: object, *, name: str) -> list[str]:
    """Ensure *x* is a list of non-empty strings.

    Returns:
        List of stripped, non-empty strings.
    """
    if not isinstance(x, (list, tuple)):
        _raise_invalid_rhs_spec(detail=f"{name} must be a list of strings")
    out: list[str] = []
    for i, v in enumerate(x):
        if not isinstance(v, str) or not v.strip():
            _raise_invalid_rhs_spec(detail=f"{name}[{i}] must be a non-empty string")
        out.append(v.strip())
    return out


def _ensure_str_dict(x: object, *, name: str) -> dict[str, str]:
    """Ensure *x* is a dict mapping non-empty strings to non-empty strings.

    Returns:
        Dict of stripped string keys to stripped non-empty string values.
    """
    if x is None:
        return {}
    if not isinstance(x, dict):
        _raise_invalid_rhs_spec(detail=f"{name} must be a mapping of string->string")
    out: dict[str, str] = {}
    for k, v in x.items():
        if not isinstance(k, str) or not k.strip():
            _raise_invalid_rhs_spec(detail=f"{name} keys must be non-empty strings")
        if not isinstance(v, str) or not v.strip():
            _raise_invalid_rhs_spec(detail=f"{name}[{k!r}] must be a non-empty string")
        out[k.strip()] = v.strip()
    return out


def _sorted_unique(xs: Iterable[str]) -> tuple[str, ...]:
    """Return a sorted tuple of unique strings from the iterable."""
    return tuple(sorted(set(xs)))


def _as_number(x: object, *, name: str) -> float:
    """Ensure *x* is a real number and return it as float.

    Returns:
        Input coerced to float.
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        _raise_invalid_rhs_spec(detail=f"{name} must be a number")
    return float(x)


def _ensure_mapping(x: object, *, name: str) -> Mapping[str, Any]:
    """Ensure *x* is a mapping.

    Returns:
        Mapping view of the input.
    """
    if not isinstance(x, dict):
        _raise_invalid_rhs_spec(detail=f"{name} must be a mapping")
    return x
