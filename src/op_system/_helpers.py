"""op_system._helpers.

Shared type-validation and coercion utilities used across op_system modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

from op_system._errors import InvalidRhsSpecError


def _ensure_str_list(x: object, *, name: str) -> list[str]:
    """Ensure *x* is a list of non-empty strings.

    Args:
        x: Value to validate; expected to be a ``list`` or ``tuple`` of
            non-empty ``str`` entries.
        name: Field name used in raised error messages.

    Returns:
        List of stripped, non-empty strings preserving input order.

    Raises:
        InvalidRhsSpecError: If ``x`` is not a list/tuple, or contains a
            non-string or empty/whitespace-only entry.

    Examples:
        >>> _ensure_str_list(["  age ", "vax"], name="axes")
        ['age', 'vax']
        >>> try:
        ...     _ensure_str_list("age", name="axes")
        ... except Exception as exc:
        ...     print(type(exc).__name__)
        InvalidRhsSpecError
        >>> try:
        ...     _ensure_str_list(["age", ""], name="axes")
        ... except Exception as exc:
        ...     print(type(exc).__name__)
        InvalidRhsSpecError
    """
    if not isinstance(x, (list, tuple)):
        raise InvalidRhsSpecError(detail=f"{name} must be a list of strings")
    out: list[str] = []
    for i, v in enumerate(x):
        if not isinstance(v, str) or not v.strip():
            raise InvalidRhsSpecError(detail=f"{name}[{i}] must be a non-empty string")
        out.append(v.strip())
    return out


def _ensure_str_dict(x: object, *, name: str) -> dict[str, str]:
    """Ensure *x* is a dict mapping non-empty strings to non-empty strings.

    Args:
        x: Value to validate; ``None`` is treated as an empty mapping.
        name: Field name used in raised error messages.

    Returns:
        Dict of stripped string keys to stripped non-empty string values.
        ``{}`` if ``x is None``.

    Raises:
        InvalidRhsSpecError: If ``x`` is not a dict, or any key/value is not
            a non-empty string.

    Examples:
        >>> _ensure_str_dict({" age ": " old "}, name="map")
        {'age': 'old'}
        >>> _ensure_str_dict(None, name="map")
        {}
    """
    if x is None:
        return {}
    if not isinstance(x, dict):
        raise InvalidRhsSpecError(detail=f"{name} must be a mapping of string->string")
    out: dict[str, str] = {}
    for k, v in x.items():
        if not isinstance(k, str) or not k.strip():
            raise InvalidRhsSpecError(detail=f"{name} keys must be non-empty strings")
        if not isinstance(v, str) or not v.strip():
            raise InvalidRhsSpecError(
                detail=f"{name}[{k!r}] must be a non-empty string"
            )
        out[k.strip()] = v.strip()
    return out


def _sorted_unique(xs: Iterable[str]) -> tuple[str, ...]:
    """Return a sorted tuple of unique strings from the iterable.

    Args:
        xs: Iterable of strings (may contain duplicates).

    Returns:
        Tuple of unique strings in ascending lexicographic order.

    Examples:
        >>> _sorted_unique(["vax", "age", "vax"])
        ('age', 'vax')
        >>> _sorted_unique([])
        ()
    """
    return tuple(sorted(set(xs)))


def _as_number(x: object, *, name: str) -> float:
    """Ensure *x* is a real number and return it as float.

    ``bool`` values are explicitly rejected (despite being a subclass of
    ``int``) since accepting them would silently coerce ``True``/``False``
    into 1.0/0.0 in numeric fields.

    Args:
        x: Value to coerce; must be ``int`` or ``float`` (not ``bool``).
        name: Field name used in raised error messages.

    Returns:
        ``x`` coerced to ``float``.

    Raises:
        InvalidRhsSpecError: If ``x`` is not a real number.

    Examples:
        >>> _as_number(3, name="x")
        3.0
        >>> _as_number(2.5, name="x")
        2.5
        >>> try:
        ...     _as_number(True, name="x")
        ... except Exception as exc:
        ...     print(type(exc).__name__)
        InvalidRhsSpecError
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise InvalidRhsSpecError(detail=f"{name} must be a number")
    return float(x)


def _ensure_mapping(x: object, *, name: str) -> Mapping[str, Any]:
    """Ensure *x* is a mapping.

    Args:
        x: Value to validate; must be a ``dict``.
        name: Field name used in raised error messages.

    Returns:
        ``x`` as a ``Mapping[str, Any]``.

    Raises:
        InvalidRhsSpecError: If ``x`` is not a ``dict``.

    Examples:
        >>> _ensure_mapping({"a": 1}, name="cfg")
        {'a': 1}
        >>> try:
        ...     _ensure_mapping([1, 2], name="cfg")
        ... except Exception as exc:
        ...     print(type(exc).__name__)
        InvalidRhsSpecError
    """
    if not isinstance(x, dict):
        raise InvalidRhsSpecError(detail=f"{name} must be a mapping")
    return x
