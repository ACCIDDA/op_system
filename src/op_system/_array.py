"""Shared array namespace discovery for backend-polymorphic code."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from array_api_compat import (  # type: ignore[import-untyped]
    array_namespace as _array_namespace,
)


def array_namespace(value: object) -> Any:  # ruff: ignore[any-type]
    """Return an Array-API-compatible namespace for ``value``.

    ``array-api-compat`` recognizes both arrays that implement
    ``__array_namespace__`` and native arrays such as ``torch.Tensor`` that
    need a compatibility namespace. Optional backends are imported only when
    one of their arrays is supplied.

    Args:
        value: Numerical array whose namespace should be selected.

    Returns:
        The Array-API-compatible namespace for ``value``.

    Raises:
        TypeError: If ``value`` is not a supported array object.
    """
    try:
        return _array_namespace(value)
    except TypeError as error:
        msg = (
            "op_system numerical inputs must be supported array objects; "
            f"got {type(value).__name__}."
        )
        raise TypeError(msg) from error


__all__ = ["array_namespace"]


def _asarray(value: object, xp: Any) -> Any:  # ruff: ignore[any-type]
    """Convert without detaching already-native Torch arrays on older releases.

    Returns:
        Native array with the namespace's existing default dtype/device rules.
    """
    if getattr(xp, "__name__", "") in {"torch", "array_api_compat.torch"}:
        return import_module("torch").as_tensor(value)
    return xp.asarray(value)
