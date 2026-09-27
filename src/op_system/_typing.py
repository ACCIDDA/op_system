"""op_system._typing.

Lightweight Array-API structural typing for op_system inputs and outputs.

Mirrors :class:`flepimop2.typing.Array` so that producers and consumers
across both packages can share the same structural contract without an
import dependency between them.

Runtime namespace discovery uses :func:`array_api_compat.array_namespace`,
which accepts standard-compliant arrays and native arrays such as
``torch.Tensor``. The stricter :class:`Array` protocol remains useful for
static typing when a producer exposes ``__array_namespace__`` directly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Array(Protocol):
    """Structural Array-API protocol.

    Any object whose runtime type implements ``shape``, ``dtype``,
    ``__array_namespace__`` and ``item`` satisfies this protocol. NumPy
    >= 2.0 ndarrays and JAX arrays (concrete and traced) qualify directly.

    Runtime evaluation is deliberately broader: it discovers namespaces via
    :func:`array_api_compat.array_namespace`, so native arrays such as
    ``torch.Tensor`` are accepted even though they do not structurally satisfy
    this protocol. No compile-time backend selector is needed.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> object: ...

    def __array_namespace__(  # ruff: ignore[bad-dunder-method-name]
        self,
        *,
        api_version: Any = None,  # ruff: ignore[any-type]
    ) -> object: ...

    def item(self) -> Any: ...  # ruff: ignore[any-type]


__all__ = ["Array"]
