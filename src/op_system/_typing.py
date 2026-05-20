"""op_system._typing.

Lightweight Array-API structural typing for op_system inputs and outputs.

Mirrors :class:`flepimop2.typing.Array` so that producers and consumers
across both packages can share the same structural contract without an
import dependency between them.

The single load-bearing capability is :meth:`Array.__array_namespace__`,
which is what allows :func:`op_system.compile.compile_rhs` to obtain the
operations namespace at *call time* from the input arrays themselves —
removing the need for a compile-time backend selector and giving users a
single, namespace-polymorphic, trace-pure ``eval_fn``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Array(Protocol):
    """Structural Array-API protocol.

    Any object whose runtime type implements ``shape``, ``dtype``,
    ``__array_namespace__`` and ``item`` satisfies this protocol. NumPy
    >= 2.0 ndarrays, JAX arrays (concrete and traced), and PyTorch tensors
    (via the array-api compat layer) all qualify.

    The namespace returned by ``__array_namespace__`` is the *only* gate
    op_system uses to dispatch operations: input → namespace → output in
    that same namespace. No conversion, no coercion, no compile-time
    backend selector.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the array shape as a tuple of ``int``."""
        ...

    @property
    def dtype(self) -> object:
        """Return the array dtype object (namespace-specific)."""
        ...

    def __array_namespace__(  # noqa: PLW3201
        self,
        *,
        api_version: Any = None,  # noqa: ANN401
    ) -> object:
        """Return the Array-API namespace that owns this array."""
        ...

    def item(self) -> Any:  # noqa: ANN401
        """Return the underlying scalar value (for 0-d arrays)."""
        ...


__all__ = ["Array"]
