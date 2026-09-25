# flepimop2-op_system: Operator-Partitioned System Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Sparse ``[axis, axis]`` tables for op_system routing transitions (#88).

A routing transition such as

.. code-block:: yaml

    - from: X[age, vax=u, imm:i]
      to:   X[age, vax=v, imm:j]
      rate: nu[time, age] * eta[time, imm:i, imm:j]

requests ``eta`` with axes ``(time, imm, imm)``. ``sparse_table`` builds
that dense array from a declared support: each entry names one
``(imm_from, imm_to)`` pair and gives its value, either a number or any
parameter configuration (for example a per-day CSV series requested with
the leading ``time`` axis). Entries outside the support are structurally
zero.

.. code-block:: yaml

    eta:
      module: sparse_table
      indices: [imm, imm]
      entries:
        - index: [x0, x3]
          value: {module: fixed, value: 0.2}
        - index: [x5, x7]
          value: 1.0
"""

__all__ = ["SparseTableEntry", "SparseTableParameter"]

from typing import TYPE_CHECKING, Any

import numpy as np
from flepimop2.axis import AxisCollection
from flepimop2.parameter.abc import (
    ParameterABC,
    ParameterRequest,
    ParameterValue,
    build,
)
from flepimop2.typing import IdentifierString
from pydantic import BaseModel, PrivateAttr, model_validator

if TYPE_CHECKING:
    from types import EllipsisType


class SparseTableEntry(BaseModel):
    """One support entry of a sparse table.

    Attributes:
        index: One coordinate label per table axis, e.g. ``["x0", "x3"]``.
        value: A number, or a parameter configuration sampled with the
            request's leading axes (for example ``(time,)``).
    """

    index: tuple[str, ...]
    value: float | dict[str, Any]


class SparseTableParameter(ParameterABC, module="sparse_table"):
    """Dense array over ``[..., *indices]`` assembled from a declared support.

    Attributes:
        indices: The table axes, which must be the trailing axes of the
            request (for example ``[imm, imm]`` for a request over
            ``(time, imm, imm)``). An axis may repeat.
        entries: The support. Each index appears at most once.

    Examples:
        >>> from flepimop2.axis import Axis, AxisCollection
        >>> from flepimop2.parameter.abc import ParameterRequest
        >>> from flepimop2.parameter.sparse_table import SparseTableParameter
        >>> imm = Axis(
        ...     name="imm", kind="categorical", size=3, labels=("x0", "x1", "x2")
        ... )
        >>> table = SparseTableParameter(
        ...     indices=("imm", "imm"),
        ...     entries=[{"index": ["x0", "x2"], "value": 0.5}],
        ... )
        >>> request = ParameterRequest(name="eta", axes=("imm", "imm"))
        >>> table.sample(axes=AxisCollection({"imm": imm}), request=request).value
        array([[0. , 0. , 0.5],
               [0. , 0. , 0. ],
               [0. , 0. , 0. ]])
        >>> table.support(AxisCollection({"imm": imm}))
        ((0, 2),)
    """

    indices: tuple[IdentifierString, ...]
    entries: list[SparseTableEntry]
    _nested: dict[int, ParameterABC] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _check_entries(self) -> "SparseTableParameter":
        """Reject empty tables, wrong index lengths, and duplicate indices.

        Returns:
            The validated table.

        Raises:
            ValueError: If the table is malformed.
        """
        if not self.indices:
            msg = "sparse_table requires at least one entry in 'indices'."
            raise ValueError(msg)
        seen: set[tuple[str, ...]] = set()
        for entry in self.entries:
            if len(entry.index) != len(self.indices):
                msg = (
                    f"sparse_table entry {list(entry.index)} has "
                    f"{len(entry.index)} labels but indices are {list(self.indices)}."
                )
                raise ValueError(msg)
            if entry.index in seen:
                msg = f"sparse_table entry {list(entry.index)} is declared twice."
                raise ValueError(msg)
            seen.add(entry.index)
        return self

    def support(self, axes: AxisCollection) -> tuple[tuple[int, ...], ...]:
        """Integer positions of the support entries along the table axes.

        Args:
            axes: Resolved runtime axes that label the table axes.

        Returns:
            One tuple of positions per entry, in declaration order.
        """
        return tuple(self._positions(axes, entry) for entry in self.entries)

    def _positions(
        self, axes: AxisCollection, entry: SparseTableEntry
    ) -> tuple[int, ...]:
        positions = []
        for axis_name, label in zip(self.indices, entry.index, strict=True):
            labels = axes[axis_name].labels
            if labels is None or label not in labels:
                msg = (
                    f"sparse_table entry {list(entry.index)}: {label!r} is not a "
                    f"label of axis {axis_name!r} ({labels})."
                )
                raise ValueError(msg)
            positions.append(labels.index(label))
        return tuple(positions)

    def _entry_value(
        self,
        position: int,
        entry: SparseTableEntry,
        *,
        axes: AxisCollection,
        request: ParameterRequest,
        leading: tuple[IdentifierString, ...],
    ) -> np.ndarray:
        if not isinstance(entry.value, dict):
            return np.asarray(float(entry.value))
        nested = self._nested.get(position)
        if nested is None:
            nested = build(entry.value)
            self._nested[position] = nested
        name = "__".join((request.name, *entry.index))
        sub_request = ParameterRequest(name=name, axes=leading, broadcast=True)
        return np.asarray(
            nested.sample(axes=axes, request=sub_request).value, dtype=np.float64
        )

    def sample(
        self,
        *,
        axes: AxisCollection | None = None,
        request: ParameterRequest | None = None,
    ) -> ParameterValue:
        """Return the dense table for ``request``.

        Args:
            axes: Resolved runtime axes; required.
            request: The system's request; its trailing axes must equal
                ``indices``.

        Returns:
            A value over ``request.axes`` that is zero off the support.

        Raises:
            ValueError: If the request or axes are missing, or the request's
                trailing axes differ from ``indices``, or an entry's value
                does not match the leading axes.
        """
        if axes is None or request is None:
            msg = "sparse_table requires both an AxisCollection and a request."
            raise ValueError(msg)
        requested = tuple(request.axes)
        k = len(self.indices)
        if requested[-k:] != self.indices:
            msg = (
                f"sparse_table for {request.name!r} has indices {list(self.indices)} "
                f"but the request's axes are {list(requested)}."
            )
            raise ValueError(msg)
        leading = requested[:-k]
        shape = axes.resolve_shape(requested)
        lead_shape = shape.sizes[: len(leading)]
        out = np.zeros(shape.sizes, dtype=np.float64)
        for position, entry in enumerate(self.entries):
            where = self._positions(axes, entry)
            value = self._entry_value(
                position, entry, axes=axes, request=request, leading=leading
            )
            if value.shape not in {(), lead_shape}:
                msg = (
                    f"sparse_table entry {list(entry.index)} has shape {value.shape}; "
                    f"expected () or {lead_shape} for leading axes {list(leading)}."
                )
                raise ValueError(msg)
            key: tuple[EllipsisType | int, ...] = (Ellipsis, *where)
            out[key] = value
        return ParameterValue(value=out, shape=shape)
