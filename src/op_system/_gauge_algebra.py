"""Small exact linear-algebra routines for declared scaling exponents."""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _rref(
    rows: Sequence[Sequence[Fraction | int | float]], n_columns: int
) -> tuple[list[list[Fraction]], tuple[int, ...]]:
    """Reduce rational rows without a numerical rank tolerance.

    Returns:
        Reduced rows and pivot-column indices.

    Raises:
        ValueError: If row widths do not match the declared matrix width.
    """
    if n_columns < 0 or any(len(row) != n_columns for row in rows):
        msg = "Scaling matrix rows must match their declared width"
        raise ValueError(msg)
    matrix = [
        [
            value if isinstance(value, Fraction) else Fraction(str(value))
            for value in row
        ]
        for row in rows
    ]
    pivots: list[int] = []
    for column in range(n_columns):
        pivot = next(
            (row for row in range(len(pivots), len(matrix)) if matrix[row][column]),
            None,
        )
        if pivot is None:
            continue
        position = len(pivots)
        matrix[position], matrix[pivot] = matrix[pivot], matrix[position]
        scale = matrix[position][column]
        matrix[position] = [value / scale for value in matrix[position]]
        for row, values in enumerate(matrix):
            if row != position and values[column]:
                coefficient = values[column]
                matrix[row] = [
                    value - coefficient * selected
                    for value, selected in zip(values, matrix[position], strict=True)
                ]
        pivots.append(column)
        if len(pivots) == len(matrix):
            break
    return matrix, tuple(pivots)


def _integer_nullspace(
    rows: Sequence[Sequence[Fraction | int | float]], n_columns: int
) -> tuple[tuple[int, ...], ...]:
    """Return a primitive integer basis for a rational matrix's nullspace.

    Returns:
        One basis vector per free column. Its free entry is positive.
    """
    matrix, pivots = _rref(rows, n_columns)
    result = []
    for free in range(n_columns):
        if free in pivots:
            continue
        vector = [Fraction(0)] * n_columns
        vector[free] = Fraction(1)
        for row, pivot in enumerate(pivots):
            vector[pivot] = -matrix[row][free]
        multiplier = lcm(*(value.denominator for value in vector))
        integers = [int(value * multiplier) for value in vector]
        divisor = reduce(gcd, (abs(value) for value in integers if value), 0)
        result.append(tuple(value // divisor for value in integers))
    return tuple(result)


def _rank(rows: Sequence[Sequence[Fraction | int | float]], n_columns: int) -> int:
    """Return the exact rational rank of the exponent matrix.

    Returns:
        Number of pivot columns.
    """
    return len(_rref(rows, n_columns)[1])
