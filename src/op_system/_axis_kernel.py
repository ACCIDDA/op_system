"""op_system._axis_kernel — reference semantics for ``axis_kernel`` operators.

An ``axis_kernel`` operator moves mass along one axis with a matrix-valued
parameter instead of one coordinate-pinned transition per coordinate pair.
Coordinate-pinned transitions make compile time grow with states times
transitions, so dense routing on a fine axis is otherwise impractical (#206).

Two forms are supported:

``generator``
    ``d x / d t += velocity * x @ G`` along the axis, where ``G`` has
    nonnegative off-diagonal rates (rows are sources) and rows summing to
    zero. Mass along the axis is conserved.

``stochastic``
    A flux ``F`` (computed by the engine, e.g. a vaccination-status transfer
    along a second axis) is redistributed along the kernel axis by the
    row-stochastic matrix ``K``: the target slice gains ``F @ K - F`` on top
    of the ordinary transfer. Mass is conserved when rows of ``K`` sum to one.

These functions are backend-agnostic (Array API) reference implementations so
engines share one definition and its tests.
"""

from __future__ import annotations

from typing import Any

AXIS_KERNEL_FORMS = frozenset({"generator", "stochastic"})


def _namespace(value: Any) -> Any:  # ruff: ignore[any-type]
    namespace = getattr(value, "__array_namespace__", None)
    if namespace is None:
        msg = "axis_kernel inputs must implement __array_namespace__"
        raise TypeError(msg)
    return namespace()


def axis_kernel_generator_rhs(
    state: Any,  # ruff: ignore[any-type]
    generator: Any,  # ruff: ignore[any-type]
    *,
    axis: int,
    velocity: Any = 1.0,  # ruff: ignore[any-type]
) -> Any:  # ruff: ignore[any-type]
    """Return ``velocity * state @ generator`` applied along ``axis``.

    Args:
        state: Array with the kernel axis at position ``axis``.
        generator: Square matrix over the kernel axis; rows are sources.
        axis: Position of the kernel axis in ``state``.
        velocity: Scalar multiplier (for example a waning rate).

    Returns:
        The derivative contribution, with the same shape as ``state``.
    """
    xp = _namespace(state)
    moved = xp.moveaxis(state, axis, -1)
    return xp.moveaxis(velocity * (moved @ generator), -1, axis)


def axis_kernel_redistribute(
    flux: Any,  # ruff: ignore[any-type]
    kernel: Any,  # ruff: ignore[any-type]
    *,
    axis: int = -1,
) -> Any:  # ruff: ignore[any-type]
    """Return ``flux @ kernel - flux`` along ``axis`` for a transferred flux.

    Args:
        flux: Per-coordinate flux already moved by an ordinary transfer.
        kernel: Row-stochastic matrix over the kernel axis.
        axis: Position of the kernel axis in ``flux``.

    Returns:
        The redistribution to add to the target slice.
    """
    xp = _namespace(flux)
    moved = xp.moveaxis(flux, axis, -1)
    return xp.moveaxis(moved @ kernel - moved, -1, axis)


def validate_axis_kernel_matrix(
    matrix: Any,  # ruff: ignore[any-type]
    *,
    form: str,
    tolerance: float = 1e-9,
) -> list[str]:
    """Return problems with a numeric kernel matrix for ``form`` (empty if valid).

    Args:
        matrix: Candidate square matrix.
        form: ``"generator"`` or ``"stochastic"``.
        tolerance: Absolute tolerance for sign and row-sum checks.

    Returns:
        Human-readable problems; an empty list means the matrix is valid.

    Raises:
        ValueError: If ``form`` is not a supported axis-kernel form.
    """
    if form not in AXIS_KERNEL_FORMS:
        msg = f"unknown axis_kernel form {form!r}"
        raise ValueError(msg)
    xp = _namespace(matrix)
    problems: list[str] = []
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return ["matrix must be square"]
    n = matrix.shape[0]
    off_diagonal = matrix * (1.0 - xp.eye(n, dtype=matrix.dtype))
    if bool(xp.any(off_diagonal < -tolerance)):
        problems.append("off-diagonal entries must be nonnegative")
    row_sums = xp.sum(matrix, axis=1)
    if form == "generator":
        if bool(xp.any(xp.abs(row_sums) > tolerance)):
            problems.append("generator rows must sum to zero")
    else:
        if bool(xp.any(xp.linalg.diagonal(matrix) < -tolerance)):
            problems.append("stochastic diagonal entries must be nonnegative")
        if bool(xp.any(xp.abs(row_sums - 1.0) > tolerance)):
            problems.append("stochastic rows must sum to one")
    return problems
