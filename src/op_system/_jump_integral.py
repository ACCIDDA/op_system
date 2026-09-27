"""Portable reference semantics for conservative jump-integral operators.

A jump kernel is a row-source, column-target matrix of non-negative off-diagonal
rate densities. Direction masks select permitted targets. Continuous axes then
multiply each target column by its quadrature weight. The diagonal is the
negative row sum, producing a conservative generator ``Q`` and the contribution
``rate * state @ Q`` along the declared axis.

Only reflecting/truncated boundaries are currently defined: jumps outside the
declared coordinates do not occur, and a source with no permitted in-domain
target has a zero generator row. Unsupported boundary behavior must be rejected
rather than guessed by an engine.
"""

from __future__ import annotations

from typing import Any

from op_system._array import array_namespace
from op_system._axis_kernel import axis_kernel_generator_rhs

JUMP_INTEGRAL_FORMS = frozenset({"matrix"})
JUMP_INTEGRAL_DIRECTIONS = frozenset({"up", "down", "both"})
JUMP_INTEGRAL_BOUNDARIES = frozenset({"reflecting"})
_AXIS_TYPES = frozenset({"categorical", "ordinal", "continuous"})


def _validate_static_contract(
    *,
    axis_type: str,
    direction: str,
    boundary: str,
) -> None:
    """Validate static jump-integral choices.

    Raises:
        ValueError: If a choice is unsupported or unordered categorical data
            is given a directional jump.
    """
    if axis_type not in _AXIS_TYPES:
        msg = f"unknown jump_integral axis type {axis_type!r}"
        raise ValueError(msg)
    if direction not in JUMP_INTEGRAL_DIRECTIONS:
        msg = f"unknown jump_integral direction {direction!r}"
        raise ValueError(msg)
    if boundary not in JUMP_INTEGRAL_BOUNDARIES:
        msg = f"unknown jump_integral boundary {boundary!r}"
        raise ValueError(msg)
    if axis_type == "categorical" and direction != "both":
        msg = "categorical jump_integral axes support only direction='both'"
        raise ValueError(msg)


def _device_kwargs(array: Any) -> dict[str, Any]:  # ruff: ignore[any-type]
    """Return an Array-API device keyword when the array exposes one."""
    device = getattr(array, "device", None)
    return {} if device is None else {"device": device}


def jump_integral_generator(
    kernel: Any,  # ruff: ignore[any-type]
    *,
    axis_type: str,
    direction: str = "both",
    boundary: str = "reflecting",
    quadrature_weights: Any | None = None,  # ruff: ignore[any-type]
) -> Any:  # ruff: ignore[any-type]
    """Build the conservative row-source generator for a jump kernel.

    ``kernel[i, j]`` is the non-negative rate density from source coordinate
    ``i`` to target coordinate ``j``. Its diagonal is ignored. ``up`` keeps
    only ``j > i``; ``down`` keeps only ``j < i``; ``both`` keeps every
    off-diagonal entry. For a continuous axis each target column ``j`` is
    multiplied by ``quadrature_weights[j]`` before the diagonal loss is set.

    Args:
        kernel: Square source-by-target rate-density matrix.
        axis_type: ``categorical``, ``ordinal``, or ``continuous``.
        direction: Permitted movement in declared coordinate order.
        boundary: Boundary contract. Only ``reflecting`` is currently defined.
        quadrature_weights: Positive target weights required for continuous
            axes and forbidden for categorical or ordinal axes.

    Returns:
        A generator in the kernel's Array-API namespace. Rows sum to zero.

    Raises:
        ValueError: If static choices, shapes, or quadrature usage are invalid.

    Notes:
        Call :func:`validate_jump_integral_kernel` with concrete parameter
        values at an orchestration boundary to check finiteness and signs.
        This builder deliberately keeps array values dynamic under transforms.
    """
    _validate_static_contract(
        axis_type=axis_type,
        direction=direction,
        boundary=boundary,
    )
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        msg = "jump_integral kernel must be square"
        raise ValueError(msg)

    xp = array_namespace(kernel)
    upper = xp.triu(xp.ones_like(kernel, dtype=xp.bool), k=1)
    lower = xp.tril(xp.ones_like(kernel, dtype=xp.bool), k=-1)
    if direction == "up":
        allowed = upper
    elif direction == "down":
        allowed = lower
    else:
        allowed = xp.logical_or(upper, lower)
    rates = xp.where(allowed, kernel, xp.zeros_like(kernel))

    if axis_type == "continuous":
        if quadrature_weights is None:
            msg = "continuous jump_integral axes require quadrature_weights"
            raise ValueError(msg)
        weights = xp.asarray(
            quadrature_weights,
            dtype=kernel.dtype,
            **_device_kwargs(kernel),
        )
        if weights.ndim != 1 or weights.shape[0] != kernel.shape[0]:
            msg = "quadrature_weights must have one entry per kernel target"
            raise ValueError(msg)
        rates *= xp.reshape(weights, (1, weights.shape[0]))
    elif quadrature_weights is not None:
        msg = "quadrature_weights are only valid for continuous jump_integral axes"
        raise ValueError(msg)

    row_sums = xp.sum(rates, axis=1)
    identity = xp.eye(
        kernel.shape[0],
        dtype=kernel.dtype,
        **_device_kwargs(kernel),
    )
    return rates - identity * xp.reshape(row_sums, (row_sums.shape[0], 1))


def jump_integral_rhs(  # ruff: ignore[too-many-arguments]
    state: Any,  # ruff: ignore[any-type]
    kernel: Any,  # ruff: ignore[any-type]
    *,
    axis: int,
    axis_type: str,
    rate: Any = 1.0,  # ruff: ignore[any-type]
    direction: str = "both",
    boundary: str = "reflecting",
    quadrature_weights: Any | None = None,  # ruff: ignore[any-type]
) -> Any:  # ruff: ignore[any-type]
    """Return a conservative jump-integral derivative contribution.

    Args:
        state: State array containing the jump axis.
        kernel: Source-by-target rate-density matrix.
        axis: Position of the jump axis in ``state``.
        axis_type: ``categorical``, ``ordinal``, or ``continuous``.
        rate: Scalar multiplier with inverse-time units after any continuous
            target quadrature has been applied.
        direction: ``up``, ``down``, or ``both``.
        boundary: Boundary contract; currently only ``reflecting``.
        quadrature_weights: Target quadrature weights for continuous axes.

    Returns:
        Derivative contribution with the same shape as ``state``.
    """
    generator = jump_integral_generator(
        kernel,
        axis_type=axis_type,
        direction=direction,
        boundary=boundary,
        quadrature_weights=quadrature_weights,
    )
    return axis_kernel_generator_rhs(state, generator, axis=axis, velocity=rate)


def validate_jump_integral_kernel(  # ruff: ignore[complex-structure, too-many-arguments, too-many-branches]
    kernel: Any,  # ruff: ignore[any-type]
    *,
    axis_type: str,
    direction: str = "both",
    boundary: str = "reflecting",
    quadrature_weights: Any | None = None,  # ruff: ignore[any-type]
    tolerance: float = 1e-9,
) -> list[str]:
    """Return problems with concrete jump-kernel values (empty if valid).

    This eager validation belongs at parameter resolution or another host
    orchestration boundary, not inside a traced numerical solve.

    Args:
        kernel: Candidate source-by-target rate-density matrix.
        axis_type: ``categorical``, ``ordinal``, or ``continuous``.
        direction: ``up``, ``down``, or ``both``.
        boundary: Boundary contract; currently only ``reflecting``.
        quadrature_weights: Target quadrature weights for continuous axes.
        tolerance: Absolute tolerance for sign and diagonal checks.

    Returns:
        Human-readable problems; an empty list means the values conform.

    Raises:
        ValueError: If a static contract choice is unsupported.
    """
    try:
        _validate_static_contract(
            axis_type=axis_type,
            direction=direction,
            boundary=boundary,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        return ["kernel must be square"]
    xp = array_namespace(kernel)
    problems: list[str] = []
    if bool(xp.any(xp.logical_not(xp.isfinite(kernel)))):
        problems.append("kernel entries must be finite")
    if bool(xp.any(kernel < -tolerance)):
        problems.append("kernel entries must be nonnegative")
    if bool(xp.any(xp.abs(xp.linalg.diagonal(kernel)) > tolerance)):
        problems.append("kernel diagonal entries must be zero")

    if axis_type == "continuous":
        if quadrature_weights is None:
            problems.append("continuous axes require quadrature_weights")
        else:
            weights = xp.asarray(
                quadrature_weights,
                dtype=kernel.dtype,
                **_device_kwargs(kernel),
            )
            if weights.ndim != 1 or weights.shape[0] != kernel.shape[0]:
                problems.append("quadrature_weights must match kernel size")
            else:
                if bool(xp.any(xp.logical_not(xp.isfinite(weights)))):
                    problems.append("quadrature_weights must be finite")
                if bool(xp.any(weights <= 0.0)):
                    problems.append("quadrature_weights must be positive")
    elif quadrature_weights is not None:
        problems.append("quadrature_weights require a continuous axis")
    return problems
