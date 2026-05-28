"""op_system._block_axes.

Block-axis analysis: verify that declared ``factorize_axes`` satisfy the
block-diagonal separability property required for engine-level
``jax.vmap`` partitioning.

An axis ``a`` is *block-diagonal* (or *factorizable*) when each cell along
``a`` evolves independently — ``dy[..., i, ...]/dt`` depends only on
``y[..., i, ...]`` (and on parameters that are either broadcast across ``a``
or are per-cell along ``a``).  This enables the engine to vmap a single-block
ODE solve over the axis rather than solving one monolithic system.

Public API
----------
- :class:`BlockAxisInfo` - pickle-stable metadata for one block-diagonal axis.
- :func:`analyze_block_axes` - walks IR and checks separability; raises
  :class:`~op_system._errors.UnsupportedFeatureError` on the first violation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from op_system._errors import UnsupportedFeatureError
from op_system._ir import Reduce, Subscript, walk

if TYPE_CHECKING:
    from op_system._ir import Expr
    from op_system._normalize import NormalizedRhs


@dataclass(frozen=True, slots=True)
class BlockAxisInfo:
    """Metadata for one block-diagonal (factorizable) axis.

    A ``BlockAxisInfo`` is attached to
    :class:`~op_system.compile.CompiledRhs` for each axis declared in
    ``spec["factorize_axes"]`` that passes the IR separability check in
    :func:`analyze_block_axes`.  Engines (e.g. the diffrax plugin) consume
    this to partition ODE solves with ``jax.vmap`` over the block axis.

    Attributes:
        name: Axis name string (e.g. ``"loc"``).
        size: Number of elements along the axis.
        state_axis_pos: Maps each state-template base name to the integer
            position of this axis within that template's shape tuple.
            Only templates that carry this axis appear in the dict.
        param_axis_pos: Maps each *shaped* parameter name to the integer
            position of this axis within the *actual runtime array* that the
            engine passes to the eval function, or ``None`` if the parameter
            does not carry this axis (broadcast).  For non-time-varying shaped
            parameters the position is the index in the parameter's axis tuple.
            For time-varying parameters the runtime array has time prepended at
            index 0, so the position equals the index of the block axis in the
            full ``(time, *spatial_axes)`` tuple.  Parameters that are entirely
            scalar (not shaped) do not appear in this dict and are always
            broadcast.

    Note:
        ``BlockAxisInfo`` uses ``dict`` fields and therefore cannot be used
        as a hash key.  It is stored on
        :class:`~op_system.compile.CompiledRhs` with ``hash=False`` and is
        fully pickle-stable.

    Examples:
        >>> info = BlockAxisInfo(
        ...     name="loc",
        ...     size=3,
        ...     state_axis_pos={"S": 1, "I": 1, "R": 1},
        ...     param_axis_pos={"rho": 0, "beta": None},
        ... )
        >>> info.name
        'loc'
        >>> info.size
        3
        >>> info.state_axis_pos["S"]
        1
        >>> info.param_axis_pos["rho"]
        0
        >>> info.param_axis_pos["beta"] is None
        True
    """

    name: str
    size: int
    state_axis_pos: dict[str, int]
    param_axis_pos: dict[str, int | None]
    # Note: for time-varying parameters the position stored here is the
    # index of the block axis in the *actual runtime array* (which has
    # time prepended at index 0).  For non-time-varying shaped parameters
    # the position is simply the index in the parameter's axis tuple.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_axis_size(
    axis_name: str,
    axes_meta: tuple[Mapping[str, Any], ...],
) -> int:
    """Return the number of coordinates for *axis_name* in *axes_meta*.

    Args:
        axis_name: Axis name to look up.
        axes_meta: Tuple of axis metadata dicts from ``rhs.meta["axes"]``.

    Returns:
        Number of coordinates (i.e. the axis size).

    Raises:
        UnsupportedFeatureError: If the axis is not found in *axes_meta* or
            has no ``coords`` list.
    """
    for ax in axes_meta:
        if ax.get("name") == axis_name:
            coords = ax.get("coords")
            if isinstance(coords, (list, tuple)):
                return len(coords)
    raise UnsupportedFeatureError(
        feature=f"factorize_axes={axis_name!r}",
        detail=(
            f"Block axis {axis_name!r} is listed in factorize_axes but was not "
            f"found in the spec axes metadata."
        ),
    )


def _check_expr_separable(
    expr: Expr,
    axis_name: str,
    *,
    label: str,
) -> None:
    """Raise if *expr* contains any cross-block coupling on *axis_name*.

    Cross-block couplings are:

    - A :class:`~op_system._ir.Reduce` node with *axis_name* in its
      ``bindings`` (explicit ``sum_over`` / ``apply_along`` /
      ``integrate_over`` over the block axis).
    - A :class:`~op_system._ir.Subscript` with *axis_name* pinned to a
      literal ``coord`` (e.g. ``S[age, loc:a]`` inside an equation cell
      where ``loc`` is the block axis).

    Args:
        expr: IR expression to walk.
        axis_name: Name of the axis being checked.
        label: Human-readable label for the expression (used in the error
            detail message).

    Raises:
        UnsupportedFeatureError: On the first violation found.
    """
    for node in walk(expr):
        if isinstance(node, Reduce):
            for binding_axis, _ in node.bindings:
                if binding_axis == axis_name:
                    raise UnsupportedFeatureError(
                        feature=f"factorize_axes={axis_name!r}",
                        detail=(
                            f"{label!r} reduces over axis {axis_name!r} "
                            f"(kind={node.kind!r}).  Block-factorizable axes must "
                            f"not appear in any sum_over / apply_along / "
                            f"integrate_over binding."
                        ),
                    )
        if isinstance(node, Subscript):
            for idx in node.indices:
                if idx.axis == axis_name and idx.coord is not None:
                    raise UnsupportedFeatureError(
                        feature=f"factorize_axes={axis_name!r}",
                        detail=(
                            f"{label!r}: subscript {node.name!r} pins block axis "
                            f"{axis_name!r} to literal coord {idx.coord!r}.  "
                            f"Block-factorizable axes must not be pinned to a "
                            f"specific coordinate inside an expression."
                        ),
                    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_param_axis_pos(
    rhs: NormalizedRhs,
    axis_name: str,
) -> dict[str, int | None]:
    """Return the runtime array axis position for each shaped param.

    For non-time-varying params the position equals the index in the
    parameter's reduced axis tuple.  For time-varying params the engine
    receives arrays with ``time`` prepended at index 0, so the position
    is looked up in the full ``(time, *spatial_axes)`` tuple stored in
    ``rhs.time_varying_params``.

    Args:
        rhs: Normalized RHS containing shaped and TV param metadata.
        axis_name: Block axis whose position is being computed.

    Returns:
        Mapping from param name to the integer axis position in the
        *actual* runtime array, or ``None`` if the param does not carry
        this axis.
    """
    tv_full: dict[str, tuple[str, ...]] = dict(rhs.time_varying_params)
    param_axis_pos: dict[str, int | None] = {}
    for param_name, reduced_axes in rhs.shaped_params:
        if param_name in tv_full:
            full = tv_full[param_name]
            param_axis_pos[param_name] = (
                full.index(axis_name) if axis_name in full else None
            )
        else:
            param_axis_pos[param_name] = (
                reduced_axes.index(axis_name) if axis_name in reduced_axes else None
            )
    return param_axis_pos


def analyze_block_axes(rhs: NormalizedRhs) -> tuple[BlockAxisInfo, ...]:
    """Verify separability and return :class:`BlockAxisInfo` for each factorize axis.

    Walks every expression in ``rhs.equations_ir_reduce`` and
    ``rhs.aliases_ir_reduce`` for each axis listed in
    ``rhs.meta["factorize_axes"]``.  Raises
    :class:`~op_system._errors.UnsupportedFeatureError` on the first
    cross-block coupling detected.

    Cross-block couplings that trigger rejection:

    - A :class:`~op_system._ir.Reduce` node with the axis in its
      ``bindings`` (explicit ``sum_over`` / ``apply_along`` /
      ``integrate_over``).
    - A :class:`~op_system._ir.Subscript` with the block axis pinned to a
      literal coord (e.g. ``S[age, loc:a]``).
    - Any operator descriptor whose ``axis`` field matches the block axis
      (a spatial operator acting on the partition axis couples adjacent
      cells and breaks separability).
    - Any non-scalar state template (``template.axes != ()``) that does not
      carry the block axis.  All axis-indexed state templates must include
      every declared ``factorize_axis`` so the engine can partition them
      uniformly.

    Args:
        rhs: Normalized RHS produced by
            :func:`~op_system.specs.normalize_rhs`.

    Returns:
        Tuple of :class:`BlockAxisInfo` — one entry per axis listed in
        ``rhs.meta["factorize_axes"]``.  Returns an empty tuple if no axes
        are declared in ``factorize_axes``.

    Raises:
        UnsupportedFeatureError: On any separability violation.  The
            ``detail`` message identifies the offending expression or
            operator and explains the reason.

    Examples:
        >>> from op_system.specs import normalize_rhs
        >>> spec = {
        ...     "kind": "transitions",
        ...     "state": ["S[age,loc]", "I[age,loc]", "R[age,loc]"],
        ...     "transitions": [
        ...         {
        ...             "from": "S[age,loc]",
        ...             "to": "I[age,loc]",
        ...             "rate": "beta * I[age,loc]",
        ...         },
        ...         {"from": "I[age,loc]", "to": "R[age,loc]", "rate": "gamma"},
        ...     ],
        ...     "axes": [
        ...         {"name": "age", "type": "categorical", "coords": ["y", "o"]},
        ...         {"name": "loc", "type": "categorical", "coords": ["a", "b"]},
        ...     ],
        ...     "factorize_axes": ["loc"],
        ... }
        >>> rhs = normalize_rhs(spec)
        >>> infos = analyze_block_axes(rhs)
        >>> len(infos)
        1
        >>> infos[0].name
        'loc'
        >>> infos[0].size
        2
        >>> infos[0].state_axis_pos["S"]
        1
    """
    factorize_names: tuple[str, ...] = tuple(
        s for s in (rhs.meta.get("factorize_axes") or ()) if isinstance(s, str)
    )
    if not factorize_names:
        return ()

    axes_meta: tuple[Mapping[str, Any], ...] = tuple(rhs.meta.get("axes") or ())
    result: list[BlockAxisInfo] = []

    for axis_name in factorize_names:
        # --- axis size -----------------------------------------------------------
        size = _find_axis_size(axis_name, axes_meta)

        # --- operator check ------------------------------------------------------
        operators_raw = rhs.meta.get("operators") or []
        for op in operators_raw:
            if isinstance(op, Mapping) and op.get("axis") == axis_name:
                op_kind = op.get("kind", "<unknown>")
                raise UnsupportedFeatureError(
                    feature=f"factorize_axes={axis_name!r}",
                    detail=(
                        f"Operator kind={op_kind!r} acts on axis {axis_name!r}.  "
                        f"Block-factorizable axes must not have spatial operators "
                        f"that couple adjacent elements."
                    ),
                )

        # --- state template check ------------------------------------------------
        # Every axis-indexed (non-scalar) template must carry the block axis so the
        # engine can partition all templated state arrays uniformly.
        for tpl in rhs.state_templates:
            if tpl.axes and axis_name not in tpl.axes:
                raise UnsupportedFeatureError(
                    feature=f"factorize_axes={axis_name!r}",
                    detail=(
                        f"State template {tpl.base!r} has axes {tpl.axes!r} which "
                        f"does not include block axis {axis_name!r}.  All non-scalar "
                        f"(axis-indexed) state templates must carry every declared "
                        f"factorize_axis."
                    ),
                )

        # --- IR walk: equations --------------------------------------------------
        for i, expr in enumerate(rhs.equations_ir_reduce):
            if expr is None:
                continue
            state_label = rhs.state_names[i] if i < len(rhs.state_names) else f"eq[{i}]"
            _check_expr_separable(expr, axis_name, label=f"equation({state_label})")

        # --- IR walk: aliases ----------------------------------------------------
        for alias_name, expr in rhs.aliases_ir_reduce.items():
            _check_expr_separable(expr, axis_name, label=f"alias({alias_name})")

        # --- state_axis_pos ------------------------------------------------------
        state_axis_pos: dict[str, int] = {
            tpl.base: tpl.axes.index(axis_name)
            for tpl in rhs.state_templates
            if axis_name in tpl.axes
        }

        # --- param_axis_pos ------------------------------------------------------
        param_axis_pos = _build_param_axis_pos(rhs, axis_name)

        result.append(
            BlockAxisInfo(
                name=axis_name,
                size=size,
                state_axis_pos=state_axis_pos,
                param_axis_pos=param_axis_pos,
            )
        )

    return tuple(result)
