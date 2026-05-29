"""op_system._ir_lower.

Lower typed IR expressions to vectorized Python AST nodes.

This module is the foundation for migrating ``_vectorize`` off the per-cell
AST-rewriting pipeline. It produces buffer-access expressions directly from
the typed IR — skipping per-cell scalar-name expansion entirely — for the
subset of expressions where every templated reference is in pure wildcard
form (e.g. ``S[age, vax]`` rather than ``S[age=ap, vax=unvac]``).

v1 scope (intentionally narrow; callers fall back on
:class:`UnsupportedIRLowering`):

- Expressions composed of :class:`Literal`, :class:`Sym`, arithmetic /
  unary / comparison :class:`Apply` nodes, and :class:`Subscript`
  references to declared templated buffers.
- Every :class:`Subscript` must reference a known templated buffer name
  and use only FREE-axis indices (no coord literals, no placeholders, no
  COORD_SYMBOL bindings). The subscript's axis set must equal the
  buffer's declared axes (in any order) and be a subset of
  ``target_axes``.
- :class:`Reduce` nodes, function-call ``Apply`` nodes whose ``op`` isn't
  a recognized arithmetic/comparison operator, and any unsupported
  subscript shape raise :class:`UnsupportedIRLowering`.

Wiring this into the main vectorize path is left to a follow-up PR; this
module exists so the lowering can be developed and tested in isolation.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from op_system._ir import (
    Apply,
    AxisIndex,
    AxisKind,
    Expr,
    HistoryOp,
    Literal,
    Reduce,
    Subscript,
    Sym,
    classify_axis_index,
    substitute,
    unparse_ir,
    walk,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "UnsupportedIRLoweringError",
    "lift_cell_ir_to_template",
    "lower_subscript_to_buffer",
    "lower_to_vector_ast",
]


class UnsupportedIRLoweringError(NotImplementedError):
    """Raised when an IR expression falls outside the v1 lowering subset.

    Callers should catch this and fall back to the existing per-cell AST
    rewriting path. The detail message identifies the offending node.
    """


# ---------------------------------------------------------------------------
# Operator tables (kept local; mirrored from _ir.ir_to_ast_expr)
# ---------------------------------------------------------------------------

_BINOP: dict[str, type[ast.operator]] = {
    "+": ast.Add,
    "-": ast.Sub,
    "*": ast.Mult,
    "/": ast.Div,
    "%": ast.Mod,
    "pow": ast.Pow,
}

_CMPOP: dict[str, type[ast.cmpop]] = {
    "==": ast.Eq,
    "!=": ast.NotEq,
    "<": ast.Lt,
    "<=": ast.LtE,
    ">": ast.Gt,
    ">=": ast.GtE,
}


def _name(ident: str) -> ast.Name:
    return ast.Name(id=ident, ctx=ast.Load())


# ---------------------------------------------------------------------------
# Subscript lowering
# ---------------------------------------------------------------------------


def lower_subscript_to_buffer(  # noqa: C901
    sub: Subscript,
    *,
    src_axes: tuple[str, ...],
    target_axes: tuple[str, ...],
    axis_names: frozenset[str],
    axis_alias: Mapping[str, str] | None = None,
) -> ast.expr:
    """Lower a wildcard IR :class:`Subscript` to a buffer-access AST.

    The returned AST evaluates to an array that broadcasts cleanly against
    a tensor of shape implied by ``target_axes`` — i.e. it carries a
    singleton (``None``) dimension for every axis in ``target_axes`` not
    present in the subscript and is transposed so kept axes appear in
    ``target_axes`` order.

    Args:
        sub: IR subscript to lower. Must have one :class:`AxisKind.FREE`
            index per element, and its axis set must equal ``src_axes`` as
            a set (any permutation accepted).
        src_axes: Declared axis order of the source buffer
            (the ordering used to flatten ``<name>_buf``).
        target_axes: Axis order of the cell layout the result must
            broadcast against.
        axis_names: Registered axis identifiers; used to classify indices.
        axis_alias: Optional mapping from synthetic axis labels (e.g.
            ``age#ap`` emitted by :func:`_lower_reduce` for same-axis-twice
            bindings) back to their real axis name. Used to match
            subscript labels against ``src_axes`` while preserving the
            synthetic label for ``target_axes`` alignment.

    Returns:
        An ``ast.expr`` accessing ``<sub.name>_buf`` with the necessary
        transpose / size-1 insertions to align with ``target_axes``.

    Raises:
        UnsupportedIRLoweringError: If any index is non-FREE, the index axis
            set doesn't match ``src_axes``, or ``src_axes`` is not a
            subset of ``target_axes``.
    """
    alias = axis_alias or {}
    if len(sub.indices) != len(src_axes):
        msg = (
            f"subscript {sub.name!r} has {len(sub.indices)} indices but "
            f"buffer has {len(src_axes)} axes"
        )
        raise UnsupportedIRLoweringError(msg)

    sub_axes: list[str] = []
    real_sub_axes: list[str] = []
    for idx in sub.indices:
        kind = idx.kind or classify_axis_index(idx, axis_names=axis_names)
        if kind is not AxisKind.FREE:
            msg = (
                f"subscript {sub.name!r} has non-FREE index "
                f"({kind.value}) — v1 lowering supports only wildcard "
                "axis references"
            )
            raise UnsupportedIRLoweringError(msg)
        sub_axes.append(idx.axis)
        real_sub_axes.append(alias.get(idx.axis, idx.axis))

    if set(real_sub_axes) != set(src_axes):
        msg = (
            f"subscript {sub.name!r} axes {tuple(sub_axes)} do not match "
            f"buffer axes {tuple(src_axes)} as a set"
        )
        raise UnsupportedIRLoweringError(msg)

    if not set(sub_axes).issubset(target_axes):
        msg = (
            f"buffer axes {tuple(src_axes)} (labels {tuple(sub_axes)}) not "
            f"a subset of target axes {tuple(target_axes)}"
        )
        raise UnsupportedIRLoweringError(msg)

    buf: ast.expr = _name(f"{sub.name}_buf")

    # Reorder buffer (stored in src_axes order) to sub_axes order if the
    # user wrote them out of declaration order (e.g. ``S[vax, age]``
    # against ``src_axes=(age, vax)``). Use the real-axis view for the
    # match; the resulting buffer carries labels in ``sub_axes`` order.
    if tuple(real_sub_axes) != tuple(src_axes):
        perm = tuple(src_axes.index(a) for a in real_sub_axes)
        buf = _transpose(buf, perm)

    # If sub_axes already cover target_axes in matching order, no further
    # alignment is needed.
    if tuple(sub_axes) == tuple(target_axes):
        return buf

    # Reorder sub_axes to match the order they appear in target_axes
    # before inserting singleton dimensions.
    target_kept = tuple(a for a in target_axes if a in sub_axes)
    if tuple(sub_axes) != target_kept:
        perm = tuple(sub_axes.index(a) for a in target_kept)
        buf = _transpose(buf, perm)

    if tuple(target_kept) == tuple(target_axes):
        return buf

    # Insert ``None`` placeholders for target axes not present.
    elts: list[ast.expr] = []
    for ax in target_axes:
        if ax in target_kept:
            elts.append(ast.Slice(lower=None, upper=None, step=None))
        else:
            elts.append(ast.Constant(value=None))
    return ast.Subscript(
        value=buf,
        slice=ast.Tuple(elts=elts, ctx=ast.Load()),
        ctx=ast.Load(),
    )


def _transpose(node: ast.expr, perm: tuple[int, ...]) -> ast.expr:
    return ast.Call(
        func=ast.Attribute(value=_name("np"), attr="transpose", ctx=ast.Load()),
        args=[
            node,
            ast.Tuple(elts=[ast.Constant(value=p) for p in perm], ctx=ast.Load()),
        ],
        keywords=[],
    )


def _lower_shaped_param_subscript(  # noqa: C901, PLR0911, PLR0912, PLR0915
    sub: Subscript,
    *,
    src_axes: tuple[str, ...],
    target_axes: tuple[str, ...],
    axis_names: frozenset[str],
    axis_alias: Mapping[str, str] | None = None,
) -> ast.expr:
    """Lower a shaped-parameter :class:`Subscript` to a buffer-access AST.

    Unlike :func:`lower_subscript_to_buffer` (which matches subscript
    axes against the source by *set* — permitting user-reordered indices
    like ``S[vax, age]`` against ``src_axes=(age, vax)``), this helper
    matches positionally: ``sub.indices[i]`` corresponds to source
    position ``i``. Positional matching is required for shaped params
    that legally repeat an axis name (e.g. a contact kernel
    ``K[age, age]``) where set-based reordering would be ambiguous.

    The index ``axis`` field acts as the *broadcast label* for the
    corresponding position: when a binding has been relabeled to a
    synthetic axis name (e.g. ``age#ap`` for an apply_along loop
    variable on a same-axis-twice ``K``), the synthetic label appears in
    the index, and the resulting buffer broadcasts to a target layout
    that carries both the original and synthetic axes as distinct
    positions.

    Args:
        sub: IR subscript to lower. Every index must be FREE on a label
            present in ``target_axes`` (literal coords / placeholders
            are not supported).
        src_axes: Declared axis order of the source buffer. Length must
            equal ``len(sub.indices)``.
        target_axes: Axis order of the cell layout the result must
            broadcast against.
        axis_names: Registered axis identifiers (including synthetic
            labels emitted by :func:`_lower_reduce`); used to classify
            indices.
        axis_alias: Optional mapping from synthetic axis labels back to
            real axis names. Used to validate positional consistency
            between ``sub.indices[i].axis`` (after alias resolution)
            and ``src_axes[i]``.

    Returns:
        An ``ast.expr`` accessing ``<sub.name>_buf`` aligned to
        ``target_axes`` via transpose / size-1 insertions.

    Raises:
        UnsupportedIRLoweringError: If any index is non-FREE, the
            position count disagrees, a sub_axes label is not in
            ``target_axes``, or the labels are not distinct (positional
            mode cannot disambiguate duplicate labels at the broadcast
            stage).
    """
    if len(sub.indices) != len(src_axes):
        msg = (
            f"shaped-param subscript {sub.name!r} has {len(sub.indices)} "
            f"indices but buffer has {len(src_axes)} axes"
        )
        raise UnsupportedIRLoweringError(msg)

    # Fast path: all indices are literal integer positions (post-expansion
    # form, e.g. ``gamma[0]`` after ``gamma[age]`` was string-expanded).
    # For axes that are in ``target_axes``, emit a full slice (``:``) so
    # the result vectorizes along that axis and the first-/last-cell AST
    # equality check in ``_vectorize_template_equations`` passes.  For axes
    # NOT in ``target_axes`` (i.e. being unrolled by the caller), emit the
    # literal integer so the access selects the correct element.
    if all(idx.coord is not None and not idx.axis for idx in sub.indices):
        buf_node: ast.expr = _name(f"{sub.name}_buf")
        elts_out: list[ast.expr] = []
        tied_ax_names: list[str] = []
        for ax_name, idx in zip(src_axes, sub.indices, strict=True):
            if ax_name in target_axes:
                elts_out.append(ast.Slice(lower=None, upper=None, step=None))
                tied_ax_names.append(ax_name)
            else:
                elts_out.append(ast.Constant(value=int(idx.coord)))  # type: ignore[arg-type]
        if len(elts_out) == 1:
            sl_node: ast.expr = elts_out[0]
        else:
            sl_node = ast.Tuple(elts=elts_out, ctx=ast.Load())
        accessed: ast.expr = ast.Subscript(
            value=buf_node, slice=sl_node, ctx=ast.Load()
        )
        if not tied_ax_names:
            return accessed
        # Broadcast tied axes into target_axes layout.
        if tuple(tied_ax_names) == tuple(target_axes):
            return accessed
        target_kept = tuple(a for a in target_axes if a in tied_ax_names)
        if tuple(tied_ax_names) != target_kept:
            perm = tuple(tied_ax_names.index(a) for a in target_kept)
            accessed = _transpose(accessed, perm)
        if tuple(target_kept) == tuple(target_axes):
            return accessed
        insert_elts: list[ast.expr] = []
        for ax in target_axes:
            if ax in tied_ax_names:
                insert_elts.append(ast.Slice(lower=None, upper=None, step=None))
            else:
                insert_elts.append(ast.Constant(value=None))
        return ast.Subscript(
            value=accessed,
            slice=ast.Tuple(elts=insert_elts, ctx=ast.Load()),
            ctx=ast.Load(),
        )

    sub_axes: list[str] = []
    alias = axis_alias or {}
    for pos, idx in enumerate(sub.indices):
        kind = idx.kind or classify_axis_index(idx, axis_names=axis_names)
        if kind is not AxisKind.FREE:
            msg = (
                f"shaped-param subscript {sub.name!r} has non-FREE index "
                f"({kind.value}) — v1 lowering supports only wildcard "
                "axis references"
            )
            raise UnsupportedIRLoweringError(msg)
        sub_axes.append(idx.axis)
        real = alias.get(idx.axis, idx.axis)
        if real != src_axes[pos]:
            msg = (
                f"shaped-param subscript {sub.name!r} index {pos} label "
                f"{idx.axis!r} (real {real!r}) does not match buffer "
                f"axis {src_axes[pos]!r} at that position"
            )
            raise UnsupportedIRLoweringError(msg)

    if len(set(sub_axes)) != len(sub_axes):
        msg = (
            f"shaped-param subscript {sub.name!r} has duplicate index "
            f"labels {tuple(sub_axes)}; positional broadcast requires "
            "distinct labels (use synthetic labels for bound positions)"
        )
        raise UnsupportedIRLoweringError(msg)

    if not set(sub_axes).issubset(target_axes):
        msg = (
            f"shaped-param subscript {sub.name!r} labels {tuple(sub_axes)} "
            f"not a subset of target axes {tuple(target_axes)}"
        )
        raise UnsupportedIRLoweringError(msg)

    buf: ast.expr = _name(f"{sub.name}_buf")

    # Positional: source position i carries label sub_axes[i]. Align to
    # target_axes by transposing kept axes into target order and
    # inserting size-1 placeholders for absent target axes.
    if tuple(sub_axes) == tuple(target_axes):
        return buf

    target_kept = tuple(a for a in target_axes if a in sub_axes)
    if tuple(sub_axes) != target_kept:
        perm = tuple(sub_axes.index(a) for a in target_kept)
        buf = _transpose(buf, perm)

    if tuple(target_kept) == tuple(target_axes):
        return buf

    elts: list[ast.expr] = []
    for ax in target_axes:
        if ax in target_kept:
            elts.append(ast.Slice(lower=None, upper=None, step=None))
        else:
            elts.append(ast.Constant(value=None))
    return ast.Subscript(
        value=buf,
        slice=ast.Tuple(elts=elts, ctx=ast.Load()),
        ctx=ast.Load(),
    )


# ---------------------------------------------------------------------------
# Full expression lowering
# ---------------------------------------------------------------------------


def _lower_history_op(  # noqa: PLR0913
    expr: HistoryOp,
    *,
    target_axes: tuple[str, ...],
    buffer_axes: Mapping[str, tuple[str, ...]],
    axis_names: frozenset[str],
    reducible_axes: frozenset[str] = frozenset(),
    axis_weights: Mapping[str, tuple[float, ...]] | None = None,
    axis_coords: Mapping[str, tuple[str, ...]] | None = None,
    axis_types: Mapping[str, str] | None = None,
    shaped_param_axes: Mapping[str, tuple[str, ...]] | None = None,
    axis_alias: Mapping[str, str] | None = None,
    history_signal_id_map: Mapping[tuple[str, str, tuple[tuple[str, str], ...]], int] | None = None,
) -> ast.expr:
    """Lower a HistoryOp to a __hist_query call.

    Args:
        expr: The HistoryOp node.
        history_signal_id_map: Maps (kind, body_repr, options_tuple) to signal_id.
            Required for convolve_history; None triggers raise.
        ... (other args match lower_to_vector_ast)

    Returns:
        ast.Call node calling __hist_query(signal_id, body, **options).

    Raises:
        UnsupportedIRLoweringError: If kind is not convolve_history or if
            signal_id_map is missing.
    """
    if expr.kind in {"history", "delay"}:
        msg = (
            f"{expr.kind} operator requires engine-managed history buffers "
            "and cannot be lowered in v1 (issue #173)"
        )
        raise UnsupportedIRLoweringError(msg)

    if expr.kind != "convolve_history":
        msg = f"unknown HistoryOp kind: {expr.kind!r}"
        raise UnsupportedIRLoweringError(msg)

    if history_signal_id_map is None:
        msg = (
            "convolve_history lowering requires a signal_id_map; "
            "this is a caller error (vectorize path should provide it)"
        )
        raise UnsupportedIRLoweringError(msg)

    # Build lookup key: (kind, body_repr, options_tuple).
    body_repr = unparse_ir(expr.body)
    options_tuple = tuple((k, unparse_ir(v)) for k, v in expr.options)
    lookup_key = (expr.kind, body_repr, options_tuple)
    signal_id = history_signal_id_map.get(lookup_key)
    if signal_id is None:
        msg = (
            f"convolve_history node not found in signal_id_map: {lookup_key}; "
            "this indicates a mismatch between compile.py and lowering traversal order"
        )
        raise UnsupportedIRLoweringError(msg)

    # Lower body expression.
    body_ast = lower_to_vector_ast(
        expr.body,
        target_axes=target_axes,
        buffer_axes=buffer_axes,
        axis_names=axis_names,
        reducible_axes=reducible_axes,
        axis_weights=axis_weights,
        axis_coords=axis_coords,
        axis_types=axis_types,
        shaped_param_axes=shaped_param_axes,
        axis_alias=axis_alias,
        history_signal_id_map=history_signal_id_map,
    )

    # Lower options: Literal → Constant, Sym → Constant(name), else recurse.
    keywords: list[ast.keyword] = []
    for opt_name, opt_expr in expr.options:
        if isinstance(opt_expr, Literal):
            opt_ast = ast.Constant(value=opt_expr.value)
        elif isinstance(opt_expr, Sym):
            # Pass symbolic name as string for runtime dispatch.
            opt_ast = ast.Constant(value=opt_expr.name)
        else:
            # Recursively lower complex option expressions.
            opt_ast = lower_to_vector_ast(
                opt_expr,
                target_axes=target_axes,
                buffer_axes=buffer_axes,
                axis_names=axis_names,
                reducible_axes=reducible_axes,
                axis_weights=axis_weights,
                axis_coords=axis_coords,
                axis_types=axis_types,
                shaped_param_axes=shaped_param_axes,
                axis_alias=axis_alias,
                history_signal_id_map=history_signal_id_map,
            )
        keywords.append(ast.keyword(arg=opt_name, value=opt_ast))

    return ast.Call(
        func=ast.Name(id="__hist_query", ctx=ast.Load()),
        args=[ast.Constant(value=signal_id), body_ast],
        keywords=keywords,
    )


def lower_to_vector_ast(  # noqa: PLR0913
    expr: Expr,
    *,
    target_axes: tuple[str, ...],
    buffer_axes: Mapping[str, tuple[str, ...]],
    axis_names: frozenset[str],
    reducible_axes: frozenset[str] = frozenset(),
    axis_weights: Mapping[str, tuple[float, ...]] | None = None,
    axis_coords: Mapping[str, tuple[str, ...]] | None = None,
    axis_types: Mapping[str, str] | None = None,
    shaped_param_axes: Mapping[str, tuple[str, ...]] | None = None,
    axis_alias: Mapping[str, str] | None = None,
    history_signal_id_map: Mapping[tuple[str, str, tuple[tuple[str, str], ...]], int] | None = None,
) -> ast.expr:
    """Lower an IR expression to a vector-shape AST.

    Walks ``expr`` and produces an :class:`ast.expr` whose value is shaped
    to broadcast against a tensor with axes ``target_axes``.

    Args:
        expr: Typed IR expression to lower. Must satisfy the v1 subset
            described in the module docstring.
        target_axes: Axis order of the cell layout the result must
            broadcast against. Use ``()`` for a scalar result.
        buffer_axes: Mapping from templated-buffer name (state or alias)
            to the buffer's declared axis order. Names not in this
            mapping are treated as scalar (``ast.Name``) leaves.
        axis_names: Registered axis identifier set; passed through to
            :func:`lower_subscript_to_buffer` for index classification.
        reducible_axes: Axes that may be summed over inside a
            :class:`Reduce` node. Typically the set of categorical/ordinal
            axes (uniform-weight kernel). Empty by default — when empty,
            any :class:`Reduce` raises :class:`UnsupportedIRLoweringError`.
        axis_weights: Optional mapping from axis name to per-coordinate
            integration weights (e.g. trapezoidal deltas) for continuous
            axes. Required for ``Reduce(kind="integrate_over", ...)`` and
            for any bound axis not in ``reducible_axes``. When supplied,
            the reduction emits ``np.sum(weights * body, axis=...)``
            instead of a plain ``np.sum``.
        axis_coords: Optional mapping from axis name to its declared
            coord labels (as strings). Required when a :class:`Reduce`
            carries non-empty ``filters`` so coord labels can be
            resolved to integer indices for ``np.take`` slicing.
        axis_types: Optional mapping from axis name to its declared type
            (``"categorical"``, ``"ordinal"``, or ``"continuous"``).
            Controls how :class:`Reduce` filter coord lists are
            interpreted: categorical → exact-label subset, ordinal →
            inclusive ``[lo_label, hi_label]`` index range, continuous →
            closed ``[lo, hi]`` numeric sub-interval (with trapezoidal
            weight recomputation for the integrate kernel).
        shaped_param_axes: Optional mapping from shaped-parameter name
            to the parameter's declared axis order. When provided,
            :class:`Subscript` references whose name is not in
            ``buffer_axes`` but is in ``shaped_param_axes`` are lowered
            via :func:`_lower_shaped_param_subscript` (positional
            broadcast) rather than rejected. This enables IR lowering
            for shaped-parameter references such as a contact kernel
            ``K[age, vax]`` — and, in combination with synthetic
            bound-axis labels emitted by :class:`Reduce`, the
            same-axis-twice case ``K[age, age:ap]``.
        axis_alias: Optional mapping from synthetic axis labels emitted
            by :func:`_lower_reduce` (e.g. ``age#ap``) back to their
            real axis name. Threaded into :func:`lower_subscript_to_buffer`
            and :func:`_lower_shaped_param_subscript` so labeled
            indices still match the real buffer axes while preserving
            the synthetic label for ``target_axes`` broadcast alignment.

    Returns:
        An ``ast.expr`` suitable for ``compile(ast.Expression(body=...))``
        once enclosed and ``ast.fix_missing_locations``-ed by the caller.

    Raises:
        UnsupportedIRLoweringError: If ``expr`` (or any subexpression)
            contains a node outside the v1 subset.
    """
    if isinstance(expr, Literal):
        return ast.Constant(value=expr.value)

    if isinstance(expr, Sym):
        return _name(expr.name)

    if isinstance(expr, Subscript):
        src_axes = buffer_axes.get(expr.name)
        if src_axes is not None:
            return lower_subscript_to_buffer(
                expr,
                src_axes=src_axes,
                target_axes=target_axes,
                axis_names=axis_names,
                axis_alias=axis_alias,
            )
        shaped_axes = (shaped_param_axes or {}).get(expr.name)
        if shaped_axes is not None:
            return _lower_shaped_param_subscript(
                expr,
                src_axes=shaped_axes,
                target_axes=target_axes,
                axis_names=axis_names,
                axis_alias=axis_alias,
            )
        msg = (
            f"subscript {expr.name!r} is not a registered templated "
            "buffer or shaped parameter — v1 lowering cannot resolve it"
        )
        raise UnsupportedIRLoweringError(msg)

    if isinstance(expr, Apply):
        return _lower_apply(
            expr,
            target_axes=target_axes,
            buffer_axes=buffer_axes,
            axis_names=axis_names,
            reducible_axes=reducible_axes,
            axis_weights=axis_weights,
            axis_coords=axis_coords,
            axis_types=axis_types,
            shaped_param_axes=shaped_param_axes,
            axis_alias=axis_alias,
            history_signal_id_map=history_signal_id_map,
        )

    if isinstance(expr, HistoryOp):
        return _lower_history_op(
            expr,
            target_axes=target_axes,
            buffer_axes=buffer_axes,
            axis_names=axis_names,
            reducible_axes=reducible_axes,
            axis_weights=axis_weights,
            axis_coords=axis_coords,
            axis_types=axis_types,
            shaped_param_axes=shaped_param_axes,
            axis_alias=axis_alias,
            history_signal_id_map=history_signal_id_map,
        )

    if not isinstance(expr, Reduce):
        msg = f"unsupported IR node in lowering: {type(expr).__name__}"
        raise UnsupportedIRLoweringError(msg)

    return _lower_reduce(
        expr,
        target_axes=target_axes,
        buffer_axes=buffer_axes,
        axis_names=axis_names,
        reducible_axes=reducible_axes,
        axis_weights=axis_weights,
        axis_coords=axis_coords,
        axis_types=axis_types,
        shaped_param_axes=shaped_param_axes,
        axis_alias=axis_alias,
        history_signal_id_map=history_signal_id_map,
    )


_REDUCE_SUM_KINDS: frozenset[str] = frozenset({
    "apply_along",
    "sum_over",
    "integrate_over",
})
_REDUCE_UNIFORM_KINDS: frozenset[str] = frozenset({"apply_along", "sum_over"})


def _resolve_ordinal_range_filter(
    axis: str, declared: tuple[str, ...], filt: tuple[str, ...]
) -> tuple[int, ...]:
    """Resolve an ordinal-axis ``[lo_label, hi_label]`` filter to indices.

    Returns:
        Tuple of declared-coord indices spanning the inclusive index
        range ``declared.index(lo)..declared.index(hi)``.

    Raises:
        UnsupportedIRLoweringError: If ``filt`` is not a 2-element list,
            either endpoint is not a declared coord label, or
            ``index(lo) > index(hi)``.
    """
    if len(filt) != 2:
        msg = (
            f"Reduce filter for ordinal axis {axis!r} must be a 2-element "
            f"[lo_label, hi_label] range (got {len(filt)} entries)"
        )
        raise UnsupportedIRLoweringError(msg)
    lo_label, hi_label = filt
    unknown = [c for c in (lo_label, hi_label) if c not in declared]
    if unknown:
        msg = f"Reduce filter for ordinal axis {axis!r} has unknown coords: {unknown}"
        raise UnsupportedIRLoweringError(msg)
    lo_idx = declared.index(lo_label)
    hi_idx = declared.index(hi_label)
    if lo_idx > hi_idx:
        msg = (
            f"Reduce filter for ordinal axis {axis!r} endpoints must satisfy "
            f"index(lo) <= index(hi) (got [{lo_label!r}, {hi_label!r}] at "
            f"indices [{lo_idx}, {hi_idx}])"
        )
        raise UnsupportedIRLoweringError(msg)
    return tuple(range(lo_idx, hi_idx + 1))


def _resolve_continuous_range_filter(  # noqa: C901
    axis: str,
    declared: tuple[str, ...],
    filt: tuple[str, ...],
    *,
    recompute_trapezoidal: bool,
) -> tuple[tuple[int, ...], tuple[float, ...] | None]:
    """Resolve a continuous-axis ``[lo, hi]`` filter to indices (+ weights).

    Selects declared coord positions ``i`` for which ``lo <= float(c) <= hi``.
    When ``recompute_trapezoidal`` is true (axis is integrated over), the
    selected coords are parsed as floats and trapezoidal weights are
    computed for the sub-interval; otherwise the second element is
    ``None``.

    Returns:
        ``(indices, sub_interval_weights_or_None)``.

    Raises:
        UnsupportedIRLoweringError: If ``filt`` is not a 2-element list,
            the endpoints aren't numeric, ``lo > hi``, the sub-interval
            selects no axis coords, or trapezoidal weight recomputation
            requires fewer than two coords / non-strictly-increasing
            coords.
    """
    if len(filt) != 2:
        msg = (
            f"Reduce filter for continuous axis {axis!r} must be a 2-element "
            f"[lo, hi] interval (got {len(filt)} entries)"
        )
        raise UnsupportedIRLoweringError(msg)
    try:
        lo = float(filt[0])
        hi = float(filt[1])
    except ValueError as exc:
        msg = (
            f"Reduce filter for continuous axis {axis!r} endpoints must be "
            f"numeric (got {list(filt)!r})"
        )
        raise UnsupportedIRLoweringError(msg) from exc
    if lo > hi:
        msg = (
            f"Reduce filter for continuous axis {axis!r} endpoints must "
            f"satisfy lo <= hi (got [{lo}, {hi}])"
        )
        raise UnsupportedIRLoweringError(msg)
    try:
        coord_floats = [float(c) for c in declared]
    except ValueError as exc:
        msg = (
            f"Reduce filter for continuous axis {axis!r} requires numeric "
            f"declared coords (got {list(declared)!r})"
        )
        raise UnsupportedIRLoweringError(msg) from exc
    indices = tuple(i for i, c in enumerate(coord_floats) if lo <= c <= hi)
    if not indices:
        msg = (
            f"Reduce filter for continuous axis {axis!r} interval [{lo}, "
            f"{hi}] selects no axis coords"
        )
        raise UnsupportedIRLoweringError(msg)
    if not recompute_trapezoidal:
        return indices, None
    sub_floats = [coord_floats[i] for i in indices]
    if len(sub_floats) < 2:
        msg = (
            f"Reduce filter for continuous axis {axis!r} sub-interval needs "
            "at least 2 coords for trapezoidal integration"
        )
        raise UnsupportedIRLoweringError(msg)
    sub_weights: list[float] = []
    for i in range(len(sub_floats)):
        if i == 0:
            width = (sub_floats[1] - sub_floats[0]) / 2.0
        elif i == len(sub_floats) - 1:
            width = (sub_floats[-1] - sub_floats[-2]) / 2.0
        else:
            width = (sub_floats[i + 1] - sub_floats[i - 1]) / 2.0
        if width <= 0.0:
            msg = (
                f"Reduce filter for continuous axis {axis!r} sub-interval "
                "coords must be strictly increasing for trapezoidal "
                "integration"
            )
            raise UnsupportedIRLoweringError(msg)
        sub_weights.append(width)
    return indices, tuple(sub_weights)


def _lower_reduce(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915
    expr: Reduce,
    *,
    target_axes: tuple[str, ...],
    buffer_axes: Mapping[str, tuple[str, ...]],
    axis_names: frozenset[str],
    reducible_axes: frozenset[str],
    axis_weights: Mapping[str, tuple[float, ...]] | None = None,
    axis_coords: Mapping[str, tuple[str, ...]] | None = None,
    axis_types: Mapping[str, str] | None = None,
    shaped_param_axes: Mapping[str, tuple[str, ...]] | None = None,
    axis_alias: Mapping[str, str] | None = None,
    history_signal_id_map: Mapping[tuple[str, str, tuple[tuple[str, str], ...]], int] | None = None,
) -> ast.expr:
    """Lower a :class:`Reduce` node to ``np.sum(body, axis=...)``.

    Supports uniform-weight kernels (``apply_along`` / ``sum_over``) over
    axes in ``reducible_axes`` and shaped-weight kernels
    (``integrate_over``, or any binding whose axis appears in
    ``axis_weights``). For weighted bindings, emits
    ``np.sum(weights * body, axis=...)`` with per-axis weight constants
    baked into the AST as ``np.array([...])`` factors broadcast into
    position.

    Same-axis-twice (e.g. a shaped contact kernel ``K[age, age:ap]``
    inside ``apply_along(..., age=ap)`` whose target also carries
    ``age``) is supported by using the binding variable itself (``ap``)
    as a synthetic axis label: the label is appended to
    ``extended_target`` as a distinct broadcast position, while
    ``axis_weights`` / ``axis_coords`` / ``axis_types`` lookups continue
    to resolve via the original axis name.  Crucially, the binding
    variable is also added to ``body_axis_names``, so bare ``I[ap, ...]``
    subscripts — where the spec uses the variable as an axis label
    rather than the ``coord=`` form — classify as FREE without rewriting.

    Returns:
        An ``ast.expr`` invoking ``np.sum`` on the (possibly
        weight-scaled) lowered body, with the bound axes collapsed.

    Raises:
        UnsupportedIRLoweringError: If ``expr.kind`` is not a recognized
            sum kind, if a binding requires weights but none were
            provided, if a uniform binding targets a non-reducible axis,
            or if the body itself violates the v1 lowering subset.
    """
    if expr.kind not in _REDUCE_SUM_KINDS:
        msg = (
            f"Reduce kind {expr.kind!r} is not a recognized sum kind; v1 "
            f"lowering supports {sorted(_REDUCE_SUM_KINDS)}"
        )
        raise UnsupportedIRLoweringError(msg)

    if expr.kernel is not None and expr.kernel not in {"sum", "integrate"}:
        msg = (
            f"Reduce kernel {expr.kernel!r} is not recognized; v1 lowering "
            "supports kernel='sum' or kernel='integrate'"
        )
        raise UnsupportedIRLoweringError(msg)

    weights_map: Mapping[str, tuple[float, ...]] = axis_weights or {}
    coords_map: Mapping[str, tuple[str, ...]] = axis_coords or {}
    types_map: Mapping[str, str] = axis_types or {}
    filter_map: dict[str, tuple[str, ...]] = dict(expr.filters)
    force_integrate = expr.kind == "integrate_over" or expr.kernel == "integrate"
    force_sum = expr.kernel == "sum"

    # Decide per-binding whether the bound position needs a synthetic
    # axis label. A synthetic label is required when any body
    # :class:`Subscript` references the bound axis with both a FREE
    # index and a coord=binding index (e.g. ``K[age, age:ap]`` where
    # ``age`` is the outer-free position and ``age:ap`` is the bound
    # inner position on a duplicate-axis shaped parameter). When
    # synthesized, the binding variable itself (e.g. ``ap``) is used as
    # the label — it is already unique within the scope and is added to
    # ``body_axis_names`` so that bare ``I[ap, ...]`` subscripts (where
    # the spec uses the binding variable as an axis label rather than
    # the ``coord=`` syntax) also classify as FREE without any rewriting.
    bound_axes_seen: set[str] = set()
    binding_label: dict[str, str] = {}  # var -> label (synthetic or real)
    axis_relabel: dict[str, str] = {}  # real axis -> synthetic label
    label_real_axis: dict[str, str] = {}  # label -> real axis (incl. real->real)
    weighted_labels: list[str] = []
    bound_labels: list[str] = []
    for axis, binding in expr.bindings:
        if axis in bound_axes_seen:
            msg = f"Reduce has duplicate axis binding {axis!r}"
            raise UnsupportedIRLoweringError(msg)
        bound_axes_seen.add(axis)
        needs_synthetic = _binding_collides_with_free_index(
            expr.body, axis=axis, binding_var=binding
        )
        label = binding if needs_synthetic else axis
        if needs_synthetic:
            axis_relabel[axis] = label
        binding_label[binding] = label
        label_real_axis[label] = axis
        if force_integrate or (not force_sum and axis not in reducible_axes):
            if axis not in weights_map:
                msg = (
                    f"Reduce binding {axis}={binding} (kind {expr.kind!r}, "
                    f"kernel {expr.kernel!r}) requires integration weights "
                    f"for axis {axis!r}, but none were provided"
                )
                raise UnsupportedIRLoweringError(msg)
            weighted_labels.append(label)
        bound_labels.append(label)

    rebound_body = _rebind_subscript_indices(
        expr.body,
        rebind={binding: axis for axis, binding in expr.bindings},
        axis_relabel=axis_relabel,
    )

    extended_target = target_axes + tuple(
        lbl for lbl in bound_labels if lbl not in target_axes
    )
    body_axis_names = axis_names | frozenset(axis_relabel.values())
    # Merge synthetic-label aliases with any inherited from an outer
    # Reduce so deeply-nested same-axis-twice cases still resolve their
    # buffer matches via real axis names.
    child_axis_alias: dict[str, str] = dict(axis_alias or {})
    child_axis_alias.update({
        label: real_axis for real_axis, label in axis_relabel.items()
    })
    body_ast = lower_to_vector_ast(
        rebound_body,
        target_axes=extended_target,
        buffer_axes=buffer_axes,
        axis_names=body_axis_names,
        reducible_axes=reducible_axes,
        axis_weights=axis_weights,
        axis_coords=axis_coords,
        axis_types=axis_types,
        shaped_param_axes=shaped_param_axes,
        axis_alias=child_axis_alias,
        history_signal_id_map=history_signal_id_map,
    )

    # Resolve per-axis filter coord lists to integer indices. The
    # interpretation depends on the axis type carried in ``axis_types``
    # (mirroring :func:`_normalize._build_apply_along_axis_options`):
    #
    # * ``categorical`` (or unknown type): exact-label subset \u2014 every
    #   filter coord must appear in the axis's declared coord list.
    # * ``ordinal``: filter must be exactly ``[lo_label, hi_label]``;
    #   resolves to the inclusive index range ``index(lo)..index(hi)``.
    # * ``continuous``: filter must be exactly ``[lo, hi]`` numeric
    #   endpoints; resolves to declared coords ``c`` with
    #   ``lo <= float(c) <= hi``. When the axis is also weighted, the
    #   integration weights for the sub-interval are recomputed via the
    #   trapezoidal rule on the selected coords (not sub-selected from
    #   the original axis weights).
    filter_indices: dict[str, tuple[int, ...]] = {}
    filter_subinterval_weights: dict[str, tuple[float, ...]] = {}
    real_bound_axes = set(label_real_axis.values())
    real_weighted_axes = {label_real_axis[lbl] for lbl in weighted_labels}
    real_to_label = {label_real_axis[lbl]: lbl for lbl in bound_labels}
    for ax, coords in filter_map.items():
        if ax not in real_bound_axes:
            msg = f"Reduce filter for axis {ax!r} has no matching binding"
            raise UnsupportedIRLoweringError(msg)
        if ax not in coords_map:
            msg = (
                f"Reduce filter for axis {ax!r} requires axis_coords, but "
                "none were provided"
            )
            raise UnsupportedIRLoweringError(msg)
        declared = coords_map[ax]
        ax_type = types_map.get(ax, "categorical")
        if ax_type == "ordinal":
            indices = _resolve_ordinal_range_filter(ax, declared, coords)
        elif ax_type == "continuous":
            indices, sub_weights = _resolve_continuous_range_filter(
                ax,
                declared,
                coords,
                recompute_trapezoidal=ax in real_weighted_axes,
            )
            if sub_weights is not None:
                filter_subinterval_weights[ax] = sub_weights
        else:
            try:
                indices = tuple(declared.index(c) for c in coords)
            except ValueError:
                msg = (
                    f"Reduce filter {ax}={list(coords)} contains coord(s) "
                    f"not declared on axis {ax!r}; categorical filters "
                    "require an exact subset of the declared coords"
                )
                raise UnsupportedIRLoweringError(msg) from None
        filter_indices[ax] = indices

    for ax, indices in filter_indices.items():
        pos = extended_target.index(real_to_label[ax])
        body_ast = ast.Call(
            func=ast.Attribute(value=_name("np"), attr="take", ctx=ast.Load()),
            args=[
                body_ast,
                ast.List(
                    elts=[ast.Constant(value=i) for i in indices],
                    ctx=ast.Load(),
                ),
            ],
            keywords=[ast.keyword(arg="axis", value=ast.Constant(value=pos))],
        )

    for lbl in weighted_labels:
        ax = label_real_axis[lbl]
        weights = weights_map[ax]
        if ax in filter_subinterval_weights:
            # Continuous-axis sub-interval integration uses freshly
            # recomputed trapezoidal weights, not the original axis
            # weights indexed by position.
            weights = filter_subinterval_weights[ax]
        elif ax in filter_indices:
            # Categorical / ordinal filter: sub-select the original
            # uniform-or-weighted vector to match the slice.
            weights = tuple(weights[i] for i in filter_indices[ax])
        body_ast = ast.BinOp(
            left=_weight_constant_for_axis(
                lbl,
                weights=weights,
                extended_target=extended_target,
            ),
            op=ast.Mult(),
            right=body_ast,
        )

    reduce_positions = tuple(extended_target.index(lbl) for lbl in bound_labels)
    if len(reduce_positions) == 1:
        axis_arg: ast.expr = ast.Constant(value=reduce_positions[0])
    else:
        axis_arg = ast.Tuple(
            elts=[ast.Constant(value=p) for p in reduce_positions],
            ctx=ast.Load(),
        )
    # Fast path: when none of the bound axes also appear in
    # ``target_axes``, every reduced dim is a trailing "extra" position
    # that ``np.sum`` (``keepdims=False``) collapses for free, leaving a
    # result whose rank already matches ``target_axes``. This is the
    # common case (every ordinary ``apply_along``/``sum_over``) and
    # avoids two extra reshape ops per reduction in the lowered XLA.
    #
    # Slow path (``keepdims=True`` + ``squeeze``) is reserved for the
    # pinned-token mask synthesis emitted by ``_normalize``, where a
    # reduced axis also appears in ``target_axes`` (e.g.
    # ``sum_over(... * mask_from[vax], vax=vax) * mask_to[vax]``). In
    # that case ``keepdims=True`` preserves the in-target reduced axis
    # as a size-1 broadcast slot, and the trailing purely-bound
    # "extras" are squeezed off so the result rank matches
    # ``target_axes``.
    n_extra = len(extended_target) - len(target_axes)
    target_axis_collapsed = n_extra < len(bound_labels)
    if not target_axis_collapsed:
        return ast.Call(
            func=ast.Attribute(value=_name("np"), attr="sum", ctx=ast.Load()),
            args=[body_ast],
            keywords=[ast.keyword(arg="axis", value=axis_arg)],
        )
    sum_call = ast.Call(
        func=ast.Attribute(value=_name("np"), attr="sum", ctx=ast.Load()),
        args=[body_ast],
        keywords=[
            ast.keyword(arg="axis", value=axis_arg),
            ast.keyword(arg="keepdims", value=ast.Constant(value=True)),
        ],
    )
    if n_extra == 0:
        return sum_call
    squeeze_positions = tuple(range(len(target_axes), len(extended_target)))
    if len(squeeze_positions) == 1:
        squeeze_axis_arg: ast.expr = ast.Constant(value=squeeze_positions[0])
    else:
        squeeze_axis_arg = ast.Tuple(
            elts=[ast.Constant(value=p) for p in squeeze_positions],
            ctx=ast.Load(),
        )
    return ast.Call(
        func=ast.Attribute(value=_name("np"), attr="squeeze", ctx=ast.Load()),
        args=[sum_call],
        keywords=[ast.keyword(arg="axis", value=squeeze_axis_arg)],
    )


def _weight_constant_for_axis(
    axis: str,
    *,
    weights: tuple[float, ...],
    extended_target: tuple[str, ...],
) -> ast.expr:
    """Build ``np.array([w0, w1, ...])[None, ..., :, ..., None]`` for ``axis``.

    Returns a constant 1-D weights vector reshaped to broadcast along the
    position of ``axis`` within ``extended_target``.

    Returns:
        An AST expression that evaluates to the broadcast-shaped weights
        constant.
    """
    pos = extended_target.index(axis)
    arr: ast.expr = ast.Call(
        func=ast.Attribute(value=_name("np"), attr="array", ctx=ast.Load()),
        args=[
            ast.List(
                elts=[ast.Constant(value=float(w)) for w in weights],
                ctx=ast.Load(),
            )
        ],
        keywords=[],
    )
    n = len(extended_target)
    if n == 1:
        return arr
    elts: list[ast.expr] = []
    for i in range(n):
        if i == pos:
            elts.append(ast.Slice(lower=None, upper=None, step=None))
        else:
            elts.append(ast.Constant(value=None))
    return ast.Subscript(
        value=arr,
        slice=ast.Tuple(elts=elts, ctx=ast.Load()),
        ctx=ast.Load(),
    )


def _binding_collides_with_free_index(
    expr: Expr, *, axis: str, binding_var: str
) -> bool:
    """Return ``True`` if a synthetic axis label is needed for ``binding_var``.

    Two cases require the binding variable to become a synthetic axis label
    in :func:`_lower_reduce`:

    1. **Same-axis-twice**: a ``Subscript`` has both a FREE index and a
       ``coord=binding_var`` index on the same ``axis`` (e.g.
       ``K[age, age:ap]`` inside ``apply_along(..., age=ap)``).  The bound
       position must be renamed so the broadcast pipeline can carry both
       the outer-free and inner-bound slots as distinct dimensions.

    2. **Bare binding label**: a ``Subscript`` uses ``binding_var`` itself
       as a bare axis label (e.g. ``I[ap]``) rather than the ``coord=``
       form.  This appears in low-rank factored specs such as
       ``apply_along(H[rank, age:ap] * I[ap], age=ap)`` where there is no
       same-axis-twice collision but the binding variable still needs to be
       a recognized axis name in ``body_axis_names`` for the index to
       classify as FREE.
    """
    if isinstance(expr, Subscript):
        has_free = any(idx.axis == axis and idx.coord is None for idx in expr.indices)
        has_bound = any(
            idx.axis == axis and idx.coord == binding_var for idx in expr.indices
        )
        has_bare_binding = any(
            idx.axis == binding_var and idx.coord is None for idx in expr.indices
        )
        return (has_free and has_bound) or has_bare_binding
    if isinstance(expr, Apply):
        return any(
            _binding_collides_with_free_index(a, axis=axis, binding_var=binding_var)
            for a in expr.args
        )
    if isinstance(expr, Reduce):
        # An inner Reduce shadows its own bindings but not the outer
        # axis/var pair we're searching for unless one of its bindings
        # rebinds the same variable name.
        if any(b == binding_var for _, b in expr.bindings):
            return False
        return _binding_collides_with_free_index(
            expr.body, axis=axis, binding_var=binding_var
        )
    return False


def _rebind_subscript_indices(  # noqa: PLR0911
    expr: Expr,
    *,
    rebind: Mapping[str, str],
    axis_relabel: Mapping[str, str] | None = None,
) -> Expr:
    """Rewrite Subscript ``AxisIndex(axis, coord=binding)`` slots to FREE.

    Walks ``expr`` recursively; for every ``Subscript``, any index whose
    ``coord`` matches a key of ``rebind`` (the per-binding coord name
    chosen by the Reduce) is rewritten to a FREE wildcard on its declared
    axis. Indices not matching are returned unchanged.

    When ``axis_relabel`` is provided, the axis name written into the
    FREE index is taken from ``axis_relabel.get(original_axis,
    original_axis)``. This is used by :func:`_lower_reduce` to give
    same-axis-twice bound positions a distinct synthetic axis label
    (the binding variable itself, e.g. ``ap``) so the broadcast-by-label
    pipeline can disambiguate the outer-free and inner-bound positions of
    a duplicate-axis shaped param such as ``K[age, age:ap]``.

    Returns:
        A new IR expression with matching Subscript indices rewritten to
        FREE, or ``expr`` itself when no rewrites apply.
    """
    relabel = axis_relabel or {}
    if isinstance(expr, Subscript):
        new_indices: list[AxisIndex] = []
        changed = False
        for idx in expr.indices:
            if idx.coord and idx.coord in rebind and rebind[idx.coord] == idx.axis:
                new_axis = relabel.get(idx.axis, idx.axis)
                new_indices.append(AxisIndex(axis=new_axis, kind=AxisKind.FREE))
                changed = True
            else:
                new_indices.append(idx)
        if changed:
            return Subscript(name=expr.name, indices=tuple(new_indices))
        return expr
    if isinstance(expr, Apply):
        new_args = tuple(
            _rebind_subscript_indices(a, rebind=rebind, axis_relabel=relabel)
            for a in expr.args
        )
        if any(na is not oa for na, oa in zip(new_args, expr.args, strict=True)):
            return Apply(op=expr.op, args=new_args)
        return expr
    if isinstance(expr, Reduce):
        # Inner Reduce shadows its own bindings: drop shadowed keys from
        # the outer rebind map before recursing into the inner body.
        inner_bindings = {b for _, b in expr.bindings}
        inner_rebind = {k: v for k, v in rebind.items() if k not in inner_bindings}
        new_body = _rebind_subscript_indices(
            expr.body, rebind=inner_rebind, axis_relabel=relabel
        )
        if new_body is not expr.body:
            return Reduce(
                kind=expr.kind,
                bindings=expr.bindings,
                body=new_body,
                filters=expr.filters,
                kernel=expr.kernel,
            )
        return expr
    return expr


def _lower_apply(  # noqa: PLR0913
    expr: Apply,
    *,
    target_axes: tuple[str, ...],
    buffer_axes: Mapping[str, tuple[str, ...]],
    axis_names: frozenset[str],
    reducible_axes: frozenset[str] = frozenset(),
    axis_weights: Mapping[str, tuple[float, ...]] | None = None,
    axis_coords: Mapping[str, tuple[str, ...]] | None = None,
    axis_types: Mapping[str, str] | None = None,
    shaped_param_axes: Mapping[str, tuple[str, ...]] | None = None,
    axis_alias: Mapping[str, str] | None = None,
    history_signal_id_map: Mapping[tuple[str, str, tuple[tuple[str, str], ...]], int] | None = None,
) -> ast.expr:
    def lower(child: Expr) -> ast.expr:
        return lower_to_vector_ast(
            child,
            target_axes=target_axes,
            buffer_axes=buffer_axes,
            axis_names=axis_names,
            reducible_axes=reducible_axes,
            axis_weights=axis_weights,
            axis_coords=axis_coords,
            axis_types=axis_types,
            shaped_param_axes=shaped_param_axes,
            axis_alias=axis_alias,
            history_signal_id_map=history_signal_id_map,
        )

    if expr.op == "neg" and len(expr.args) == 1:
        return ast.UnaryOp(op=ast.USub(), operand=lower(expr.args[0]))
    if expr.op == "pos" and len(expr.args) == 1:
        return ast.UnaryOp(op=ast.UAdd(), operand=lower(expr.args[0]))

    if expr.op in _BINOP and len(expr.args) >= 2:
        # Fold left-associatively; expand_reduce_pointwise may produce N-ary
        # flat Apply('+', ...) to avoid deep recursion in the tree.
        ast_op_cls = _BINOP[expr.op]
        result: ast.expr = lower(expr.args[0])
        for arg in expr.args[1:]:
            result = ast.BinOp(left=result, op=ast_op_cls(), right=lower(arg))
        return result
    if expr.op in _CMPOP and len(expr.args) == 2:
        return ast.Compare(
            left=lower(expr.args[0]),
            ops=[_CMPOP[expr.op]()],
            comparators=[lower(expr.args[1])],
        )
    if expr.op in {"and", "or"}:
        bool_op: ast.boolop = ast.And() if expr.op == "and" else ast.Or()
        return ast.BoolOp(op=bool_op, values=[lower(a) for a in expr.args])
    if expr.op == "ifelse" and len(expr.args) == 3:
        return ast.IfExp(
            test=lower(expr.args[0]),
            body=lower(expr.args[1]),
            orelse=lower(expr.args[2]),
        )

    msg = (
        f"Apply op {expr.op!r} (arity {len(expr.args)}) is outside the "
        "v1 lowering subset — only arithmetic, comparison, boolean, and "
        "ifelse operators are supported"
    )
    raise UnsupportedIRLoweringError(msg)


# ---------------------------------------------------------------------------
# Per-cell → template-form lift
# ---------------------------------------------------------------------------


def lift_cell_ir_to_template(
    expr: Expr,
    *,
    cell_to_template: Mapping[str, tuple[str, tuple[str, ...]]],
) -> Expr:
    """Lift per-cell IR to template-form IR.

    The normalizer expands templated states/aliases/params into per-cell
    scalar names (e.g. ``S__age_y__loc_a``) before parsing into IR, so the
    resulting :class:`Sym` leaves carry expanded names rather than
    :class:`Subscript` nodes against base buffers. This function walks
    ``expr`` and rewrites each ``Sym(name)`` whose ``name`` appears in
    ``cell_to_template`` (with non-empty axes) into a wildcard
    :class:`Subscript` against the buffer's base name, suitable as input to
    :func:`lower_to_vector_ast`.

    Symbols not in ``cell_to_template`` — scalar parameters or unbound
    names — are left unchanged. Mappings whose template has ``axes == ()``
    (e.g. scalar aliases) rewrite the symbol to a bare ``Sym(f"{base}_buf")``
    leaf so the result evaluates against the runtime's ``<base>_buf``
    binding.

    Args:
        expr: Per-cell IR expression (typically from
            ``NormalizedRhs.equations_ir_raw`` or
            ``NormalizedRhs.aliases_ir``).
        cell_to_template: Mapping from expanded per-cell name to a
            ``(base_name, axes)`` pair describing the buffer the cell
            belongs to. Empty-axes entries are treated as scalar aliases
            (rewritten to ``Sym(f"{base}_buf")``); non-empty entries are
            rewritten to FREE-index :class:`Subscript` nodes.

    Returns:
        A new IR expression equivalent to ``expr`` but with all matched
        cell-name :class:`Sym` leaves replaced — by FREE-index
        :class:`Subscript` nodes (templated buffers) or scalar-buffer
        :class:`Sym` leaves (scalar aliases).

    Raises:
        UnsupportedIRLoweringError: If two distinct per-cell symbols
            sharing the same templated ``base`` co-occur in ``expr``
            (signals a string-expanded axis reduction that cannot be
            represented as a single FREE-axis subscript).
    """
    mapping: dict[str, Expr] = {}
    for cell_name, (base, axes) in cell_to_template.items():
        if not axes:
            mapping[cell_name] = Sym(name=f"{base}_buf")
            continue
        mapping[cell_name] = Subscript(
            name=base,
            indices=tuple(AxisIndex(axis=ax, kind=AxisKind.FREE) for ax in axes),
        )
    if not mapping:
        return expr
    # Refuse the lift when multiple distinct per-cell symbols of the same
    # templated buffer co-occur in ``expr``: that signals an axis reduction
    # (e.g. ``apply_along`` or ``sum_over``) that the normalizer expanded
    # to a per-cell sum at the string level. Collapsing those cells back to
    # a single FREE-axis subscript would silently broadcast instead of
    # reduce, producing wrong results.
    seen_bases: dict[str, str] = {}
    for node in walk(expr):
        if not isinstance(node, Sym):
            continue
        entry = cell_to_template.get(node.name)
        if entry is None or not entry[1]:
            continue
        base = entry[0]
        prior = seen_bases.get(base)
        if prior is None:
            seen_bases[base] = node.name
        elif prior != node.name:
            msg = (
                f"cannot lift per-cell IR for buffer {base!r}: multiple "
                f"distinct cells co-occur ({prior!r} and {node.name!r}); "
                "this signals a string-expanded axis reduction"
            )
            raise UnsupportedIRLoweringError(msg)
    return substitute(expr, mapping)
