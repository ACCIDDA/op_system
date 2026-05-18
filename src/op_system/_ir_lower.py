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
    Literal,
    Reduce,
    Subscript,
    Sym,
    classify_axis_index,
    substitute,
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

    Returns:
        An ``ast.expr`` accessing ``<sub.name>_buf`` with the necessary
        transpose / size-1 insertions to align with ``target_axes``.

    Raises:
        UnsupportedIRLoweringError: If any index is non-FREE, the index axis
            set doesn't match ``src_axes``, or ``src_axes`` is not a
            subset of ``target_axes``.
    """
    if len(sub.indices) != len(src_axes):
        msg = (
            f"subscript {sub.name!r} has {len(sub.indices)} indices but "
            f"buffer has {len(src_axes)} axes"
        )
        raise UnsupportedIRLoweringError(msg)

    sub_axes: list[str] = []
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

    if set(sub_axes) != set(src_axes):
        msg = (
            f"subscript {sub.name!r} axes {tuple(sub_axes)} do not match "
            f"buffer axes {tuple(src_axes)} as a set"
        )
        raise UnsupportedIRLoweringError(msg)

    if not set(src_axes).issubset(target_axes):
        msg = (
            f"buffer axes {tuple(src_axes)} not a subset of target axes "
            f"{tuple(target_axes)}"
        )
        raise UnsupportedIRLoweringError(msg)

    buf: ast.expr = _name(f"{sub.name}_buf")

    # Reorder buffer (stored in src_axes order) to sub_axes order if the
    # user wrote them out of declaration order (e.g. ``S[vax, age]``
    # against ``src_axes=(age, vax)``).
    if tuple(sub_axes) != tuple(src_axes):
        perm = tuple(src_axes.index(a) for a in sub_axes)
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


# ---------------------------------------------------------------------------
# Full expression lowering
# ---------------------------------------------------------------------------


def lower_to_vector_ast(
    expr: Expr,
    *,
    target_axes: tuple[str, ...],
    buffer_axes: Mapping[str, tuple[str, ...]],
    axis_names: frozenset[str],
    reducible_axes: frozenset[str] = frozenset(),
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
        if src_axes is None:
            msg = (
                f"subscript {expr.name!r} is not a registered templated "
                "buffer — v1 lowering cannot resolve it"
            )
            raise UnsupportedIRLoweringError(msg)
        return lower_subscript_to_buffer(
            expr,
            src_axes=src_axes,
            target_axes=target_axes,
            axis_names=axis_names,
        )

    if isinstance(expr, Apply):
        return _lower_apply(
            expr,
            target_axes=target_axes,
            buffer_axes=buffer_axes,
            axis_names=axis_names,
            reducible_axes=reducible_axes,
        )

    return _lower_reduce(
        expr,
        target_axes=target_axes,
        buffer_axes=buffer_axes,
        axis_names=axis_names,
        reducible_axes=reducible_axes,
    )


_REDUCE_SUM_KINDS: frozenset[str] = frozenset({"apply_along", "sum_over"})


def _lower_reduce(
    expr: Reduce,
    *,
    target_axes: tuple[str, ...],
    buffer_axes: Mapping[str, tuple[str, ...]],
    axis_names: frozenset[str],
    reducible_axes: frozenset[str],
) -> ast.expr:
    """Lower a :class:`Reduce` node to ``np.sum(body, axis=...)``.

    Restricted to uniform-weight kernels (``apply_along`` / ``sum_over``)
    over axes that appear in ``reducible_axes`` — i.e. categorical or
    ordinal axes where the reduction collapses to a plain sum. The
    ``integrate_over`` kind and any continuous-axis binding raise
    :class:`UnsupportedIRLoweringError` so callers fall back to the
    legacy string-expansion path.

    Returns:
        An ``ast.expr`` invoking ``np.sum`` on the lowered body, with the
        bound axes collapsed.

    Raises:
        UnsupportedIRLoweringError: If ``expr.kind`` is not a uniform-sum
            kind, if any bound axis is not in ``reducible_axes``, or if
            the body itself violates the v1 lowering subset.
    """
    if expr.kind not in _REDUCE_SUM_KINDS:
        msg = (
            f"Reduce kind {expr.kind!r} requires shaped weights; v1 "
            "lowering supports only uniform-weight reductions "
            f"({sorted(_REDUCE_SUM_KINDS)})"
        )
        raise UnsupportedIRLoweringError(msg)

    bound_axes: list[str] = []
    rebind: dict[str, str] = {}
    for axis, binding in expr.bindings:
        if axis not in reducible_axes:
            msg = (
                f"Reduce binding {axis}={binding} targets a non-reducible "
                "axis (likely continuous) — v1 lowering supports "
                "uniform-weight categorical reductions only"
            )
            raise UnsupportedIRLoweringError(msg)
        if axis in bound_axes:
            msg = f"Reduce has duplicate axis binding {axis!r}"
            raise UnsupportedIRLoweringError(msg)
        bound_axes.append(axis)
        rebind[binding] = axis

    rebound_body = _rebind_subscript_indices(expr.body, rebind=rebind)

    extended_target = target_axes + tuple(
        ax for ax in bound_axes if ax not in target_axes
    )
    body_ast = lower_to_vector_ast(
        rebound_body,
        target_axes=extended_target,
        buffer_axes=buffer_axes,
        axis_names=axis_names,
        reducible_axes=reducible_axes,
    )

    reduce_positions = tuple(extended_target.index(ax) for ax in bound_axes)
    if len(reduce_positions) == 1:
        axis_arg: ast.expr = ast.Constant(value=reduce_positions[0])
    else:
        axis_arg = ast.Tuple(
            elts=[ast.Constant(value=p) for p in reduce_positions],
            ctx=ast.Load(),
        )
    return ast.Call(
        func=ast.Attribute(value=_name("np"), attr="sum", ctx=ast.Load()),
        args=[body_ast],
        keywords=[ast.keyword(arg="axis", value=axis_arg)],
    )


def _rebind_subscript_indices(  # noqa: PLR0911
    expr: Expr, *, rebind: Mapping[str, str]
) -> Expr:
    """Rewrite Subscript ``AxisIndex(axis, coord=binding)`` slots to FREE.

    Walks ``expr`` recursively; for every ``Subscript``, any index whose
    ``coord`` matches a key of ``rebind`` (the per-binding coord name
    chosen by the Reduce) is rewritten to a FREE wildcard on its declared
    axis. Indices not matching are returned unchanged.

    Returns:
        A new IR expression with matching Subscript indices rewritten to
        FREE, or ``expr`` itself when no rewrites apply.
    """
    if isinstance(expr, Subscript):
        new_indices: list[AxisIndex] = []
        changed = False
        for idx in expr.indices:
            if idx.coord and idx.coord in rebind and rebind[idx.coord] == idx.axis:
                new_indices.append(AxisIndex(axis=idx.axis, kind=AxisKind.FREE))
                changed = True
            else:
                new_indices.append(idx)
        if changed:
            return Subscript(name=expr.name, indices=tuple(new_indices))
        return expr
    if isinstance(expr, Apply):
        new_args = tuple(_rebind_subscript_indices(a, rebind=rebind) for a in expr.args)
        if any(na is not oa for na, oa in zip(new_args, expr.args, strict=True)):
            return Apply(op=expr.op, args=new_args)
        return expr
    if isinstance(expr, Reduce):
        # Inner Reduce shadows its own bindings: drop shadowed keys from
        # the outer rebind map before recursing into the inner body.
        inner_bindings = {b for _, b in expr.bindings}
        inner_rebind = {k: v for k, v in rebind.items() if k not in inner_bindings}
        new_body = _rebind_subscript_indices(expr.body, rebind=inner_rebind)
        if new_body is not expr.body:
            return Reduce(kind=expr.kind, bindings=expr.bindings, body=new_body)
        return expr
    return expr


def _lower_apply(
    expr: Apply,
    *,
    target_axes: tuple[str, ...],
    buffer_axes: Mapping[str, tuple[str, ...]],
    axis_names: frozenset[str],
    reducible_axes: frozenset[str] = frozenset(),
) -> ast.expr:
    def lower(child: Expr) -> ast.expr:
        return lower_to_vector_ast(
            child,
            target_axes=target_axes,
            buffer_axes=buffer_axes,
            axis_names=axis_names,
            reducible_axes=reducible_axes,
        )

    if expr.op == "neg" and len(expr.args) == 1:
        return ast.UnaryOp(op=ast.USub(), operand=lower(expr.args[0]))
    if expr.op == "pos" and len(expr.args) == 1:
        return ast.UnaryOp(op=ast.UAdd(), operand=lower(expr.args[0]))

    if expr.op in _BINOP and len(expr.args) == 2:
        return ast.BinOp(
            left=lower(expr.args[0]),
            op=_BINOP[expr.op](),
            right=lower(expr.args[1]),
        )
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
