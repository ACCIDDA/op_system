"""IR-level template expansion (issue #112).

Mirrors the inline template-substitution semantics of
:func:`op_system._templates._apply_template_substitutions` but operates on
typed IR rather than expression strings.

Two rewrite modes are supported per ``Subscript``:

- **Named expansion**: ``S[age]`` with ``assignment={"age": "0_5"}`` becomes
  ``Sym(name="S__age_0_5")`` - a scalar reference to the expanded state.
- **Shaped-parameter indexing**: when the base name is a registered shaped
  parameter and the placeholder axes match the registered axes, the
  ``AxisIndex`` entries are rewritten to literal integer coords resolved
  via ``axis_lookup`` (so ``theta[imm]`` becomes ``theta[5]`` into the
  shaped buffer).

This module is intentionally pure: it does not touch compile / normalize.
It exists to back the eventual replacement of string-based alias and
template expansion in ``_normalize.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._ir import Apply, AxisIndex, Expr, Literal, Reduce, Subscript, Sym
from ._templates import _render_template_name

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _expandable_axes(indices: Sequence[AxisIndex]) -> list[str] | None:
    """Return the list of bare-identifier axis names from ``indices``.

    Returns ``None`` if any index carries a literal coord or placeholder, in
    which case the subscript is not eligible for template expansion.
    """
    out: list[str] = []
    for idx in indices:
        if idx.coord is not None or idx.placeholder is not None:
            return None
        out.append(idx.axis)
    return out


def _rewrite_shaped_indices(
    axes: Sequence[str],
    *,
    assignment: Mapping[str, str],
    registered: tuple[str, ...],
    axis_lookup: Mapping[str, Sequence[str]],
) -> tuple[AxisIndex, ...] | None:
    """Rewrite shaped-parameter placeholders to literal integer coords.

    Returns:
        Tuple of literal-coord ``AxisIndex`` entries, or ``None`` if the
        axes don't match the registered axes order, an axis is missing
        from ``axis_lookup``, or a coord isn't in its axis.
    """
    if tuple(axes) != registered:
        return None
    out: list[AxisIndex] = []
    for ax in registered:
        coords = axis_lookup.get(ax)
        if coords is None:
            return None
        try:
            i = coords.index(assignment[ax])
        except (KeyError, ValueError):
            return None
        out.append(AxisIndex(axis="", coord=str(i)))
    return tuple(out)


def _expand_subscript(
    sub: Subscript,
    *,
    assignment: Mapping[str, str],
    shaped_params: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, Sequence[str]] | None,
) -> Expr:
    axes = _expandable_axes(sub.indices)
    if axes is None or not axes:
        return sub
    if any(ax not in assignment for ax in axes):
        return sub

    if sub.name in shaped_params and axis_lookup is not None:
        new_indices = _rewrite_shaped_indices(
            axes,
            assignment=assignment,
            registered=shaped_params[sub.name],
            axis_lookup=axis_lookup,
        )
        if new_indices is None:
            return sub
        return Subscript(name=sub.name, indices=new_indices)

    rendered = _render_template_name(sub.name, list(axes), assignment)
    return Sym(name=rendered)


def expand_inline_templates(
    expr: Expr,
    *,
    assignment: Mapping[str, str],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    axis_lookup: Mapping[str, Sequence[str]] | None = None,
) -> Expr:
    """Expand placeholder subscripts in ``expr`` using ``assignment``.

    Args:
        expr: Root IR expression.
        assignment: Mapping from placeholder axis name to concrete coord
            string (e.g. ``{"age": "0_5", "pop": "p1"}``).
        shaped_params: Optional mapping from shaped-parameter base name to
            its registered axis tuple. When provided alongside
            ``axis_lookup``, subscripts whose base names match are rewritten
            to literal integer coords instead of being expanded to scalar
            symbols.
        axis_lookup: Optional mapping from axis name to its coord list,
            used to resolve shaped-parameter indices.

    Returns:
        A new IR expression with templated subscripts expanded. Subscripts
        that are not eligible (literal coords, placeholders, unknown axis
        names) are left untouched.

    Raises:
        TypeError: If ``expr`` contains an unsupported IR node type.
    """
    shaped = shaped_params or {}

    if isinstance(expr, (Literal, Sym)):
        return expr
    if isinstance(expr, Subscript):
        return _expand_subscript(
            expr,
            assignment=assignment,
            shaped_params=shaped,
            axis_lookup=axis_lookup,
        )
    if isinstance(expr, Apply):
        new_args = tuple(
            expand_inline_templates(
                arg,
                assignment=assignment,
                shaped_params=shaped,
                axis_lookup=axis_lookup,
            )
            for arg in expr.args
        )
        if new_args == expr.args:
            return expr
        return Apply(op=expr.op, args=new_args)
    if isinstance(expr, Reduce):
        # Reduce bindings shadow placeholder names: a bound name in this
        # scope should not be substituted from the outer assignment.
        bound = {bind for _, bind in expr.bindings}
        inner_assignment = (
            assignment
            if not bound
            else {k: v for k, v in assignment.items() if k not in bound}
        )
        new_body = expand_inline_templates(
            expr.body,
            assignment=inner_assignment,
            shaped_params=shaped,
            axis_lookup=axis_lookup,
        )
        if new_body is expr.body:
            return expr
        return Reduce(kind=expr.kind, bindings=expr.bindings, body=new_body)
    msg = f"unsupported IR node in expand_inline_templates: {type(expr).__name__}"
    raise TypeError(msg)


__all__ = ["expand_inline_templates"]
