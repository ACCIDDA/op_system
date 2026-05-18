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

from itertools import product
from typing import TYPE_CHECKING

from ._ir import (
    Apply,
    AxisIndex,
    Expr,
    Literal,
    Reduce,
    Subscript,
    Sym,
    free_symbols,
    substitute,
)
from ._templates import _render_template_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


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


__all__ = [
    "expand_inline_templates",
    "expand_over_axes",
    "inline_aliases",
    "placeholder_axes",
]

_CYCLE_WHITE = 0
_CYCLE_GREY = 1
_CYCLE_BLACK = 2


def _collect_subscript_axes(
    sub: Subscript,
    *,
    shaped: Mapping[str, tuple[str, ...]],
    out: dict[str, None],
) -> None:
    if sub.name in shaped:
        return
    for idx in sub.indices:
        if idx.coord is not None or idx.placeholder is not None:
            continue
        if idx.axis:
            out.setdefault(idx.axis, None)


def _walk_for_axes(
    node: Expr,
    *,
    shaped: Mapping[str, tuple[str, ...]],
    seen: dict[str, None],
) -> None:
    if isinstance(node, (Literal, Sym)):
        return
    if isinstance(node, Subscript):
        _collect_subscript_axes(node, shaped=shaped, out=seen)
        return
    if isinstance(node, Apply):
        for arg in node.args:
            _walk_for_axes(arg, shaped=shaped, seen=seen)
        return
    if isinstance(node, Reduce):
        _walk_for_axes(node.body, shaped=shaped, seen=seen)
        for _, bind in node.bindings:
            seen.pop(bind, None)


def placeholder_axes(
    expr: Expr,
    *,
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Collect placeholder axis names referenced by ``expr``.

    Walks the IR and gathers every bare-identifier axis appearing inside a
    ``Subscript`` whose base name is *not* a registered shaped parameter.
    Literal-coord and ``$``-placeholder positions are skipped. Order is
    deterministic: first appearance in pre-order traversal wins.

    Args:
        expr: Root IR expression.
        shaped_params: Optional mapping of shaped-parameter base names
            (whose subscripts are rewritten to literal indices rather than
            expanded into scalar symbols).

    Returns:
        Tuple of distinct axis names in first-seen order.
    """
    shaped = shaped_params or {}
    seen: dict[str, None] = {}
    _walk_for_axes(expr, shaped=shaped, seen=seen)
    return tuple(seen)


def expand_over_axes(
    expr: Expr,
    *,
    axes: Iterable[str],
    axis_lookup: Mapping[str, Sequence[str]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
) -> list[tuple[dict[str, str], Expr]]:
    """Cross-expand ``expr`` over the coordinate product of ``axes``.

    For every combination of coords drawn from ``axis_lookup[ax]`` for each
    ``ax in axes``, builds an assignment and applies
    :func:`expand_inline_templates` to ``expr``.

    Args:
        expr: Root IR expression to expand.
        axes: Axis names to cross-expand over. Typical callers pass
            ``placeholder_axes(expr, shaped_params=...)``.
        axis_lookup: Mapping of axis name to its ordered coord list. Must
            contain every entry in ``axes``.
        shaped_params: Optional shaped-parameter registry forwarded to
            :func:`expand_inline_templates`.

    Returns:
        List of ``(assignment, expanded_expr)`` pairs, one per coordinate
        combination. When ``axes`` is empty, returns ``[({}, expr)]``.
    """
    axis_list = list(axes)
    if not axis_list:
        return [({}, expr)]

    coord_lists = [list(axis_lookup[ax]) for ax in axis_list]
    out: list[tuple[dict[str, str], Expr]] = []
    for combo in product(*coord_lists):
        assignment = dict(zip(axis_list, combo, strict=True))
        out.append((
            assignment,
            expand_inline_templates(
                expr,
                assignment=assignment,
                shaped_params=shaped_params,
                axis_lookup=axis_lookup,
            ),
        ))
    return out


def _detect_alias_cycle(aliases: Mapping[str, Expr]) -> list[str] | None:
    """Return a cycle of alias names if one exists, else ``None``.

    Builds a name -> referenced-alias-names graph (only edges into the
    alias namespace itself count) and DFS-checks for a back edge.
    """
    keys = set(aliases)
    graph: dict[str, frozenset[str]] = {
        name: frozenset(free_symbols(body) & keys) for name, body in aliases.items()
    }

    color: dict[str, int] = dict.fromkeys(graph, _CYCLE_WHITE)
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        color[node] = _CYCLE_GREY
        stack.append(node)
        for nxt in graph[node]:
            if color[nxt] == _CYCLE_GREY:
                i = stack.index(nxt)
                return [*stack[i:], nxt]
            if color[nxt] == _CYCLE_WHITE:
                found = visit(nxt)
                if found is not None:
                    return found
        color[node] = _CYCLE_BLACK
        stack.pop()
        return None

    for n in graph:
        if color[n] == _CYCLE_WHITE:
            cyc = visit(n)
            if cyc is not None:
                return cyc
    return None


def inline_aliases(
    expr: Expr,
    aliases: Mapping[str, Expr],
    *,
    max_depth: int = 64,
) -> Expr:
    """Fixed-point inline ``Sym`` references that match ``aliases`` keys.

    Repeatedly applies :func:`op_system._ir.substitute` until no free
    symbols in the result are in ``aliases``, or ``max_depth`` is reached.

    Aliases may reference one another transitively; this function expands
    chains in one pass per iteration. Self-referential or mutually
    recursive aliases are rejected up front via a cycle check.

    Args:
        expr: Root IR expression to inline into.
        aliases: Mapping from alias name (a ``Sym`` name) to its IR body.
            Keys are expected to be fully expanded alias names (after
            template expansion), and bodies are expected to be IR.
        max_depth: Safety bound on the number of substitution rounds.

    Returns:
        A new IR expression with all alias references resolved.

    Raises:
        ValueError: If the alias graph contains a cycle, or if expansion
            does not converge within ``max_depth`` iterations.
    """
    if not aliases:
        return expr

    cycle = _detect_alias_cycle(aliases)
    if cycle is not None:
        msg = f"alias cycle detected: {' -> '.join(cycle)}"
        raise ValueError(msg)

    keys = set(aliases)
    current = expr
    for _ in range(max_depth):
        live = free_symbols(current) & keys
        if not live:
            return current
        mapping = {name: aliases[name] for name in live}
        current = substitute(current, mapping)

    msg = f"alias inlining did not converge within {max_depth} iterations"
    raise ValueError(msg)
