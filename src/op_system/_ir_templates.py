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


# Identity-keyed cache mapping ``id(aliases)`` -> ``(aliases, body_refs)``.
# Many ``inline_aliases`` calls in batch passes share the same ``aliases``
# mapping, and the ``body_refs`` precomputation is the dominant non-substitute
# cost (free_symbols over each large alias body). Cache by identity but also
# store the dict object so we can defensively guard against ``id()`` reuse
# across distinct ``aliases`` instances (issue #145).
_BODY_REFS_CACHE: dict[int, tuple[Mapping[str, Expr], dict[str, frozenset[str]]]] = {}


def _expandable_axes(indices: Sequence[AxisIndex]) -> list[str] | None:
    """Return the list of bare-identifier axis names from ``indices``.

    Returns ``None`` if any index carries a literal coord, in
    which case the subscript is not eligible for template expansion.
    """
    out: list[str] = []
    for idx in indices:
        if idx.coord is not None:
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
        that are not eligible (literal coords, unknown axis
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
        # Reduce bindings shadow outer names: a bound name in this
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
    "free_axes",
    "inline_aliases",
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
        if idx.coord is not None:
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


def free_axes(
    expr: Expr,
    *,
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Collect free (unbound, non-literal) axis names referenced by ``expr``.

    Walks the IR and gathers every bare-identifier axis appearing inside a
    ``Subscript`` whose base name is *not* a registered shaped parameter.
    Literal-coord positions are skipped. Order is deterministic: first
    appearance in pre-order traversal wins.

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


def _detect_alias_cycle(
    aliases: Mapping[str, Expr],
    *,
    memo: dict[int, frozenset[str]] | None = None,
) -> list[str] | None:
    """Return a cycle of alias names if one exists, else ``None``.

    Builds a name -> referenced-alias-names graph (only edges into the
    alias namespace itself count) and DFS-checks for a back edge.

    Args:
        aliases: Mapping of alias name to IR body.
        memo: Optional identity-keyed ``free_symbols`` cache (shared
            with downstream ``inline_aliases`` calls to avoid
            recomputing per-body free-symbol sets).
    """
    keys = set(aliases)
    if memo is None:
        memo = {}
    graph: dict[str, frozenset[str]] = {
        name: frozenset(free_symbols(body, memo=memo) & keys)
        for name, body in aliases.items()
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
    memo: dict[int, frozenset[str]] | None = None,
    skip_cycle_check: bool = False,
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
        memo: Optional identity-keyed cache shared with
            :func:`op_system._ir.free_symbols`. Pass a single dict across
            many inline calls that share the same alias bodies to avoid
            re-walking each body per equation.
        skip_cycle_check: If ``True``, bypass :func:`_detect_alias_cycle`.
            Callers that batch-inline many expressions against the same
            ``aliases`` mapping should validate once and set this flag on
            subsequent calls.

    Returns:
        A new IR expression with all alias references resolved.

    Raises:
        ValueError: If the alias graph contains a cycle, or if expansion
            does not converge within ``max_depth`` iterations.
    """
    if not aliases:
        return expr

    if not skip_cycle_check:
        cycle = _detect_alias_cycle(aliases)
        if cycle is not None:
            msg = f"alias cycle detected: {' -> '.join(cycle)}"
            raise ValueError(msg)

    if memo is None:
        memo = {}
    keys = set(aliases)
    # Precompute, for each alias body, which other alias names it references.
    # The next "live" set after a substitution is exactly the union of these
    # entries over the just-inlined names, so we never need to re-walk the
    # substituted expression with ``free_symbols`` (which on large inlined
    # bodies dominates inline cost when many equations share one alias).
    # Cache by ``id(aliases)`` so batch-inlining many expressions against the
    # same alias mapping pays this O(|aliases| * body_size) cost only once
    # rather than once per call (issue #145).
    cached_refs = _BODY_REFS_CACHE.get(id(aliases))
    if cached_refs is None:
        body_refs: dict[str, frozenset[str]] = {
            name: free_symbols(body, memo) & keys for name, body in aliases.items()
        }
        _BODY_REFS_CACHE[id(aliases)] = (aliases, body_refs)
    else:
        # Guard against id() reuse: only trust the cache when the mapping
        # object is the same instance.
        cached_obj, cached_body_refs = cached_refs
        body_refs = (
            cached_body_refs
            if cached_obj is aliases
            else {
                name: free_symbols(body, memo) & keys for name, body in aliases.items()
            }
        )
    current = expr
    live = free_symbols(current, memo) & keys
    for _ in range(max_depth):
        if not live:
            return current
        mapping = {name: aliases[name] for name in live}
        current = substitute(current, mapping, memo)
        next_live: set[str] = set()
        for name in live:
            next_live |= body_refs[name]
        live = frozenset(next_live)

    msg = f"alias inlining did not converge within {max_depth} iterations"
    raise ValueError(msg)
