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
    HistoryOp,
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


_EMPTY_AXES: frozenset[str] = frozenset()


def _free_axes_in(
    node: Expr,
    *,
    shaped: Mapping[str, tuple[str, ...]],
    memo: dict[int, frozenset[str]],
) -> frozenset[str]:
    """Return the set of placeholder axis names referenced under ``node``.

    Walks the IR subtree counting any unbound axis appearing inside a
    ``Subscript``. For shaped-parameter subscripts (whose axes are rewritten
    to literal integer coords by ``_rewrite_shaped_indices``), the
    registered axis tuple is included so callers using this set for
    disjoint-check pruning still descend when ``assignment`` covers them.
    ``Reduce`` bindings shadow outer names and are subtracted from the body
    set. Memoized by ``id(node)`` so repeated calls with the same memo
    visit each unique subtree at most once (issue #145).
    """
    cached = memo.get(id(node))
    if cached is not None:
        return cached
    result = _compute_free_axes(node, shaped=shaped, memo=memo)
    memo[id(node)] = result
    return result


def _free_axes_subscript(
    node: Subscript, *, shaped: Mapping[str, tuple[str, ...]]
) -> frozenset[str]:
    if node.name in shaped:
        # Treat all registered axes as referenced so shaped rewrite
        # still fires when ``assignment`` covers them.
        return frozenset(shaped[node.name])
    out: set[str] = set()
    for idx in node.indices:
        if idx.coord is None and idx.axis:
            out.add(idx.axis)
    return frozenset(out) or _EMPTY_AXES


def _compute_free_axes(
    node: Expr,
    *,
    shaped: Mapping[str, tuple[str, ...]],
    memo: dict[int, frozenset[str]],
) -> frozenset[str]:
    if isinstance(node, Subscript):
        return _free_axes_subscript(node, shaped=shaped)
    if isinstance(node, Apply):
        acc: set[str] = set()
        for arg in node.args:
            acc |= _free_axes_in(arg, shaped=shaped, memo=memo)
        return frozenset(acc) or _EMPTY_AXES
    if isinstance(node, Reduce):
        body_axes = _free_axes_in(node.body, shaped=shaped, memo=memo)
        bound = {b for _, b in node.bindings}
        if not (bound and body_axes):
            return body_axes
        return (body_axes - bound) or _EMPTY_AXES
    if isinstance(node, HistoryOp):
        acc: set[str] = set(_free_axes_in(node.body, shaped=shaped, memo=memo))
        for _, opt_expr in node.options:
            acc |= _free_axes_in(opt_expr, shaped=shaped, memo=memo)
        return frozenset(acc) or _EMPTY_AXES
    # Literal, Sym, and any other leaf node carry no free axes.
    return _EMPTY_AXES


def expand_inline_templates(
    expr: Expr,
    *,
    assignment: Mapping[str, str],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    axis_lookup: Mapping[str, Sequence[str]] | None = None,
    _free_axes_memo: dict[int, frozenset[str]] | None = None,
    _expand_result_memo: dict[tuple[object, ...], Expr] | None = None,
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
        _free_axes_memo: Optional id-keyed cache of the per-subtree
            unbound-axis set. When supplied, the walk short-circuits at
            compound nodes whose free-axis set is disjoint from
            ``assignment``. Callers that invoke this function repeatedly
            with the same root (e.g. per-coord alias pinning) should
            share one dict to avoid recomputing the cache. Caller-owned
            so callers control lifetime; pass an empty dict to enable.
        _expand_result_memo: Optional result cache keyed by
            ``(id(node), *sorted_projected_assignment_items)`` where the
            projected assignment contains only the axes actually free in
            the subtree.  Two calls that agree on the axes a subtree
            references but differ on irrelevant axes return the same
            expanded object, eliminating redundant rebuilds of heavy
            subtrees when only a "thin" LHS axis (e.g. ``loc`` in
            ``foi[age, loc]``) changes between cells (issue #147).
            Caller-owned; pass an empty dict to enable.  Must NOT be
            shared across calls with a different root template.

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
    # Compound nodes (Apply / Reduce / HistoryOp): skip the recursion entirely when
    # the cached free-axis set is disjoint from ``assignment``. This is
    # the dominant win for per-coord alias pinning where ``assignment``
    # is ``{<one-axis>: ...}`` but the body spans many axes (issue #145).
    if (
        _free_axes_memo is not None
        and assignment
        and _free_axes_in(expr, shaped=shaped, memo=_free_axes_memo).isdisjoint(
            assignment
        )
    ):
        return expr
    # Result cache: for compound nodes whose free-axis set is a STRICT
    # SUBSET of ``assignment.keys()``, two assignments that agree on the
    # relevant axes but differ on irrelevant ones produce the same
    # expanded subtree.  By projecting the assignment onto the free-axis
    # set before building the cache key we share that result instead of
    # rebuilding.  For ``foi[age, loc]`` the heavy 21-term sum only
    # references ``age``; its cache key omits ``loc``, so all 51 loc
    # variants of the same age value hit the same entry (issue #147).
    if _expand_result_memo is not None and assignment:
        fa_memo_used: dict[int, frozenset[str]] = (
            _free_axes_memo if _free_axes_memo is not None else {}
        )
        node_free = _free_axes_in(expr, shaped=shaped, memo=fa_memo_used)
        relevant = node_free & frozenset(assignment)
        cache_key: tuple[object, ...] = (
            id(expr),
            *sorted((k, assignment[k]) for k in relevant),
        )
        cached_result = _expand_result_memo.get(cache_key)
        if cached_result is not None:
            return cached_result
    else:
        cache_key = ()
    if isinstance(expr, Apply):
        result = _expand_apply(
            expr,
            assignment=assignment,
            shaped_params=shaped,
            axis_lookup=axis_lookup,
            free_axes_memo=_free_axes_memo,
            expand_result_memo=_expand_result_memo,
        )
    elif isinstance(expr, Reduce):
        result = _expand_reduce(
            expr,
            assignment=assignment,
            shaped_params=shaped,
            axis_lookup=axis_lookup,
            free_axes_memo=_free_axes_memo,
            expand_result_memo=_expand_result_memo,
        )
    elif isinstance(expr, HistoryOp):
        result = _expand_history_op(
            expr,
            assignment=assignment,
            shaped_params=shaped,
            axis_lookup=axis_lookup,
            free_axes_memo=_free_axes_memo,
            expand_result_memo=_expand_result_memo,
        )
    else:
        msg = f"unsupported IR node in expand_inline_templates: {type(expr).__name__}"
        raise TypeError(msg)
    if _expand_result_memo is not None and cache_key:
        _expand_result_memo[cache_key] = result
    return result


def _expand_apply(  # noqa: PLR0913
    expr: Apply,
    *,
    assignment: Mapping[str, str],
    shaped_params: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, Sequence[str]] | None,
    free_axes_memo: dict[int, frozenset[str]] | None,
    expand_result_memo: dict[tuple[object, ...], Expr] | None = None,
) -> Expr:
    new_args = tuple(
        expand_inline_templates(
            arg,
            assignment=assignment,
            shaped_params=shaped_params,
            axis_lookup=axis_lookup,
            _free_axes_memo=free_axes_memo,
            _expand_result_memo=expand_result_memo,
        )
        for arg in expr.args
    )
    if new_args == expr.args:
        return expr
    return Apply(op=expr.op, args=new_args)


def _expand_reduce(  # noqa: PLR0913
    expr: Reduce,
    *,
    assignment: Mapping[str, str],
    shaped_params: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, Sequence[str]] | None,
    free_axes_memo: dict[int, frozenset[str]] | None,
    expand_result_memo: dict[tuple[object, ...], Expr] | None = None,
) -> Expr:
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
        shaped_params=shaped_params,
        axis_lookup=axis_lookup,
        _free_axes_memo=free_axes_memo,
        _expand_result_memo=expand_result_memo,
    )
    if new_body is expr.body:
        return expr
    return Reduce(
        kind=expr.kind,
        bindings=expr.bindings,
        body=new_body,
        filters=expr.filters,
        kernel=expr.kernel,
    )


def _expand_history_op(  # noqa: PLR0913
    expr: HistoryOp,
    *,
    assignment: Mapping[str, str],
    shaped_params: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, Sequence[str]] | None,
    free_axes_memo: dict[int, frozenset[str]] | None,
    expand_result_memo: dict[tuple[object, ...], Expr] | None = None,
) -> Expr:
    new_body = expand_inline_templates(
        expr.body,
        assignment=assignment,
        shaped_params=shaped_params,
        axis_lookup=axis_lookup,
        _free_axes_memo=free_axes_memo,
        _expand_result_memo=expand_result_memo,
    )
    new_options = tuple(
        (
            k,
            expand_inline_templates(
                opt_expr,
                assignment=assignment,
                shaped_params=shaped_params,
                axis_lookup=axis_lookup,
                _free_axes_memo=free_axes_memo,
                _expand_result_memo=expand_result_memo,
            ),
        )
        for k, opt_expr in expr.options
    )
    if new_body is expr.body and new_options == expr.options:
        return expr
    return HistoryOp(kind=expr.kind, body=new_body, options=new_options)


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
        return
    if isinstance(node, HistoryOp):
        _walk_for_axes(node.body, shaped=shaped, seen=seen)
        for _, opt_expr in node.options:
            _walk_for_axes(opt_expr, shaped=shaped, seen=seen)


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


def inline_aliases(  # noqa: C901, PLR0913
    expr: Expr,
    aliases: Mapping[str, Expr],
    *,
    max_depth: int = 64,
    memo: dict[int, frozenset[str]] | None = None,
    skip_cycle_check: bool = False,
    result_memo: dict[int, Expr] | None = None,
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
        result_memo: Optional identity-keyed cache mapping ``id(expr)`` to
            the fully-inlined result. When the same IR object is inlined
            many times against the same ``aliases`` (e.g. one template's
            synthesized IR shared across many state cells), pass a
            single dict to amortize the substitution work across calls.

    Returns:
        A new IR expression with all alias references resolved.

    Raises:
        ValueError: If the alias graph contains a cycle, or if expansion
            does not converge within ``max_depth`` iterations.
    """
    if not aliases:
        return expr
    if result_memo is not None:
        cached_result = result_memo.get(id(expr))
        if cached_result is not None:
            return cached_result

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
    #
    # The precompute is LAZY: ``body_refs`` is built one alias at a time,
    # only for aliases that actually appear in the expression being inlined
    # (and transitively in their bodies). Eagerly precomputing for every
    # alias OOMs on continuum specs where ``aliases`` has O(thousands) of
    # per-cell entries, even when the expression being inlined references
    # only one of them (issue #147 followup).
    cached_refs = _BODY_REFS_CACHE.get(id(aliases))
    if cached_refs is None or cached_refs[0] is not aliases:
        body_refs: dict[str, frozenset[str]] = {}
        _BODY_REFS_CACHE[id(aliases)] = (aliases, body_refs)
    else:
        body_refs = cached_refs[1]

    def _refs_for(name: str) -> frozenset[str]:
        cached = body_refs.get(name)
        if cached is not None:
            return cached
        refs = free_symbols(aliases[name], memo) & keys
        body_refs[name] = refs
        return refs

    current = expr
    live = free_symbols(current, memo) & keys
    for _ in range(max_depth):
        if not live:
            if result_memo is not None:
                result_memo[id(expr)] = current
            return current
        mapping = {name: aliases[name] for name in live}
        current = substitute(current, mapping, memo)
        next_live: set[str] = set()
        for name in live:
            next_live |= _refs_for(name)
        live = frozenset(next_live)

    msg = f"alias inlining did not converge within {max_depth} iterations"
    raise ValueError(msg)
