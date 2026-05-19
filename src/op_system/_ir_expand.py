"""IR-side expansion of ``Reduce`` nodes into pointwise IR.

This module mirrors the string-level :func:`_normalize._expand_apply_along`
pipeline but operates on typed IR. After running
:func:`expand_reduce_pointwise` on an expression every ``Reduce`` node
(``apply_along``, ``sum_over``, ``integrate_over``) is folded into an
explicit ``Apply("+", ...)`` chain of per-coord pointwise terms, where
each :class:`Subscript` reference whose bound positions are resolved has
its name mangled (e.g. ``I__pop_p1``) or — for multi-axis shaped
parameters whose every position is resolved — is rewritten as a
literal-integer subscript (e.g. ``K[2, 0]``).

The pass is bottom-up: nested ``Reduce`` nodes are expanded first, so
the surrounding pass only ever sees already-pointwise children.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any

from op_system._ir import (
    Apply,
    AxisIndex,
    Expr,
    Literal,
    Reduce,
    Subscript,
    Sym,
)
from op_system._templates import _sanitize_fragment

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def expand_reduce_pointwise(
    expr: Expr,
    *,
    axes: Sequence[Mapping[str, Any]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    lhs_assignment: Mapping[str, str] | None = None,
    axis_coords: Mapping[str, Sequence[str]] | None = None,
) -> Expr:
    """Recursively expand ``Reduce`` nodes into pointwise IR.

    Args:
        expr: Root IR expression to expand.
        axes: Normalized axis dicts (each with ``name``, ``type``,
            ``coords``, and optional ``deltas``).
        shaped_params: Mapping from shaped-parameter name to its declared
            axis tuple. Multi-axis entries enable integer-indexed
            rewriting when every position is resolved.
        lhs_assignment: Free-axis assignment from the equation's LHS
            template row (e.g. ``{"age": "a0"}``). Required to
            disambiguate same-axis-twice references like
            ``K[age, age:ap]`` inside a templated equation.
        axis_coords: ``{axis_name: [coord, ...]}`` for resolving coord
            names to integer positions on shaped-parameter axes.

    Returns:
        A pointwise IR expression: same shape as ``expr`` but with every
        ``Reduce`` replaced by its expanded weighted-sum form.
    """
    shaped_params = shaped_params or {}
    lhs_assignment = lhs_assignment or {}
    axis_coords = axis_coords or {}

    # Bottom-up: expand children first so the current node only sees
    # already-pointwise sub-expressions.
    expanded = _expand_children(
        expr,
        axes=axes,
        shaped_params=shaped_params,
        lhs_assignment=lhs_assignment,
        axis_coords=axis_coords,
    )
    if isinstance(expanded, Reduce):
        return _expand_one_reduce(
            expanded,
            axes=axes,
            shaped_params=shaped_params,
            lhs_assignment=lhs_assignment,
            axis_coords=axis_coords,
        )
    return expanded


def _expand_children(
    expr: Expr,
    *,
    axes: Sequence[Mapping[str, Any]],
    shaped_params: Mapping[str, tuple[str, ...]],
    lhs_assignment: Mapping[str, str],
    axis_coords: Mapping[str, Sequence[str]],
) -> Expr:
    """Recurse into ``expr``'s children, expanding ``Reduce`` nodes.

    Returns:
        ``expr`` with every nested ``Reduce`` replaced by its expansion.
    """
    if isinstance(expr, (Literal, Sym, Subscript)):
        return expr
    if isinstance(expr, Apply):
        return Apply(
            op=expr.op,
            args=tuple(
                expand_reduce_pointwise(
                    a,
                    axes=axes,
                    shaped_params=shaped_params,
                    lhs_assignment=lhs_assignment,
                    axis_coords=axis_coords,
                )
                for a in expr.args
            ),
        )
    if isinstance(expr, Reduce):
        new_body = expand_reduce_pointwise(
            expr.body,
            axes=axes,
            shaped_params=shaped_params,
            lhs_assignment=lhs_assignment,
            axis_coords=axis_coords,
        )
        return Reduce(
            kind=expr.kind,
            bindings=expr.bindings,
            body=new_body,
            filters=expr.filters,
            kernel=expr.kernel,
        )
    return expr


def _expand_one_reduce(
    expr: Reduce,
    *,
    axes: Sequence[Mapping[str, Any]],
    shaped_params: Mapping[str, tuple[str, ...]],
    lhs_assignment: Mapping[str, str],
    axis_coords: Mapping[str, Sequence[str]],
) -> Expr:
    """Expand a single ``Reduce`` node into a pointwise weighted-sum IR.

    Returns:
        The pointwise IR replacement for ``expr``.
    """
    # Late import to avoid a hard cycle: _normalize imports _ir_expand
    # for its IR-build path.
    from op_system._normalize import (  # noqa: PLC0415
        _build_apply_along_axis_options,
        _select_apply_along_kernel,
    )

    axis_lookup: dict[str, dict[str, Any]] = {
        str(ax.get("name")): dict(ax) for ax in axes if ax.get("name")
    }
    axis_order = tuple(str(ax["name"]) for ax in axes if ax.get("name"))
    filters_dict: dict[str, list[str] | None] = {
        ax: list(coords) if coords else None for ax, coords in expr.filters
    }
    bindings: list[tuple[str, str, list[str] | None]] = [
        (ax, var, filters_dict.get(ax)) for ax, var in expr.bindings
    ]

    # Determine the kernel form (sum|integrate|None=auto) from Reduce.kind
    # and explicit Reduce.kernel; mirror the string-side selection.
    kind = expr.kind
    if kind == "sum_over":
        kernel_form: str | None = "sum"
    elif kind == "integrate_over":
        kernel_form = "integrate"
    else:  # "apply_along"
        kernel_form = expr.kernel
    axes_list: list[dict[str, Any]] = [dict(ax) for ax in axes]
    kernel = _select_apply_along_kernel(bindings, kernel_form, axes=axes_list)

    axis_options = [
        _build_apply_along_axis_options(
            ax_name, filt, kernel=kernel, axis=axis_lookup[ax_name]
        )
        for ax_name, _var, filt in bindings
    ]

    terms: list[Expr] = []
    for combo in product(*axis_options):
        var_to_coord: dict[str, str] = {}
        bound: dict[str, str] = {}
        weight = 1.0
        for (ax_name, var_name, _filt), (coord, w) in zip(bindings, combo, strict=True):
            var_to_coord[var_name] = coord
            bound[ax_name] = coord
            weight *= w
        replaced = _substitute_in_body(
            expr.body,
            var_to_coord=var_to_coord,
            bound=bound,
            axis_order=axis_order,
            shaped_params=shaped_params,
            lhs_assignment=lhs_assignment,
            axis_coords=axis_coords,
        )
        if kernel == "integrate":
            replaced = Apply(op="*", args=(Literal(value=weight), replaced))
        terms.append(replaced)

    if not terms:  # pragma: no cover - guarded by axis_options non-empty
        return Literal(value=0.0)
    if len(terms) == 1:
        return terms[0]
    # Flat N-ary Apply avoids O(N) recursion depth in _unparse_ir / lower.
    return Apply(op="+", args=tuple(terms))


def _substitute_in_body(  # noqa: PLR0911, PLR0913
    body: Expr,
    *,
    var_to_coord: Mapping[str, str],
    bound: Mapping[str, str],
    axis_order: Sequence[str],
    shaped_params: Mapping[str, tuple[str, ...]],
    lhs_assignment: Mapping[str, str],
    axis_coords: Mapping[str, Sequence[str]],
) -> Expr:
    """Substitute bound coords into ``body`` (Subscripts and arithmetic Syms).

    Inner ``Reduce`` nodes are *not* re-expanded here (the caller handles
    that bottom-up); but their inner body is recursively substituted while
    shadowing any inner binding variable from ``var_to_coord``.

    Returns:
        A new IR expression with substitutions applied.
    """
    if isinstance(body, Literal):
        return body
    if isinstance(body, Sym):
        if body.name in var_to_coord:
            # Mirror the string-side re.sub(rf"\b{var}\b", coord, ...);
            # the coord token becomes a bare identifier (which downstream
            # treats as either a literal coord name or a defined Sym).
            return Sym(name=var_to_coord[body.name])
        return body
    if isinstance(body, Apply):
        return Apply(
            op=body.op,
            args=tuple(
                _substitute_in_body(
                    a,
                    var_to_coord=var_to_coord,
                    bound=bound,
                    axis_order=axis_order,
                    shaped_params=shaped_params,
                    lhs_assignment=lhs_assignment,
                    axis_coords=axis_coords,
                )
                for a in body.args
            ),
        )
    if isinstance(body, Subscript):
        return _rewrite_subscript(
            body,
            var_to_coord=var_to_coord,
            bound=bound,
            axis_order=axis_order,
            shaped_params=shaped_params,
            lhs_assignment=lhs_assignment,
            axis_coords=axis_coords,
        )
    if isinstance(body, Reduce):
        inner_vars = {var for _, var in body.bindings}
        outer_var_to_coord = {
            k: v for k, v in var_to_coord.items() if k not in inner_vars
        }
        new_inner = _substitute_in_body(
            body.body,
            var_to_coord=outer_var_to_coord,
            bound=bound,
            axis_order=axis_order,
            shaped_params=shaped_params,
            lhs_assignment=lhs_assignment,
            axis_coords=axis_coords,
        )
        return Reduce(
            kind=body.kind,
            bindings=body.bindings,
            body=new_inner,
            filters=body.filters,
            kernel=body.kernel,
        )
    return body  # pragma: no cover - exhaustive over Expr union


def _rewrite_subscript(  # noqa: C901, PLR0912, PLR0913, PLR0915
    sub: Subscript,
    *,
    var_to_coord: Mapping[str, str],
    bound: Mapping[str, str],
    axis_order: Sequence[str],
    shaped_params: Mapping[str, tuple[str, ...]],
    lhs_assignment: Mapping[str, str],
    axis_coords: Mapping[str, Sequence[str]],
) -> Expr:
    """IR-level equivalent of :func:`_substitute_apply_along_brackets`.

    For each :class:`AxisIndex` position:

    * ``placeholder`` indices are passed through unchanged.
    * ``coord`` indices whose value names a binding variable resolve to
      that binding's current coord (consumed).
    * Bare ``axis`` tokens consume to ``bound[axis]`` (or, in the
      same-axis-twice case, to ``lhs_assignment[axis]``).
    * Literal ``coord`` indices that happen to equal ``bound[axis]`` are
      consumed; otherwise they remain as-is.

    If every position of a multi-axis shaped parameter is resolved, the
    result is a literal-integer :class:`Subscript`. Otherwise the
    consumed (axis, coord) pairs are folded into a canonical-order
    ``__axis_<coord>`` suffix on the name and any remaining positions
    are kept in brackets.

    Returns:
        The rewritten IR node (``Sym`` if all positions are consumed,
        ``Subscript`` if some positions remain, or the original ``sub``
        unchanged if no positions were consumed).
    """
    name = sub.name

    # Detect same-axis-twice: an axis appears with both a bare token AND
    # a coord token that names a binding variable.
    pinned_axes: set[str] = set()
    bare_axes: set[str] = set()
    for idx in sub.indices:
        if idx.placeholder is not None:
            continue
        if idx.coord is not None:
            if idx.coord in var_to_coord:
                pinned_axes.add(idx.axis)
        elif idx.axis:
            bare_axes.add(idx.axis)
    same_axis_twice = pinned_axes & bare_axes

    consumed: list[tuple[str, str]] = []
    remaining: list[AxisIndex] = []
    resolved_full: list[tuple[str, str | None]] = []
    for idx in sub.indices:
        if idx.placeholder is not None:
            resolved_full.append((idx.axis, None))
            remaining.append(idx)
            continue
        if idx.coord is not None:
            if idx.coord in var_to_coord:
                c = var_to_coord[idx.coord]
                resolved_full.append((idx.axis, c))
                consumed.append((idx.axis, c))
            elif idx.axis in bound and bound[idx.axis] == idx.coord:
                resolved_full.append((idx.axis, idx.coord))
                consumed.append((idx.axis, idx.coord))
            else:
                resolved_full.append((idx.axis, idx.coord))
                remaining.append(idx)
            continue
        # Bare axis token.
        if idx.axis in same_axis_twice and idx.axis in lhs_assignment:
            c = lhs_assignment[idx.axis]
            resolved_full.append((idx.axis, c))
            consumed.append((idx.axis, c))
        elif idx.axis in bound:
            c = bound[idx.axis]
            resolved_full.append((idx.axis, c))
            consumed.append((idx.axis, c))
        else:
            resolved_full.append((idx.axis, None))
            remaining.append(idx)

    if not consumed:
        return sub

    # Multi-axis shaped parameter → integer-indexed Subscript when every
    # position is resolved and every axis matches the param's declared
    # order positionally.
    if name in shaped_params and len(shaped_params[name]) > 1:
        param_axes = shaped_params[name]
        if (
            len(resolved_full) == len(param_axes)
            and not remaining
            and all(c is not None for _, c in resolved_full)
            and all(ax == param_axes[i] for i, (ax, _) in enumerate(resolved_full))
            and all(ax in axis_coords for ax in param_axes)
        ):
            try:
                int_idx = [
                    list(axis_coords[param_axes[i]]).index(c)
                    for i, (_, c) in enumerate(resolved_full)
                    if c is not None
                ]
            except ValueError:
                int_idx = None
            if int_idx is not None:
                return Subscript(
                    name=name,
                    indices=tuple(AxisIndex(axis="", coord=str(i)) for i in int_idx),
                )

    # Otherwise: fold consumed pairs into a name suffix (canonical axis
    # order). Merge with any pre-existing suffix on the name so nested
    # apply_along expansions converge on the same canonical state name.
    base, existing = _split_name_suffix(name, axis_order)
    suffix = _emit_suffix(existing + consumed, axis_order)
    new_name = f"{base}{suffix}"
    if remaining:
        return Subscript(name=new_name, indices=tuple(remaining))
    return Sym(name=new_name)


def _split_name_suffix(
    name: str, axis_order: Sequence[str]
) -> tuple[str, list[tuple[str, str]]]:
    """Strip trailing ``__<axis>_<coord>`` tokens recognised by ``axis_order``.

    Mirrors :func:`_normalize._substitute_apply_along_brackets._split_suffix`
    so that nested ``apply_along`` expansions converge on the same
    canonical mangled state name regardless of binding order.

    Returns:
        ``(base_name, [(axis, coord), ...])`` where pairs are in left-to-
        right order as they appeared in ``name``.
    """
    if not axis_order:
        return name, []
    parts = name.split("__")
    if len(parts) <= 1:
        return name, []
    suffix_pairs: list[tuple[str, str]] = []
    cut = len(parts)
    for i in range(len(parts) - 1, 0, -1):
        tok = parts[i]
        best: str | None = None
        for ax in axis_order:
            if (
                tok.startswith(f"{ax}_")
                and len(tok) > len(ax) + 1
                and (best is None or len(ax) > len(best))
            ):
                best = ax
        if best is None:
            break
        coord = tok[len(best) + 1 :]
        suffix_pairs.append((best, coord))
        cut = i
    if not suffix_pairs:
        return name, []
    suffix_pairs.reverse()
    base = "__".join(parts[:cut])
    return base, suffix_pairs


def _emit_suffix(pairs: list[tuple[str, str]], axis_order: Sequence[str]) -> str:
    """Render ``pairs`` as ``__axis_<coord>`` tokens in canonical axis order.

    Returns:
        The concatenated suffix string (empty when ``pairs`` is empty).
    """
    priority = {ax: i for i, ax in enumerate(axis_order)}
    if not priority:
        return "".join(f"__{ax}_{_sanitize_fragment(c)}" for ax, c in pairs)
    known = sorted(
        (p for p in pairs if p[0] in priority),
        key=lambda p: priority[p[0]],
    )
    unknown = [p for p in pairs if p[0] not in priority]
    return "".join(f"__{ax}_{_sanitize_fragment(c)}" for ax, c in known + unknown)
