"""op_system._normalize_ir.

IR-level normalization helpers: ``StateTemplate``, shaped-param scanning,
time-varying stripping, alias/equation IR builders, and string derivation.

All public entry points remain in ``_normalize.py``.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import re
    from collections.abc import Iterable, Mapping, Sequence

from op_system._axes import _normalize_bracket_key
from op_system._errors import InvalidRhsSpecError
from op_system._ir import Expr, free_symbols, parse_expr_to_ir, unparse_ir
from op_system._ir_templates import (
    _detect_alias_cycle,
    expand_inline_templates,
    inline_aliases,
)
from op_system._templates import (
    _INLINE_TEMPLATE_RE,
    PinnedToken,
    WildcardToken,
    build_axis_lookup,
    expand_selector,
    parse_selector,
)

# ---------------------------------------------------------------------------
# StateTemplate dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StateTemplate:
    """Structural record for a state template prior to scalar expansion.

    A spec like ``state: ["S[age, vax]"]`` over axes ``age`` (size 4) and
    ``vax`` (size 2) produces a single `StateTemplate` with `base="S"`,
    `axes=("age", "vax")`, `shape=(4, 2)`, and `expanded_names` listing the
    eight scalar state names in cartesian-product order (`age` outer,
    `vax` inner — matching `itertools.product` and the order used to expand
    the flat state vector).

    Non-templated entries (e.g. a bare ``"D"`` in `state`) are also reported
    as `StateTemplate` records with ``axes=()``, ``shape=()``, and a single
    `expanded_names = ("D",)` so consumers can iterate templates uniformly.

    Attributes:
        base: Compartment name without selector brackets (e.g. ``"S"``).
        axes: Wildcard axes in declaration order. Empty for scalar templates.
        shape: Per-axis sizes in `axes` order. Empty for scalar templates.
        expanded_names: Flat scalar state names, in cartesian-product order
            over `axes` (consistent with `state_names`).
        coord_assignments: For each entry in `expanded_names`, the
            ``axis -> coord`` mapping. Empty dict for scalar templates.
        offset: Index of `expanded_names[0]` within the parent
            `NormalizedRhs.state_names` tuple (i.e. where this template's
            slice starts in the flat state vector).
    """

    base: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    expanded_names: tuple[str, ...]
    coord_assignments: tuple[Mapping[str, str], ...]
    offset: int


# ---------------------------------------------------------------------------
# Shaped-param constants and helpers
# ---------------------------------------------------------------------------

_SHAPED_PARAM_BUILTIN_NAMES: frozenset[str] = frozenset({
    "np",
    "t",
    "sum_state",
    "sum_prefix",
})


def _scan_shaped_param_refs(  # noqa: C901, PLR0912
    expressions: Iterable[str],
    *,
    name_blocklist: set[str],
    axis_lookup: Mapping[str, list[str]],
) -> dict[str, tuple[str, ...]]:
    """Scan expressions for shaped-parameter references.

    Returns:
        Mapping from base name to its registered axes tuple.

    Raises:
        InvalidRhsSpecError: If a name is referenced with inconsistent axes.
    """
    result: dict[str, tuple[str, ...]] = {}
    for expr in expressions:
        if not expr:
            continue
        for match in _INLINE_TEMPLATE_RE.finditer(expr):
            base = match.group(1)
            if base in name_blocklist or base in _SHAPED_PARAM_BUILTIN_NAMES:
                continue
            inner = match.group(2)
            parts = [p.strip() for p in inner.split(",")]
            if not parts:
                continue
            axes: list[str] = []
            valid = True
            for p in parts:
                if not p:
                    valid = False
                    break
                if ":" in p:
                    ax_name, _, _ = p.partition(":")
                    ax_name = ax_name.strip()
                elif "=" in p:
                    ax_name, _, _ = p.partition("=")
                    ax_name = ax_name.strip()
                else:
                    ax_name = p
                if ax_name not in axis_lookup:
                    valid = False
                    break
                axes.append(ax_name)
            if not valid:
                continue
            axes_tuple = tuple(axes)
            if base in result:
                if result[base] != axes_tuple:
                    raise InvalidRhsSpecError(
                        detail=(
                            f"shaped parameter {base!r} referenced with "
                            f"inconsistent axes: {result[base]} vs {axes_tuple}"
                        )
                    )
            else:
                result[base] = axes_tuple
    return result


def _resolve_time_axis_name(spec: Mapping[str, Any]) -> str:
    """Return the configured time-axis name (default ``"time"``).

    Raises:
        InvalidRhsSpecError: If ``time_axis`` is present but not a non-empty
            string.
    """
    raw = spec.get("time_axis", "time")
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidRhsSpecError(
            detail="time_axis must be a non-empty identifier string"
        )
    return raw.strip()


def _reject_legacy_time_varying_field(spec: Mapping[str, Any]) -> None:
    """Raise a migration-friendly error if the legacy ``time_varying`` field is present.

    Raises:
        InvalidRhsSpecError: If ``spec`` contains a ``time_varying`` key.
    """
    if "time_varying" in spec:
        raise InvalidRhsSpecError(
            detail=(
                "the top-level 'time_varying' field has been removed; declare "
                "time-varying parameters implicitly by subscripting them with "
                "the configured time axis (default 'time'), e.g. "
                "'beta[time, age]'."
            )
        )


def _partition_time_varying_shaped(
    shaped_params: Mapping[str, tuple[str, ...]],
    *,
    time_axis_name: str,
    axis_lookup: Mapping[str, list[str]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Split shaped-parameter axes into non-time-varying and time-varying.

    Returns:
        Pair ``(shaped_reduced, time_varying_full)``.

    Raises:
        InvalidRhsSpecError: If a parameter declares the time axis but the
            axis itself was not declared in the spec's ``axes`` block.
    """
    shaped_reduced: dict[str, tuple[str, ...]] = {}
    time_varying_full: dict[str, tuple[str, ...]] = {}
    for name, axes in shaped_params.items():
        if time_axis_name not in axes:
            shaped_reduced[name] = axes
            continue
        if time_axis_name not in axis_lookup:
            raise InvalidRhsSpecError(
                detail=(
                    f"parameter {name!r} is subscripted with the time axis "
                    f"{time_axis_name!r} but no such axis is declared in 'axes'"
                )
            )
        reduced = tuple(ax for ax in axes if ax != time_axis_name)
        shaped_reduced[name] = reduced
        time_varying_full[name] = axes
    return shaped_reduced, time_varying_full


def _strip_time_axis_in_expr(
    expr: str,
    *,
    tv_full_axes: Mapping[str, tuple[str, ...]],
    time_axis_name: str,
) -> str:
    """Rewrite ``name[time, age]`` → ``name[age]`` (or bare ``name``) in ``expr``.

    Returns:
        The rewritten expression.
    """
    if not tv_full_axes or not expr:
        return expr

    def _rewrite(match: re.Match[str]) -> str:
        """Rewrite one ``base[time, ...]`` match by dropping the time axis."""
        base = match.group(1)
        full = tv_full_axes.get(base)
        if full is None:
            return match.group(0)
        parts = tuple(p.strip() for p in match.group(2).split(","))
        if parts != full:
            return match.group(0)
        reduced = tuple(p for p in parts if p != time_axis_name)
        if not reduced:
            return base
        return f"{base}[{', '.join(reduced)}]"

    return _INLINE_TEMPLATE_RE.sub(_rewrite, expr)


def _strip_time_axis_in_mapping(
    mapping: dict[str, Any],
    *,
    tv_full_axes: Mapping[str, tuple[str, ...]],
    time_axis_name: str,
) -> None:
    """In-place rewrite of all string values in ``mapping``.

    Delegates to :func:`_strip_time_axis_in_expr` for each value.
    """
    if not tv_full_axes:
        return
    for key, value in list(mapping.items()):
        if isinstance(value, str):
            mapping[key] = _strip_time_axis_in_expr(
                value, tv_full_axes=tv_full_axes, time_axis_name=time_axis_name
            )


# ---------------------------------------------------------------------------
# Template expansion helpers
# ---------------------------------------------------------------------------


def _expand_state_templates(
    state_raw: list[str],
    *,
    axes: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[tuple[str, dict[str, str]]]]]:
    """Expand state templates with categorical axes into concrete names.

    Returns:
        A tuple of ``(expanded_state_names, template_map)``.
    """
    axis_lookup = build_axis_lookup(axes)
    expanded: list[str] = []
    template_map: dict[str, list[tuple[str, dict[str, str]]]] = {}

    for entry in state_raw:
        base, tokens = parse_selector(entry)
        results = expand_selector(
            entry, axis_lookup=axis_lookup, context=f"state entry {entry!r}"
        )
        if not tokens:
            expanded.append(base)
        else:
            wildcards = [t for t in tokens if isinstance(t, WildcardToken)]
            if wildcards:
                template_key = f"{base}[{','.join(wt.axis for wt in wildcards)}]"
            else:
                pinned = (t for t in tokens if isinstance(t, PinnedToken))
                template_key = (
                    f"{base}[{','.join(f'{t.axis}={t.coord}' for t in pinned)}]"
                )
            template_map[template_key] = results
            for name, _ in results:
                expanded.append(name)

    return expanded, template_map


def _build_state_templates(
    state_raw: list[str],
    *,
    axes: list[dict[str, Any]],
    state_template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
    state_expanded: list[str],
) -> tuple[StateTemplate, ...]:
    """Build per-template structural records aligned with `state_expanded`.

    Returns:
        Tuple of `StateTemplate`, one per entry in `state_raw`, in order.
    """
    axis_size = {ax["name"]: len(ax.get("coords", ())) for ax in axes}
    name_to_idx = {n: i for i, n in enumerate(state_expanded)}
    templates: list[StateTemplate] = []
    for entry in state_raw:
        base, tokens = parse_selector(entry)
        wildcards = [t for t in tokens if isinstance(t, WildcardToken)]
        pinned = [t for t in tokens if isinstance(t, PinnedToken)]
        if wildcards:
            template_key = f"{base}[{','.join(wt.axis for wt in wildcards)}]"
            results = state_template_map.get(template_key, [])
            ax_names = tuple(wt.axis for wt in wildcards)
            shape = tuple(axis_size[a] for a in ax_names)
            expanded_names = tuple(name for name, _ in results)
            coords = tuple(dict(coord_map) for _, coord_map in results)
            offset = name_to_idx[expanded_names[0]] if expanded_names else 0
            templates.append(
                StateTemplate(
                    base=base,
                    axes=ax_names,
                    shape=shape,
                    expanded_names=expanded_names,
                    coord_assignments=coords,
                    offset=offset,
                )
            )
        elif pinned:
            template_key = f"{base}[{','.join(f'{t.axis}={t.coord}' for t in pinned)}]"
            results = state_template_map.get(template_key, [])
            for scalar_name, coord_map in results:
                templates.append(
                    StateTemplate(
                        base=base,
                        axes=(),
                        shape=(),
                        expanded_names=(scalar_name,),
                        coord_assignments=(dict(coord_map),),
                        offset=name_to_idx[scalar_name],
                    )
                )
        else:
            templates.append(
                StateTemplate(
                    base=base,
                    axes=(),
                    shape=(),
                    expanded_names=(base,),
                    coord_assignments=({},),
                    offset=name_to_idx[base],
                )
            )
    return tuple(templates)


def _build_alias_templates(
    aliases_raw: Mapping[str, Any],
    *,
    axes: list[dict[str, Any]],
    alias_template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> tuple[StateTemplate, ...]:
    """Build per-template structural records for alias names.

    Returns:
        Tuple of :class:`StateTemplate`, one per unique alias template base,
        in declaration order.
    """
    axis_size = {ax["name"]: len(ax.get("coords", ())) for ax in axes}
    templates: list[StateTemplate] = []
    seen: set[str] = set()
    for raw_name in aliases_raw:
        canonical_name = _normalize_bracket_key(raw_name)
        if canonical_name in alias_template_map:
            base, tokens = parse_selector(raw_name)
            wildcards = [t for t in tokens if isinstance(t, WildcardToken)]
            if wildcards:
                ax_names = tuple(wt.axis for wt in wildcards)
                tpl_key = f"{base}[{','.join(ax_names)}]"
                if tpl_key in seen:
                    continue
                seen.add(tpl_key)
                shape = tuple(axis_size.get(a, 0) for a in ax_names)
                results = alias_template_map[canonical_name]
                templates.append(
                    StateTemplate(
                        base=base,
                        axes=ax_names,
                        shape=shape,
                        expanded_names=tuple(name for name, _ in results),
                        coord_assignments=tuple(
                            dict(coord_map) for _, coord_map in results
                        ),
                        offset=0,
                    )
                )
            else:
                for expanded_name, coord_map in alias_template_map[canonical_name]:
                    if expanded_name in seen:
                        continue
                    seen.add(expanded_name)
                    templates.append(
                        StateTemplate(
                            base=base,
                            axes=(),
                            shape=(),
                            expanded_names=(expanded_name,),
                            coord_assignments=(dict(coord_map),),
                            offset=0,
                        )
                    )
        else:
            if raw_name in seen:
                continue
            seen.add(raw_name)
            templates.append(
                StateTemplate(
                    base=raw_name,
                    axes=(),
                    shape=(),
                    expanded_names=(raw_name,),
                    coord_assignments=({},),
                    offset=0,
                )
            )
    return tuple(templates)


# ---------------------------------------------------------------------------
# Alias and equation IR builders
# ---------------------------------------------------------------------------


def _build_aliases_ir(
    aliases: Mapping[str, str],
    *,
    lower_helpers: bool = False,
) -> dict[str, Expr]:
    """Parse each alias body to IR and inline alias-to-alias references.

    Returns:
        Mapping from alias name to its (fully inlined) IR expression.
    """
    old_limit = sys.getrecursionlimit()
    needed = max(old_limit, 10_000)
    try:
        if needed > old_limit:
            sys.setrecursionlimit(needed)
        parsed: dict[str, Expr] = {}
        for name, body in aliases.items():
            parsed[name] = parse_expr_to_ir(body, lower_helpers=lower_helpers)
        memo: dict[int, frozenset[str]] = {}
        cycle_validated = False
        try:
            cycle = _detect_alias_cycle(parsed)
            cycle_validated = cycle is None
        except (ValueError, RecursionError):
            cycle_validated = False
        inlined: dict[str, Expr] = {}
        for name, expr in parsed.items():
            try:
                inlined[name] = inline_aliases(
                    expr,
                    parsed,
                    memo=memo,
                    skip_cycle_check=cycle_validated,
                )
            except (ValueError, RecursionError):
                inlined[name] = expr
        return inlined
    finally:
        if needed > old_limit:
            sys.setrecursionlimit(old_limit)


def _build_equations_ir(
    equations: tuple[str, ...],
    aliases_ir: Mapping[str, Expr] | None = None,
    *,
    lower_helpers: bool = False,
) -> tuple[Expr | None, ...]:
    """Parse each equation RHS to IR, optionally inlining alias references.

    Returns:
        Tuple of IR expressions aligned positionally with ``equations``.
    """
    old_limit = sys.getrecursionlimit()
    needed = max(old_limit, 10_000)
    try:
        if needed > old_limit:
            sys.setrecursionlimit(needed)
        memo: dict[int, frozenset[str]] = {}
        cycle_validated = False
        if aliases_ir:
            try:
                cycle = _detect_alias_cycle(aliases_ir)
                cycle_validated = cycle is None
            except (ValueError, RecursionError):
                cycle_validated = False
        out: list[Expr] = []
        for eq in equations:
            expr = parse_expr_to_ir(eq, lower_helpers=lower_helpers)
            if aliases_ir is None:
                out.append(expr)
                continue
            try:
                out.append(
                    inline_aliases(
                        expr,
                        aliases_ir,
                        memo=memo,
                        skip_cycle_check=cycle_validated,
                    )
                )
            except (ValueError, RecursionError):
                out.append(expr)
        return tuple(out)
    finally:
        if needed > old_limit:
            sys.setrecursionlimit(old_limit)


def _build_aliases_ir_from_raw(  # noqa: C901
    aliases_raw: Mapping[str, str],
    *,
    axes: list[dict[str, Any]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    axis_lookup: Mapping[str, list[str]] | None = None,
) -> tuple[
    dict[str, Expr],
    dict[str, Expr],
    dict[str, list[tuple[str, dict[str, str]]]],
]:
    """Build alias IR directly from raw (pre-expansion) strings.

    Returns:
        ``(aliases_ir, aliases_ir_reduce, alias_template_map)``.

    Raises:
        InvalidRhsSpecError: If any alias body is an invalid expression.
    """
    from op_system._ir_expand import expand_reduce_pointwise  # noqa: PLC0415

    shaped = shaped_params or {}
    ax_lookup = dict(axis_lookup or {})

    _, alias_template_map = _expand_state_templates(list(aliases_raw.keys()), axes=axes)

    reduce_parsed: dict[str, Expr] = {}
    full_parsed: dict[str, Expr] = {}
    old_limit = sys.getrecursionlimit()
    needed = max(old_limit, 10_000)
    try:
        if needed > old_limit:
            sys.setrecursionlimit(needed)

        for raw_name, expr_str in aliases_raw.items():
            canonical_name = _normalize_bracket_key(raw_name)
            expr_s = expr_str.strip() if isinstance(expr_str, str) else ""
            if not expr_s:
                raise InvalidRhsSpecError(
                    detail=f"aliases[{raw_name!r}] must be a non-empty string"
                )
            try:
                ir_raw = parse_expr_to_ir(expr_s, lower_helpers=True)
            except Exception as exc:
                raise InvalidRhsSpecError(
                    detail=f"aliases[{raw_name!r}] has invalid expression: {exc}"
                ) from exc

            if canonical_name in alias_template_map:
                for expanded_name, assignment in alias_template_map[canonical_name]:
                    ir_tmpl = expand_inline_templates(
                        ir_raw,
                        assignment=assignment,
                        shaped_params=shaped,
                        axis_lookup=ax_lookup,
                    )
                    reduce_parsed[expanded_name] = ir_tmpl
                    full_parsed[expanded_name] = expand_reduce_pointwise(
                        ir_tmpl,
                        axes=list(axes),
                        shaped_params=shaped,
                        lhs_assignment=assignment,
                        axis_coords=ax_lookup,
                    )
            else:
                reduce_parsed[raw_name] = ir_raw
                full_parsed[raw_name] = expand_reduce_pointwise(
                    ir_raw,
                    axes=list(axes),
                    shaped_params=shaped,
                    lhs_assignment={},
                    axis_coords=ax_lookup,
                )

        def _inline_all(parsed: dict[str, Expr]) -> dict[str, Expr]:
            """Inline alias references through the alias map (cycle-safe)."""
            memo: dict[int, frozenset[str]] = {}
            cycle_ok = False
            with contextlib.suppress(ValueError, RecursionError):
                cycle_ok = _detect_alias_cycle(parsed) is None
            inlined: dict[str, Expr] = {}
            for name, expr in parsed.items():
                try:
                    inlined[name] = inline_aliases(
                        expr, parsed, memo=memo, skip_cycle_check=cycle_ok
                    )
                except (ValueError, RecursionError):
                    inlined[name] = expr
            return inlined

        return _inline_all(full_parsed), _inline_all(reduce_parsed), alias_template_map
    finally:
        if needed > old_limit:
            sys.setrecursionlimit(old_limit)


def _lookup_cell_expr(
    cell: str,
    equations_map: Mapping[str, Any],
    template_map: Mapping[str, Sequence[tuple[str, Mapping[str, str]]]],
) -> str:
    """Return the raw equation string for *cell*.

    Raises:
        InvalidRhsSpecError: If the cell has no equation or its body is not a
            non-empty string.
    """
    if cell in equations_map:
        raw = equations_map[cell]
        if not isinstance(raw, str) or not raw.strip():
            raise InvalidRhsSpecError(
                detail=f"equations[{cell!r}] must be a non-empty string"
            )
        return raw.strip()
    for template_key, variants in template_map.items():
        if template_key not in equations_map:
            continue
        for expanded_name, _ in variants:
            if expanded_name == cell:
                raw = equations_map[template_key]
                if not isinstance(raw, str) or not raw.strip():
                    raise InvalidRhsSpecError(
                        detail=(
                            f"equations[{template_key!r}] must be a non-empty string"
                        )
                    )
                return raw.strip()
    raise InvalidRhsSpecError(detail=f"Missing equation for state {cell!r}")


def _build_equations_ir_from_raw(  # noqa: PLR0913
    *,
    state_expanded: Sequence[str],
    equations_map: Mapping[str, Any],
    template_map: Mapping[str, Sequence[tuple[str, Mapping[str, str]]]],
    axes: Sequence[Mapping[str, Any]],
    shaped_params: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, Sequence[str]],
    aliases_ir: Mapping[str, Expr] | None = None,
) -> tuple[
    tuple[Expr | None, ...],
    tuple[Expr | None, ...],
    set[str],
]:
    """Build per-cell equation IR directly from raw spec expressions.

    Returns:
        ``(equations_ir, equations_ir_reduce, all_syms)``
    """
    from op_system._ir_expand import expand_reduce_pointwise  # noqa: PLC0415

    cell_to_assignment: dict[str, dict[str, str]] = {}
    for variants in template_map.values():
        for cell, asgmt in variants:
            cell_to_assignment[cell] = dict(asgmt)

    cell_to_expr = {
        cell: _lookup_cell_expr(cell, equations_map, template_map)
        for cell in state_expanded
    }

    alias_memo: dict[int, frozenset[str]] = {}
    alias_cycle_ok = False
    if aliases_ir:
        with contextlib.suppress(ValueError, RecursionError):
            alias_cycle_ok = _detect_alias_cycle(aliases_ir) is None

    out_reduce: list[Expr | None] = []
    out_full: list[Expr | None] = []
    all_syms: set[str] = set()
    old_limit = sys.getrecursionlimit()
    needed = max(old_limit, 10_000)
    try:
        if needed > old_limit:
            sys.setrecursionlimit(needed)
        for cell in state_expanded:
            raw_expr = cell_to_expr[cell]
            assignment = cell_to_assignment.get(cell, {})
            ir = parse_expr_to_ir(raw_expr, lower_helpers=True)
            ir_tmpl = expand_inline_templates(
                ir,
                assignment=assignment,
                shaped_params=shaped_params,
                axis_lookup=axis_lookup,
            )
            out_reduce.append(ir_tmpl)
            ir_expanded = expand_reduce_pointwise(
                ir_tmpl,
                axes=list(axes),
                shaped_params=shaped_params,
                lhs_assignment=assignment,
                axis_coords=dict(axis_lookup),
            )
            if aliases_ir:
                try:
                    ir_inlined = inline_aliases(
                        ir_expanded,
                        aliases_ir,
                        memo=alias_memo,
                        skip_cycle_check=alias_cycle_ok,
                    )
                except (ValueError, RecursionError):
                    ir_inlined = ir_expanded
            else:
                ir_inlined = ir_expanded
            all_syms |= free_symbols(ir_inlined)
            out_full.append(ir_inlined)
        return tuple(out_full), tuple(out_reduce), all_syms
    finally:
        if needed > old_limit:
            sys.setrecursionlimit(old_limit)


# ---------------------------------------------------------------------------
# String derivation from IR
# ---------------------------------------------------------------------------


def _derive_equation_strings(
    equations_ir: tuple[Expr | None, ...],
) -> tuple[str, ...]:
    """Render each equation's RHS directly from typed IR.

    Returns:
        Tuple of equation strings aligned with ``equations_ir``.

    Raises:
        InvalidRhsSpecError: If any equation IR is missing or cannot be
            rendered back to source.
    """
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10_000))
    try:
        rendered: list[str] = []
        for idx, ir in enumerate(equations_ir):
            if ir is None:
                raise InvalidRhsSpecError(
                    detail=f"equations[{idx}] is missing typed IR during rendering"
                )
            try:
                rendered.append(unparse_ir(ir))
            except (ValueError, RecursionError) as exc:
                raise InvalidRhsSpecError(
                    detail=f"equations[{idx}] could not be rendered from typed IR"
                ) from exc
        return tuple(rendered)
    finally:
        with contextlib.suppress(ValueError, RecursionError):
            sys.setrecursionlimit(old_limit)


def _derive_alias_strings(
    aliases_ir: Mapping[str, Expr],
    alias_order: Iterable[str],
) -> dict[str, str]:
    """Render each alias body directly from typed IR.

    Returns:
        New alias mapping with IR-derived bodies.

    Raises:
        InvalidRhsSpecError: If any alias IR is missing or cannot be rendered
            back to source.
    """
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10_000))
    try:
        out: dict[str, str] = {}
        for name in alias_order:
            ir = aliases_ir.get(name)
            if ir is None:
                raise InvalidRhsSpecError(
                    detail=f"alias {name!r} is missing typed IR during rendering"
                )
            try:
                out[name] = unparse_ir(ir)
            except (ValueError, RecursionError) as exc:
                raise InvalidRhsSpecError(
                    detail=f"alias {name!r} could not be rendered from typed IR"
                ) from exc
        return out
    finally:
        with contextlib.suppress(ValueError, RecursionError):
            sys.setrecursionlimit(old_limit)
