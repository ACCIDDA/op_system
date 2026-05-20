"""op_system._normalize.

Core normalization entry points for op_system RHS specifications.

Public entry points:

- ``normalize_rhs``:             dispatch to expr or transitions normalizer
- ``normalize_expr_rhs``:        normalize an ``expr``-kind spec
- ``normalize_transitions_rhs``: normalize a ``transitions``-kind spec

The large helper subsystems have been extracted into dedicated modules:

- :mod:`op_system._normalize_ir`: :class:`StateTemplate`, shaped-param
  scanning, time-varying stripping, alias/equation IR builders, string
  derivation.
- :mod:`op_system._normalize_kernels`: kernel, operator, state-axes, and
  apply_along normalization.
- :mod:`op_system._normalize_initial_state`: ``initial_state`` block
  expansion.
- :mod:`op_system._normalize_chains`: chain helper and coord-shift expansion.

The public types and functions are re-exported from ``specs.py`` for backward
compatibility.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

from op_system._axes import _normalize_axes, _normalize_bracket_key
from op_system._errors import InvalidRhsSpecError, UnsupportedFeatureError
from op_system._helpers import (
    _ensure_mapping,
    _ensure_str_dict,
    _ensure_str_list,
    _get_required_str,
    _sorted_unique,
)
from op_system._ir import (
    Apply,
    Expr,
    Literal,
    Sym,
    free_symbols,
    parse_expr_to_ir,
    unparse_ir,
)
from op_system._ir_templates import expand_inline_templates
from op_system._normalize_chains import (
    _apply_coord_shifts,
    _apply_expr_chains,
    _apply_transition_chains,
)
from op_system._normalize_initial_state import _maybe_attach_initial_state
from op_system._normalize_ir import (
    _SHAPED_PARAM_BUILTIN_NAMES,
    StateTemplate,
    _build_alias_templates,
    _build_aliases_ir_from_raw,
    _build_equations_ir,
    _build_equations_ir_from_raw,
    _build_state_templates,
    _derive_alias_strings,
    _derive_equation_strings,
    _expand_state_templates,
    _partition_time_varying_shaped,
    _reject_legacy_time_varying_field,
    _resolve_time_axis_name,
    _scan_shaped_param_refs,
    _strip_time_axis_in_expr,
    _strip_time_axis_in_mapping,
)
from op_system._normalize_kernels import (
    _normalize_kernels,
    _normalize_operators,
    _normalize_state_axes,
)
from op_system._templates import (
    WildcardToken,
    _apply_template_substitutions,
    _extract_placeholders_from_expr,
    build_axis_lookup,
    parse_selector,
    render_selector,
)

__all__ = [
    "ExprRhs",
    "NormalizedRhs",
    "StateTemplate",
    "TransitionsRhs",
    "normalize_expr_rhs",
    "normalize_rhs",
    "normalize_transitions_rhs",
]


# ---------------------------------------------------------------------------
# Normalized RHS representation (backend-facing public type)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RhsBase:
    """Shared fields for all normalized-RHS kinds.  Not part of the public API.

    Use :data:`NormalizedRhs`, :class:`ExprRhs`, or :class:`TransitionsRhs`
    instead.
    """

    state_names: tuple[str, ...]
    equations: tuple[str, ...]
    aliases: Mapping[str, str]
    param_names: tuple[str, ...]
    all_symbols: frozenset[str]
    meta: Mapping[str, Any]
    state_templates: tuple[StateTemplate, ...] = ()
    shaped_params: tuple[tuple[str, tuple[str, ...]], ...] = ()
    time_varying_params: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Post-expansion, alias-inlined IR (Reduce nodes expanded).
    # Used by the scalar compile path (``compile.py._make_eval_fn``).
    aliases_ir: Mapping[str, Expr] = field(default_factory=dict)
    equations_ir: tuple[Expr | None, ...] = ()
    # Reduce-bearing IR (Reduce nodes preserved).
    # Used by the vector compile path (``_vectorize.py``).
    aliases_ir_reduce: Mapping[str, Expr] = field(default_factory=dict)
    equations_ir_reduce: tuple[Expr | None, ...] = ()
    alias_templates: tuple[StateTemplate, ...] = ()


@dataclass(frozen=True, slots=True)
class ExprRhs(_RhsBase):
    """Normalized RHS for ``kind="expr"`` specs (explicit d(state)/dt equations).

    Produced by :func:`normalize_expr_rhs`.  Use :data:`NormalizedRhs` as the
    union type when you need to accept both kinds.
    """


@dataclass(frozen=True, slots=True)
class TransitionsRhs(_RhsBase):
    """Normalized RHS for ``kind="transitions"`` specs (per-capita hazard diagram).

    Produced by :func:`normalize_transitions_rhs`.  Use :data:`NormalizedRhs`
    as the union type when you need to accept both kinds.
    """


#: Discriminated union of the two normalized-RHS kinds.
#:
#: Use ``isinstance(rhs, ExprRhs)`` / ``isinstance(rhs, TransitionsRhs)`` to
#: dispatch, rather than the legacy ``rhs.kind == "expr"`` string check.
NormalizedRhs = ExprRhs | TransitionsRhs


# ---------------------------------------------------------------------------
# Small shared validators
# ---------------------------------------------------------------------------


def _validate_transition_mapping(tr: object, *, idx: int) -> Mapping[str, Any]:
    """Validate and return a transition mapping.

    Returns:
        Validated transition mapping.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if not isinstance(tr, dict):
        raise InvalidRhsSpecError(detail=f"transitions[{idx}] must be a mapping")
    if "name" in tr:
        name_val = tr.get("name")
        if not isinstance(name_val, str) or not name_val.strip():
            raise InvalidRhsSpecError(
                detail=f"transitions[{idx}].name must be a non-empty string"
            )
    return tr


# ---------------------------------------------------------------------------
# Common meta normalization
# ---------------------------------------------------------------------------


def _get_meta_field(spec: Mapping[str, Any], key: str) -> object:
    """Look up a field inside an optional nested ``meta`` block.

    Returns:
        The field value, or ``None`` if *meta* is absent or the key is missing.
    """
    meta_block = spec.get("meta")
    if meta_block is None:
        return None
    return _ensure_mapping(meta_block, name="meta").get(key)


def _normalize_common_meta(
    spec: Mapping[str, Any],
    *,
    axis_names: set[str],
    state_set: set[str] | None,
    operator_state_set: set[str] | None = None,
    axes: list[dict[str, Any]] | None = None,
) -> tuple[
    dict[str, str],
    dict[str, tuple[str, ...]],
    list[dict[str, Any]],
    object,
]:
    """Normalize aliases, state_axes, kernels, and operators from *spec*.

    Returns:
        Tuple of ``(aliases, state_axes, kernels, operators)``.
    """
    aliases = _ensure_str_dict(spec.get("aliases"), name="aliases")
    state_axes = (
        _normalize_state_axes(
            spec.get("state_axes"), axis_names=axis_names, state_set=state_set
        )
        if state_set is not None
        else {}
    )
    kernels_meta = _normalize_kernels(
        spec.get("kernels") or _get_meta_field(spec, "kernels"),
        axis_names=axis_names,
    )
    operators_raw = spec.get("operators") or _get_meta_field(spec, "operators")
    operators_meta = (
        _normalize_operators(
            operators_raw,
            axis_names=axis_names,
            state_set=operator_state_set,
            axes=axes,
        )
        if isinstance(operators_raw, list)
        else operators_raw
    )
    return aliases, state_axes, kernels_meta, operators_meta


# ---------------------------------------------------------------------------
# Public normalization entry points
# ---------------------------------------------------------------------------


def normalize_rhs(spec: Mapping[str, Any] | None) -> NormalizedRhs:
    """Normalize a RHS specification dict into a backend-facing representation.

    Args:
        spec: Raw RHS specification mapping.

    Returns:
        Backend-facing normalized RHS representation.

    Raises:
        InvalidRhsSpecError: If validation fails.
        UnsupportedFeatureError: If validation fails.
    """
    if spec is None:
        raise InvalidRhsSpecError(detail="rhs specification is required")

    kind = str(spec.get("kind", "expr")).strip().lower()

    if kind == "expr":
        return normalize_expr_rhs(spec)

    if kind == "transitions":
        return normalize_transitions_rhs(spec)

    raise UnsupportedFeatureError(
        feature=f"rhs.kind={kind}",
        detail="Only 'expr' and 'transitions' are supported in v1.",
    )


def normalize_expr_rhs(spec: Mapping[str, Any]) -> ExprRhs:  # noqa: C901, PLR0914, PLR0915
    """Normalize an expression-based RHS specification.

    Args:
        spec: Raw RHS specification mapping.

    Returns:
        Backend-facing normalized RHS representation.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    state_raw = _ensure_str_list(spec.get("state"), name="state")
    if len(state_raw) != len(set(state_raw)):
        raise InvalidRhsSpecError(detail="state contains duplicate names")

    equations_map = spec.get("equations")
    if not isinstance(equations_map, dict):
        raise InvalidRhsSpecError(detail="equations must be a mapping of state->expr")

    equations_map = {_normalize_bracket_key(k): v for k, v in equations_map.items()}

    axes_meta = _normalize_axes(spec.get("axes"))
    meta_parts = _normalize_common_meta(
        spec,
        axis_names={"subgroup"} | {ax["name"] for ax in axes_meta},
        state_set=set(state_raw),
        operator_state_set=set(state_raw),
        axes=axes_meta,
    )

    meta: dict[str, Any] = {
        "axes": axes_meta,
        "state_axes": meta_parts[1],
        "kernels": meta_parts[2],
        "operators": meta_parts[3],
    }
    for reserved_key in ("sources", "couplings", "constraints"):
        if reserved_key in spec:
            meta[reserved_key] = spec.get(reserved_key)

    state_expanded, state_template_map = _expand_state_templates(
        state_raw, axes=axes_meta
    )
    if len(state_expanded) != len(set(state_expanded)):
        raise InvalidRhsSpecError(detail="expanded state contains duplicates")

    # Pre-scan raw aliases + equations for shaped-parameter references before
    # alias/template expansion mangles bracketed bases into per-coord names.
    axis_lookup_dict: dict[str, list[str]] = build_axis_lookup(axes_meta)
    aliases_raw_map = meta_parts[0] or {}
    name_blocklist = (
        {parse_selector(s)[0] for s in state_raw}
        | set(state_expanded)
        | {parse_selector(_normalize_bracket_key(k))[0] for k in aliases_raw_map}
        | set(aliases_raw_map.keys())
    )
    raw_expressions: list[str] = [
        v for v in aliases_raw_map.values() if isinstance(v, str)
    ]
    raw_expressions.extend(v for v in equations_map.values() if isinstance(v, str))
    shaped_params = _scan_shaped_param_refs(
        raw_expressions,
        name_blocklist=name_blocklist,
        axis_lookup=axis_lookup_dict,
    )
    _reject_legacy_time_varying_field(spec)
    time_axis_name = _resolve_time_axis_name(spec)
    shaped_params, time_varying_full = _partition_time_varying_shaped(
        shaped_params,
        time_axis_name=time_axis_name,
        axis_lookup=axis_lookup_dict,
    )
    if time_varying_full:
        _strip_time_axis_in_mapping(
            equations_map,
            tv_full_axes=time_varying_full,
            time_axis_name=time_axis_name,
        )
        if isinstance(meta_parts[0], dict):
            _strip_time_axis_in_mapping(
                meta_parts[0],
                tv_full_axes=time_varying_full,
                time_axis_name=time_axis_name,
            )

    aliases_ir_map, aliases_ir_reduce_map, alias_template_map = (
        _build_aliases_ir_from_raw(
            meta_parts[0],
            axes=axes_meta,
            shaped_params=shaped_params,
            axis_lookup=axis_lookup_dict,
        )
    )
    template_map_all = {**state_template_map, **alias_template_map}

    chain_block = spec.get("chain")
    if chain_block:
        if not isinstance(chain_block, list):
            raise InvalidRhsSpecError(detail="chain must be a list if provided")
        _apply_expr_chains(
            chains=chain_block,
            state_expanded=state_expanded,
            equations_map=equations_map,
        )

    unknown_keys = [
        k
        for k in equations_map
        if k not in state_expanded and k not in template_map_all
    ]
    if unknown_keys:
        raise InvalidRhsSpecError(
            detail=f"unknown equation key(s): {sorted(unknown_keys)}"
        )

    equations_ir_built, equations_ir_reduce, all_syms = _build_equations_ir_from_raw(
        state_expanded=state_expanded,
        equations_map=equations_map,
        template_map=template_map_all,
        axes=axes_meta,
        shaped_params=shaped_params,
        axis_lookup=axis_lookup_dict,
        aliases_ir=aliases_ir_map,
    )
    # Collect free symbols from alias bodies (alias bodies may reference
    # params not appearing in any equation directly).
    for alias_ir_val in aliases_ir_map.values():
        all_syms |= free_symbols(alias_ir_val)

    _maybe_attach_initial_state(
        meta,
        spec.get("initial_state"),
        axes=axes_meta,
        template_map=template_map_all,
    )

    meta["shaped_params"] = tuple(sorted(shaped_params.items()))
    meta["time_axis"] = time_axis_name
    meta["time_varying_params"] = tuple(sorted(time_varying_full.items()))

    shaped_set = set(shaped_params)
    time_varying_set = set(time_varying_full)
    axis_name_set = set(axis_lookup_dict)
    template_base_set = {parse_selector(k)[0] for k in template_map_all}
    equations_strings = _derive_equation_strings(equations_ir_built)
    aliases_strings = _derive_alias_strings(aliases_ir_map, aliases_ir_map)

    return ExprRhs(
        state_names=tuple(state_expanded),
        equations=equations_strings,
        aliases=aliases_strings,
        param_names=_sorted_unique(
            sym
            for sym in all_syms
            if sym not in set(state_expanded)
            and sym not in aliases_ir_map
            and sym not in shaped_set
            and sym not in time_varying_set
            and sym not in axis_name_set
            and sym not in template_base_set
            and sym not in _SHAPED_PARAM_BUILTIN_NAMES
        ),
        all_symbols=frozenset(all_syms | set(aliases_ir_map.keys())),
        meta=meta,
        state_templates=_build_state_templates(
            state_raw,
            axes=axes_meta,
            state_template_map=state_template_map,
            state_expanded=state_expanded,
        ),
        shaped_params=tuple(sorted(shaped_params.items())),
        time_varying_params=tuple(sorted(time_varying_full.items())),
        aliases_ir=aliases_ir_map,
        equations_ir=equations_ir_built,
        aliases_ir_reduce=aliases_ir_reduce_map,
        equations_ir_reduce=equations_ir_reduce,
        alias_templates=_build_alias_templates(
            aliases_raw_map,
            axes=axes_meta,
            alias_template_map=alias_template_map,
        ),
    )


# ---------------------------------------------------------------------------
# transitions kind helpers
# ---------------------------------------------------------------------------


def _build_transition_equations_ir(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915
    transitions_raw: list[Mapping[str, Any]],
    *,
    state_set: set[str],
    state_expanded: list[str],
    axes: list[dict[str, Any]],
    axis_lookup: dict[str, list[str]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[
    tuple[Expr | None, ...],
    tuple[Expr | None, ...],
    list[dict[str, Any]],
    set[str],
]:
    """Build per-state equation IR and expanded transitions from raw transition specs.

    Returns:
        ``(equations_ir_pre, equations_ir_reduce_pre, transitions_expanded, all_syms)``

    Raises:
        InvalidRhsSpecError: If any transition is invalid.
    """
    from op_system._ir_expand import expand_reduce_pointwise  # noqa: PLC0415

    shaped = shaped_params or {}
    ax_lookup_dict = dict(axis_lookup)
    d_ir_full: dict[str, list[Expr]] = {s: [] for s in state_expanded}
    d_ir_reduce: dict[str, list[Expr]] = {s: [] for s in state_expanded}
    all_syms: set[str] = set()
    transitions_expanded_out: list[dict[str, Any]] = []

    old_limit = sys.getrecursionlimit()
    needed = max(old_limit, 10_000)
    try:
        if needed > old_limit:
            sys.setrecursionlimit(needed)
        for tr_idx, tr_map in enumerate(transitions_raw):
            tr_valid = _validate_transition_mapping(dict(tr_map), idx=tr_idx)
            frm_s = _get_required_str(tr_valid, idx=tr_idx, key="from")
            to_s = _get_required_str(tr_valid, idx=tr_idx, key="to")
            rate_s = _get_required_str(tr_valid, idx=tr_idx, key="rate")
            name_s = (
                tr_valid.get("name") if isinstance(tr_valid.get("name"), str) else None
            )

            frm_base, frm_tokens = parse_selector(frm_s)
            to_base, to_tokens = parse_selector(to_s)

            # Collect wildcard axes
            wildcard_axes: list[str] = []
            seen_wc: set[str] = set()
            for tok in frm_tokens + to_tokens:
                if isinstance(tok, WildcardToken) and tok.axis not in seen_wc:
                    wildcard_axes.append(tok.axis)
                    seen_wc.add(tok.axis)
            skip_shaped = set(shaped)
            expr_phs = _extract_placeholders_from_expr(
                rate_s, shaped_param_names=skip_shaped
            )
            if name_s:
                expr_phs |= _extract_placeholders_from_expr(
                    name_s, shaped_param_names=skip_shaped
                )
            for ph in sorted(expr_phs):
                if ":" in ph or "=" in ph:
                    continue
                if ph not in seen_wc:
                    if ph not in ax_lookup_dict:
                        raise InvalidRhsSpecError(
                            detail=(
                                f"transition placeholder {ph!r} references unknown axis"
                            )
                        )
                    wildcard_axes.append(ph)
                    seen_wc.add(ph)

            # Parse rate to IR once (Reduce nodes preserved for apply_along)
            ir_rate_raw = parse_expr_to_ir(rate_s, lower_helpers=True)

            # Build wildcard combos
            if not wildcard_axes:
                combos: Iterable[tuple[str, ...]] = [()]
            else:
                coords_lists = [ax_lookup_dict[ph] for ph in wildcard_axes]
                combos = product(*coords_lists)

            for combo in combos:
                assignment = dict(zip(wildcard_axes, combo, strict=True))
                from_name = render_selector(
                    frm_base, frm_tokens, assignment, axis_lookup=ax_lookup_dict
                )
                to_name = render_selector(
                    to_base, to_tokens, assignment, axis_lookup=ax_lookup_dict
                )
                if from_name not in state_set:
                    raise InvalidRhsSpecError(
                        detail=f"transitions[{tr_idx}].from={from_name!r} not in state"
                    )
                if to_name not in state_set:
                    raise InvalidRhsSpecError(
                        detail=f"transitions[{tr_idx}].to={to_name!r} not in state"
                    )

                # Rate IR: template-substituted, Reduce nodes preserved
                ir_rate_reduce = expand_inline_templates(
                    ir_rate_raw,
                    assignment=assignment,
                    shaped_params=shaped,
                    axis_lookup=ax_lookup_dict,
                )
                # Rate IR: fully expanded, no Reduce nodes
                ir_rate_full = expand_reduce_pointwise(
                    ir_rate_reduce,
                    axes=list(axes),
                    shaped_params=shaped,
                    lhs_assignment=assignment,
                    axis_coords=ax_lookup_dict,
                )

                # Build flow IR: rate * from_state
                from_sym = Sym(name=from_name)
                flow_full = Apply(op="*", args=(ir_rate_full, from_sym))
                flow_reduce = Apply(op="*", args=(ir_rate_reduce, from_sym))

                # Accumulate: from_state gets -flow, to_state gets +flow
                d_ir_full[from_name].append(Apply(op="neg", args=(flow_full,)))
                d_ir_full[to_name].append(flow_full)
                d_ir_reduce[from_name].append(Apply(op="neg", args=(flow_reduce,)))
                d_ir_reduce[to_name].append(flow_reduce)

                # Collect rate symbols (alias Syms kept unresolved)
                all_syms |= free_symbols(ir_rate_full)

                # Build expanded transition dict for meta["transitions"]
                tr_out: dict[str, Any] = dict(tr_valid)
                tr_out["from"] = from_name
                tr_out["to"] = to_name
                tr_out["rate"] = unparse_ir(ir_rate_full)
                if name_s:
                    tr_out["name"] = _apply_template_substitutions(
                        name_s,
                        assignment=assignment,
                        template_map=template_map,
                        shaped_params=shaped,
                        axis_lookup=ax_lookup_dict,
                    )
                transitions_expanded_out.append(tr_out)
    finally:
        if needed > old_limit:
            sys.setrecursionlimit(old_limit)

    def _sum_terms(terms: list[Expr]) -> Expr:
        if not terms:
            return Literal(value=0.0)
        if len(terms) == 1:
            return terms[0]
        return Apply(op="+", args=tuple(terms))

    equations_ir_pre = tuple(_sum_terms(d_ir_full[s]) for s in state_expanded)
    equations_ir_reduce_pre = tuple(_sum_terms(d_ir_reduce[s]) for s in state_expanded)
    return equations_ir_pre, equations_ir_reduce_pre, transitions_expanded_out, all_syms


def normalize_transitions_rhs(  # noqa: C901, PLR0912, PLR0914, PLR0915
    spec: Mapping[str, Any],
) -> TransitionsRhs:
    """Normalize a transition-based RHS specification (diagram/hazard semantics).

    Returns:
        Backend-facing normalized RHS representation for the transitions kind.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    state_raw = _ensure_str_list(spec.get("state"), name="state")
    if len(state_raw) != len(set(state_raw)):
        raise InvalidRhsSpecError(detail="state contains duplicate names")

    transitions_raw = spec.get("transitions")
    if transitions_raw is None:
        transitions_raw = []
    elif isinstance(transitions_raw, list):
        transitions_raw = list(transitions_raw)
    else:
        raise InvalidRhsSpecError(detail="transitions must be a list")

    axes_meta = _normalize_axes(spec.get("axes"))

    meta_parts = _normalize_common_meta(
        spec,
        axis_names={"subgroup"} | {ax["name"] for ax in axes_meta},
        state_set=None,
        operator_state_set=set(state_raw),
        axes=axes_meta,
    )

    meta: dict[str, Any] = {
        "transitions": transitions_raw,
        "axes": axes_meta,
        "kernels": meta_parts[2],
        "operators": meta_parts[3],
    }
    meta.update({
        k: spec[k] for k in ("sources", "couplings", "constraints") if k in spec
    })

    chain_block = spec.get("chain")
    if chain_block:
        if not isinstance(chain_block, list):
            raise InvalidRhsSpecError(detail="chain must be a list if provided")
        _apply_transition_chains(
            chains=chain_block,
            state_raw=state_raw,
            transitions_raw=transitions_raw,
            state_set=set(state_raw),
        )

    state_expanded, state_template_map = _expand_state_templates(
        state_raw, axes=axes_meta
    )
    if len(state_expanded) != len(set(state_expanded)):
        raise InvalidRhsSpecError(detail="expanded state contains duplicates")

    # Pre-scan raw aliases + transition rates for shaped-parameter references.
    axis_lookup_dict: dict[str, list[str]] = build_axis_lookup(axes_meta)
    aliases_raw_map = meta_parts[0] or {}
    name_blocklist = (
        {parse_selector(s)[0] for s in state_raw}
        | set(state_expanded)
        | {parse_selector(_normalize_bracket_key(k))[0] for k in aliases_raw_map}
        | set(aliases_raw_map.keys())
    )
    raw_expressions: list[str] = [
        v for v in aliases_raw_map.values() if isinstance(v, str)
    ]
    for tr in transitions_raw:
        if isinstance(tr, _MappingABC):
            r = tr.get("rate")
            if isinstance(r, str):
                raw_expressions.append(r)
            n = tr.get("name")
            if isinstance(n, str):
                raw_expressions.append(n)
    shaped_params = _scan_shaped_param_refs(
        raw_expressions,
        name_blocklist=name_blocklist,
        axis_lookup=axis_lookup_dict,
    )
    _reject_legacy_time_varying_field(spec)
    time_axis_name = _resolve_time_axis_name(spec)
    shaped_params, time_varying_full = _partition_time_varying_shaped(
        shaped_params,
        time_axis_name=time_axis_name,
        axis_lookup=axis_lookup_dict,
    )
    if time_varying_full:
        if isinstance(meta_parts[0], dict):
            _strip_time_axis_in_mapping(
                meta_parts[0],
                tv_full_axes=time_varying_full,
                time_axis_name=time_axis_name,
            )
        for tr in transitions_raw:
            if not isinstance(tr, _MappingABC):
                continue
            r = tr.get("rate")
            if isinstance(r, str):
                tr["rate"] = _strip_time_axis_in_expr(  # type: ignore[index]
                    r,
                    tv_full_axes=time_varying_full,
                    time_axis_name=time_axis_name,
                )
            n = tr.get("name")
            if isinstance(n, str):
                tr["name"] = _strip_time_axis_in_expr(  # type: ignore[index]
                    n,
                    tv_full_axes=time_varying_full,
                    time_axis_name=time_axis_name,
                )

    aliases_ir_map, aliases_ir_reduce_map, alias_template_map = (
        _build_aliases_ir_from_raw(
            aliases_raw_map,
            axes=axes_meta,
            shaped_params=shaped_params,
            axis_lookup=axis_lookup_dict,
        )
    )
    template_map_all = {**state_template_map, **alias_template_map}

    state_set = set(state_expanded)
    # Collect alias symbols from IR
    all_syms: set[str] = set()
    for alias_expr in aliases_ir_map.values():
        all_syms |= free_symbols(alias_expr)

    _apply_coord_shifts(
        transitions_raw=transitions_raw,
        state_expanded=state_expanded,
        axes=axes_meta,
    )

    if not transitions_raw:
        raise InvalidRhsSpecError(
            detail="transitions must be non-empty after applying chain expansion"
        )

    # Build equations and collect expanded transitions in one IR-native pass
    equations_ir_pre_inline, equations_ir_reduce, transitions_expanded, rate_syms = (
        _build_transition_equations_ir(
            transitions_raw,
            state_set=state_set,
            state_expanded=state_expanded,
            axes=axes_meta,
            axis_lookup=axis_lookup_dict,
            template_map=template_map_all,
            shaped_params=shaped_params,
        )
    )
    all_syms |= rate_syms

    _maybe_attach_initial_state(
        meta,
        spec.get("initial_state"),
        axes=axes_meta,
        template_map=template_map_all,
    )

    meta["shaped_params"] = tuple(sorted(shaped_params.items()))
    meta["time_axis"] = time_axis_name
    meta["time_varying_params"] = tuple(sorted(time_varying_full.items()))

    shaped_set = set(shaped_params)
    time_varying_set = set(time_varying_full)
    axis_name_set = set(axis_lookup_dict)
    template_base_set = {parse_selector(k)[0] for k in template_map_all}
    eqs_tuple = _derive_equation_strings(equations_ir_pre_inline)
    equations_ir_built = _build_equations_ir(eqs_tuple, aliases_ir_map)
    return TransitionsRhs(
        state_names=tuple(state_expanded),
        equations=eqs_tuple,
        aliases=_derive_alias_strings(aliases_ir_map, aliases_ir_map),
        param_names=_sorted_unique(
            sym
            for sym in all_syms
            if sym not in state_set
            and sym not in aliases_ir_map
            and sym not in shaped_set
            and sym not in time_varying_set
            and sym not in axis_name_set
            and sym not in template_base_set
            and sym not in _SHAPED_PARAM_BUILTIN_NAMES
        ),
        all_symbols=frozenset(all_syms | set(aliases_ir_map.keys())),
        meta={**meta, "transitions": transitions_expanded},
        state_templates=_build_state_templates(
            state_raw,
            axes=axes_meta,
            state_template_map=state_template_map,
            state_expanded=state_expanded,
        ),
        shaped_params=tuple(sorted(shaped_params.items())),
        time_varying_params=tuple(sorted(time_varying_full.items())),
        aliases_ir=aliases_ir_map,
        equations_ir=equations_ir_built,
        aliases_ir_reduce=aliases_ir_reduce_map,
        equations_ir_reduce=equations_ir_reduce,
        alias_templates=_build_alias_templates(
            aliases_raw_map,
            axes=axes_meta,
            alias_template_map=alias_template_map,
        ),
    )
