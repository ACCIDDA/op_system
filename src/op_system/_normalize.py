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
    AxisIndex,
    Expr,
    Literal,
    Reduce,
    Subscript,
    Sym,
    free_symbols,
    parse_expr_to_ir,
    unparse_ir,
    walk,
)
from op_system._ir_templates import (
    expand_inline_templates,
    inline_aliases,
)
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
    PinnedToken,
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
    # params not appearing in any equation directly). Walk the *reduce*
    # map (Reduce nodes still folded) rather than the fully expanded
    # map: alias inlining is identical between the two, but the reduce
    # form is orders of magnitude smaller for continuum specs. Share a
    # single id-keyed memo across the per-cell entries so common
    # subtrees are visited at most once (issue #145).
    fs_memo: dict[int, frozenset[str]] = {}
    for alias_ir_val in aliases_ir_reduce_map.values():
        all_syms |= free_symbols(alias_ir_val, memo=fs_memo)

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


def _discover_pinned_token_masks(  # noqa: C901
    transitions_raw: list[Mapping[str, Any]],
    *,
    axis_lookup: dict[str, list[str]],
) -> tuple[dict[tuple[str, str], str], dict[str, tuple[float, ...]]]:
    """Discover pinned-token masks and one-hot value tuples.

    Scans transition ``from``/``to`` selectors for ``PinnedToken`` entries
    and allocates a unique mask shaped-param name plus a one-hot value
    tuple for each ``(axis, coord)`` pair encountered.

    Returns:
        ``(mask_names, mask_values)`` where ``mask_names`` maps
        ``(axis, coord) -> mask_name`` and ``mask_values`` maps
        ``mask_name -> tuple[float, ...]`` (1.0 at the pinned coord index,
        0.0 elsewhere; length equals the axis cardinality).
    """
    mask_names: dict[tuple[str, str], str] = {}
    mask_values: dict[str, tuple[float, ...]] = {}
    for tr in transitions_raw:
        if not isinstance(tr, _MappingABC):
            continue
        for key in ("from", "to"):
            s = tr.get(key)
            if not isinstance(s, str):
                continue
            try:
                _, tokens = parse_selector(s)
            except (InvalidRhsSpecError, ValueError):
                continue
            for tok in tokens:
                if not isinstance(tok, PinnedToken):
                    continue
                if tok.axis not in axis_lookup:
                    continue
                coords = axis_lookup[tok.axis]
                if tok.coord not in coords:
                    continue
                key2 = (tok.axis, tok.coord)
                if key2 in mask_names:
                    continue
                idx = coords.index(tok.coord)
                name = f"__op_system_mask__{tok.axis}__{idx}"
                mask_names[key2] = name
                mask_values[name] = tuple(
                    1.0 if i == idx else 0.0 for i in range(len(coords))
                )
    return mask_names, mask_values


def _build_transition_equations_ir(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915
    transitions_raw: list[Mapping[str, Any]],
    *,
    state_set: set[str],
    state_expanded: list[str],
    axes: list[dict[str, Any]],
    axis_lookup: dict[str, list[str]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    mask_names: Mapping[tuple[str, str], str] | None = None,
    time_axis_name: str | None = None,
    alias_bases: set[str] | None = None,
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
    masks = mask_names or {}
    ax_lookup_dict = dict(axis_lookup)
    alias_base_set: set[str] = set(alias_bases or ())
    d_ir_full: dict[str, list[Expr]] = {s: [] for s in state_expanded}
    d_ir_reduce: dict[str, list[Expr]] = {s: [] for s in state_expanded}
    all_syms: set[str] = set()
    transitions_expanded_out: list[dict[str, Any]] = []
    # Group ``state_expanded`` by base name once so the per-transition
    # synthesis step can look up "all cells of this template" via a
    # dict access instead of re-rendering every coord combo through
    # ``render_selector`` (which dominated 3+ s on the COVID19_USA
    # continuum spec). Every cell name is ``{base}__{axis_suffix}``
    # (or just ``{base}`` for axis-less templates) and the transition
    # ``state_set`` check below guarantees tokens are in the canonical
    # state-template order, so the grouped lookup matches the result
    # of enumerating the template's wildcard combos. Issue #145.
    cells_by_base: dict[str, list[str]] = {}
    for _name in state_expanded:
        _base = _name.split("__", 1)[0]
        cells_by_base.setdefault(_base, []).append(_name)
    # Cache: ``(base, template_tokens) -> list[str]`` for
    # ``_enumerate_template_cell_names``. The same (base, tokens) pair is
    # queried once per (from, to) side per combo expansion, so caching cuts
    # the dominant ``render_selector`` walk on continuum specs (issue #145).
    enumerate_template_cell_names_cache: dict[
        tuple[str, tuple[Any, ...]], list[str]
    ] = {}

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

            # Partition wildcard axes by where they appear, to decide whether
            # a template-uniform Reduce node can stand in for per-combo flow
            # accumulation on the destination side. The fully-expanded
            # ``d_ir_full`` always uses per-cell scalar names (needed by the
            # scalar engine and by per-cell fallback), but ``d_ir_reduce``
            # can carry a single Reduce per to-cell when wildcard axes
            # appear only on the ``from`` side — that lets the vectorizer
            # template-lift the to-template equation without tripping the
            # "multiple cells of buffer X co-occur" guard in
            # ``lift_cell_ir_to_template``.
            frm_wc_axes: list[str] = [
                tok.axis for tok in frm_tokens if isinstance(tok, WildcardToken)
            ]
            to_wc_axes: list[str] = [
                tok.axis for tok in to_tokens if isinstance(tok, WildcardToken)
            ]
            frm_wc_set = set(frm_wc_axes)
            to_wc_set = set(to_wc_axes)
            from_only_axes = [ax for ax in frm_wc_axes if ax not in to_wc_set]
            to_only_axes = [ax for ax in to_wc_axes if ax not in frm_wc_set]
            expr_only_axes = [
                ax
                for ax in wildcard_axes
                if ax not in frm_wc_set and ax not in to_wc_set and ax != time_axis_name
            ]
            pinned_tokens_from: list[PinnedToken] = [
                tok for tok in frm_tokens if isinstance(tok, PinnedToken)
            ]
            pinned_tokens_to: list[PinnedToken] = [
                tok for tok in to_tokens if isinstance(tok, PinnedToken)
            ]
            has_pinned = bool(pinned_tokens_from or pinned_tokens_to)
            # All pinned tokens must have a registered mask (discovered in
            # the normalize pre-pass) for the synthesized template-uniform
            # form to be expressible. If any mask is missing we conservatively
            # fall back to per-cell accumulation.
            pinned_masks_ok = all(
                (tok.axis, tok.coord) in masks
                for tok in (*pinned_tokens_from, *pinned_tokens_to)
            )
            # Synthesis combines: source side template-uniform ``-flow`` IR,
            # destination side template-uniform ``Reduce(sum_over)`` over
            # axes present on the from-template but absent from the
            # to-template, with one-hot mask multiplications collapsing
            # pinned-coord slabs.
            synthesize = (
                bool(frm_wc_axes)
                and not to_only_axes
                and not expr_only_axes
                and pinned_masks_ok
                and (bool(from_only_axes) or has_pinned)
            )
            to_names_for_synthesis: set[str] = set()
            from_names_for_synthesis: set[str] = set()
            # Template-uniform non-synth fast-path. When the from-template
            # and the to-template have the same wildcard axis set and
            # there are no pinned tokens, no destination-only axes, and
            # no rate-only axes, every (from_cell, to_cell) combo of this
            # transition contributes the SAME ``rate * from_state``
            # template-form flow to its respective cell. Building that
            # IR once at the template level (``assignment={}``, with the
            # ``from`` reference as a ``Subscript`` carrying symbolic
            # ``AxisIndex(coord=None)`` slots) lets every cell of both
            # templates share the same IR by object identity, which
            # collapses the dominant per-cell ``inline_aliases`` /
            # ``unparse_ir`` / ``free_symbols`` work from O(n_state) to
            # O(n_template) on continuum specs (issue #145).
            tpl_uniform = (
                not synthesize
                and bool(frm_wc_axes)
                and frm_wc_set == to_wc_set
                and not from_only_axes
                and not to_only_axes
                and not expr_only_axes
                and not has_pinned
            )
            if tpl_uniform and alias_base_set:
                # Disable the fast-path when the rate references an
                # alias. The downstream alias-inline pass keys on
                # fully-pinned ``Sym("alias__axis_coord")`` references
                # (produced by per-combo ``expand_inline_templates`` with
                # a non-empty ``assignment``); a template-form
                # ``Subscript("alias", (AxisIndex(coord=None), ...))``
                # would slip through unresolved and break compilation.
                for node in walk(ir_rate_raw):
                    if (
                        isinstance(node, (Sym, Subscript))
                        and node.name in alias_base_set
                    ):
                        tpl_uniform = False
                        break
            tpl_neg_flow_full: Expr | None = None
            tpl_flow_full: Expr | None = None
            tpl_neg_flow_reduce: Expr | None = None
            tpl_flow_reduce: Expr | None = None
            tpl_ir_rate_full: Expr | None = None
            tpl_rate_string: str | None = None
            if tpl_uniform:
                # Build template-form (axes symbolic) rate IR once.
                tpl_ir_rate_reduce = expand_inline_templates(
                    ir_rate_raw,
                    assignment={},
                    shaped_params=shaped,
                    axis_lookup=ax_lookup_dict,
                )
                tpl_ir_rate_full = expand_reduce_pointwise(
                    tpl_ir_rate_reduce,
                    axes=list(axes),
                    shaped_params=shaped,
                    lhs_assignment={},
                    axis_coords=ax_lookup_dict,
                )
                # ``from`` reference as a template-form Subscript over the
                # from-template's wildcard axes.
                tpl_from_sub = Subscript(
                    name=frm_base,
                    indices=tuple(AxisIndex(axis=ax, coord=None) for ax in frm_wc_axes),
                )
                tpl_flow_full = Apply(op="*", args=(tpl_ir_rate_full, tpl_from_sub))
                tpl_flow_reduce = Apply(op="*", args=(tpl_ir_rate_reduce, tpl_from_sub))
                tpl_neg_flow_full = Apply(op="neg", args=(tpl_flow_full,))
                tpl_neg_flow_reduce = Apply(op="neg", args=(tpl_flow_reduce,))
                all_syms |= free_symbols(tpl_ir_rate_full)
                tpl_rate_string = unparse_ir(tpl_ir_rate_full)
            # Reduce bindings: from-template axes that need to be summed
            # away when evaluating the to-side. Always include from-pinned
            # axes (the mask inside the reduce filters to the pinned
            # slab) plus any from-wildcard axes absent from the to-side.
            reduce_bindings_axes = [
                *(ax for ax in frm_wc_axes if ax not in set(to_wc_axes)),
                *(t.axis for t in pinned_tokens_from),
            ]

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

                # Rate IR: template-substituted, Reduce nodes preserved.
                # In the ``tpl_uniform`` fast-path the template-form
                # rate IR was built once before the combo loop and is
                # shared by identity across all combos via
                # ``tpl_neg_flow_full`` / ``tpl_flow_full`` (etc.); we
                # therefore skip per-combo rate construction entirely.
                if tpl_uniform:
                    ir_rate_reduce: Expr = tpl_ir_rate_reduce
                    ir_rate_full: Expr = tpl_ir_rate_full  # type: ignore[assignment]
                else:
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

                # Build flow IR: rate * from_state. In the tpl_uniform
                # fast-path the per-cell ``Sym(from_name)`` reference is
                # replaced by a single shared ``Subscript`` over the
                # from-template's wildcard axes, and the resulting
                # ``flow_full`` / ``flow_reduce`` / negated forms are the
                # SAME IR object for every combo of this transition.
                if tpl_uniform:
                    flow_full: Expr = tpl_flow_full  # type: ignore[assignment]
                    flow_reduce: Expr = tpl_flow_reduce  # type: ignore[assignment]
                    neg_flow_full: Expr = tpl_neg_flow_full  # type: ignore[assignment]
                    neg_flow_reduce: Expr = tpl_neg_flow_reduce  # type: ignore[assignment]
                else:
                    from_sym = Sym(name=from_name)
                    flow_full = Apply(op="*", args=(ir_rate_full, from_sym))
                    flow_reduce = Apply(op="*", args=(ir_rate_reduce, from_sym))
                    neg_flow_full = Apply(op="neg", args=(flow_full,))
                    neg_flow_reduce = Apply(op="neg", args=(flow_reduce,))

                # Accumulate: from_state gets -flow, to_state gets +flow.
                # When synthesizing template-uniform IR for this transition
                # (see post-loop block below), skip the per-combo appends
                # into BOTH ``d_ir_full`` and ``d_ir_reduce``. The synthesized
                # nodes installed after the combo loop carry the same total
                # contribution but are SHARED across all cells of the
                # template. Sharing collapses per-cell ``inline_aliases``
                # work from O(n_state) to O(n_template) when the per-cell
                # inline loop is passed a ``result_memo`` keyed on
                # ``id(expr)`` (issue #145).
                if synthesize:
                    # Synthesis installs template-uniform IR into ALL
                    # cells of both templates after the combo loop; the
                    # per-combo (pinned-slab-only) names from
                    # ``render_selector`` would miss non-pinned cells
                    # whose mask-multiplied contribution is 0.
                    pass
                else:
                    d_ir_full[from_name].append(neg_flow_full)
                    d_ir_full[to_name].append(flow_full)
                    d_ir_reduce[from_name].append(neg_flow_reduce)
                    d_ir_reduce[to_name].append(flow_reduce)

                # Collect rate symbols (alias Syms kept unresolved). Skip
                # in the tpl_uniform fast-path where this was hoisted out
                # of the combo loop.
                if not tpl_uniform:
                    all_syms |= free_symbols(ir_rate_full)

                # Build expanded transition dict for meta["transitions"].
                # In the tpl_uniform fast-path the rate string is the
                # same for every combo (axes symbolic), so reuse the
                # pre-rendered ``tpl_rate_string`` instead of unparsing
                # per combo.
                tr_out: dict[str, Any] = dict(tr_valid)
                tr_out["from"] = from_name
                tr_out["to"] = to_name
                tr_out["rate"] = (
                    tpl_rate_string if tpl_uniform else unparse_ir(ir_rate_full)
                )
                if name_s:
                    tr_out["name"] = _apply_template_substitutions(
                        name_s,
                        assignment=assignment,
                        template_map=template_map,
                        shaped_params=shaped,
                        axis_lookup=ax_lookup_dict,
                    )
                transitions_expanded_out.append(tr_out)

            # Synthesize template-uniform IR for this transition when the
            # source-template's wildcard axes (possibly with pinned-coord
            # selectors handled by one-hot masks) admit a uniform
            # representation. The source side installs a ``-flow`` body in
            # every cell of the from-template; the destination side installs
            # a ``Reduce(sum_over, ...)`` (or just the body when there are
            # no from-template-only axes) in every cell of the to-template.
            # Each pinned (axis, coord) selector multiplies the body by a
            # one-hot mask shaped param so that only the pinned slab
            # contributes; the mask zeroes out the non-pinned slabs before
            # the surrounding Reduce sums them away.
            if synthesize:
                # Enumerate ALL cells of both templates (treating pinned
                # axes as wildcards over their full coord ranges) so the
                # template-uniform synthesized IR is installed in every
                # cell. Pinned-coord selection lives inside the body via
                # the one-hot mask multiplications below.
                #
                # ``cells_by_base`` (built once at the top of this function)
                # already holds every concrete cell name grouped by base
                # in the canonical state-template ordering, so the set
                # of cells matching a wildcard-only template is just the
                # bucket for that base. We only fall back to the previous
                # ``render_selector`` enumeration when a template has no
                # axis tokens at all (the lookup would then equal
                # ``[base]`` anyway, but the empty/no-axes case is rare
                # and not on the hot path). Issue #145.
                enum_cache = enumerate_template_cell_names_cache

                def _enumerate_template_cell_names(
                    base: str,
                    template_tokens: tuple[Any, ...],
                    _cache: dict[tuple[str, tuple[Any, ...]], list[str]] = enum_cache,
                ) -> list[str]:
                    if template_tokens:
                        cached = cells_by_base.get(base)
                        if cached is not None:
                            return cached
                        return [base]
                    cache_key = (base, template_tokens)
                    cached_full = _cache.get(cache_key)
                    if cached_full is not None:
                        return cached_full
                    _cache[cache_key] = [base]
                    return [base]

                to_names_for_synthesis = set(
                    _enumerate_template_cell_names(to_base, tuple(to_tokens))
                )
                from_names_for_synthesis = set(
                    _enumerate_template_cell_names(frm_base, tuple(frm_tokens))
                )

                from_subscript = Subscript(
                    name=frm_base,
                    indices=tuple(
                        AxisIndex(axis=tok.axis, coord=None)
                        for tok in frm_tokens
                        if isinstance(tok, (WildcardToken, PinnedToken))
                    ),
                )

                def _mask_sub(tok: PinnedToken) -> Subscript:
                    return Subscript(
                        name=masks[tok.axis, tok.coord],
                        indices=(AxisIndex(axis=tok.axis, coord=None),),
                    )

                # To-side body: rate * from-state * from-pinned masks,
                # all summed over reduce bindings; then multiplied by
                # to-side pinned masks (which use to-cell coords). Rate
                # lives inside the reduce so per-axis terms such as
                # ``theta[imm]`` correctly bind to the reduce loop.
                inner_to: Expr = Apply(op="*", args=(ir_rate_raw, from_subscript))
                for tok in pinned_tokens_from:
                    inner_to = Apply(op="*", args=(inner_to, _mask_sub(tok)))
                synth_to: Expr
                if reduce_bindings_axes:
                    synth_to = Reduce(
                        kind="sum_over",
                        bindings=tuple((ax, ax) for ax in reduce_bindings_axes),
                        body=inner_to,
                    )
                else:
                    synth_to = inner_to
                for tok in pinned_tokens_to:
                    synth_to = Apply(op="*", args=(synth_to, _mask_sub(tok)))

                # From-side body: rate * from-state * masks for FROM-side
                # pinned tokens only (to-side pinned axes may not be in
                # the from-template and would broadcast incorrectly).
                from_body: Expr = Apply(op="*", args=(ir_rate_raw, from_subscript))
                for tok in pinned_tokens_from:
                    from_body = Apply(op="*", args=(from_body, _mask_sub(tok)))
                synth_neg = Apply(op="neg", args=(from_body,))

                all_syms |= free_symbols(synth_to)
                # Pointwise-expand the synthesized IR ONCE at the template
                # level so ``d_ir_full`` (the per-cell equation IR consumed
                # by downstream code via ``ir_to_ast_expr``) stays Reduce-
                # free. With ``lhs_assignment={}``, only the explicit
                # reduce bindings are enumerated; wildcard template axes
                # remain symbolic (``AxisIndex(coord=None)``) so the
                # resulting IR is identical across every cell of the
                # template and can be shared by object identity (issue
                # #145). Without this, every cell of a synthesized
                # transition would need its own per-cell pointwise
                # expansion and the per-cell ``inline_aliases`` would not
                # hit the shared ``result_memo``.
                synth_to_full = expand_reduce_pointwise(
                    synth_to,
                    axes=list(axes),
                    shaped_params=shaped,
                    lhs_assignment={},
                    axis_coords=ax_lookup_dict,
                )
                synth_neg_full = expand_reduce_pointwise(
                    synth_neg,
                    axes=list(axes),
                    shaped_params=shaped,
                    lhs_assignment={},
                    axis_coords=ax_lookup_dict,
                )
                for to_name in to_names_for_synthesis:
                    d_ir_reduce[to_name].append(synth_to)
                    d_ir_full[to_name].append(synth_to_full)
                for from_name in from_names_for_synthesis:
                    d_ir_reduce[from_name].append(synth_neg)
                    d_ir_full[from_name].append(synth_neg_full)
    finally:
        if needed > old_limit:
            sys.setrecursionlimit(old_limit)

    # Cells whose transition participation is identical end up with
    # identical term lists by identity (same shared synth_to /
    # synth_neg / flow IR objects). Dedup the outer Apply by the tuple
    # of term ids so downstream identity-keyed memos (unparse_ir,
    # free_symbols, inline_aliases) see one wrapper instead of one
    # per cell. Issue #145.
    _sum_dedup_full: dict[tuple[int, ...], Expr] = {}
    _sum_dedup_reduce: dict[tuple[int, ...], Expr] = {}

    def _sum_terms(
        terms: list[Expr],
        _dedup: dict[tuple[int, ...], Expr],
    ) -> Expr:
        if not terms:
            return Literal(value=0.0)
        if len(terms) == 1:
            return terms[0]
        key = tuple(id(t) for t in terms)
        cached = _dedup.get(key)
        if cached is not None:
            return cached
        node: Expr = Apply(op="+", args=tuple(terms))
        _dedup[key] = node
        return node

    equations_ir_pre = tuple(
        _sum_terms(d_ir_full[s], _sum_dedup_full) for s in state_expanded
    )
    equations_ir_reduce_pre = tuple(
        _sum_terms(d_ir_reduce[s], _sum_dedup_reduce) for s in state_expanded
    )
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
    # Collect alias symbols from IR. Walk the Reduce-folded map (same
    # inlined symbol set as the full-expansion map but vastly smaller
    # on continuum specs) and share an id-keyed memo across per-cell
    # entries that share subtrees by identity (issue #145).
    all_syms: set[str] = set()
    fs_memo: dict[int, frozenset[str]] = {}
    for alias_expr in aliases_ir_reduce_map.values():
        all_syms |= free_symbols(alias_expr, memo=fs_memo)

    _apply_coord_shifts(
        transitions_raw=transitions_raw,
        state_expanded=state_expanded,
        axes=axes_meta,
        state_template_map=state_template_map,
    )

    if not transitions_raw:
        raise InvalidRhsSpecError(
            detail="transitions must be non-empty after applying chain expansion"
        )

    # Build equations and collect expanded transitions in one IR-native pass
    pinned_mask_names, pinned_mask_values = _discover_pinned_token_masks(
        transitions_raw, axis_lookup=axis_lookup_dict
    )
    if pinned_mask_values:
        # Register one-hot masks as shaped params so the vectorizer's
        # extra-param-buffers plumbing assembles them at eval time; stash
        # the actual values under ``meta`` so ``compile_rhs`` can inject
        # them into the eval_fn's ``params`` automatically.
        for (axis, _coord), mask_name in pinned_mask_names.items():
            shaped_params[mask_name] = (axis,)
        meta["op_system_synth_constants"] = dict(pinned_mask_values)

    equations_ir_pre_inline, equations_ir_reduce, transitions_expanded, rate_syms = (
        _build_transition_equations_ir(
            transitions_raw,
            state_set=state_set,
            state_expanded=state_expanded,
            axes=axes_meta,
            axis_lookup=axis_lookup_dict,
            template_map=template_map_all,
            shaped_params=shaped_params,
            mask_names=pinned_mask_names,
            time_axis_name=time_axis_name,
            alias_bases={
                parse_selector(_normalize_bracket_key(k))[0] for k in aliases_raw_map
            },
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
    # Inline aliases directly into the per-cell IR we already built in
    # ``_build_transition_equations_ir`` rather than re-parsing the
    # round-tripped equation strings. Avoids 73k x ``parse_expr_to_ir``
    # plus a redundant IR rebuild on large continuum specs (issue #145).
    # ``aliases_ir_map`` is already fully alias-inlined inside
    # ``_build_aliases_ir_from_raw`` so cycle detection is redundant here.
    #
    # The dominant cost on large specs is the per-cell ``inline_aliases``
    # call. Because synthesized transitions install the SAME ``synth_to`` /
    # ``synth_neg`` IR object into every cell of a template, the *terms*
    # of the per-cell sum are shared across many cells even though the
    # outer ``Apply(op="+", ...)`` wrapper is unique per cell. Inlining
    # term-by-term with a shared ``result_memo`` keyed on ``id(term)``
    # collapses the alias-substitution work from O(n_state) to
    # O(n_unique_terms) (issue #145).
    alias_inline_memo: dict[int, frozenset[str]] = {}
    alias_inline_result_memo: dict[int, Expr] = {}

    def _inline_one(expr: Expr) -> Expr:
        return inline_aliases(
            expr,
            aliases_ir_map,
            memo=alias_inline_memo,
            skip_cycle_check=True,
            result_memo=alias_inline_result_memo,
        )

    # Dedup the post-inline outer Apply by tuple of arg ids so cells
    # whose inlined-term tuple is identical share one wrapper, letting
    # downstream identity-keyed memos (e.g. unparse_ir) cache the
    # rendered string once per unique equation. Issue #145.
    apply_plus_dedup: dict[tuple[int, ...], Expr] = {}

    equations_ir_built_list: list[Expr | None] = []
    for expr in equations_ir_pre_inline:
        if expr is None:
            equations_ir_built_list.append(None)
            continue
        if not aliases_ir_map:
            equations_ir_built_list.append(expr)
            continue
        try:
            if isinstance(expr, Apply) and expr.op == "+":
                new_args = tuple(_inline_one(a) for a in expr.args)
                if all(n is o for n, o in zip(new_args, expr.args, strict=True)):
                    equations_ir_built_list.append(expr)
                else:
                    dedup_key = tuple(id(a) for a in new_args)
                    cached_apply = apply_plus_dedup.get(dedup_key)
                    if cached_apply is None:
                        cached_apply = Apply(op="+", args=new_args)
                        apply_plus_dedup[dedup_key] = cached_apply
                    equations_ir_built_list.append(cached_apply)
            else:
                equations_ir_built_list.append(_inline_one(expr))
        except (ValueError, RecursionError):
            equations_ir_built_list.append(expr)
    equations_ir_built = tuple(equations_ir_built_list)
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
