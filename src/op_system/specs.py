"""op_system.specs.

RHS specification models and normalization utilities for op_system.

Design goals
------------
- Domain-agnostic core: no imports from flepimop2 or other adapters.
- YAML-friendly RHS specifications that compile into a normalized representation
  consumable by op_engine (and other numerical backends).
- Minimal v1 implementation that demonstrates the idea without blocking future
  multiphysics extensions (IMEX operators, PDE terms, sources, etc.).

Current supported RHS kinds
---------------------------
1) kind: "expr"
   - User provides explicit expressions for d(state)/dt per state variable.

2) kind: "transitions"
   - User provides diagram-style transitions and per-capita hazard expressions.
   - Each transition contributes a flow:
        flow = hazard_expr * from_state
     and updates derivatives:
        d(from)/dt -= flow
        d(to)/dt   += flow

Future-facing (not implemented, but reserved)
---------------------------------------------
- kind: "multiphysics" or additional top-level keys such as:
    - sources: explicit additive per-state terms (births/imports/forcing)
    - operators: implicit operator specs/factories for IMEX (diffusion, transport)
    - couplings: structured couplings across axes (space/age/traits) and fields
  The normalization outputs include placeholders to carry these blocks forward,
  so adapters/backends can extend without changing the fundamental contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from re import Match

from op_system._axes import _normalize_axes, _normalize_bracket_key
from op_system._errors import (
    _raise_invalid_rhs_spec,
    _raise_unsupported_feature,
)
from op_system._helpers import (
    _ensure_mapping,
    _ensure_str_dict,
    _ensure_str_list,
    _sorted_unique,
)
from op_system._symbols import _collect_names, _parse_expr
from op_system._templates import (
    PinnedToken,
    SelectorToken,
    WildcardToken,
    _apply_template_substitutions,
    _extract_placeholders_from_expr,
    _sanitize_fragment,
    build_axis_lookup,
    expand_apply_to,
    expand_selector,
    parse_selector,
    render_selector,
)

# These types are part of the public API of this module (re-exported for callers).
__all__ = ["PinnedToken", "SelectorToken", "WildcardToken"]

_INTEGRATE_OVER_RE = re.compile(
    r"integrate_over\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(.*?)\)",
    re.DOTALL,
)
_SUM_OVER_RE = re.compile(
    r"sum_over\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s+IN\s+\[([^\[\]]*)\])?\s*,\s*(.*?)\)",
    re.DOTALL,
)

_ALLOWED_KERNEL_FORMS: dict[str, tuple[str, ...]] = {
    "erfc": ("scale", "sigma"),
    "gaussian": ("scale", "sigma"),
    "exponential": ("scale", "lambda"),
    "gamma": ("scale", "k", "theta"),
    "power_law": ("scale", "sigma", "p"),
    "custom_value": (),
}


# -----------------------------------------------------------------------------
# Normalized RHS representation (backend-facing)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalizedRhs:
    """Normalized RHS representation suitable for compilation/execution."""

    kind: str
    state_names: tuple[str, ...]
    equations: tuple[str, ...]
    aliases: Mapping[str, str]
    param_names: tuple[str, ...]
    all_symbols: frozenset[str]
    meta: Mapping[str, Any]


# (Constraint normalization is in _constraints.py and imported above.)


def _normalize_state_axes(
    raw_state_axes: object,
    *,
    axis_names: set[str],
    state_set: set[str],
) -> dict[str, tuple[str, ...]]:
    """Normalize mapping of state -> list of axis names.

    Returns:
        Mapping of state names to tuples of axis names.
    """
    if raw_state_axes is None:
        return {}
    mapping = _ensure_mapping(raw_state_axes, name="state_axes")
    out: dict[str, tuple[str, ...]] = {}
    for state, axes in mapping.items():
        if state not in state_set:
            _raise_invalid_rhs_spec(detail=f"state_axes key {state!r} not in state")
        if not isinstance(axes, (list, tuple)) or not axes:
            _raise_invalid_rhs_spec(
                detail=f"state_axes[{state!r}] must be a non-empty list of axis names"
            )
        resolved: list[str] = []
        seen: set[str] = set()
        for i, ax in enumerate(axes):
            if not isinstance(ax, str) or not ax.strip():
                _raise_invalid_rhs_spec(
                    detail=f"state_axes[{state!r}][{i}] must be a non-empty string"
                )
            ax_name = ax.strip()
            if ax_name not in axis_names:
                _raise_invalid_rhs_spec(
                    detail=f"state_axes[{state!r}] references unknown axis {ax_name!r}"
                )
            if ax_name in seen:
                _raise_invalid_rhs_spec(
                    detail=f"state_axes[{state!r}] contains duplicate axis {ax_name!r}"
                )
            seen.add(ax_name)
            resolved.append(ax_name)
        out[state] = tuple(resolved)
    return out


def _normalize_kernel_axes_field(
    axes_field: object, *, idx: int, axis_names: set[str]
) -> tuple[str, ...] | None:
    if axes_field is None:
        return None
    if not isinstance(axes_field, (list, tuple)) or not axes_field:
        _raise_invalid_rhs_spec(
            detail=f"kernels[{idx}].axes must be a non-empty list if provided"
        )
    resolved_axes: list[str] = []
    for ax_idx, ax in enumerate(axes_field):
        if not isinstance(ax, str) or not ax.strip():
            _raise_invalid_rhs_spec(
                detail=f"kernels[{idx}].axes[{ax_idx}] must be a non-empty string"
            )
        ax_name = ax.strip()
        if ax_name not in axis_names:
            _raise_invalid_rhs_spec(
                detail=f"kernels[{idx}] references unknown axis {ax_name!r}"
            )
        resolved_axes.append(ax_name)
    return tuple(resolved_axes)


def _normalize_kernel_form_and_params(
    *, value: object, form: object, params_field: object, idx: int
) -> tuple[str, Mapping[str, Any] | None]:
    if value is None:
        if not isinstance(form, str) or not form.strip():
            _raise_invalid_rhs_spec(
                detail=(f"kernels[{idx}].form is required when value is not provided")
            )
        form_name = form.strip().lower()
        if form_name not in _ALLOWED_KERNEL_FORMS:
            _raise_invalid_rhs_spec(
                detail=f"kernels[{idx}] form {form_name!r} is not supported"
            )
        required_keys = _ALLOWED_KERNEL_FORMS[form_name]
        params_map_required = _ensure_mapping(
            params_field, name=f"kernels[{idx}].params"
        )
        for req in required_keys:
            if req not in params_map_required:
                _raise_invalid_rhs_spec(
                    detail=f"kernels[{idx}].params missing required key {req!r}"
                )
        return form_name, params_map_required

    form_name = (
        str(form).strip().lower()
        if isinstance(form, str) and form.strip()
        else "custom_value"
    )
    if form_name and form_name not in _ALLOWED_KERNEL_FORMS:
        _raise_invalid_rhs_spec(
            detail=f"kernels[{idx}] form {form_name!r} is not supported"
        )
    params_map_optional: Mapping[str, Any] | None
    if params_field is not None:
        params_map_optional = _ensure_mapping(
            params_field, name=f"kernels[{idx}].params"
        )
    else:
        params_map_optional = None
    return form_name, params_map_optional


def _normalize_single_kernel(
    mk_map: Mapping[str, Any],
    *,
    idx: int,
    axis_names: set[str],
    seen: set[str],
) -> dict[str, Any]:
    name = _get_required_str(mk_map, idx=idx, key="name")
    if name in seen:
        _raise_invalid_rhs_spec(detail=f"duplicate kernel name: {name!r}")
    seen.add(name)

    axes_resolved = _normalize_kernel_axes_field(
        mk_map.get("axes"), idx=idx, axis_names=axis_names
    )
    form_name, params_map = _normalize_kernel_form_and_params(
        value=mk_map.get("value"),
        form=mk_map.get("form"),
        params_field=mk_map.get("params"),
        idx=idx,
    )

    mk_out: dict[str, Any] = {
        "name": name,
        "kind": str(mk_map.get("kind", "analytic")).strip().lower(),
        "form": form_name,
    }
    if axes_resolved is not None:
        mk_out["axes"] = axes_resolved
    if params_map is not None:
        mk_out["params"] = params_map
    if mk_map.get("value") is not None:
        mk_out["value"] = mk_map.get("value")
    return mk_out


def _normalize_kernels(
    raw_kernels: object,
    *,
    axis_names: set[str],
) -> list[dict[str, Any]]:
    """Normalize kernel metadata.

    Returns:
        List of normalized kernel definitions.
    """
    if raw_kernels is None:
        return []
    if not isinstance(raw_kernels, list):
        _raise_invalid_rhs_spec(detail="kernels must be a list")

    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for idx, mk in enumerate(raw_kernels):
        mk_map = _ensure_mapping(mk, name=f"kernels[{idx}]")
        mk_out = _normalize_single_kernel(
            mk_map, idx=idx, axis_names=axis_names, seen=seen
        )
        out.append(mk_out)

    return out


def _validate_op_apply_to(
    apply_to_raw: object,
    idx: int,
    state_set: set[str] | None,
    *,
    axes: list[dict[str, Any]] | None = None,
) -> list[str]:
    if not isinstance(apply_to_raw, (list, tuple)) or not apply_to_raw:
        _raise_invalid_rhs_spec(
            detail=(
                f"operators[{idx}].apply_to must be a non-empty list "
                "of state names if provided"
            )
        )
    if axes:
        axis_lookup = build_axis_lookup(axes)
        result = expand_apply_to(
            list(apply_to_raw),
            axis_lookup=axis_lookup,
            state_set=state_set,
            context=f"operators[{idx}].apply_to",
        )
    else:
        result = []
        for j, state_name in enumerate(apply_to_raw):
            if not isinstance(state_name, str) or not state_name.strip():
                _raise_invalid_rhs_spec(
                    detail=f"operators[{idx}].apply_to[{j}] must be a non-empty string"
                )
            state_name_s = state_name.strip()
            if state_set is not None and state_name_s not in state_set:
                _raise_invalid_rhs_spec(
                    detail=(
                        f"operators[{idx}].apply_to[{j}]={state_name_s!r} not in state"
                    )
                )
            result.append(state_name_s)
    return result


def _validate_scalar_or_expr(value: object, field: str) -> None:
    """Raise if *value* is not a non-empty string or a finite number."""
    if isinstance(value, str):
        if not value.strip():
            _raise_invalid_rhs_spec(
                detail=f"{field} must be a non-empty string or number"
            )
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        _raise_invalid_rhs_spec(detail=f"{field} must be a non-empty string or number")


def _validate_op_advection(op_map: Mapping[str, Any], idx: int, kind_s: str) -> None:
    velocity_val = op_map.get("velocity")
    if velocity_val is None:
        _raise_invalid_rhs_spec(
            detail=f"operators[{idx}].velocity is required for {kind_s!r}"
        )
    _validate_scalar_or_expr(velocity_val, f"operators[{idx}].velocity")


def _validate_op_jump_integral(op_map: Mapping[str, Any], idx: int) -> None:
    rate_val = op_map.get("rate")
    if rate_val is None:
        _raise_invalid_rhs_spec(
            detail=f"operators[{idx}].rate is required for 'jump_integral'"
        )
    _validate_scalar_or_expr(rate_val, f"operators[{idx}].rate")

    kernel_val = op_map.get("kernel")
    if not isinstance(kernel_val, dict):
        _raise_invalid_rhs_spec(
            detail=f"operators[{idx}].kernel must be a mapping for 'jump_integral'"
        )
    kernel_form = kernel_val.get("form")
    if not isinstance(kernel_form, str) or not kernel_form.strip():
        _raise_invalid_rhs_spec(
            detail=(
                f"operators[{idx}].kernel.form must be a non-empty "
                "string for 'jump_integral'"
            )
        )
    kernel_params = kernel_val.get("params")
    if kernel_params is not None and not isinstance(kernel_params, dict):
        _raise_invalid_rhs_spec(
            detail=(
                f"operators[{idx}].kernel.params must be a mapping if "
                "provided for 'jump_integral'"
            )
        )

    direction_val = op_map.get("direction")
    if direction_val is not None:
        if not isinstance(direction_val, str) or not direction_val.strip():
            _raise_invalid_rhs_spec(
                detail=(
                    f"operators[{idx}].direction must be one of 'up', 'down', or 'both'"
                )
            )
        if direction_val.strip().lower() not in {"up", "down", "both"}:
            _raise_invalid_rhs_spec(
                detail=(
                    f"operators[{idx}].direction must be one of 'up', 'down', or 'both'"
                )
            )


def _validate_op_header(
    op_map: Mapping[str, Any],
    idx: int,
    axis_names: set[str],
) -> tuple[str | None, str, str, str | None]:
    """Validate and strip the simple header fields of one operator entry.

    Returns:
        Tuple of (op_name_s, axis_name_s, kind_s, bc_val_s).
    """
    op_name = op_map.get("name")
    if op_name is not None and (not isinstance(op_name, str) or not op_name.strip()):
        _raise_invalid_rhs_spec(
            detail=f"operators[{idx}].name must be a non-empty string if provided"
        )

    axis_name = op_map.get("axis")
    if not isinstance(axis_name, str) or not axis_name.strip():
        _raise_invalid_rhs_spec(
            detail=f"operators[{idx}].axis must be a non-empty string"
        )
    axis_name_s = axis_name.strip()
    if axis_name_s not in axis_names:
        _raise_invalid_rhs_spec(
            detail=f"operators[{idx}] references unknown axis {axis_name_s!r}"
        )

    kind_val = op_map.get("kind")
    if not isinstance(kind_val, str) or not kind_val.strip():
        _raise_invalid_rhs_spec(
            detail=(
                f"operators[{idx}].kind must be a non-empty string "
                "(e.g., advection/jump_integral)"
            )
        )
    kind_s = kind_val.strip().lower()

    bc_val = op_map.get("bc")
    if bc_val is not None and (not isinstance(bc_val, str) or not bc_val.strip()):
        _raise_invalid_rhs_spec(
            detail=f"operators[{idx}].bc must be a non-empty string if provided"
        )

    op_name_s: str | None = op_name.strip() if isinstance(op_name, str) else None
    bc_val_s: str | None = bc_val.strip() if isinstance(bc_val, str) else None
    return op_name_s, axis_name_s, kind_s, bc_val_s


def _enrich_op_kind_fields(op_out: dict[str, Any], kind_s: str) -> None:
    """Normalize kind-specific fields in *op_out* in-place."""
    if kind_s in {"advection", "transport"}:
        velocity_val = op_out["velocity"]
        op_out["velocity"] = (
            velocity_val.strip()
            if isinstance(velocity_val, str)
            else float(velocity_val)
        )
    elif kind_s == "jump_integral":
        rate_val = op_out["rate"]
        op_out["rate"] = (
            rate_val.strip() if isinstance(rate_val, str) else float(rate_val)
        )
        kernel_val = dict(op_out["kernel"])
        kernel_val["form"] = str(kernel_val["form"]).strip()
        op_out["kernel"] = kernel_val
        if "direction" in op_out and isinstance(op_out["direction"], str):
            op_out["direction"] = op_out["direction"].strip().lower()


def _normalize_single_operator(
    op: object,
    idx: int,
    *,
    axis_names: set[str],
    state_set: set[str] | None,
    axes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    op_map = _ensure_mapping(op, name=f"operators[{idx}]")
    op_name_s, axis_name_s, kind_s, bc_val_s = _validate_op_header(
        op_map, idx, axis_names
    )

    apply_to_raw = op_map.get("apply_to")
    apply_to_clean: list[str] | None = (
        _validate_op_apply_to(apply_to_raw, idx, state_set, axes=axes)
        if apply_to_raw is not None
        else None
    )

    if kind_s in {"advection", "transport"}:
        _validate_op_advection(op_map, idx, kind_s)
    elif kind_s == "jump_integral":
        _validate_op_jump_integral(op_map, idx)

    op_out = dict(op_map)
    if op_name_s is not None:
        op_out["name"] = op_name_s
    op_out["axis"] = axis_name_s
    op_out["kind"] = kind_s
    if bc_val_s is not None:
        op_out["bc"] = bc_val_s
    if apply_to_clean is not None:
        op_out["apply_to"] = apply_to_clean
    _enrich_op_kind_fields(op_out, kind_s)
    return op_out


def _normalize_operators(
    raw_ops: object,
    *,
    axis_names: set[str],
    state_set: set[str] | None = None,
    axes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize operator metadata and validate axis references.

    Returns:
        List of normalized operator definitions.
    """
    if raw_ops is None:
        return []
    if not isinstance(raw_ops, list):
        _raise_invalid_rhs_spec(detail="operators must be a list")

    return [
        _normalize_single_operator(
            op, idx, axis_names=axis_names, state_set=state_set, axes=axes
        )
        for idx, op in enumerate(raw_ops)
    ]


def _get_meta_field(spec: Mapping[str, Any], key: str) -> object:
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


def _collect_alias_symbols(
    aliases: Mapping[str, str], *, axes: list[dict[str, Any]]
) -> set[str]:
    symbols: set[str] = set()
    for expr in aliases.values():
        expr_s = _expand_helpers(expr, axes=axes)
        tree = _parse_expr(expr_s)
        symbols |= _collect_names(tree)
    return symbols


def _expand_state_templates(
    state_raw: list[str],
    *,
    axes: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[tuple[str, dict[str, str]]]]]:
    """Expand state templates with categorical axes into concrete names.

    Supports wildcard tokens (``axis``), pinned tokens (``axis=coord``), and
    mixed selectors.  Wildcard-only selectors also create a template_map entry
    keyed by ``"base[ax1,ax2]"`` for use in subsequent expression substitution.

    Returns:
        A tuple of (expanded_state_names, template_map).
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
            # Bare name — no expansion, no template_map entry.
            expanded.append(base)
        else:
            wildcards = [t for t in tokens if isinstance(t, WildcardToken)]
            if wildcards:
                # Canonical template key: only wildcard axes (for expr substitution).
                template_key = f"{base}[{','.join(wt.axis for wt in wildcards)}]"
            else:
                # All-pinned selector: full key (no expression substitution expected).
                pinned = (t for t in tokens if isinstance(t, PinnedToken))
                template_key = (
                    f"{base}[{','.join(f'{t.axis}={t.coord}' for t in pinned)}]"
                )
            template_map[template_key] = results
            for name, _ in results:
                expanded.append(name)

    return expanded, template_map


def _expand_alias_templates(
    aliases_raw: Mapping[str, str],
    *,
    axes: list[dict[str, Any]],
    template_map_seed: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> tuple[dict[str, str], dict[str, list[tuple[str, dict[str, str]]]]]:
    """Expand templated alias names and substitute inline placeholders.

    Returns:
        A tuple of (expanded_aliases, alias_template_map).
    """
    alias_names = list(aliases_raw.keys())
    alias_expanded, alias_template_map = _expand_state_templates(alias_names, axes=axes)
    if len(alias_expanded) != len(set(alias_expanded)):
        _raise_invalid_rhs_spec(detail="expanded aliases contain duplicates")

    combined_template_map = {**template_map_seed, **alias_template_map}
    aliases_out: dict[str, str] = {}

    for raw_name, expr in aliases_raw.items():
        # Normalize the key so spaces inside brackets don't prevent lookup.
        canonical_name = _normalize_bracket_key(raw_name)
        expr_s = expr.strip()
        if not expr_s:
            _raise_invalid_rhs_spec(
                detail=f"aliases[{raw_name!r}] must be a non-empty string"
            )
        expr_s = _expand_helpers(expr_s, axes=axes)

        if canonical_name in alias_template_map:
            for expanded_name, assignment in alias_template_map[canonical_name]:
                substituted = _apply_template_substitutions(
                    expr_s,
                    assignment=assignment,
                    template_map=combined_template_map,
                )
                aliases_out[expanded_name] = substituted
        else:
            aliases_out[raw_name] = expr_s

    return aliases_out, alias_template_map


def _expand_initial_state_templates(
    initial_state_raw: Mapping[str, str] | None,
    *,
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> dict[str, str] | None:
    """Expand a templated initial_state mapping into concrete state→param pairs.

    Supports wildcard selectors (``X[age, vax]``), pinned selectors
    (``X[age, vax, imm=X0]``), and bare state names.

    Returns:
        Expanded ``dict[str, str]`` mapping each concrete state name to its
        concrete parameter name, or ``None`` if *initial_state_raw* is ``None``.
    """
    if initial_state_raw is None:
        return None

    axis_lookup = build_axis_lookup(axes)
    result: dict[str, str] = {}
    expanded_keys: list[str] = []

    for raw_key, raw_val in initial_state_raw.items():
        val_s = str(raw_val).strip()
        if not val_s:
            _raise_invalid_rhs_spec(
                detail=f"initial_state[{raw_key!r}] must be a non-empty string",
            )
        results = expand_selector(
            raw_key,
            axis_lookup=axis_lookup,
            context=f"initial_state key {raw_key!r}",
        )
        for expanded_key, assignment in results:
            expanded_keys.append(expanded_key)
            result[expanded_key] = _apply_template_substitutions(
                val_s,
                assignment=assignment,
                template_map=template_map,
            )

    if len(expanded_keys) != len(set(expanded_keys)):
        _raise_invalid_rhs_spec(detail="expanded initial_state keys contain duplicates")

    return result


def _maybe_attach_initial_state(
    meta: dict[str, Any],
    initial_state_raw: Mapping[str, str] | None,
    *,
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> None:
    """Expand *initial_state_raw* and attach it to *meta* when present."""
    expanded = _expand_initial_state_templates(
        initial_state_raw,
        axes=axes,
        template_map=template_map,
    )
    if expanded is not None:
        meta["initial_state"] = expanded


# (Template primitives _extract_placeholders_from_expr, _render_template_or_literal,
#  _apply_template_substitutions, _render_template_name, _parse_template_key,
#  _parse_transition_endpoint_tokens, _render_transition_endpoint, and
#  _build_chain_stage_names are now in _templates.py and imported above.)


@dataclass(frozen=True)
class _TransitionEndpoints:
    """Parsed endpoint data and expansion context for a single transition template."""

    frm_base: str
    frm_tokens: list[SelectorToken]
    to_base: str
    to_tokens: list[SelectorToken]
    rate_s: str
    name_s: str | None
    axes: list[dict[str, Any]]
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]]
    axis_lookup: dict[str, list[str]]


def _collect_transition_wildcard_axes(
    endpoints: _TransitionEndpoints,
) -> list[str]:
    """Return ordered list of unique wildcard axes from endpoints and expressions."""
    wildcard_axes: list[str] = []
    seen: set[str] = set()
    for tok in endpoints.frm_tokens + endpoints.to_tokens:
        if isinstance(tok, WildcardToken) and tok.axis not in seen:
            wildcard_axes.append(tok.axis)
            seen.add(tok.axis)
    expr_placeholders = _extract_placeholders_from_expr(endpoints.rate_s)
    if endpoints.name_s:
        expr_placeholders |= _extract_placeholders_from_expr(endpoints.name_s)
    for ph in sorted(expr_placeholders):
        if ph not in seen:
            if ph not in endpoints.axis_lookup:
                _raise_invalid_rhs_spec(
                    detail=f"transition placeholder {ph!r} references unknown axis"
                )
            wildcard_axes.append(ph)
            seen.add(ph)
    return wildcard_axes


def _render_transition_combo(
    tr_base: dict[str, Any],
    endpoints: _TransitionEndpoints,
    *,
    assignment: dict[str, str],
) -> dict[str, Any]:
    """Build one expanded transition dict for a given axis assignment.

    Returns:
        Expanded transition mapping with concrete from/to/rate fields.
    """
    rate_sub = _apply_template_substitutions(
        endpoints.rate_s, assignment=assignment, template_map=endpoints.template_map
    )
    tr_out: dict[str, Any] = dict(tr_base)
    tr_out["from"] = render_selector(
        endpoints.frm_base,
        endpoints.frm_tokens,
        assignment,
        axis_lookup=endpoints.axis_lookup,
    )
    tr_out["to"] = render_selector(
        endpoints.to_base,
        endpoints.to_tokens,
        assignment,
        axis_lookup=endpoints.axis_lookup,
    )
    tr_out["rate"] = _expand_helpers(rate_sub, axes=endpoints.axes)
    if endpoints.name_s:
        tr_out["name"] = _apply_template_substitutions(
            endpoints.name_s, assignment=assignment, template_map=endpoints.template_map
        )
    return tr_out


def _expand_single_transition(
    tr_map: Mapping[str, Any],
    *,
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
    axis_lookup: dict[str, list[str]],
) -> list[dict[str, Any]]:
    tr_valid = _validate_transition_mapping(tr_map, idx=0)
    frm_s = _get_required_str(tr_valid, idx=0, key="from")
    to_s = _get_required_str(tr_valid, idx=0, key="to")
    rate_s = _get_required_str(tr_valid, idx=0, key="rate")
    name_s = tr_valid.get("name") if isinstance(tr_valid.get("name"), str) else None

    frm_base, frm_tokens = parse_selector(frm_s)
    to_base, to_tokens = parse_selector(to_s)
    endpoints = _TransitionEndpoints(
        frm_base=frm_base,
        frm_tokens=frm_tokens,
        to_base=to_base,
        to_tokens=to_tokens,
        rate_s=rate_s,
        name_s=name_s,
        axes=axes,
        template_map=template_map,
        axis_lookup=axis_lookup,
    )
    wildcard_axes = _collect_transition_wildcard_axes(endpoints)

    if not wildcard_axes:
        return [_render_transition_combo(dict(tr_valid), endpoints, assignment={})]

    coords_lists: list[list[str]] = []
    for ph in wildcard_axes:
        if ph not in axis_lookup:
            _raise_invalid_rhs_spec(
                detail=f"transition placeholder {ph!r} references unknown axis"
            )
        coords_lists.append(axis_lookup[ph])

    return [
        _render_transition_combo(
            dict(tr_valid),
            endpoints,
            assignment=dict(zip(wildcard_axes, combo, strict=True)),
        )
        for combo in product(*coords_lists)
    ]


def _expand_transition_templates(
    transitions_raw: list[Mapping[str, Any]],
    *,
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> list[dict[str, Any]]:
    """Expand templated transitions over categorical axes.

    Supports placeholders in from/to/name/rate fields (e.g., S[vacc,age]).

    Returns:
        List of expanded transition mappings with concrete names and rates.
    """
    axis_lookup: dict[str, list[str]] = {
        ax["name"]: [str(c) for c in ax.get("coords", [])] for ax in axes
    }

    expanded: list[dict[str, Any]] = []

    for tr_map in transitions_raw:
        expanded.extend(
            _expand_single_transition(
                tr_map,
                axes=axes,
                template_map=template_map,
                axis_lookup=axis_lookup,
            )
        )

    return expanded


def _resolve_template_equation(
    *,
    name: str,
    equations_map: Mapping[str, Any],
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
    all_syms: set[str],
) -> str | None:
    """Resolve an equation for a templated state name.

    Returns:
        Expanded expression string if found, otherwise None.
    """
    for template_key, variants in template_map.items():
        for expanded_name, assignment in variants:
            if expanded_name != name or template_key not in equations_map:
                continue
            expr = equations_map[template_key]
            if not isinstance(expr, str) or not expr.strip():
                _raise_invalid_rhs_spec(
                    detail=(f"equations[{template_key!r}] must be a non-empty string")
                )
            expr_s = _expand_helpers(expr.strip(), axes=axes)
            expr_s = _apply_template_substitutions(
                expr_s, assignment=assignment, template_map=template_map
            )
            tree = _parse_expr(expr_s)
            all_syms |= _collect_names(tree)
            return expr_s
    return None


def _expand_integrate_over(expr: str, *, axes: list[dict[str, Any]]) -> str:
    """Expand integrate_over(axis=var, inner_expr) for continuous axes.

    Uses trapezoidal weights derived from axis coords; rejects categorical axes.

    Returns:
        Expression string with integrate_over expanded to weighted sums.
    """

    def _axis_coords_and_deltas(ax_name: str) -> tuple[list[str], list[float]]:
        for ax in axes:
            if ax.get("name") == ax_name:
                if ax.get("type") != "continuous":
                    _raise_invalid_rhs_spec(
                        detail=f"integrate_over axis {ax_name!r} must be continuous"
                    )
                coords = [str(c) for c in ax.get("coords", [])]
                deltas = ax.get("deltas") or []
                if not coords or not deltas or len(coords) != len(deltas):
                    _raise_invalid_rhs_spec(
                        detail=f"integrate_over axis {ax_name!r} missing coords/deltas"
                    )
                return coords, [float(d) for d in deltas]
        return _raise_invalid_rhs_spec(
            detail=f"integrate_over axis {ax_name!r} not found"
        )

    out = expr
    while True:
        m = _INTEGRATE_OVER_RE.search(out)
        if not m:
            break
        axis_name = m.group(1)
        var_name = m.group(2)
        inner = m.group(3)
        coords, deltas = _axis_coords_and_deltas(axis_name)
        terms: list[str] = []
        for coord, delta in zip(coords, deltas, strict=True):
            replaced = re.sub(
                rf"\b{re.escape(var_name)}\b",
                str(coord),
                inner,
            )

            def _template_replacer(
                match: Match[str], *, axis: str = axis_name, coord_val: str = str(coord)
            ) -> str:
                return f"{match.group(1)}__{axis}_{_sanitize_fragment(coord_val)}"

            replaced = re.sub(
                rf"([A-Za-z_][A-Za-z0-9_]*)\[\s*{re.escape(axis_name)}\s*=\s*{re.escape(str(coord))}\s*\]",
                _template_replacer,
                replaced,
            )
            terms.append(f"({delta})*({replaced})")
        replacement = " + ".join(terms)
        out = out[: m.start()] + f"({replacement})" + out[m.end() :]
    return out


def _apply_coord_filter(
    filter_str: str | None,
    *,
    axis_name: str,
    all_coords: list[str],
) -> list[str]:
    """Return the coord subset to iterate over, or all coords if no filter.

    Args:
        filter_str: Raw ``IN [...]`` content (e.g. ``"v, w"``), or ``None``.
        axis_name: Axis name — used in error messages only.
        all_coords: All valid coords for the axis.

    Returns:
        Filtered coord list (preserves order of the filter, not the axis).
    """
    if filter_str is None:
        return all_coords
    requested = [c.strip() for c in filter_str.split(",") if c.strip()]
    if not requested:
        _raise_invalid_rhs_spec(
            detail=f"sum_over IN filter for axis {axis_name!r} is empty",
        )
    seen: set[str] = set()
    for coord in requested:
        if coord in seen:
            _raise_invalid_rhs_spec(
                detail=(
                    f"sum_over IN filter for axis {axis_name!r} "
                    f"contains duplicate coord {coord!r}"
                ),
            )
        seen.add(coord)
        if coord not in all_coords:
            _raise_invalid_rhs_spec(
                detail=(
                    f"sum_over IN filter references unknown coord {coord!r} "
                    f"for axis {axis_name!r} (valid: {all_coords})"
                ),
            )
    return requested


def _expand_sum_over(expr: str, *, axes: list[dict[str, Any]]) -> str:
    """Expand sum_over(axis=var, inner_expr) for categorical axes.

    This is a simple string-level unroller; it does not support nested
    sum_over inside the inner_expr beyond repeated passes. Continuous axes
    are rejected.

    Returns:
        Expression string with sum_over expanded to explicit sums.
    """

    def _axis_coords(ax_name: str) -> list[str]:
        for ax in axes:
            if ax.get("name") == ax_name:
                if ax.get("type") != "categorical":
                    _raise_invalid_rhs_spec(
                        detail=f"sum_over axis {ax_name!r} must be categorical"
                    )
                coords = ax.get("coords", [])
                if not coords:
                    _raise_invalid_rhs_spec(
                        detail=f"sum_over axis {ax_name!r} has no coords"
                    )
                return [str(c) for c in coords]
        return _raise_invalid_rhs_spec(detail=f"sum_over axis {ax_name!r} not found")

    out = expr
    # Iterate until no more matches to handle multiple sum_over occurrences.
    while True:
        m = _SUM_OVER_RE.search(out)
        if not m:
            break
        axis_name = m.group(1)
        var_name = m.group(2)
        inner = m.group(4)
        coords = _apply_coord_filter(
            m.group(3),
            axis_name=axis_name,
            all_coords=_axis_coords(axis_name),
        )
        terms: list[str] = []
        for coord in coords:
            # Replace var_name occurrences with coord (as identifier-safe string)
            replaced = re.sub(
                rf"\b{re.escape(var_name)}\b",
                str(coord),
                inner,
            )
            coord_s = str(coord)

            def _template_replacer(
                match: Match[str], *, axis: str = axis_name, coord: str = coord_s
            ) -> str:
                return f"{match.group(1)}__{axis}_{_sanitize_fragment(coord)}"

            replaced = re.sub(
                rf"([A-Za-z_][A-Za-z0-9_]*)\[\s*{re.escape(axis_name)}\s*=\s*{re.escape(coord_s)}\s*\]",
                _template_replacer,
                replaced,
            )
            terms.append(f"({replaced})")
        replacement = " + ".join(terms)
        out = out[: m.start()] + f"({replacement})" + out[m.end() :]
    return out


def _expand_helpers(expr: str, *, axes: list[dict[str, Any]]) -> str:
    """Expand helper calls (integrate_over, sum_over) before AST parse.

    Returns:
        Expression string with helper calls expanded.
    """
    return _expand_sum_over(_expand_integrate_over(expr, axes=axes), axes=axes)


def _gather_equations(
    state: list[str],
    equations_map: Mapping[str, Any],
    all_syms: set[str],
    *,
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]] | None = None,
) -> list[str]:
    eqs: list[str] = []
    template_map = template_map or {}
    for name in state:
        if name in equations_map:
            expr = equations_map[name]
            if not isinstance(expr, str) or not expr.strip():
                _raise_invalid_rhs_spec(
                    detail=f"equations[{name!r}] must be a non-empty string"
                )
            expr_s = _expand_helpers(expr.strip(), axes=axes)
            tree = _parse_expr(expr_s)
            all_syms |= _collect_names(tree)
            eqs.append(expr_s)
            continue

        expr_res = _resolve_template_equation(
            name=name,
            equations_map=equations_map,
            axes=axes,
            template_map=template_map,
            all_syms=all_syms,
        )
        if expr_res is None:
            _raise_invalid_rhs_spec(detail=f"Missing equation for state {name!r}")
        eqs.append(expr_res)
    return eqs


def _chain_rate_expr(value: object, *, field: str) -> str:
    """Normalize a chain rate value into an expression string.

    Returns:
        A non-empty expression string.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(float(value))
    return _raise_invalid_rhs_spec(
        detail=f"{field} must be a non-empty string or number"
    )


def _normalize_chain_forward_rates(
    forward_raw: object,
    *,
    idx: int,
    length: int,
) -> list[str]:
    """Normalize chain forward rates into per-edge expressions.

    Returns:
        A list of length ``length - 1``.
    """
    if isinstance(forward_raw, (str, int, float)) and not isinstance(forward_raw, bool):
        expr = _chain_rate_expr(forward_raw, field=f"chain[{idx}].forward")
        return [expr] * (length - 1)

    if not isinstance(forward_raw, (list, tuple)):
        _raise_invalid_rhs_spec(
            detail=(
                f"chain[{idx}].forward must be a string/number or a list of "
                f"{length - 1} rates"
            )
        )

    rates = [
        _chain_rate_expr(v, field=f"chain[{idx}].forward[{i}]")
        for i, v in enumerate(forward_raw)
    ]
    if len(rates) != length - 1:
        _raise_invalid_rhs_spec(
            detail=(
                f"chain[{idx}].forward list length must be {length - 1} "
                f"for chain length {length}"
            )
        )
    return rates


def _build_chain_stage_names(cname: str, *, length: int) -> list[str]:
    """Build chain stage names, preserving template placeholders when present.

    Returns:
        Ordered list of stage names for the chain.
    """
    base, tokens = parse_selector(cname)
    if tokens:
        suffix = (
            "["
            + ",".join(
                (f"{t.axis}={t.coord}" if isinstance(t, PinnedToken) else t.axis)
                for t in tokens
            )
            + "]"
        )
        return [f"{base}{i}{suffix}" for i in range(1, length + 1)]
    return [f"{cname}{i}" for i in range(1, length + 1)]


def _normalize_chain_entry(
    chain: Mapping[str, Any], *, idx: int
) -> tuple[str, str] | None:
    """Normalize optional chain entry block.

    Returns:
        Tuple ``(from_state, rate_expr)`` or ``None`` if no entry provided.
    """
    entry_raw = chain.get("entry")
    if entry_raw is None:
        return None
    entry_map = _ensure_mapping(entry_raw, name=f"chain[{idx}].entry")
    frm = entry_map.get("from")
    if not isinstance(frm, str) or not frm.strip():
        _raise_invalid_rhs_spec(
            detail=f"chain[{idx}].entry.from must be a non-empty string"
        )
    rate = _chain_rate_expr(entry_map.get("rate"), field=f"chain[{idx}].entry.rate")
    return frm.strip(), rate


def _normalize_chain_exit(
    chain: Mapping[str, Any],
    *,
    idx: int,
) -> tuple[str, str | None] | None:
    """Normalize optional chain exit configuration.

    Accepts either ``exit: {to, rate?}`` or legacy ``to``.

    Returns:
        Tuple ``(to_state, rate_expr_or_none)`` or ``None``.
    """
    exit_raw = chain.get("exit")
    if exit_raw is not None:
        exit_map = _ensure_mapping(exit_raw, name=f"chain[{idx}].exit")
        to_raw = exit_map.get("to")
        if not isinstance(to_raw, str) or not to_raw.strip():
            _raise_invalid_rhs_spec(
                detail=f"chain[{idx}].exit.to must be a non-empty string"
            )
        rate_raw = exit_map.get("rate")
        rate_expr = (
            _chain_rate_expr(rate_raw, field=f"chain[{idx}].exit.rate")
            if rate_raw is not None
            else None
        )
        return to_raw.strip(), rate_expr

    to_legacy = chain.get("to")
    if to_legacy is None:
        return None
    if not isinstance(to_legacy, str) or not to_legacy.strip():
        _raise_invalid_rhs_spec(detail=f"chain[{idx}].to must be a non-empty string")
    return to_legacy.strip(), None


def _validate_chain_entry(
    *,
    chain: Mapping[str, Any],
    idx: int,
    state_set: set[str],
) -> tuple[list[str], list[str], tuple[str, str] | None, tuple[str, str | None] | None]:
    """Validate a chain entry and return normalized chain configuration.

    Returns:
        (stage_names, forward_rates, entry, exit_cfg).
    """
    if not isinstance(chain, dict):
        _raise_invalid_rhs_spec(detail=f"chain[{idx}] must be a mapping")
    cname = _get_required_str(chain, idx=idx, key="name")
    length_obj = chain.get("length")
    if not isinstance(length_obj, (int, float)) or isinstance(length_obj, bool):
        _raise_invalid_rhs_spec(detail=f"chain[{idx}].length must be an integer >= 2")
    clen = int(length_obj)
    if clen < 2:
        _raise_invalid_rhs_spec(detail=f"chain[{idx}].length must be >= 2")

    forward_rates = _normalize_chain_forward_rates(
        chain.get("forward"),
        idx=idx,
        length=clen,
    )

    stage_names = _build_chain_stage_names(cname, length=clen)

    entry_cfg = _normalize_chain_entry(chain, idx=idx)
    if (
        entry_cfg is not None
        and not parse_selector(entry_cfg[0])[1]
        and entry_cfg[0] not in state_set
    ):
        _raise_invalid_rhs_spec(
            detail=f"chain[{idx}].entry.from={entry_cfg[0]!r} not in state"
        )

    exit_cfg = _normalize_chain_exit(chain, idx=idx)
    if (
        exit_cfg is not None
        and not parse_selector(exit_cfg[0])[1]
        and exit_cfg[0] not in state_set
    ):
        _raise_invalid_rhs_spec(
            detail=f"chain[{idx}] exit.to={exit_cfg[0]!r} not in state"
        )

    return stage_names, forward_rates, entry_cfg, exit_cfg


def _apply_expr_chains(
    *,
    chains: list[Any],
    state_expanded: list[str],
    equations_map: dict[str, Any],
) -> None:
    """Apply chain helper for expr kind by auto-filling equations when missing."""
    state_set = set(state_expanded)
    for c_idx, chain in enumerate(chains):
        stage_names, forward_rates, _, exit_cfg = _validate_chain_entry(
            chain=chain,
            idx=c_idx,
            state_set=state_set,
        )
        exit_rate = exit_cfg[1] if exit_cfg is not None else None
        final_out_rate = exit_rate or forward_rates[-1]
        for stage_name in stage_names:
            if stage_name not in state_set:
                state_expanded.append(stage_name)
                state_set.add(stage_name)

        if stage_names[0] not in equations_map:
            equations_map[stage_names[0]] = f"-({forward_rates[0]})*{stage_names[0]}"
        for i in range(1, len(stage_names)):
            if stage_names[i] not in equations_map:
                out_rate = (
                    forward_rates[i] if i < len(stage_names) - 1 else final_out_rate
                )
                equations_map[stage_names[i]] = (
                    f"({forward_rates[i - 1]})*{stage_names[i - 1]} - "
                    f"({out_rate})*{stage_names[i]}"
                )
        if exit_cfg is not None:
            sink_s, sink_rate = exit_cfg
            out_rate = sink_rate or forward_rates[-1]
            if sink_s not in equations_map:
                equations_map[sink_s] = f"({out_rate})*{stage_names[-1]}"


def _apply_transition_chains(
    *,
    chains: list[Any],
    state_raw: list[str],
    transitions_raw: list[dict[str, Any]],
    state_set: set[str],
) -> None:
    """Apply chain helper for transitions kind by appending transitions."""
    for c_idx, chain in enumerate(chains):
        stage_names, forward_rates, entry_cfg, exit_cfg = _validate_chain_entry(
            chain=chain,
            idx=c_idx,
            state_set=state_set,
        )

        for stage_name in stage_names:
            if stage_name not in state_set:
                state_raw.append(stage_name)
                state_set.add(stage_name)

        if entry_cfg is not None:
            entry_from, entry_rate = entry_cfg
            transitions_raw.append({
                "from": entry_from,
                "to": stage_names[0],
                "rate": entry_rate,
            })

        transitions_raw.extend(
            {
                "from": stage_names[i],
                "to": stage_names[i + 1],
                "rate": forward_rates[i],
            }
            for i in range(len(stage_names) - 1)
        )

        if exit_cfg is not None:
            sink_s, sink_rate = exit_cfg
            transitions_raw.append({
                "from": stage_names[-1],
                "to": sink_s,
                "rate": sink_rate or forward_rates[-1],
            })


def _validate_coord_shift_entry(
    tr: dict[str, Any],
    axis_lookup: Mapping[str, list[str]],
) -> tuple[str, str, str, list[Any], str]:
    """Parse and validate a single ``coord_shift`` transition entry.

    Returns:
        ``(axis_name, from_coord, to_coord, apply_to, rate)``
    """
    shift_spec = tr["coord_shift"]
    if not isinstance(shift_spec, dict) or len(shift_spec) != 1:
        _raise_invalid_rhs_spec(
            detail="coord_shift must be a mapping with exactly one axis entry",
        )

    axis_name, arrow = next(iter(shift_spec.items()))
    if axis_name not in axis_lookup:
        _raise_invalid_rhs_spec(
            detail=f"coord_shift axis {axis_name!r} is not defined",
        )

    if not isinstance(arrow, str) or "->" not in arrow:
        _raise_invalid_rhs_spec(
            detail=f"coord_shift[{axis_name}] must be 'from_coord -> to_coord'",
        )
    parts = [p.strip() for p in arrow.split("->")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        _raise_invalid_rhs_spec(
            detail=f"coord_shift[{axis_name}] must be 'from_coord -> to_coord'",
        )
    from_coord, to_coord = parts
    valid_coords = axis_lookup[axis_name]
    for coord in (from_coord, to_coord):
        if coord not in valid_coords:
            _raise_invalid_rhs_spec(
                detail=(
                    f"coord_shift coordinate {coord!r} not in "
                    f"axis {axis_name!r} coords {valid_coords}"
                ),
            )

    apply_to = tr.get("apply_to")
    if not isinstance(apply_to, list) or not apply_to:
        _raise_invalid_rhs_spec(
            detail="coord_shift requires a non-empty 'apply_to' list",
        )

    rate_s = tr.get("rate")
    if not isinstance(rate_s, str) or not rate_s.strip():
        _raise_invalid_rhs_spec(detail="coord_shift requires a 'rate' string")

    return axis_name, from_coord, to_coord, apply_to, rate_s.strip()


def _apply_coord_shifts(
    *,
    transitions_raw: list[dict[str, Any]],
    state_expanded: list[str],
    axes: list[dict[str, Any]],
) -> None:
    """Expand ``coord_shift`` entries into concrete transitions.

    Each ``coord_shift`` entry describes movement along one axis coordinate for
    a set of states.  The entry is replaced *in-place* by one concrete
    transition per state listed in ``apply_to``, per combination of the
    remaining (non-shifted) axes those states carry.

    Args:
        transitions_raw: Mutable transition list — ``coord_shift`` entries are
            replaced by concrete transition dicts.
        state_expanded: Expanded state names (used to discover which axes each
            ``apply_to`` state carries).
        axes: Normalized axis definitions.
    """
    axis_lookup: dict[str, list[str]] = {
        ax["name"]: [str(c) for c in ax.get("coords", [])] for ax in axes
    }

    i = 0
    while i < len(transitions_raw):
        tr = transitions_raw[i]
        if "coord_shift" not in tr:
            i += 1
            continue

        axis_name, from_coord, to_coord, apply_to, rate_s = _validate_coord_shift_entry(
            tr, axis_lookup
        )

        concrete: list[dict[str, Any]] = []
        from_frag = f"{axis_name}_{_sanitize_fragment(from_coord)}"
        to_frag = f"{axis_name}_{_sanitize_fragment(to_coord)}"
        expanded_apply_to = expand_apply_to(
            apply_to,
            axis_lookup=axis_lookup,
            context=f"coord_shift[{axis_name}].apply_to",
        )
        for base in expanded_apply_to:
            concrete.extend(
                _expand_coord_shift_for_base(
                    base=base,
                    from_frag=from_frag,
                    to_frag=to_frag,
                    rate_s=rate_s,
                    state_expanded=state_expanded,
                )
            )

        transitions_raw[i : i + 1] = concrete
        i += len(concrete)


def _expand_coord_shift_for_base(
    *,
    base: str,
    from_frag: str,
    to_frag: str,
    rate_s: str,
    state_expanded: list[str],
) -> list[dict[str, Any]]:
    """Emit concrete transitions for one ``apply_to`` base state.

    Discovers which axes the base carries by inspecting ``state_expanded``
    for names starting with ``base__``.  If the base has extra axes beyond
    the shifted one, a transition is emitted for each coordinate combination
    of those extra axes.  If only the shifted axis is present, a single
    transition is emitted.

    Returns:
        Concrete ``{"from", "to", "rate"}`` transition dicts.
    """
    prefix = f"{base}__"
    matching = [s for s in state_expanded if s.startswith(prefix)]

    if not matching:
        _raise_invalid_rhs_spec(
            detail=(
                f"coord_shift apply_to state {base!r} has no expanded states "
                f"starting with '{prefix}'"
            ),
        )

    concrete: list[dict[str, Any]] = []
    expanded_set = set(state_expanded)

    for state_name in matching:
        if from_frag not in state_name:
            continue

        target = state_name.replace(from_frag, to_frag, 1)
        if target not in expanded_set:
            _raise_invalid_rhs_spec(
                detail=(
                    f"coord_shift would create transition to {target!r} "
                    f"which is not an expanded state"
                ),
            )

        concrete.append({"from": state_name, "to": target, "rate": rate_s})

    if not concrete:
        _raise_invalid_rhs_spec(
            detail=(
                f"coord_shift apply_to state {base!r} has no expanded states "
                f"with fragment {from_frag!r}"
            ),
        )

    return concrete


# -----------------------------------------------------------------------------
# Public normalization entrypoint
# -----------------------------------------------------------------------------


def normalize_rhs(spec: Mapping[str, Any] | None) -> NormalizedRhs:
    """
    Normalize a RHS specification dict into a backend-facing representation.

    Args:
        spec: Raw RHS specification mapping.

    Returns:
        Backend-facing normalized RHS representation.
    """
    if spec is None:
        _raise_invalid_rhs_spec(detail="rhs specification is required")
    # spec is not None past this point; the above call is NoReturn when spec is None.

    kind = str(spec.get("kind", "expr")).strip().lower()

    if kind == "expr":  # lowest-level escape hatch
        return normalize_expr_rhs(spec)

    if kind == "transitions":  # diagram-style hazards
        return normalize_transitions_rhs(spec)

    _raise_unsupported_feature(
        feature=f"rhs.kind={kind}",
        detail="Only 'expr' and 'transitions' are supported in v1.",
    )
    return normalize_expr_rhs(spec)  # unreachable; satisfies return type checker


# -----------------------------------------------------------------------------
# expr kind
# -----------------------------------------------------------------------------


def normalize_expr_rhs(spec: Mapping[str, Any]) -> NormalizedRhs:
    """
    Normalize an expression-based RHS specification.

    Args:
        spec: Raw RHS specification mapping.

    Returns:
        Backend-facing normalized RHS representation.
    """
    state_raw = _ensure_str_list(spec.get("state"), name="state")
    if len(state_raw) != len(set(state_raw)):
        _raise_invalid_rhs_spec(detail="state contains duplicate names")

    equations_map = spec.get("equations")
    if not isinstance(equations_map, dict):
        _raise_invalid_rhs_spec(detail="equations must be a mapping of state->expr")

    # Normalize equation keys so that e.g. "u[x, y]" matches template "u[x,y]"
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
        _raise_invalid_rhs_spec(detail="expanded state contains duplicates")

    aliases, alias_template_map = _expand_alias_templates(
        meta_parts[0], axes=axes_meta, template_map_seed=state_template_map
    )
    template_map_all = {**state_template_map, **alias_template_map}

    chain_block = spec.get("chain")
    if chain_block:
        if not isinstance(chain_block, list):
            _raise_invalid_rhs_spec(detail="chain must be a list if provided")
        _apply_expr_chains(
            chains=chain_block,
            state_expanded=state_expanded,
            equations_map=equations_map,
        )

    # Validate equation keys: allow either concrete states or template keys
    unknown_keys = [
        k
        for k in equations_map
        if k not in state_expanded and k not in template_map_all
    ]
    if unknown_keys:
        _raise_invalid_rhs_spec(
            detail=f"unknown equation key(s): {sorted(unknown_keys)}"
        )

    all_syms = _collect_alias_symbols(aliases, axes=axes_meta)
    eqs = _gather_equations(
        state_expanded,
        equations_map,
        all_syms,
        axes=axes_meta,
        template_map=template_map_all,
    )

    _maybe_attach_initial_state(
        meta,
        spec.get("initial_state"),
        axes=axes_meta,
        template_map=template_map_all,
    )

    return NormalizedRhs(
        kind="expr",
        state_names=tuple(state_expanded),
        equations=tuple(eqs),
        aliases=aliases,
        param_names=_sorted_unique(
            sym
            for sym in all_syms
            if sym not in set(state_expanded) and sym not in aliases
        ),
        all_symbols=frozenset(all_syms | set(aliases.keys())),
        meta=meta,
    )


# -----------------------------------------------------------------------------
# transitions kind (hazard semantics)
# -----------------------------------------------------------------------------


def _validate_transition_mapping(tr: object, *, idx: int) -> Mapping[str, Any]:
    """
    Validate and return a transition mapping.

    Args:
        tr: Transition object.
        idx: Transition index (for error messages).

    Returns:
        Transition mapping.
    """
    if not isinstance(tr, dict):
        _raise_invalid_rhs_spec(detail=f"transitions[{idx}] must be a mapping")
    if "name" in tr:
        name_val = tr.get("name")
        if not isinstance(name_val, str) or not name_val.strip():
            _raise_invalid_rhs_spec(
                detail=f"transitions[{idx}].name must be a non-empty string"
            )
    return tr


def _get_required_str(tr: Mapping[str, Any], *, idx: int, key: str) -> str:
    """
    Fetch a required string field from a transition mapping.

    Args:
        tr: Transition mapping.
        idx: Transition index (for error messages).
        key: Field key.

    Returns:
        Stripped string value.
    """
    val = tr.get(key)
    if not isinstance(val, str) or not val.strip():
        _raise_invalid_rhs_spec(detail=f"transitions[{idx}].{key} must be a string")
    return val.strip()


def _apply_transition(
    *,
    idx: int,
    tr: Mapping[str, Any],
    state_set: set[str],
    all_syms: set[str],
    d_terms: dict[str, list[str]],
) -> None:
    """Apply a transition to the derivative-term accumulator."""
    frm_s = _get_required_str(tr, idx=idx, key="from")
    to_s = _get_required_str(tr, idx=idx, key="to")
    rate_s = _get_required_str(tr, idx=idx, key="rate")

    if frm_s not in state_set:
        _raise_invalid_rhs_spec(
            detail=f"transitions[{idx}].from={frm_s!r} not in state"
        )
    if to_s not in state_set:
        _raise_invalid_rhs_spec(detail=f"transitions[{idx}].to={to_s!r} not in state")

    tree = _parse_expr(rate_s)
    all_syms |= _collect_names(tree)

    flow = f"({rate_s})*({frm_s})"
    d_terms[frm_s].append(f"-({flow})")
    d_terms[to_s].append(f"+({flow})")


def _build_transition_equations(
    state: list[str], d_terms: Mapping[str, list[str]]
) -> list[str]:
    equations: list[str] = []
    for name in state:
        terms = d_terms[name]
        if not terms:
            equations.append("0.0")
            continue
        expr = " ".join(terms)
        if expr.startswith("+") and expr[1:2] == "(":
            expr = expr[1:]
        equations.append(expr)
    return equations


def normalize_transitions_rhs(
    spec: Mapping[str, Any],
) -> NormalizedRhs:
    """Normalize a transition-based RHS specification (diagram/hazard semantics).

    Returns:
        Backend-facing normalized RHS representation for transitions kind.
    """
    state_raw = _ensure_str_list(spec.get("state"), name="state")
    if len(state_raw) != len(set(state_raw)):
        _raise_invalid_rhs_spec(detail="state contains duplicate names")

    transitions_raw = spec.get("transitions")
    if transitions_raw is None:
        transitions_raw = []
    elif isinstance(transitions_raw, list):
        transitions_raw = list(transitions_raw)
    else:
        _raise_invalid_rhs_spec(detail="transitions must be a list")

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
            _raise_invalid_rhs_spec(detail="chain must be a list if provided")
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
        _raise_invalid_rhs_spec(detail="expanded state contains duplicates")

    aliases, alias_template_map = _expand_alias_templates(
        meta_parts[0], axes=axes_meta, template_map_seed=state_template_map
    )
    template_map_all = {**state_template_map, **alias_template_map}

    state_set = set(state_expanded)
    d_terms: dict[str, list[str]] = {s: [] for s in state_expanded}
    all_syms = _collect_alias_symbols(aliases, axes=axes_meta)

    _apply_coord_shifts(
        transitions_raw=transitions_raw,
        state_expanded=state_expanded,
        axes=axes_meta,
    )

    for state_name in state_expanded:
        d_terms.setdefault(state_name, [])

    if not transitions_raw:
        _raise_invalid_rhs_spec(
            detail="transitions must be non-empty after applying chain expansion"
        )

    transitions_expanded = _expand_transition_templates(
        transitions_raw,
        axes=axes_meta,
        template_map=template_map_all,
    )

    for idx, tr_map in enumerate(transitions_expanded):
        _apply_transition(
            idx=idx,
            tr=tr_map,
            state_set=state_set,
            all_syms=all_syms,
            d_terms=d_terms,
        )

    _maybe_attach_initial_state(
        meta,
        spec.get("initial_state"),
        axes=axes_meta,
        template_map=template_map_all,
    )

    return NormalizedRhs(
        kind="transitions",
        state_names=tuple(state_expanded),
        equations=tuple(_build_transition_equations(state_expanded, d_terms)),
        aliases=aliases,
        param_names=_sorted_unique(
            sym for sym in all_syms if sym not in state_set and sym not in aliases
        ),
        all_symbols=frozenset(all_syms | set(aliases.keys())),
        meta={**meta, "transitions": transitions_expanded},
    )
