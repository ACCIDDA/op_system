"""op_system._normalize.

Core normalization implementations for op_system RHS specifications.

Public entry points:

- ``normalize_rhs``:             dispatch to expr or transitions normalizer
- ``normalize_expr_rhs``:        normalize an ``expr``-kind spec
- ``normalize_transitions_rhs``: normalize a ``transitions``-kind spec

All internal helpers for both normalization paths live here.  The public
types and functions are re-exported from ``specs.py`` for backward compat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from op_system._axes import _normalize_axes, _normalize_bracket_key
from op_system._errors import InvalidRhsSpecError, UnsupportedFeatureError
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

# ---------------------------------------------------------------------------
# Regex patterns for helper expression expansion
# ---------------------------------------------------------------------------

_ALLOWED_KERNEL_FORMS: dict[str, tuple[str, ...]] = {
    "erfc": ("scale", "sigma"),
    "gaussian": ("scale", "sigma"),
    "exponential": ("scale", "lambda"),
    "gamma": ("scale", "k", "theta"),
    "power_law": ("scale", "sigma", "p"),
    "custom_value": (),
}


# ---------------------------------------------------------------------------
# Normalized RHS representation (backend-facing public type)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Small shared validators
# ---------------------------------------------------------------------------


def _get_required_str(tr: Mapping[str, Any], *, idx: int, key: str) -> str:
    """Fetch a required non-empty string field from a mapping.

    Returns:
        Stripped string value.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    val = tr.get(key)
    if not isinstance(val, str) or not val.strip():
        raise InvalidRhsSpecError(detail=f"transitions[{idx}].{key} must be a string")
    return val.strip()


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


def _validate_scalar_or_expr(value: object, field: str) -> None:
    """Raise if *value* is not a non-empty string or a finite number.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if isinstance(value, str):
        if not value.strip():
            raise InvalidRhsSpecError(
                detail=f"{field} must be a non-empty string or number"
            )
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRhsSpecError(
            detail=f"{field} must be a non-empty string or number"
        )


# ---------------------------------------------------------------------------
# State-axes normalization
# ---------------------------------------------------------------------------


def _normalize_state_axes(
    raw_state_axes: object,
    *,
    axis_names: set[str],
    state_set: set[str],
) -> dict[str, tuple[str, ...]]:
    """Normalize mapping of state -> list of axis names.

    Returns:
        Mapping of state names to tuples of axis names.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if raw_state_axes is None:
        return {}
    mapping = _ensure_mapping(raw_state_axes, name="state_axes")
    out: dict[str, tuple[str, ...]] = {}
    for state, axes in mapping.items():
        if state not in state_set:
            raise InvalidRhsSpecError(detail=f"state_axes key {state!r} not in state")
        if not isinstance(axes, (list, tuple)) or not axes:
            raise InvalidRhsSpecError(
                detail=f"state_axes[{state!r}] must be a non-empty list of axis names"
            )
        resolved: list[str] = []
        seen: set[str] = set()
        for i, ax in enumerate(axes):
            if not isinstance(ax, str) or not ax.strip():
                raise InvalidRhsSpecError(
                    detail=f"state_axes[{state!r}][{i}] must be a non-empty string"
                )
            ax_name = ax.strip()
            if ax_name not in axis_names:
                raise InvalidRhsSpecError(
                    detail=f"state_axes[{state!r}] references unknown axis {ax_name!r}"
                )
            if ax_name in seen:
                raise InvalidRhsSpecError(
                    detail=f"state_axes[{state!r}] contains duplicate axis {ax_name!r}"
                )
            seen.add(ax_name)
            resolved.append(ax_name)
        out[state] = tuple(resolved)
    return out


# ---------------------------------------------------------------------------
# Kernel normalization
# ---------------------------------------------------------------------------


def _normalize_kernel_axes_field(
    axes_field: object, *, idx: int, axis_names: set[str]
) -> tuple[str, ...] | None:
    """Normalize a kernel's ``axes`` field.

    Returns:
        Tuple of axis name strings, or ``None`` if not provided.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if axes_field is None:
        return None
    if not isinstance(axes_field, (list, tuple)) or not axes_field:
        raise InvalidRhsSpecError(
            detail=f"kernels[{idx}].axes must be a non-empty list if provided"
        )
    resolved_axes: list[str] = []
    for ax_idx, ax in enumerate(axes_field):
        if not isinstance(ax, str) or not ax.strip():
            raise InvalidRhsSpecError(
                detail=f"kernels[{idx}].axes[{ax_idx}] must be a non-empty string"
            )
        ax_name = ax.strip()
        if ax_name not in axis_names:
            raise InvalidRhsSpecError(
                detail=f"kernels[{idx}] references unknown axis {ax_name!r}"
            )
        resolved_axes.append(ax_name)
    return tuple(resolved_axes)


def _normalize_kernel_form_and_params(
    *, value: object, form: object, params_field: object, idx: int
) -> tuple[str, Mapping[str, Any] | None]:
    """Normalize a kernel's form name and params.

    Returns:
        Tuple of (form_name, params_mapping_or_none).

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if value is None:
        if not isinstance(form, str) or not form.strip():
            raise InvalidRhsSpecError(
                detail=(f"kernels[{idx}].form is required when value is not provided")
            )
        form_name = form.strip().lower()
        if form_name not in _ALLOWED_KERNEL_FORMS:
            raise InvalidRhsSpecError(
                detail=f"kernels[{idx}] form {form_name!r} is not supported"
            )
        required_keys = _ALLOWED_KERNEL_FORMS[form_name]
        params_map_required = _ensure_mapping(
            params_field, name=f"kernels[{idx}].params"
        )
        for req in required_keys:
            if req not in params_map_required:
                raise InvalidRhsSpecError(
                    detail=f"kernels[{idx}].params missing required key {req!r}"
                )
        return form_name, params_map_required

    form_name = (
        str(form).strip().lower()
        if isinstance(form, str) and form.strip()
        else "custom_value"
    )
    if form_name and form_name not in _ALLOWED_KERNEL_FORMS:
        raise InvalidRhsSpecError(
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
    """Normalize one kernel entry.

    Returns:
        Normalized kernel dict.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    name = _get_required_str(mk_map, idx=idx, key="name")
    if name in seen:
        raise InvalidRhsSpecError(detail=f"duplicate kernel name: {name!r}")
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

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if raw_kernels is None:
        return []
    if not isinstance(raw_kernels, list):
        raise InvalidRhsSpecError(detail="kernels must be a list")

    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for idx, mk in enumerate(raw_kernels):
        mk_map = _ensure_mapping(mk, name=f"kernels[{idx}]")
        mk_out = _normalize_single_kernel(
            mk_map, idx=idx, axis_names=axis_names, seen=seen
        )
        out.append(mk_out)

    return out


# ---------------------------------------------------------------------------
# Operator normalization
# ---------------------------------------------------------------------------


def _validate_op_apply_to(
    apply_to_raw: object,
    idx: int,
    state_set: set[str] | None,
    *,
    axes: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Validate and expand an operator's ``apply_to`` field.

    Returns:
        List of concrete (or bare) state name strings.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if not isinstance(apply_to_raw, (list, tuple)) or not apply_to_raw:
        raise InvalidRhsSpecError(
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
                raise InvalidRhsSpecError(
                    detail=f"operators[{idx}].apply_to[{j}] must be a non-empty string"
                )
            state_name_s = state_name.strip()
            if state_set is not None and state_name_s not in state_set:
                raise InvalidRhsSpecError(
                    detail=(
                        f"operators[{idx}].apply_to[{j}]={state_name_s!r} not in state"
                    )
                )
            result.append(state_name_s)
    return result


def _validate_op_advection(op_map: Mapping[str, Any], idx: int, kind_s: str) -> None:
    """Validate advection/transport operator fields.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    velocity_val = op_map.get("velocity")
    if velocity_val is None:
        raise InvalidRhsSpecError(
            detail=f"operators[{idx}].velocity is required for {kind_s!r}"
        )
    _validate_scalar_or_expr(velocity_val, f"operators[{idx}].velocity")


def _validate_op_jump_integral(op_map: Mapping[str, Any], idx: int) -> None:
    """Validate jump_integral operator fields.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    rate_val = op_map.get("rate")
    if rate_val is None:
        raise InvalidRhsSpecError(
            detail=f"operators[{idx}].rate is required for 'jump_integral'"
        )
    _validate_scalar_or_expr(rate_val, f"operators[{idx}].rate")

    kernel_val = op_map.get("kernel")
    if not isinstance(kernel_val, dict):
        raise InvalidRhsSpecError(
            detail=f"operators[{idx}].kernel must be a mapping for 'jump_integral'"
        )
    kernel_form = kernel_val.get("form")
    if not isinstance(kernel_form, str) or not kernel_form.strip():
        raise InvalidRhsSpecError(
            detail=(
                f"operators[{idx}].kernel.form must be a non-empty "
                "string for 'jump_integral'"
            )
        )
    kernel_params = kernel_val.get("params")
    if kernel_params is not None and not isinstance(kernel_params, dict):
        raise InvalidRhsSpecError(
            detail=(
                f"operators[{idx}].kernel.params must be a mapping if "
                "provided for 'jump_integral'"
            )
        )

    direction_val = op_map.get("direction")
    if direction_val is not None:
        if not isinstance(direction_val, str) or not direction_val.strip():
            raise InvalidRhsSpecError(
                detail=(
                    f"operators[{idx}].direction must be one of 'up', 'down', or 'both'"
                )
            )
        if direction_val.strip().lower() not in {"up", "down", "both"}:
            raise InvalidRhsSpecError(
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
        Tuple of ``(op_name_s, axis_name_s, kind_s, bc_val_s)``.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    op_name = op_map.get("name")
    if op_name is not None and (not isinstance(op_name, str) or not op_name.strip()):
        raise InvalidRhsSpecError(
            detail=f"operators[{idx}].name must be a non-empty string if provided"
        )

    axis_name = op_map.get("axis")
    if not isinstance(axis_name, str) or not axis_name.strip():
        raise InvalidRhsSpecError(
            detail=f"operators[{idx}].axis must be a non-empty string"
        )
    axis_name_s = axis_name.strip()
    if axis_name_s not in axis_names:
        raise InvalidRhsSpecError(
            detail=f"operators[{idx}] references unknown axis {axis_name_s!r}"
        )

    kind_val = op_map.get("kind")
    if not isinstance(kind_val, str) or not kind_val.strip():
        raise InvalidRhsSpecError(
            detail=(
                f"operators[{idx}].kind must be a non-empty string "
                "(e.g., advection/jump_integral)"
            )
        )
    kind_s = kind_val.strip().lower()

    bc_val = op_map.get("bc")
    if bc_val is not None and (not isinstance(bc_val, str) or not bc_val.strip()):
        raise InvalidRhsSpecError(
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
    """Normalize one operator entry.

    Returns:
        Normalized operator dict.
    """
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

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if raw_ops is None:
        return []
    if not isinstance(raw_ops, list):
        raise InvalidRhsSpecError(detail="operators must be a list")

    return [
        _normalize_single_operator(
            op, idx, axis_names=axis_names, state_set=state_set, axes=axes
        )
        for idx, op in enumerate(raw_ops)
    ]


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
# Template expansion helpers
# ---------------------------------------------------------------------------


def _collect_alias_symbols(
    aliases: Mapping[str, str], *, axes: list[dict[str, Any]]
) -> set[str]:
    """Collect all symbol names referenced in alias expressions.

    Returns:
        Set of symbol name strings found across all alias expressions.
    """
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


def _expand_alias_templates(
    aliases_raw: Mapping[str, str],
    *,
    axes: list[dict[str, Any]],
    template_map_seed: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> tuple[dict[str, str], dict[str, list[tuple[str, dict[str, str]]]]]:
    """Expand templated alias names and substitute inline placeholders.

    Returns:
        A tuple of ``(expanded_aliases, alias_template_map)``.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    alias_names = list(aliases_raw.keys())
    alias_expanded, alias_template_map = _expand_state_templates(alias_names, axes=axes)
    if len(alias_expanded) != len(set(alias_expanded)):
        raise InvalidRhsSpecError(detail="expanded aliases contain duplicates")

    combined_template_map = {**template_map_seed, **alias_template_map}
    aliases_out: dict[str, str] = {}

    for raw_name, expr in aliases_raw.items():
        canonical_name = _normalize_bracket_key(raw_name)
        expr_s = expr.strip()
        if not expr_s:
            raise InvalidRhsSpecError(
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
        Expanded ``dict[str, str]`` mapping, or ``None`` if *initial_state_raw*
        is ``None``.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if initial_state_raw is None:
        return None

    axis_lookup = build_axis_lookup(axes)
    result: dict[str, str] = {}
    expanded_keys: list[str] = []

    for raw_key, raw_val in initial_state_raw.items():
        val_s = str(raw_val).strip()
        if not val_s:
            raise InvalidRhsSpecError(
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
        raise InvalidRhsSpecError(
            detail="expanded initial_state keys contain duplicates"
        )

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


# ---------------------------------------------------------------------------
# Transition expansion
# ---------------------------------------------------------------------------


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
    """Return ordered list of unique wildcard axes from endpoints and expressions.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
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
                raise InvalidRhsSpecError(
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
    """Expand one transition template over all wildcard combinations.

    Returns:
        List of concrete transition dicts.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
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
            raise InvalidRhsSpecError(
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

    Supports placeholders in from/to/name/rate fields (e.g., ``S[vax,age]``).

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


# ---------------------------------------------------------------------------
# Equation expansion (apply_along, gather)
# ---------------------------------------------------------------------------


def _find_call_span(expr: str, name: str) -> tuple[int, int] | None:
    """Locate the first balanced ``name(...)`` call in *expr*.

    Returns:
        ``(start, end)`` half-open span covering ``name(...)`` (end is the
        index just past the closing paren), or ``None`` if no such call
        exists.  Matches whole-word identifiers only.

    Raises:
        InvalidRhsSpecError: If parentheses inside the call are unbalanced.
    """
    pat = re.compile(rf"\b{re.escape(name)}\s*\(")
    m = pat.search(expr)
    if m is None:
        return None
    start = m.start()
    i = m.end()
    depth = 1
    n = len(expr)
    while i < n:
        c = expr[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return (start, i + 1)
        i += 1
    raise InvalidRhsSpecError(detail=f"unbalanced parentheses in {name}(...) call")


def _split_top_level_commas(s: str) -> list[str]:
    """Split *s* on commas at the top paren/bracket nesting level.

    Returns:
        List of comma-separated argument strings (each stripped of
        surrounding whitespace).  Empty input yields ``[]``.
    """
    out: list[str] = []
    depth = 0
    last = 0
    for i, c in enumerate(s):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(s[last:i].strip())
            last = i + 1
    tail = s[last:].strip()
    if tail or out:
        out.append(tail)
    return out


_KERNEL_KW_RE = re.compile(r"^kernel\s*=\s*(.+)$", re.DOTALL)
_AXIS_BIND_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", re.DOTALL)
_APPLY_ALONG_KERNELS = frozenset({"sum", "integrate"})


def _parse_apply_along_args(  # noqa: C901
    args_str: str,
    *,
    axis_names: set[str],
) -> tuple[list[tuple[str, str]], str | None, str]:
    """Parse the argument list of an ``apply_along(...)`` call.

    Args:
        args_str: The raw text between the call's parentheses.
        axis_names: Known axis names (used to distinguish axis bindings
            from the inner expression).

    Returns:
        Tuple ``(axis_bindings, kernel_form, inner_expr)`` where
        ``axis_bindings`` is a list of ``(axis_name, var_name)`` pairs in
        declaration order, ``kernel_form`` is ``"sum"``, ``"integrate"``
        or ``None`` (meaning auto-select from axis types), and
        ``inner_expr`` is the bound inner expression.

    Raises:
        InvalidRhsSpecError: If the argument list is malformed (no args,
            empty arg, duplicate ``kernel=``, unknown kernel name, missing
            axis bindings, non-identifier var name, or wrong number of
            inner expressions).
    """
    parts = _split_top_level_commas(args_str)
    if not parts or all(not p for p in parts):
        raise InvalidRhsSpecError(detail="apply_along(...) requires arguments")

    bindings: list[tuple[str, str]] = []
    kernel_form: str | None = None
    inner_parts: list[str] = []

    for part in parts:
        if not part:
            raise InvalidRhsSpecError(detail="apply_along(...) has empty argument")
        km = _KERNEL_KW_RE.match(part)
        if km is not None:
            if kernel_form is not None:
                raise InvalidRhsSpecError(
                    detail="apply_along(...) accepts at most one 'kernel=' kwarg"
                )
            kf = km.group(1).strip().lower()
            if kf not in _APPLY_ALONG_KERNELS:
                raise InvalidRhsSpecError(
                    detail=(
                        f"apply_along(...) kernel must be one of "
                        f"{sorted(_APPLY_ALONG_KERNELS)}, got {kf!r}"
                    )
                )
            kernel_form = kf
            continue
        bm = _AXIS_BIND_RE.match(part)
        if bm is not None and bm.group(1) in axis_names:
            ax_name = bm.group(1)
            var_name = bm.group(2).strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var_name):
                raise InvalidRhsSpecError(
                    detail=(
                        f"apply_along(...) axis binding {ax_name}={var_name!r} "
                        "must bind to an identifier"
                    )
                )
            bindings.append((ax_name, var_name))
            continue
        inner_parts.append(part)

    if not bindings:
        raise InvalidRhsSpecError(
            detail="apply_along(...) requires at least one axis=var binding"
        )
    if len(inner_parts) != 1:
        raise InvalidRhsSpecError(
            detail=(
                "apply_along(...) requires exactly one inner expression argument "
                f"(got {len(inner_parts)})"
            )
        )
    return bindings, kernel_form, inner_parts[0]


def _select_apply_along_kernel(
    bindings: list[tuple[str, str]],
    kernel_form: str | None,
    *,
    axes: list[dict[str, Any]],
) -> str:
    """Select the per-axis helper form for an ``apply_along`` call.

    Returns:
        Either ``"sum"`` or ``"integrate"``.

    Raises:
        InvalidRhsSpecError: If an explicit kernel is incompatible with
            the bound axis types, or if the kernel cannot be inferred from
            mixed/unknown axis types.
    """
    axis_types: dict[str, str] = {
        str(ax.get("name")): str(ax.get("type", "")) for ax in axes
    }
    types = {axis_types.get(ax) for ax, _ in bindings}
    if kernel_form == "sum":
        bad = [ax for ax, _ in bindings if axis_types.get(ax) != "categorical"]
        if bad:
            raise InvalidRhsSpecError(
                detail=(
                    f"apply_along(kernel=sum) requires categorical axes; "
                    f"got non-categorical: {bad}"
                )
            )
        return "sum"
    if kernel_form == "integrate":
        bad = [ax for ax, _ in bindings if axis_types.get(ax) != "continuous"]
        if bad:
            raise InvalidRhsSpecError(
                detail=(
                    f"apply_along(kernel=integrate) requires continuous axes; "
                    f"got non-continuous: {bad}"
                )
            )
        return "integrate"
    if types == {"categorical"}:
        return "sum"
    if types == {"continuous"}:
        return "integrate"
    raise InvalidRhsSpecError(
        detail=(
            "apply_along(...) cannot infer kernel for mixed/unknown axis types "
            f"({sorted(t for t in types if t)}); pass kernel=sum or kernel=integrate"
        )
    )


def _substitute_apply_along_brackets(expr: str, bound: Mapping[str, str]) -> str:
    """Rewrite ``name[ax=c, ...]`` to ``name__ax_<c>__...`` for bound axes.

    For each ``name[...]`` subexpression, axis assignments whose axis is in
    ``bound`` and whose coord matches ``bound[axis]`` are consumed into a
    ``__axis_<sanitized-coord>`` suffix on ``name``.  Remaining assignments
    stay inside the brackets.  The bracket itself is dropped if every
    assignment is consumed.

    Returns:
        Expression string with bound bracket assignments folded into
        ``__axis_coord`` suffixes.
    """
    pat = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[([^\[\]]*)\]")

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        body = match.group(2)
        entries = [e.strip() for e in body.split(",") if e.strip()]
        consumed: list[tuple[str, str]] = []
        remaining: list[str] = []
        for entry in entries:
            if "=" in entry:
                ax, _, coord = entry.partition("=")
                ax = ax.strip()
                coord = coord.strip()
                if ax in bound and bound[ax] == coord:
                    consumed.append((ax, coord))
                    continue
            remaining.append(entry)
        if not consumed:
            return match.group(0)
        suffix = "".join(
            f"__{ax}_{_sanitize_fragment(coord)}" for ax, coord in consumed
        )
        if remaining:
            return f"{name}{suffix}[{', '.join(remaining)}]"
        return f"{name}{suffix}"

    return pat.sub(_sub, expr)


def _expand_apply_along(expr: str, *, axes: list[dict[str, Any]]) -> str:  # noqa: PLR0914
    """Expand ``apply_along(...)`` calls directly to weighted sums.

    The ``apply_along`` primitive contracts an expression along one or more
    axes using a single shared kernel form.  Categorical axes use uniform
    weights of 1; continuous axes use trapezoidal weights derived from the
    axis ``deltas``.  All bound axes are substituted in a single Cartesian
    expansion, so ``name[ax1=c1, ax2=c2, ...]`` brackets that fully bind
    apply_along axes collapse to ``name__ax1_<c1>__ax2_<c2>``.

    Returns:
        Expression string with ``apply_along`` calls fully expanded.

    Raises:
        InvalidRhsSpecError: If a bound axis has no coords, or if a
            continuous axis is missing trapezoidal ``deltas``.
    """
    axis_lookup = {str(ax.get("name")): ax for ax in axes if ax.get("name")}
    out = expr
    while True:
        span = _find_call_span(out, "apply_along")
        if span is None:
            return out
        start, end = span
        args_str = out[start + len("apply_along(") : end - 1]
        bindings, kernel_form, inner = _parse_apply_along_args(
            args_str, axis_names=set(axis_lookup)
        )
        kernel = _select_apply_along_kernel(bindings, kernel_form, axes=axes)
        # Recursively expand any inner apply_along first so we work on a flat
        # inner expression for substitution.
        inner = _expand_apply_along(inner, axes=axes)

        # Build per-axis (coord, weight) lists.
        axis_options: list[list[tuple[str, float]]] = []
        for ax_name, _var in bindings:
            ax = axis_lookup[ax_name]
            coords = [str(c) for c in ax.get("coords", [])]
            if not coords:
                raise InvalidRhsSpecError(
                    detail=f"apply_along axis {ax_name!r} has no coords"
                )
            if kernel == "integrate":
                deltas = ax.get("deltas") or []
                if len(deltas) != len(coords):
                    raise InvalidRhsSpecError(
                        detail=(
                            f"apply_along axis {ax_name!r} missing or mismatched deltas"
                        )
                    )
                axis_options.append([
                    (c, float(d)) for c, d in zip(coords, deltas, strict=True)
                ])
            else:
                axis_options.append([(c, 1.0) for c in coords])

        terms: list[str] = []
        for combo in product(*axis_options):
            replaced = inner
            bound: dict[str, str] = {}
            for (ax_name, var_name), (coord, _w) in zip(bindings, combo, strict=True):
                replaced = re.sub(rf"\b{re.escape(var_name)}\b", coord, replaced)
                bound[ax_name] = coord
            replaced = _substitute_apply_along_brackets(replaced, bound)
            if kernel == "integrate":
                weight = "*".join(str(w) for _, w in combo)
                terms.append(f"({weight})*({replaced})")
            else:
                terms.append(f"({replaced})")
        replacement = " + ".join(terms)
        out = out[:start] + f"({replacement})" + out[end:]


def _expand_helpers(expr: str, *, axes: list[dict[str, Any]]) -> str:
    """Expand helper calls (currently only ``apply_along``) before AST parsing.

    Returns:
        Expression string with helper calls expanded.
    """
    return _expand_apply_along(expr, axes=axes)


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
        Expanded expression string if found, otherwise ``None``.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    for template_key, variants in template_map.items():
        for expanded_name, assignment in variants:
            if expanded_name != name or template_key not in equations_map:
                continue
            expr = equations_map[template_key]
            if not isinstance(expr, str) or not expr.strip():
                raise InvalidRhsSpecError(
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


def _gather_equations(
    state: list[str],
    equations_map: Mapping[str, Any],
    all_syms: set[str],
    *,
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]] | None = None,
) -> list[str]:
    """Gather and expand one equation string per state variable.

    Returns:
        List of expanded equation strings in the same order as *state*.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    eqs: list[str] = []
    template_map = template_map or {}
    for name in state:
        if name in equations_map:
            expr = equations_map[name]
            if not isinstance(expr, str) or not expr.strip():
                raise InvalidRhsSpecError(
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
            raise InvalidRhsSpecError(detail=f"Missing equation for state {name!r}")
        eqs.append(expr_res)
    return eqs


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------


def _chain_rate_expr(value: object, *, field: str) -> str:
    """Normalize a chain rate value into an expression string.

    Returns:
        A non-empty expression string.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(float(value))
    raise InvalidRhsSpecError(detail=f"{field} must be a non-empty string or number")


def _normalize_chain_forward_rates(
    forward_raw: object,
    *,
    idx: int,
    length: int,
) -> list[str]:
    """Normalize chain forward rates into per-edge expressions.

    Returns:
        A list of ``length - 1`` rate expression strings.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if isinstance(forward_raw, (str, int, float)) and not isinstance(forward_raw, bool):
        expr = _chain_rate_expr(forward_raw, field=f"chain[{idx}].forward")
        return [expr] * (length - 1)

    if not isinstance(forward_raw, (list, tuple)):
        raise InvalidRhsSpecError(
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
        raise InvalidRhsSpecError(
            detail=(
                f"chain[{idx}].forward list length must be {length - 1} "
                f"for chain length {length}"
            )
        )
    return rates


def _build_chain_stage_names(cname: str, *, length: int) -> list[str]:
    """Build chain stage names, preserving template placeholders when present.

    Returns:
        Ordered list of stage name strings for the chain.
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

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    entry_raw = chain.get("entry")
    if entry_raw is None:
        return None
    entry_map = _ensure_mapping(entry_raw, name=f"chain[{idx}].entry")
    frm = entry_map.get("from")
    if not isinstance(frm, str) or not frm.strip():
        raise InvalidRhsSpecError(
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

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    exit_raw = chain.get("exit")
    if exit_raw is not None:
        exit_map = _ensure_mapping(exit_raw, name=f"chain[{idx}].exit")
        to_raw = exit_map.get("to")
        if not isinstance(to_raw, str) or not to_raw.strip():
            raise InvalidRhsSpecError(
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
        raise InvalidRhsSpecError(detail=f"chain[{idx}].to must be a non-empty string")
    return to_legacy.strip(), None


def _validate_chain_entry(
    *,
    chain: Mapping[str, Any],
    idx: int,
    state_set: set[str],
) -> tuple[list[str], list[str], tuple[str, str] | None, tuple[str, str | None] | None]:
    """Validate a chain entry and return normalized chain configuration.

    Returns:
        ``(stage_names, forward_rates, entry_cfg, exit_cfg)``.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if not isinstance(chain, dict):
        raise InvalidRhsSpecError(detail=f"chain[{idx}] must be a mapping")
    cname = _get_required_str(chain, idx=idx, key="name")
    length_obj = chain.get("length")
    if not isinstance(length_obj, (int, float)) or isinstance(length_obj, bool):
        raise InvalidRhsSpecError(detail=f"chain[{idx}].length must be an integer >= 2")
    clen = int(length_obj)
    if clen < 2:
        raise InvalidRhsSpecError(detail=f"chain[{idx}].length must be >= 2")

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
        raise InvalidRhsSpecError(
            detail=f"chain[{idx}].entry.from={entry_cfg[0]!r} not in state"
        )

    exit_cfg = _normalize_chain_exit(chain, idx=idx)
    if (
        exit_cfg is not None
        and not parse_selector(exit_cfg[0])[1]
        and exit_cfg[0] not in state_set
    ):
        raise InvalidRhsSpecError(
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


# ---------------------------------------------------------------------------
# Coord-shift expansion
# ---------------------------------------------------------------------------


def _validate_coord_shift_entry(
    tr: dict[str, Any],
    axis_lookup: Mapping[str, list[str]],
) -> tuple[str, str, str, list[Any], str]:
    """Parse and validate a single ``coord_shift`` transition entry.

    Returns:
        ``(axis_name, from_coord, to_coord, apply_to, rate)``

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    shift_spec = tr["coord_shift"]
    if not isinstance(shift_spec, dict) or len(shift_spec) != 1:
        raise InvalidRhsSpecError(
            detail="coord_shift must be a mapping with exactly one axis entry",
        )

    axis_name, arrow = next(iter(shift_spec.items()))
    if axis_name not in axis_lookup:
        raise InvalidRhsSpecError(
            detail=f"coord_shift axis {axis_name!r} is not defined",
        )

    if not isinstance(arrow, str) or "->" not in arrow:
        raise InvalidRhsSpecError(
            detail=f"coord_shift[{axis_name}] must be 'from_coord -> to_coord'",
        )
    parts = [p.strip() for p in arrow.split("->")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise InvalidRhsSpecError(
            detail=f"coord_shift[{axis_name}] must be 'from_coord -> to_coord'",
        )
    from_coord, to_coord = parts
    valid_coords = axis_lookup[axis_name]
    for coord in (from_coord, to_coord):
        if coord not in valid_coords:
            raise InvalidRhsSpecError(
                detail=(
                    f"coord_shift coordinate {coord!r} not in "
                    f"axis {axis_name!r} coords {valid_coords}"
                ),
            )

    apply_to = tr.get("apply_to")
    if not isinstance(apply_to, list) or not apply_to:
        raise InvalidRhsSpecError(
            detail="coord_shift requires a non-empty 'apply_to' list",
        )

    rate_s = tr.get("rate")
    if not isinstance(rate_s, str) or not rate_s.strip():
        raise InvalidRhsSpecError(detail="coord_shift requires a 'rate' string")

    return axis_name, from_coord, to_coord, apply_to, rate_s.strip()


def _apply_coord_shifts(
    *,
    transitions_raw: list[dict[str, Any]],
    state_expanded: list[str],
    axes: list[dict[str, Any]],
) -> None:
    """Expand ``coord_shift`` entries into concrete transitions in-place.

    Each ``coord_shift`` entry describes movement along one axis coordinate for
    a set of states.  The entry is replaced in-place by one concrete transition
    per state listed in ``apply_to``, per combination of the remaining axes.

    Args:
        transitions_raw: Mutable transition list — ``coord_shift`` entries are
            replaced by concrete transition dicts.
        state_expanded: Expanded state names (used to discover axes per state).
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

    Discovers which axes the base carries by inspecting ``state_expanded`` for
    names starting with ``base__``.  A transition is emitted for each
    combination that matches the shifted fragment.

    Returns:
        Concrete ``{"from", "to", "rate"}`` transition dicts.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    prefix = f"{base}__"
    matching = [s for s in state_expanded if s.startswith(prefix)]

    if not matching:
        raise InvalidRhsSpecError(
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
            raise InvalidRhsSpecError(
                detail=(
                    f"coord_shift would create transition to {target!r} "
                    f"which is not an expanded state"
                ),
            )

        concrete.append({"from": state_name, "to": target, "rate": rate_s})

    if not concrete:
        raise InvalidRhsSpecError(
            detail=(
                f"coord_shift apply_to state {base!r} has no expanded states "
                f"with fragment {from_frag!r}"
            ),
        )

    return concrete


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
    return normalize_expr_rhs(spec)  # unreachable; satisfies return type checker


def normalize_expr_rhs(spec: Mapping[str, Any]) -> NormalizedRhs:
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

    aliases, alias_template_map = _expand_alias_templates(
        meta_parts[0], axes=axes_meta, template_map_seed=state_template_map
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


# ---------------------------------------------------------------------------
# transitions kind helpers
# ---------------------------------------------------------------------------


def _apply_transition(
    *,
    idx: int,
    tr: Mapping[str, Any],
    state_set: set[str],
    all_syms: set[str],
    d_terms: dict[str, list[str]],
) -> None:
    """Apply a transition to the derivative-term accumulator.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    frm_s = _get_required_str(tr, idx=idx, key="from")
    to_s = _get_required_str(tr, idx=idx, key="to")
    rate_s = _get_required_str(tr, idx=idx, key="rate")

    if frm_s not in state_set:
        raise InvalidRhsSpecError(
            detail=f"transitions[{idx}].from={frm_s!r} not in state"
        )
    if to_s not in state_set:
        raise InvalidRhsSpecError(detail=f"transitions[{idx}].to={to_s!r} not in state")

    tree = _parse_expr(rate_s)
    all_syms |= _collect_names(tree)

    flow = f"({rate_s})*({frm_s})"
    d_terms[frm_s].append(f"-({flow})")
    d_terms[to_s].append(f"+({flow})")


def _build_transition_equations(
    state: list[str], d_terms: Mapping[str, list[str]]
) -> list[str]:
    """Build one equation string per state from accumulated derivative terms.

    Returns:
        List of equation strings (``"0.0"`` for states with no terms).
    """
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
        raise InvalidRhsSpecError(
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
