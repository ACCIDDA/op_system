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
import sys
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

from op_system._axes import (
    _compute_axis_deltas,
    _normalize_axes,
    _normalize_bracket_key,
)
from op_system._errors import InvalidRhsSpecError, UnsupportedFeatureError
from op_system._helpers import (
    _ensure_mapping,
    _ensure_str_dict,
    _ensure_str_list,
    _sorted_unique,
)
from op_system._ir import Expr, parse_expr_to_ir
from op_system._ir_templates import _detect_alias_cycle, inline_aliases
from op_system._symbols import _collect_names, _parse_expr
from op_system._templates import (
    _INLINE_TEMPLATE_RE,
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
    state_templates: tuple[StateTemplate, ...] = ()
    shaped_params: tuple[tuple[str, tuple[str, ...]], ...] = ()
    time_varying_params: tuple[tuple[str, tuple[str, ...]], ...] = ()
    aliases_ir: Mapping[str, Expr] = field(default_factory=dict)
    equations_ir: tuple[Expr | None, ...] = ()
    equations_ir_raw: tuple[Expr | None, ...] = ()
    # Helper-bearing IR variants: aliases/equations parsed from their
    # *pre-expansion* string form (i.e. before ``_expand_helpers`` lowers
    # ``apply_along`` / ``sum_over`` / ``integrate_over`` to per-cell
    # sums) with ``parse_expr_to_ir(lower_helpers=True)``. Carries
    # ``Reduce`` nodes; intended for downstream IR-fast-path consumers.
    # Best-effort, parallel to ``aliases_ir`` / ``equations_ir`` — entries
    # that fail to parse or inline are omitted (aliases) or stored as
    # ``None`` (equations). Empty for ``kind == "transitions"`` (rates are
    # not yet plumbed through this surface).
    aliases_ir_reduce: Mapping[str, Expr] = field(default_factory=dict)
    equations_ir_reduce: tuple[Expr | None, ...] = ()


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

    A shaped parameter reference is a token of the form ``name[ax1,ax2,...]``
    where ``name`` is not in ``name_blocklist`` (states, aliases, recognized
    built-ins) and every bracket entry is either:
    - A bare axis name (e.g., ``age``) registered in ``axis_lookup``
    - An axis binding (e.g., ``age:ap`` or legacy ``age=ap``) where the
      axis name is registered

    Each such ``name`` is recorded as shaped over the tuple of axes appearing
    in its first occurrence; subsequent occurrences must use the same axes in
    the same order. Duplicate axes are allowed (e.g., ``K[age, age]`` for a
    contact kernel).

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
            # For each part, extract the axis name (before "=" if present)
            axes: list[str] = []
            valid = True
            for p in parts:
                if not p:
                    valid = False
                    break
                # Handle bare axes (e.g., "age"), legacy bindings
                # (``age=ap``), and the canonical slice-form bindings
                # (``age:ap``).
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

    Time-varying parameters are now declared implicitly: any shaped reference
    such as ``beta[time, age]`` (where ``time`` matches the configured
    ``time_axis``) is interpolated at runtime.

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

    A shaped parameter is "time-varying" when the configured time axis
    appears in its axes tuple.  The returned ``shaped_reduced`` mapping
    drops the time axis from each tv parameter so downstream substitution
    treats it as a shape over the remaining (non-time) axes; the
    ``time_varying_full`` mapping retains the full axes tuple (with time)
    for the parameter request and the runtime interpolation wrapper.

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

    Only matches bracket templates whose base appears in ``tv_full_axes``
    and whose bracket entries equal the registered full axes tuple.

    Returns:
        The rewritten expression.
    """
    if not tv_full_axes or not expr:
        return expr

    def _rewrite(match: re.Match[str]) -> str:
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


def _build_state_templates(
    state_raw: list[str],
    *,
    axes: list[dict[str, Any]],
    state_template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
    state_expanded: list[str],
) -> tuple[StateTemplate, ...]:
    """Build per-template structural records aligned with `state_expanded`.

    Walks the user-declared `state_raw` in order. For each entry, looks up
    the template in `state_template_map` (if present) or treats it as a
    scalar. Records the offset of each template's first expanded name in
    the flat `state_expanded` list so consumers can recover contiguous
    per-template slices of the state vector.

    Returns:
        Tuple of `StateTemplate`, one per entry in `state_raw`, in order.

    Notes:
        Pinned-only templates (e.g. `S[vax=u]`) are reported as scalar
        templates because they expand to exactly one cell each. Only
        wildcard templates carry shape information used by vectorized
        evaluation.
    """
    axis_size = {ax["name"]: len(ax.get("coords", ())) for ax in axes}
    name_to_idx = {n: i for i, n in enumerate(state_expanded)}
    templates: list[StateTemplate] = []
    for entry in state_raw:
        base, tokens = parse_selector(entry)
        wildcards = [t for t in tokens if isinstance(t, WildcardToken)]
        pinned = [t for t in tokens if isinstance(t, PinnedToken)]
        if wildcards:
            # Canonical key matches what `_expand_state_templates` builds.
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


def _expand_alias_templates(  # noqa: PLR0913
    aliases_raw: Mapping[str, str],
    *,
    axes: list[dict[str, Any]],
    template_map_seed: Mapping[str, list[tuple[str, dict[str, str]]]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    axis_lookup: Mapping[str, list[str]] | None = None,
    passthrough_helpers: bool = False,
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

        if canonical_name in alias_template_map:
            # Defer ``_expand_helpers`` until the per-row LHS assignment is
            # known so that same-axis-twice references like
            # ``K[age, age:ap]`` inside a templated alias (e.g. ``foi[age]``)
            # resolve the bare LHS-free position from the row assignment
            # rather than collapsing onto the bound ``apply_along`` coord.
            # Mirrors the equations path in ``_resolve_template_equation``.
            for expanded_name, assignment in alias_template_map[canonical_name]:
                expanded_expr = _expand_helpers(
                    expr_s,
                    axes=axes,
                    shaped_params=shaped_params,
                    lhs_assignment=assignment,
                    axis_coords=axis_lookup,
                    passthrough=passthrough_helpers,
                )
                substituted = _apply_template_substitutions(
                    expanded_expr,
                    assignment=assignment,
                    template_map=combined_template_map,
                    shaped_params=shaped_params,
                    axis_lookup=axis_lookup,
                )
                aliases_out[expanded_name] = substituted
        else:
            aliases_out[raw_name] = _expand_helpers(
                expr_s,
                axes=axes,
                shaped_params=shaped_params,
                passthrough=passthrough_helpers,
            )

    return aliases_out, alias_template_map


def _build_aliases_ir(
    aliases: Mapping[str, str],
    *,
    lower_helpers: bool = False,
) -> dict[str, Expr]:
    """Parse each alias body to IR and inline alias-to-alias references.

    Best-effort: when parsing or inlining fails (cyclic aliases, AST recursion
    on very long expanded chains, unsupported syntax), the failing alias is
    omitted rather than aborting normalization. Existing string-based consumers
    rely on lazy cycle detection at evaluation time, so this preserves
    backward-compatible behaviour while still exposing IR where available.

    Args:
        aliases: Mapping from alias name to body string.
        lower_helpers: When ``True``, parse with ``lower_helpers=True`` so
            helper-call ``Apply`` nodes (``apply_along``, ``sum_over``,
            ``integrate_over``) become :class:`Reduce` nodes. Use with a
            pre-expansion ``aliases`` mapping to expose structured
            reduction IR to downstream consumers.

    Returns:
        Mapping from alias name to its (fully inlined) IR expression. Entries
        that could not be parsed or inlined are omitted.
    """
    # Long expanded Add chains (e.g. ``apply_along`` over many coords) can
    # exceed Python's default recursion limit during AST descent / inlining.
    old_limit = sys.getrecursionlimit()
    needed = max(old_limit, 10_000)
    try:
        if needed > old_limit:
            sys.setrecursionlimit(needed)
        parsed: dict[str, Expr] = {}
        for name, body in aliases.items():
            try:
                expr = parse_expr_to_ir(body, lower_helpers=lower_helpers)
            except (ValueError, RecursionError):
                continue
            parsed[name] = expr
        # Validate the alias graph once and share a free_symbols memo across
        # all per-alias inline_aliases calls: alias bodies are reused by
        # identity, so id-keyed caching turns the inner traversal cost from
        # O(#aliases * body_size) into O(body_size).
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

    Best-effort: entries that fail to parse or inline are returned as ``None``
    so positional alignment with ``equations`` is preserved. Mirrors the
    fallback policy of :func:`_build_aliases_ir` to keep this slice purely
    additive — existing string-based consumers remain authoritative.

    Args:
        equations: Tuple of equation RHS strings.
        aliases_ir: Optional alias IR map to inline references against.
        lower_helpers: When ``True``, parse with ``lower_helpers=True`` so
            helper-call ``Apply`` nodes become :class:`Reduce` nodes. Use
            with pre-expansion equation strings.

    Returns:
        Tuple of IR expressions (or ``None`` for failed entries) aligned
        positionally with ``equations``.
    """
    old_limit = sys.getrecursionlimit()
    needed = max(old_limit, 10_000)
    try:
        if needed > old_limit:
            sys.setrecursionlimit(needed)
        # Share one free_symbols memo across every equation and validate the
        # alias cycle once: alias bodies are reused by identity, so id-keyed
        # caching turns the per-equation inline cost from O(alias_size) into
        # O(equation_size) after the first call warms the memo.
        memo: dict[int, frozenset[str]] = {}
        cycle_validated = False
        if aliases_ir:
            try:
                cycle = _detect_alias_cycle(aliases_ir)
                cycle_validated = cycle is None
            except (ValueError, RecursionError):
                cycle_validated = False
        out: list[Expr | None] = []
        for eq in equations:
            try:
                expr = parse_expr_to_ir(eq, lower_helpers=lower_helpers)
            except (ValueError, RecursionError):
                out.append(None)
                continue
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


def _build_equations_ir_via_pointwise(  # noqa: PLR0913
    *,
    equations_ir_reduce: tuple[Expr | None, ...],
    aliases_ir_reduce: Mapping[str, Expr],
    state_expanded: Sequence[str],
    template_map: Mapping[str, Sequence[tuple[str, Mapping[str, str]]]],
    axes: Sequence[Mapping[str, Any]],
    shaped_params: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, Sequence[str]],
) -> tuple[Expr | None, ...]:
    """Build pointwise equation IR by IR-side expansion of ``Reduce`` nodes.

    For each state cell, inline alias references (resolved against the
    pre-expansion ``aliases_ir_reduce`` map so nested ``Reduce`` nodes are
    preserved) and then call :func:`expand_reduce_pointwise` with the cell's
    LHS template assignment so that same-axis-twice references resolve
    correctly. Avoids the string roundtrip that :func:`_build_equations_ir`
    performs against ``_expand_apply_along`` output.

    Best-effort: cells whose IR cannot be inlined or expanded are returned
    as ``None`` to preserve positional alignment with ``state_expanded``.

    Returns:
        Tuple of expanded pointwise IR (or ``None`` for failed entries)
        aligned positionally with ``state_expanded``.
    """
    from op_system._ir_expand import expand_reduce_pointwise  # noqa: PLC0415

    if not equations_ir_reduce:
        return ()
    cell_to_assignment: dict[str, Mapping[str, str]] = {}
    for variants in template_map.values():
        cell_to_assignment.update(dict(variants))

    memo: dict[int, frozenset[str]] = {}
    try:
        cycle_validated = _detect_alias_cycle(aliases_ir_reduce) is None
    except (ValueError, RecursionError):
        cycle_validated = False

    out: list[Expr | None] = []
    old_limit = sys.getrecursionlimit()
    needed = max(old_limit, 10_000)
    try:
        if needed > old_limit:
            sys.setrecursionlimit(needed)
        for cell, ir in zip(state_expanded, equations_ir_reduce, strict=False):
            if ir is None:
                out.append(None)
                continue
            try:
                inlined = inline_aliases(
                    ir,
                    aliases_ir_reduce,
                    memo=memo,
                    skip_cycle_check=cycle_validated,
                )
            except (ValueError, RecursionError):
                inlined = ir
            assignment = cell_to_assignment.get(cell, {})
            try:
                expanded = expand_reduce_pointwise(
                    inlined,
                    axes=axes,
                    shaped_params=shaped_params,
                    lhs_assignment=assignment,
                    axis_coords=axis_lookup,
                )
            except (ValueError, RecursionError, KeyError):
                expanded = None
            out.append(expanded)
        return tuple(out)
    finally:
        if needed > old_limit:
            sys.setrecursionlimit(old_limit)


_INITIAL_STATE_SHAPED_KEY = "shaped"
_INITIAL_STATE_SHAPED_AXES_KEY = "axes"
_INITIAL_STATE_SHAPED_ALLOWED_KEYS = frozenset((
    _INITIAL_STATE_SHAPED_KEY,
    _INITIAL_STATE_SHAPED_AXES_KEY,
))


def _normalize_shaped_initial_state_value(
    raw_val: Mapping[str, Any],
    *,
    raw_key: str,
    axis_lookup: dict[str, list[str]],
) -> tuple[str, tuple[str, ...]]:
    """Validate a shaped-initial-state value mapping.

    A shaped IC entry looks like::

        X[age, vax, loc, imm]:
          shaped: x_init
          axes: [age, vax, loc, imm]

    The ``shaped`` field names a single parameter that is shaped over
    ``axes``; downstream provider/engine code is responsible for emitting
    one ParameterRequest per ``(name, axes)`` pair and indexing per cell
    by the assignment of the LHS wildcards.

    Returns:
        ``(name, axes_tuple)`` where ``axes_tuple`` preserves the user's
        declared axis order (no implicit reordering — engine plugins may
        choose to transpose the resolved value into the LHS order).

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    extra_keys = set(raw_val.keys()) - _INITIAL_STATE_SHAPED_ALLOWED_KEYS
    if extra_keys:
        raise InvalidRhsSpecError(
            detail=(
                f"initial_state[{raw_key!r}] shaped entry has unknown "
                f"keys {sorted(extra_keys)!r}; allowed keys are "
                f"{sorted(_INITIAL_STATE_SHAPED_ALLOWED_KEYS)!r}"
            ),
        )
    name_obj = raw_val.get(_INITIAL_STATE_SHAPED_KEY)
    if not isinstance(name_obj, str) or not name_obj.strip():
        raise InvalidRhsSpecError(
            detail=(
                f"initial_state[{raw_key!r}] shaped entry must set "
                f"{_INITIAL_STATE_SHAPED_KEY!r} to a non-empty string"
            ),
        )
    name = name_obj.strip()
    if not name.isidentifier():
        raise InvalidRhsSpecError(
            detail=(
                f"initial_state[{raw_key!r}] shaped name {name!r} is not "
                "a valid identifier"
            ),
        )
    axes_obj = raw_val.get(_INITIAL_STATE_SHAPED_AXES_KEY)
    if not isinstance(axes_obj, (list, tuple)) or not axes_obj:
        raise InvalidRhsSpecError(
            detail=(
                f"initial_state[{raw_key!r}] shaped entry must set "
                f"{_INITIAL_STATE_SHAPED_AXES_KEY!r} to a non-empty list "
                "of axis names"
            ),
        )
    axes_list: list[str] = []
    seen: set[str] = set()
    for ax in axes_obj:
        if not isinstance(ax, str) or not ax.strip():
            raise InvalidRhsSpecError(
                detail=(
                    f"initial_state[{raw_key!r}] shaped axes must be non-empty strings"
                ),
            )
        ax_s = ax.strip()
        if ax_s in seen:
            raise InvalidRhsSpecError(
                detail=(
                    f"initial_state[{raw_key!r}] shaped axes contain duplicate {ax_s!r}"
                ),
            )
        if ax_s not in axis_lookup:
            raise InvalidRhsSpecError(
                detail=(
                    f"initial_state[{raw_key!r}] shaped axis {ax_s!r} not "
                    "defined in spec axes"
                ),
            )
        seen.add(ax_s)
        axes_list.append(ax_s)
    return name, tuple(axes_list)


def _expand_initial_state_templates(
    initial_state_raw: Mapping[str, Any] | None,
    *,
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> dict[str, Any] | None:
    """Expand a templated initial_state mapping into concrete state→entry pairs.

    Supports wildcard selectors (``X[age, vax]``), pinned selectors
    (``X[age, vax, imm=X0]``), and bare state names on the key side.

    Each value is one of:

    * A non-empty string — the legacy *scalar* form.  Per-cell template
      substitution (``S0[vax]`` → ``S0__vax_v1``) still applies, so the
      expanded entry is a single parameter name string.
    * A mapping with keys ``shaped`` (parameter identifier) and ``axes``
      (list of axis names) — the new *shaped* form.  Every expanded state
      cell is stored as a dict ``{"shaped": name, "axes": (...,),
      "coords": {axis: coord, ...}}`` where ``coords`` selects which cell
      of the shaped parameter fills this state.  All shaped axes must be
      LHS wildcards (or pinned coords) so that every expansion has a
      coordinate for each axis.

    Returns:
        Expanded ``dict[str, str | dict[str, Any]]`` mapping, or ``None``
        if *initial_state_raw* is ``None``.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if initial_state_raw is None:
        return None

    axis_lookup = build_axis_lookup(axes)
    result: dict[str, Any] = {}
    expanded_keys: list[str] = []

    for raw_key, raw_val in initial_state_raw.items():
        if isinstance(raw_val, _MappingABC):
            shaped_name, shaped_axes = _normalize_shaped_initial_state_value(
                raw_val,
                raw_key=raw_key,
                axis_lookup=axis_lookup,
            )
            results = expand_selector(
                raw_key,
                axis_lookup=axis_lookup,
                context=f"initial_state key {raw_key!r}",
            )
            for expanded_key, assignment in results:
                missing = [ax for ax in shaped_axes if ax not in assignment]
                if missing:
                    raise InvalidRhsSpecError(
                        detail=(
                            f"initial_state[{raw_key!r}] shaped axes "
                            f"{missing!r} are not bound by the LHS "
                            "selector (each shaped axis must appear as a "
                            "wildcard or pinned coord on the key)"
                        ),
                    )
                expanded_keys.append(expanded_key)
                result[expanded_key] = {
                    "shaped": shaped_name,
                    "axes": shaped_axes,
                    "coords": {ax: assignment[ax] for ax in shaped_axes},
                }
            continue

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
    initial_state_raw: Mapping[str, Any] | None,
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
    shaped_params: Mapping[str, tuple[str, ...]] | None = None


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
    skip = set(endpoints.shaped_params or {})
    expr_placeholders = _extract_placeholders_from_expr(
        endpoints.rate_s, shaped_param_names=skip
    )
    if endpoints.name_s:
        expr_placeholders |= _extract_placeholders_from_expr(
            endpoints.name_s, shaped_param_names=skip
        )
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
        endpoints.rate_s,
        assignment=assignment,
        template_map=endpoints.template_map,
        shaped_params=endpoints.shaped_params,
        axis_lookup=endpoints.axis_lookup,
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
    tr_out["rate"] = _expand_helpers(
        rate_sub, axes=endpoints.axes, shaped_params=endpoints.shaped_params
    )
    if endpoints.name_s:
        tr_out["name"] = _apply_template_substitutions(
            endpoints.name_s,
            assignment=assignment,
            template_map=endpoints.template_map,
            shaped_params=endpoints.shaped_params,
            axis_lookup=endpoints.axis_lookup,
        )
    return tr_out


def _expand_single_transition(
    tr_map: Mapping[str, Any],
    *,
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
    axis_lookup: dict[str, list[str]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
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
        shaped_params=shaped_params,
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
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
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
                shaped_params=shaped_params,
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
_AXIS_FILTER_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s+in\s+\[([^\[\]]*)\]\s*$",
    re.IGNORECASE | re.DOTALL,
)
_APPLY_ALONG_KERNELS = frozenset({"sum", "integrate"})


def _parse_apply_along_args(  # noqa: C901, PLR0912
    args_str: str,
    *,
    axis_names: set[str],
) -> tuple[list[tuple[str, str, list[str] | None]], str | None, str]:
    """Parse the argument list of an ``apply_along(...)`` call.

    Args:
        args_str: The raw text between the call's parentheses.
        axis_names: Known axis names (used to distinguish axis bindings
            from the inner expression).

    Returns:
        Tuple ``(axis_bindings, kernel_form, inner_expr)`` where
        ``axis_bindings`` is a list of ``(axis_name, var_name, filter)``
        triples in declaration order.  ``filter`` is the optional list of
        coord names parsed from ``axis=var in [c1, c2, ...]``, or ``None``
        if the binding takes the full axis.  ``kernel_form`` is ``"sum"``,
        ``"integrate"`` or ``None`` (meaning auto-select from axis types),
        and ``inner_expr`` is the bound inner expression.

    Raises:
        InvalidRhsSpecError: If the argument list is malformed (no args,
            empty arg, duplicate ``kernel=``, unknown kernel name, missing
            axis bindings, non-identifier var name, empty filter list, or
            wrong number of inner expressions).
    """
    parts = _split_top_level_commas(args_str)
    if not parts or all(not p for p in parts):
        raise InvalidRhsSpecError(detail="apply_along(...) requires arguments")

    bindings: list[tuple[str, str, list[str] | None]] = []
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
            rhs = bm.group(2).strip()
            fm = _AXIS_FILTER_RE.match(rhs)
            if fm is not None:
                var_name = fm.group(1).strip()
                filter_coords = [c.strip() for c in fm.group(2).split(",") if c.strip()]
                if not filter_coords:
                    raise InvalidRhsSpecError(
                        detail=(
                            f"apply_along(...) axis filter {ax_name}={var_name} "
                            "in [...] requires at least one coord"
                        )
                    )
            else:
                var_name = rhs
                filter_coords = None
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var_name):
                raise InvalidRhsSpecError(
                    detail=(
                        f"apply_along(...) axis binding {ax_name}={var_name!r} "
                        "must bind to an identifier"
                    )
                )
            bindings.append((ax_name, var_name, filter_coords))
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
    bindings: list[tuple[str, str, list[str] | None]],
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
    types = {axis_types.get(ax) for ax, _, _ in bindings}
    sum_compatible = {"categorical", "ordinal"}
    if kernel_form == "sum":
        bad = [ax for ax, _, _ in bindings if axis_types.get(ax) not in sum_compatible]
        if bad:
            raise InvalidRhsSpecError(
                detail=(
                    f"apply_along(kernel=sum) requires categorical or ordinal axes; "
                    f"got incompatible: {bad}"
                )
            )
        return "sum"
    if kernel_form == "integrate":
        bad = [ax for ax, _, _ in bindings if axis_types.get(ax) != "continuous"]
        if bad:
            raise InvalidRhsSpecError(
                detail=(
                    f"apply_along(kernel=integrate) requires continuous axes; "
                    f"got non-continuous: {bad}"
                )
            )
        return "integrate"
    if types and types.issubset(sum_compatible):
        return "sum"
    if types == {"continuous"}:
        return "integrate"
    raise InvalidRhsSpecError(
        detail=(
            "apply_along(...) cannot infer kernel for mixed/unknown axis types "
            f"({sorted(t for t in types if t)}); pass kernel=sum or kernel=integrate"
        )
    )


def _substitute_apply_along_brackets(  # noqa: C901, PLR0913, PLR0915
    expr: str,
    bound: Mapping[str, str],
    *,
    axis_order: Sequence[str] = (),
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    lhs_assignment: Mapping[str, str] | None = None,
    axis_coords: Mapping[str, list[str]] | None = None,
) -> str:
    """Rewrite ``name[ax=c, ...]`` to ``name__ax_<c>__...`` for bound axes.

    For each ``name[...]`` subexpression, axis assignments whose axis is in
    ``bound`` and whose coord matches ``bound[axis]`` are consumed into a
    ``__axis_<sanitized-coord>`` suffix on ``name``.  Remaining assignments
    stay inside the brackets.  The bracket itself is dropped if every
    assignment is consumed.

    When ``axis_order`` is provided (the canonical ordering taken from the
    spec's ``axes:`` list), any pre-existing ``__<axis>_<coord>`` suffix on
    ``name`` is parsed off, merged with the newly-consumed pairs, and the
    combined suffix is re-emitted in canonical order.  This makes nested
    ``apply_along`` calls (where the inner pass appends a partial suffix and
    the outer pass appends more) yield the same canonical state name as a
    direct expansion, e.g. ``X__age_..__vax_..__loc_..__imm_..`` rather than
    the binding-order accident ``X__imm_..__age_..__vax_..__loc_..``.

    Same-axis-twice disambiguation
    ------------------------------
    For a shaped parameter declared on duplicate axes (e.g. a contact kernel
    ``K[age, age]``), the natural reference inside an ``apply_along`` is
    ``K[age, age:ap]`` where the **bare** ``age`` token is the *free outer*
    axis (taken from the LHS template assignment of the equation being
    expanded) and the ``age:ap`` token is the *bound inner* axis (assigned
    by the surrounding ``apply_along``).  ``lhs_assignment`` carries the
    LHS-free axis bindings so this case can be lowered to per-row
    indexing/mangling rather than collapsing both positions onto the bound
    coord (which would produce only the diagonal of ``K``).

    If ``shaped_params`` is provided and contains ``name`` with multiple axes
    (e.g., a contact kernel ``K[age, age]``), and all those axes are bound,
    the bracket assignments are converted to direct indexing instead of
    name mangling. For example, ``K[age, age:ap]`` with bound ``age=c1`` is
    converted to ``K[c1, ap]`` rather than ``K__age_<c1>[age:ap]``.

    Returns:
        Expression string with bound bracket assignments folded into
        ``__axis_coord`` suffixes or indexing operations.
    """
    pat = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[([^\[\]]*)\]")
    priority = {ax: i for i, ax in enumerate(axis_order)}
    shaped_params = shaped_params or {}
    lhs_assignment = lhs_assignment or {}
    axis_coords = axis_coords or {}

    def _split_suffix(name: str) -> tuple[str, list[tuple[str, str]]]:
        """Strip trailing ``__<axis>_<coord>`` tokens recognised by axis_order.

        Walks ``__``-separated segments from the right, consuming each one
        whose ``<axis>_`` prefix matches a known axis.  Stops at the first
        non-matching segment; everything to its left is treated as the base
        name.

        Returns:
            ``(base, [(axis, coord), ...])`` where the suffix pairs are in
            left-to-right (original) order.
        """
        if not priority:
            return name, []
        parts = name.split("__")
        if len(parts) <= 1:
            return name, []
        suffix_pairs: list[tuple[str, str]] = []
        cut = len(parts)
        for i in range(len(parts) - 1, 0, -1):
            tok = parts[i]
            # Longest matching axis name keeps the parse unambiguous when
            # axis names share a prefix.
            best: str | None = None
            for ax in priority:
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

    def _emit_suffix(pairs: list[tuple[str, str]]) -> str:
        if not priority:
            return "".join(f"__{ax}_{_sanitize_fragment(c)}" for ax, c in pairs)
        # Sort by canonical priority; pairs whose axis is unknown to priority
        # (shouldn't occur for well-formed specs) preserve their original
        # order at the end.
        known = sorted(
            (p for p in pairs if p[0] in priority),
            key=lambda p: priority[p[0]],
        )
        unknown = [p for p in pairs if p[0] not in priority]
        return "".join(f"__{ax}_{_sanitize_fragment(c)}" for ax, c in known + unknown)

    def _sub(match: re.Match[str]) -> str:  # noqa: C901, PLR0914
        name = match.group(1)
        body = match.group(2)
        entries = [e.strip() for e in body.split(",") if e.strip()]

        def _entry_parts(e: str) -> tuple[str, str | None]:
            """Return (axis, coord) for one bracket entry.

            Accepts both the canonical slice form ``axis:coord`` (valid
            Python) and the legacy ``axis=coord`` form used by some
            substitution outputs. Returns ``coord=None`` for bare
            wildcard tokens.
            """
            for sep in (":", "="):
                if sep in e:
                    ax, _, c = e.partition(sep)
                    return ax.strip(), c.strip()
            return e, None

        parsed_entries = [_entry_parts(e) for e in entries]

        # Same-axis-twice disambiguation: detect axes that appear in this
        # bracket both as a bare token and as ``axis:coord``. The bare token
        # then refers to the LHS-free axis (taken from ``lhs_assignment``);
        # the ``axis:coord`` token refers to the bound apply_along axis.
        bare_axes = {ax for ax, c in parsed_entries if c is None}
        pinned_axes = {ax for ax, c in parsed_entries if c is not None}
        same_axis_twice = bare_axes & pinned_axes

        # Resolve each entry positionally to (axis_name, coord, source) where
        # source is "bound", "lhs", or None (unresolved → kept in brackets).
        resolved: list[tuple[str, str | None, str | None]] = []
        consumed: list[tuple[str, str]] = []
        remaining: list[str] = []
        for entry, (ax, coord) in zip(entries, parsed_entries, strict=True):
            if coord is not None:
                if ax in bound and bound[ax] == coord:
                    consumed.append((ax, coord))
                    resolved.append((ax, coord, "bound"))
                    continue
                resolved.append((ax, None, None))
                remaining.append(entry)
                continue
            # Bare axis token.
            if entry in same_axis_twice:
                # Free outer axis — must be supplied by the LHS template
                # assignment. If we don't have it (e.g., a non-templated
                # equation), leave the token in the brackets for downstream
                # template substitution.
                if entry in lhs_assignment:
                    lhs_coord = lhs_assignment[entry]
                    consumed.append((entry, lhs_coord))
                    resolved.append((entry, lhs_coord, "lhs"))
                    continue
                resolved.append((entry, None, None))
                remaining.append(entry)
                continue
            if entry in bound:
                # Plain bound axis (no same-axis-twice ambiguity).
                consumed.append((entry, bound[entry]))
                resolved.append((entry, bound[entry], "bound"))
                continue
            resolved.append((entry, None, None))
            remaining.append(entry)

        if not consumed:
            return match.group(0)

        # Multi-axis shaped parameter → direct indexing when every position
        # is resolved. Use positional coords from ``resolved`` so that
        # same-axis-twice references like ``K[age, age:ap]`` get distinct
        # row/column indices instead of collapsing onto the bound coord.
        if name in shaped_params and len(shaped_params[name]) > 1:
            param_axes = shaped_params[name]
            if (
                len(resolved) == len(param_axes)
                and not remaining
                and all(coord is not None for _, coord, _ in resolved)
                and all(ax == param_axes[i] for i, (ax, _, _) in enumerate(resolved))
                and all(ax in axis_coords for ax in param_axes)
            ):
                # Convert coord names to integer positions on each axis so
                # the downstream compiler sees a literal subscript.
                try:
                    int_idx = [
                        axis_coords[param_axes[i]].index(coord)  # type: ignore[arg-type]
                        for i, (_, coord, _) in enumerate(resolved)
                    ]
                except ValueError:
                    int_idx = None
                if int_idx is not None:
                    indices = ", ".join(str(i) for i in int_idx)
                    return f"{name}[{indices}]"

        base, existing = _split_suffix(name)
        suffix = _emit_suffix(existing + consumed)
        if remaining:
            return f"{base}{suffix}[{', '.join(remaining)}]"
        return f"{base}{suffix}"

    return pat.sub(_sub, expr)


def _build_apply_along_axis_options(
    ax_name: str,
    filt: list[str] | None,
    *,
    kernel: str,
    axis: dict[str, Any],
) -> list[tuple[str, float]]:
    """Build the per-axis ``(coord, weight)`` list for one apply_along binding.

    Filter semantics depend on axis type:

    - **Categorical**: ``filt`` is the explicit list of coord names to retain.
      Unknown coords are rejected.
    - **Ordinal**: ``filt`` must be exactly ``[lo_label, hi_label]`` -- two
      coord names interpreted as the inclusive index range from
      ``index(lo_label)`` to ``index(hi_label)`` along the declared axis
      order.  All coords in that contiguous slice are retained with weight 1.
    - **Continuous**: ``filt`` is interpreted as the closed sub-interval
      endpoints ``[lo, hi]`` (must be exactly two numeric values).  All axis
      coords ``c`` with ``lo <= c <= hi`` are retained, and trapezoidal
      weights are recomputed for that sub-interval.

    Args:
        ax_name: Name of the bound axis.
        filt: Optional ``in [...]`` filter; ``None`` for full axis.
        kernel: ``"sum"`` or ``"integrate"``.
        axis: The axis dict from the normalized spec.

    Returns:
        List of ``(coord_str, weight)`` pairs to be combined under ``product``.

    Raises:
        InvalidRhsSpecError: If the axis has no coords; if a categorical
            filter lists an unknown coord; if an ordinal filter is not a
            2-element ``[lo_label, hi_label]`` pair, names an unknown coord,
            or has ``index(lo) > index(hi)``; if a continuous-axis filter is
            not a 2-element ``[lo, hi]`` pair, has ``lo > hi``, or selects no
            axis coords; or if a continuous axis is missing trapezoidal
            ``deltas``.
    """
    coords = [str(c) for c in axis.get("coords", [])]
    if not coords:
        raise InvalidRhsSpecError(detail=f"apply_along axis {ax_name!r} has no coords")
    ax_type = str(axis.get("type", ""))
    is_continuous = ax_type == "continuous"
    is_ordinal = ax_type == "ordinal"
    if kernel != "integrate":
        if filt is None:
            return [(c, 1.0) for c in coords]
        if is_continuous:
            sub_coords = _continuous_subinterval_coords(ax_name, coords, filt)
            return [(c, 1.0) for c in sub_coords]
        if is_ordinal:
            sub_coords = _ordinal_subrange_coords(ax_name, coords, filt)
            return [(c, 1.0) for c in sub_coords]
        unknown = [c for c in filt if c not in coords]
        if unknown:
            raise InvalidRhsSpecError(
                detail=(
                    f"apply_along axis {ax_name!r} filter has unknown coords: {unknown}"
                )
            )
        return [(c, 1.0) for c in filt]
    deltas = axis.get("deltas") or []
    if len(deltas) != len(coords):
        raise InvalidRhsSpecError(
            detail=f"apply_along axis {ax_name!r} missing or mismatched deltas"
        )
    if filt is None:
        return [(c, float(d)) for c, d in zip(coords, deltas, strict=True)]
    sub_coords = _continuous_subinterval_coords(ax_name, coords, filt)
    sub_floats = [float(c) for c in sub_coords]
    sub_deltas = _compute_axis_deltas(sub_floats, idx=0)
    return list(zip(sub_coords, sub_deltas, strict=True))


def _continuous_subinterval_coords(
    ax_name: str, coords: list[str], filt: list[str]
) -> list[str]:
    """Resolve a continuous-axis ``in [lo, hi]`` filter to retained coord strings.

    Args:
        ax_name: Name of the bound axis (used in error messages).
        coords: All axis coords as strings, in axis order.
        filt: Filter list parsed from the spec; must contain exactly two
            numeric values ``[lo, hi]`` with ``lo <= hi``.

    Returns:
        List of coord strings (in axis order) for which ``lo <= c <= hi``.

    Raises:
        InvalidRhsSpecError: If ``filt`` is not a 2-element list, the
            endpoints are not numeric, ``lo > hi``, or no axis coord falls in
            the closed interval ``[lo, hi]``.
    """
    if len(filt) != 2:
        raise InvalidRhsSpecError(
            detail=(
                f"apply_along axis {ax_name!r} continuous filter must be a "
                f"2-element [lo, hi] interval (got {len(filt)} entries)"
            )
        )
    try:
        lo = float(filt[0])
        hi = float(filt[1])
    except ValueError as exc:
        raise InvalidRhsSpecError(
            detail=(
                f"apply_along axis {ax_name!r} continuous filter endpoints must "
                f"be numeric (got {filt!r})"
            )
        ) from exc
    if lo > hi:
        raise InvalidRhsSpecError(
            detail=(
                f"apply_along axis {ax_name!r} continuous filter endpoints must "
                f"satisfy lo <= hi (got [{lo}, {hi}])"
            )
        )
    sub = [c for c in coords if lo <= float(c) <= hi]
    if not sub:
        raise InvalidRhsSpecError(
            detail=(
                f"apply_along axis {ax_name!r} continuous filter [{lo}, {hi}] "
                "selects no axis coords"
            )
        )
    return sub


def _ordinal_subrange_coords(
    ax_name: str, coords: list[str], filt: list[str]
) -> list[str]:
    """Resolve an ordinal-axis ``in [lo_label, hi_label]`` filter to retained coords.

    The two filter entries must be coord labels declared on the axis.  The
    inclusive index range from ``index(lo_label)`` to ``index(hi_label)``
    along the declared axis order is returned.

    Args:
        ax_name: Name of the bound axis (used in error messages).
        coords: All axis coords as strings, in declared axis order.
        filt: Filter list parsed from the spec; must contain exactly two
            coord labels ``[lo_label, hi_label]`` with
            ``index(lo_label) <= index(hi_label)``.

    Returns:
        List of coord strings (in axis order) for the inclusive index range.

    Raises:
        InvalidRhsSpecError: If ``filt`` is not a 2-element list, an endpoint
            is not a declared coord label, or ``index(lo) > index(hi)``.
    """
    if len(filt) != 2:
        raise InvalidRhsSpecError(
            detail=(
                f"apply_along axis {ax_name!r} ordinal filter must be a "
                f"2-element [lo_label, hi_label] range (got {len(filt)} entries)"
            )
        )
    lo_label, hi_label = filt[0], filt[1]
    unknown = [c for c in (lo_label, hi_label) if c not in coords]
    if unknown:
        raise InvalidRhsSpecError(
            detail=(
                f"apply_along axis {ax_name!r} ordinal filter has unknown "
                f"coords: {unknown}"
            )
        )
    lo_idx = coords.index(lo_label)
    hi_idx = coords.index(hi_label)
    if lo_idx > hi_idx:
        raise InvalidRhsSpecError(
            detail=(
                f"apply_along axis {ax_name!r} ordinal filter endpoints must "
                f"satisfy index(lo) <= index(hi) (got [{lo_label!r}, {hi_label!r}] "
                f"at indices [{lo_idx}, {hi_idx}])"
            )
        )
    return coords[lo_idx : hi_idx + 1]


def _expand_apply_along(  # noqa: PLR0914
    expr: str,
    *,
    axes: list[dict[str, Any]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    lhs_assignment: Mapping[str, str] | None = None,
    axis_coords: Mapping[str, list[str]] | None = None,
) -> str:
    """Expand ``apply_along(...)`` calls directly to weighted sums.

    The ``apply_along`` primitive contracts an expression along one or more
    axes using a single shared kernel form.  Categorical axes use uniform
    weights of 1; continuous axes use trapezoidal weights derived from the
    axis ``deltas``.  All bound axes are substituted in a single Cartesian
    expansion, so ``name[ax1=c1, ax2=c2, ...]`` brackets that fully bind
    apply_along axes collapse to ``name__ax1_<c1>__ax2_<c2>``.

    A binding may restrict expansion to a subset of an axis with the
    ``axis=var in [c1, c2, ...]`` form.  For categorical axes the listed
    coords just filter the expansion set.  For continuous axes the listed
    coords must be a contiguous slice of the axis coords (in axis order),
    and trapezoidal weights are recomputed for that sub-interval.

    Returns:
        Expression string with ``apply_along`` calls fully expanded.
    """
    shaped_params = shaped_params or {}
    axis_lookup = {str(ax.get("name")): ax for ax in axes if ax.get("name")}
    axis_order = tuple(str(ax["name"]) for ax in axes if ax.get("name"))
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
        inner = _expand_apply_along(
            inner,
            axes=axes,
            shaped_params=shaped_params,
            lhs_assignment=lhs_assignment,
            axis_coords=axis_coords,
        )

        axis_options = [
            _build_apply_along_axis_options(
                ax_name, filt, kernel=kernel, axis=axis_lookup[ax_name]
            )
            for ax_name, _var, filt in bindings
        ]

        terms: list[str] = []
        for combo in product(*axis_options):
            replaced = inner
            bound: dict[str, str] = {}
            for (ax_name, var_name, _filt), (coord, _w) in zip(
                bindings, combo, strict=True
            ):
                replaced = re.sub(rf"\b{re.escape(var_name)}\b", coord, replaced)
                bound[ax_name] = coord
            replaced = _substitute_apply_along_brackets(
                replaced,
                bound,
                axis_order=axis_order,
                shaped_params=shaped_params,
                lhs_assignment=lhs_assignment,
                axis_coords=axis_coords,
            )
            if kernel == "integrate":
                weight = "*".join(str(w) for _, w in combo)
                terms.append(f"({weight})*({replaced})")
            else:
                terms.append(f"({replaced})")
        replacement = " + ".join(terms)
        out = out[:start] + f"({replacement})" + out[end:]


def _expand_helpers(  # noqa: PLR0913
    expr: str,
    *,
    axes: list[dict[str, Any]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    lhs_assignment: Mapping[str, str] | None = None,
    axis_coords: Mapping[str, list[str]] | None = None,
    passthrough: bool = False,
) -> str:
    """Expand helper calls (currently only ``apply_along``) before AST parsing.

    ``lhs_assignment`` carries the LHS template assignment of the equation
    being expanded (e.g. ``{"age": "a0"}`` for the row ``foi__age_a0`` of
    ``foi[age]``). It is forwarded to :func:`_substitute_apply_along_brackets`
    so that same-axis-twice references like ``K[age, age:ap]`` can resolve
    the bare (free outer) axis position from the LHS row rather than
    collapsing it onto the bound apply_along coord.

    ``axis_coords`` maps each axis name to its coord list and is used to
    rewrite multi-axis shaped-parameter references to literal integer
    subscripts (``K[2, 0]``) when every position is resolved.

    When ``passthrough`` is true, helpers are *not* expanded — the input
    is returned unchanged. Used by the second-pass IR build to capture
    pre-expansion strings for ``parse_expr_to_ir(lower_helpers=True)``.

    Returns:
        Expression string with helper calls expanded (or the input
        unchanged when ``passthrough`` is true).
    """
    if passthrough:
        return expr
    return _expand_apply_along(
        expr,
        axes=axes,
        shaped_params=shaped_params,
        lhs_assignment=lhs_assignment,
        axis_coords=axis_coords,
    )


def _resolve_template_equation(  # noqa: PLR0913
    *,
    name: str,
    equations_map: Mapping[str, Any],
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
    all_syms: set[str],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    axis_lookup: Mapping[str, list[str]] | None = None,
    passthrough_helpers: bool = False,
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
            expr_s = _expand_helpers(
                expr.strip(),
                axes=axes,
                shaped_params=shaped_params,
                lhs_assignment=assignment,
                axis_coords=axis_lookup,
                passthrough=passthrough_helpers,
            )
            expr_s = _apply_template_substitutions(
                expr_s,
                assignment=assignment,
                template_map=template_map,
                shaped_params=shaped_params,
                axis_lookup=axis_lookup,
            )
            if not passthrough_helpers:
                tree = _parse_expr(expr_s)
                all_syms |= _collect_names(tree)
            return expr_s
    return None


def _gather_equations(  # noqa: PLR0913
    state: list[str],
    equations_map: Mapping[str, Any],
    all_syms: set[str],
    *,
    axes: list[dict[str, Any]],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]] | None = None,
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    axis_lookup: Mapping[str, list[str]] | None = None,
    passthrough_helpers: bool = False,
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
            expr_s = _expand_helpers(
                expr.strip(),
                axes=axes,
                shaped_params=shaped_params,
                passthrough=passthrough_helpers,
            )
            if not passthrough_helpers:
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
            shaped_params=shaped_params,
            axis_lookup=axis_lookup,
            passthrough_helpers=passthrough_helpers,
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


def normalize_expr_rhs(spec: Mapping[str, Any]) -> NormalizedRhs:  # noqa: C901, PLR0914, PLR0915
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

    aliases, alias_template_map = _expand_alias_templates(
        meta_parts[0],
        axes=axes_meta,
        template_map_seed=state_template_map,
        shaped_params=shaped_params,
        axis_lookup=axis_lookup_dict,
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
        shaped_params=shaped_params,
        axis_lookup=axis_lookup_dict,
    )

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
    eqs_tuple = tuple(eqs)
    aliases_ir_map = _build_aliases_ir(aliases)

    # Second pass: capture pre-expansion (helper-bearing) strings so we can
    # build IR with ``Reduce`` nodes for downstream IR-fast-path consumers.
    # Best-effort: failures leave the reduce-bearing fields empty.
    aliases_ir_reduce_map: Mapping[str, Expr] = {}
    equations_ir_reduce: tuple[Expr | None, ...] = ()
    try:
        aliases_pre, _alias_template_map_pre = _expand_alias_templates(
            meta_parts[0],
            axes=axes_meta,
            template_map_seed=state_template_map,
            shaped_params=shaped_params,
            axis_lookup=axis_lookup_dict,
            passthrough_helpers=True,
        )
        eqs_pre = _gather_equations(
            state_expanded,
            equations_map,
            set(),  # throwaway: don't pollute all_syms with helper names
            axes=axes_meta,
            template_map=template_map_all,
            shaped_params=shaped_params,
            axis_lookup=axis_lookup_dict,
            passthrough_helpers=True,
        )
        aliases_ir_reduce_map = _build_aliases_ir(aliases_pre, lower_helpers=True)
        equations_ir_reduce = _build_equations_ir(
            tuple(eqs_pre), aliases_ir_reduce_map, lower_helpers=True
        )
    except (ValueError, RecursionError, InvalidRhsSpecError):
        aliases_ir_reduce_map = {}
        equations_ir_reduce = ()

    return NormalizedRhs(
        kind="expr",
        state_names=tuple(state_expanded),
        equations=eqs_tuple,
        aliases=aliases,
        param_names=_sorted_unique(
            sym
            for sym in all_syms
            if sym not in set(state_expanded)
            and sym not in aliases
            and sym not in shaped_set
            and sym not in time_varying_set
            and sym not in axis_name_set
            and sym not in template_base_set
            and sym not in _SHAPED_PARAM_BUILTIN_NAMES
        ),
        all_symbols=frozenset(all_syms | set(aliases.keys())),
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
        equations_ir=_build_equations_ir_via_pointwise(
            equations_ir_reduce=equations_ir_reduce,
            aliases_ir_reduce=aliases_ir_reduce_map,
            state_expanded=state_expanded,
            template_map=template_map_all,
            axes=axes_meta,
            shaped_params=shaped_params,
            axis_lookup=axis_lookup_dict,
        )
        or _build_equations_ir(eqs_tuple, aliases_ir_map),
        equations_ir_raw=_build_equations_ir(eqs_tuple),
        aliases_ir_reduce=aliases_ir_reduce_map,
        equations_ir_reduce=equations_ir_reduce,
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


def normalize_transitions_rhs(  # noqa: C901, PLR0912, PLR0914, PLR0915
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

    aliases, alias_template_map = _expand_alias_templates(
        meta_parts[0],
        axes=axes_meta,
        template_map_seed=state_template_map,
        shaped_params=shaped_params,
        axis_lookup=axis_lookup_dict,
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
        shaped_params=shaped_params,
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

    meta["shaped_params"] = tuple(sorted(shaped_params.items()))
    meta["time_axis"] = time_axis_name
    meta["time_varying_params"] = tuple(sorted(time_varying_full.items()))

    shaped_set = set(shaped_params)
    time_varying_set = set(time_varying_full)
    axis_name_set = set(axis_lookup_dict)
    template_base_set = {parse_selector(k)[0] for k in template_map_all}
    eqs_tuple = tuple(_build_transition_equations(state_expanded, d_terms))
    aliases_ir_map = _build_aliases_ir(aliases)
    return NormalizedRhs(
        kind="transitions",
        state_names=tuple(state_expanded),
        equations=eqs_tuple,
        aliases=aliases,
        param_names=_sorted_unique(
            sym
            for sym in all_syms
            if sym not in state_set
            and sym not in aliases
            and sym not in shaped_set
            and sym not in time_varying_set
            and sym not in axis_name_set
            and sym not in template_base_set
            and sym not in _SHAPED_PARAM_BUILTIN_NAMES
        ),
        all_symbols=frozenset(all_syms | set(aliases.keys())),
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
        equations_ir=_build_equations_ir(eqs_tuple, aliases_ir_map),
        equations_ir_raw=_build_equations_ir(eqs_tuple),
    )
