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

import ast
import re
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from re import Match

# -----------------------------------------------------------------------------
# Error message constants
# -----------------------------------------------------------------------------

_INVALID_RHS_SPEC_PREFIX = "Invalid op_system RHS specification."
_INVALID_EXPRESSION_PREFIX = "Invalid op_system expression."
_UNSUPPORTED_FEATURE_PREFIX = "Unsupported op_system feature."

_ALLOWED_KERNEL_FORMS: dict[str, tuple[str, ...]] = {
    "erfc": ("scale", "sigma"),
    "gaussian": ("scale", "sigma"),
    "exponential": ("scale", "lambda"),
    "gamma": ("scale", "k", "theta"),
    "power_law": ("scale", "sigma", "p"),
    "custom_value": (),
}


def _raise_invalid_rhs_spec(
    *, missing: list[str] | None = None, detail: str | None = None
) -> NoReturn:
    """Raise a standardized RHS specification error.

    Args:
        missing: Optional list of missing field names.
        detail: Optional additional detail string.

    Raises:
        ValueError: Always.
    """
    parts: list[str] = [_INVALID_RHS_SPEC_PREFIX]
    if missing:
        parts.append(f"Missing required field(s): {sorted(set(missing))}.")
    if detail:
        parts.append(f"Detail: {detail}")
    raise ValueError(" ".join(parts))


def _raise_invalid_expression(*, detail: str) -> NoReturn:
    """Raise a standardized expression error.

    Args:
        detail: Error detail.

    Raises:
        ValueError: Always.
    """
    msg = f"{_INVALID_EXPRESSION_PREFIX} Detail: {detail}"
    raise ValueError(msg)


def _raise_unsupported_feature(*, feature: str, detail: str | None = None) -> NoReturn:
    """Raise a standardized unsupported feature error.

    Args:
        feature: Feature identifier.
        detail: Optional additional detail.

    Raises:
        NotImplementedError: Always.
    """
    msg = f"{_UNSUPPORTED_FEATURE_PREFIX} Feature '{feature}' is not supported."
    if detail:
        msg = f"{msg} Detail: {detail}"
    raise NotImplementedError(msg)


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


# -----------------------------------------------------------------------------
# AST helpers (minimal v1)
# -----------------------------------------------------------------------------


def _parse_expr(expr: str) -> ast.AST:
    """
    Parse a Python expression and return the AST node.

    Args:
        expr: Expression string to parse.

    Returns:
        The parsed AST node.
    """
    try:
        return ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        _raise_invalid_expression(detail=f"invalid expression syntax: {exc.msg}")


def _collect_names(tree: ast.AST) -> set[str]:
    """
    Collect all Name identifiers used in an expression AST.

    Args:
        tree: Parsed AST.

    Returns:
        Set of identifier names referenced in the expression.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(str(node.id))
    return names


def _ensure_str_list(x: object, *, name: str) -> list[str]:
    """
    Ensure x is a list of non-empty strings.

    Args:
        x: Input value.
        name: Field name for error messages.

    Returns:
        List of stripped, non-empty strings.
    """
    if not isinstance(x, (list, tuple)):
        _raise_invalid_rhs_spec(detail=f"{name} must be a list of strings")
    out: list[str] = []
    for i, v in enumerate(x):
        if not isinstance(v, str) or not v.strip():
            _raise_invalid_rhs_spec(detail=f"{name}[{i}] must be a non-empty string")
        out.append(v.strip())
    return out


def _ensure_str_dict(x: object, *, name: str) -> dict[str, str]:
    """
    Ensure x is a dict of non-empty strings.

    Args:
        x: Input value (mapping) or None.
        name: Field name for error messages.

    Returns:
        Dict of stripped string keys to stripped non-empty string values.
    """
    if x is None:
        return {}
    if not isinstance(x, dict):
        _raise_invalid_rhs_spec(detail=f"{name} must be a mapping of string->string")
    out: dict[str, str] = {}
    for k, v in x.items():
        if not isinstance(k, str) or not k.strip():
            _raise_invalid_rhs_spec(detail=f"{name} keys must be non-empty strings")
        if not isinstance(v, str) or not v.strip():
            _raise_invalid_rhs_spec(detail=f"{name}[{k!r}] must be a non-empty string")
        out[k.strip()] = v.strip()
    return out


def _sorted_unique(xs: Iterable[str]) -> tuple[str, ...]:
    """Return a sorted tuple of unique strings from the iterable."""
    return tuple(sorted(set(xs)))


def _as_number(x: object, *, name: str) -> float:
    """Ensure x is a real number (int/float) and return float.

    Returns:
        Input coerced to float.
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        _raise_invalid_rhs_spec(detail=f"{name} must be a number")
    return float(x)


def _ensure_mapping(x: object, *, name: str) -> Mapping[str, Any]:
    """Ensure x is a mapping.

    Returns:
        Mapping view of the input.
    """
    if not isinstance(x, dict):
        _raise_invalid_rhs_spec(detail=f"{name} must be a mapping")
    return x


def _normalize_axis_name(ax_map: Mapping[str, Any], *, idx: int, seen: set[str]) -> str:
    name_val = ax_map.get("name")
    if not isinstance(name_val, str) or not name_val.strip():
        _raise_invalid_rhs_spec(detail=f"axes[{idx}].name must be a non-empty string")
    name = name_val.strip()
    if name in seen:
        _raise_invalid_rhs_spec(detail=f"duplicate axis name: {name!r}")
    seen.add(name)
    return name


def _normalize_axis_type(ax_map: Mapping[str, Any], *, idx: int) -> str:
    ax_type = str(ax_map.get("type", "categorical")).strip().lower()
    if ax_type not in {"categorical", "continuous"}:
        _raise_invalid_rhs_spec(
            detail=f"axes[{idx}].type must be 'categorical' or 'continuous'"
        )
    return ax_type


def _normalize_axis_units(ax_map: Mapping[str, Any], *, idx: int) -> str | None:
    units_obj = ax_map.get("units")
    if units_obj is None:
        return None
    if not isinstance(units_obj, str) or not units_obj.strip():
        _raise_invalid_rhs_spec(
            detail=f"axes[{idx}].units must be a non-empty string if provided"
        )
    return units_obj.strip()


def _normalize_axis_coords(
    coords_obj: object,
    *,
    idx: int,
    ax_type: str,
) -> tuple[list[Any], int]:
    if not isinstance(coords_obj, (list, tuple)) or not coords_obj:
        _raise_invalid_rhs_spec(detail=f"axes[{idx}].coords must be a non-empty list")
    coords = list(coords_obj)
    if ax_type == "categorical":
        for j, v in enumerate(coords):
            if not isinstance(v, str) or not str(v).strip():
                _raise_invalid_rhs_spec(
                    detail=f"axes[{idx}].coords[{j}] must be a non-empty string"
                )
            coords[j] = str(v).strip()
        return coords, len(coords)

    for j, v in enumerate(coords):
        coords[j] = _as_number(v, name=f"axes[{idx}].coords[{j}]")
    if len(coords) >= 2:
        for j in range(1, len(coords)):
            if coords[j] < coords[j - 1]:
                _raise_invalid_rhs_spec(
                    detail=(
                        f"axes[{idx}].coords must be non-decreasing for continuous axes"
                    )
                )
    return coords, len(coords)


def _compute_axis_deltas(coords: list[float], *, idx: int) -> list[float]:
    """Compute integration weights (trapezoidal) for a continuous axis.

    Weights are half-interval widths at boundaries and centered widths inside.
    Raises if any interval width is non-positive.

    Returns:
        List of trapezoidal weights matching coords length.
    """
    if len(coords) < 2:
        _raise_invalid_rhs_spec(detail=f"axes[{idx}] continuous requires >=2 coords")

    deltas: list[float] = []
    for i in range(len(coords)):
        if i == 0:
            width = (coords[1] - coords[0]) / 2.0
        elif i == len(coords) - 1:
            width = (coords[-1] - coords[-2]) / 2.0
        else:
            width = (coords[i + 1] - coords[i - 1]) / 2.0
        if width <= 0.0:
            _raise_invalid_rhs_spec(
                detail=f"axes[{idx}] coords must be strictly increasing for integration"
            )
        deltas.append(width)
    return deltas


def _generate_continuous_coords(
    *, domain: object, size_obj: object, spacing: str, idx: int
) -> tuple[list[float], int]:
    domain_map = (
        _ensure_mapping(domain, name=f"axes[{idx}].domain")
        if domain is not None
        else None
    )
    lb = (
        _as_number(domain_map.get("lb"), name=f"axes[{idx}].domain.lb")
        if domain_map
        else None
    )
    ub = (
        _as_number(domain_map.get("ub"), name=f"axes[{idx}].domain.ub")
        if domain_map
        else None
    )
    if lb is None or ub is None:
        _raise_invalid_rhs_spec(
            detail=(
                f"axes[{idx}] continuous requires domain.lb and domain.ub "
                "when coords are absent"
            )
        )
    if ub <= lb:
        _raise_invalid_rhs_spec(detail=f"axes[{idx}].domain.ub must be greater than lb")

    if not isinstance(size_obj, (int, float)) or isinstance(size_obj, bool):
        _raise_invalid_rhs_spec(detail=f"axes[{idx}].size must be an integer >= 2")
    resolved_size = int(size_obj)
    if resolved_size < 2:
        _raise_invalid_rhs_spec(detail=f"axes[{idx}].size must be >= 2")

    spacing_allowed = {"linear", "log", "geom"}
    if spacing not in spacing_allowed:
        _raise_invalid_rhs_spec(
            detail=f"axes[{idx}].spacing must be one of {sorted(spacing_allowed)}"
        )

    coords: list[float] = []
    if spacing == "linear":
        step = (ub - lb) / (resolved_size - 1)
        coords = [lb + step * i for i in range(resolved_size)]
    elif spacing in {"log", "geom"}:
        if lb <= 0 or ub <= 0:
            _raise_invalid_rhs_spec(
                detail=f"axes[{idx}] log/geom spacing requires positive lb/ub"
            )
        ratio = (ub / lb) ** (1.0 / (resolved_size - 1))
        coords = [lb * (ratio**i) for i in range(resolved_size)]

    return coords, resolved_size


def _normalize_single_axis(
    ax_map: Mapping[str, Any], *, idx: int, seen: set[str]
) -> dict[str, Any]:
    name = _normalize_axis_name(ax_map, idx=idx, seen=seen)
    ax_type = _normalize_axis_type(ax_map, idx=idx)
    spacing = str(ax_map.get("spacing", "linear")).strip().lower()
    coords_obj = ax_map.get("coords")
    domain = ax_map.get("domain")
    size_obj = ax_map.get("size")

    if coords_obj is not None:
        coords, resolved_size = _normalize_axis_coords(
            coords_obj, idx=idx, ax_type=ax_type
        )
    elif ax_type == "categorical":
        _raise_invalid_rhs_spec(detail=f"axes[{idx}] categorical requires coords")
    else:
        coords, resolved_size = _generate_continuous_coords(
            domain=domain, size_obj=size_obj, spacing=spacing, idx=idx
        )

    axis_out: dict[str, Any] = {
        "name": name,
        "type": ax_type,
        "coords": coords,
        "size": resolved_size,
    }
    if ax_type == "continuous":
        axis_out["deltas"] = _compute_axis_deltas(coords, idx=idx)
    if domain is not None:
        axis_out["domain"] = domain
    if spacing:
        axis_out["spacing"] = spacing
    units = _normalize_axis_units(ax_map, idx=idx)
    if units is not None:
        axis_out["units"] = units
    return axis_out


def _normalize_axes(raw_axes: object) -> list[dict[str, Any]]:
    """Normalize axis specifications (categorical or continuous).

    Returns:
        Normalized axis definitions with coords and sizes.
    """
    if raw_axes is None:
        return []
    if not isinstance(raw_axes, list):
        _raise_invalid_rhs_spec(detail="axes must be a list of axis definitions")

    seen: set[str] = set()
    axes_out: list[dict[str, Any]] = []

    for idx, ax in enumerate(raw_axes):
        axis_out = _normalize_single_axis(
            _ensure_mapping(ax, name=f"axes[{idx}]"), idx=idx, seen=seen
        )
        axes_out.append(axis_out)

    return axes_out


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


def _normalize_operators(
    raw_ops: object,
    *,
    axis_names: set[str],
) -> list[dict[str, Any]]:
    """Normalize operator metadata and validate axis references.

    Returns:
        List of normalized operator definitions.
    """
    if raw_ops is None:
        return []
    if not isinstance(raw_ops, list):
        _raise_invalid_rhs_spec(detail="operators must be a list")
    out: list[dict[str, Any]] = []
    for idx, op in enumerate(raw_ops):
        op_map = _ensure_mapping(op, name=f"operators[{idx}]")
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
        op_out = dict(op_map)
        op_out["axis"] = axis_name_s
        out.append(op_out)
    return out


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
        _normalize_operators(operators_raw, axis_names=axis_names)
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


def _sanitize_fragment(val: object) -> str:
    """Sanitize coord fragments into identifier-friendly pieces.

    Returns:
        A safe identifier fragment with non-alphanumerics replaced by underscores.
    """
    s = str(val)
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_") or "_"


def _expand_state_templates(
    state_raw: list[str],
    *,
    axes: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[tuple[str, dict[str, str]]]]]:
    """Expand state templates with categorical axes into concrete names.

    Returns:
        A tuple of (expanded_state_names, template_map) where template_map maps
        template keys to lists of (expanded_name, assignment dict).
    """
    axis_lookup: dict[str, list[str]] = {
        ax["name"]: [str(c) for c in ax.get("coords", [])] for ax in axes
    }

    expanded: list[str] = []
    template_map: dict[str, list[tuple[str, dict[str, str]]]] = {}

    for entry in state_raw:
        if "[" not in entry or "]" not in entry:
            expanded.append(entry)
            continue
        m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\[(.+)\]\s*", entry)
        if not m:
            _raise_invalid_rhs_spec(detail=f"invalid state template: {entry!r}")
        base = m.group(1)
        placeholders = [p.strip() for p in m.group(2).split(",")]
        if not all(placeholders):
            _raise_invalid_rhs_spec(detail=f"invalid state template axes in {entry!r}")
        coords_lists: list[list[str]] = []
        for ph in placeholders:
            if ph not in axis_lookup:
                _raise_invalid_rhs_spec(
                    detail=f"state template axis {ph!r} not defined"
                )
            coords_lists.append(axis_lookup[ph])
        combos = list(product(*coords_lists))
        template_key = f"{base}[{','.join(placeholders)}]"
        template_map[template_key] = []
        for combo in combos:
            parts = [
                f"{placeholders[i]}_{_sanitize_fragment(combo[i])}"
                for i in range(len(placeholders))
            ]
            name = base + "__" + "__".join(parts)
            template_map[template_key].append((
                name,
                {placeholders[i]: combo[i] for i in range(len(placeholders))},
            ))
            expanded.append(name)

    return expanded, template_map


def _expand_alias_templates(
    aliases_raw: Mapping[str, str],
    *,
    axes: list[dict[str, Any]],
    template_map_seed: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> tuple[dict[str, str], dict[str, list[tuple[str, dict[str, str]]]]]:
    """Expand templated alias names and substitute inline placeholders.

    Args:
        aliases_raw: Mapping of alias name (templated or concrete) to expr.
        axes: Normalized axis definitions.
        template_map_seed: Existing template map (e.g., state templates) to use
            for substitutions inside alias expressions.

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
        expr_s = expr.strip()
        if not expr_s:
            _raise_invalid_rhs_spec(
                detail=f"aliases[{raw_name!r}] must be a non-empty string"
            )
        expr_s = _expand_helpers(expr_s, axes=axes)

        if raw_name in alias_template_map:
            for expanded_name, assignment in alias_template_map[raw_name]:
                substituted = _apply_template_substitutions(
                    expr_s,
                    assignment=assignment,
                    template_map=combined_template_map,
                )
                aliases_out[expanded_name] = substituted
        else:
            aliases_out[raw_name] = expr_s

    return aliases_out, alias_template_map


def _extract_placeholders_from_expr(expr: str) -> set[str]:
    """Extract placeholder symbols found inside [...] tokens in an expression.

    Returns:
        Set of placeholder names referenced inside bracket templates.
    """
    placeholders: set[str] = set()
    for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*\[(.*?)\]", expr):
        inner = match.group(1)
        for part in inner.split(","):
            part_s = part.strip()
            if part_s:
                placeholders.add(part_s)
    return placeholders


def _render_template_or_literal(name_s: str, assignment: Mapping[str, str]) -> str:
    base, placeholders = _parse_template_key(name_s)
    if not placeholders or any(ph not in assignment for ph in placeholders):
        return name_s
    return _render_template_name(base, placeholders, assignment)


def _expand_transition_templates(  # noqa: PLR0914
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

    for idx, tr_map in enumerate(transitions_raw):
        tr_valid = _validate_transition_mapping(tr_map, idx=idx)
        frm_s = _get_required_str(tr_valid, idx=idx, key="from")
        to_s = _get_required_str(tr_valid, idx=idx, key="to")
        rate_s = _get_required_str(tr_valid, idx=idx, key="rate")
        name_s = tr_valid.get("name") if isinstance(tr_valid.get("name"), str) else None

        placeholders: set[str] = set()
        placeholders |= set(_parse_template_key(frm_s)[1])
        placeholders |= set(_parse_template_key(to_s)[1])
        placeholders |= _extract_placeholders_from_expr(rate_s)
        if name_s:
            placeholders |= _extract_placeholders_from_expr(name_s)

        if not placeholders:
            rate_expanded = _expand_helpers(rate_s, axes=axes)
            tr_out = dict(tr_valid)
            tr_out["from"] = frm_s
            tr_out["to"] = to_s
            tr_out["rate"] = rate_expanded
            expanded.append(tr_out)
            continue

        coords_lists: list[list[str]] = []
        for ph in placeholders:
            if ph not in axis_lookup:
                _raise_invalid_rhs_spec(
                    detail=f"transition placeholder {ph!r} references unknown axis"
                )
            coords_lists.append(axis_lookup[ph])

        combos = list(product(*coords_lists))
        for combo in combos:
            assignment = {ph: combo[i] for i, ph in enumerate(placeholders)}

            frm_render = _render_template_or_literal(frm_s, assignment)
            to_render = _render_template_or_literal(to_s, assignment)

            rate_sub = _apply_template_substitutions(
                rate_s,
                assignment=assignment,
                template_map=template_map,
            )
            rate_expanded = _expand_helpers(rate_sub, axes=axes)

            tr_expanded: dict[str, Any] = dict(tr_valid)
            tr_expanded["from"] = frm_render
            tr_expanded["to"] = to_render
            tr_expanded["rate"] = rate_expanded
            if name_s:
                tr_expanded["name"] = _apply_template_substitutions(
                    name_s, assignment=assignment, template_map=template_map
                )
            expanded.append(tr_expanded)

    return expanded


def _parse_template_key(template_key: str) -> tuple[str, list[str]]:
    """Parse a template key like "S[pop,age]" into (base, placeholders).

    Returns:
        Tuple of (base, placeholder list). Returns (template_key, []) if it
        does not match the expected format.
    """
    m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\[(.+)\]\s*", template_key)
    if not m:
        return template_key, []
    base = m.group(1)
    placeholders = [p.strip() for p in m.group(2).split(",") if p.strip()]
    return base, placeholders


def _render_template_name(
    base: str, placeholders: list[str], assignment: Mapping[str, str]
) -> str:
    """Render an expanded name from base + placeholders using assignment.

    Returns:
        Concrete state name (e.g., S__pop_p1__age_0_5).
    """
    parts = [f"{ph}_{_sanitize_fragment(assignment[ph])}" for ph in placeholders]
    return base + "__" + "__".join(parts)


def _apply_template_substitutions(
    expr_s: str,
    *,
    assignment: Mapping[str, str],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> str:
    """Replace template references in ``expr_s`` using a concrete assignment.

    Handles both explicit template keys (e.g., ``S[age]`` present in a template
    map) and inline placeholder syntax like ``theta[age,pop]`` when the
    placeholders are covered by ``assignment``.

    Returns:
        Expression with template tokens replaced using the provided assignment.
    """
    expr_out = expr_s

    # Explicit template keys (e.g., from state or alias template maps)
    for template_key in template_map:
        base, placeholders = _parse_template_key(template_key)
        if not placeholders or any(ph not in assignment for ph in placeholders):
            continue
        rendered = _render_template_name(base, placeholders, assignment)
        expr_out = re.sub(re.escape(template_key), rendered, expr_out)

    # Inline placeholder syntax without explicit template map entry
    pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[(.*?)\]")

    def _inline_replacer(match: Match[str]) -> str:
        base = match.group(1)
        inner = match.group(2)
        placeholders = [p.strip() for p in inner.split(",") if p.strip()]
        if not placeholders or any(ph not in assignment for ph in placeholders):
            return match.group(0)
        return _render_template_name(base, placeholders, assignment)

    return pattern.sub(_inline_replacer, expr_out)


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
        _raise_invalid_rhs_spec(detail=f"integrate_over axis {ax_name!r} not found")

    pattern = re.compile(
        r"integrate_over\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(.*?)\)",
        re.DOTALL,
    )

    out = expr
    while True:
        m = pattern.search(out)
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
        _raise_invalid_rhs_spec(detail=f"sum_over axis {ax_name!r} not found")

    pattern = re.compile(
        r"sum_over\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(.*?)\)",
        re.DOTALL,
    )

    out = expr
    # Iterate until no more matches to handle multiple sum_over occurrences.
    while True:
        m = pattern.search(out)
        if not m:
            break
        axis_name = m.group(1)
        var_name = m.group(2)
        inner = m.group(3)
        coords = _axis_coords(axis_name)
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


def _validate_chain_entry(
    *,
    chain: Mapping[str, Any],
    idx: int,
    state_set: set[str],
    allow_templates: bool,
) -> tuple[list[str], str, str | None]:
    """Validate a chain entry and return stage names, rate, and sink.

    Returns:
        (stage_names, rate_expr, sink_name or None).
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
    rate_expr = _get_required_str(chain, idx=idx, key="forward")
    sink = chain.get("to")
    stage_names = [f"{cname}{i}" for i in range(1, clen + 1)]
    missing = [s for s in stage_names if s not in state_set]
    if missing:
        tail = "Define them in state."
        if allow_templates:
            tail = "Define them (or templates) in state."
        _raise_invalid_rhs_spec(
            detail=(f"chain[{idx}] references missing states: {missing}. {tail}")
        )
    sink_s: str | None = None
    if sink is not None:
        sink_s = sink.strip() if isinstance(sink, str) else None
        if not sink_s:
            _raise_invalid_rhs_spec(
                detail=f"chain[{idx}].to must be a non-empty string"
            )
        if sink_s not in state_set:
            _raise_invalid_rhs_spec(detail=f"chain[{idx}].to={sink_s!r} not in state")
    return stage_names, rate_expr, sink_s


def _apply_expr_chains(
    *,
    chains: list[Any],
    state_expanded: list[str],
    equations_map: dict[str, Any],
) -> None:
    """Apply chain helper for expr kind by auto-filling equations when missing."""
    state_set = set(state_expanded)
    for c_idx, chain in enumerate(chains):
        stage_names, rate_expr, sink_s = _validate_chain_entry(
            chain=chain,
            idx=c_idx,
            state_set=state_set,
            allow_templates=True,
        )
        if stage_names[0] not in equations_map:
            equations_map[stage_names[0]] = f"-({rate_expr})*{stage_names[0]}"
        for i in range(1, len(stage_names)):
            if stage_names[i] not in equations_map:
                equations_map[stage_names[i]] = (
                    f"({rate_expr})*{stage_names[i - 1]} - "
                    f"({rate_expr})*{stage_names[i]}"
                )
        if sink_s is not None and sink_s not in equations_map:
            equations_map[sink_s] = f"({rate_expr})*{stage_names[-1]}"


def _apply_transition_chains(
    *,
    chains: list[Any],
    transitions_raw: list[dict[str, Any]],
    state_set: set[str],
) -> None:
    """Apply chain helper for transitions kind by appending transitions."""
    for c_idx, chain in enumerate(chains):
        stage_names, rate_expr, sink_s = _validate_chain_entry(
            chain=chain,
            idx=c_idx,
            state_set=state_set,
            allow_templates=False,
        )
        transitions_raw.extend(
            {
                "from": stage_names[i],
                "to": stage_names[i + 1],
                "rate": rate_expr,
            }
            for i in range(len(stage_names) - 1)
        )
        if sink_s is not None:
            transitions_raw.append({
                "from": stage_names[-1],
                "to": sink_s,
                "rate": rate_expr,
            })


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

    kind = str(spec.get("kind", "expr")).strip().lower()

    if kind == "expr":  # lowest-level escape hatch
        return normalize_expr_rhs(spec)

    if kind == "transitions":  # diagram-style hazards
        return normalize_transitions_rhs(spec)

    _raise_unsupported_feature(
        feature=f"rhs.kind={kind}",
        detail="Only 'expr' and 'transitions' are supported in v1.",
    )


# -----------------------------------------------------------------------------
# expr kind
# -----------------------------------------------------------------------------


def normalize_expr_rhs(spec: Mapping[str, Any]) -> NormalizedRhs:  # noqa: PLR0914
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

    axes_meta = _normalize_axes(spec.get("axes"))
    aliases_raw, state_axes, kernels_meta, operators_meta = _normalize_common_meta(
        spec,
        axis_names={"subgroup"} | {ax["name"] for ax in axes_meta},
        state_set=set(state_raw),
    )

    meta: dict[str, Any] = {
        "axes": axes_meta,
        "state_axes": state_axes,
        "kernels": kernels_meta,
        "operators": operators_meta,
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
        aliases_raw, axes=axes_meta, template_map_seed=state_template_map
    )
    template_map_all = {**state_template_map, **alias_template_map}

    chains = spec.get("chain") or []
    if chains:
        if not isinstance(chains, list):
            _raise_invalid_rhs_spec(detail="chain must be a list if provided")
        _apply_expr_chains(
            chains=chains,
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


def normalize_transitions_rhs(spec: Mapping[str, Any]) -> NormalizedRhs:  # noqa: PLR0914
    """Normalize a transition-based RHS specification (diagram/hazard semantics).

    Returns:
        Backend-facing normalized RHS representation for transitions kind.
    """
    state_raw = _ensure_str_list(spec.get("state"), name="state")
    if len(state_raw) != len(set(state_raw)):
        _raise_invalid_rhs_spec(detail="state contains duplicate names")

    transitions_raw = spec.get("transitions")
    if not isinstance(transitions_raw, list) or not transitions_raw:
        _raise_invalid_rhs_spec(detail="transitions must be a non-empty list")

    axes_meta = _normalize_axes(spec.get("axes"))

    aliases_raw, _, kernels_meta, operators_meta = _normalize_common_meta(
        spec,
        axis_names={"subgroup"} | {ax["name"] for ax in axes_meta},
        state_set=None,
    )

    meta: dict[str, Any] = {
        "transitions": transitions_raw,
        "axes": axes_meta,
        "kernels": kernels_meta,
        "operators": operators_meta,
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
        aliases_raw, axes=axes_meta, template_map_seed=state_template_map
    )
    template_map_all = {**state_template_map, **alias_template_map}

    state_set = set(state_expanded)
    d_terms: dict[str, list[str]] = {s: [] for s in state_expanded}
    all_syms = _collect_alias_symbols(aliases, axes=axes_meta)

    chains = spec.get("chain") or []
    if chains:
        if not isinstance(chains, list):
            _raise_invalid_rhs_spec(detail="chain must be a list if provided")
        _apply_transition_chains(
            chains=chains,
            transitions_raw=transitions_raw,
            state_set=state_set,
        )

    transitions_expanded = _expand_transition_templates(
        transitions_raw,
        axes=axes_meta,
        template_map=template_map_all,
    )

    for idx, tr_map in enumerate(transitions_expanded):
        tr_valid = _validate_transition_mapping(tr_map, idx=idx)
        _apply_transition(
            idx=idx,
            tr=tr_valid,
            state_set=state_set,
            all_syms=all_syms,
            d_terms=d_terms,
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
