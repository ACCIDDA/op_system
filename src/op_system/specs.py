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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# -----------------------------------------------------------------------------
# Error message constants
# -----------------------------------------------------------------------------

_INVALID_RHS_SPEC_PREFIX = "Invalid op_system RHS specification."
_INVALID_EXPRESSION_PREFIX = "Invalid op_system expression."
_UNSUPPORTED_FEATURE_PREFIX = "Unsupported op_system feature."

_ALLOWED_MIXING_FORMS: dict[str, tuple[str, ...]] = {
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


def _normalize_mixing_axes_field(
    axes_field: object, *, idx: int, axis_names: set[str]
) -> tuple[str, ...] | None:
    if axes_field is None:
        return None
    if not isinstance(axes_field, (list, tuple)) or not axes_field:
        _raise_invalid_rhs_spec(
            detail=f"mixing[{idx}].axes must be a non-empty list if provided"
        )
    resolved_axes: list[str] = []
    for ax_idx, ax in enumerate(axes_field):
        if not isinstance(ax, str) or not ax.strip():
            _raise_invalid_rhs_spec(
                detail=f"mixing[{idx}].axes[{ax_idx}] must be a non-empty string"
            )
        ax_name = ax.strip()
        if ax_name not in axis_names:
            _raise_invalid_rhs_spec(
                detail=f"mixing[{idx}] references unknown axis {ax_name!r}"
            )
        resolved_axes.append(ax_name)
    return tuple(resolved_axes)


def _normalize_mixing_form_and_params(
    *, value: object, form: object, params_field: object, idx: int
) -> tuple[str, Mapping[str, Any] | None]:
    if value is None:
        if not isinstance(form, str) or not form.strip():
            _raise_invalid_rhs_spec(
                detail=(f"mixing[{idx}].form is required when value is not provided")
            )
        form_name = form.strip().lower()
        if form_name not in _ALLOWED_MIXING_FORMS:
            _raise_invalid_rhs_spec(
                detail=f"mixing[{idx}] form {form_name!r} is not supported"
            )
        required_keys = _ALLOWED_MIXING_FORMS[form_name]
        params_map_required = _ensure_mapping(
            params_field, name=f"mixing[{idx}].params"
        )
        for req in required_keys:
            if req not in params_map_required:
                _raise_invalid_rhs_spec(
                    detail=f"mixing[{idx}].params missing required key {req!r}"
                )
        return form_name, params_map_required

    form_name = (
        str(form).strip().lower()
        if isinstance(form, str) and form.strip()
        else "custom_value"
    )
    if form_name and form_name not in _ALLOWED_MIXING_FORMS:
        _raise_invalid_rhs_spec(
            detail=f"mixing[{idx}] form {form_name!r} is not supported"
        )
    params_map_optional: Mapping[str, Any] | None
    if params_field is not None:
        params_map_optional = _ensure_mapping(
            params_field, name=f"mixing[{idx}].params"
        )
    else:
        params_map_optional = None
    return form_name, params_map_optional


def _normalize_single_mixing(
    mk_map: Mapping[str, Any],
    *,
    idx: int,
    axis_names: set[str],
    seen: set[str],
) -> dict[str, Any]:
    name = _get_required_str(mk_map, idx=idx, key="name")
    if name in seen:
        _raise_invalid_rhs_spec(detail=f"duplicate mixing name: {name!r}")
    seen.add(name)

    axes_resolved = _normalize_mixing_axes_field(
        mk_map.get("axes"), idx=idx, axis_names=axis_names
    )
    form_name, params_map = _normalize_mixing_form_and_params(
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


def _normalize_mixing(
    raw_mixing: object,
    *,
    axis_names: set[str],
) -> list[dict[str, Any]]:
    """Normalize mixing kernel metadata.

    Returns:
        List of normalized mixing kernel definitions.
    """
    if raw_mixing is None:
        return []
    if not isinstance(raw_mixing, list):
        _raise_invalid_rhs_spec(detail="mixing must be a list")

    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for idx, mk in enumerate(raw_mixing):
        mk_map = _ensure_mapping(mk, name=f"mixing[{idx}]")
        mk_out = _normalize_single_mixing(
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
    mixing_meta = _normalize_mixing(
        spec.get("mixing") or _get_meta_field(spec, "mixing"),
        axis_names=axis_names,
    )
    operators_raw = spec.get("operators") or _get_meta_field(spec, "operators")
    operators_meta = (
        _normalize_operators(operators_raw, axis_names=axis_names)
        if isinstance(operators_raw, list)
        else operators_raw
    )
    return aliases, state_axes, mixing_meta, operators_meta


def _collect_alias_symbols(aliases: Mapping[str, str]) -> set[str]:
    symbols: set[str] = set()
    for expr in aliases.values():
        tree = _parse_expr(expr)
        symbols |= _collect_names(tree)
    return symbols


def _gather_equations(
    state: list[str],
    equations_map: Mapping[str, Any],
    all_syms: set[str],
) -> list[str]:
    eqs: list[str] = []
    for name in state:
        expr = equations_map[name]
        if not isinstance(expr, str) or not expr.strip():
            _raise_invalid_rhs_spec(
                detail=f"equations[{name!r}] must be a non-empty string"
            )
        expr_s = expr.strip()
        tree = _parse_expr(expr_s)
        all_syms |= _collect_names(tree)
        eqs.append(expr_s)
    return eqs


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


def normalize_expr_rhs(spec: Mapping[str, Any]) -> NormalizedRhs:
    """
    Normalize an expression-based RHS specification.

    Args:
        spec: Raw RHS specification mapping.

    Returns:
        Backend-facing normalized RHS representation.
    """
    state = _ensure_str_list(spec.get("state"), name="state")
    if len(state) != len(set(state)):
        _raise_invalid_rhs_spec(detail="state contains duplicate names")

    equations_map = spec.get("equations")
    if not isinstance(equations_map, dict):
        _raise_invalid_rhs_spec(detail="equations must be a mapping of state->expr")

    axes_meta = _normalize_axes(spec.get("axes"))
    axis_names_set: set[str] = {"subgroup"} | {ax["name"] for ax in axes_meta}
    state_set = set(state)
    aliases, state_axes, mixing_meta, operators_meta = _normalize_common_meta(
        spec, axis_names=axis_names_set, state_set=state_set
    )

    meta: dict[str, Any] = {
        "axes": axes_meta,
        "state_axes": state_axes,
        "mixing": mixing_meta,
        "operators": operators_meta,
    }
    for reserved_key in ("sources", "couplings", "constraints"):
        if reserved_key in spec:
            meta[reserved_key] = spec.get(reserved_key)

    missing_eqs = [s for s in state if s not in equations_map]
    if missing_eqs:
        _raise_invalid_rhs_spec(
            missing=missing_eqs, detail="Missing equation(s) for state"
        )

    unknown_keys = [k for k in equations_map if k not in state_set]
    if unknown_keys:
        _raise_invalid_rhs_spec(
            detail=f"unknown equation key(s): {sorted(unknown_keys)}"
        )

    all_syms = _collect_alias_symbols(aliases)
    eqs = _gather_equations(state, equations_map, all_syms)

    params = _sorted_unique(
        sym for sym in all_syms if sym not in state_set and sym not in aliases
    )

    return NormalizedRhs(
        kind="expr",
        state_names=tuple(state),
        equations=tuple(eqs),
        aliases=aliases,
        param_names=params,
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


def normalize_transitions_rhs(spec: Mapping[str, Any]) -> NormalizedRhs:
    """
    Normalize a transition-based RHS specification (diagram/hazard semantics).

    Args:
        spec: Raw RHS specification mapping.

    Returns:
        Backend-facing normalized RHS representation.
    """
    state = _ensure_str_list(spec.get("state"), name="state")
    if len(state) != len(set(state)):
        _raise_invalid_rhs_spec(detail="state contains duplicate names")

    transitions = spec.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        _raise_invalid_rhs_spec(detail="transitions must be a non-empty list")

    axes_meta = _normalize_axes(spec.get("axes"))

    aliases, _, mixing_meta, operators_meta = _normalize_common_meta(
        spec,
        axis_names={"subgroup"} | {ax["name"] for ax in axes_meta},
        state_set=None,
    )

    meta: dict[str, Any] = {
        "transitions": transitions,
        "axes": axes_meta,
        "mixing": mixing_meta,
        "operators": operators_meta,
    }
    for reserved_key in ("sources", "couplings", "constraints"):
        if reserved_key in spec:
            meta[reserved_key] = spec.get(reserved_key)

    state_set = set(state)
    d_terms: dict[str, list[str]] = {s: [] for s in state}
    all_syms = _collect_alias_symbols(aliases)

    for idx, tr_map in enumerate(transitions):
        _apply_transition(
            idx=idx,
            tr=_validate_transition_mapping(tr_map, idx=idx),
            state_set=state_set,
            all_syms=all_syms,
            d_terms=d_terms,
        )

    equations = _build_transition_equations(state, d_terms)

    params = _sorted_unique(
        sym for sym in all_syms if sym not in state_set and sym not in aliases
    )

    return NormalizedRhs(
        kind="transitions",
        state_names=tuple(state),
        equations=tuple(equations),
        aliases=aliases,
        param_names=params,
        all_symbols=frozenset(all_syms | set(aliases.keys())),
        meta=meta,
    )
