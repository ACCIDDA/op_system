"""op_system._normalize_kernels.

Kernel, operator, state-axes, and apply_along normalization helpers.

These functions validate and normalize the ``kernels``, ``operators``,
``state_axes``, and ``apply_along`` portions of an op_system RHS spec.
They are pure helpers — all public entry points remain in ``_normalize.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from op_system._axes import _compute_axis_deltas
from op_system._errors import InvalidRhsSpecError
from op_system._helpers import _ensure_mapping, _get_required_str
from op_system._templates import (
    PinnedToken,
    WildcardToken,
    build_axis_lookup,
    expand_apply_to,
    parse_selector,
)

# ---------------------------------------------------------------------------
# Kernel form registry
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
# Shared validator used by both kernels and operators
# ---------------------------------------------------------------------------


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
# Equation expansion (apply_along / gather)
# ---------------------------------------------------------------------------


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
