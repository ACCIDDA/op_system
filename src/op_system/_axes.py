"""op_system._axes.

Axis normalization utilities: parse, validate, and compute metadata for
categorical and continuous axes declared in an op_system spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

import re

from op_system._errors import InvalidRhsSpecError
from op_system._helpers import _as_number, _ensure_mapping

_STATE_TEMPLATE_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\[(.+)\]\s*")


def _normalize_bracket_key(key: str) -> str:
    """Normalize whitespace inside bracket notation.

    Strips spaces around commas and bracket edges so that user-provided keys
    like ``"u[x, y]"`` match the canonical template form ``"u[x,y]"``.
    Inputs without brackets are returned with leading/trailing whitespace
    stripped; inputs whose bracket contents do not match the template grammar
    are returned stripped but otherwise unchanged.

    Args:
        key: Input key string, possibly containing bracket notation.

    Returns:
        Canonical bracket key string.

    Examples:
        >>> _normalize_bracket_key("u[x, y]")
        'u[x,y]'
        >>> _normalize_bracket_key("  S  ")
        'S'
        >>> _normalize_bracket_key("X[ age , imm = X0 ]")
        'X[age,imm = X0]'
    """
    if "[" not in key:
        return key.strip()
    m = _STATE_TEMPLATE_RE.fullmatch(key)
    if not m:
        return key.strip()
    base = m.group(1)
    parts = [p.strip() for p in m.group(2).split(",")]
    return f"{base}[{','.join(parts)}]"


def _normalize_axis_name(ax_map: Mapping[str, Any], *, idx: int, seen: set[str]) -> str:
    name_val = ax_map.get("name")
    if not isinstance(name_val, str) or not name_val.strip():
        raise InvalidRhsSpecError(detail=f"axes[{idx}].name must be a non-empty string")
    name = name_val.strip()
    if name in seen:
        raise InvalidRhsSpecError(detail=f"duplicate axis name: {name!r}")
    seen.add(name)
    return name


def _normalize_axis_type(ax_map: Mapping[str, Any], *, idx: int) -> str:
    ax_type = str(ax_map.get("type", "categorical")).strip().lower()
    if ax_type not in {"categorical", "ordinal", "continuous"}:
        raise InvalidRhsSpecError(
            detail=(
                f"axes[{idx}].type must be 'categorical', 'ordinal', or 'continuous'"
            )
        )
    return ax_type


def _normalize_axis_units(ax_map: Mapping[str, Any], *, idx: int) -> str | None:
    units_obj = ax_map.get("units")
    if units_obj is None:
        return None
    if not isinstance(units_obj, str) or not units_obj.strip():
        raise InvalidRhsSpecError(
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
        raise InvalidRhsSpecError(detail=f"axes[{idx}].coords must be a non-empty list")
    coords = list(coords_obj)
    if ax_type in {"categorical", "ordinal"}:
        for j, v in enumerate(coords):
            if not isinstance(v, str) or not str(v).strip():
                raise InvalidRhsSpecError(
                    detail=f"axes[{idx}].coords[{j}] must be a non-empty string"
                )
            coords[j] = str(v).strip()
        if len(set(coords)) != len(coords):
            raise InvalidRhsSpecError(detail=f"axes[{idx}].coords must be unique")
        return coords, len(coords)

    for j, v in enumerate(coords):
        coords[j] = _as_number(v, name=f"axes[{idx}].coords[{j}]")
    if len(coords) >= 2:
        for j in range(1, len(coords)):
            if coords[j] < coords[j - 1]:
                raise InvalidRhsSpecError(
                    detail=(
                        f"axes[{idx}].coords must be non-decreasing for continuous axes"
                    )
                )
    return coords, len(coords)


def _compute_axis_deltas(coords: list[float], *, idx: int) -> list[float]:
    """Compute trapezoidal integration weights for a continuous axis.

    Each weight equals the half-sum of the gaps to the neighboring points
    (one-sided at the boundaries), so summing ``f(c) * w(c)`` over coords
    approximates the trapezoidal-rule integral of ``f``.

    Args:
        coords: Strictly increasing list of coordinate values.
        idx: Axis index, used in raised error messages.

    Returns:
        List of weights with the same length as ``coords``.

    Raises:
        InvalidRhsSpecError: If ``coords`` has fewer than two entries, or
            adjacent coordinates are not strictly increasing.

    Examples:
        >>> _compute_axis_deltas([0.0, 1.0, 3.0], idx=0)
        [0.5, 1.5, 1.0]
        >>> _compute_axis_deltas([0.0, 2.0], idx=0)
        [1.0, 1.0]
    """
    if len(coords) < 2:
        raise InvalidRhsSpecError(detail=f"axes[{idx}] continuous requires >=2 coords")

    deltas: list[float] = []
    for i in range(len(coords)):
        if i == 0:
            width = (coords[1] - coords[0]) / 2.0
        elif i == len(coords) - 1:
            width = (coords[-1] - coords[-2]) / 2.0
        else:
            width = (coords[i + 1] - coords[i - 1]) / 2.0
        if width <= 0.0:
            raise InvalidRhsSpecError(
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
        raise InvalidRhsSpecError(
            detail=(
                f"axes[{idx}] continuous requires domain.lb and domain.ub "
                "when coords are absent"
            )
        )
    if ub <= lb:
        raise InvalidRhsSpecError(
            detail=f"axes[{idx}].domain.ub must be greater than lb"
        )

    if not isinstance(size_obj, (int, float)) or isinstance(size_obj, bool):
        raise InvalidRhsSpecError(detail=f"axes[{idx}].size must be an integer >= 2")
    resolved_size = int(size_obj)
    if resolved_size < 2:
        raise InvalidRhsSpecError(detail=f"axes[{idx}].size must be >= 2")

    spacing_allowed = {"linear", "log", "geom"}
    if spacing not in spacing_allowed:
        raise InvalidRhsSpecError(
            detail=f"axes[{idx}].spacing must be one of {sorted(spacing_allowed)}"
        )

    coords: list[float] = []
    if spacing == "linear":
        step = (ub - lb) / (resolved_size - 1)
        coords = [lb + step * i for i in range(resolved_size)]
    elif spacing in {"log", "geom"}:
        if lb <= 0 or ub <= 0:
            raise InvalidRhsSpecError(
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
    elif ax_type in {"categorical", "ordinal"}:
        raise InvalidRhsSpecError(detail=f"axes[{idx}] {ax_type} requires coords")
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

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if raw_axes is None:
        return []
    if not isinstance(raw_axes, list):
        raise InvalidRhsSpecError(detail="axes must be a list of axis definitions")

    seen: set[str] = set()
    axes_out: list[dict[str, Any]] = []

    for idx, ax in enumerate(raw_axes):
        axis_out = _normalize_single_axis(
            _ensure_mapping(ax, name=f"axes[{idx}]"), idx=idx, seen=seen
        )
        axes_out.append(axis_out)

    return axes_out
