"""op_system._constraints.

Cross-axis constraint normalization: parse, validate, and store
allow/exclude rules for coordinate-combination filtering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from op_system._errors import _raise_invalid_rhs_spec
from op_system._helpers import _ensure_mapping


class ConstraintRule(NamedTuple):
    """Validated constraint rule produced by ``_normalize_constraints``."""

    axes: tuple[str, ...]
    mode: str
    rules: tuple[dict[str, list[str]], ...]


def _validate_constraint_axes(
    rule_axes_raw: Sequence[str] | object,
    *,
    idx: int,
    axis_names: set[str],
) -> tuple[list[str], set[str]]:
    """Validate and return the axes list for a single constraint rule.

    Returns:
        Tuple of (ordered axis name list, axis name set).

    Examples:
        >>> axes, seen = _validate_constraint_axes(
        ...     ["age", "vax"], idx=0, axis_names={"age", "vax"}
        ... )
        >>> axes
        ['age', 'vax']
        >>> sorted(seen)
        ['age', 'vax']
    """
    if not isinstance(rule_axes_raw, (list, tuple)) or len(rule_axes_raw) < 2:
        _raise_invalid_rhs_spec(
            detail=(
                f"constraints[{idx}].axes must be a list of at least two axis names"
            )
        )
    rule_axes: list[str] = []
    seen: set[str] = set()
    for j, ax_name in enumerate(rule_axes_raw):
        if not isinstance(ax_name, str) or not (ax_s := ax_name.strip()):
            _raise_invalid_rhs_spec(
                detail=f"constraints[{idx}].axes[{j}] must be a non-empty string"
            )
        if ax_s not in axis_names:
            _raise_invalid_rhs_spec(
                detail=f"constraints[{idx}].axes references unknown axis {ax_s!r}"
            )
        if ax_s in seen:
            _raise_invalid_rhs_spec(
                detail=f"constraints[{idx}].axes contains duplicate axis {ax_s!r}"
            )
        seen.add(ax_s)
        rule_axes.append(ax_s)
    return rule_axes, seen


def _resolve_constraint_mode(
    entry_map: Mapping[str, Any], *, idx: int
) -> tuple[str, list[Any]]:
    """Determine allow/exclude mode and return raw rules list.

    Returns:
        Tuple of (mode string, raw rule list).
    """
    has_allow = "allow" in entry_map
    has_exclude = "exclude" in entry_map
    if has_allow and has_exclude:
        _raise_invalid_rhs_spec(
            detail=(
                f"constraints[{idx}] must specify either 'allow' or 'exclude', not both"
            )
        )
    if not has_allow and not has_exclude:
        _raise_invalid_rhs_spec(
            detail=f"constraints[{idx}] must specify 'allow' or 'exclude'"
        )
    mode = "allow" if has_allow else "exclude"
    raw_rules = entry_map[mode]
    if not isinstance(raw_rules, list) or not raw_rules:
        _raise_invalid_rhs_spec(
            detail=f"constraints[{idx}].{mode} must be a non-empty list"
        )
    return mode, raw_rules


def _validate_constraint_rule(
    rule: object,
    *,
    label: str,
    rule_axis_set: set[str],
    axis_lookup: Mapping[str, set[str]],
) -> dict[str, list[str]]:
    """Validate a single rule mapping inside a constraint entry.

    Returns:
        Validated mapping of axis name to list of coordinate strings.
    """
    rule_map = _ensure_mapping(rule, name=label)
    validated: dict[str, list[str]] = {}
    for key, val in rule_map.items():
        key_s = str(key).strip()
        if key_s not in rule_axis_set:
            _raise_invalid_rhs_spec(
                detail=(
                    f"{label} references axis {key_s!r} not in "
                    f"constraint axes {sorted(rule_axis_set)}"
                )
            )
        if isinstance(val, str):
            coords = [val.strip()]
        elif isinstance(val, (list, tuple)):
            coords = [str(v).strip() for v in val]
        else:
            coords = [str(val).strip()]
        for coord in coords:
            if coord not in axis_lookup[key_s]:
                _raise_invalid_rhs_spec(
                    detail=(
                        f"{label} references unknown coord {coord!r} for axis {key_s!r}"
                    )
                )
        validated[key_s] = coords
    if not validated:
        _raise_invalid_rhs_spec(detail=f"{label} must specify at least one axis")
    return validated


def _normalize_constraints(
    raw_constraints: object,
    *,
    axes: list[dict[str, Any]],
) -> list[ConstraintRule]:
    """Normalize cross-axis constraint rules.

    Each rule references two or more axes and declares either an ``allow``
    list (allowlist of valid coordinate combinations) or an ``exclude``
    list (blocklist of invalid combinations).

    Returns:
        List of validated ``ConstraintRule`` named tuples.
    """
    if raw_constraints is None:
        return []
    if not isinstance(raw_constraints, list):
        _raise_invalid_rhs_spec(detail="constraints must be a list")
    if not raw_constraints:
        return []

    axis_lookup: dict[str, set[str]] = {
        ax["name"]: {str(c) for c in ax.get("coords", [])} for ax in axes
    }
    axis_names = set(axis_lookup)
    out: list[ConstraintRule] = []

    for idx, entry in enumerate(raw_constraints):
        entry_map = _ensure_mapping(entry, name=f"constraints[{idx}]")
        rule_axes, rule_axis_set = _validate_constraint_axes(
            entry_map.get("axes"), idx=idx, axis_names=axis_names
        )
        mode, raw_rules = _resolve_constraint_mode(entry_map, idx=idx)

        validated_rules = [
            _validate_constraint_rule(
                rule,
                label=f"constraints[{idx}].{mode}[{r_idx}]",
                rule_axis_set=rule_axis_set,
                axis_lookup=axis_lookup,
            )
            for r_idx, rule in enumerate(raw_rules)
        ]

        out.append(
            ConstraintRule(
                axes=tuple(rule_axes),
                mode=mode,
                rules=tuple(validated_rules),
            )
        )

    return out
