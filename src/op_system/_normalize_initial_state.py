"""op_system._normalize_initial_state.

Initial-state normalization helpers for op_system RHS specs.

Handles the ``initial_state`` block: scalar string form and the structured
``{shaped, axes}`` form.  All public entry points remain in ``_normalize.py``.
"""

from __future__ import annotations

from collections.abc import Mapping as _MappingABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from op_system._errors import InvalidRhsSpecError
from op_system._helpers import _ensure_mapping
from op_system._templates import (
    _apply_template_substitutions,
    build_axis_lookup,
    expand_selector,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INITIAL_STATE_SHAPED_KEY = "shaped"
_INITIAL_STATE_SHAPED_AXES_KEY = "axes"
_INITIAL_STATE_SHAPED_ALLOWED_KEYS = frozenset((
    _INITIAL_STATE_SHAPED_KEY,
    _INITIAL_STATE_SHAPED_AXES_KEY,
))


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


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

    Returns:
        ``(name, axes_tuple)`` where ``axes_tuple`` preserves the user's
        declared axis order.

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
