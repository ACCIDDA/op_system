"""op_system._normalize_chains.

Chain helper and coord-shift expansion for op_system RHS normalization.

Contains ``_apply_expr_chains``, ``_apply_transition_chains``,
``_apply_coord_shifts``, and their supporting private helpers.
All public entry points remain in ``_normalize.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from op_system._errors import InvalidRhsSpecError
from op_system._helpers import _ensure_mapping, _get_required_str
from op_system._templates import (
    PinnedToken,
    _sanitize_fragment,
    build_axis_lookup,
    expand_apply_to,
    parse_selector,
)


# ---------------------------------------------------------------------------
# Chain normalization helpers
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
