"""op_system._normalize_chains.

Chain helper and coord-shift expansion for op_system RHS normalization.

Contains ``_apply_expr_chains``, ``_apply_transition_chains``,
``_apply_coord_shifts``, and their supporting private helpers.
All public entry points remain in ``_normalize.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Mapping

from op_system._errors import InvalidRhsSpecError
from op_system._helpers import _ensure_mapping, _get_required_str
from op_system._templates import (
    PinnedToken,
    _sanitize_fragment,
    expand_apply_to,
    parse_selector,
)

#: Private transition key naming where a normalized transition came from in
#: the spec (``transitions[2]``, ``chain[0].forward[1]``), so reaction
#: coverage records can point users at the entry to fix.
ORIGIN_KEY = "_op_system_origin"

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


def _normalize_catalysts(raw: object, *, field: str) -> list[dict[str, Any]] | None:
    """Validate an optional ``catalysts`` list for generated transitions.

    Catalysts are the molecular reactants beyond the consumed source, which
    the generator adds itself. Shapes and selectors are validated with the
    generated transition's ``reactants``.

    Returns:
        The catalyst mappings, or ``None`` when ``raw`` is absent.

    Raises:
        InvalidRhsSpecError: If ``raw`` is not a list of mappings.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise InvalidRhsSpecError(detail=f"{field} must be a list")
    return [
        dict(_ensure_mapping(item, name=f"{field}[{i}]")) for i, item in enumerate(raw)
    ]


def _with_reactants(
    transition: dict[str, Any], catalysts: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Declare a generated transition's complete reactants.

    Returns:
        ``transition``, with ``reactants`` set to its consumed source at
        order one plus ``catalysts`` when catalysts were declared.
    """
    if catalysts is not None:
        transition["reactants"] = [
            {"state": transition["from"], "order": 1},
            *catalysts,
        ]
    return transition


def _apply_transition_chains(
    *,
    chains: list[Any],
    state_raw: list[str],
    transitions_raw: list[dict[str, Any]],
    state_set: set[str],
) -> None:
    """Apply chain helper for transitions kind by appending transitions.

    Generated transitions are named ``{base}_entry``,
    ``{base}_advance_{k}`` (stage ``k`` to ``k + 1``), and ``{base}_exit``,
    so each publishes a reaction artifact. ``entry.catalysts`` and the
    chain's ``catalysts`` declare their reactants beyond the consumed stage.
    """
    for c_idx, chain in enumerate(chains):
        stage_names, forward_rates, entry_cfg, exit_cfg = _validate_chain_entry(
            chain=chain,
            idx=c_idx,
            state_set=state_set,
        )
        base = parse_selector(chain["name"])[0]
        stage_catalysts = _normalize_catalysts(
            chain.get("catalysts"), field=f"chain[{c_idx}].catalysts"
        )

        for stage_name in stage_names:
            if stage_name not in state_set:
                state_raw.append(stage_name)
                state_set.add(stage_name)

        if entry_cfg is not None:
            entry_from, entry_rate = entry_cfg
            transitions_raw.append(
                _with_reactants(
                    {
                        "name": f"{base}_entry",
                        "from": entry_from,
                        "to": stage_names[0],
                        "rate": entry_rate,
                        ORIGIN_KEY: f"chain[{c_idx}].entry",
                    },
                    _normalize_catalysts(
                        chain["entry"].get("catalysts"),
                        field=f"chain[{c_idx}].entry.catalysts",
                    ),
                )
            )

        transitions_raw.extend(
            _with_reactants(
                {
                    "name": f"{base}_advance_{i + 1}",
                    "from": stage_names[i],
                    "to": stage_names[i + 1],
                    "rate": forward_rates[i],
                    ORIGIN_KEY: f"chain[{c_idx}].forward[{i}]",
                },
                stage_catalysts,
            )
            for i in range(len(stage_names) - 1)
        )

        if exit_cfg is not None:
            sink_s, sink_rate = exit_cfg
            transitions_raw.append(
                _with_reactants(
                    {
                        "name": f"{base}_exit",
                        "from": stage_names[-1],
                        "to": sink_s,
                        "rate": sink_rate or forward_rates[-1],
                        ORIGIN_KEY: f"chain[{c_idx}].exit",
                    },
                    stage_catalysts,
                )
            )


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
    _validate_coord_shift_common(tr)
    return axis_name, from_coord, to_coord, tr["apply_to"], tr["rate"].strip()


def _validate_coord_shift_common(tr: Mapping[str, Any]) -> None:
    """Validate the ``apply_to`` and ``rate`` fields shared by both forms.

    Raises:
        InvalidRhsSpecError: If either field is missing or malformed.
    """
    apply_to = tr.get("apply_to")
    if not isinstance(apply_to, list) or not apply_to:
        raise InvalidRhsSpecError(
            detail="coord_shift requires a non-empty 'apply_to' list",
        )

    rate_s = tr.get("rate")
    if not isinstance(rate_s, str) or not rate_s.strip():
        raise InvalidRhsSpecError(detail="coord_shift requires a 'rate' string")


_AXIS_SHIFT_KEYS = frozenset({"axis", "step", "rate", "boundary"})
_AXIS_SHIFT_BOUNDARIES = ("absorb", "stay", "error")

#: Private transition key carrying a validated axis-wide shift from
#: ``_apply_coord_shifts`` to equation synthesis and reaction artifacts.
COORD_SHIFT_KEY = "_op_system_coord_shift"


def _is_axis_wide_shift(shift_spec: object) -> TypeGuard[dict[str, Any]]:
    """Return whether a ``coord_shift`` value uses the axis-wide form.

    The pairwise form is ``{axis_name: "from -> to"}``. An axis that is
    itself named ``axis`` keeps that meaning, distinguished by the arrow.

    Returns:
        ``True`` for ``{axis: ..., step: ..., ...}`` mappings.
    """
    if not isinstance(shift_spec, dict) or "axis" not in shift_spec:
        return False
    value = shift_spec["axis"]
    return not (len(shift_spec) == 1 and isinstance(value, str) and "->" in value)


class _AxisShift(NamedTuple):
    """Validated axis-wide ``coord_shift``: every coordinate moves ``step`` bins.

    ``boundary`` says what happens to source bins whose destination
    ``index + step`` falls outside the axis: ``absorb`` removes the mass
    from the system and ``stay`` leaves it in place.
    """

    axis: str
    step: int
    boundary: str

    @classmethod
    def from_mapping(
        cls,
        shift_spec: Mapping[str, Any],
        *,
        axis_lookup: Mapping[str, list[str]],
    ) -> _AxisShift:
        """Parse ``{axis, step, boundary}`` (``rate`` is hoisted already).

        Returns:
            The validated shift.

        Raises:
            InvalidRhsSpecError: If a field is missing, unknown, or invalid,
                or if ``boundary`` is ``error`` (the default).
        """
        unknown = set(shift_spec) - _AXIS_SHIFT_KEYS
        if unknown:
            raise InvalidRhsSpecError(
                detail=f"coord_shift has unknown fields: {sorted(unknown)!r}"
            )
        axis = shift_spec["axis"]
        if not isinstance(axis, str) or axis not in axis_lookup:
            raise InvalidRhsSpecError(
                detail=f"coord_shift axis {axis!r} is not defined",
            )
        n_coords = len(axis_lookup[axis])
        step = shift_spec.get("step", 1)
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step == 0
            or abs(step) >= n_coords
        ):
            raise InvalidRhsSpecError(
                detail=(
                    f"coord_shift step on axis {axis!r} must be a nonzero integer "
                    f"smaller in magnitude than the axis length {n_coords}; "
                    f"got {step!r}"
                ),
            )
        boundary = shift_spec.get("boundary", "error")
        if boundary not in _AXIS_SHIFT_BOUNDARIES:
            raise InvalidRhsSpecError(
                detail=(
                    f"coord_shift boundary must be one of "
                    f"{list(_AXIS_SHIFT_BOUNDARIES)}; got {boundary!r}"
                ),
            )
        if boundary == "error":
            edge = "last" if step > 0 else "first"
            raise InvalidRhsSpecError(
                detail=(
                    f"coord_shift step {step:+d} on axis {axis!r} moves the {edge} "
                    f"{abs(step)} coordinate(s) off the axis; set boundary to "
                    "'absorb' (mass leaves the system) or 'stay' (mass remains)"
                ),
            )
        return cls(axis=axis, step=step, boundary=boundary)

    def tag(self) -> str:
        """Return a name-safe step label such as ``p1`` or ``m2``.

        Returns:
            ``p{step}`` for forward shifts and ``m{-step}`` for backward ones.
        """
        return f"p{self.step}" if self.step > 0 else f"m{-self.step}"

    def shift_matrix(self, n_coords: int) -> tuple[tuple[float, ...], ...]:
        """Return the one-hot source-row, target-column shift matrix.

        Returns:
            ``M[i][j] == 1.0`` exactly when ``j == i + step`` is on the axis.
        """
        return tuple(
            tuple(1.0 if j == i + self.step else 0.0 for j in range(n_coords))
            for i in range(n_coords)
        )

    def keep_mask(self, n_coords: int) -> tuple[float, ...]:
        """Return ``1.0`` for source bins whose destination is on the axis.

        Returns:
            One weight per coordinate along the shifted axis.
        """
        return tuple(
            1.0 if 0 <= i + self.step < n_coords else 0.0 for i in range(n_coords)
        )


def coord_shift_constant_names(axis: str, tag: str) -> tuple[str, str]:
    """Return synthesized ``(shift_matrix, keep_mask)`` parameter names.

    Returns:
        Shaped-parameter names for the ``[axis, axis]`` shift matrix and the
        ``[axis]`` in-domain source mask.
    """
    return (
        f"__op_system_shift__{axis}__{tag}",
        f"__op_system_shift_keep__{axis}__{tag}",
    )


def _hoist_coord_shift_rates(transitions_raw: list[Any]) -> None:
    """Move an axis-wide ``coord_shift.rate`` to the transition's ``rate``.

    Runs before shaped-parameter discovery and time-axis stripping, which
    both read only top-level rates. Entries are replaced, not mutated, so a
    caller's nested ``coord_shift`` mapping is left untouched.

    Raises:
        InvalidRhsSpecError: If both places declare a rate.
    """
    for idx, tr in enumerate(transitions_raw):
        if not isinstance(tr, dict):
            continue
        shift_spec = tr.get("coord_shift")
        if not _is_axis_wide_shift(shift_spec) or "rate" not in shift_spec:
            continue
        if "rate" in tr:
            raise InvalidRhsSpecError(
                detail=(
                    f"transitions[{idx}] declares a rate both in coord_shift "
                    "and on the transition; keep one"
                )
            )
        transitions_raw[idx] = {
            **tr,
            "rate": shift_spec["rate"],
            "coord_shift": {k: v for k, v in shift_spec.items() if k != "rate"},
        }


def _wildcard_template_axes(
    base: str,
    state_template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> list[str] | None:
    """Return ``base``'s axes when it has one all-wildcard state template.

    Returns:
        Axis names in declaration order, or ``None`` when ``base`` has no
        unique template or that template pins a coordinate.
    """
    prefix = f"{base}["
    matches = [
        k for k in state_template_map if k.startswith(prefix) and k.endswith("]")
    ]
    if len(matches) != 1:
        return None
    inside = matches[0][len(prefix) : -1]
    tokens = [t.strip() for t in inside.split(",") if t.strip()]
    if any("=" in tok for tok in tokens):
        return None
    return tokens


def _expand_axis_wide_shift(
    tr: Mapping[str, Any],
    *,
    axis_lookup: dict[str, list[str]],
    state_template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> list[dict[str, Any]]:
    """Emit one template-form shift transition per ``apply_to`` base.

    Each output carries :data:`COORD_SHIFT_KEY`; equation synthesis lowers it
    once per template and reaction-artifact construction publishes it as one
    offset reaction. A named entry yields names ``{name}_{base}``.

    Returns:
        Template-form transition dicts.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    shift = _AxisShift.from_mapping(tr["coord_shift"], axis_lookup=axis_lookup)
    _validate_coord_shift_common(tr)
    catalysts = _coord_shift_catalysts(tr)
    name = tr.get("name")
    out: list[dict[str, Any]] = []
    for base in expand_apply_to(
        tr["apply_to"],
        axis_lookup=axis_lookup,
        context=f"coord_shift[{shift.axis}].apply_to",
    ):
        axes = _wildcard_template_axes(base, state_template_map)
        if axes is None or shift.axis not in axes:
            raise InvalidRhsSpecError(
                detail=(
                    f"axis-wide coord_shift apply_to state {base!r} needs one "
                    f"state template with a wildcard {shift.axis!r} axis and no "
                    "pinned coordinates"
                ),
            )
        selector = f"{base}[{', '.join(axes)}]"
        entry: dict[str, Any] = {
            "from": selector,
            "to": selector,
            "rate": tr["rate"].strip(),
            COORD_SHIFT_KEY: shift,
        }
        if isinstance(name, str) and name.strip():
            entry["name"] = f"{name.strip()}_{base}"
        out.append(_with_reactants(entry, catalysts))
    return out


def _coord_shift_catalysts(tr: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Read a ``coord_shift`` entry's ``catalysts``, rejecting ``reactants``.

    One entry generates a transition per ``apply_to`` state, so it cannot
    name each one's consumed source; the generator adds it.

    Returns:
        The catalyst mappings, or ``None`` when none are declared.

    Raises:
        InvalidRhsSpecError: If the entry declares ``reactants``.
    """
    if "reactants" in tr:
        raise InvalidRhsSpecError(
            detail=(
                "coord_shift does not accept 'reactants'; declare reactants "
                "beyond the shifted state as 'catalysts'"
            ),
        )
    return _normalize_catalysts(tr.get("catalysts"), field="coord_shift.catalysts")


def _discover_coord_shift_constants(
    transitions_raw: list[Any],
    *,
    axis_lookup: Mapping[str, list[str]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Collect synthesized shift matrices and masks for axis-wide shifts.

    Returns:
        ``(shaped_axes, values)`` keyed by synthesized parameter name.
    """
    shaped: dict[str, tuple[str, ...]] = {}
    values: dict[str, Any] = {}
    for tr in transitions_raw:
        shift = tr.get(COORD_SHIFT_KEY) if isinstance(tr, dict) else None
        if not isinstance(shift, _AxisShift):
            continue
        matrix_name, keep_name = coord_shift_constant_names(shift.axis, shift.tag())
        if matrix_name in values:
            continue
        n_coords = len(axis_lookup[shift.axis])
        shaped[matrix_name] = (shift.axis, shift.axis)
        shaped[keep_name] = (shift.axis,)
        values[matrix_name] = shift.shift_matrix(n_coords)
        values[keep_name] = shift.keep_mask(n_coords)
    return shaped, values


def _apply_coord_shifts(
    *,
    transitions_raw: list[dict[str, Any]],
    state_expanded: list[str],
    axes: list[dict[str, Any]],
    state_template_map: Mapping[str, list[tuple[str, dict[str, str]]]] | None = None,
) -> None:
    """Expand ``coord_shift`` entries into concrete transitions in-place.

    A pairwise ``coord_shift`` entry describes movement between two
    coordinates of one axis for a set of states.  When ``state_template_map``
    is supplied and the ``apply_to`` base has a unique all-wildcard state
    template, the entry is replaced by a single template-form transition
    (selectors with the shifted axis pinned and remaining axes left as
    wildcards); otherwise it falls back to one concrete transition per
    matching cell.  Template-form output lets downstream synthesis lift the
    transition into a vectorizable Reduce.

    An axis-wide entry (``{axis, step, boundary}``) instead becomes one
    template-form transition per ``apply_to`` base, marked with
    :data:`COORD_SHIFT_KEY`, so it compiles once per template rather than
    once per coordinate pair.

    Args:
        transitions_raw: Mutable transition list — ``coord_shift`` entries are
            replaced by concrete transition dicts.
        state_expanded: Expanded state names (used to discover axes per state).
        axes: Normalized axis definitions.
        state_template_map: Mapping of template-keyed entries (e.g.
            ``"X[age,vax,loc,imm]"``) to expanded ``(name, coord_map)`` pairs.
            Used to derive axis order for template-form emission.
    """
    axis_lookup: dict[str, list[str]] = {
        ax["name"]: [str(c) for c in ax.get("coords", [])] for ax in axes
    }
    tmpl_map: Mapping[str, list[tuple[str, dict[str, str]]]] = state_template_map or {}

    i = 0
    while i < len(transitions_raw):
        tr = transitions_raw[i]
        if "coord_shift" not in tr:
            i += 1
            continue
        if _is_axis_wide_shift(tr["coord_shift"]):
            shifted = _expand_axis_wide_shift(
                tr, axis_lookup=axis_lookup, state_template_map=tmpl_map
            )
        else:
            shifted = _expand_pairwise_shift(
                tr,
                axis_lookup=axis_lookup,
                state_template_map=tmpl_map,
                state_expanded=state_expanded,
            )
        if ORIGIN_KEY in tr:
            for entry in shifted:
                entry[ORIGIN_KEY] = tr[ORIGIN_KEY]
        transitions_raw[i : i + 1] = shifted
        i += len(shifted)


def _expand_pairwise_shift(
    tr: dict[str, Any],
    *,
    axis_lookup: dict[str, list[str]],
    state_template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
    state_expanded: list[str],
) -> list[dict[str, Any]]:
    """Expand one ``{axis: "from -> to"}`` entry for every ``apply_to`` base.

    Returns:
        One template-form transition per base where possible, otherwise one
        concrete transition per matching cell.
    """
    axis_name, from_coord, to_coord, apply_to, rate_s = _validate_coord_shift_entry(
        tr, axis_lookup
    )
    catalysts = _coord_shift_catalysts(tr)
    name = tr.get("name")
    prefix = name.strip() if isinstance(name, str) and name.strip() else None
    concrete: list[dict[str, Any]] = []
    from_frag = f"{axis_name}_{_sanitize_fragment(from_coord)}"
    to_frag = f"{axis_name}_{_sanitize_fragment(to_coord)}"
    for base in expand_apply_to(
        apply_to,
        axis_lookup=axis_lookup,
        context=f"coord_shift[{axis_name}].apply_to",
    ):
        templated = _build_templated_coord_shift_transition(
            base=base,
            axis_name=axis_name,
            from_coord=from_coord,
            to_coord=to_coord,
            rate_s=rate_s,
            state_template_map=state_template_map,
        )
        if templated is not None:
            if prefix is not None:
                templated["name"] = f"{prefix}_{base}"
            concrete.append(_with_reactants(templated, catalysts))
            continue
        # Per-cell fallback: concrete cell names are not reactant selectors,
        # so these keep the consumed-source fallback.
        for cell in _expand_coord_shift_for_base(
            base=base,
            from_frag=from_frag,
            to_frag=to_frag,
            rate_s=rate_s,
            state_expanded=state_expanded,
        ):
            if prefix is not None:
                cell["name"] = f"{prefix}_{cell['from']}"
            concrete.append(cell)
    return concrete


def _build_templated_coord_shift_transition(  # ruff: ignore[too-many-arguments]
    *,
    base: str,
    axis_name: str,
    from_coord: str,
    to_coord: str,
    rate_s: str,
    state_template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
) -> dict[str, Any] | None:
    """Build a single template-form coord_shift transition for ``base``.

    Returns a transition dict with selectors of the form
    ``"{base}[ax1, ..., axis_name={coord}, ...]"`` when ``base`` has a unique
    state-template entry whose tokens are all wildcards and include
    ``axis_name``; returns ``None`` otherwise (caller falls back to per-cell
    expansion).

    Returns:
        Template-form transition dict, or ``None`` if no unique all-wildcard
        template for ``base`` exists (or ``axis_name`` is not among its axes).
    """
    axes = _wildcard_template_axes(base, state_template_map)
    if axes is None or axis_name not in axes:
        return None

    from_parts = [f"{ax}={from_coord}" if ax == axis_name else ax for ax in axes]
    to_parts = [f"{ax}={to_coord}" if ax == axis_name else ax for ax in axes]
    return {
        "from": f"{base}[{', '.join(from_parts)}]",
        "to": f"{base}[{', '.join(to_parts)}]",
        "rate": rate_s,
    }


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
