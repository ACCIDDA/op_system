"""op_system._reactions.

Per-transition ("reaction") compiled artifacts for ``kind: transitions``
specs.

The deterministic RHS path (``_normalize._build_transition_equations_ir``)
deliberately discards per-transition identity: every transition's flow
contribution is folded into a per-state accumulator and summed into one
combined ``d(state)/dt`` expression, with heavy identity-based sharing and
deduplication (documented against issue #145) that makes reconstructing
per-transition structure from the summed equations impractical (a shared
``Expr`` object can legitimately serve many transitions/cells at once).

This module builds a SEPARATE, independent artifact instead of trying to
extend that hot path: for every *named* transition whose ``to``-side
wildcard axes are a subset of its ``from``-side wildcard axes (i.e. no
axis is "created" transitioning from -> to, and the rate expression
doesn't reference any axis outside that set), it captures:

- a template-form propensity expression, shaped like the ``from``-side
  template (one independent rate per source cell), and
- lightweight axis bookkeeping (which ``from``-axes get pinned to a fixed
  ``to``-coordinate, which get summed away) describing how a firing event
  at a given source cell maps onto the destination state.

This is deliberately narrower than the full transitions grammar (no
``from: null`` source-only transitions, no ``to``-side axis not present on
``from``, no rate expression referencing an axis absent from ``from``) --
see ``docs`` / the originating issue for why: a stochastic/CTMC consumer
needs "how many independent source cells are firing, and where does each
firing land", which these excluded shapes don't have a well-defined answer
for without design work beyond this artifact's scope. Transitions outside
this scope are simply omitted from the artifact tuple, not an error --
mirrors how ``history_requirements`` is built opportunistically elsewhere
in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from op_system._helpers import _get_required_str
from op_system._ir import (
    Apply,
    AxisIndex,
    Expr,
    Subscript,
    iter_subscripts,
    parse_expr_to_ir,
    unparse_ir,
)
from op_system._ir_expand import expand_reduce_pointwise
from op_system._ir_templates import expand_inline_templates
from op_system._templates import (
    PinnedToken,
    WildcardToken,
    parse_selector,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class ReactionArtifactIR:
    """Normalize-time record for one in-scope named transition.

    Attributes:
        name: The transition's own ``name:`` (required -- unnamed
            transitions are never included).
        from_base: State base name the reaction depletes.
        from_axes: Wildcard axes of the ``from`` template, in declaration
            order -- this is the shape of the compiled propensity.
        full_axes: EVERY axis of ``from_base``'s true (unreduced) template,
            in declaration order, whether wildcard or pinned on this
            transition's ``from``-side selector -- i.e. ``from_axes`` plus
            any axis this transition pins on the from-side (e.g. a
            vaccination-dose-progression transition pinned at
            ``vax=unvaccinated`` on ``from``). Used to correctly reference
            ``from_base``'s true shape (see ``propensity_ir_full``) and,
            by a consumer, to build a complete scatter-target index into
            ``to_base`` (assumed to share this same axis order -- see
            ``run_hybrid_ctmc`` in diphtheria_outbreakvacc for the one
            current consumer's validation of that assumption).
        to_base: State base name the reaction replenishes.
        to_axes: Wildcard axes of the ``to`` template, in declaration
            order. Always a subset of ``from_axes`` (enforced at build
            time) -- axes in ``from_axes`` but not here are summed away
            when a firing event is scattered onto the destination.
        pinned: ``(axis, coord)`` pairs for every axis pinned on the
            ``to``-side selector -- both axes that are wildcard on
            ``from`` and pinned on ``to`` (a "collapse to a fixed target"
            transition, e.g. recovery landing in a single ``vax=full``
            stratum regardless of the firing cell's own vax value) and
            axes that are ALSO pinned on ``from`` (a "point-to-point"
            transition between two specific coordinates on the same axis,
            e.g. ``vax=unvaccinated -> vax=partial`` dose progression --
            not a collapse, just a fixed single-cell shift, but the
            scatter target still needs this axis's coordinate fixed).
        from_pinned: ``(axis, coord)`` pairs for every axis pinned on the
            ``from``-side selector -- i.e. every axis in ``full_axes`` but
            not ``from_axes``. Distinct from ``pinned``: for a
            point-to-point transition, the same axis appears in both, but
            with DIFFERENT coordinates (e.g. ``vax=unvaccinated`` here vs.
            ``vax=partial`` in ``pinned``). A consumer needs this to know
            which single coordinate of ``from_base``'s true shape to
            deplete -- the compiled ``propensity_fn`` already reads from
            this coordinate internally (see ``propensity_ir_full``), but
            that isn't otherwise visible from the propensity array's own
            shape (``from_axes``), which carries no trace of it.
        rate_ir_full: Template-form per-capita RATE IR (axes symbolic,
            Reduce nodes resolved) -- the bare rate expression as written
            in the transition's ``rate:`` field, kept for display/
            debugging. NOT what gets compiled as the propensity (see
            ``propensity_ir_full``).
        rate_string: Unparsed ``rate_ir_full``, for display/debugging.
        propensity_ir_full: Template-form PROPENSITY IR -- ``rate *
            from_state``, i.e. the actual per-cell hazard (events per unit
            time), matching the standard CTMC/tau-leaping definition. This
            is what compile-time lowering compiles.
        propensity_ir_reduce: Same, with Reduce nodes preserved (for the
            vector compile path, consistent with the rest of this
            package).
    """

    name: str
    from_base: str
    from_axes: tuple[str, ...]
    full_axes: tuple[str, ...]
    to_base: str
    to_axes: tuple[str, ...]
    pinned: tuple[tuple[str, str], ...]
    from_pinned: tuple[tuple[str, str], ...]
    rate_ir_full: Expr
    rate_string: str
    propensity_ir_full: Expr
    propensity_ir_reduce: Expr


def _in_order_wildcard_axes(tokens: list[Any]) -> list[str]:
    """Return wildcard axis names from parsed selector tokens, in order.

    Returns:
        Axis names for each :class:`WildcardToken` in ``tokens``, in the
        order they appear, without duplicates.
    """
    axes: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if isinstance(tok, WildcardToken) and tok.axis not in seen:
            axes.append(tok.axis)
            seen.add(tok.axis)
    return axes


def build_reaction_artifacts_ir(  # noqa: PLR0914
    transitions_raw: list[Mapping[str, Any]],
    *,
    axes: list[dict[str, Any]],
    axis_lookup: dict[str, list[str]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    time_axis_name: str | None = None,
) -> tuple[ReactionArtifactIR, ...]:
    """Build per-transition reaction artifacts for in-scope named transitions.

    Runs independently of (and has no effect on) the deterministic
    per-state equation construction in ``_build_transition_equations_ir``.

    Returns:
        One :class:`ReactionArtifactIR` per named, in-scope transition, in
        declaration order. Transitions without a ``name:``, source-only
        (``from: null``) transitions, transitions whose ``to``-side
        introduces a wildcard axis absent from ``from``, and transitions
        whose rate references an axis outside the ``from``-side wildcard
        set are silently omitted (not an error -- see module docstring).
    """
    shaped = shaped_params or {}
    out: list[ReactionArtifactIR] = []

    for tr_map in transitions_raw:
        if not isinstance(tr_map, dict):
            continue
        name_s = tr_map.get("name")
        if not isinstance(name_s, str) or not name_s.strip():
            continue  # unnamed: not addressable, skip.

        frm_raw = tr_map.get("from")
        if frm_raw is None:
            continue  # source-only: out of scope for v1, see module docstring.

        to_s = _get_required_str(tr_map, idx=-1, key="to")
        rate_s = _get_required_str(tr_map, idx=-1, key="rate")
        frm_s = _get_required_str(tr_map, idx=-1, key="from")

        frm_base, frm_tokens = parse_selector(frm_s)
        to_base, to_tokens = parse_selector(to_s)

        frm_wc_axes = _in_order_wildcard_axes(list(frm_tokens))
        to_wc_axes = _in_order_wildcard_axes(list(to_tokens))
        frm_wc_set = set(frm_wc_axes)

        # to-side must not introduce a wildcard axis absent from from-side.
        if any(ax not in frm_wc_set for ax in to_wc_axes):
            continue

        ir_rate_raw = parse_expr_to_ir(rate_s, lower_helpers=True)
        # Rate must not reference an axis outside the from-side wildcard
        # set (other than the time axis, which is handled separately by
        # the engine, not baked into the propensity template).
        rate_axes = {
            ix.axis
            for sub in iter_subscripts(ir_rate_raw)
            for ix in sub.indices
            if ix.axis is not None
        }
        if any(
            ax not in frm_wc_set and ax != time_axis_name and ax in axis_lookup
            for ax in rate_axes
        ):
            continue

        rate_ir_reduce = expand_inline_templates(
            ir_rate_raw,
            assignment={},
            shaped_params=shaped,
            axis_lookup=axis_lookup,
        )
        rate_ir_full = expand_reduce_pointwise(
            rate_ir_reduce,
            axes=list(axes),
            shaped_params=shaped,
            lhs_assignment={},
            axis_coords=axis_lookup,
        )

        # Propensity = rate * from_state (the actual per-cell hazard), NOT
        # the bare rate -- mirrors the deterministic path's tpl_flow_full
        # construction in _normalize._build_transition_equations_ir.
        #
        # from_sub must reference from_base's TRUE full shape, not just its
        # wildcard axes: a from-side PinnedToken (e.g. `S[age, vax=u, loc]`)
        # still needs that coordinate baked into the Subscript, or the
        # reference silently points at the wrong (or a shape-mismatched)
        # slice. full_axes -- every token's axis, wildcard or pinned, in
        # the selector's own declared order -- is exactly from_base's true
        # axis order (a well-formed selector mentions every axis of the
        # state it references), so build indices from ALL of frm_tokens,
        # not just the wildcard subset.
        full_axes = tuple(tok.axis for tok in frm_tokens)
        from_sub = Subscript(
            name=frm_base,
            indices=tuple(
                AxisIndex(
                    axis=tok.axis,
                    coord=(tok.coord if isinstance(tok, PinnedToken) else None),
                )
                for tok in frm_tokens
            ),
        )
        propensity_ir_full = Apply(op="*", args=(rate_ir_full, from_sub))
        propensity_ir_reduce = Apply(op="*", args=(rate_ir_reduce, from_sub))

        # Every to-side pinned axis needs a fixed scatter-target coordinate,
        # not just ones that are wildcard on from (the "collapse" case) --
        # an axis pinned on BOTH sides (e.g. vax=unvaccinated -> vax=partial
        # dose progression) is a "point-to-point" shift on that axis, not a
        # collapse, but the target coordinate still needs recording.
        pinned = tuple(
            (tok.axis, tok.coord) for tok in to_tokens if isinstance(tok, PinnedToken)
        )
        # Every from-side pinned axis (full_axes minus from_axes) needs its
        # own coordinate recorded separately from `pinned` -- for a
        # point-to-point transition the same axis is pinned on both sides
        # but to DIFFERENT coordinates (e.g. vax=unvaccinated here vs.
        # vax=partial in `pinned`), so this can't be derived from `pinned`.
        from_pinned = tuple(
            (tok.axis, tok.coord) for tok in frm_tokens if isinstance(tok, PinnedToken)
        )

        out.append(
            ReactionArtifactIR(
                name=name_s,
                from_base=frm_base,
                from_axes=tuple(frm_wc_axes),
                full_axes=full_axes,
                to_base=to_base,
                to_axes=tuple(to_wc_axes),
                pinned=pinned,
                from_pinned=from_pinned,
                rate_ir_full=rate_ir_full,
                rate_string=unparse_ir(rate_ir_full),
                propensity_ir_full=propensity_ir_full,
                propensity_ir_reduce=propensity_ir_reduce,
            )
        )

    return tuple(out)


__all__ = ["ReactionArtifactIR", "build_reaction_artifacts_ir"]
