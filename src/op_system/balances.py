"""Conditional scalar balance identities derived from reaction topology.

This analysis uses the compiled unit-transfer incidence structure, without
sampling propensities or inferring physical conservation from state names.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, NamedTuple

from op_system.compile import CompiledRhs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from op_system._reactions import ReactionGap
    from op_system._scientific_meta import AxisDeclaration, StateDeclaration
    from op_system.compile import CompiledReaction

_ASSUMPTIONS = (
    (
        "The supplied reaction artifacts and metadata faithfully describe "
        "the compiled RHS."
    ),
    (
        "Execution uses these represented channels without added state updates, "
        "fluxes, clipping, projections, resets or impulses."
    ),
    (
        "Coordinates summed in a law use compatible units and representations; "
        "labels are retained without dimensional verification or unit conversion."
    ),
    (
        "A coordinate-sum identity does not establish physical mass conservation, "
        "a spatial integral or numerical integration accuracy."
    ),
)


class BalanceCoverageGap(NamedTuple):
    """An omission or unsupported contribution preventing issued laws.

    Attributes:
        origin: Location in compiler metadata or reaction artifacts.
        reason: Machine-readable coverage limitation.
        name: Associated reaction/state name when known.
        reaction_gap: Original compiler omission, retaining its selectors/reason.
    """

    origin: str
    reason: str
    name: str | None = None
    reaction_gap: ReactionGap | None = None


class ReactionBalanceSource(NamedTuple):
    """One external flux term in a component's balance equation.

    Attributes:
        reaction_name: Source-only reaction whose propensity supplies the flux.
        coefficient: Exact multiplier of that propensity in the balance.
        target_state: Compiled scalar state receiving the source.
    """

    reaction_name: str
    coefficient: int
    target_state: str


class ReactionBalanceLaw(NamedTuple):
    """A conditional identity ``d(coefficients @ state)/dt = source fluxes``.

    Attributes:
        name: Component label stable within this report's compiled state order.
        coefficients: Exact integer weights in the report's state order.
        reaction_names: All reaction identities covered by the analysis.
        sources: Nonzero external terms, resolved by reaction name.
        assumptions: Conditions needed to apply this structural identity.
        origin: Compiler unit-transfer derivation, rather than a user assertion.
    """

    name: str
    coefficients: tuple[int, ...]
    reaction_names: tuple[str, ...]
    sources: tuple[ReactionBalanceSource, ...]
    assumptions: tuple[str, ...]
    origin: Literal["compiled_unit_transfers"] = "compiled_unit_transfers"

    @property
    def closed(self) -> bool:
        """Whether this identity has no represented external source term."""
        return not self.sources


class ReactionBalanceReport(NamedTuple):
    """Immutable scalar balance analysis with explicit scope and coverage.

    Attributes:
        state_names: Coefficient-vector coordinate order.
        laws: Issued identities; empty if any coverage gap exists.
        coverage_gaps: Unsupported/omitted dynamics or inconsistent metadata.
        state_meta: State declarations in compiled state order; unknowns stay absent.
        axis_meta: Retained scientific axis declarations; no weights are inferred.
        assumptions: Shared applicability conditions, including unknown physical units.
        scope: Bounded analysis capability, excluding general stoichiometry/meshes.
    """

    state_names: tuple[str, ...]
    laws: tuple[ReactionBalanceLaw, ...]
    coverage_gaps: tuple[BalanceCoverageGap, ...]
    state_meta: tuple[tuple[str, StateDeclaration], ...]
    axis_meta: tuple[tuple[str, AxisDeclaration], ...]
    assumptions: tuple[str, ...]
    scope: Literal["scalar_unit_transfers"] = "scalar_unit_transfers"

    @property
    def fully_covered(self) -> bool:
        """Whether this bounded analysis covers the supplied structure."""
        return not self.coverage_gaps


def _transition_coverage(compiled: CompiledRhs) -> list[BalanceCoverageGap]:
    """Check artifact identities against normalized scalar transition metadata.

    Returns:
        Explicit coverage limitations, without assigning missing fluxes zero.
    """
    if compiled.reaction_gaps:
        return []
    transitions = compiled.meta.get("transitions")
    if not isinstance(transitions, (list, tuple)):
        return [BalanceCoverageGap("transitions", "missing_transition_metadata")]
    names = [
        entry.get("name") if isinstance(entry, Mapping) else None
        for entry in transitions
    ]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        return [BalanceCoverageGap("transitions", "unnamed_transition")]
    expected = Counter(names)
    actual = Counter(reaction.name for reaction in compiled.reactions)
    if expected != actual:
        return [BalanceCoverageGap("transitions", "transition_artifact_mismatch")]
    return []


def _reaction_coverage(
    reaction: CompiledReaction, states: set[str], index: int
) -> list[BalanceCoverageGap]:
    """Validate the supported scalar incidence-column interpretation.

    Returns:
        Explicit coverage limitations, without assigning missing fluxes zero.
    """
    origin = f"reactions[{index}]"
    gaps = []
    if reaction.from_base is not None and reaction.from_base not in states:
        gaps.append(BalanceCoverageGap(origin, "unknown_source_state", reaction.name))
    if reaction.to_base not in states:
        gaps.append(BalanceCoverageGap(origin, "unknown_target_state", reaction.name))
    if any((
        reaction.from_axes,
        reaction.full_axes,
        reaction.to_axes,
        reaction.to_full_axes,
        reaction.sum_axes,
        reaction.pinned,
        reaction.from_pinned,
        reaction.offsets,
        reaction.routed_axes,
    )):
        gaps.append(BalanceCoverageGap(origin, "non_scalar_reaction", reaction.name))
    return gaps


def _coverage(compiled: CompiledRhs) -> list[BalanceCoverageGap]:
    """Collect structural limitations before attempting any balance derivation.

    Returns:
        Explicit coverage limitations, without assigning missing fluxes zero.
    """
    gaps = [
        BalanceCoverageGap(gap.origin, gap.reason, gap.name, gap)
        for gap in compiled.reaction_gaps
    ]
    states = set(compiled.state_names)
    if (
        not states
        or len(states) != len(compiled.state_names)
        or any(not name.strip() for name in states)
    ):
        gaps.append(BalanceCoverageGap("state_names", "invalid_state_identities"))
    shapes = compiled.template_shapes
    if shapes is None:
        gaps.append(BalanceCoverageGap("template_shapes", "missing_state_layout"))
    elif set(shapes) != states:
        gaps.append(BalanceCoverageGap("template_shapes", "state_layout_mismatch"))
    elif any(shapes.values()):
        gaps.append(BalanceCoverageGap("template_shapes", "non_scalar_state_layout"))
    if set(compiled.state_meta) - states:
        gaps.append(BalanceCoverageGap("state_meta", "unknown_state_metadata"))
    if compiled.operators or compiled.meta.get("operators"):
        gaps.append(BalanceCoverageGap("operators", "operator_dynamics"))
    gaps.extend(
        BalanceCoverageGap(name, "additional_declared_dynamics")
        for name in ("sources", "couplings", "constraints")
        if name in compiled.meta
    )
    names = [reaction.name for reaction in compiled.reactions]
    if any(not name.strip() for name in names) or len(names) != len(set(names)):
        gaps.append(BalanceCoverageGap("reactions", "invalid_reaction_identities"))
    gaps.extend(_transition_coverage(compiled))
    for index, reaction in enumerate(compiled.reactions):
        gaps.extend(_reaction_coverage(reaction, states, index))
    return gaps


def _components(
    state_names: tuple[str, ...], reactions: Sequence[CompiledReaction]
) -> tuple[frozenset[str], ...]:
    """Find exact incidence components, keeping compiled first-state order.

    Returns:
        Components in compiled first-state order.
    """
    adjacency: dict[str, set[str]] = {name: set() for name in state_names}
    for reaction in reactions:
        if reaction.from_base is not None:
            adjacency[reaction.from_base].add(reaction.to_base)
            adjacency[reaction.to_base].add(reaction.from_base)
    remaining = set(state_names)
    components = []
    for seed in state_names:
        if seed not in remaining:
            continue
        remaining.remove(seed)
        component = {seed}
        stack = [seed]
        while stack:
            for neighbor in adjacency[stack.pop()] - component:
                component.add(neighbor)
                remaining.remove(neighbor)
                stack.append(neighbor)
        components.append(frozenset(component))
    return tuple(components)


def _scientific_coverage(
    compiled: CompiledRhs, components: Sequence[frozenset[str]]
) -> list[BalanceCoverageGap]:
    """Keep unknown labels conditional, rejecting known component conflicts.

    Returns:
        Explicit coverage limitations, without assigning missing fluxes zero.
    """
    gaps = []
    for index, component in enumerate(components):
        declarations = [
            compiled.state_meta[name]
            for name in compiled.state_names
            if name in component and name in compiled.state_meta
        ]
        units = {entry.units for entry in declarations if entry.units is not None}
        representations = {
            entry.representation
            for entry in declarations
            if entry.representation is not None
        }
        origin = f"component:{index}"
        if len(units) > 1:
            gaps.append(BalanceCoverageGap(origin, "incompatible_state_units"))
        if len(representations) > 1:
            gaps.append(
                BalanceCoverageGap(origin, "incompatible_state_representations")
            )
        if any(entry.measure_axes for entry in declarations):
            gaps.append(BalanceCoverageGap(origin, "unsupported_state_measure"))
    return gaps


def reaction_balances(compiled: CompiledRhs) -> ReactionBalanceReport:
    """Derive conditional scalar totals from fully represented unit transfers.

    Each internal incidence column has one consumed and one produced unit.
    Component-indicator vectors therefore give an exact integer left-nullspace
    basis. Source-only channels appear explicitly on the right-hand side of
    their component balances. Any unsupported contribution prevents all laws;
    omitted terms are never assigned zero flux. Propensities are not evaluated.

    Args:
        compiled: Compiled RHS with complete named scalar reaction artifacts.

    Returns:
        Immutable identities, coverage records, declarations and assumptions.
        Physical interpretation and numerical/runtime realization require
        independent qualification; ``fully_covered`` is a structural result.

    Raises:
        TypeError: If ``compiled`` is not a CompiledRhs container.
    """
    if not isinstance(compiled, CompiledRhs):
        msg = "compiled must be a CompiledRhs"
        raise TypeError(msg)
    states = tuple(compiled.state_names)
    gaps = _coverage(compiled)
    components = () if gaps else _components(states, compiled.reactions)
    gaps.extend(_scientific_coverage(compiled, components))
    names = tuple(reaction.name for reaction in compiled.reactions)
    laws = (
        ()
        if gaps
        else tuple(
            ReactionBalanceLaw(
                name=f"component:{index}",
                coefficients=tuple(int(name in component) for name in states),
                reaction_names=names,
                sources=tuple(
                    ReactionBalanceSource(reaction.name, 1, reaction.to_base)
                    for reaction in compiled.reactions
                    if reaction.from_base is None and reaction.to_base in component
                ),
                assumptions=_ASSUMPTIONS,
            )
            for index, component in enumerate(components)
        )
    )
    return ReactionBalanceReport(
        states,
        laws,
        tuple(gaps),
        tuple(
            (name, compiled.state_meta[name])
            for name in states
            if name in compiled.state_meta
        ),
        tuple(compiled.axis_meta.items()),
        _ASSUMPTIONS,
    )
