"""Infer molecular reactants from a reaction rate (issue #255).

``reactants: auto`` derives a transition's reactants from its rate, after
aliases are inlined, instead of requiring an explicit list. Inference is
exact only when the rate is a single product of state references:
``beta * I[age]``, ``k * S**2``, or a rate that reads no state at all. Each
state factor is then a molecular reactant whose order is its integer power.

Sums of states, states in a denominator, reductions over states, functions
of a state, and history operators are not molecular mass action. A
frequency-dependent force of infection such as ``beta * S * sum(I) / N``
needs a different description of its state dependence (issue #256), so
inference rejects these rates instead of guessing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from op_system._ir import Apply, HistoryOp, Literal, Reduce, Subscript, Sym, walk

if TYPE_CHECKING:
    from collections.abc import Mapping

    from op_system._ir import Expr

REACTANTS_AUTO = "auto"


@dataclass(frozen=True, slots=True)
class StateReference:
    """One state selection read by a rate, with axes in declared order.

    Attributes:
        base: State template base name.
        axes: Axes the reference leaves free, one value per channel.
        full_axes: The state's declared axes.
        pinned: ``(axis, coord)`` pairs fixed by the reference.
    """

    base: str
    axes: tuple[str, ...]
    full_axes: tuple[str, ...]
    pinned: tuple[tuple[str, str], ...]


class NotInferableError(ValueError):
    """A rate whose state dependence is not a single product of states."""


def may_read_state(expr: Expr, state_axes: Mapping[str, tuple[str, ...]]) -> bool:
    """Report whether ``expr`` could depend on the current state.

    History operators count as reading state, because their bodies are
    evaluated against stored state.

    Returns:
        ``True`` when any node names a state or is a history operator.
    """
    return any(
        isinstance(node, HistoryOp)
        or (isinstance(node, Sym | Subscript) and node.name in state_axes)
        for node in walk(expr)
    )


def infer_state_factors(
    rate: Expr,
    *,
    state_axes: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, list[str]],
) -> Counter[StateReference]:
    """Return the state factors of a single-product rate and their powers.

    Args:
        rate: Alias-inlined rate IR.
        state_axes: Each state base's declared axes.
        axis_lookup: Each axis's declared coordinates.

    Returns:
        Each state reference with its total integer power. Empty when the
        rate reads no state.

    Raises:
        NotInferableError: If the rate reads state other than through a
            product of state references and positive integer powers.
    """
    if not may_read_state(rate, state_axes):
        return Counter()
    if isinstance(rate, Sym | Subscript):
        return Counter({_reference(rate, state_axes, axis_lookup): 1})
    if isinstance(rate, Apply):
        return _apply_factors(rate, state_axes=state_axes, axis_lookup=axis_lookup)
    if isinstance(rate, Reduce):
        msg = f"reduces states with {rate.kind}"
        raise NotInferableError(msg)
    if isinstance(rate, HistoryOp):
        msg = f"reads state history through {rate.kind}"
        raise NotInferableError(msg)
    msg = "reads state in an unsupported form"
    raise NotInferableError(msg)


def _apply_factors(
    rate: Apply,
    *,
    state_axes: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, list[str]],
) -> Counter[StateReference]:
    """Combine the state factors of a product, quotient, or power.

    Returns:
        Each state reference with its total integer power.

    Raises:
        NotInferableError: If the operation is not a product of states.
    """
    if rate.op == "*":
        total: Counter[StateReference] = Counter()
        for arg in rate.args:
            total.update(
                infer_state_factors(arg, state_axes=state_axes, axis_lookup=axis_lookup)
            )
        return total
    if rate.op == "/" and len(rate.args) == 2:
        numerator, denominator = rate.args
        if may_read_state(denominator, state_axes):
            msg = "divides by a state"
            raise NotInferableError(msg)
        return infer_state_factors(
            numerator, state_axes=state_axes, axis_lookup=axis_lookup
        )
    if rate.op == "pow" and len(rate.args) == 2:
        base, exponent = rate.args
        power = exponent.value if isinstance(exponent, Literal) else None
        if not isinstance(power, int) or isinstance(power, bool) or power < 1:
            msg = "raises a state to a power that is not a positive integer literal"
            raise NotInferableError(msg)
        factors = infer_state_factors(
            base, state_axes=state_axes, axis_lookup=axis_lookup
        )
        return Counter({ref: count * power for ref, count in factors.items()})
    msg = f"combines a state with {rate.op!r}"
    raise NotInferableError(msg)


def _reference(
    node: Sym | Subscript,
    state_axes: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, list[str]],
) -> StateReference:
    """Describe one state reference in declared axis order.

    Returns:
        The reference's free axes and pinned coordinates.

    Raises:
        NotInferableError: If the reference does not select each declared
            axis once, as a free axis or a declared coordinate.
    """
    declared = state_axes[node.name]
    if isinstance(node, Sym):
        if declared:
            msg = f"names templated state {node.name!r} without its axes"
            raise NotInferableError(msg)
        return StateReference(node.name, (), (), ())
    coords: dict[str, str | None] = {}
    for index in node.indices:
        if index.axis not in declared or index.axis in coords:
            msg = f"indexes state {node.name!r} by {index.axis!r}"
            raise NotInferableError(msg)
        if index.coord is not None and index.coord not in axis_lookup.get(
            index.axis, ()
        ):
            msg = (
                f"indexes state {node.name!r} with {index.axis}:{index.coord}, "
                "which is not a declared coordinate"
            )
            raise NotInferableError(msg)
        coords[index.axis] = index.coord
    if set(coords) != set(declared):
        msg = f"does not index every axis of state {node.name!r}"
        raise NotInferableError(msg)
    return StateReference(
        base=node.name,
        axes=tuple(axis for axis in declared if coords[axis] is None),
        full_axes=declared,
        pinned=tuple(
            (axis, coord) for axis in declared if (coord := coords[axis]) is not None
        ),
    )
