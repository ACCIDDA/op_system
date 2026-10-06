"""Infer reaction reactants and state dependencies from a rate.

``reactants: auto`` derives a transition's reactants from its rate, after
aliases are inlined, instead of requiring an explicit list (issue #255).
When the rate is a single product of state references (``beta * I[age]``,
``k * S**2``, or a rate that reads no state) each state factor is a
molecular reactant whose order is its integer power.

Other rates are not molecular mass action. A frequency-dependent force of
infection such as ``beta * S * sum(I) / N`` instead publishes the states its
propensity reads and a bound on its elasticity (issue #256):

    E >= sum_i |d log a / d log x_i|

Adaptive tau-leaping keeps each propensity's relative change within
epsilon from that bound alone. The bound is derived from the expression's
structure: 0 for a factor that reads no state, 1 for a state, the sum of the
operands for a product or quotient, ``|p|`` times the base for a literal
power ``p``, and the maximum of the terms for a sum or reduction whose terms
are provably non-negative. The last rule holds because the elasticities of a
non-negative sum are a convex combination of its terms' elasticities.
Parameters are assumed non-negative and independent of state. Subtraction,
negation, other functions of a state, and history operators have no such
bound and are rejected.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from op_system._ir import Apply, HistoryOp, Literal, Reduce, Subscript, Sym, walk

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

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

    A reference that does not select each declared axis once, as a free
    axis or a declared coordinate, raises :class:`NotInferableError`.

    Returns:
        The reference's free axes and pinned coordinates.
    """
    (reference,) = _expand_reference(node, state_axes, axis_lookup, bound={})
    return reference


def _expand_reference(
    node: Sym | Subscript,
    state_axes: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, list[str]],
    *,
    bound: Mapping[str, tuple[str, ...]],
) -> Iterator[StateReference]:
    """Expand a state reference into one record per bound coordinate.

    An index bound by an enclosing reduction (``I[age:a]`` inside
    ``sum_over(..., age=a)``) reads every coordinate the reduction visits,
    so it expands to one pinned reference per coordinate.

    Yields:
        Each state reference in declared axis order.

    Raises:
        NotInferableError: If the reference does not select each declared
            axis once, as a free axis, a declared coordinate, or a bound
            reduction variable.
    """
    declared = state_axes[node.name]
    if isinstance(node, Sym):
        if declared:
            msg = f"names templated state {node.name!r} without its axes"
            raise NotInferableError(msg)
        yield StateReference(node.name, (), (), ())
        return
    choices: dict[str, tuple[str | None, ...]] = {}
    for index in node.indices:
        if index.axis not in declared or index.axis in choices:
            msg = f"indexes state {node.name!r} by {index.axis!r}"
            raise NotInferableError(msg)
        if index.coord is None:
            choices[index.axis] = (None,)
        elif index.coord in bound:
            choices[index.axis] = bound[index.coord]
        elif index.coord in axis_lookup.get(index.axis, ()):
            choices[index.axis] = (index.coord,)
        else:
            msg = (
                f"indexes state {node.name!r} with {index.axis}:{index.coord}, "
                "which is not a declared coordinate"
            )
            raise NotInferableError(msg)
    if set(choices) != set(declared):
        msg = f"does not index every axis of state {node.name!r}"
        raise NotInferableError(msg)
    for picked in itertools.product(*(choices[axis] for axis in declared)):
        coords = dict(zip(declared, picked, strict=True))
        yield StateReference(
            base=node.name,
            axes=tuple(axis for axis in declared if coords[axis] is None),
            full_axes=declared,
            pinned=tuple(
                (axis, coord)
                for axis in declared
                if (coord := coords[axis]) is not None
            ),
        )


def state_dependencies(
    rate: Expr,
    *,
    state_axes: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, list[str]],
) -> tuple[StateReference, ...]:
    """Return every state selection the rate reads, in first-seen order.

    Args:
        rate: Alias-inlined rate IR.
        state_axes: Each state base's declared axes.
        axis_lookup: Each axis's declared coordinates.

    A reference that cannot be described, or a read of state history,
    raises :class:`NotInferableError`.

    Returns:
        Distinct state references, with reductions expanded per coordinate.
    """
    found: dict[StateReference, None] = {}
    _collect_dependencies(rate, state_axes, axis_lookup, bound={}, found=found)
    return tuple(found)


def _collect_dependencies(
    expr: Expr,
    state_axes: Mapping[str, tuple[str, ...]],
    axis_lookup: Mapping[str, list[str]],
    *,
    bound: Mapping[str, tuple[str, ...]],
    found: dict[StateReference, None],
) -> None:
    """Add the state references under ``expr`` to ``found``.

    Raises:
        NotInferableError: If the expression reads state history.
    """
    if isinstance(expr, Sym | Subscript):
        if expr.name in state_axes:
            found.update(
                dict.fromkeys(
                    _expand_reference(expr, state_axes, axis_lookup, bound=bound)
                )
            )
        return
    if isinstance(expr, HistoryOp):
        msg = f"reads state history through {expr.kind}"
        raise NotInferableError(msg)
    if isinstance(expr, Reduce):
        filters = dict(expr.filters)
        inner = dict(bound)
        for axis, var in expr.bindings:
            coords = tuple(axis_lookup.get(axis, ()))
            subset = filters.get(axis, coords)
            # A filter that is not a coordinate list (a continuous
            # sub-interval) keeps every coordinate: a superset is safe.
            inner[var] = subset if set(subset) <= set(coords) else coords
        _collect_dependencies(
            expr.body, state_axes, axis_lookup, bound=inner, found=found
        )
        return
    if isinstance(expr, Apply):
        for arg in expr.args:
            _collect_dependencies(
                arg, state_axes, axis_lookup, bound=bound, found=found
            )


def elasticity_order(rate: Expr, *, state_axes: Mapping[str, tuple[str, ...]]) -> int:
    """Return a whole-number bound on the rate's total elasticity.

    A rate with no structural bound raises :class:`NotInferableError`.

    Returns:
        ``ceil(E)`` for the structural bound ``E`` described in the module
        docstring.
    """
    return math.ceil(_elasticity(rate, state_axes))


def _elasticity(expr: Expr, state_axes: Mapping[str, tuple[str, ...]]) -> Fraction:
    """Bound ``sum_i |d log expr / d log x_i|`` from the expression's form.

    Returns:
        The bound as an exact fraction.

    Raises:
        NotInferableError: If the form has no bound.
    """
    if not may_read_state(expr, state_axes):
        return Fraction(0)
    if isinstance(expr, Sym | Subscript):
        return Fraction(1)
    if isinstance(expr, Reduce):
        if not _nonnegative(expr.body):
            msg = f"reduces terms with {expr.kind} that may be negative"
            raise NotInferableError(msg)
        return _elasticity(expr.body, state_axes)
    if isinstance(expr, Apply):
        return _apply_elasticity(expr, state_axes)
    if isinstance(expr, HistoryOp):
        msg = f"reads state history through {expr.kind}"
        raise NotInferableError(msg)
    msg = "reads state in an unsupported form"
    raise NotInferableError(msg)


def _apply_elasticity(
    expr: Apply, state_axes: Mapping[str, tuple[str, ...]]
) -> Fraction:
    """Bound the elasticity of a product, quotient, power, or sum.

    Returns:
        The bound as an exact fraction.

    Raises:
        NotInferableError: If the operation has no bound.
    """
    if expr.op in {"*", "/"}:
        return sum((_elasticity(arg, state_axes) for arg in expr.args), Fraction(0))
    if expr.op == "pow" and len(expr.args) == 2:
        base, exponent = expr.args
        power = exponent.value if isinstance(exponent, Literal) else None
        if not isinstance(power, int | float) or isinstance(power, bool):
            msg = "raises a state to a power that is not a numeric literal"
            raise NotInferableError(msg)
        return abs(Fraction(power)) * _elasticity(base, state_axes)
    if expr.op == "+":
        if not all(_nonnegative(arg) for arg in expr.args):
            msg = "adds terms that may be negative"
            raise NotInferableError(msg)
        return max(_elasticity(arg, state_axes) for arg in expr.args)
    msg = f"combines a state with {expr.op!r}"
    raise NotInferableError(msg)


def _nonnegative(expr: Expr) -> bool:
    """Report whether ``expr`` is non-negative from its form alone.

    States are populations and parameters are assumed non-negative.

    Returns:
        ``True`` when the form guarantees a non-negative value.
    """
    if isinstance(expr, Literal):
        value = expr.value
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and value >= 0
        )
    if isinstance(expr, Reduce):
        return _nonnegative(expr.body)
    if not isinstance(expr, Apply):
        return isinstance(expr, Sym | Subscript)
    if expr.op in {"*", "/", "+"}:
        return all(_nonnegative(arg) for arg in expr.args)
    if expr.op == "pow":
        return bool(expr.args) and _nonnegative(expr.args[0])
    return expr.op == "exp"
