"""Static state maps evaluated simultaneously from the pre-event state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from numbers import Real
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

from op_system._array import array_namespace
from op_system._errors import InvalidRhsSpecError
from op_system._ir import Literal as LiteralExpr
from op_system._ir import walk
from op_system._observables import (
    CompiledObservable,
    ObservableEvalFn,
    ObservablePytreeEvalFn,
    _compile_observables,
    _normalize_observables,
    _ObservableIR,
)
from op_system._templates import PinnedToken, parse_selector

if TYPE_CHECKING:
    from op_system.compile import _Indexable
    from op_system.specs import NormalizedRhs


class ImpulseUpdate(NamedTuple):
    """One disjoint state selection and its pre-event expression.

    Attributes:
        target: Existing named state selector, including any pinned coordinates.
        operation: Add, multiply or replace selected pre-event values.
        expression: Pointwise arithmetic expression in the existing state layout.
    """

    target: str
    operation: Literal["add", "multiply", "set"]
    expression: str


class CompiledImpulse(NamedTuple):
    """A static, right-continuous state map with native array evaluators.

    Attributes:
        time: Finite recorded time, fixed before solving or differentiation.
        updates: Immutable disjoint update definitions, evaluated simultaneously.
        parameter_axes: Required named parameter layouts, separate from the RHS.
        parameter_shapes: Required concrete parameter shapes.
        eval_fn: Map a complete flat pre-event state to the post-event state.
        pytree_eval_fn: Equivalent map on complete shaped state buffers.
    """

    time: float
    updates: tuple[ImpulseUpdate, ...]
    parameter_axes: tuple[tuple[str, tuple[str, ...]], ...]
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    eval_fn: ObservableEvalFn
    pytree_eval_fn: ObservablePytreeEvalFn


class _ImpulseUpdateIR(NamedTuple):
    definition: ImpulseUpdate
    base: str
    mask: tuple[bool, ...]
    observable: _ObservableIR


class _ImpulseIR(NamedTuple):
    time: float
    updates: tuple[_ImpulseUpdateIR, ...]


def _target(value: object, rhs: NormalizedRhs) -> tuple[str, tuple[bool, ...]]:
    """Resolve a selector without expanding or mutating state expressions.

    Returns:
        State base and flattened mask in its declared axis order.

    Raises:
        InvalidRhsSpecError: If target, axes or coordinates are unknown.
    """
    if not isinstance(value, str):
        raise InvalidRhsSpecError(detail="impulse target must be a state selector")
    base, tokens = parse_selector(value)
    template = next((item for item in rhs.state_templates if item.base == base), None)
    if template is None or {token.axis for token in tokens} != set(template.axes):
        raise InvalidRhsSpecError(
            detail="impulse target must match a declared state and its axes"
        )
    coordinates = {
        axis["name"]: tuple(str(c) for c in axis["coords"])
        for axis in rhs.meta.get("axes", ())
    }
    pinned = {
        token.axis: token.coord for token in tokens if isinstance(token, PinnedToken)
    }
    if any(coord not in coordinates[axis] for axis, coord in pinned.items()):
        raise InvalidRhsSpecError(detail="impulse target coordinate is not declared")
    mask = tuple(
        all(
            pinned.get(axis, coord) == coord
            for axis, coord in zip(template.axes, cell, strict=True)
        )
        for cell in product(*(coordinates[axis] for axis in template.axes))
    )
    return base, mask


def _update(value: object, rhs: NormalizedRhs, index: int) -> _ImpulseUpdateIR:
    """Validate one restricted pointwise map and reuse native expression IR.

    Returns:
        Immutable definition, selector mask and validated expression.

    Raises:
        InvalidRhsSpecError: If fields, expression literals or operation are invalid.
    """
    if not isinstance(value, Mapping) or set(value) != {"target", "operation", "expr"}:
        raise InvalidRhsSpecError(
            detail="impulse updates require target, operation and expr only"
        )
    base, mask = _target(value["target"], rhs)
    operation = value["operation"]
    if not isinstance(operation, str) or operation not in {"add", "multiply", "set"}:
        raise InvalidRhsSpecError(
            detail="impulse operation must be add, multiply or set"
        )
    axes = next(
        template.axes for template in rhs.state_templates if template.base == base
    )
    observable = _normalize_observables(
        {f"impulse_update_{index}": {"expr": value["expr"], "axes": axes}}, rhs=rhs
    )[0]
    if any(
        isinstance(node, LiteralExpr)
        and isinstance(node.value, Real)
        and not math.isfinite(node.value)
        for node in walk(observable.expression)
    ):
        raise InvalidRhsSpecError(detail="impulse expression literals must be finite")
    return _ImpulseUpdateIR(
        ImpulseUpdate(
            value["target"],
            cast('Literal["add", "multiply", "set"]', operation),
            observable.definition.expression,
        ),
        base,
        mask,
        observable,
    )


def _normalize_impulses(value: object, *, rhs: NormalizedRhs) -> tuple[_ImpulseIR, ...]:
    """Snapshot strictly increasing static events and reject ambiguous targets.

    Returns:
        Immutable events; absent or empty schedules have no metadata effect.

    Raises:
        InvalidRhsSpecError: If schedule fields, times, targets or layouts are invalid.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise InvalidRhsSpecError(detail="impulses must be a list of static events")
    result = []
    previous = -math.inf
    layouts = dict(rhs.shaped_params)
    layouts.update(dict.fromkeys(rhs.param_names, ()))
    for observable in rhs.observables_ir:
        layouts.update(dict(observable.parameter_axes))
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {"time", "updates"}:
            raise InvalidRhsSpecError(detail="impulse requires time and updates only")
        time = raw["time"]
        if (
            isinstance(time, bool)
            or not isinstance(time, Real)
            or not math.isfinite(time)
            or time <= previous
        ):
            raise InvalidRhsSpecError(
                detail="impulse times must be finite and strictly increasing"
            )
        updates = raw["updates"]
        if not isinstance(updates, (list, tuple)) or not updates:
            raise InvalidRhsSpecError(detail="impulse updates must be a nonempty list")
        normalized = tuple(
            _update(update, rhs, index) for index, update in enumerate(updates)
        )
        _check_updates(normalized, layouts)
        previous = float(time)
        result.append(_ImpulseIR(previous, normalized))
    return tuple(result)


def _check_updates(
    updates: tuple[_ImpulseUpdateIR, ...], layouts: dict[str, tuple[str, ...]]
) -> None:
    """Reject overlapping selections and conflicting parameter declarations.

    Raises:
        InvalidRhsSpecError: If a target overlaps or a parameter layout conflicts.
    """
    selected: dict[str, set[int]] = {}
    for update in updates:
        indices = {i for i, active in enumerate(update.mask) if active}
        prior = selected.setdefault(update.base, set())
        if prior & indices:
            raise InvalidRhsSpecError(
                detail="impulse updates must have disjoint targets"
            )
        prior.update(indices)
        for name, axes in update.observable.parameter_axes:
            if layouts.setdefault(name, axes) != axes:
                raise InvalidRhsSpecError(
                    detail=f"impulse parameter {name!r} has inconsistent axes"
                )


def _map_evaluators(
    event: _ImpulseIR, values: tuple[CompiledObservable, ...], rhs: NormalizedRhs
) -> tuple[ObservableEvalFn, ObservablePytreeEvalFn]:
    def pytree_eval_fn(t: object, y: Mapping[str, object], **params: object) -> object:
        # Every expression sees y; no expression reads an earlier update's result.
        evaluated = [value.pytree_eval_fn(t, y, **params) for value in values]
        result = dict(y)
        for update, value in zip(event.updates, evaluated, strict=True):
            reference = y[update.base]
            xp = array_namespace(reference)
            operation = update.definition.operation
            changed = (
                xp.add(reference, value)
                if operation == "add"
                else xp.multiply(reference, value)
                if operation == "multiply"
                else value
            )
            mask = xp.reshape(
                xp.asarray(update.mask, dtype=xp.bool),
                getattr(reference, "shape", None),
            )
            result[update.base] = xp.where(mask, changed, result[update.base])
        return result

    def eval_fn(t: object, y: object, **params: object) -> object:
        xp = array_namespace(y)
        if getattr(y, "shape", None) != (len(rhs.state_names),):
            message = "impulse flat state shape must match compiled state_names"
            raise ValueError(message)
        vector = cast("_Indexable", y)
        states = {
            template.base: xp.reshape(
                vector[template.offset : template.offset + math.prod(template.shape)],
                template.shape,
            )
            for template in rhs.state_templates
        }
        result = cast("Mapping[str, object]", pytree_eval_fn(t, states, **params))
        return xp.concat(
            tuple(
                xp.reshape(result[template.base], (-1,))
                for template in rhs.state_templates
            )
        )

    return eval_fn, pytree_eval_fn


def _compile_impulses(rhs: NormalizedRhs) -> tuple[CompiledImpulse, ...]:
    """Compile immutable maps with the qualified pointwise expression machinery.

    Returns:
        Native maps; models without events retain empty metadata and evaluators.
    """
    events = cast("tuple[_ImpulseIR, ...]", rhs.meta.get("op_system_impulses_ir", ()))
    result = []
    for event in events:
        metadata = {
            **rhs.meta,
            "op_system_observables_ir": tuple(
                update.observable for update in event.updates
            ),
        }
        values = tuple(_compile_observables(replace(rhs, meta=metadata)).values())
        axes = tuple(
            sorted({
                entry
                for update in event.updates
                for entry in update.observable.parameter_axes
            })
        )
        shapes = tuple(
            sorted({entry for value in values for entry in value.parameter_shapes})
        )
        eval_fn, pytree_eval_fn = _map_evaluators(event, values, rhs)
        result.append(
            CompiledImpulse(
                event.time,
                tuple(update.definition for update in event.updates),
                axes,
                shapes,
                eval_fn,
                pytree_eval_fn,
            )
        )
    return tuple(result)
