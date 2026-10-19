"""Bind gauge anchors consistently across compiled evaluator contracts."""

from __future__ import annotations

import math
from dataclasses import replace
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._errors import InvalidRhsSpecError
from ._gauge_algebra import _rank
from ._reactions import ReactionGap
from .gauges import GaugeFixMap

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .compile import CompiledRhs
    from .gauges import GaugeInfo


def _fixed_values(
    value: Mapping[str, float], compiled: CompiledRhs
) -> tuple[tuple[str, float], ...]:
    """Validate scalar positive parameter anchors.

    Returns:
        Detached targets in mapping order.

    Raises:
        InvalidRhsSpecError: If targets or parameter names are invalid.
    """
    scalar_names = set(compiled.param_names)
    scalar_names.update(
        name
        for observable in compiled.observables.values()
        for name, shape in observable.parameter_shapes
        if not shape
    )
    result: list[tuple[str, float]] = []
    for name, target in value.items():
        if (
            name not in scalar_names
            or isinstance(target, bool)
            or not isinstance(target, Real)
            or not math.isfinite(float(target))
            or target <= 0
        ):
            raise InvalidRhsSpecError(
                detail=(
                    f"Gauge anchor {name!r} needs a scalar "
                    "parameter and positive target"
                )
            )
        support = compiled.param_meta.get(name)
        if (
            support is not None
            and support.support is not None
            and support.support.kind not in {"real", "positive"}
        ):
            raise InvalidRhsSpecError(
                detail=f"Gauge anchor {name!r} has a bounded support"
            )
        result.append((name, float(target)))
    return tuple(result)


def _bind_anchors(
    fn: Callable[..., Any], fixed: Mapping[str, float]
) -> Callable[..., Any]:
    """Bind constants and reject accidental runtime overrides.

    Returns:
        Same evaluator contract with named constants removed from its arguments.
    """

    def evaluate(*args: object, **params: object) -> Any:  # ruff: ignore[any-type]
        overlap = fixed.keys() & params.keys()
        if overlap:
            msg = f"Cannot override gauge-fixed parameters: {sorted(overlap)}"
            raise TypeError(msg)
        return fn(*args, **params, **fixed)

    return evaluate


def _fix_gauge(
    compiled: CompiledRhs, info: GaugeInfo, anchors: Mapping[str, float]
) -> CompiledRhs:
    """Bind a full-rank set of scalar anchors and retain the coordinate map.

    Returns:
        Compiled evaluators with identical physical equations at the anchors.

    Raises:
        InvalidRhsSpecError: If anchors do not remove all reported directions.
    """
    values = _fixed_values(anchors, compiled)
    matrix = [
        [dict(gauge.parameters).get(name, 0.0) for gauge in info.generators]
        for name, _ in values
    ]
    if (
        not info.rank
        or len(values) != info.rank
        or _rank(matrix, info.rank) != info.rank
        or not compiled.observables
    ):
        raise InvalidRhsSpecError(
            detail="Gauge fixing needs invariant outputs and rank-many scalar anchors"
        )
    inverse = np.linalg.inv(np.asarray(matrix, dtype=float))
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    assert rhs is not None  # ruff: ignore[assert]
    bases = {
        name: template.base
        for template in rhs.state_templates
        for name in template.expanded_names
    }
    state_weights = tuple(
        tuple(dict(gauge.states).get(bases[name], 0.0) for gauge in info.generators)
        for name in compiled.state_names
    )
    mapping = GaugeFixMap(
        values,
        info.generators,
        state_weights,
        tuple(tuple(float(value) for value in row) for row in inverse),
    )
    if rhs.meta.get("initial_state"):
        raise InvalidRhsSpecError(
            detail=(
                "Remove spec initial_state before gauge fixing; "
                "encode initial states with original parameters explicitly"
            )
        )
    fixed = dict(values)
    observables = {
        name: observable._replace(
            parameter_shapes=tuple(
                (key, shape)
                for key, shape in observable.parameter_shapes
                if key not in fixed
            ),
            eval_fn=_bind_anchors(observable.eval_fn, fixed),
            pytree_eval_fn=_bind_anchors(observable.pytree_eval_fn, fixed),
        )
        for name, observable in compiled.observables.items()
    }
    reactions = tuple(
        replace(
            reaction,
            propensity_fn=_bind_anchors(reaction.propensity_fn, fixed),
            resolved_propensity_fn=_bind_anchors(reaction.resolved_propensity_fn, fixed)
            if reaction.resolved_propensity_fn
            else None,
        )
        for reaction in compiled.reactions
    )
    gaps = compiled.reaction_gaps
    if reactions:
        gaps += tuple(
            ReactionGap(
                "gauges",
                reaction.name,
                reaction.from_base,
                reaction.to_base,
                "gauge_fix_requires_jump_contract",
            )
            for reaction in reactions
        )
        reactions = ()
    declarations = {
        name: declaration._replace(fixed=True) if name in fixed else declaration
        for name, declaration in compiled.param_meta.items()
    }
    metadata = {**compiled.meta, "parameter_declarations": declarations}

    return replace(
        compiled,
        param_names=tuple(name for name in compiled.param_names if name not in fixed),
        eval_fn=_bind_anchors(compiled.eval_fn, fixed),
        pytree_eval_fn=cast(
            "Any",
            _bind_anchors(compiled.pytree_eval_fn, fixed)
            if compiled.pytree_eval_fn
            else None,
        ),
        block_pytree_eval_fn=cast(
            "Any",
            _bind_anchors(compiled.block_pytree_eval_fn, fixed)
            if compiled.block_pytree_eval_fn
            else None,
        ),
        observables=MappingProxyType(observables),
        reactions=reactions,
        reaction_gaps=gaps,
        param_meta=MappingProxyType(declarations),
        meta=metadata,
        gauge_fix=mapping,
    )
