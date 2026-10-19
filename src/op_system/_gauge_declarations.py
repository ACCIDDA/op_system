"""Immutable declarations of shared positive scaling actions."""

from __future__ import annotations

import keyword
import math
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, NamedTuple

from ._errors import InvalidRhsSpecError

if TYPE_CHECKING:
    from collections.abc import Collection


class ScalingGauge(NamedTuple):
    """One positive scaling generator, shared over each named template.

    For scale ``s > 0``, an exponent ``w`` acts as ``value * s**w``.
    Declaration does not establish covariance or output invariance; compiled
    analysis records those checks separately.

    Attributes:
        name: Unique generator label.
        states: State base names and finite scaling exponents.
        parameters: Parameter base names and finite scaling exponents.
    """

    name: str
    states: tuple[tuple[str, float], ...] = ()
    parameters: tuple[tuple[str, float], ...] = ()


def _exponents(
    value: object, *, known: Collection[str], field: str
) -> tuple[tuple[str, float], ...]:
    """Detach and validate base-name exponent pairs.

    Returns:
        Nonzero exponents in declaration order.

    Raises:
        InvalidRhsSpecError: If names, exponents or the declaration shape fail.
    """
    if not isinstance(value, Mapping):
        raise InvalidRhsSpecError(
            detail=f"gauge {field} must be a base-name exponent mapping"
        )
    result = []
    for name, exponent in value.items():
        if not isinstance(name, str) or name not in known:
            raise InvalidRhsSpecError(
                detail=f"Unknown gauge {field} base {name!r}; use declared base names"
            )
        if (
            isinstance(exponent, bool)
            or not isinstance(exponent, Real)
            or not math.isfinite(float(exponent))
        ):
            raise InvalidRhsSpecError(
                detail=f"Gauge exponent for {name!r} must be a finite real number"
            )
        if exponent:
            result.append((name, float(exponent)))
    return tuple(result)


def _normalize_gauges(
    value: object, *, states: Collection[str], parameters: Collection[str]
) -> tuple[ScalingGauge, ...]:
    """Validate optional raw declarations without assigning verification status.

    Returns:
        Detached immutable generators.

    Raises:
        InvalidRhsSpecError: If a declaration is malformed, duplicated or empty.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise InvalidRhsSpecError(
            detail="gauges must be a list of scaling declarations"
        )
    result = []
    names: set[str] = set()
    for declaration in value:
        if not isinstance(declaration, Mapping) or set(declaration) - {
            "name",
            "states",
            "parameters",
        }:
            raise InvalidRhsSpecError(
                detail="A gauge declaration allows only name, states and parameters"
            )
        name = declaration.get("name")
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
            or name in names
        ):
            raise InvalidRhsSpecError(detail="Gauge names must be unique identifiers")
        state_weights = _exponents(
            declaration.get("states", {}), known=states, field="states"
        )
        parameter_weights = _exponents(
            declaration.get("parameters", {}), known=parameters, field="parameters"
        )
        if not state_weights and not parameter_weights:
            raise InvalidRhsSpecError(
                detail=f"Gauge {name!r} must have a nonzero exponent"
            )
        names.add(name)
        result.append(ScalingGauge(name, state_weights, parameter_weights))
    return tuple(result)
