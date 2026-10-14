"""Scientific declarations independent of numerical reductions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

from op_system._errors import InvalidRhsSpecError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from op_system._normalize_ir import StateTemplate

_Representation = Literal[
    "counts", "cell_mass", "density", "concentration", "cumulative"
]
_MeasureKind = Literal["counting", "cell", "quadrature"]


class AxisMeasure(NamedTuple):
    """Explicit physical measure in the axis coordinate order.

    Attributes:
        kind: Counting, cell widths/volumes, or point quadrature weights.
        weights: Finite strictly positive weights, one per coordinate.
        units: Optional measure units, without dimensional verification.
    """

    kind: _MeasureKind
    weights: tuple[float, ...]
    units: str | None = None

    @classmethod
    def from_mapping(cls, value: object, *, size: int) -> AxisMeasure:
        """Validate a measure without inferring geometry from coordinates.

        Args:
            value: Mapping with kind, optional units, and weights for cell/quadrature.
            size: Number of axis coordinates.

        Returns:
            Immutable measure in coordinate order.

        Raises:
            InvalidRhsSpecError: If the kind, units, weights or length are invalid.
        """
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise InvalidRhsSpecError(
                detail="axis measure size must be a positive integer"
            )
        if not isinstance(value, Mapping) or set(value) - {"kind", "weights", "units"}:
            raise InvalidRhsSpecError(
                detail="axis measure must contain kind, weights and/or units"
            )
        kind = value.get("kind")
        if not isinstance(kind, str) or kind not in {"counting", "cell", "quadrature"}:
            raise InvalidRhsSpecError(
                detail="axis measure kind must be counting, cell or quadrature"
            )
        if kind == "counting":
            if "weights" in value:
                raise InvalidRhsSpecError(
                    detail="counting measure has unit weights; omit weights"
                )
            weights = (1.0,) * size
        else:
            raw_weights = value.get("weights")
            if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) != size:
                raise InvalidRhsSpecError(
                    detail="axis measure weights must match the coordinate count"
                )
            weights = tuple(_positive_weight(weight) for weight in raw_weights)
        return cls(cast("_MeasureKind", kind), weights, _optional_text(value, "units"))


class AxisDeclaration(NamedTuple):
    """Physical interpretation of a named axis.

    Attributes:
        units: Coordinate units, independent of measure units.
        measure: Explicit measure, or None when undeclared.
    """

    units: str | None = None
    measure: AxisMeasure | None = None


class StateDeclaration(NamedTuple):
    """Optional representation and physical reduction requirements of a state.

    Attributes:
        representation: Extensive counts/mass/counter or intensive density/
            concentration; None when unknown. Cumulative means an extensive
            accumulated counter, without asserting monotonicity of its equations.
        units: State value units, without dimensional verification.
        description: Human-readable annotation.
        measure_axes: Axes whose measures are required for an intensive integral.
            None means undeclared; an empty tuple explicitly needs none.
    """

    representation: _Representation | None = None
    units: str | None = None
    description: str | None = None
    measure_axes: tuple[str, ...] | None = None

    @classmethod
    def from_mapping(cls, value: object) -> StateDeclaration:
        """Validate a state annotation without changing its equations.

        Args:
            value: Mapping of representation, units, description and measure_axes.

        Returns:
            Immutable declaration; omitted information remains unknown.

        Raises:
            InvalidRhsSpecError: If fields or their values are invalid.
        """
        if not isinstance(value, Mapping) or set(value) - {
            "representation",
            "units",
            "description",
            "measure_axes",
        }:
            raise InvalidRhsSpecError(
                detail="state metadata contains unknown fields or is not a mapping"
            )
        representation = value.get("representation")
        if "representation" in value and (
            not isinstance(representation, str)
            or representation
            not in {"counts", "cell_mass", "density", "concentration", "cumulative"}
        ):
            raise InvalidRhsSpecError(
                detail=(
                    "state representation must be counts, cell_mass, density, "
                    "concentration or cumulative"
                )
            )
        measure_axes = None
        if "measure_axes" in value:
            axes = value["measure_axes"]
            if (
                not isinstance(axes, (list, tuple))
                or any(not isinstance(axis, str) or not axis.strip() for axis in axes)
                or len(set(axes)) != len(axes)
            ):
                raise InvalidRhsSpecError(
                    detail="state measure_axes must list unique nonempty axis names"
                )
            measure_axes = tuple(axes)
            if representation in {"counts", "cell_mass", "cumulative"} and measure_axes:
                raise InvalidRhsSpecError(
                    detail="extensive states do not require measure weights"
                )
        return cls(
            cast("_Representation | None", representation),
            _optional_text(value, "units"),
            _optional_text(value, "description"),
            measure_axes,
        )


def _positive_weight(value: object) -> float:
    """Validate one supported physical weight.

    Returns:
        Weight as a finite positive float.

    Raises:
        InvalidRhsSpecError: If the weight is not finite and strictly positive.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidRhsSpecError(
            detail="axis measure weights must be finite positive numbers"
        )
    try:
        weight = float(value)
    except OverflowError as exc:
        raise InvalidRhsSpecError(
            detail="axis measure weights must be finite positive numbers"
        ) from exc
    if not math.isfinite(weight) or weight <= 0:
        raise InvalidRhsSpecError(
            detail="axis measure weights must be finite positive numbers"
        )
    return weight


def _optional_text(value: Mapping[str, object], name: str) -> str | None:
    """Read a nonempty optional scientific annotation.

    Returns:
        Supplied text, or None when absent.

    Raises:
        InvalidRhsSpecError: If a supplied annotation is not nonempty text.
    """
    if name not in value:
        return None
    text = value[name]
    if not isinstance(text, str) or not text.strip():
        raise InvalidRhsSpecError(
            detail=f"scientific metadata {name} must be nonempty text"
        )
    return text


def _state_declaration_metadata(
    value: object,
    *,
    templates: Sequence[StateTemplate],
    axes: Sequence[Mapping[str, object]],
    state_axes: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    """Validate state metadata against resolved states and physical axes.

    Returns:
        Optional declarations, with absent metadata left unchanged.

    Raises:
        InvalidRhsSpecError: If a state or required physical measure is unknown.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvalidRhsSpecError(detail="state_metadata must be a declaration mapping")
    template_axes: dict[str, set[str]] = {}
    for template in templates:
        template_axes.setdefault(template.base, set()).update(template.axes)
    measures = {axis["name"]: axis.get("measure") for axis in axes}
    declarations = {}
    for name, raw_declaration in value.items():
        if not isinstance(name, str) or name not in template_axes:
            raise InvalidRhsSpecError(
                detail=f"state metadata {name!r} is not a declared state base"
            )
        declaration = StateDeclaration.from_mapping(raw_declaration)
        available = template_axes[name] | set(state_axes.get(name, ()))
        for axis in declaration.measure_axes or ():
            if axis not in available:
                raise InvalidRhsSpecError(
                    detail=f"state {name!r} measure axis {axis!r} is not a state axis"
                )
            if measures.get(axis) is None:
                raise InvalidRhsSpecError(
                    detail=f"state {name!r} requires a measure for axis {axis!r}"
                )
        declarations[name] = declaration
    return {"state_declarations": declarations}


def _axis_declarations(
    axes: Sequence[Mapping[str, object]],
) -> dict[str, AxisDeclaration]:
    """Detach typed physical declarations from normalized axes.

    Returns:
        Declarations for all named axes, including unknown units/measure.
    """
    return {
        cast("str", axis["name"]): AxisDeclaration(
            cast("str | None", axis.get("units")),
            cast("AxisMeasure | None", axis.get("measure")),
        )
        for axis in axes
    }
