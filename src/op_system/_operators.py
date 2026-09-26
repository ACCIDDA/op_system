"""op_system._operators — typed operator descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class OperatorDescriptor:
    """Typed description of a spatial operator declared in an op_system RHS spec.

    Captures the model-level description of an operator that is known at
    compile time: which axis it acts on, its kind (advection, diffusion,
    etc.), optional boundary condition, and the names of any runtime
    parameters (velocity, rate) it consumes.  Grid geometry, CN matrix
    construction, and solver staging remain downstream concerns handled by
    the engine.

    Attributes:
        axis: Name of the axis the operator acts on (e.g. ``"loc"``).
        kind: Operator type string, e.g. ``"advection"``, ``"diffusion"``,
            ``"transport"``, ``"jump_integral"``, or ``None`` if unspecified.
        bc: Boundary condition, e.g. ``"absorbing"``, ``"periodic"``,
            ``"neumann"``, ``"reflecting"``, or ``None`` if unspecified.
        velocity: Parameter name or numeric constant for an advection velocity,
            or ``None``.
        rate: Parameter name or numeric constant for a rate/coefficient, or
            ``None``.
        kernel: Mixing-kernel sub-specification, or ``None``.
        name: Optional operator name from the specification.
        apply_to: Concrete state names selected by the operator, or ``None`` when
            the operator applies to every compatible state.
        direction: Optional normalized direction. For advection and transport,
            ``"increasing"`` preserves the resolved velocity sign and
            ``"decreasing"`` reverses it; when omitted, velocity is used as a
            signed coefficient. Other operator kinds may define their own
            direction vocabulary.

    Examples:
        >>> od = OperatorDescriptor(axis="loc")
        >>> od.axis
        'loc'
        >>> od.kind is None
        True
        >>> od.bc is None
        True
        >>> od.velocity is None
        True
        >>> od2 = OperatorDescriptor(
        ...     axis="loc",
        ...     kind="advection",
        ...     bc="absorbing",
        ...     velocity=0.25,
        ...     name="drift",
        ...     apply_to=("S",),
        ... )
        >>> od2.kind
        'advection'
        >>> od2.bc
        'absorbing'
        >>> od2.velocity
        0.25
        >>> od2.name
        'drift'
        >>> od2.apply_to
        ('S',)
    """

    axis: str
    kind: str | None = None
    bc: str | None = None
    velocity: str | float | None = None
    rate: str | float | None = None
    kernel: Mapping[str, Any] | None = None
    name: str | None = None
    apply_to: tuple[str, ...] | None = None
    direction: str | None = None
