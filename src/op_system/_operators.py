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
    compile time: which axis it acts on, and the names of any runtime
    parameters (velocity, rate) it consumes.  Grid geometry, CN matrix
    construction, and solver staging remain downstream concerns handled by
    the engine.

    Attributes:
        axis: Name of the axis the operator acts on (e.g. ``"loc"``).
        velocity: Parameter name for an advection velocity, or ``None``.
        rate: Parameter name for a diffusion rate/coefficient, or ``None``.
        kernel: Mixing-kernel sub-specification, or ``None``.

    Examples:
        >>> od = OperatorDescriptor(axis="loc")
        >>> od.axis
        'loc'
        >>> od.velocity is None
        True
        >>> od2 = OperatorDescriptor(axis="loc", velocity="v_advec", rate="diff_r")
        >>> od2.velocity
        'v_advec'
        >>> od2.rate
        'diff_r'
    """

    axis: str
    velocity: str | None = None
    rate: str | None = None
    kernel: Mapping[str, Any] | None = None
