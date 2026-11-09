"""Immutable eager kernel tables retaining named conditional host/antigen axes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["ConditionalKernel", "KernelAxis", "KernelReduction"]


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be a nonempty string"
        raise ValueError(msg)


def _immutable(values: ArrayLike) -> NDArray[np.float64]:
    """Copy into a bytes-backed table that cannot become writable again.

    Returns:
        Real float64 values backed by immutable bytes.

    Raises:
        ValueError: Values are not real numeric arrays or contain nonfinite entries.
    """
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        msg = "Kernel values and weights must be real numeric arrays"
        raise ValueError(msg)
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        msg = "Kernel values and weights must be finite"
        raise ValueError(msg)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True)
class KernelAxis:
    """Ordered named conditional axis; coordinates are semantic labels."""

    name: str
    coordinates: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject ambiguous axes.

        Raises:
            ValueError: Names or coordinates are empty, reserved or duplicated.
            TypeError: Coordinates are a string rather than a sequence of labels.
        """
        _identity(self.name, "axis name")
        if self.name == "infection_age":
            msg = "infection_age is reserved for the leading grid"
            raise ValueError(msg)
        if isinstance(self.coordinates, str):
            msg = "Axis coordinates must be a sequence of labels, not a string"
            raise TypeError(msg)
        coordinates = tuple(self.coordinates)
        if not coordinates or len(set(coordinates)) != len(coordinates):
            msg = "Axis coordinates must be nonempty and unique"
            raise ValueError(msg)
        for coordinate in coordinates:
            _identity(coordinate, "axis coordinate")
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True, eq=False)
class KernelReduction:
    """Explicit joint probability weights in declared conditional-axis order."""

    axes: tuple[str, ...]
    weights: ArrayLike = field(repr=False, compare=False)
    distribution_identity: str

    def __post_init__(self) -> None:
        """Snapshot normalized joint weights and their scientific identity.

        Raises:
            ValueError: Axes, distribution identity or weights are invalid.
        """
        axes = tuple(self.axes)
        if not axes or len(set(axes)) != len(axes):
            msg = "Reduction axes must be nonempty and unique"
            raise ValueError(msg)
        for axis in axes:
            _identity(axis, "reduction axis")
        _identity(self.distribution_identity, "distribution identity")
        weights = _immutable(self.weights)
        if (
            weights.ndim != len(axes)
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
        ):
            msg = "Joint probability weights require matching rank and unit total mass"
            raise ValueError(msg)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True, eq=False)
class ConditionalKernel:
    """Snapshot of a source observable; no axes are aggregated implicitly.

    Values have shape ``(len(infection_age), *conditional_axis_sizes)``.
    Source settings identify parameter fixtures, conditioning, solver and
    precision. A caller must regenerate this eager artifact when they change.
    It is not a differentiable binding or a two-way runtime coupling.
    """

    infection_age: tuple[float, ...]
    axes: tuple[KernelAxis, ...]
    values: ArrayLike = field(repr=False, compare=False)
    source_identity: str
    source_settings_identity: str
    observable: str
    units: str
    age_units: str
    conditioning: tuple[str, ...] = ()
    reductions: tuple[KernelReduction, ...] = ()

    def __post_init__(self) -> None:
        """Validate the grid, axis order, provenance and immutable values.

        Raises:
            ValueError: Metadata, grid or value shape is invalid.
            TypeError: Axes or reduction history have incorrect types.
        """
        for name in (
            "source_identity",
            "source_settings_identity",
            "observable",
            "units",
            "age_units",
        ):
            _identity(getattr(self, name), name)
        age = np.asarray(self.infection_age, dtype=np.float64)
        if (
            age.ndim != 1
            or not age.size
            or not np.all(np.isfinite(age))
            or np.any(age < 0)
            or np.any(np.diff(age) <= 0)
        ):
            msg = (
                "Infection-age grid must be finite, nonnegative and strictly increasing"
            )
            raise ValueError(msg)
        axes = tuple(self.axes)
        reductions = tuple(self.reductions)
        if any(not isinstance(axis, KernelAxis) for axis in axes):
            msg = "Conditional axes must be KernelAxis instances"
            raise TypeError(msg)
        if any(not isinstance(r, KernelReduction) for r in reductions):
            msg = "Reduction history must contain KernelReduction instances"
            raise TypeError(msg)
        if len({axis.name for axis in axes}) != len(axes):
            msg = "Conditional-axis names must be unique"
            raise ValueError(msg)
        conditioning = tuple(self.conditioning)
        for condition in conditioning:
            _identity(condition, "conditioning")
        values = _immutable(self.values)
        if values.shape != (len(age), *(len(axis.coordinates) for axis in axes)):
            msg = (
                "Kernel value shape must match infection-age and named conditional axes"
            )
            raise ValueError(msg)
        object.__setattr__(self, "infection_age", tuple(float(t) for t in age))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "conditioning", conditioning)
        object.__setattr__(self, "reductions", reductions)

    def reduce(self, reduction: KernelReduction) -> ConditionalKernel:
        """Average only declared axes with explicit joint probability weights.

        Apply scientifically intended nonlinear mappings before this reduction.
        Weights are static, independent of age and remaining axes; conditional
        or parameter-dependent distributions require caller regeneration.

        Returns:
            A new kernel retaining the grid, remaining axes and reduction history.

        Raises:
            TypeError: A reduction artifact was not supplied.
            ValueError: Named axes or weight shape do not match.
        """
        if not isinstance(reduction, KernelReduction):
            msg = "Explicit KernelReduction required"
            raise TypeError(msg)
        names = tuple(axis.name for axis in self.axes)
        if any(name not in names for name in reduction.axes):
            msg = "Reduction axes must name retained conditional axes"
            raise ValueError(msg)
        indices = tuple(names.index(name) + 1 for name in reduction.axes)
        weights = np.asarray(reduction.weights)
        values = np.asarray(self.values)
        if weights.shape != tuple(values.shape[i] for i in indices):
            msg = "Joint weights do not match the declared axis coordinate order"
            raise ValueError(msg)
        order = np.argsort(indices)
        aligned = weights.transpose(tuple(int(i) for i in order))
        broadcast = [1] * values.ndim
        for index in indices:
            broadcast[index] = values.shape[index]
        reduced = np.sum(
            values * aligned.reshape(broadcast), axis=tuple(sorted(indices))
        )
        return ConditionalKernel(
            self.infection_age,
            tuple(a for a in self.axes if a.name not in reduction.axes),
            reduced,
            self.source_identity,
            self.source_settings_identity,
            self.observable,
            self.units,
            self.age_units,
            self.conditioning,
            (*self.reductions, reduction),
        )

    def content_digest(self) -> str:
        """Hash numerical content and ordered scientific provenance.

        Returns:
            SHA-256 of canonical float64 tables and ordered immutable metadata.
        """
        metadata = (
            self.infection_age,
            self.axes,
            self.source_identity,
            self.source_settings_identity,
            self.observable,
            self.units,
            self.age_units,
            self.conditioning,
            tuple((r.axes, r.distribution_identity) for r in self.reductions),
        )
        digest = hashlib.sha256(repr(metadata).encode("utf-8"))
        digest.update(np.asarray(self.values, dtype="<f8").tobytes())
        for reduction in self.reductions:
            weights = np.asarray(reduction.weights, dtype="<f8")
            digest.update(repr(weights.shape).encode("ascii"))
            digest.update(weights.tobytes())
        return digest.hexdigest()
