"""Scaling declarations, numerical verification records and quotient maps.

Equation covariance and output invariance do not establish identifiability.
Initial conditions must transform with the states; priors and known initial
conditions may remove a scaling freedom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from ._array import array_namespace
from ._gauge_declarations import ScalingGauge

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ._typing import Array


class GaugeVerification(NamedTuple):
    """Reproducible finite probes, rather than a global numerical proof.

    Attributes:
        seed: Local NumPy random seed; no global random state is changed.
        n_probes: Points tested per generator, at both reported scales.
        scales: Positive group elements tested independently per generator.
        rtol: Relative tolerance used for every compared component.
        atol: Absolute tolerance used for every compared component.
        max_absolute_error: Largest absolute covariance/invariance residual.
        observables: All declared output names checked for invariance.
        structural: Whether supported homogeneity rules also verify declarations.
    """

    seed: int
    n_probes: int
    scales: tuple[float, ...]
    rtol: float
    atol: float
    max_absolute_error: float
    observables: tuple[str, ...]
    structural: bool


class InvariantCoordinate(NamedTuple):
    """A monomial invariant of the reported parameter exponent matrix.

    Attributes:
        exponents: Parameter base names and nonzero integer powers.
    """

    exponents: tuple[tuple[str, int], ...]

    def evaluate(self, parameters: Mapping[str, object]) -> object:
        """Evaluate a monomial using supplied values, without domain correction.

        Args:
            parameters: Values with native array broadcasting semantics.

        Returns:
            Product of the named powers. Negative powers require nonzero values.

        Examples:
            >>> invariant = InvariantCoordinate((("K", 1), ("c", 1)))
            >>> invariant.evaluate({"K": 4.0, "c": 3.0})
            12.0
        """
        result: object = 1.0
        for name, exponent in self.exponents:
            # Arithmetic is namespace-native for concrete values and tracers.
            result *= parameters[name] ** exponent  # type: ignore[operator]
        return result


class GaugeInfo(NamedTuple):
    """Declared/detected shared-template scalings and their rational ranks.

    Attributes:
        declared: User-supplied generators verified at compile time.
        detected: Conservative structural nullspace basis, if detection requested.
        generators: Independent basis spanning the generators being reported.
        rank: Rank on states and parameters together.
        parameter_rank: Rank of the parameter projection of that action.
        parameter_names: Scalar/shaped parameter bases in coordinate order.
        invariant_coordinates: Integer nullspace monomials for parameters.
            This count is not an identifiable dimension estimate.
        verification: Finite-probe record for declared generators, or None.
        detection_requested: Distinguishes no detection from an empty nullspace.
    """

    declared: tuple[ScalingGauge, ...]
    detected: tuple[ScalingGauge, ...]
    generators: tuple[ScalingGauge, ...]
    rank: int
    parameter_rank: int
    parameter_names: tuple[str, ...]
    invariant_coordinates: tuple[InvariantCoordinate, ...]
    verification: GaugeVerification | None
    detection_requested: bool


class GaugeCoordinates(NamedTuple):
    """A representative together with the group coordinates it discarded.

    Attributes:
        state: Flat state in the fixed representative's units.
        parameters: Remaining parameters; anchor values are omitted.
        log_scales: Group coordinates needed to invert this particular map.
        valid: Native scalar flag for positive anchors and finite coordinates.
            Invalid domains produce nonfinite arithmetic; no floor is applied.
    """

    state: Array
    parameters: dict[str, object]
    log_scales: Array
    valid: Array


class GaugeFixMap(NamedTuple):
    """Positive scalar anchors for a full-rank shared scaling action.

    Attributes:
        fixed_values: Scalar parameter anchors and target values.
        generators: Independent group basis used by the map.
        state_weights: One generator-weight row per expanded state component.
        inverse_anchor_matrix: Inverse anchor-weight matrix, stored immutably.
    """

    fixed_values: tuple[tuple[str, float], ...]
    generators: tuple[ScalingGauge, ...]
    state_weights: tuple[tuple[float, ...], ...]
    inverse_anchor_matrix: tuple[tuple[float, ...], ...]

    def encode(
        self, state: Array, parameters: Mapping[str, object]
    ) -> GaugeCoordinates:
        """Map natural state/parameters to the anchored representative.

        Args:
            state: Real floating flat state, with optional trailing batch axes.
            parameters: Original parameter values, including positive anchors.

        Returns:
            Representative, original group coordinates and native domain flag.
            Check ``valid`` before using a result in an inference pipeline.

        Raises:
            ValueError: If state axes or anchor shapes do not match this map.
            TypeError: If states are not real floating arrays.
        """
        if not state.shape or state.shape[0] != len(self.state_weights):
            msg = "Gauge map state axis does not match the compiled model"
            raise ValueError(msg)
        xp = array_namespace(state)
        if not xp.isdtype(state.dtype, "real floating"):
            msg = "Gauge coordinates require real floating state arrays"
            raise TypeError(msg)
        values = xp.stack([
            xp.asarray(parameters[name], dtype=state.dtype)
            for name, _ in self.fixed_values
        ])
        if len(values.shape) != 1:
            msg = "Gauge anchors must be scalar; use vmap for separate parameter draws"
            raise ValueError(msg)
        targets = xp.asarray(
            [value for _, value in self.fixed_values], dtype=state.dtype
        )
        matrix = xp.asarray(self.inverse_anchor_matrix, dtype=state.dtype)
        logs = matrix @ (xp.log(targets) - xp.log(values))
        valid = xp.all(xp.logical_and(xp.isfinite(values), values > 0))
        valid = xp.logical_and(valid, xp.all(xp.isfinite(logs)))
        weights = xp.asarray(self.state_weights, dtype=state.dtype)
        factors = xp.exp(weights @ logs)
        valid = xp.logical_and(
            valid, xp.all(xp.logical_and(xp.isfinite(factors), factors > 0))
        )
        factors = xp.reshape(
            factors, (len(self.state_weights),) + (1,) * (len(state.shape) - 1)
        )
        transformed = {
            name: xp.asarray(value, dtype=state.dtype)
            * xp.exp(
                xp.sum(
                    xp.asarray(
                        [
                            dict(generator.parameters).get(name, 0.0)
                            for generator in self.generators
                        ],
                        dtype=state.dtype,
                    )
                    * logs
                )
            )
            for name, value in parameters.items()
            if name not in dict(self.fixed_values)
        }
        mapped_state = state * factors
        valid = xp.logical_and(valid, xp.all(xp.isfinite(mapped_state)))
        valid = xp.logical_and(
            valid,
            xp.all(xp.logical_or(xp.not_equal(mapped_state, 0), xp.equal(state, 0))),
        )
        for name, value in transformed.items():
            original = xp.asarray(parameters[name], dtype=state.dtype)
            finite = xp.all(xp.isfinite(value))
            retained = xp.all(xp.logical_or(value != 0, original == 0))
            valid = xp.logical_and(valid, xp.logical_and(finite, retained))
        return GaugeCoordinates(mapped_state, transformed, logs, valid)

    def decode(self, coordinates: GaugeCoordinates) -> tuple[Array, dict[str, object]]:
        """Recover original values using retained group coordinates.

        Args:
            coordinates: Encoded representative with its original log scales.

        Returns:
            Original flat state and complete parameter mapping. A quotient alone
            has no inverse; these retained scales select the original member.
        """
        state, remaining, logs, _ = coordinates
        xp = array_namespace(state)
        factors = xp.exp(xp.asarray(self.state_weights, dtype=state.dtype) @ logs)
        factors = xp.reshape(
            factors, (len(self.state_weights),) + (1,) * (len(state.shape) - 1)
        )
        params = {**remaining, **dict(self.fixed_values)}
        original = {
            name: xp.asarray(value, dtype=state.dtype)
            * xp.exp(
                -xp.sum(
                    xp.asarray(
                        [
                            dict(generator.parameters).get(name, 0.0)
                            for generator in self.generators
                        ],
                        dtype=state.dtype,
                    )
                    * logs
                )
            )
            for name, value in params.items()
        }
        return state / factors, original


__all__ = [
    "GaugeCoordinates",
    "GaugeFixMap",
    "GaugeInfo",
    "GaugeVerification",
    "InvariantCoordinate",
    "ScalingGauge",
]
