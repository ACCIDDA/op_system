"""Opt-in shared scaling analysis of compiled deterministic equations."""

from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import TYPE_CHECKING

import numpy as np

from ._errors import InvalidRhsSpecError, UnsupportedFeatureError
from ._gauge_algebra import _integer_nullspace, _rank
from ._gauge_declarations import ScalingGauge
from ._gauge_homogeneity import _Homogeneity
from .gauges import GaugeInfo, GaugeVerification, InvariantCoordinate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .compile import CompiledRhs
    from .specs import NormalizedRhs


def _parameter_names(rhs: NormalizedRhs) -> tuple[str, ...]:
    """Collect RHS and observable parameter bases in deterministic order.

    Returns:
        Parameter names excluding injected compiler constants.
    """
    names = set(rhs.param_names) | set(dict(rhs.shaped_params))
    names.update(name for obs in rhs.observables_ir for name, _ in obs.parameter_axes)
    return tuple(sorted(names - set(rhs.meta.get("op_system_synth_constants", {}))))


def _weight_row(gauge: ScalingGauge, variables: Sequence[str]) -> tuple[float, ...]:
    """Return a generator in the declared variable order."""
    weights = {**dict(gauge.states), **dict(gauge.parameters)}
    return tuple(weights.get(name, 0.0) for name in variables)


def _constraints(rhs: NormalizedRhs, variables: Sequence[str]) -> _Homogeneity:
    """Form supported structural covariance and output-invariance constraints.

    Returns:
        Rational constraints; no finite differences or fitted exponents.

    Raises:
        UnsupportedFeatureError: If an equation is unavailable to the analysis.
    """
    bases = {
        name: template.base
        for template in rhs.state_templates
        for name in template.expanded_names
    }
    analysis = _Homogeneity(
        variables,
        bases,
        rhs.aliases_ir,
        state_bases=tuple(t.base for t in rhs.state_templates),
    )
    for name, expr in zip(rhs.state_names, rhs.equations_ir, strict=True):
        if expr is None:
            raise UnsupportedFeatureError(
                feature="scaling gauges", detail="Missing equation IR"
            )
        analysis.equate(analysis.degree(expr), analysis.variable(name))
    for observable in rhs.observables_ir:
        analysis.equate(analysis.degree(observable.expression), analysis.zero)
    return analysis


def _supported(compiled: CompiledRhs, rhs: NormalizedRhs) -> None:
    """Reject dynamics omitted by the instantaneous compiled RHS.

    Raises:
        UnsupportedFeatureError: If full-system verification cannot be claimed.
        InvalidRhsSpecError: If a scaled parameter has a restricted fixed domain.
    """
    if compiled.operators or compiled.history_requirements or rhs.time_varying_params:
        raise UnsupportedFeatureError(
            feature="scaling gauges",
            detail=(
                "Operator/history/time-table dynamics need explicit scaling contracts"
            ),
        )
    for gauge in rhs.meta.get("op_system_gauge_declarations", ()):
        for name, _ in gauge.parameters:
            declaration = compiled.param_meta.get(name)
            if declaration is not None and (
                declaration.fixed
                or (
                    declaration.support is not None
                    and declaration.support.kind not in {"real", "positive"}
                )
            ):
                raise InvalidRhsSpecError(
                    detail=(
                        f"Scaling {name!r} conflicts with a fixed/bounded declaration"
                    )
                )


def _probe_parameters(
    compiled: CompiledRhs, rng: np.random.Generator
) -> dict[str, object]:
    """Sample finite positive scalar/shaped inputs locally.

    Returns:
        Parameters covering equations and every declared observable.
    """
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    assert rhs is not None  # ruff: ignore[assert]
    sizes = {axis["name"]: len(axis["coords"]) for axis in rhs.meta.get("axes", ())}
    shapes = {
        name: tuple(sizes[axis] for axis in axes) for name, axes in rhs.shaped_params
    }
    for observable in compiled.observables.values():
        shapes.update(dict(observable.parameter_shapes))
    params: dict[str, object] = {}
    for name in _parameter_names(rhs):
        value = rng.uniform(0.6, 1.4, size=shapes.get(name, ()))
        declaration = compiled.param_meta.get(name)
        support = declaration.support if declaration else None
        if support is not None and support.kind == "unit_interval":
            value *= 0.5
        elif support is not None and support.kind == "bounded":
            assert support.lower is not None  # ruff: ignore[assert]
            assert support.upper is not None  # ruff: ignore[assert]
            value = support.lower + value * 0.5 * (support.upper - support.lower)
        params[name] = value
    return params


def _compare(left: object, right: object, label: str) -> float:
    """Check a finite numerical comparison with a local, useful failure.

    Returns:
        Maximum absolute error.

    Raises:
        InvalidRhsSpecError: If any component is nonfinite or exceeds tolerance.
    """
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (
        a.shape != b.shape
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
        or not np.allclose(a, b, rtol=1e-9, atol=1e-11)
    ):
        raise InvalidRhsSpecError(
            detail=f"Gauge failed {label}; seeded probe rtol=1e-9, atol=1e-11"
        )
    return float(np.max(np.abs(a - b), initial=0))


def _verify(
    compiled: CompiledRhs, declared: tuple[ScalingGauge, ...], *, structural: bool
) -> GaugeVerification:
    """Probe RHS covariance and all declared observable invariances.

    Returns:
        Seed, scales, tolerances and residuals for reproducibility.
    """
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    assert rhs is not None  # ruff: ignore[assert]
    bases = {
        name: template.base
        for template in rhs.state_templates
        for name in template.expanded_names
    }
    rng = np.random.default_rng(19471)
    error = 0.0
    for gauge in declared:
        state_weights = np.array([
            dict(gauge.states).get(bases[name], 0.0) for name in compiled.state_names
        ])
        for _ in range(4):
            state = rng.uniform(0.6, 1.4, size=len(compiled.state_names))
            params = _probe_parameters(compiled, rng)
            t = float(rng.uniform(0.1, 0.9))
            reference = compiled.eval_fn(t, state, **params)
            for scale in (0.5, 2.0):
                transformed = {
                    name: value * scale ** dict(gauge.parameters).get(name, 0.0)
                    for name, value in params.items()
                }
                scaled_state = state * scale**state_weights
                result = compiled.eval_fn(t, scaled_state, **transformed)
                error = max(
                    error,
                    _compare(
                        result,
                        reference * scale**state_weights,
                        f"RHS covariance for {gauge.name!r}",
                    ),
                )
                for name, observable in compiled.observables.items():
                    error = max(
                        error,
                        _compare(
                            observable.eval_fn(t, scaled_state, **transformed),
                            observable.eval_fn(t, state, **params),
                            f"observable {name!r} invariance for {gauge.name!r}",
                        ),
                    )
    return GaugeVerification(
        19471,
        4,
        (0.5, 2.0),
        1e-9,
        1e-11,
        error,
        tuple(compiled.observables),
        structural,
    )


def _analyze_gauges(compiled: CompiledRhs, *, detect: bool) -> GaugeInfo:
    """Build optional structural directions and verify user declarations.

    Returns:
        Reported exponent ranks and parameter-invariant monomials.

    Raises:
        UnsupportedFeatureError: If requested detection has unsupported operations.
    """
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    assert rhs is not None  # ruff: ignore[assert]
    _supported(compiled, rhs)
    states = tuple(template.base for template in rhs.state_templates)
    parameters = _parameter_names(rhs)
    variables = (*states, *parameters)
    declared = tuple(rhs.meta.get("op_system_gauge_declarations", ()))
    rows = None
    try:
        rows = _constraints(rhs, variables).rows
    except UnsupportedFeatureError:
        if detect:
            raise
    structural = rows is not None and all(
        sum(
            (
                coefficient * Fraction(str(weight))
                for coefficient, weight in zip(
                    row, _weight_row(gauge, variables), strict=True
                )
            ),
            Fraction(0),
        )
        == 0
        for gauge in declared
        for row in rows
    )
    verification = (
        _verify(compiled, declared, structural=structural) if declared else None
    )
    detected: tuple[ScalingGauge, ...] = ()
    if detect and rows is not None:
        # A fixed or bounded parameter is not a global positive scaling freedom.
        for name, declaration in compiled.param_meta.items():
            if declaration.fixed or (
                declaration.support is not None
                and declaration.support.kind not in {"real", "positive"}
            ):
                rows.append(tuple(Fraction(variable == name) for variable in variables))
        detected = tuple(
            ScalingGauge(
                f"detected_{index}",
                tuple(
                    (name, float(value))
                    for name, value in zip(states, vector[: len(states)], strict=True)
                    if value
                ),
                tuple(
                    (name, float(value))
                    for name, value in zip(
                        parameters, vector[len(states) :], strict=True
                    )
                    if value
                ),
            )
            for index, vector in enumerate(_integer_nullspace(rows, len(variables)))
        )
    generators: list[ScalingGauge] = []
    matrix: list[tuple[float, ...]] = []
    for gauge in (*declared, *detected):
        row = _weight_row(gauge, variables)
        if _rank([*matrix, row], len(variables)) > len(matrix):
            generators.append(gauge)
            matrix.append(row)
    parameter_matrix = [row[len(states) :] for row in matrix]
    invariants = tuple(
        InvariantCoordinate(
            tuple(
                (name, power)
                for name, power in zip(parameters, vector, strict=True)
                if power
            )
        )
        for vector in _integer_nullspace(parameter_matrix, len(parameters))
    )
    return GaugeInfo(
        declared,
        detected,
        tuple(generators),
        len(matrix),
        _rank(parameter_matrix, len(parameters)),
        parameters,
        invariants,
        verification,
        detect,
    )


def _gauge_summary(compiled: CompiledRhs) -> dict[str, object] | None:
    """Summarize a reported action at generic nonzero parameter values.

    Returns:
        CLI-safe counts and scope, without asserting identifiable dimensions.
    """
    info = compiled.gauges
    if info is None:
        return None
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    assert rhs is not None  # ruff: ignore[assert]
    sizes = {axis["name"]: len(axis["coords"]) for axis in rhs.meta.get("axes", ())}
    shapes = {
        name: tuple(sizes[axis] for axis in axes) for name, axes in rhs.shaped_params
    }
    for observable in compiled.observables.values():
        shapes.update(dict(observable.parameter_shapes))
    count = sum(prod(shapes.get(name, ())) for name in info.parameter_names)
    return {
        "rank": info.rank,
        "parameter_rank": info.parameter_rank,
        "free_reported_scaling_directions": 0 if compiled.gauge_fix else info.rank,
        "parameter_coordinate_count": count,
        "parameter_coordinates_after_reported_action": count - info.parameter_rank,
        "parameter_base_invariant_monomials": len(info.invariant_coordinates),
        "detection_requested": info.detection_requested,
        "verification": info.verification._asdict() if info.verification else None,
        "identifiability_assessed": False,
        "scope": (
            "Instantaneous deterministic equations and declared outputs; "
            "generic nonzero parameter values; initial states must transform."
        ),
    }
