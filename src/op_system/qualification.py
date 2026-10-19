"""Opt-in, eager numerical qualification of compiled RHS functions.

This module checks user-supplied fixtures against independent equations and
optional balance and JAX directional-derivative checks. It does not replace
structural validation or establish scientific validity or integration accuracy.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np

from op_system._array import array_namespace

if TYPE_CHECKING:
    from op_system import Array, CompiledRhs
    from op_system.compile import StateDict

Evaluator = Literal["flat", "structured", "block"]
ReferenceRhs = Callable[
    [float, Mapping[str, "Array"], Mapping[str, object]], Mapping[str, "Array"]
]
BalanceCheck = Callable[
    [float, Mapping[str, "Array"], Mapping[str, "Array"], Mapping[str, object]], "Array"
]


class QualificationFixture(NamedTuple):
    """Named structured state/parameter fixture and optional state direction."""

    name: str
    state: Mapping[str, Array]
    params: Mapping[str, object]
    time: float = 0.0
    direction: Mapping[str, Array] | None = None


class InvariantCheck(NamedTuple):
    """A residual-to-zero balance check with explicit applicability conditions."""

    name: str
    callback: BalanceCheck
    condition: str
    atol: float = 1e-8


class NumericalCheck(NamedTuple):
    """A named numerical check and any informative failure messages."""

    name: str
    passed: bool
    max_abs_error: float | None
    errors: tuple[str, ...] = ()
    condition: str | None = None
    rtol: float = 0.0
    atol: float = 0.0
    step: float | None = None


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    """Reference and derivative tolerances for eager qualification.

    Defaults assume double precision. Choose wider tolerances and a larger
    finite-difference step for float32 fixtures.
    """

    rtol: float = 1e-7
    atol: float = 1e-9
    derivative_step: float = 1e-5
    derivative_rtol: float = 1e-4
    derivative_atol: float = 1e-6

    def __post_init__(self) -> None:
        """Validate controls.

        Raises:
            ValueError: If a tolerance or finite-difference step is invalid.
        """
        for name in (
            "rtol",
            "atol",
            "derivative_rtol",
            "derivative_atol",
            "derivative_step",
        ):
            value = getattr(self, name)
            numeric = isinstance(value, Real) and not isinstance(value, bool)
            if (
                not numeric
                or not math.isfinite(value)
                or value < 0
                or (name == "derivative_step" and value == 0)
            ):
                bound = "positive" if name == "derivative_step" else "non-negative"
                msg = f"{name} must be finite and {bound}"
                raise ValueError(msg)


class ArrayComparison(NamedTuple):
    """Reference parity, shape/dtype preservation, and finite-value diagnostics."""

    state: str
    shape: tuple[int, ...]
    reference_shape: tuple[int, ...]
    input_dtype: str
    output_dtype: str
    reference_dtype: str
    finite: bool
    passed: bool
    max_abs_error: float | None
    max_relative_error: float | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FixtureQualification:
    """Qualification of one named fixture through one evaluator.

    All diagnostics are host values: this report is an eager authoring tool,
    even when the evaluated RHS uses JAX or Torch arrays.
    """

    fixture: str
    evaluator: Evaluator
    backend: str
    time: float
    rtol: float
    atol: float
    arrays: tuple[ArrayComparison, ...] = ()
    invariants: tuple[NumericalCheck, ...] = ()
    derivative: NumericalCheck | None = None
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every requested check passed."""
        return (
            not self.errors
            and bool(self.arrays)
            and all(item.passed for item in self.arrays)
            and all(item.passed for item in self.invariants)
            and (self.derivative is None or self.derivative.passed)
        )


@dataclass(frozen=True, slots=True)
class QualificationReport:
    """Inspectable results for all requested fixture/evaluator combinations."""

    fixtures: tuple[FixtureQualification, ...]

    @property
    def ok(self) -> bool:
        """Whether all fixtures and requested evaluators qualified."""
        return bool(self.fixtures) and all(item.ok for item in self.fixtures)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible fields, including per-fixture pass/fail status.

        Returns:
            A mapping suitable for ``json.dumps(..., allow_nan=False)``.
        """
        return {
            "ok": self.ok,
            "fixtures": [
                {
                    **asdict(item),
                    "ok": item.ok,
                    "arrays": [value._asdict() for value in item.arrays],
                    "invariants": [value._asdict() for value in item.invariants],
                    "derivative": item.derivative._asdict()
                    if item.derivative
                    else None,
                }
                for item in self.fixtures
            ],
        }


def _host(value: object) -> np.ndarray:
    values = cast("Any", value)
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _select(value: object, axis: int | None, index: int) -> Any:  # ruff: ignore[any-type]
    if axis is None:
        return value
    xp = array_namespace(value)
    shape = cast("Any", value).shape
    return xp.permute_dims(value, (axis, *(i for i in range(len(shape)) if i != axis)))[
        index
    ]


def _evaluate(
    compiled: CompiledRhs, fixture: QualificationFixture, evaluator: Evaluator
) -> Mapping[str, Array]:
    shapes = compiled.template_shapes or dict.fromkeys(compiled.state_names, ())
    if set(fixture.state) != set(shapes):
        msg = f"state keys must match compiled templates {tuple(shapes)}"
        raise ValueError(msg)
    for name, shape in shapes.items():
        if fixture.state[name].shape != shape:
            msg = (
                f"state {name!r} has shape {fixture.state[name].shape}; "
                f"expected {shape}"
            )
            raise ValueError(msg)
    if evaluator == "structured":
        if compiled.pytree_eval_fn is None:
            msg = "structured evaluator is unavailable"
            raise ValueError(msg)
        return compiled.pytree_eval_fn(
            fixture.time, cast("StateDict", dict(fixture.state)), **fixture.params
        )
    xp = array_namespace(next(iter(fixture.state.values())))
    if evaluator == "flat":
        return _evaluate_flat(compiled, fixture, shapes, xp)
    block = compiled.block_pytree_eval_fn
    if block is None or not compiled.block_axes:
        msg = (
            "block evaluator is unavailable; "
            "coupled models cannot be treated as independent"
        )
        raise ValueError(msg)
    info = compiled.block_axes[0]
    if set(info.state_axis_pos) != set(shapes):
        msg = "block qualification requires the block axis on every state template"
        raise ValueError(msg)
    blocks = [
        block(
            fixture.time,
            {
                name: _select(value, info.state_axis_pos[name], index)
                for name, value in fixture.state.items()
            },
            **{
                name: _select(value, info.param_axis_pos.get(name), index)
                for name, value in fixture.params.items()
            },
        )
        for index in range(info.size)
    ]
    return {
        name: xp.stack([item[name] for item in blocks], axis=info.state_axis_pos[name])
        for name in shapes
    }


def _evaluate_flat(
    compiled: CompiledRhs,
    fixture: QualificationFixture,
    shapes: Mapping[str, tuple[int, ...]],
    xp: Any,  # ruff: ignore[any-type]
) -> Mapping[str, Array]:
    flat = xp.concat([xp.reshape(fixture.state[name], (-1,)) for name in shapes])
    result = compiled.eval_fn(fixture.time, flat, **fixture.params)
    if result.shape != flat.shape:
        msg = f"flat evaluator returned shape {result.shape}; expected {flat.shape}"
        raise ValueError(msg)
    if array_namespace(result) is not xp:
        msg = "flat evaluator changed the input array namespace"
        raise ValueError(msg)
    output = {}
    offset = 0
    for name, shape in shapes.items():
        size = math.prod(shape)
        output[name] = xp.reshape(cast("Any", result)[offset : offset + size], shape)
        offset += size
    return output


def _compare(
    name: str,
    initial: Array,
    actual: Array,
    expected: Array,
    config: QualificationConfig,
) -> ArrayComparison:
    value, reference = _host(actual), _host(expected)
    errors = []
    if value.shape != initial.shape or reference.shape != initial.shape:
        errors.append(
            f"shape mismatch: input {initial.shape}, result {value.shape}, "
            f"reference {reference.shape}"
        )
    if actual.dtype != initial.dtype:
        errors.append(f"dtype changed from {initial.dtype} to {actual.dtype}")
    if array_namespace(actual) is not array_namespace(initial):
        errors.append("output changed the input array namespace")
    finite = bool(
        np.isfinite(_host(initial)).all()
        and np.isfinite(value).all()
        and np.isfinite(reference).all()
    )
    maximum, relative = None, None
    if not finite:
        errors.append("input, result, or reference contains non-finite values")
    if value.size == 0:
        errors.append("empty arrays do not provide numerical qualification")
    if finite and value.size and value.shape == reference.shape:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            difference = np.abs(value - reference)
            raw_maximum = float(np.max(difference))
            maximum = raw_maximum if math.isfinite(raw_maximum) else None
            denominator = np.maximum(
                np.abs(reference), max(config.atol, np.finfo(np.float64).tiny)
            )
            raw_relative = float(np.max(difference / denominator))
            relative = raw_relative if math.isfinite(raw_relative) else None
        if not np.allclose(value, reference, rtol=config.rtol, atol=config.atol):
            errors.append(f"reference disagreement: maximum absolute error {maximum}")
    return ArrayComparison(
        name,
        value.shape,
        reference.shape,
        str(initial.dtype),
        str(actual.dtype),
        str(expected.dtype),
        finite,
        not errors,
        maximum,
        relative,
        tuple(errors),
    )


def _balance(
    check: InvariantCheck,
    fixture: QualificationFixture,
    derivative: Mapping[str, Array],
) -> NumericalCheck:
    try:
        residual = _host(
            check.callback(fixture.time, fixture.state, derivative, fixture.params)
        )
    except Exception as exc:  # ruff: ignore[blind-except]
        return NumericalCheck(
            check.name,
            passed=False,
            max_abs_error=None,
            errors=(f"{type(exc).__name__}: {exc}",),
            condition=check.condition,
            atol=check.atol,
        )
    finite = bool(residual.size and np.isfinite(residual).all())
    maximum = float(np.max(np.abs(residual), initial=0)) if finite else None
    passed = maximum is not None and maximum <= check.atol
    errors = (
        () if passed else (f"balance residual exceeds {check.atol:g} or is non-finite",)
    )
    return NumericalCheck(
        check.name, passed, maximum, errors, check.condition, atol=check.atol
    )


def _directional(
    compiled: CompiledRhs,
    fixture: QualificationFixture,
    reference: ReferenceRhs,
    evaluator: Evaluator,
    config: QualificationConfig,
) -> NumericalCheck:
    direction = fixture.direction
    if direction is None:
        msg = "a state direction is required"
        raise ValueError(msg)
    xp = array_namespace(next(iter(fixture.state.values())))
    if not xp.__name__.startswith("jax"):
        return _derivative_result(
            config,
            passed=False,
            maximum=None,
            errors=("directional qualification requires JAX state arrays",),
        )
    jax = importlib.import_module("jax")
    step = config.derivative_step

    state = dict(fixture.state)
    _, tangent = jax.jvp(
        lambda values: _evaluate(compiled, fixture._replace(state=values), evaluator),
        (state,),
        (dict(direction),),
    )
    plus = {
        name: xp.add(value, xp.multiply(step, direction[name]))
        for name, value in state.items()
    }
    minus = {
        name: xp.subtract(value, xp.multiply(step, direction[name]))
        for name, value in state.items()
    }
    upper = reference(fixture.time, plus, fixture.params)
    lower = reference(fixture.time, minus, fixture.params)
    maximum = 0.0
    for name in state:
        expected = (_host(upper[name]) - _host(lower[name])) / (2 * step)
        actual = _host(tangent[name])
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            return _derivative_result(
                config,
                passed=False,
                maximum=None,
                errors=(f"non-finite derivative for {name!r}",),
            )
        maximum = max(maximum, float(np.max(np.abs(actual - expected), initial=0)))
        if not np.allclose(
            actual, expected, rtol=config.derivative_rtol, atol=config.derivative_atol
        ):
            return _derivative_result(
                config,
                passed=False,
                maximum=maximum,
                errors=(f"directional derivative disagrees for state {name!r}",),
            )
    return _derivative_result(config, passed=True, maximum=maximum)


def _derivative_result(
    config: QualificationConfig,
    *,
    passed: bool,
    maximum: float | None,
    errors: tuple[str, ...] = (),
) -> NumericalCheck:
    return NumericalCheck(
        "state_directional_derivative",
        passed,
        maximum,
        errors,
        rtol=config.derivative_rtol,
        atol=config.derivative_atol,
        step=config.derivative_step,
    )


def _inspect_fixture(  # ruff: ignore[too-many-arguments]
    compiled: CompiledRhs,
    fixture: QualificationFixture,
    reference: ReferenceRhs,
    evaluator: Evaluator,
    config: QualificationConfig,
    *,
    invariants: Sequence[InvariantCheck],
    backend: str,
) -> FixtureQualification:
    actual = _evaluate(compiled, fixture, evaluator)
    expected = reference(fixture.time, fixture.state, fixture.params)
    if set(actual) != set(fixture.state) or set(expected) != set(fixture.state):
        msg = "reference and result keys must match the fixture state"
        raise ValueError(msg)
    arrays = tuple(
        _compare(name, fixture.state[name], actual[name], expected[name], config)
        for name in fixture.state
    )
    checks = tuple(_balance(item, fixture, actual) for item in invariants)
    derivative = None
    if fixture.direction is not None:
        try:
            derivative = _directional(compiled, fixture, reference, evaluator, config)
        except Exception as exc:  # ruff: ignore[blind-except]
            derivative = _derivative_result(
                config,
                passed=False,
                maximum=None,
                errors=(f"{type(exc).__name__}: {exc}",),
            )
    return FixtureQualification(
        fixture.name,
        evaluator,
        backend,
        float(fixture.time),
        config.rtol,
        config.atol,
        arrays,
        checks,
        derivative,
    )


def _qualify_fixture(  # ruff: ignore[too-many-arguments]
    compiled: CompiledRhs,
    fixture: QualificationFixture,
    reference: ReferenceRhs,
    evaluator: Evaluator,
    config: QualificationConfig,
    *,
    invariants: Sequence[InvariantCheck],
) -> FixtureQualification:
    backend = "unknown"
    try:
        backend = array_namespace(next(iter(fixture.state.values()))).__name__
        return _inspect_fixture(
            compiled,
            fixture,
            reference,
            evaluator,
            config,
            invariants=invariants,
            backend=backend,
        )
    except Exception as exc:  # ruff: ignore[blind-except]
        # Arbitrary user callbacks are qualification failures. Cancellation
        # and other BaseException subclasses still propagate normally.
        return FixtureQualification(
            fixture.name,
            evaluator,
            backend,
            float(fixture.time),
            config.rtol,
            config.atol,
            errors=(f"{type(exc).__name__}: {exc}",),
        )


def qualify_rhs(  # ruff: ignore[too-many-arguments]
    compiled: CompiledRhs,
    fixtures: Sequence[QualificationFixture],
    reference: ReferenceRhs,
    *,
    evaluators: Sequence[Evaluator] = ("flat", "structured"),
    config: QualificationConfig | None = None,
    invariants: Sequence[InvariantCheck] = (),
) -> QualificationReport:
    """Compare compiled layouts against independently supplied equations.

    Args:
        compiled: Compiled RHS to inspect; structural validation is separate.
        fixtures: Named structured state/parameter fixtures. Supply a JAX
            state ``direction`` to request an additional derivative check.
        reference: Independent ``reference(t, state, params)`` returning
            derivatives keyed by the same state templates.
        evaluators: Requested layouts. Unavailable layouts produce failed
            records rather than silently falling back to another evaluator.
        config: Reference and derivative tolerances. Defaults assume double
            precision; adjust the tolerances and difference step for float32.
        invariants: Residual-to-zero callbacks with stated conditions and
            individual absolute tolerances, such as closed-system balance.

    Returns:
        Eager host diagnostics for every requested combination. Fixture-based
        agreement is not a proof of scientific validity or solution accuracy.

    Raises:
        ValueError: If fixtures/evaluators are empty, fixture names/time are
            invalid, or an invariant lacks valid conditions/tolerances.

    Examples:
        >>> import numpy as np
        >>> from op_system import compile_spec
        >>> compiled = compile_spec({
        ...     "kind": "expr",
        ...     "state": ["x"],
        ...     "equations": {"x": "-rate*x"},
        ... })
        >>> fixture = QualificationFixture(
        ...     "decay", {"x": np.array(4.0)}, {"rate": 0.5}
        ... )
        >>> def reference(time, state, params):
        ...     return {"x": -params["rate"] * state["x"]}
        >>> report = qualify_rhs(compiled, [fixture], reference)
        >>> report.ok
        True
        >>> len(report.fixtures)
        2
    """
    controls = config or QualificationConfig()
    if not fixtures or not evaluators:
        msg = "at least one fixture and evaluator are required"
        raise ValueError(msg)
    names = [item.name for item in fixtures]
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        msg = "fixture names must be unique and non-empty"
        raise ValueError(msg)
    if any(not math.isfinite(item.time) for item in fixtures):
        msg = "fixture times must be finite"
        raise ValueError(msg)
    if any(name not in {"flat", "structured", "block"} for name in evaluators):
        msg = "unknown evaluator"
        raise ValueError(msg)
    if any(
        not item.condition.strip() or not math.isfinite(item.atol) or item.atol < 0
        for item in invariants
    ):
        msg = "invariant checks need conditions and finite non-negative tolerances"
        raise ValueError(msg)
    return QualificationReport(
        tuple(
            _qualify_fixture(
                compiled, item, reference, evaluator, controls, invariants=invariants
            )
            for item in fixtures
            for evaluator in evaluators
        )
    )


__all__ = [
    "ArrayComparison",
    "FixtureQualification",
    "InvariantCheck",
    "NumericalCheck",
    "QualificationConfig",
    "QualificationFixture",
    "QualificationReport",
    "qualify_rhs",
]
