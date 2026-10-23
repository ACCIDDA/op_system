"""User-facing numerical qualification and informative failed reports (#264)."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarks.hierarchical_model import fixture, model_spec, reference_rhs
from op_system import (
    InvariantCheck,
    QualificationConfig,
    QualificationFixture,
    compile_spec,
    qualify_rhs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from op_system import Array
    from op_system.qualification import Evaluator


def _reference(
    _t: float, state: Mapping[str, Array], params: Mapping[str, object]
) -> dict[str, np.ndarray]:
    return reference_rhs(state, params)


def _balance(
    _t: float,
    _state: Mapping[str, Array],
    derivative: Mapping[str, Array],
    _params: Mapping[str, object],
) -> np.ndarray:
    balance = (
        np
        .stack([np.asarray(value) for value in derivative.values()])
        .sum(axis=0)
        .sum(axis=-1)
    )
    return np.asarray(balance)


CLOSED = InvariantCheck(
    "group_mass", _balance, "Closed network with conservative stage transport"
)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_qualification_reports_every_fixture_layout_and_balance(
    backend: str, dtype: str
) -> None:
    """Fixtures receive inspectable layout, precision, and balance results."""
    module = (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )
    jax = pytest.importorskip("jax") if backend == "jax" else None
    previous = jax.config.read("jax_enable_x64") if jax else None
    if jax:
        enable = True
        jax.config.update("jax_enable_x64", enable)
    try:
        raw, params = fixture(dtype=dtype)
        state = {
            key: module.asarray(value, dtype=getattr(module, dtype))
            for key, value in raw.items()
        }
        values = {
            key: module.asarray(value, dtype=getattr(module, dtype))
            for key, value in params.items()
        }
        compiled = compile_spec(model_spec(3, 2))
        cases = [
            QualificationFixture("positive", state, values),
            QualificationFixture(
                "scaled",
                {key: value * 1.7 for key, value in state.items()},
                values,
                2.0,
            ),
        ]
        controls = (
            QualificationConfig(rtol=2e-5, atol=1e-6)
            if dtype == "float32"
            else QualificationConfig()
        )
        invariant = CLOSED._replace(atol=2e-6 if dtype == "float32" else 1e-10)

        # Torch references explicitly cross the host boundary; the utility
        # preserves input tensors and records eager diagnostics.
        def reference(_t: float, y: Any, p: Any) -> Any:  # ruff: ignore[any-type]
            def host(value: Any) -> Any:  # ruff: ignore[any-type]
                return (
                    value.detach().cpu().numpy()
                    if hasattr(value, "detach")
                    else np.asarray(value)
                )

            return reference_rhs(
                {k: host(v) for k, v in y.items()}, {k: host(v) for k, v in p.items()}
            )

        report = qualify_rhs(
            compiled,
            cases,
            reference,
            evaluators=("flat", "structured", "block"),
            config=controls,
            invariants=[invariant],
        )
        assert report.ok
        assert len(report.fixtures) == 6
        assert {item.fixture for item in report.fixtures} == {"positive", "scaled"}
        for item in report.fixtures:
            assert item.ok
            assert backend in item.backend
            assert item.rtol == controls.rtol
            assert all(value.finite and value.passed for value in item.arrays)
            assert all(value.shape == (3, 2) for value in item.arrays)
            assert item.invariants[0].condition == invariant.condition
            assert item.invariants[0].atol == invariant.atol
        encoded = json.loads(json.dumps(report.to_dict(), allow_nan=False))
        assert encoded["fixtures"][0]["arrays"][0]["state"] == "A"
        assert (
            encoded["fixtures"][0]["invariants"][0]["condition"] == invariant.condition
        )
    finally:
        if jax:
            jax.config.update("jax_enable_x64", previous)


def test_wrong_reference_reports_the_state_and_error() -> None:
    """An independent reference with a wrong equation cannot qualify."""
    state, params = fixture()
    compiled = compile_spec(model_spec(3, 2))

    def wrong(
        t: float, y: Mapping[str, Array], p: Mapping[str, object]
    ) -> dict[str, np.ndarray]:
        result = _reference(t, y, p)
        result["B"] += 1
        return result

    result = qualify_rhs(
        compiled, [QualificationFixture("wrong-equation", state, params)], wrong
    )
    assert not result.ok
    assert all(not item.ok for item in result.fixtures)
    comparison = result.fixtures[0].arrays[1]
    assert comparison.state == "B"
    assert comparison.max_abs_error == pytest.approx(1)
    assert "reference disagreement" in comparison.errors[0]


def test_wrong_compiled_rhs_and_violated_balance_are_reported() -> None:
    """A defective compiled derivative fails both parity and conserved mass."""
    state, params = fixture()
    compiled = compile_spec(model_spec(3, 2))
    original = compiled.pytree_eval_fn
    assert original is not None

    def wrong(t: Any, y: Any, **p: Any) -> Any:  # ruff: ignore[any-type]
        result = original(t, y, **p)
        return {**result, "A": result["A"] + 1}

    result = qualify_rhs(
        replace(compiled, pytree_eval_fn=wrong),
        [QualificationFixture("broken-rhs", state, params)],
        _reference,
        evaluators=("structured",),
        invariants=[CLOSED],
    )
    assert not result.ok
    assert result.fixtures[0].invariants[0].max_abs_error == pytest.approx(2)
    assert not result.fixtures[0].invariants[0].passed


@pytest.mark.parametrize("failure", ["shape", "dtype", "nan"])
def test_reference_shape_dtype_and_finite_checks(failure: str) -> None:
    """Loose numerical tolerance cannot hide malformed or non-finite outputs."""
    state, params = fixture()
    compiled = compile_spec(model_spec(3, 2))
    original = compiled.pytree_eval_fn
    assert original is not None

    def invalid(t: Any, y: Any, **p: Any) -> Any:  # ruff: ignore[any-type]
        output = original(t, y, **p)
        output["A"] = (
            output["A"].T
            if failure == "shape"
            else output["A"].astype(np.float32)
            if failure == "dtype"
            else np.full_like(output["A"], np.nan)
        )
        return output

    result = qualify_rhs(
        replace(compiled, pytree_eval_fn=invalid),
        [QualificationFixture("malformed", state, params)],
        _reference,
        evaluators=("structured",),
        config=QualificationConfig(rtol=1, atol=1),
    )
    assert not result.ok
    assert result.fixtures[0].arrays[0].errors
    json.dumps(result.to_dict(), allow_nan=False)


def test_coupled_model_never_falls_back_to_independent_blocks() -> None:
    """Requesting unavailable block evaluation returns a failed named record."""
    state, params = fixture(coupled=True)
    compiled = compile_spec(model_spec(3, 2, coupled=True))

    def reference(
        _t: float, y: Mapping[str, Array], p: Mapping[str, object]
    ) -> dict[str, np.ndarray]:
        return reference_rhs(y, p, coupled=True)

    result = qualify_rhs(
        compiled,
        [QualificationFixture("coupled", state, params)],
        reference,
        evaluators=("flat", "structured", "block"),
    )
    assert result.fixtures[0].ok
    assert result.fixtures[1].ok
    assert not result.fixtures[2].ok
    assert "coupled" in result.fixtures[2].errors[0]


def test_callback_failure_preserves_value_results_and_conditions() -> None:
    """A failed user balance callback remains an informative per-check failure."""
    state, params = fixture()

    def invalid(*_args: object) -> Array:
        msg = "the closed-system assumption is violated"
        raise ValueError(msg)

    check = InvariantCheck("closure", invalid, "No external source")
    result = qualify_rhs(
        compile_spec(model_spec(3, 2)),
        [QualificationFixture("open", state, params)],
        _reference,
        invariants=[check],
    )
    assert not result.ok
    assert all(item.passed for item in result.fixtures[0].arrays)
    assert "closed-system assumption" in result.fixtures[0].invariants[0].errors[0]


@pytest.mark.parametrize("layout", ["flat", "structured", "block"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_jax_directional_checks_against_independent_reference(
    layout: Evaluator, dtype: str
) -> None:
    """JAX state directions match central differences of independent equations."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    previous = jax.config.read("jax_enable_x64")
    enable = True
    jax.config.update("jax_enable_x64", enable)
    try:
        state, params = fixture(dtype=dtype)
        state = {name: jnp.asarray(value) for name, value in state.items()}
        params = {name: jnp.asarray(value) for name, value in params.items()}
        direction = {name: jnp.full_like(value, 0.1) for name, value in state.items()}
        case = QualificationFixture("jax-direction", state, params, direction=direction)
        config = (
            QualificationConfig(
                rtol=2e-5,
                atol=1e-6,
                derivative_step=2e-3,
                derivative_rtol=2e-3,
                derivative_atol=1e-5,
            )
            if dtype == "float32"
            else QualificationConfig()
        )
        result = qualify_rhs(
            compile_spec(model_spec(3, 2)),
            [case],
            _reference,
            evaluators=(layout,),
            config=config,
        )
        assert result.ok
        check = result.fixtures[0].derivative
        assert check is not None
        assert check.passed
        assert check.step == pytest.approx(config.derivative_step)
        assert check.max_abs_error is not None
        assert check.max_abs_error < (2e-5 if dtype == "float32" else 1e-8)

        def bad_derivative(
            t: float, y: Mapping[str, Array], p: Mapping[str, object]
        ) -> dict[str, np.ndarray]:
            values = _reference(t, y, p)
            return {
                name: value + 4 * (np.asarray(y[name]) - np.asarray(state[name]))
                for name, value in values.items()
            }

        failed = qualify_rhs(
            compile_spec(model_spec(3, 2)),
            [case],
            bad_derivative,
            evaluators=(layout,),
            config=config,
        )
        assert not failed.ok
        assert all(item.passed for item in failed.fixtures[0].arrays)
        assert failed.fixtures[0].derivative is not None
        assert not failed.fixtures[0].derivative.passed
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_directional_check_is_explicitly_unqualified_on_numpy() -> None:
    """The report does not pretend NumPy has the same native AD contract as JAX."""
    state, params = fixture()
    direction = {name: np.ones_like(value) for name, value in state.items()}
    result = qualify_rhs(
        compile_spec(model_spec(3, 2)),
        [QualificationFixture("numpy-direction", state, params, direction=direction)],
        _reference,
        evaluators=("structured",),
    )
    assert not result.ok
    assert result.fixtures[0].derivative is not None
    assert "requires JAX" in result.fixtures[0].derivative.errors[0]


def test_scalar_flat_evaluator_and_template_order() -> None:
    """Flat qualification supports scalar models and ignores fixture key order."""
    compiled = compile_spec({
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {"x": "-rate*x", "y": "rate*x-y"},
    })
    state = {"y": np.asarray(3.0), "x": np.asarray(2.0)}
    params = {"rate": 0.5}

    def reference(
        _t: float, y: Mapping[str, Array], _p: Mapping[str, object]
    ) -> dict[str, np.ndarray]:
        flow = np.asarray(_p["rate"]) * np.asarray(y["x"])
        return {"x": -flow, "y": flow - np.asarray(y["y"])}

    result = qualify_rhs(
        compiled,
        [QualificationFixture("scalar", state, params)],
        reference,
        evaluators=("flat",),
    )
    assert result.ok


def test_block_axis_can_be_inside_the_state_layout() -> None:
    """Block slicing/reassembly follow metadata rather than axis-zero guesses."""
    spec: dict[str, object] = {
        "kind": "expr",
        "axes": [
            {"name": "stage", "coords": ["a", "b"]},
            {"name": "group", "coords": ["one", "two", "three"]},
        ],
        "state": ["X[stage,group]", "Y[stage,group]"],
        "equations": {
            "X[stage,group]": "-rate[group] * X[stage,group]",
            "Y[stage,group]": "rate[group] * X[stage,group]",
        },
        "factorize_axes": ["group"],
    }
    state = {"X": np.arange(1.0, 7.0).reshape(2, 3), "Y": np.ones((2, 3))}
    params = {"rate": np.array([0.1, 0.2, 0.3])}

    def reference(
        _t: float, y: Mapping[str, Array], p: Mapping[str, object]
    ) -> dict[str, np.ndarray]:
        flux = np.empty((2, 3))
        for stage in range(2):
            for group in range(3):
                flux[stage, group] = (
                    np.asarray(p["rate"])[group] * np.asarray(y["X"])[stage, group]
                )
        return {"X": -flux, "Y": flux}

    result = qualify_rhs(
        compile_spec(spec),
        [QualificationFixture("axis-one", state, params)],
        reference,
        evaluators=("flat", "structured", "block"),
    )
    assert result.ok


def test_nonfinite_reference_has_json_safe_diagnostics() -> None:
    """Non-finite values fail qualification without producing invalid JSON numbers."""
    state, params = fixture()

    def reference(
        _t: float, y: Mapping[str, Array], _p: Mapping[str, object]
    ) -> dict[str, np.ndarray]:
        return {name: np.full_like(value, np.inf) for name, value in y.items()}

    result = qualify_rhs(
        compile_spec(model_spec(3, 2)),
        [QualificationFixture("nonfinite", state, params)],
        reference,
    )
    assert not result.ok
    assert not result.fixtures[0].arrays[0].finite
    assert result.fixtures[0].arrays[0].max_abs_error is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_flat_result_cannot_hide_extra_output_entries() -> None:
    """Unpacking cannot silently discard a malformed flat derivative suffix."""
    state, params = fixture()
    compiled = compile_spec(model_spec(3, 2))
    original = compiled.eval_fn

    def wrong(t: Any, y: Any, **p: Any) -> Any:  # ruff: ignore[any-type]
        return np.concatenate([original(t, y, **p), np.asarray([1.0])])

    report = qualify_rhs(
        replace(compiled, eval_fn=wrong),
        [QualificationFixture("extra-output", state, params)],
        _reference,
        evaluators=("flat",),
    )
    assert not report.ok
    assert "shape" in report.fixtures[0].errors[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [("rtol", -1), ("atol", np.nan), ("derivative_step", 0), ("rtol", True)],
)
def test_invalid_controls_fail_before_numerical_evaluation(
    field: str, value: float
) -> None:
    """Qualification controls are validated at construction."""
    with pytest.raises(ValueError, match=field):
        QualificationConfig(**{field: value})
