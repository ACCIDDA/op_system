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
    return (
        np
        .stack([np.asarray(value) for value in derivative.values()])
        .sum(axis=0)
        .sum(axis=-1)
    )


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


