"""Catalytic serology over a weighted frailty axis (examples/frailty_serology.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from examples.frailty_serology import (
    closed_form_marginal,
    convergence,
    evaluate,
    gamma_quadrature,
    main,
    model_spec,
    seropositive,
)
from op_system import compile_spec


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0, 8.0])
def test_quadrature_matches_gamma_moments(shape: float) -> None:
    """Weights sum to one; mean one and variance 1/k are exact at order 8."""
    z, w = gamma_quadrature(shape, 8)
    assert w.sum() == pytest.approx(1.0)
    assert float(w @ z) == pytest.approx(1.0)
    assert float(w @ (z - 1.0) ** 2) == pytest.approx(1.0 / shape)


def test_quadrature_rejects_invalid_inputs() -> None:
    """Shape must be positive and order at least one."""
    with pytest.raises(ValueError, match="shape must be positive"):
        gamma_quadrature(0.0, 4)
    with pytest.raises(ValueError, match="order"):
        gamma_quadrature(1.0, 0)


@pytest.mark.parametrize("shape", [0.5, 2.0])
def test_marginal_matches_closed_form_without_reversion(shape: float) -> None:
    """The integral observable equals the Laplace-transform closed form."""
    marginal, _, _ = evaluate(
        shape, 32, lam=0.02, rho=0.0, age=30.0, sens=0.9, spec=0.99
    )
    expected = closed_form_marginal(shape, 0.02, 30.0, sens=0.9, spec=0.99)
    assert marginal == pytest.approx(expected, rel=1e-6 if shape < 1 else 1e-10)


def test_conditional_axis_and_conservation() -> None:
    """Conditional values keep the frailty axis; transitions conserve S + P."""
    z, weights = gamma_quadrature(2.0, 6)
    compiled = compile_spec(model_spec(weights))
    observable = compiled.observables["apparent_conditional"]
    assert observable.output_shape == (6,)
    positive = seropositive(z, 0.03, 0.1, 20.0)
    state = np.concatenate([1.0 - positive, positive])
    derivative = np.asarray(compiled.eval_fn(20.0, state, lam=0.03, z=z, rho=0.1))
    assert np.allclose(derivative[:6] + derivative[6:], 0.0)
    expected = 0.03 * z * (1.0 - positive) - 0.1 * positive
    assert np.allclose(derivative[6:], expected)
    marginal = compiled.observables["apparent_marginal"].eval_fn(
        20.0, state, sens=1.0, spec=1.0
    )
    assert float(np.asarray(marginal)) == pytest.approx(float(weights @ positive))


def test_quadrature_converges_with_reversion() -> None:
    """Error against a high-order reference falls with quadrature order."""
    errors = convergence()
    assert errors[2] > errors[4] > errors[8]
    assert errors[8] < 1e-10
    assert not any(math.isnan(e) for e in errors.values())


def test_example_main_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """The documented command runs its own checks."""
    main()
    assert "order 16" in capsys.readouterr().out
