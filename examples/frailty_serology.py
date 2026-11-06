"""Catalytic serology with a discretized gamma frailty on the force of infection.

Run ``uv run python examples/frailty_serology.py``. All inputs are synthetic.

Seronegative ``S`` and seropositive ``P`` fractions evolve in age ``a`` for a
cohort initialized seronegative at age zero. Each quadrature node ``z_q`` scales
the force of infection ``lam``; ``rho`` is the seroreversion rate. The
``frailty`` axis carries fixed generalized Gauss-Laguerre weights for a
Gamma(k, 1/k) multiplier (mean one, coefficient of variation ``1/sqrt(k)``) at
a fixed shape and order. Measures are static, so inferring ``k`` would require
recompiling the model.

Two observables expose an imperfect assay (sensitivity ``sens``, specificity
``spec``):

* ``apparent_conditional[frailty]`` keeps the frailty axis. A clustered survey
  likelihood needs it, because members of one cluster share ``z``: sum their
  log-probabilities under each node, then log-sum-exp over nodes with the
  weights.
* ``apparent_marginal`` integrates over the frailty measure, giving the
  population-marginal seroprevalence of an unclustered random sample.

The checks compare the derivative with the transition rates, and the
observables with closed forms (``rho = 0``) and with a high-order reference
(``rho > 0``). Age integration belongs to a solver such as op_engine. The
analytic solution here only evaluates the compiled observables.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from op_system import compile_spec


def gamma_quadrature(shape: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights integrating against a Gamma(shape, 1/shape) density.

    Golub-Welsch on the generalized Laguerre recurrence with
    ``alpha = shape - 1``; nodes are rescaled by ``1/shape`` and weights are
    normalized to sum to one.

    Returns:
        Multiplier nodes ``z`` (mean-one scale) and probability weights.

    Raises:
        ValueError: If ``shape`` is not positive or ``order`` is below one.
    """
    if shape <= 0 or order < 1:
        msg = "shape must be positive and order at least one"
        raise ValueError(msg)
    index = np.arange(order, dtype=float)
    alpha = shape - 1.0
    diagonal = 2.0 * index + alpha + 1.0
    off = np.sqrt(index[1:] * (index[1:] + alpha))
    nodes, vectors = np.linalg.eigh(
        np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
    )
    return nodes / shape, vectors[0] ** 2


def model_spec(weights: np.ndarray) -> dict[str, Any]:
    """Describe catalytic serology over a weighted frailty axis.

    Returns:
        A transitions specification with conditional and marginal observables.
    """
    labels = [f"q{i}" for i in range(len(weights))]
    intensive = {"representation": "density", "measure_axes": ["frailty"]}
    apparent = "sens*P[frailty] + (1 - spec)*S[frailty]"
    return {
        "kind": "transitions",
        "axes": [
            {
                "name": "frailty",
                "type": "categorical",
                "coords": labels,
                "measure": {"kind": "quadrature", "weights": weights.tolist()},
            }
        ],
        "state": ["S[frailty]", "P[frailty]"],
        "state_metadata": {"S": intensive, "P": intensive},
        "transitions": [
            {
                "name": "seroconversion",
                "from": "S[frailty]",
                "to": "P[frailty]",
                "rate": "lam*z[frailty]",
            },
            {
                "name": "seroreversion",
                "from": "P[frailty]",
                "to": "S[frailty]",
                "rate": "rho",
            },
        ],
        "parameters": {
            "sens": {"support": "unit_interval"},
            "spec": {"support": "unit_interval"},
        },
        "observables": {
            "apparent_conditional": {"expr": apparent, "axes": ["frailty"]},
            "apparent_marginal": {
                "expr": apparent,
                "reduction": {"kind": "integral", "axes": ["frailty"]},
            },
        },
    }


def seropositive(z: np.ndarray, lam: float, rho: float, age: float) -> np.ndarray:
    """Analytic seropositive fraction at ``age`` for each node.

    Returns:
        ``P(a) = lam z / (lam z + rho) (1 - exp(-(lam z + rho) a))``.
    """
    total = lam * z + rho
    return lam * z / total * (1.0 - np.exp(-total * age))


def closed_form_marginal(
    shape: float, lam: float, age: float, *, sens: float, spec: float
) -> float:
    """Population-marginal apparent seroprevalence without seroreversion.

    Returns:
        ``sens (1 - L) + (1 - spec) L`` with ``L = (1 + lam a / k)^(-k)``.
    """
    never = (1.0 + lam * age / shape) ** (-shape)
    return float(sens * (1.0 - never) + (1.0 - spec) * never)


def evaluate(  # ruff: ignore[too-many-arguments] - every argument is a model input.
    shape: float,
    order: int,
    *,
    lam: float,
    rho: float,
    age: float,
    sens: float,
    spec: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compile at one order and evaluate observables on the analytic state.

    Returns:
        Marginal and conditional apparent seroprevalence, and the derivative.
    """
    z, weights = gamma_quadrature(shape, order)
    compiled = compile_spec(model_spec(weights))
    positive = seropositive(z, lam, rho, age)
    state = np.concatenate([1.0 - positive, positive])
    observables = compiled.observables
    marginal = observables["apparent_marginal"].eval_fn(
        age, state, sens=sens, spec=spec
    )
    conditional = observables["apparent_conditional"].eval_fn(
        age, state, sens=sens, spec=spec
    )
    derivative = compiled.eval_fn(age, state, lam=lam, z=z, rho=rho)
    return float(np.asarray(marginal)), np.asarray(conditional), np.asarray(derivative)


SETTINGS = {"lam": 0.02, "age": 30.0, "sens": 0.9, "spec": 0.99}


def check_without_reversion(shape: float = 2.0, order: int = 8) -> float:
    """Compare compiled values with closed forms when ``rho = 0``.

    Returns:
        The closed-form marginal apparent seroprevalence.
    """
    lam, age = SETTINGS["lam"], SETTINGS["age"]
    sens, spec = SETTINGS["sens"], SETTINGS["spec"]
    marginal, conditional, derivative = evaluate(shape, order, rho=0.0, **SETTINGS)
    expected = closed_form_marginal(shape, lam, age, sens=sens, spec=spec)
    np.testing.assert_allclose(marginal, expected, rtol=1e-10)
    z, _ = gamma_quadrature(shape, order)
    positive = seropositive(z, lam, 0.0, age)
    np.testing.assert_allclose(
        conditional, sens * positive + (1 - spec) * (1 - positive)
    )
    flow = lam * z * (1 - positive)
    np.testing.assert_allclose(derivative, np.concatenate([-flow, flow]))
    return expected


def convergence(shape: float = 2.0, rho: float = 0.05) -> dict[int, float]:
    """Absolute marginal error by quadrature order against order 64.

    Returns:
        Errors keyed by order.
    """
    reference = evaluate(shape, 64, rho=rho, **SETTINGS)[0]
    return {
        order: abs(evaluate(shape, order, rho=rho, **SETTINGS)[0] - reference)
        for order in (2, 4, 8, 16)
    }


def main() -> None:
    """Run the checks and print convergence across quadrature orders."""
    expected = check_without_reversion()
    errors = convergence()
    np.testing.assert_array_less(errors[16], 1e-10)
    print(f"closed form (rho=0): {expected:.12f}")  # ruff: ignore[print]
    for order, error in errors.items():
        print(f"order {order:2d}, rho=0.05: |error| {error:.2e}")  # ruff: ignore[print]


if __name__ == "__main__":
    main()
