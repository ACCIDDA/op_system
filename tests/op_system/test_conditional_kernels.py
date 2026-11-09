"""Independent conditional-kernel references and explicit joint-weight checks."""

from dataclasses import replace

import numpy as np
import pytest

from op_system import compile_spec
from op_system.kernels import ConditionalKernel, KernelAxis, KernelReduction


def _kernel(values: np.ndarray) -> ConditionalKernel:
    return ConditionalKernel(
        infection_age=(0.0, 1.0, 2.0),
        axes=(KernelAxis("host", ("low", "high")), KernelAxis("antigen", ("a", "b"))),
        values=values,
        source_identity="source-sha256-fixture",
        source_settings_identity="fixed-rates-and-independent-solver-fixture",
        observable="shedding-times-participation",
        units="transmission/day",
        age_units="day",
        conditioning=("initial-load=1",),
    )


def test_snapshot_is_immutable_and_content_identity_covers_values() -> None:
    """Mutation of input or an array view cannot change a published table."""
    values = np.arange(12.0).reshape(3, 2, 2)
    kernel = _kernel(values)
    identity = kernel.content_digest()
    values[:] = -1
    assert np.array_equal(kernel.values, np.arange(12.0).reshape(3, 2, 2))
    with pytest.raises(ValueError, match="WRITEABLE"):
        np.asarray(kernel.values).setflags(write=True)
    assert kernel.content_digest() == identity
    assert replace(kernel, values=values).content_digest() != identity
    assert replace(kernel, units="count").content_digest() != identity
    assert (
        replace(kernel, source_settings_identity="other").content_digest() != identity
    )


def test_joint_weights_and_reversed_axis_order_match_hand_average() -> None:
    """Correlated host-antigen weights do not become a product of marginals."""
    values = np.arange(12.0).reshape(3, 2, 2)
    kernel = _kernel(values)
    joint = np.array([[0.4, 0.1], [0.2, 0.3]])
    reduction = KernelReduction(("antigen", "host"), joint.T, "joint-law-v1")
    averaged = kernel.reduce(reduction)
    expected = np.array([
        sum(values[t, h, a] * joint[h, a] for h in range(2) for a in range(2))
        for t in range(3)
    ])
    assert np.allclose(averaged.values, expected)
    assert averaged.axes == ()
    assert averaged.reductions == (reduction,)
    marginal = kernel.reduce(KernelReduction(("host",), (0.25, 0.75), "host-law"))
    assert marginal.axes == (kernel.axes[1],)
    assert np.allclose(marginal.values, values[:, 0, :] * 0.25 + values[:, 1, :] * 0.75)


def test_nonlinear_mapping_before_average_retains_heterogeneity() -> None:
    """Population seropositivity is not thresholded mean antibody activity."""
    activity = _kernel(np.broadcast_to([[[0.0, 0.0], [4.0, 4.0]]], (3, 2, 2)))
    weights = KernelReduction(("host",), (0.75, 0.25), "host-law")
    mapped = replace(
        activity,
        values=(np.asarray(activity.values) > 0.5).astype(float),
        observable="seropositive",
        units="probability",
        source_settings_identity="original-settings-plus-cutoff-0.5",
    )
    assert np.allclose(mapped.reduce(weights).values, 0.25)
    assert np.all(np.asarray(activity.reduce(weights).values) > 0.5)


def test_ode_to_discrete_renewal_matches_independent_hand_coupling() -> None:
    """A source ODE retains joint traits before explicit population reduction."""
    rates = np.array([[0.2, 0.4], [0.6, 0.8]])
    compiled = compile_spec({
        "kind": "expr",
        "axes": [
            {"name": "host", "coords": ["low", "high"]},
            {"name": "antigen", "coords": ["a", "b"]},
        ],
        "state": ["X[host,antigen]"],
        "equations": {"X[host,antigen]": "-rate[host,antigen] * X[host,antigen]"},
    })
    state = np.ones((2, 2))
    samples = [state.copy()]
    step = 0.01
    for index in range(200):
        time = index * step

        def rhs(x: np.ndarray, time: float = time) -> np.ndarray:
            assert compiled.pytree_eval_fn is not None
            return np.asarray(compiled.pytree_eval_fn(time, {"X": x}, rate=rates)["X"])

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step * k1, time + 0.5 * step)
        k3 = rhs(state + 0.5 * step * k2, time + 0.5 * step)
        k4 = rhs(state + step * k3, time + step)
        state += step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (index + 1) % 100 == 0:
            samples.append(state.copy())
    weights = np.array([[0.4, 0.1], [0.1, 0.4]])
    population = _kernel(np.stack(samples)).reduce(
        KernelReduction(("host", "antigen"), weights, "joint-law")
    )
    analytic = np.array([
        sum(
            weights[h, a] * np.exp(-rates[h, a] * t) for h in range(2) for a in range(2)
        )
        for t in population.infection_age
    ])
    assert np.allclose(population.values, analytic, rtol=1e-10, atol=1e-12)
    incidence = [1.0]
    hand_incidence = [1.0]
    for generation in range(1, 7):
        incidence.append(
            sum(
                incidence[generation - lag] * np.asarray(population.values)[lag]
                for lag in range(1, min(2, generation) + 1)
            )
        )
        hand_incidence.append(
            sum(
                hand_incidence[generation - lag] * analytic[lag]
                for lag in range(1, min(2, generation) + 1)
            )
        )
    assert np.allclose(incidence, hand_incidence, rtol=1e-10)


@pytest.mark.parametrize("age", [(0.0, 0.0, 2.0), (-1.0, 1.0, 2.0), (0.0, np.nan, 2.0)])
def test_invalid_age_grid_is_refused(age: tuple[float, ...]) -> None:
    """Duplicate, negative or missing infection ages cannot define a kernel."""
    with pytest.raises(ValueError, match="grid"):
        replace(_kernel(np.ones((3, 2, 2))), infection_age=age)


def test_shape_axis_and_probability_errors_are_explicit() -> None:
    """No silent broadcast, automatic normalization or implicit axis removal."""
    kernel = _kernel(np.ones((3, 2, 2)))
    with pytest.raises(ValueError, match="unit total mass"):
        KernelReduction(("host",), (1.0, 1.0), "distribution")
    with pytest.raises(ValueError, match="shape"):
        replace(kernel, values=np.ones(3))
    with pytest.raises(ValueError, match="retained"):
        kernel.reduce(KernelReduction(("infection_age",), (1.0,), "distribution"))
    with pytest.raises(ValueError, match="coordinate order"):
        kernel.reduce(KernelReduction(("host",), (0.2, 0.3, 0.5), "distribution"))
