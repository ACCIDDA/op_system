"""Renewal birth reductions into pinned destination cells (issue #239)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from op_system import compile_spec
from op_system._errors import InvalidRhsSpecError
from op_system.specs import normalize_transitions_rhs


def _renewal_spec(rate: str) -> dict[str, object]:
    return {
        "kind": "transitions",
        "axes": [{"name": "age", "coords": ["a0", "a1", "a2"]}],
        "state": ["N[age]"],
        "transitions": [
            {
                "name": "renewal",
                "from": None,
                "to": "N[age=a0]",
                "rate": rate,
                "reactants": [],
            },
        ],
    }


@pytest.mark.parametrize(
    "rate",
    [
        "sum_over(B[age:a] * N[age:a], age=a)",
        "sum_over(B[age] * N[age], age=age)",
        "apply_along(B[age:a] * N[age:a], age=a, kernel=sum)",
    ],
)
def test_renewal_reduction_has_one_channel_and_matches_rhs(rate: str) -> None:
    """A reduction is one total birth hazard, with no donor depletion."""
    compiled = compile_spec(_renewal_spec(rate))
    assert len(compiled.reactions) == 1
    (birth,) = compiled.reactions
    assert birth.from_base is None
    assert birth.from_axes == birth.to_axes == ()
    assert birth.full_axes == ("age",)
    assert birth.pinned == (("age", 0),)
    assert birth.from_pinned == ()
    assert birth.sum_axes == ()
    assert birth.reactants == ()
    assert birth.reactants_complete is True
    assert compiled.pytree_eval_fn is not None

    rng = np.random.default_rng(239)
    for _ in range(5):
        population = rng.integers(0, 100, size=3).astype(float)
        fertility = rng.uniform(0.0, 1.0, size=3)
        expected = np.array([fertility @ population, 0.0, 0.0])
        params = {"B": fertility}
        state = {"N": population}
        propensity = np.asarray(birth.propensity_fn(0.0, state, **params))
        assert propensity.shape == ()
        np.testing.assert_allclose(propensity, expected[0])
        np.testing.assert_allclose(
            compiled.eval_fn(np.asarray(0.0), population, **params), expected
        )
        np.testing.assert_allclose(
            compiled.pytree_eval_fn(np.asarray(0.0), state, **params)["N"], expected
        )


def test_reduced_and_free_axes_are_distinguished() -> None:
    """Reducing one occurrence must not hide a free occurrence of its axis."""
    spec = _renewal_spec("B[age] * sum_over(N[age:a], age=a)")
    assert normalize_transitions_rhs(spec).reactions_ir == ()


@pytest.mark.parametrize("helper", ["sum_over", "apply_along"])
def test_renewal_unknown_reduction_axis_raises_spec_error(helper: str) -> None:
    """An undefined reduction axis gives a structural spec error."""
    spec = _renewal_spec(f"{helper}(N[age:a], missing=a)")
    with pytest.raises(InvalidRhsSpecError, match=r"missing|unknown axis"):
        compile_spec(spec)


@pytest.mark.parametrize("age_first", [True, False])
@pytest.mark.parametrize("aliased", [True, False])
def test_renewal_retains_group_channels_and_pins_age(
    *, age_first: bool, aliased: bool
) -> None:
    """Each group gets its own reduced hazard, independently of axis order."""
    axes = "age,group" if age_first else "group,age"
    bound_axes = "age:a,group" if age_first else "group,age:a"
    destination = "age=a0,group" if age_first else "group,age=a0"
    rate = f"sum_over(B[age:a] * (N[{bound_axes}] + M[{bound_axes}]), age=a)"
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["a0", "a1", "a2"]},
            {"name": "group", "coords": ["g0", "g1"]},
        ],
        "state": [f"N[{axes}]", f"M[{axes}]"],
        "aliases": {"birth_flux[group]": rate} if aliased else {},
        "transitions": [
            {
                "name": "renewal",
                "to": f"N[{destination}]",
                "rate": "birth_flux[group]" if aliased else rate,
                "reactants": [],
            },
        ],
    }
    compiled = compile_spec(spec)
    (birth,) = compiled.reactions
    assert birth.from_axes == birth.to_axes == ("group",)
    assert birth.full_axes == tuple(axes.split(","))
    assert birth.pinned == (("age", 0),)
    assert birth.sum_axes == ()
    assert compiled.pytree_eval_fn is not None

    population = np.arange(6, dtype=float).reshape(3, 2) + 1.0
    other = population + 10.0
    fertility = np.array([0.1, 0.2, 0.3])
    expected_rate = fertility @ (population + other)
    expected_rhs = np.zeros_like(population)
    expected_rhs[0] = expected_rate
    if not age_first:
        population = population.T
        other = other.T
        expected_rhs = expected_rhs.T
    state = {"N": population, "M": other}
    np.testing.assert_allclose(
        np.asarray(birth.propensity_fn(0.0, state, B=fertility)), expected_rate
    )
    rhs = compiled.pytree_eval_fn(np.asarray(0.0), state, B=fertility)
    np.testing.assert_allclose(rhs["N"], expected_rhs)
    np.testing.assert_allclose(rhs["M"], np.zeros_like(other))
    flat = np.concatenate([population.ravel(), other.ravel()])
    np.testing.assert_allclose(
        compiled.eval_fn(np.asarray(0.0), flat, B=fertility),
        np.concatenate([expected_rhs.ravel(), np.zeros(other.size)]),
    )


def test_filtered_renewal_reduction_matches_rhs() -> None:
    """A fertility reduction may include only selected age coordinates."""
    compiled = compile_spec(
        _renewal_spec("sum_over(B[age:a] * N[age:a], age=a in [a1, a2])")
    )
    (birth,) = compiled.reactions
    assert compiled.pytree_eval_fn is not None
    state = {"N": np.array([100.0, 20.0, 30.0])}
    fertility = np.array([5.0, 0.2, 0.3])
    expected = np.array([13.0, 0.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(birth.propensity_fn(0.0, state, B=fertility)), 13.0
    )
    np.testing.assert_allclose(
        compiled.eval_fn(np.asarray(0.0), state["N"], B=fertility), expected
    )
    np.testing.assert_allclose(
        compiled.pytree_eval_fn(np.asarray(0.0), state, B=fertility)["N"], expected
    )


def _age_population_spec() -> dict[str, object]:
    """Build three living age bins and an absorbing departure counter.

    Returns:
        Named renewal, aging, and departure transitions.
    """
    spec = _renewal_spec("sum_over(B[age:a] * N[age:a], age=a)")
    spec["state"] = ["N[age]", "D[age]"]
    transitions = spec["transitions"]
    assert isinstance(transitions, list)
    transitions.extend([
        {
            "name": "depart",
            "from": "N[age]",
            "to": "D[age]",
            "rate": "mu",
            "reactants": [{"state": "N[age]", "order": 1}],
        },
        *(
            {
                "name": f"age_{k}",
                "from": f"N[age=a{k}]",
                "to": f"N[age=a{k + 1}]",
                "rate": "aging",
                "reactants": [{"state": f"N[age=a{k}]", "order": 1}],
            }
            for k in range(2)
        ),
    ])
    return spec


@pytest.mark.parametrize("mu", [0.1, 0.8])
@pytest.mark.parametrize("width", [0.25, 2.0])
def test_renewal_age_population_has_exponential_stationary_live_distribution(
    mu: float, width: float
) -> None:
    """Exponential-fitted aging preserves bin masses of the continuum density."""
    compiled = compile_spec(_age_population_spec())
    assert compiled.pytree_eval_fn is not None
    survival = math.exp(-mu * width)
    # Integrate mu*exp(-mu*a) over [0,h), [h,2h), and [2h,infinity).
    population = 1000.0 * np.array([
        1.0 - survival,
        survival * (1.0 - survival),
        survival**2,
    ])
    params = {
        "mu": np.asarray(mu),
        "aging": np.asarray(mu / math.expm1(mu * width)),
        "B": np.full(3, mu),
    }
    state = {"N": population, "D": np.zeros(3)}
    rhs = compiled.pytree_eval_fn(np.asarray(0.0), state, **params)
    np.testing.assert_allclose(rhs["N"], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(rhs["D"], mu * population)
    np.testing.assert_allclose(
        compiled.eval_fn(
            np.asarray(0.0), np.concatenate(list(state.values())), **params
        ),
        np.concatenate([np.zeros(3), mu * population]),
        atol=1e-12,
    )
    birth = next(r for r in compiled.reactions if r.name == "renewal")
    np.testing.assert_allclose(
        np.asarray(birth.propensity_fn(0.0, state, **params)), mu * population.sum()
    )


def test_reaction_scatter_reconstructs_age_population_rhs_at_random_states() -> None:
    """Birth, aging, and departure artifacts reproduce both population drifts."""
    compiled = compile_spec(_age_population_spec())
    assert compiled.pytree_eval_fn is not None
    assert len(compiled.reactions) == 4
    rng = np.random.default_rng(239)
    for _ in range(5):
        state = {"N": rng.uniform(0.0, 100.0, 3), "D": rng.uniform(0.0, 10.0, 3)}
        params = {"B": np.full(3, 0.2), "mu": np.asarray(0.2), "aging": np.asarray(2.0)}
        reconstructed = {"N": np.zeros(3), "D": np.zeros(3)}
        for reaction in compiled.reactions:
            propensity = np.asarray(reaction.propensity_fn(0.0, state, **params))
            if reaction.from_base is not None:
                source = (
                    reaction.from_pinned[0][1] if reaction.from_pinned else slice(None)
                )
                reconstructed[reaction.from_base][source] -= propensity
            target = reaction.pinned[0][1] if reaction.pinned else slice(None)
            reconstructed[reaction.to_base][target] += propensity
        rhs = compiled.pytree_eval_fn(np.asarray(0.0), state, **params)
        for base in state:
            np.testing.assert_allclose(reconstructed[base], rhs[base])
        # Constant fertility equal to mortality conserves the living total
        # at every state, while the absorbing departure counter increases.
        np.testing.assert_allclose(rhs["N"].sum(), 0.0, atol=1e-12)
        np.testing.assert_allclose(rhs["D"].sum(), 0.2 * state["N"].sum())


def test_jax_renewal_reduction_is_dynamic_under_jit_and_vmap() -> None:
    """Changing state and fertility values must change compiled birth hazards."""
    jax = pytest.importorskip("jax")
    xp = pytest.importorskip("jax.numpy")
    compiled = compile_spec(_renewal_spec("sum_over(B[age:a] * N[age:a], age=a)"))
    assert compiled.pytree_eval_fn is not None
    population = xp.asarray([10.0, 20.0, 30.0])
    fertility = xp.asarray([0.1, 0.2, 0.3])
    tree_fn = jax.jit(compiled.pytree_eval_fn)
    np.testing.assert_allclose(
        np.asarray(tree_fn(0.0, {"N": population}, B=fertility)["N"]),
        [14.0, 0.0, 0.0],
    )
    flat_fn = jax.jit(compiled.eval_fn)
    np.testing.assert_allclose(
        np.asarray(flat_fn(0.0, population, B=2 * fertility)), [28.0, 0.0, 0.0]
    )
    propensity_fn = jax.jit(
        jax.vmap(compiled.reactions[0].propensity_fn, in_axes=(None, 0))
    )
    rates = propensity_fn(
        0.0,
        {"N": xp.stack([population, 2 * population])},
        B=xp.stack([fertility, fertility]),
    )
    assert rates.__array_namespace__() is xp
    np.testing.assert_allclose(np.asarray(rates), [14.0, 28.0])


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_pinned_renewal_equations_assemble_unrolled_pytree_bins(backend: str) -> None:
    """An explicit boundary equation assembles scalar bins without extra axes."""
    compiled = compile_spec({
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["a0", "a1", "a2"]}],
        "state": ["N[age]"],
        "equations": {
            "N__age_a0": "sum_over(B[age:a] * N[age:a], age=a)",
            "N__age_a1": "0.0",
            "N__age_a2": "0.0",
        },
    })
    assert compiled.pytree_eval_fn is not None
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    state = {"N": xp.asarray([10.0, 20.0, 30.0])}
    fertility = xp.asarray([0.1, 0.2, 0.3])
    tree = compiled.pytree_eval_fn(0.0, state, B=fertility)
    assert tree["N"].shape == (3,)
    assert tree["N"].__array_namespace__() is xp
    np.testing.assert_allclose(np.asarray(tree["N"]), [14.0, 0.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(compiled.eval_fn(0.0, state["N"], B=fertility)), [14.0, 0.0, 0.0]
    )
