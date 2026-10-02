"""Renewal birth reductions into pinned destination cells (issue #239)."""

from __future__ import annotations

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
