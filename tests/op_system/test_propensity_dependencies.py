"""State dependencies and elasticity bounds under ``reactants: auto`` (#256)."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import Any

import numpy as np
import pytest

from op_system import CompiledReaction, compile_spec

AGE = {"name": "age", "coords": ["a0", "a1", "a2"]}
STATE = ["S[age]", "I[age]", "R[age]", "Z"]
PARAMS: dict[str, object] = {"beta": 0.3, "c": np.full((3, 3), 0.5)}

FREQUENCY_DEPENDENT = (
    "beta * sum_over(I[age:a], age=a) / sum_over(S[age:a] + I[age:a] + R[age:a], age=a)"
)
CONTACT = (
    "beta * sum_over(c[age, age:a] * I[age:a], age=a) / (S[age] + I[age] + R[age])"
)


def _reaction(
    rate: str, *, frm: str | None = "S[age]", to: str = "I[age]"
) -> CompiledReaction:
    transition: dict[str, Any] = {
        "name": "r",
        "to": to,
        "rate": rate,
        "reactants": "auto",
    }
    if frm is not None:
        transition["from"] = frm
    (reaction,) = compile_spec({
        "kind": "transitions",
        "axes": [AGE],
        "state": STATE,
        "transitions": [transition],
    }).reactions
    return reaction


def _pinned(
    base: str,
) -> list[tuple[str, tuple[str, ...], tuple[tuple[str, int], ...]]]:
    return [(base, (), (("age", k),)) for k in range(3)]


def _dependencies(
    reaction: CompiledReaction,
) -> list[tuple[str, tuple[str, ...], tuple[tuple[str, int], ...]]]:
    assert all(d.order == 1 for d in reaction.dependencies)
    return [(d.state_base, d.state_axes, d.pinned) for d in reaction.dependencies]


def test_frequency_dependent_rate_publishes_dependencies_not_reactants() -> None:
    """``beta * S * sum(I) / N``: order 3, every state in N, reactants open."""
    reaction = _reaction(FREQUENCY_DEPENDENT)
    assert reaction.reactants_complete is False
    assert [(r.state_base, r.order) for r in reaction.reactants] == [("S", 1)]
    assert reaction.dependencies_complete is True
    assert reaction.propensity_order == 3
    assert _dependencies(reaction) == [
        ("S", ("age",), ()),
        *_pinned("I"),
        *_pinned("S"),
        *_pinned("R"),
    ]


def test_contact_reduction_pins_one_dependency_per_contact_coordinate() -> None:
    """A contact ``Reduce`` reads every infectious age class."""
    reaction = _reaction(CONTACT)
    assert reaction.propensity_order == 3
    assert _dependencies(reaction) == [
        ("S", ("age",), ()),
        *_pinned("I"),
        ("I", ("age",), ()),
        ("R", ("age",), ()),
    ]


def test_filtered_reduction_reads_only_its_coordinates() -> None:
    """``age=a in [a0, a2]`` depends on those two coordinates only."""
    reaction = _reaction("beta * sum_over(I[age:a], age=a in [a0, a2])")
    assert _dependencies(reaction) == [
        ("S", ("age",), ()),
        ("I", (), (("age", 0),)),
        ("I", (), (("age", 2),)),
    ]


@pytest.mark.parametrize(
    ("rate", "order"),
    [
        pytest.param("beta * (I[age] + R[age])", 2, id="sum-of-states"),
        pytest.param("beta * Z**0.5", 2, id="fractional-power"),
        pytest.param("beta / (Z + 1.0)", 2, id="state-in-denominator"),
        pytest.param("beta * (I[age] * R[age] + Z)", 3, id="sum-takes-max"),
    ],
)
def test_order_bounds_follow_the_expression(rate: str, order: int) -> None:
    """The bound adds over products and takes the max over sums."""
    reaction = _reaction(rate)
    assert reaction.dependencies_complete is True
    assert reaction.propensity_order == order


def test_source_only_births_read_the_population_without_reactants() -> None:
    """Births from a population sum consume nothing but read every cell."""
    reaction = _reaction(
        "beta * sum_over(S[age:a] + I[age:a], age=a)", frm=None, to="S[age]"
    )
    assert reaction.reactants == ()
    assert reaction.reactants_complete is False
    assert reaction.propensity_order == 1
    assert _dependencies(reaction) == [*_pinned("S"), *_pinned("I")]


def test_single_product_rates_keep_complete_reactants_only() -> None:
    """Mass action takes the exact reactant path and publishes no bound."""
    reaction = _reaction("beta * I[age]")
    assert reaction.reactants_complete is True
    assert reaction.dependencies == ()
    assert reaction.propensity_order is None
    assert reaction.dependencies_complete is False


def test_omitted_reactants_publish_no_dependencies() -> None:
    """Dependencies are opt-in through ``reactants: auto``."""
    (reaction,) = compile_spec({
        "kind": "transitions",
        "axes": [AGE],
        "state": STATE,
        "transitions": [
            {"name": "r", "from": "S[age]", "to": "I[age]", "rate": CONTACT}
        ],
    }).reactions
    assert reaction.dependencies == ()
    assert reaction.dependencies_complete is False


def _measured_elasticity(reaction: CompiledReaction, seed: int) -> float:
    """Measure ``sum_i |d log a / d log x_i|`` by finite differences.

    Returns:
        The largest total over channels at one random state.
    """
    rng = np.random.default_rng(seed)
    state = {
        "S": rng.uniform(1.0, 100.0, 3),
        "I": rng.uniform(1.0, 100.0, 3),
        "R": rng.uniform(1.0, 100.0, 3),
        "Z": np.asarray(rng.uniform(1.0, 100.0)),
    }
    base = np.log(np.asarray(reaction.propensity_fn(0.0, state, **PARAMS)))
    total = np.zeros_like(base)
    step = 1e-6
    for name, values in state.items():
        for cell in np.ndindex(values.shape):
            bumped = {key: np.array(value, copy=True) for key, value in state.items()}
            bumped[name][cell] *= 1.0 + step
            moved = np.log(np.asarray(reaction.propensity_fn(0.0, bumped, **PARAMS)))
            total += np.abs(moved - base) / step
    return float(total.max())


@pytest.mark.parametrize(
    "rate",
    [
        pytest.param(FREQUENCY_DEPENDENT, id="frequency-dependent"),
        pytest.param(CONTACT, id="contact"),
        pytest.param("beta * (I[age] * R[age] + Z)", id="sum-of-products"),
        pytest.param("beta * Z**0.5 / (I[age] + R[age])", id="power-over-sum"),
    ],
)
def test_order_bounds_the_measured_elasticity(rate: str) -> None:
    """The published order bounds finite-difference elasticities."""
    reaction = _reaction(rate)
    assert reaction.propensity_order is not None
    for seed in range(20):
        assert _measured_elasticity(reaction, seed) <= reaction.propensity_order + 1e-5


def test_dependencies_survive_pickling() -> None:
    """A round-tripped RHS rebuilds the same dependency metadata."""
    compiled = compile_spec({
        "kind": "transitions",
        "axes": [AGE],
        "state": STATE,
        "transitions": [
            {
                "name": "r",
                "from": "S[age]",
                "to": "I[age]",
                "rate": FREQUENCY_DEPENDENT,
                "reactants": "auto",
            }
        ],
    })
    restored = pickle.loads(pickle.dumps(compiled))  # ruff: ignore[suspicious-pickle-usage]
    (before,) = compiled.reactions
    (after,) = restored.reactions
    assert after.dependencies == before.dependencies
    assert after.propensity_order == before.propensity_order
    assert after.dependencies_complete is True
