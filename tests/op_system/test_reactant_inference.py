"""Reactant completeness from the rate, and ``reactants: auto`` (issue #255)."""

from __future__ import annotations

import re
from typing import Any

import pytest

from op_system import compile_spec
from op_system._errors import InvalidRhsSpecError

AXES = [
    {"name": "age", "coords": ["a0", "a1"]},
    {"name": "vax", "coords": ["u", "v"]},
]
STATE = ["S[age,vax]", "I[age,vax]", "J[age]", "Z", "R[age,vax]"]


def _spec(
    rate: str,
    *,
    reactants: object = None,
    frm: str | None = "S[age,vax]",
    to: str = "I[age,vax]",
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    transition: dict[str, Any] = {"name": "r", "to": to, "rate": rate}
    if frm is not None:
        transition["from"] = frm
    if reactants is not None:
        transition["reactants"] = reactants
    spec: dict[str, Any] = {
        "kind": "transitions",
        "axes": AXES,
        "state": STATE,
        "transitions": [transition],
    }
    if aliases:
        spec["aliases"] = aliases
    return spec


def _reactants(spec: dict[str, Any]) -> tuple[bool, list[tuple[Any, ...]]]:
    (reaction,) = compile_spec(spec).reactions
    return reaction.reactants_complete, [
        (r.state_base, r.state_axes, r.pinned, r.order) for r in reaction.reactants
    ]


SOURCE = ("S", ("age", "vax"), (), 1)


@pytest.mark.parametrize(
    ("rate", "complete"),
    [
        pytest.param("gamma", True, id="parameter"),
        pytest.param("k[age] * 2.0", True, id="shaped-parameter"),
        pytest.param("beta * I[age,vax]", False, id="templated-state"),
        pytest.param("beta * Z", False, id="axis-less-state"),
        pytest.param("lam", False, id="state-behind-alias"),
        pytest.param("beta * sum_over(J[age:a], age=a)", False, id="reduction"),
    ],
)
def test_omitted_reactants_are_complete_only_when_the_rate_reads_no_state(
    rate: str, *, complete: bool
) -> None:
    """The synthesized source is complete exactly when no catalyst can exist."""
    assert _reactants(_spec(rate, aliases={"lam": "beta * Z"})) == (
        complete,
        [SOURCE],
    )


def test_constant_source_only_rate_is_complete_without_reactants() -> None:
    """A constant hazard into a cell has no reactants and nothing missing."""
    assert _reactants(_spec("eta", frm=None, to="S[age,vax]")) == (True, [])


@pytest.mark.parametrize(
    ("rate", "inferred"),
    [
        pytest.param("gamma", [], id="no-state"),
        pytest.param(
            "beta * I[age,vax]", [("I", ("age", "vax"), (), 1)], id="same-axes"
        ),
        pytest.param("beta * J[age]", [("J", ("age",), (), 1)], id="fewer-axes"),
        pytest.param(
            "beta * I[age,vax:u]", [("I", ("age",), (("vax", 0),), 1)], id="pinned"
        ),
        pytest.param("beta * Z", [("Z", (), (), 1)], id="axis-less"),
        pytest.param("lam", [("Z", (), (), 1)], id="through-alias"),
        pytest.param(
            "beta * I[age,vax] * J[age] / c",
            [("I", ("age", "vax"), (), 1), ("J", ("age",), (), 1)],
            id="product-over-parameter",
        ),
        pytest.param("k * Z**2", [("Z", (), (), 2)], id="integer-power"),
    ],
)
def test_auto_infers_the_source_and_each_state_factor(
    rate: str, inferred: list[tuple[Any, ...]]
) -> None:
    """Each state factor is a reactant at its power, after the source."""
    spec = _spec(rate, reactants="auto", aliases={"lam": "beta * Z"})
    assert _reactants(spec) == (True, [SOURCE, *inferred])


def test_auto_adds_a_rate_that_reads_the_source_to_its_order() -> None:
    """``k * S`` consumes one ``S`` and reads another: order two."""
    assert _reactants(_spec("k * S[age,vax]", reactants="auto")) == (
        True,
        [("S", ("age", "vax"), (), 2)],
    )


def test_auto_matches_the_equivalent_explicit_declaration() -> None:
    """Inference publishes exactly what the explicit list declares."""
    rate = "beta * I[age,vax:u] * J[age]"
    explicit = [
        {"state": "S[age,vax]", "order": 1},
        {"state": "I[age,vax=u]", "order": 1},
        {"state": "J[age]", "order": 1},
    ]
    assert _reactants(_spec(rate, reactants="auto")) == _reactants(
        _spec(rate, reactants=explicit)
    )


def test_auto_on_a_source_only_transition_infers_from_the_rate() -> None:
    """Births proportional to a state read that state only."""
    spec = _spec("b * J[age]", reactants="auto", frm=None, to="S[age,vax]")
    assert _reactants(spec) == (True, [("J", ("age",), (), 1)])


@pytest.mark.parametrize(
    ("rate", "reason"),
    [
        pytest.param("beta * (I[age,vax] + R[age,vax])", "with '+'", id="sum"),
        pytest.param(
            "beta * I[age,vax] / (S[age,vax] + I[age,vax])",
            "divides by a state",
            id="frequency-dependent",
        ),
        pytest.param(
            "beta * sum_over(J[age:a], age=a)", "reduces states", id="reduction"
        ),
        pytest.param("exp(-Z)", "with 'exp'", id="function"),
        pytest.param("Z**0.5", "positive integer literal", id="fractional-power"),
        pytest.param("beta * I", "without its axes", id="bare-templated-state"),
    ],
)
def test_auto_rejects_rates_that_are_not_a_product_of_states(
    rate: str, reason: str
) -> None:
    """Inference refuses instead of guessing; see issue #256."""
    with pytest.raises(
        InvalidRhsSpecError, match=f"reactants: auto.*{re.escape(reason)}"
    ):
        compile_spec(_spec(rate, reactants="auto"))


def test_auto_rejects_a_state_read_along_a_routed_axis() -> None:
    """A fan-out target axis is not a channel axis of the reactant metadata."""
    spec: dict[str, Any] = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["a0", "a1"]},
            {"name": "imm", "coords": ["x0", "x1"]},
        ],
        "state": ["I[age]", "X[age,imm]", "Y[imm]"],
        "transitions": [
            {
                "name": "fan",
                "from": "I[age]",
                "to": "X[age,imm:j]",
                "rate": "w[imm:j] * Y[imm:j]",
                "reactants": "auto",
            }
        ],
    }
    with pytest.raises(InvalidRhsSpecError, match="outside the reaction channels"):
        compile_spec(spec)


def test_reactants_must_be_a_list_or_auto() -> None:
    """Any other string is rejected at validation."""
    with pytest.raises(InvalidRhsSpecError, match="must be a list or 'auto'"):
        compile_spec(_spec("gamma", reactants="all"))


def test_chain_catalysts_auto_infers_each_stage() -> None:
    """``catalysts: auto`` infers the entry's catalysts from its rate."""
    spec: dict[str, Any] = {
        "kind": "transitions",
        "state": ["S", "R"],
        "chain": [
            {
                "name": "I",
                "length": 2,
                "forward": ["k"],
                "entry": {"from": "S", "rate": "beta * I1", "catalysts": "auto"},
                "exit": {"to": "R", "rate": "gamma"},
                "catalysts": "auto",
            }
        ],
    }
    compiled = compile_spec(spec)
    assert {
        r.name: (r.reactants_complete, [(x.state_base, x.order) for x in r.reactants])
        for r in compiled.reactions
    } == {
        "I_entry": (True, [("S", 1), ("I1", 1)]),
        "I_advance_1": (True, [("I1", 1)]),
        "I_exit": (True, [("I2", 1)]),
    }


def test_coord_shift_catalysts_auto_infers_from_the_rate() -> None:
    """``catalysts: auto`` on a coord_shift entry reads its rate."""
    spec: dict[str, Any] = {
        "kind": "transitions",
        "axes": [{"name": "age", "coords": ["a0", "a1", "a2"]}],
        "state": ["S[age]", "Z"],
        "transitions": [
            {
                "name": "aging",
                "coord_shift": {"axis": "age", "boundary": "stay"},
                "rate": "g * Z",
                "apply_to": ["S"],
                "catalysts": "auto",
            }
        ],
    }
    (reaction,) = compile_spec(spec).reactions
    assert reaction.reactants_complete is True
    assert [(r.state_base, r.order) for r in reaction.reactants] == [
        ("S", 1),
        ("Z", 1),
    ]
