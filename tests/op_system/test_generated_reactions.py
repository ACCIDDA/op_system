"""Reactions for chain stages and coord_shift entries (issue #247)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_system import CompiledReaction, compile_spec
from op_system._errors import InvalidRhsSpecError

if TYPE_CHECKING:
    from op_system.compile import StateDict

AGE = {"name": "age", "type": "ordinal", "coords": ["a0", "a1", "a2"]}


def _chain_spec(*, templated: bool, catalysts: bool) -> dict[str, Any]:
    """Build an SIR model with a three-stage infectious period.

    Returns:
        A transitions spec using the ``chain`` helper.
    """
    sel = "[age]" if templated else ""
    stages = [f"I{k}{sel}" for k in (1, 2, 3)]
    entry: dict[str, Any] = {
        "from": f"S{sel}",
        "rate": f"beta * ({' + '.join(stages)})",
    }
    chain: dict[str, Any] = {
        "name": f"I{sel}",
        "length": 3,
        "entry": entry,
        "forward": ["gamma", "gamma"],
        "exit": {"to": f"R{sel}", "rate": "gamma"},
    }
    if catalysts:
        entry["catalysts"] = [{"state": stage, "order": 1} for stage in stages]
        chain["catalysts"] = []
    return {
        "kind": "transitions",
        "axes": [AGE] if templated else [],
        "state": [f"S{sel}", f"R{sel}"],
        "chain": [chain],
        "transitions": [],
    }


def _drift(
    reactions: tuple[CompiledReaction, ...], state: StateDict, **params: object
) -> dict[str, np.ndarray]:
    """Scatter every propensity through the published reaction contract.

    Returns:
        Expected drift per state template.
    """
    drift = {base: np.zeros(np.shape(value)) for base, value in state.items()}
    for reaction in reactions:
        assert reaction.from_base is not None
        assert reaction.to_full_axes is not None
        propensity = np.asarray(reaction.propensity_fn(0.0, state, **params))
        for index in np.ndindex(propensity.shape):
            channel = dict(zip(reaction.from_axes, index, strict=True))
            source = {**channel, **dict(reaction.from_pinned)}
            drift[reaction.from_base][tuple(source[a] for a in reaction.full_axes)] -= (
                propensity[index]
            )
            target = {a: channel[a] for a in reaction.to_axes} | dict(reaction.pinned)
            target |= {a: channel[a] + s for a, s in reaction.offsets}
            if all(
                0 <= target[a] < dim
                for a, dim in zip(
                    reaction.to_full_axes,
                    np.shape(state[reaction.to_base]),
                    strict=True,
                )
            ):
                drift[reaction.to_base][
                    tuple(target[a] for a in reaction.to_full_axes)
                ] += propensity[index]
    return drift


@pytest.mark.parametrize("templated", [False, True])
def test_chain_stages_publish_named_complete_reactions(*, templated: bool) -> None:
    """Every chain transition is named and, with catalysts, complete."""
    compiled = compile_spec(_chain_spec(templated=templated, catalysts=True))
    assert compiled.reaction_gaps == ()
    by_name = {r.name: r for r in compiled.reactions}
    assert list(by_name) == ["I_entry", "I_advance_1", "I_advance_2", "I_exit"]
    assert [(r.from_base, r.to_base) for r in by_name.values()] == [
        ("S", "I1"),
        ("I1", "I2"),
        ("I2", "I3"),
        ("I3", "R"),
    ]
    assert all(r.reactants_complete for r in compiled.reactions)
    assert [(x.state_base, x.order) for x in by_name["I_entry"].reactants] == [
        ("S", 1),
        ("I1", 1),
        ("I2", 1),
        ("I3", 1),
    ]
    assert [(x.state_base, x.order) for x in by_name["I_advance_2"].reactants] == [
        ("I2", 1)
    ]

    rng = np.random.default_rng(247)
    assert compiled.template_shapes is not None
    state = {
        base: rng.uniform(1.0, 20.0, size=shape)
        for base, shape in compiled.template_shapes.items()
    }
    params = {"beta": 0.01, "gamma": 0.3}
    assert compiled.pytree_eval_fn is not None
    rhs = compiled.pytree_eval_fn(0.0, state, **params)
    drift = _drift(compiled.reactions, state, **params)
    for base, value in rhs.items():
        np.testing.assert_allclose(drift[base], np.asarray(value), rtol=1e-12)


def test_chain_without_catalysts_keeps_the_fallback() -> None:
    """Without declarations the chain is named but not claimed complete."""
    compiled = compile_spec(_chain_spec(templated=False, catalysts=False))
    assert [r.name for r in compiled.reactions] == [
        "I_entry",
        "I_advance_1",
        "I_advance_2",
        "I_exit",
    ]
    assert not any(r.reactants_complete for r in compiled.reactions)


@pytest.mark.parametrize("axis_wide", [False, True])
def test_coord_shift_entries_publish_named_complete_reactions(
    *, axis_wide: bool
) -> None:
    """Named shifts publish ``{name}_{state}``; catalysts make them complete."""
    shift: dict[str, Any] = (
        {"coord_shift": {"axis": "age", "boundary": "stay"}}
        if axis_wide
        else {"coord_shift": {"age": "a0 -> a1"}}
    )
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [AGE],
        "state": ["S[age]", "I[age]"],
        "transitions": [
            {
                "name": "aging",
                **shift,
                "rate": "g",
                "apply_to": ["S", "I"],
                "catalysts": [],
            }
        ],
    }
    compiled = compile_spec(spec)
    assert compiled.reaction_gaps == ()
    assert [r.name for r in compiled.reactions] == ["aging_S", "aging_I"]
    assert all(r.reactants_complete for r in compiled.reactions)
    for reaction in compiled.reactions:
        (source,) = reaction.reactants
        assert source.state_base == reaction.from_base
        assert source.order == 1
    state = {"S": np.array([1.0, 2.0, 3.0]), "I": np.array([4.0, 5.0, 6.0])}
    assert compiled.pytree_eval_fn is not None
    rhs = compiled.pytree_eval_fn(0.0, state, g=0.5)
    drift = _drift(compiled.reactions, state, g=0.5)
    for base, value in rhs.items():
        np.testing.assert_allclose(drift[base], np.asarray(value), rtol=1e-12)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        (
            {"reactants": []},
            "declare reactants beyond the shifted state as 'catalysts'",
        ),
        ({"catalysts": {"state": "I[age]"}}, "catalysts must be a list"),
        (
            {"catalysts": [{"state": "S[age]", "order": 1}]},
            "repeats reactant state|outside reaction channels",
        ),
        ({"catalysts": [{"state": "Q", "order": 1}]}, "unknown state base"),
    ],
)
@pytest.mark.parametrize("axis_wide", [False, True])
def test_invalid_catalysts_raise(
    changes: dict[str, Any], match: str, *, axis_wide: bool
) -> None:
    """Catalyst declarations are validated like explicit reactants."""
    shift = {"axis": "age", "boundary": "absorb"} if axis_wide else {"age": "a0 -> a1"}
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [AGE],
        "state": ["S[age]", "I[age]"],
        "transitions": [
            {
                "name": "aging",
                "coord_shift": shift,
                "rate": "g",
                "apply_to": ["S"],
                **changes,
            }
        ],
    }
    with pytest.raises(InvalidRhsSpecError, match=match):
        compile_spec(spec)
