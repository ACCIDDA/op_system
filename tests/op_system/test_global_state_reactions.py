"""Global population totals preserve reaction export and chain state semantics."""

import pickle  # ruff: ignore[suspicious-pickle-import]

import numpy as np
import pytest

from op_system import compile_spec


def test_total_rate_does_not_disable_unrelated_reaction_export() -> None:
    """An infection rate's total leaves recovery and all RHS layouts available."""
    compiled = compile_spec({
        "kind": "transitions",
        "state": ["S", "I", "R"],
        "transitions": [
            {"name": "inf", "from": "S", "to": "I", "rate": "beta * I / sum_state()"},
            {"name": "rec", "from": "I", "to": "R", "rate": "gamma"},
        ],
    })
    assert compiled.reaction_gaps == ()
    assert len(compiled.reactions) == 2
    assert compiled.pytree_eval_fn is not None
    state = {
        name: np.asarray(value)
        for name, value in {"S": 9.0, "I": 1.0, "R": 2.0}.items()
    }
    flow, recovery = 0.4 * 9 / 12, 0.1
    np.testing.assert_allclose(
        compiled.eval_fn(0.0, np.array([9.0, 1.0, 2.0]), beta=0.4, gamma=0.1),
        [-flow, flow - recovery, recovery],
    )
    assert [
        np.asarray(reaction.propensity_fn(0.0, state, beta=0.4, gamma=0.1)).item()
        for reaction in compiled.reactions
    ] == pytest.approx([flow, recovery])
    restored = pickle.loads(pickle.dumps(compiled))  # ruff: ignore[suspicious-pickle-usage]
    assert np.asarray(
        restored.reactions[0].propensity_fn(0.0, state, beta=0.4, gamma=0.1)
    ).item() == pytest.approx(flow)
    with pytest.raises(ValueError, match=r"sum_state requires.*R"):
        compiled.reactions[0].propensity_fn(
            0.0, {"S": state["S"], "I": state["I"]}, beta=0.4, gamma=0.1
        )


@pytest.mark.parametrize("base_state", [False, True])
def test_chain_total_includes_only_actual_declared_and_generated_compartments(
    *, base_state: bool
) -> None:
    """The chain name is a prefix; declaring its base adds a separate compartment."""
    states = ["S", "R", "I"] if base_state else ["S", "R"]
    compiled = compile_spec({
        "kind": "transitions",
        "state": states,
        "aliases": {"infectious": "I1 + I2 + I3"},
        "chain": [
            {
                "name": "I",
                "length": 3,
                "entry": {"from": "S", "rate": "beta * infectious / sum_state()"},
                "forward": ["g12", "g23"],
                "exit": {"to": "R", "rate": "g3r"},
            }
        ],
        "transitions": [],
    })
    assert compiled.reaction_gaps == ()
    assert len(compiled.reactions) == 4
    assert ("I" in compiled.state_names) == base_state
    state = {
        name: np.asarray(value)
        for name, value in {"S": 9.0, "R": 2.0, "I1": 1.0, "I2": 2.0, "I3": 3.0}.items()
    }
    if base_state:
        state["I"] = np.asarray(7.0)
    flow = 0.4 * 6 * 9 / (17 + (7 if base_state else 0))
    assert np.asarray(
        compiled.reactions[0].propensity_fn(
            0.0, state, beta=0.4, g12=0.1, g23=0.2, g3r=0.3
        )
    ).item() == pytest.approx(flow)
