"""Axis-less states on the vectorized path (issues #245 and #246)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_system import CompiledReaction, compile_spec

if TYPE_CHECKING:
    from op_system.compile import StateDict

IMM = {"name": "imm", "coords": ["x0", "x1", "x2"]}
AGE = {"name": "age", "coords": ["c", "a"]}
W = np.array([0.2, 0.3, 0.5])


def _transitions(
    state: list[str], transitions: list[dict[str, Any]], *axes: dict[str, Any]
) -> dict[str, Any]:
    return {
        "kind": "transitions",
        "axes": list(axes),
        "state": state,
        "transitions": transitions,
    }


def _state_axes(spec: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Map each declared state base to its template axes.

    Returns:
        Axes per base, ``()`` for axis-less states.
    """
    out: dict[str, tuple[str, ...]] = {}
    for selector in spec["state"]:
        base, _, rest = selector.partition("[")
        out[base] = tuple(
            ax.strip() for ax in rest.rstrip("]").split(",") if ax.strip()
        )
    return out


def _reaction_drift(
    reactions: tuple[CompiledReaction, ...], state: StateDict, **params: object
) -> dict[str, np.ndarray]:
    """Scatter every propensity through the published reaction contract.

    Returns:
        Expected drift per state template.
    """
    drift = {base: np.zeros(np.shape(value)) for base, value in state.items()}
    for reaction in reactions:
        propensity = np.asarray(reaction.propensity_fn(0.0, state, **params))
        for index in np.ndindex(propensity.shape):
            channel = dict(zip(reaction.from_axes, index, strict=True))
            if reaction.from_base is not None:
                source = {**channel, **dict(reaction.from_pinned)}
                cell = tuple(source[ax] for ax in reaction.full_axes)
                drift[reaction.from_base][cell] -= propensity[index]
            target = {ax: channel[ax] for ax in reaction.to_axes}
            target.update(dict(reaction.pinned))
            to_full_axes = reaction.to_full_axes
            assert to_full_axes is not None
            cell = tuple(target[ax] for ax in to_full_axes)
            drift[reaction.to_base][cell] += propensity[index]
    return drift


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_fan_out_from_axis_less_source_evaluates(backend: str) -> None:
    """#245: each target gains ``r w_j I``; the source loses ``r sum(w) I``."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    compiled = compile_spec(
        _transitions(
            ["I", "X[imm]"],
            [{"from": "I", "to": "X[imm:j]", "rate": "r * w[imm:j]"}],
            IMM,
        )
    )
    expected = np.array([-0.3, *(0.3 * W)])
    flat = compiled.eval_fn(
        0.0, xp.asarray([1.0, 2.0, 3.0, 4.0]), r=0.3, w=xp.asarray(W)
    )
    np.testing.assert_allclose(np.asarray(flat), expected, rtol=1e-6)
    assert compiled.pytree_eval_fn is not None
    tree = compiled.pytree_eval_fn(
        0.0,
        {"I": xp.asarray(1.0), "X": xp.asarray([2.0, 3.0, 4.0])},
        r=0.3,
        w=xp.asarray(W),
    )
    np.testing.assert_allclose(np.asarray(tree["I"]), -0.3, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(tree["X"]), 0.3 * W, rtol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_axis_less_sir_publishes_scalar_reactions(backend: str) -> None:
    """#246: the textbook CTMC gets two 0-d reactions matching its RHS."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    compiled = compile_spec(
        _transitions(
            ["S", "I", "R"],
            [
                {
                    "name": "infect",
                    "from": "S",
                    "to": "I",
                    "rate": "beta * I / (S + I + R)",
                },
                {"name": "recover", "from": "I", "to": "R", "rate": "gamma"},
            ],
        )
    )
    assert compiled.template_shapes == {"S": (), "I": (), "R": ()}
    assert compiled.reaction_gaps == ()
    infect, recover = compiled.reactions
    for reaction in (infect, recover):
        assert reaction.from_axes == reaction.full_axes == reaction.to_axes == ()
        assert reaction.pinned == reaction.from_pinned == reaction.offsets == ()
    state = {"S": xp.asarray(90.0), "I": xp.asarray(10.0), "R": xp.asarray(0.0)}
    params = {"beta": 0.5, "gamma": 0.2}
    np.testing.assert_allclose(
        np.asarray(infect.propensity_fn(0.0, state, **params)), 4.5
    )
    np.testing.assert_allclose(
        np.asarray(recover.propensity_fn(0.0, state, **params)), 2.0
    )
    assert compiled.pytree_eval_fn is not None
    rhs = compiled.pytree_eval_fn(0.0, state, **params)
    numpy_state = {base: np.asarray(value) for base, value in state.items()}
    drift = _reaction_drift(compiled.reactions, numpy_state, **params)
    for base in ("S", "I", "R"):
        np.testing.assert_allclose(drift[base], np.asarray(rhs[base]), rtol=1e-6)


@pytest.mark.parametrize(
    ("spec", "state", "params"),
    [
        pytest.param(
            _transitions(
                ["I", "X[imm]"],
                [{"name": "seed", "from": "I", "to": "X[imm=x1]", "rate": "r"}],
                IMM,
            ),
            {"I": np.asarray(1.0), "X": np.array([2.0, 3.0, 4.0])},
            {"r": 0.3},
            id="scalar-to-pinned-cell",
        ),
        pytest.param(
            _transitions(
                ["X[imm]", "R"],
                [
                    {
                        "name": "collapse",
                        "from": "X[imm]",
                        "to": "R",
                        "rate": "r * w[imm]",
                    }
                ],
                IMM,
            ),
            {"X": np.array([2.0, 3.0, 4.0]), "R": np.asarray(1.0)},
            {"r": 0.3, "w": W},
            id="templated-to-scalar",
        ),
        pytest.param(
            _transitions(
                ["S[age]", "I[age]", "V"],
                [
                    {
                        "name": "infect",
                        "from": "S[age]",
                        "to": "I[age]",
                        "rate": "beta * V",
                    },
                ],
                AGE,
            ),
            {
                "S": np.array([10.0, 20.0]),
                "I": np.array([1.0, 2.0]),
                "V": np.asarray(3.0),
            },
            {"beta": 0.1},
            id="scalar-catalyst",
        ),
    ],
)
def test_mixed_scalar_and_templated_reactions_reconstruct_rhs(
    spec: dict[str, Any], state: StateDict, params: dict[str, Any]
) -> None:
    """#246: reactions touching scalar states compile and match the RHS.

    The scalar-to-pinned case also guards the vectorized RHS: before the
    donor used coordinate masks, the template's first and last cells were
    zero and the deposit into the middle cell was lost.
    """
    compiled = compile_spec(spec)
    assert compiled.reaction_gaps == ()
    declared = _state_axes(spec)
    for reaction in compiled.reactions:
        assert reaction.from_base is not None
        assert reaction.full_axes == declared[reaction.from_base]
        assert reaction.to_full_axes == declared[reaction.to_base]
    assert compiled.pytree_eval_fn is not None
    assert compiled.template_shapes is not None
    rhs = compiled.pytree_eval_fn(0.0, state, **params)
    drift = _reaction_drift(compiled.reactions, state, **params)
    for base, value in rhs.items():
        np.testing.assert_allclose(drift[base], np.asarray(value), atol=1e-12)
    order = list(compiled.template_shapes)
    flat = np.concatenate([np.ravel(state[b]) for b in order])
    np.testing.assert_allclose(
        np.asarray(compiled.eval_fn(0.0, flat, **params), dtype=float),
        np.concatenate([np.ravel(np.asarray(rhs[b], dtype=float)) for b in order]),
    )


def test_scalar_to_pinned_cell_deposits_into_the_pinned_coordinate() -> None:
    """The vectorized RHS deposits into the middle coordinate, not nowhere."""
    compiled = compile_spec(
        _transitions(
            ["I", "X[imm]"],
            [{"from": "I", "to": "X[imm=x1]", "rate": "r"}],
            IMM,
        )
    )
    np.testing.assert_allclose(
        compiled.eval_fn(0.0, np.array([1.0, 2.0, 3.0, 4.0]), r=0.3),
        [-0.3, 0.0, 0.3, 0.0],
    )


def test_axis_less_history_spec_evaluates() -> None:
    """Axis-less history specs gain a history evaluator through the 0-d plan."""
    queries: list[tuple[int, dict[str, object]]] = []

    class ConstantHistory:
        @staticmethod
        def query(signal_id: int, body: object, **options: object) -> object:
            queries.append((signal_id, options))
            return np.full_like(np.asarray(body, dtype=float), 7.0)

    compiled = compile_spec({
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {
            "x": "convolve_history(x, kernel=gamma, window=3) - y",
            "y": "x",
        },
    })
    assert compiled.history_eval_fn is not None
    out = compiled.history_eval_fn(
        0.0,
        {"x": np.asarray(2.0), "y": np.asarray(1.0)},
        history_provider=ConstantHistory(),
    )
    np.testing.assert_allclose([out["x"], out["y"]], [6.0, 2.0])
    assert len(queries) == 1
