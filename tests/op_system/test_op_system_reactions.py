"""Unit tests for op_system._reactions / CompiledRhs.reactions (pytest).

These tests cover:
- which named transitions get a reaction artifact, and which are (silently)
  excluded as out of scope for v1
- correctness of the compiled propensity: rate * from_state, not the bare
  rate, cross-checked against the existing deterministic eval_fn as the
  correctness oracle
- axis bookkeeping (from_axes/to_axes/sum_axes/pinned) for both the
  same-axes case and the collapse-to-a-pinned-target-coordinate case
- factorize_axes + a shaped-param (axis-indexed) rate, matching real usage
- a rate referencing a mixing kernel via apply_along (regression, see
  test_reduce_bearing_kernel_rate_propensity below)
"""

from __future__ import annotations

import numpy as np

from op_system import compile_spec
from op_system.specs import normalize_transitions_rhs


def _sir_like_spec(*, named: bool = True) -> dict[str, object]:
    name_kw: dict[str, object] = {"name": "expose"} if named else {}
    return {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["a0", "a1"]},
            {"name": "vax", "coords": ["u", "p", "f"]},
        ],
        "state": ["S[age,vax]", "E[age,vax]", "C[age,vax]"],
        "transitions": [
            {**name_kw, "from": "S[age,vax]", "to": "E[age,vax]", "rate": "foi"},
            {
                "name": "recover",
                "from": "C[age,vax]",
                "to": "S[age,vax=f]",
                "rate": "gamma_c",
            },
        ],
    }


def test_unnamed_transition_excluded_from_reactions_ir() -> None:
    """Unnamed transitions are not addressable and get no artifact."""
    rhs = normalize_transitions_rhs(_sir_like_spec(named=False))
    names = [r.name for r in rhs.reactions_ir]
    assert names == ["recover"]


def test_source_only_transition_excluded_from_reactions_ir() -> None:
    """`from: null` transitions have no well-defined firing-cell count."""
    spec = _sir_like_spec()
    transitions = spec["transitions"]
    assert isinstance(transitions, list)
    transitions.append({"name": "seed", "to": "S[age,vax]", "rate": "lambda_seed"})
    rhs = normalize_transitions_rhs(spec)
    names = {r.name for r in rhs.reactions_ir}
    assert "seed" not in names
    assert names == {"expose", "recover"}


def test_to_side_extra_axis_excluded_from_reactions_ir() -> None:
    """A to-side wildcard axis absent from from-side is out of scope."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["a0", "a1"]},
            {"name": "loc", "coords": ["d1", "d2"]},
        ],
        "state": ["Import[age]", "S[age,loc]"],
        "transitions": [
            {
                "name": "distribute",
                "from": "Import[age]",
                "to": "S[age,loc]",
                "rate": "k",
            },
        ],
    }
    rhs = normalize_transitions_rhs(spec)
    assert rhs.reactions_ir == ()


def test_same_axes_reaction_metadata_and_propensity() -> None:
    """A same-shape transition has empty sum_axes/pinned and rate*from_state."""
    c = compile_spec(_sir_like_spec())
    expose = next(r for r in c.reactions if r.name == "expose")
    assert expose.from_base == "S"
    assert expose.from_axes == ("age", "vax")
    assert expose.to_base == "E"
    assert expose.to_axes == ("age", "vax")
    assert expose.sum_axes == ()
    assert expose.pinned == ()

    y = {
        "S": np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]]),
        "E": np.zeros((2, 3)),
        "C": np.zeros((2, 3)),
    }
    got = np.asarray(
        expose.propensity_fn(0.0, y, foi=np.asarray(0.1), gamma_c=np.asarray(0.05))
    )
    np.testing.assert_allclose(got, 0.1 * y["S"])


def test_collapse_to_pinned_target_metadata_and_propensity() -> None:
    """A transition collapsing to a fixed to-side coordinate sums that axis."""
    c = compile_spec(_sir_like_spec())
    recover = next(r for r in c.reactions if r.name == "recover")
    assert recover.from_base == "C"
    assert recover.from_axes == ("age", "vax")
    assert recover.to_base == "S"
    assert recover.to_axes == ("age",)
    assert recover.sum_axes == ("vax",)
    assert recover.pinned == (("vax", 2),)  # "f" is coord index 2

    y = {
        "S": np.zeros((2, 3)),
        "E": np.zeros((2, 3)),
        "C": np.array([[100.0, 200.0, 300.0], [10.0, 20.0, 30.0]]),
    }
    got = np.asarray(
        recover.propensity_fn(0.0, y, foi=np.asarray(0.1), gamma_c=np.asarray(0.05))
    )
    np.testing.assert_allclose(got, 0.05 * y["C"])


def test_reactions_reconstruct_deterministic_eval_fn() -> None:
    """Summing reaction propensities (with axis bookkeeping) matches eval_fn.

    This is the correctness oracle: the deterministic path and this
    independent artifact are built by completely separate code, so
    agreement here is a strong signal neither has a sign/shape bug.
    """
    c = compile_spec(_sir_like_spec())
    assert c.pytree_eval_fn is not None
    y = {
        "S": np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]]),
        "E": np.zeros((2, 3)),
        "C": np.array([[100.0, 200.0, 300.0], [10.0, 20.0, 30.0]]),
    }
    params = {"foi": np.asarray(0.1), "gamma_c": np.asarray(0.05)}
    dy = c.pytree_eval_fn(np.asarray(0.0), y, **params)

    expose = next(r for r in c.reactions if r.name == "expose")
    recover = next(r for r in c.reactions if r.name == "recover")
    p_expose = np.asarray(expose.propensity_fn(0.0, y, **params))
    p_recover = np.asarray(recover.propensity_fn(0.0, y, **params))

    expected_s = -p_expose.copy()
    expected_s[:, 2] += p_recover.sum(axis=1)  # sum_axes=("vax",), pinned f=2
    expected_e = p_expose.copy()
    expected_c = -p_recover.copy()

    np.testing.assert_allclose(dy["S"], expected_s)
    np.testing.assert_allclose(dy["E"], expected_e)
    np.testing.assert_allclose(dy["C"], expected_c)


def test_reactions_with_factorize_axes_and_shaped_param_rate() -> None:
    """factorize_axes + a vax-indexed shaped-param rate (real-usage shape)."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "factorize_axes": ["loc"],
        "axes": [
            {"name": "age", "coords": ["a0", "a1"]},
            {"name": "vax", "coords": ["u", "p", "f"]},
            {"name": "loc", "coords": ["d1", "d2"]},
        ],
        "state": ["E[age,vax,loc]", "C1[age,vax,loc]", "S[age,vax,loc]"],
        "transitions": [
            {
                "name": "to_carrier",
                "from": "E[age,vax,loc]",
                "to": "C1[age,vax,loc]",
                "rate": "delta[vax] * tau",
            },
            {
                "name": "recover",
                "from": "C1[age,vax,loc]",
                "to": "S[age,vax=f,loc]",
                "rate": "3.0 * gamma_c",
            },
        ],
    }
    c = compile_spec(spec)
    assert c.pytree_eval_fn is not None
    assert {r.name for r in c.reactions} == {"to_carrier", "recover"}

    y = {
        "E": np.arange(12, dtype=float).reshape(2, 3, 2) + 1,
        "C1": np.arange(12, dtype=float).reshape(2, 3, 2) + 10,
        "S": np.zeros((2, 3, 2)),
    }
    params = {
        "delta": np.array([0.31, 0.48, 0.88]),
        "tau": np.asarray(0.5263),
        "gamma_c": np.asarray(0.05405),
    }

    to_carrier = next(r for r in c.reactions if r.name == "to_carrier")
    recover = next(r for r in c.reactions if r.name == "recover")
    p_carrier = np.asarray(to_carrier.propensity_fn(0.0, y, **params))
    p_recover = np.asarray(recover.propensity_fn(0.0, y, **params))

    expected_carrier = y["E"] * (params["delta"] * params["tau"])[None, :, None]
    expected_recover = y["C1"] * (3.0 * params["gamma_c"])
    np.testing.assert_allclose(p_carrier, expected_carrier)
    np.testing.assert_allclose(p_recover, expected_recover)

    dy = c.pytree_eval_fn(np.asarray(0.0), y, **params)
    hand_s = np.zeros((2, 3, 2))
    hand_s[:, 2, :] = p_recover.sum(axis=1)
    np.testing.assert_allclose(dy["S"], hand_s)
    np.testing.assert_allclose(dy["E"], -p_carrier)


def _vax_progression_spec() -> dict[str, object]:
    return {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["a0", "a1"]},
            {"name": "vax", "coords": ["u", "p", "f"]},
        ],
        "state": ["S[age,vax]"],
        "transitions": [
            {
                "name": "dose1",
                "from": "S[age,vax=u]",
                "to": "S[age,vax=p]",
                "rate": "v_p",
            },
        ],
    }


def test_from_pinned_transition_metadata_and_propensity() -> None:
    """A from-side-pinned (point-to-point) transition compiles correctly.

    Regression test: ``S[age,vax=u] -> S[age,vax=p]`` pins a coordinate on
    the *from*-side selector, which previously made
    ``_in_order_wildcard_axes`` silently drop that axis from the
    propensity's ``Subscript`` reference, producing a shape-mismatched
    reference that failed to lower and dropped the reaction from
    ``c.reactions`` entirely (no error, just silently absent).
    """
    c = compile_spec(_vax_progression_spec())
    dose1 = next(r for r in c.reactions if r.name == "dose1")
    assert dose1.from_base == "S"
    assert dose1.from_axes == ("age",)
    assert dose1.full_axes == ("age", "vax")
    assert dose1.to_base == "S"
    assert dose1.to_axes == ("age",)
    assert dose1.sum_axes == ()
    assert dose1.pinned == (("vax", 1),)  # "p" is coord index 1
    assert dose1.from_pinned == (("vax", 0),)  # "u" is coord index 0

    y = {"S": np.array([[100.0, 200.0, 300.0], [10.0, 20.0, 30.0]])}
    params = {"v_p": np.asarray(0.02)}
    got = np.asarray(dose1.propensity_fn(0.0, y, **params))
    # Only the vax="u" (coord 0) column feeds this propensity -- it must
    # NOT be the bare rate, and must NOT include the vax="p"/"f" columns.
    np.testing.assert_allclose(got, 0.02 * y["S"][:, 0])

    dy = c.pytree_eval_fn(np.asarray(0.0), y, **params)
    expected_s = np.zeros_like(y["S"])
    expected_s[:, 0] = -got
    expected_s[:, 1] = got
    np.testing.assert_allclose(dy["S"], expected_s)


def _foi_like_alias_spec() -> dict[str, object]:
    return {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["a0", "a1"]},
            {"name": "loc", "coords": ["d1", "d2"]},
        ],
        "aliases": {"foi[age, loc]": "r0 * S[age, loc] + 1.0"},
        "state": ["S[age,loc]", "E[age,loc]"],
        "transitions": [
            {
                "name": "expose",
                "from": "S[age,loc]",
                "to": "E[age,loc]",
                "rate": "foi[age, loc]",
            },
        ],
    }


def test_single_level_alias_reference_compiles_and_matches_deterministic() -> None:
    """A rate referencing a non-chained alias (issue #189) gets a reaction.

    Regression test: a rate like ``foi[age, loc]`` referencing a
    spec-level alias used to compile-fail silently (``foi`` isn't a
    registered buffer or shaped param at the vector-lowering stage) and
    be dropped from ``c.reactions`` with no error -- the real diphtheria
    config's ``expose`` transition hit exactly this.
    """
    c = compile_spec(_foi_like_alias_spec())
    expose = next((r for r in c.reactions if r.name == "expose"), None)
    assert expose is not None

    y = {"S": np.array([[10.0, 20.0], [30.0, 40.0]]), "E": np.zeros((2, 2))}
    params = {"r0": np.asarray(0.5)}
    got = np.asarray(expose.propensity_fn(0.0, y, **params))
    expected = (0.5 * y["S"] + 1.0) * y["S"]
    np.testing.assert_allclose(got, expected)

    dy = c.pytree_eval_fn(np.asarray(0.0), y, **params)
    np.testing.assert_allclose(dy["S"], -expected)
    np.testing.assert_allclose(dy["E"], expected)


def test_multi_level_alias_chain_stays_out_of_scope() -> None:
    """A rate referencing an alias that itself references another alias.

    Still excluded -- single-level inlining only (deferred, tracked
    separately; not a regression from before alias inlining existed).
    """
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [{"name": "age", "coords": ["a0", "a1"]}],
        "aliases": {
            "bar[age]": "2.0 * age",
            "foi[age]": "bar[age] + 1.0",
        },
        "state": ["S[age]", "E[age]"],
        "transitions": [
            {"name": "expose", "from": "S[age]", "to": "E[age]", "rate": "foi[age]"},
        ],
    }
    c = compile_spec(spec)
    assert c.reactions == ()


def test_expr_kind_has_no_reactions() -> None:
    """ExprRhs specs (no transitions grammar) compile with empty reactions."""
    spec: dict[str, object] = {
        "kind": "expr",
        "state": ["S", "I", "R"],
        "equations": {
            "S": "-(beta * S * I) / (S + I + R)",
            "I": "(beta * S * I) / (S + I + R) - gamma * I",
            "R": "gamma * I",
        },
    }
    c = compile_spec(spec)
    assert c.reactions == ()


def test_reduce_bearing_kernel_rate_propensity() -> None:
    """A rate referencing a mixing kernel via apply_along compiles correctly.

    Regression test: compiling propensity_ir_full (rather than
    propensity_ir_reduce) produced a malformed AxisIndex(axis='', ...)
    subscript for this case, because expand_reduce_pointwise's
    template-symbolic (empty lhs_assignment) expansion doesn't handle a
    Reduce whose bound axis coincides with the target's own free axis. The
    deterministic path never hits this because it uses the Reduce-preserving
    form for exactly this kind of expression; _build_reaction_artifacts now
    does too.
    """
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [{"name": "age", "coords": ["a", "b"]}],
        "state": ["S[age]", "I[age]"],
        "kernels": [
            {
                "name": "k",
                "axes": ["age"],
                "form": "gaussian",
                "params": {"scale": 1.0, "sigma": 0.5},
            },
        ],
        "transitions": [
            {
                "name": "infect",
                "from": "S[age]",
                "to": "I[age]",
                "rate": "apply_along(k[age, age:ap] * S[age:ap], age=ap)",
            },
        ],
    }
    c = compile_spec(spec)
    assert len(c.reactions) == 1
    reaction = c.reactions[0]

    y = {"S": np.array([2.0, 3.0]), "I": np.zeros(2)}
    k = np.array([[1.0, 0.5], [0.5, 1.0]])
    got = np.asarray(reaction.propensity_fn(0.0, y, k=k))

    # hand-computed: propensity = (k @ S) * S, elementwise
    expected = (k @ y["S"]) * y["S"]
    np.testing.assert_allclose(got, expected)

    assert c.pytree_eval_fn is not None
    dy = c.pytree_eval_fn(np.asarray(0.0), y, k=k)
    np.testing.assert_allclose(dy["S"], -got)
    np.testing.assert_allclose(dy["I"], got)
