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


def test_source_only_transition_included_in_reactions_ir() -> None:
    """`from: null` transitions DO get a reaction artifact (exogenous hazard).

    Unlike the depleting-transition exclusions below, a source-only
    transition's firing-cell count is well-defined: one independent
    Poisson process per destination cell, no source population involved.
    """
    spec = _sir_like_spec()
    transitions = spec["transitions"]
    assert isinstance(transitions, list)
    transitions.append({"name": "seed", "to": "S[age,vax]", "rate": "lambda_seed"})
    rhs = normalize_transitions_rhs(spec)
    names = {r.name for r in rhs.reactions_ir}
    assert names == {"expose", "recover", "seed"}

    seed = next(r for r in rhs.reactions_ir if r.name == "seed")
    assert seed.from_base is None
    assert seed.from_axes == ("age", "vax")  # taken from the to-side template
    assert seed.full_axes == ("age", "vax")
    assert seed.to_base == "S"
    assert seed.to_axes == ("age", "vax")
    assert seed.pinned == ()
    assert seed.from_pinned == ()


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

    assert c.pytree_eval_fn is not None
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

    assert c.pytree_eval_fn is not None
    dy = c.pytree_eval_fn(np.asarray(0.0), y, **params)
    np.testing.assert_allclose(dy["S"], -expected)
    np.testing.assert_allclose(dy["E"], expected)


def test_multi_level_alias_chain_is_inlined() -> None:
    """A rate referencing an alias that itself references another alias.

    The whole chain resolves (issue #193); this asserted exclusion until
    single-level inlining was generalised.
    """
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [{"name": "age", "coords": ["a0", "a1"]}],
        "aliases": {
            "bar[age]": "2.0 * k[age]",
            "foi[age]": "bar[age] + 1.0",
        },
        "state": ["S[age]", "E[age]"],
        "transitions": [
            {"name": "expose", "from": "S[age]", "to": "E[age]", "rate": "foi[age]"},
        ],
    }
    c = compile_spec(spec)
    expose = next(r for r in c.reactions if r.name == "expose")
    y = {"S": np.array([10.0, 20.0]), "E": np.zeros(2)}
    k = np.array([3.0, 5.0])
    got = np.asarray(expose.propensity_fn(0.0, y, k=k))
    np.testing.assert_allclose(got, (2.0 * k + 1.0) * y["S"])


def test_alias_reference_cycle_stays_out_of_scope() -> None:
    """Mutually-referencing aliases resolve to nothing, rather than hanging."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [{"name": "age", "coords": ["a0", "a1"]}],
        "aliases": {"foi[age]": "ping[age]", "ping[age]": "foi[age] + 1.0"},
        "state": ["S[age]", "E[age]"],
        "transitions": [
            {"name": "expose", "from": "S[age]", "to": "E[age]", "rate": "foi[age]"},
        ],
    }
    assert compile_spec(spec).reactions == ()


def _importation_spec() -> dict[str, object]:
    return {
        "kind": "transitions",
        "factorize_axes": ["loc"],
        "axes": [
            {"name": "age", "coords": ["a0", "a1"]},
            {"name": "loc", "coords": ["d1", "d2", "d3"]},
        ],
        "state": ["S[age,loc]", "E[age,loc]"],
        "transitions": [
            {
                "name": "expose",
                "from": "S[age,loc]",
                "to": "E[age,loc]",
                "rate": "foi",
            },
            {
                "name": "import_case",
                "to": "E[age,loc]",
                "rate": "lambda_import[loc]",
            },
        ],
    }


def _time_varying_rate_spec() -> dict[str, object]:
    return {
        "kind": "transitions",
        "axes": [
            {"name": "loc", "coords": ["d1", "d2", "d3"]},
            {
                "name": "time",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 4.0},
                "size": 5,
            },
        ],
        "state": ["S[loc]", "E[loc]"],
        "transitions": [
            {
                "name": "expose",
                "from": "S[loc]",
                "to": "E[loc]",
                "rate": "lambda_import[time, loc]",
            },
        ],
    }


def test_source_only_transition_metadata_and_propensity() -> None:
    """A source-only reaction's propensity is the bare rate, no from_state factor."""
    c = compile_spec(_importation_spec())
    import_case = next(r for r in c.reactions if r.name == "import_case")
    assert import_case.from_base is None
    assert import_case.from_axes == ("age", "loc")
    assert import_case.to_base == "E"
    assert import_case.to_axes == ("age", "loc")
    assert import_case.sum_axes == ()
    assert import_case.pinned == ()

    y = {"S": np.ones((2, 3)) * 100.0, "E": np.zeros((2, 3))}
    params = {"foi": np.asarray(0.0), "lambda_import": np.array([0.01, 0.02, 0.03])}
    got = np.asarray(import_case.propensity_fn(0.0, y, **params))
    # Bare rate, broadcast to (age, loc) -- NOT multiplied by any state.
    expected = np.broadcast_to(params["lambda_import"], (2, 3))
    np.testing.assert_allclose(got, expected)


def test_source_only_transition_matches_deterministic_inflow_no_depletion() -> None:
    """The source-only reaction's propensity matches the deterministic E inflow.

    Correctness oracle, same pattern as
    ``test_reactions_reconstruct_deterministic_eval_fn``: the deterministic
    path already supports ``from: null`` (see ``_normalize.py``), so
    agreement here confirms the independent reaction artifact adds the
    SAME inflow to E, and -- critically -- doesn't erroneously deplete any
    compartment (there is no from_base to deplete).
    """
    c = compile_spec(_importation_spec())
    assert c.pytree_eval_fn is not None
    y = {"S": np.ones((2, 3)) * 100.0, "E": np.zeros((2, 3))}
    params = {"foi": np.asarray(0.0), "lambda_import": np.array([0.01, 0.02, 0.03])}
    dy = c.pytree_eval_fn(np.asarray(0.0), y, **params)

    import_case = next(r for r in c.reactions if r.name == "import_case")
    p_import = np.asarray(import_case.propensity_fn(0.0, y, **params))

    # foi=0, so "expose" contributes nothing -- dE should be exactly the
    # importation inflow, and dS should be exactly zero (nothing depleted).
    np.testing.assert_allclose(dy["E"], p_import)
    np.testing.assert_allclose(dy["S"], np.zeros((2, 3)))


def test_time_varying_rate_propensity_matches_deterministic() -> None:
    """A rate referencing a ``[time, ...]``-shaped param interpolates correctly.

    Regression test: ``_build_reaction_artifacts`` compiles each reaction's
    propensity independently of ``_wrap_eval_fn_for_time_varying`` /
    ``_wrap_pytree_eval_fn_for_time_varying`` -- before this fix, a rate
    like ``lambda_import[time, loc]`` (declaring a time-varying parameter
    by subscripting it with the time axis) compiled fine but reached
    ``propensity_fn`` expecting the raw, un-interpolated full ``(time,
    loc)`` grid array where its compiled code actually expected the
    already-time-stripped, current-timestep ``(loc,)``-shaped slice --
    ``ValueError: cannot reshape array of size N into shape (...)``. The
    correctness oracle here is the same pattern as
    ``test_reactions_reconstruct_deterministic_eval_fn``: the propensity
    must agree with the deterministic path's own (already-correct)
    time-interpolated inflow at a fractional (off-grid) ``t``.
    """
    c = compile_spec(_time_varying_rate_spec())
    expose = next((r for r in c.reactions if r.name == "expose"), None)
    assert expose is not None

    y = {"S": np.array([100.0, 100.0, 100.0]), "E": np.zeros(3)}
    # (time=5, loc=3) grid -- values jump between t=1 and t=2, so t=1.5
    # below exercises genuine linear interpolation, not just an exact
    # grid-point lookup.
    lam = np.array([
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])
    got = np.asarray(expose.propensity_fn(1.5, y, lambda_import=lam))

    assert c.pytree_eval_fn is not None
    dy = c.pytree_eval_fn(np.asarray(1.5), y, lambda_import=lam)
    np.testing.assert_allclose(got, np.asarray(dy["E"]))
    # Sanity: genuinely interpolated (halfway between [0.1,0.2,0.3]*100 and
    # [0.5,0.5,0.5]*100), not just clamped to one grid point's value.
    np.testing.assert_allclose(got, [30.0, 35.0, 40.0])


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


def test_alias_reference_using_a_binding_variable_as_axis_label() -> None:
    """``trv[v]`` inside ``apply_along(..., vax=v)`` resolves.

    The reference labels the position with the reduction's binding
    variable rather than the axis name, so the alias's declared ``vax``
    must be bound to ``v`` when its body is inlined (issue #193).
    """
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["a0", "a1"]},
            {"name": "vax", "coords": ["u", "f", "b"]},
        ],
        "state": ["S[age,vax]", "E[age,vax]", "I[age,vax]"],
        "aliases": {
            "trv[vax]": "tr[vax]",
            "foi[age]": "apply_along(I[age, vax:v] * (1.0 - trv[v]), vax=v)",
        },
        "transitions": [
            {
                "name": "expose",
                "from": "S[age,vax]",
                "to": "E[age,vax]",
                "rate": "foi[age]",
            },
        ],
    }
    c = compile_spec(spec)
    expose = next(r for r in c.reactions if r.name == "expose")
    rng = np.random.default_rng(0)
    y = {n: rng.uniform(1.0, 9.0, size=(2, 3)) for n in ("S", "E", "I")}
    tr = np.array([0.0, 0.25, 0.6])
    got = np.asarray(expose.propensity_fn(0.0, y, tr=tr))
    expected = (y["I"] * (1.0 - tr)).sum(axis=1)[:, None] * y["S"]
    np.testing.assert_allclose(got, expected)


def test_alias_reference_bound_to_a_reduction_coord() -> None:
    """``pool[loc:lp]`` inside ``apply_along(..., loc=lp)`` resolves.

    The alias's declared ``loc`` is bound to the reduction variable, so
    its body's free ``loc`` references must be rewritten to ``loc:lp``.
    """
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [{"name": "loc", "coords": ["d0", "d1", "d2"]}],
        "state": ["S[loc]", "E[loc]", "I[loc]"],
        "aliases": {
            "pool[loc]": "I[loc]",
            "tot[loc]": "S[loc] + I[loc]",
            "press[loc]": (
                "apply_along(w[loc, loc:lp] * (pool[loc:lp] / tot[loc:lp]), loc=lp)"
            ),
        },
        "transitions": [
            {"name": "expose", "from": "S[loc]", "to": "E[loc]", "rate": "press[loc]"},
        ],
    }
    c = compile_spec(spec)
    expose = next(r for r in c.reactions if r.name == "expose")
    rng = np.random.default_rng(1)
    y = {n: rng.uniform(1.0, 9.0, size=3) for n in ("S", "E", "I")}
    w = rng.uniform(size=(3, 3))
    got = np.asarray(expose.propensity_fn(0.0, y, w=w))
    expected = (w @ (y["I"] / (y["S"] + y["I"]))) * y["S"]
    np.testing.assert_allclose(got, expected)


def test_alias_inlining_avoids_capturing_a_reused_binding_variable() -> None:
    """An alias that binds the same variable name as its reference site.

    ``inner`` binds ``lp`` itself and is referenced as ``inner[loc:lp]``
    from inside another ``apply_along(..., loc=lp)``. Substituting
    naively would let ``inner``'s own binding capture the incoming
    ``loc:lp``, silently collapsing a double sum into a single one.
    """
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [{"name": "loc", "coords": ["d0", "d1", "d2"]}],
        "state": ["S[loc]", "E[loc]", "I[loc]"],
        "aliases": {
            "inner[loc]": "apply_along(w[loc, loc:lp] * I[loc:lp], loc=lp)",
            "outer[loc]": "apply_along(w[loc, loc:lp] * inner[loc:lp], loc=lp)",
        },
        "transitions": [
            {"name": "expose", "from": "S[loc]", "to": "E[loc]", "rate": "outer[loc]"},
        ],
    }
    c = compile_spec(spec)
    expose = next(r for r in c.reactions if r.name == "expose")
    rng = np.random.default_rng(2)
    y = {n: rng.uniform(1.0, 9.0, size=3) for n in ("S", "E", "I")}
    w = rng.uniform(size=(3, 3))
    got = np.asarray(expose.propensity_fn(0.0, y, w=w))
    np.testing.assert_allclose(got, (w @ (w @ y["I"])) * y["S"])


def test_alias_reference_pinned_to_a_literal_coord() -> None:
    """``foi[age, loc:d1]`` resolves, pinning the body to that coordinate.

    A literal pin shares the COORD index shape with a reduction binding,
    so it falls out of the same substitution; the single-level pass
    rejected it. Cross-checked against the deterministic RHS, whose
    ``dE/dt`` is exactly this transition's inflow.
    """
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["a0", "a1"]},
            {"name": "loc", "coords": ["d0", "d1", "d2"]},
        ],
        "state": ["S[age,loc]", "E[age,loc]", "I[age,loc]"],
        "aliases": {"foi[age, loc]": "beta * I[age, loc]"},
        "transitions": [
            {
                "name": "expose",
                "from": "S[age,loc]",
                "to": "E[age,loc]",
                "rate": "foi[age, loc:d1]",
            },
        ],
    }
    c = compile_spec(spec)
    expose = next(r for r in c.reactions if r.name == "expose")
    rng = np.random.default_rng(3)
    y = {n: rng.uniform(1.0, 9.0, size=(2, 3)) for n in ("S", "E", "I")}
    expected = (0.5 * y["I"][:, 1])[:, None] * y["S"]
    np.testing.assert_allclose(
        np.asarray(expose.propensity_fn(0.0, y, beta=0.5)), expected
    )
    assert c.pytree_eval_fn is not None
    np.testing.assert_allclose(
        np.asarray(c.pytree_eval_fn(0.0, y, beta=0.5)["E"]), expected
    )
