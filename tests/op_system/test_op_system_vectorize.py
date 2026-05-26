"""Unit tests for the vectorized compile path (``op_system._vectorize``).

Covers:
- Numerical equivalence between scalar and vectorized eval on a templated SIR.
- Sum-pattern recognition (alias summing over all cells of a template).
- Partial unrolling for templates with structural variation along one axis.
- Templated param buffer assembly from per-cell scalars.
- Silent fallback when the spec is unsupported.
"""

from __future__ import annotations

import dis
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import CodeType

    import pytest

from op_system._vectorize import build_vector_plan, last_vector_plan_bail_reason
from op_system.compile import _make_eval_fn, compile_rhs
from op_system.specs import normalize_rhs


def _sir_two_axis_spec() -> dict[str, object]:
    """Templated SIR over (age, loc) with a sum-over-all-cells alias.

    Returns:
        A spec dict suitable for passing to ``normalize_rhs``.
    """
    return {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "loc", "coords": ["a", "b", "c"]},
        ],
        "state": ["S[age, loc]", "I[age, loc]", "R[age, loc]"],
        "params": ["beta", "gamma"],
        "aliases": {
            # Pure sum across all cells of the I template — this is the
            # pattern the recognizer should collapse to ``np.sum(I_buf)``.
            "I_total": (
                "I__age_y__loc_a + I__age_y__loc_b + I__age_y__loc_c + "
                "I__age_o__loc_a + I__age_o__loc_b + I__age_o__loc_c"
            ),
        },
        "equations": {
            "S[age, loc]": "-beta * S[age, loc] * I_total",
            "I[age, loc]": "beta * S[age, loc] * I_total - gamma * I[age, loc]",
            "R[age, loc]": "gamma * I[age, loc]",
        },
    }


def _sir_templated_param_spec() -> dict[str, object]:
    """SIR with templated param ``gamma`` per (age,) — exercises param buffer.

    Returns:
        A spec dict suitable for passing to ``normalize_rhs``.
    """
    return {
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["y", "o"]}],
        "state": ["S[age]", "I[age]"],
        "params": ["beta", "gamma[age]"],
        "equations": {
            "S[age]": "-beta * S[age] * I[age]",
            "I[age]": "beta * S[age] * I[age] - gamma[age] * I[age]",
        },
    }


def _eval_equal(spec: dict[str, object], **call_params: object) -> None:
    rhs = normalize_rhs(spec)
    # ``compile_rhs`` always attempts the vectorized path; bypass it via
    # ``_make_eval_fn`` to keep an independent scalar reference for this
    # equivalence check.
    scalar_eval = _make_eval_fn(
        state_names=rhs.state_names,
        aliases=rhs.aliases,
        equations=rhs.equations,
    )
    c_v = compile_rhs(rhs)
    rng = np.random.RandomState(0)
    y = rng.rand(len(rhs.state_names))
    out_s = scalar_eval(0.0, y, **call_params)
    out_v = c_v.eval_fn(0.0, y, **call_params)
    assert np.allclose(out_s, out_v, atol=1e-12, rtol=0.0), (
        f"max abs diff = {np.max(np.abs(out_s - out_v))}"
    )


def test_vectorized_matches_scalar_two_axis_sir() -> None:
    """Vectorized eval matches scalar eval on a two-axis templated SIR."""
    _eval_equal(_sir_two_axis_spec(), beta=0.3, gamma=0.1)


def test_vectorized_matches_scalar_templated_param() -> None:
    """Vectorized eval matches scalar eval with a per-axis templated param."""
    _eval_equal(
        _sir_templated_param_spec(),
        beta=0.4,
        gamma=np.array([0.1, 0.2]),
    )


def test_sum_pattern_recognized_for_alias_over_full_template() -> None:
    """The ``I_total`` alias should collapse to a single ``np.sum`` call."""
    rhs = normalize_rhs(_sir_two_axis_spec())
    plan = build_vector_plan(rhs)
    assert plan is not None
    alias_codes = {base: code for base, code, _shape in plan.alias_codes}
    assert "I_total" in alias_codes
    code = alias_codes["I_total"]
    # After collapse, the only buffer name referenced is ``I_buf``.
    assert "I_buf" in code.co_names
    # Bytecode should be very small after collapse (compared to the dozens
    # of indexed loads the un-collapsed version would emit).
    assert len(code.co_code) < 50, (
        f"alias bytecode unexpectedly large: {len(code.co_code)}"
    )


def test_vectorizer_uses_aliases_ir_for_alias_code() -> None:
    """Alias vectorization should consume ``aliases_ir`` when available."""
    rhs = normalize_rhs(_sir_two_axis_spec())
    rhs = replace(rhs, aliases={**rhs.aliases, "I_total": "not valid python !!!"})

    plan = build_vector_plan(rhs)
    assert plan is not None
    alias_codes = {base: code for base, code, _shape in plan.alias_codes}
    assert "I_total" in alias_codes
    assert "I_buf" in alias_codes["I_total"].co_names


def test_vectorizer_uses_equations_ir_for_alias_free_equations() -> None:
    """Alias-free equation vectorization should consume ``equations_ir``."""
    rhs = normalize_rhs(_sir_templated_param_spec())
    rhs = replace(
        rhs,
        equations=tuple("not valid python !!!" for _ in rhs.equations),
    )

    plan = build_vector_plan(rhs)
    assert plan is not None
    assert len(plan.eq_groups) == 2
    assert all(group.codes for group in plan.eq_groups)


def test_vectorizer_uses_raw_equations_ir_with_alias_buffers() -> None:
    """Equation vectorization should use raw IR while preserving aliases."""
    rhs = normalize_rhs(_sir_two_axis_spec())
    rhs = replace(
        rhs,
        equations=tuple("not valid python !!!" for _ in rhs.equations),
    )

    plan = build_vector_plan(rhs)
    assert plan is not None
    alias_codes = {base: code for base, code, _shape in plan.alias_codes}
    assert "I_total" in alias_codes
    assert all(group.codes for group in plan.eq_groups)


def test_full_vectorization_when_no_structural_variation() -> None:
    """Fully-vectorize when all cells of a template are structurally identical.

    The vectorizer should produce a single shaped code per template (no
    unrolling) when no axis introduces structural variation.
    """
    rhs = normalize_rhs(_sir_two_axis_spec())
    plan = build_vector_plan(rhs)
    assert plan is not None
    for grp in plan.eq_groups:
        assert grp.unroll_axes == ()
        assert len(grp.codes) == 1


def test_fallback_on_non_templated_spec() -> None:
    """Silently fall back to the scalar engine for non-templated specs.

    A purely scalar (non-templated) RHS yields ``None`` for the plan and the
    compile path silently falls back to the scalar engine.
    """
    spec = {
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {"x": "-x", "y": "x - y"},
    }
    rhs = normalize_rhs(spec)
    assert build_vector_plan(rhs) is None
    # Compiling must transparently fall back and not raise.
    c = compile_rhs(rhs, xp=np)
    out = c.eval_fn(0.0, np.array([1.0, 2.0]))
    assert np.allclose(out, np.array([-1.0, -1.0]))


def test_bail_reason_recorded_for_non_templated_spec() -> None:
    """``last_vector_plan_bail_reason`` exposes why the plan was rejected.

    A scalar-only RHS should record a non-empty diagnostic explaining the
    bail (here: "scalar (non-wildcard) state template present").
    """
    rhs = normalize_rhs({
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {"x": "-x", "y": "x - y"},
    })
    assert build_vector_plan(rhs) is None
    reason = last_vector_plan_bail_reason()
    assert reason is not None
    assert "scalar" in reason or "no state templates" in reason


def test_bail_reason_cleared_on_success() -> None:
    """A successful ``build_vector_plan`` clears any prior bail reason."""
    # First trigger a bail to populate state.
    scalar_rhs = normalize_rhs({
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "-x"},
    })
    assert build_vector_plan(scalar_rhs) is None
    assert last_vector_plan_bail_reason() is not None
    # Then a supported templated spec should clear it.
    ok_rhs = normalize_rhs(_sir_two_axis_spec())
    plan = build_vector_plan(ok_rhs)
    assert plan is not None
    assert last_vector_plan_bail_reason() is None


def test_bail_reason_printed_to_stderr_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Setting ``OP_SYSTEM_DEBUG_VECTOR_PLAN`` prints bail reasons to stderr."""
    monkeypatch.setenv("OP_SYSTEM_DEBUG_VECTOR_PLAN", "1")
    rhs = normalize_rhs({
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "-x"},
    })
    assert build_vector_plan(rhs) is None
    captured = capsys.readouterr()
    assert "[op_system vector-plan] bail:" in captured.err


def test_compile_rhs_returns_finite_output() -> None:
    """``compile_rhs`` produces a finite, correctly-shaped eval output.

    Exercises the default (vectorized-with-fallback) path on a supported
    templated spec.
    """
    rhs = normalize_rhs(_sir_two_axis_spec())
    c = compile_rhs(rhs, xp=np)
    rng = np.random.RandomState(1)
    y = rng.rand(len(rhs.state_names))
    out = c.eval_fn(0.0, y, beta=0.3, gamma=0.1)
    assert out.shape == (len(rhs.state_names),)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Regression coverage for op_system #95: shaped-param subscripts in equations
# / aliases (e.g. ``r0_loc[loc]``) must not force per-cell unrolling along
# the subscripted axis.
# ---------------------------------------------------------------------------


def _sir_shaped_param_in_rate_spec() -> dict[str, object]:
    """Two-axis templated SIR with a shaped param subscripted by axis name.

    The transition rate uses ``k[loc]``, which the normalizer expands to
    ``k[<int>]`` per cell. The vectorizer must recognize this and rewrite
    the cell expression so the loc axis stays vectorized rather than
    unrolling to one code per loc.

    Returns:
        A spec dict suitable for passing to ``normalize_rhs``.
    """
    return {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "loc", "coords": ["a", "b", "c"]},
        ],
        "state": ["S[age, loc]", "I[age, loc]"],
        "params": ["beta", "gamma"],
        "equations": {
            "S[age, loc]": "-beta * k[loc] * S[age, loc] * I[age, loc]",
            "I[age, loc]": (
                "beta * k[loc] * S[age, loc] * I[age, loc] - gamma * I[age, loc]"
            ),
        },
    }


def test_shaped_param_subscript_keeps_axis_vectorized() -> None:
    """Subscripting a shaped param by axis name must not unroll that axis.

    Regression test for op_system #95: a transition rate of the form
    ``k[loc] * S[age, loc] * ...`` is normalized into per-cell expansions
    ``k[0] * ...``, ``k[1] * ...`` etc. Pre-fix this defeated the equation
    vectorizer's first/last-cell AST-equality check, causing it to fall
    back to per-cell unrolling along ``loc``. Post-fix, ``loc`` stays in
    the vec-axes set and the template is compiled to a single shaped code.
    """
    rhs = normalize_rhs(_sir_shaped_param_in_rate_spec())
    plan = build_vector_plan(rhs)
    assert plan is not None
    by_base = {grp.base: grp for grp in plan.eq_groups}
    for base in ("S", "I"):
        grp = by_base[base]
        assert "loc" in grp.vec_axes, (
            f"{base!r} should keep loc vectorized, got "
            f"vec_axes={grp.vec_axes} unroll_axes={grp.unroll_axes}"
        )
        assert "loc" not in grp.unroll_axes
        assert len(grp.codes) == 1, (
            f"{base!r} should compile to a single shaped code, got {len(grp.codes)}"
        )


def test_shaped_param_subscript_eval_matches_scalar() -> None:
    """Numerical equivalence with a shaped param subscripted in equations."""
    _eval_equal(
        _sir_shaped_param_in_rate_spec(),
        beta=0.3,
        gamma=0.1,
        k=np.array([1.0, 2.0, 0.5]),
    )


def test_shaped_param_subscript_in_alias_keeps_axis_vectorized() -> None:
    """Subscripting a shaped param by axis name inside an alias body.

    Same regression target as the equation form, but driven through an
    alias used by downstream transitions.
    """
    spec: dict[str, object] = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "loc", "coords": ["a", "b", "c"]},
        ],
        "state": ["S[age, loc]", "I[age, loc]"],
        "params": ["beta", "gamma"],
        "aliases": {
            "scaled[age, loc]": "beta * k[loc] * S[age, loc] * I[age, loc]",
        },
        "equations": {
            "S[age, loc]": "-scaled[age, loc]",
            "I[age, loc]": "scaled[age, loc] - gamma * I[age, loc]",
        },
    }
    rhs = normalize_rhs(spec)
    plan = build_vector_plan(rhs)
    assert plan is not None
    for grp in plan.eq_groups:
        assert grp.unroll_axes == ()
        assert len(grp.codes) == 1
    _eval_equal(spec, beta=0.3, gamma=0.1, k=np.array([1.0, 2.0, 0.5]))


def test_shaped_param_subscript_partial_axes_eval_matches_scalar() -> None:
    """Shaped param indexed by a single axis when target has multiple axes.

    Verifies the broadcast path that inserts size-1 dims for target axes
    not in the subscript axis set.
    """
    spec: dict[str, object] = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "loc", "coords": ["a", "b"]},
        ],
        "state": ["S[age, loc]", "I[age, loc]"],
        "params": ["beta"],
        "equations": {
            "S[age, loc]": "-beta * c[age] * S[age, loc] * I[age, loc]",
            "I[age, loc]": (
                "beta * c[age] * S[age, loc] * I[age, loc] - d[loc] * I[age, loc]"
            ),
        },
    }
    _eval_equal(spec, beta=0.3, c=np.array([1.0, 0.5]), d=np.array([0.1, 0.2]))


def test_continuous_axis_with_pinned_selector_transition_compiles() -> None:
    """Regression for #97: continuous axis with pinned-coord selector must compile.

    ``kind: transitions`` with a continuous axis and a pinned-coord selector on
    a categorical axis must compile.

    Previously, ``build_vector_plan`` keyed ``axis_index`` by the raw
    continuous-axis coords (e.g. ``float`` ``0.0``) while per-cell
    ``coord_assignments`` stringified them (``'0.0'``), causing a
    ``KeyError: '0.0'`` from ``_build_access_ast`` when a transition's
    ``to:`` template needed a scalar lookup against the continuous axis.
    """
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "type": "continuous", "coords": [0.0, 5.0, 10.0]},
            {"name": "imm", "coords": ["x0", "x10"]},
        ],
        "state": ["S[age]", "X[age, imm]"],
        "transitions": [
            {"from": "S[age]", "to": "X[age, imm=x10]", "rate": "0.1"},
        ],
    }
    rhs = normalize_rhs(spec)
    plan = build_vector_plan(rhs)
    assert plan is not None


# ---------------------------------------------------------------------------
# Weighted-sum collapse: fused multiply+reduce for ``apply_along(integrate)``
# ---------------------------------------------------------------------------


def _continuum_integrate_spec() -> dict[str, object]:
    """Templated SIR with an alias that ``apply_along``-integrates over age.

    The continuous-age axis is integrated via trapezoidal weights inside an
    alias scalar. Post-normalize this is a long Add chain of
    ``<weight_const> * <S>_buf[<age_idx>, <loc_idx>]`` terms covering every
    cell of ``S``; the weighted-sum collapser must fold it into a single
    fused ``np.sum(weights * S_buf)`` call.

    Returns:
        A spec dict suitable for passing to ``normalize_rhs``.
    """
    return {
        "kind": "expr",
        "axes": [
            {"name": "age", "type": "continuous", "coords": [0.0, 1.0, 3.0, 4.0]},
            {"name": "loc", "coords": ["a", "b"]},
        ],
        "state": ["S[age, loc]", "I[age, loc]"],
        "params": ["beta", "gamma"],
        "aliases": {
            # Integrate S over age, sum over loc → scalar.
            "S_total": (
                "apply_along("
                "apply_along(S[age:a, loc:l], loc=l, kernel=sum), "
                "age=a, kernel=integrate)"
            ),
        },
        "equations": {
            "S[age, loc]": "-beta * S[age, loc] * S_total",
            "I[age, loc]": ("beta * S[age, loc] * S_total - gamma * I[age, loc]"),
        },
    }


def test_weighted_apply_along_collapses_to_fused_reduction() -> None:
    """``apply_along(integrate)`` should fuse weights into one reduction.

    The ``S_total`` alias expands to a long chain of
    ``w_age[i] * S_buf[i, j]`` terms covering every cell of ``S_buf``. The
    weighted-sum collapser must rewrite this into a single
    ``np.sum(<weights> * S_buf)`` fused multiply+reduce (no broadcast
    intermediate, no per-cell load chain).
    """
    rhs = normalize_rhs(_continuum_integrate_spec())
    plan = build_vector_plan(rhs)
    assert plan is not None
    alias_codes = {base: code for base, code, _shape in plan.alias_codes}
    assert "S_total" in alias_codes
    code = alias_codes["S_total"]
    assert "S_buf" in code.co_names
    # Structural assertions: post-collapse the alias must reference its
    # buffer at most once and reference both ``asarray`` and ``sum`` (a
    # single fused multiply-reduce). The un-collapsed form would reference
    # ``S_buf`` once per cell and never call ``asarray``/``sum``.
    s_buf_loads = sum(
        1
        for instr in dis.get_instructions(code)
        if instr.opname.startswith("LOAD_") and instr.argval == "S_buf"
    )
    assert s_buf_loads == 1, (
        f"expected a single fused S_buf load after collapse, got {s_buf_loads}"
    )
    # The fused form materializes the weight vector with either
    # ``np.asarray`` (string-collapser path) or ``np.array`` (IR fast path);
    # both are correct and observably equivalent for a literal-list argument.
    assert "asarray" in code.co_names or "array" in code.co_names
    assert "sum" in code.co_names

    # The collapser must emit at least one numeric weight constant. Python
    # may fold a list of numeric literals into a tuple constant, so search
    # both flat and nested constants.
    def _has_float(obj: object) -> bool:
        if isinstance(obj, float):
            return True
        if isinstance(obj, (list, tuple, set, frozenset)):
            return any(_has_float(x) for x in obj)
        return False

    assert any(_has_float(c) for c in code.co_consts)


def test_weighted_apply_along_eval_matches_scalar() -> None:
    """Numerical equivalence between fused-reduce and scalar eval paths."""
    _eval_equal(_continuum_integrate_spec(), beta=0.3, gamma=0.1)


# ---------------------------------------------------------------------------
# Uniform-weight collapse (op_system#103 regression guard)
#
# When ``_distribute_const_over_add`` pushes an outer constant factor over a
# ``kernel=sum`` apply_along chain, every leaf in the resulting Add chain
# carries the same weight ``c``. The collapser must emit
# ``c * np.sum(buf)`` rather than ``np.sum(np.asarray([c, ..., c]) * buf)``;
# the inlined uniform array form bloats the JAX trace and balloons XLA
# compile time on large network/categorical models (see issue #103).
# ---------------------------------------------------------------------------


def _network_outer_const_spec() -> dict[str, object]:
    """Categorical SIR whose alias has a constant factor outside ``kernel=sum``.

    The ``S_scaled`` alias multiplies a numeric constant by a full-axis
    ``apply_along(kernel=sum)`` over a templated state. Post-normalize this
    expands to ``0.25 * (S__age_y__loc_a + S__age_y__loc_b + ...)``; after
    ``_distribute_const_over_add`` the chain becomes
    ``0.25*S_buf[0,0] + 0.25*S_buf[0,1] + ...`` — a uniform-weight chain
    that the collapser must fold back to ``0.25 * np.sum(S_buf)``.

    Returns:
        A spec dict suitable for passing to ``normalize_rhs``.
    """
    return {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "loc", "coords": ["a", "b"]},
        ],
        "state": ["S[age, loc]", "I[age, loc]"],
        "params": ["beta", "gamma"],
        "aliases": {
            "S_scaled": (
                "0.25 * apply_along("
                "apply_along(S[age:a, loc:l], loc=l, kernel=sum), "
                "age=a, kernel=sum)"
            ),
        },
        "equations": {
            "S[age, loc]": "-beta * S[age, loc] * S_scaled",
            "I[age, loc]": ("beta * S[age, loc] * S_scaled - gamma * I[age, loc]"),
        },
    }


def test_uniform_weight_chain_collapses_without_inlined_array() -> None:
    """Uniform-weight full-cover chain must emit ``c*np.sum(buf)``.

    Regression test for op_system#103: the previous emission used
    ``np.sum(np.asarray([c, c, ..., c]).reshape(shape) * buf)``, embedding
    an inlined Python literal of length ``prod(shape)`` in every alias
    that had any outer constant factor multiplying a ``kernel=sum``
    apply_along. On large network configs this produced multi-minute XLA
    compiles. The fix must (a) keep ``S_buf`` referenced once, (b) keep
    ``sum`` referenced, and (c) NOT emit ``asarray`` for this case.
    """
    rhs = normalize_rhs(_network_outer_const_spec())
    plan = build_vector_plan(rhs)
    assert plan is not None
    alias_codes = {base: code for base, code, _shape in plan.alias_codes}
    assert "S_scaled" in alias_codes
    code = alias_codes["S_scaled"]
    s_buf_loads = sum(
        1
        for instr in dis.get_instructions(code)
        if instr.opname.startswith("LOAD_") and instr.argval == "S_buf"
    )
    assert s_buf_loads == 1, (
        f"expected a single fused S_buf load after collapse, got {s_buf_loads}"
    )
    assert "sum" in code.co_names, (
        "uniform-weight chain should collapse to a single np.sum call"
    )
    assert "asarray" not in code.co_names, (
        "uniform-weight chain must NOT emit an inlined np.asarray weight "
        "array (regression of op_system#103); expected c * np.sum(buf) form"
    )
    # The constant factor 0.25 must survive somewhere in co_consts.
    assert any(
        isinstance(c, float) and c == 0.25  # noqa: RUF069
        for c in code.co_consts
    ), f"expected 0.25 constant in co_consts, got {code.co_consts}"


def test_uniform_weight_chain_eval_matches_scalar() -> None:
    """Numerical equivalence on the uniform-weight collapse path."""
    _eval_equal(_network_outer_const_spec(), beta=0.3, gamma=0.1)


# ---------------------------------------------------------------------------
# Recursion-safety on long Add chains (op_system#103 regression guard)
#
# Aggregator aliases that sum across a large templated state expand at
# normalize time to a single Add chain with one term per cell. With the
# previous recursive ``ast.NodeTransformer``-based ``_distribute_const_over_add``
# pass, chains beyond Python's default ~1000-frame limit raised a silent
# ``RecursionError`` that was swallowed by the blanket ``except`` in
# ``build_vector_plan``, dropping the model to the scalar fallback (a 100x
# slowdown on real workloads). The pass must now be iterative on the outer
# chain and survive arbitrarily long aliases.
# ---------------------------------------------------------------------------


def _huge_chain_spec(n_loc: int) -> dict[str, object]:
    """Categorical SIR with ``n_loc`` locations and a constant-scaled aggregator.

    The ``S_scaled`` alias forces ``_distribute_const_over_add`` to walk a
    chain of ``n_loc`` Add terms; the outer ``0.25`` factor must be pushed
    inside before the collapser can fold the chain into ``c * np.sum(buf)``.

    Args:
        n_loc: Number of synthetic location coords (chain length).

    Returns:
        A spec dict suitable for passing to ``normalize_rhs``.
    """
    coords = [f"l{i:04d}" for i in range(n_loc)]
    return {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": coords}],
        "state": ["S[loc]", "I[loc]"],
        "params": ["beta", "gamma"],
        "aliases": {
            "S_scaled": "0.25 * apply_along(S[loc:l], loc=l, kernel=sum)",
        },
        "equations": {
            "S[loc]": "-beta * S[loc] * S_scaled",
            "I[loc]": "beta * S[loc] * S_scaled - gamma * I[loc]",
        },
    }


def test_long_add_chain_does_not_raise_recursionerror() -> None:
    """Chains far beyond ``sys.getrecursionlimit()`` must vectorize cleanly.

    Regression guard for op_system#103: the previous recursive pass blew the
    Python C-stack on any aggregator alias whose expanded Add chain exceeded
    the default 1000-frame recursion limit. Picks ``n_loc`` well above any
    plausible default limit so that a recursive implementation would fail
    even if the user has bumped ``setrecursionlimit`` modestly.
    """
    rhs = normalize_rhs(_huge_chain_spec(n_loc=2500))
    plan = build_vector_plan(rhs)
    assert plan is not None, (
        "build_vector_plan returned None on a long-chain alias - "
        "_distribute_const_over_add likely raised RecursionError that "
        "was swallowed by the blanket except in build_vector_plan"
    )
    alias_codes = {base: code for base, code, _shape in plan.alias_codes}
    assert "S_scaled" in alias_codes
    code = alias_codes["S_scaled"]
    # Must still collapse to the fused np.sum(buf) form, not unroll 2500 loads.
    s_buf_loads = sum(
        1
        for instr in dis.get_instructions(code)
        if instr.opname.startswith("LOAD_") and instr.argval == "S_buf"
    )
    assert s_buf_loads == 1, (
        f"long chain should still collapse to a single S_buf load, got {s_buf_loads}"
    )


# ---------------------------------------------------------------------------
# IR-based vectorization fast path (issue #112)
# ---------------------------------------------------------------------------


def test_ir_fast_path_emits_no_per_cell_names_on_clean_templated_spec() -> None:
    """Equations lowered via the IR fast path reference only buffer names.

    On a templated spec whose equations contain no string-expanded axis
    reductions (no ``apply_along``, ``sum_over``, etc.), the IR-based
    lowering in ``_rewrite_cell_to_vector`` should produce code objects
    that reference only ``<base>_buf`` / scalar-param names — never the
    per-cell expanded names like ``S__age_y__loc_a`` that the legacy
    ``_NameRewriter`` path produces as intermediates.
    """
    rhs = normalize_rhs(_sir_two_axis_spec())
    plan = build_vector_plan(rhs)
    assert plan is not None
    for grp in plan.eq_groups:
        for code in grp.codes:
            for name in code.co_names:
                assert "__" not in name, (
                    f"per-cell name {name!r} leaked into vectorized "
                    f"code for template {grp.base!r}"
                )


def test_ir_fast_path_preserves_numerical_parity_with_scalar() -> None:
    """The IR fast path produces numerically identical output to scalar eval."""
    # Same equivalence check as test_vectorized_matches_scalar_two_axis_sir,
    # but kept distinct so a regression that disables the fast path is
    # caught by the structural test above without masking the parity check.
    _eval_equal(_sir_two_axis_spec(), beta=0.3, gamma=0.1)


def test_template_level_cse_extracts_shared_subexpressions() -> None:
    """Template-level CSE finds shared sub-expressions across state templates.

    For the two-axis SIR model the S and I equations both contain
    ``beta * S_buf * I_total_buf`` (or equivalent product involving those
    buffers).  After CSE the plan must carry at least one CSE binding, and
    the binding code must not contain any per-cell ``__`` names.
    """
    rhs = normalize_rhs(_sir_two_axis_spec())
    plan = build_vector_plan(rhs)
    assert plan is not None
    assert len(plan.cse_codes) > 0, (
        "expected at least one CSE binding for the two-axis SIR model "
        f"(got cse_codes={plan.cse_codes!r})"
    )
    # CSE binding codes must reference only buffer/param names, not per-cell names.
    for name, code in plan.cse_codes:
        for cn in code.co_names:
            assert "__" not in cn, (
                f"per-cell name {cn!r} leaked into CSE binding {name!r}"
            )


def _apply_along_categorical_spec() -> dict[str, object]:
    """SIR over (pop,) with an ``apply_along(I[pop:j], pop=j)`` reduction.

    Used to exercise Stage 1b: the reduce-bearing IR carries a
    ``Reduce(kind='apply_along', ...)`` node that lowers directly to
    ``np.sum`` over the bound axis, bypassing the post-expansion string
    fallback.

    Returns:
        A spec dict suitable for passing to ``normalize_rhs``.
    """
    return {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1", "p2", "p3"]}],
        "state": ["S[pop]", "I[pop]", "R[pop]"],
        "params": ["beta", "gamma"],
        "equations": {
            "S[pop]": "-beta * S[pop] * apply_along(I[pop:j], pop=j)",
            "I[pop]": ("beta * S[pop] * apply_along(I[pop:j], pop=j) - gamma * I[pop]"),
            "R[pop]": "gamma * I[pop]",
        },
    }


def test_reduce_ir_fast_path_emits_np_sum_for_apply_along() -> None:
    """Reduce-bearing IR lowers apply_along directly to ``np.sum`` on the buffer.

    Stage 1b: the IR fast path now consumes
    ``NormalizedRhs.equations_ir_reduce`` (Reduce nodes preserved) and
    emits a ``np.sum(I_buf, axis=...)`` reduction without falling back to
    the string-expanded ``I__pop_p1 + I__pop_p2 + ...`` form. The compiled
    code must therefore reference ``I_buf`` and ``sum`` somewhere in the plan
    (either in the S-equation code or in a CSE binding it references), with no
    per-cell ``I__pop_*`` names leaking through.
    """
    rhs = normalize_rhs(_apply_along_categorical_spec())
    assert rhs.equations_ir_reduce, "Stage 1a should have populated reduce IR"
    plan = build_vector_plan(rhs)
    assert plan is not None

    # Collect all code objects: CSE bindings + equation codes.
    all_codes: list[CodeType] = [c for _, c in plan.cse_codes]
    for grp in plan.eq_groups:
        all_codes.extend(grp.codes)

    all_names: set[str] = set()
    for code in all_codes:
        all_names.update(code.co_names)

    # No per-cell name leakage from string expansion in any code.
    assert not any("__" in n for n in all_names), (
        f"per-cell names leaked into reduce-lowered code: "
        f"{sorted(n for n in all_names if '__' in n)}"
    )
    assert "I_buf" in all_names, (
        f"expected I_buf somewhere in plan codes: {sorted(all_names)}"
    )
    assert "sum" in all_names, (
        f"expected np.sum somewhere in plan codes: {sorted(all_names)}"
    )
    # A direct np.sum(I_buf, axis=0) collapse loads I_buf at most once in any
    # single code object (CSE binding, S-equation, or I-equation).  More than
    # one load in a single object would indicate per-cell scalar expansion.
    for code in all_codes:
        loads = sum(
            1
            for instr in dis.get_instructions(code)
            if instr.opname.startswith("LOAD_") and instr.argval == "I_buf"
        )
        assert loads <= 1, (
            f"expected at most one I_buf load per code object, got {loads} in {code}"
        )


def test_reduce_ir_fast_path_numerical_parity() -> None:
    """Reduce-lowered apply_along eval matches the scalar reference."""
    _eval_equal(_apply_along_categorical_spec(), beta=0.3, gamma=0.1)


def _same_axis_twice_apply_along_spec() -> dict[str, object]:
    """SIR-like spec with a same-axis-twice contact kernel reduction.

    The S-equation contains ``apply_along(K[age, age:ap] * I[age:ap],
    age=ap)`` — a same-axis-twice contraction representing
    ``sum_{ap} K[age, ap] * I[ap]`` (an age-structured force-of-infection
    matvec). Stage 1c lowers this via the IR fast path by using the
    binding variable ``ap`` itself as a synthetic axis label so ``K_buf``
    broadcasts as ``(N_age, N_ap)`` while ``I_buf`` broadcasts as
    ``(1, N_ap)``.

    Returns:
        A spec dict suitable for ``normalize_rhs``.
    """
    return {
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["a1", "a2", "a3"]}],
        "state": ["S[age]", "I[age]", "R[age]"],
        "params": [
            "beta",
            "gamma",
            {"name": "K", "axes": ["age", "age"]},
        ],
        "equations": {
            "S[age]": (
                "-beta * S[age] * apply_along(K[age, age:ap] * I[age:ap], age=ap)"
            ),
            "I[age]": (
                "beta * S[age] * apply_along(K[age, age:ap] * I[age:ap], age=ap)"
                " - gamma * I[age]"
            ),
            "R[age]": "gamma * I[age]",
        },
    }


def test_same_axis_twice_apply_along_ir_fast_path() -> None:
    """Same-axis-twice apply_along lowers via the IR fast path (Stage 1c).

    The compiled S-equation code must reference ``K_buf``, ``I_buf``,
    and ``sum`` directly — proving the lowering does NOT fall back to
    the string-expanded ``K__age_a1__age_a1 * I__age_a1 + ...`` form.
    """
    rhs = normalize_rhs(_same_axis_twice_apply_along_spec())
    assert rhs.equations_ir_reduce
    plan = build_vector_plan(rhs)
    assert plan is not None
    s_group = next(g for g in plan.eq_groups if g.base == "S")
    code = s_group.codes[0]
    names = set(code.co_names)
    assert not any("__" in n for n in names), (
        f"per-cell names leaked into reduce-lowered code: "
        f"{sorted(n for n in names if '__' in n)}"
    )
    assert "K_buf" in names, f"expected K_buf in {sorted(names)}"
    assert "I_buf" in names, f"expected I_buf in {sorted(names)}"
    assert "sum" in names, f"expected np.sum in {sorted(names)}"


def test_same_axis_twice_apply_along_numerical_parity() -> None:
    """Same-axis-twice apply_along eval matches the scalar reference."""
    spec = _same_axis_twice_apply_along_spec()
    k_mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    _eval_equal(spec, beta=0.3, gamma=0.1, K=k_mat)


def _same_axis_twice_bare_label_spec() -> dict[str, object]:
    """Same-axis-twice spec using the binding variable as a bare axis label.

    Identical semantics to :func:`_same_axis_twice_apply_along_spec` but
    writes ``I[ap]`` (bare axis-label form) instead of ``I[age:ap]``
    (coord form) inside the ``apply_along`` body.  This exercises the
    branch added in issue #153: a ``Subscript`` index with ``coord=None``
    whose ``axis`` is a binding variable must be treated as a FREE axis
    reference once the binding variable is added to ``body_axis_names``.

    Returns:
        A spec dict suitable for ``normalize_rhs``.
    """
    return {
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["a1", "a2", "a3"]}],
        "state": ["S[age]", "I[age]", "R[age]"],
        "params": [
            "beta",
            "gamma",
            {"name": "K", "axes": ["age", "age"]},
        ],
        "equations": {
            "S[age]": ("-beta * S[age] * apply_along(K[age, age:ap] * I[ap], age=ap)"),
            "I[age]": (
                "beta * S[age] * apply_along(K[age, age:ap] * I[ap], age=ap)"
                " - gamma * I[age]"
            ),
            "R[age]": "gamma * I[age]",
        },
    }


def test_same_axis_twice_bare_label_ir_fast_path() -> None:
    """Bare-label ``I[ap]`` same-axis-twice contraction uses the IR fast path.

    Regression for issue #153: when the spec uses the binding variable
    as a bare axis label (``I[ap]``) rather than the ``coord=`` form
    (``I[age:ap]``), the vectorizer must still succeed and must not fall
    back to per-cell scalar expansion.
    """
    rhs = normalize_rhs(_same_axis_twice_bare_label_spec())
    plan = build_vector_plan(rhs)
    assert plan is not None, (
        f"vectorizer fell back to scalar path; bail reason: "
        f"{last_vector_plan_bail_reason()}"
    )
    s_group = next(g for g in plan.eq_groups if g.base == "S")
    code = s_group.codes[0]
    names = set(code.co_names)
    assert not any("__" in n for n in names), (
        f"per-cell names leaked into reduce-lowered code: "
        f"{sorted(n for n in names if '__' in n)}"
    )
    assert "K_buf" in names, f"expected K_buf in {sorted(names)}"
    assert "I_buf" in names, f"expected I_buf in {sorted(names)}"
    assert "sum" in names, f"expected np.sum in {sorted(names)}"


def test_same_axis_twice_bare_label_numerical_parity() -> None:
    """Bare-label ``I[ap]`` same-axis-twice eval matches the scalar reference.

    Regression for issue #153.
    """
    spec = _same_axis_twice_bare_label_spec()
    k_mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    _eval_equal(spec, beta=0.3, gamma=0.1, K=k_mat)


# ---------------------------------------------------------------------------
# Low-rank factored kernel with bare binding label (issue #152)
#
# A separable contact kernel  K[age, ap] = sum_r F[r, age] * H[r, ap]  can
# be expressed as nested ``apply_along`` with factor arrays F and H and a
# non-state ``rank`` axis.  The inner apply_along body uses ``I[ap]`` (bare
# binding-variable label) rather than ``I[age:ap]`` (coord form).  Without
# the extension to ``_binding_collides_with_free_index``, the binding
# variable ``ap`` never enters ``body_axis_names``, causing it to classify
# as COORD_SYMBOL → ``UnsupportedIRLoweringError`` → scalar fallback.
# ---------------------------------------------------------------------------


def _low_rank_bare_label_spec() -> dict[str, object]:
    """Low-rank factored SIR using bare-label ``I[ap]`` (no same-axis-twice).

    The force-of-infection is ``sum_r F[r, age] * (sum_ap H[r, ap] * I[ap])``
    — a rank-R approximation to a dense contact kernel.  The inner
    ``apply_along`` uses the binding variable ``ap`` as a bare axis label on
    ``I`` (rather than the ``coord=`` form ``I[age:ap]``), which is the
    natural way to write a pure summation index.

    Returns:
        A spec dict suitable for ``normalize_rhs``.
    """
    return {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["a1", "a2", "a3"]},
            {"name": "rank", "coords": ["r1", "r2"]},
        ],
        "state": ["S[age]", "I[age]"],
        "params": [
            {"name": "F", "axes": ["rank", "age"]},
            {"name": "H", "axes": ["rank", "age"]},
            "beta",
            "gamma",
        ],
        "equations": {
            "S[age]": (
                "-beta * S[age] * apply_along("
                "F[rank, age]"
                " * apply_along(H[rank, age:ap] * I[ap], age=ap, kernel=sum),"
                " rank=c, kernel=sum)"
            ),
            "I[age]": (
                "beta * S[age] * apply_along("
                "F[rank, age]"
                " * apply_along(H[rank, age:ap] * I[ap], age=ap, kernel=sum),"
                " rank=c, kernel=sum)"
                " - gamma * I[age]"
            ),
        },
    }


def test_low_rank_bare_label_ir_fast_path() -> None:
    """Low-rank bare-label ``I[ap]`` nested apply_along uses the IR fast path.

    Regression for issue #152: without the ``has_bare_binding`` extension to
    ``_binding_collides_with_free_index``, the binding variable ``ap`` is
    never added to ``body_axis_names`` in the no-same-axis-twice case, so
    ``I[ap]`` classifies as COORD_SYMBOL and the vectorizer falls back to the
    scalar path.  Post-fix the plan must be non-None and must not contain any
    per-cell ``__`` names.
    """
    rhs = normalize_rhs(_low_rank_bare_label_spec())
    plan = build_vector_plan(rhs)
    assert plan is not None, (
        f"vectorizer fell back to scalar path; bail reason: "
        f"{last_vector_plan_bail_reason()}"
    )
    all_codes: list[CodeType] = [c for _, c in plan.cse_codes]
    for grp in plan.eq_groups:
        all_codes.extend(grp.codes)
    all_names: set[str] = set()
    for code in all_codes:
        all_names.update(code.co_names)
    assert not any("__" in n for n in all_names), (
        f"per-cell names leaked into vectorized code: "
        f"{sorted(n for n in all_names if '__' in n)}"
    )
    assert "F_buf" in all_names, f"expected F_buf in {sorted(all_names)}"
    assert "H_buf" in all_names, f"expected H_buf in {sorted(all_names)}"
    assert "I_buf" in all_names, f"expected I_buf in {sorted(all_names)}"
    assert "sum" in all_names, f"expected np.sum in {sorted(all_names)}"


def test_low_rank_bare_label_numerical_parity() -> None:
    """Low-rank bare-label ``I[ap]`` vectorized output matches numpy reference.

    The scalar-path eval cannot resolve ``F__rank_r1[age]``-style partial
    expansions for non-state shaped axes, so the parity check is performed
    against a direct numpy reference computation instead of ``_eval_equal``.
    The reference collapses the low-rank factored kernel to a dense matrix
    ``K[i, j] = sum_r F[r, i] * H[r, j]`` (i.e. ``F.T @ H``), then
    evaluates the standard force-of-infection matvec.
    """
    rhs = normalize_rhs(_low_rank_bare_label_spec())
    c = compile_rhs(rhs)
    rng = np.random.RandomState(0)
    y = rng.rand(len(rhs.state_names))
    f_mat = np.array([[1.0, 2.0, 0.5], [0.3, 0.7, 1.2]])
    h_mat = np.array([[0.6, 0.4, 0.8], [1.1, 0.9, 0.2]])
    out = c.eval_fn(0.0, y, beta=0.3, gamma=0.1, F=f_mat, H=h_mat)
    s_vec, i_vec = y[:3], y[3:]
    k_dense = f_mat.T @ h_mat  # (n_age, n_age)
    foi = k_dense @ i_vec
    expected = np.concatenate([
        -0.3 * s_vec * foi,
        0.3 * s_vec * foi - 0.1 * i_vec,
    ])
    assert np.allclose(out, expected, atol=1e-12, rtol=0.0), (
        f"max abs diff = {np.max(np.abs(out - expected))}"
    )
