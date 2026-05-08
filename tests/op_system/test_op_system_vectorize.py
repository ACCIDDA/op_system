"""Unit tests for the vectorized compile path (``op_system._vectorize``).

Covers:
- Numerical equivalence between scalar and vectorized eval on a templated SIR.
- Sum-pattern recognition (alias summing over all cells of a template).
- Partial unrolling for templates with structural variation along one axis.
- Templated param buffer assembly from per-cell scalars.
- Silent fallback when the spec is unsupported.
"""

from __future__ import annotations

import numpy as np

from op_system._vectorize import build_vector_plan
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
        xp=np,
    )
    c_v = compile_rhs(rhs, xp=np)
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
