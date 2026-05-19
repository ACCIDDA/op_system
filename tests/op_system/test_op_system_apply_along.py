"""Unit tests for the ``apply_along`` expression primitive.

``apply_along(..., inner_expr, [kernel=sum|integrate], axis1=var1)``
expands directly to weighted sums over the declared axes (categorical
axes use uniform weights of 1; continuous axes use trapezoidal weights
derived from the axis ``deltas``).  These tests cover:

- single-axis categorical contraction
- multi-axis categorical contraction
- multi-axis continuous contraction
- explicit ``kernel=`` override and validation
- inference failure on mixed axis types
- nested ``apply_along`` calls
"""

from __future__ import annotations

import pytest

from op_system.specs import normalize_expr_rhs


def test_apply_along_single_axis_categorical_expands_to_sum() -> None:
    """Single-axis apply_along should expand to a flat sum over coords."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1", "p2"]}],
        "state": ["S[pop]", "I[pop]"],
        "equations": {
            "S[pop]": "-beta * S[pop] * apply_along(I[pop:j], pop=j)",
            "I[pop]": "beta * S[pop] * apply_along(I[pop:j], pop=j) - gamma * I[pop]",
        },
    }
    out = normalize_expr_rhs(spec)
    eq_s = out.equations[0]
    assert "I__pop_p1" in eq_s
    assert "I__pop_p2" in eq_s


def test_apply_along_multi_axis_categorical_expands_outer_product() -> None:
    """Two-axis apply_along should sum over the full Cartesian product."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
        ],
        "state": ["I[age, vax]", "N"],
        "equations": {
            "I[age, vax]": "0.0",
            "N": "apply_along(I[age:a, vax:b], age=a, vax=b)",
        },
    }
    out = normalize_expr_rhs(spec)
    eq_n = out.equations[-1]
    for a in ("y", "o"):
        for v in ("u", "v"):
            assert f"I__age_{a}__vax_{v}" in eq_n


def test_apply_along_multi_axis_continuous_uses_trapezoidal_weights() -> None:
    """Two-axis apply_along over continuous axes should multiply trapezoidal weights."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "x", "type": "continuous", "coords": [0.0, 1.0, 3.0]},
            {"name": "y", "type": "continuous", "coords": [0.0, 2.0]},
        ],
        "state": ["u[x, y]", "v"],
        "equations": {
            "u[x, y]": "0.0",
            "v": "apply_along(u[x:i, y:j], x=i, y=j)",
        },
    }
    out = normalize_expr_rhs(spec)
    eq_v = out.equations[-1]
    # x deltas: [0.5, 1.5, 1.0]; y deltas: [1.0, 1.0]
    for w in ("0.5", "1.5", "1.0"):
        assert w in eq_v
    for xc in ("0_0", "1_0", "3_0"):
        for yc in ("0_0", "2_0"):
            assert f"u__x_{xc}__y_{yc}" in eq_v


def test_apply_along_explicit_kernel_sum() -> None:
    """Explicit kernel=sum on a categorical axis should succeed."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "g", "coords": ["a", "b"]}],
        "state": ["x[g]", "tot"],
        "equations": {
            "x[g]": "0.0",
            "tot": "apply_along(x[g:i], g=i, kernel=sum)",
        },
    }
    out = normalize_expr_rhs(spec)
    eq_tot = out.equations[-1]
    assert "x__g_a" in eq_tot
    assert "x__g_b" in eq_tot


def test_apply_along_explicit_kernel_integrate_on_categorical_axis_errors() -> None:
    """kernel=integrate on a categorical axis should raise."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "g", "coords": ["a", "b"]}],
        "state": ["x[g]", "tot"],
        "equations": {
            "x[g]": "0.0",
            "tot": "apply_along(x[g:i], g=i, kernel=integrate)",
        },
    }
    with pytest.raises(ValueError, match=r"requires continuous axes"):
        normalize_expr_rhs(spec)


def test_apply_along_explicit_kernel_sum_on_continuous_axis_errors() -> None:
    """kernel=sum on a continuous axis should raise."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "x", "type": "continuous", "coords": [0.0, 1.0]}],
        "state": ["u[x]", "tot"],
        "equations": {
            "u[x]": "0.0",
            "tot": "apply_along(u[x:i], x=i, kernel=sum)",
        },
    }
    with pytest.raises(ValueError, match=r"requires categorical or ordinal axes"):
        normalize_expr_rhs(spec)


def test_apply_along_mixed_axis_types_without_kernel_errors() -> None:
    """Mixed categorical+continuous axes without explicit kernel should raise."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "g", "coords": ["a", "b"]},
            {"name": "x", "type": "continuous", "coords": [0.0, 1.0]},
        ],
        "state": ["u[g, x]", "tot"],
        "equations": {
            "u[g, x]": "0.0",
            "tot": "apply_along(u[g:i, x:j], g=i, x=j)",
        },
    }
    with pytest.raises(ValueError, match=r"cannot infer kernel"):
        normalize_expr_rhs(spec)


def test_apply_along_requires_at_least_one_axis_binding() -> None:
    """apply_along with no axis bindings should raise."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "g", "coords": ["a", "b"]}],
        "state": ["x[g]"],
        "equations": {"x[g]": "apply_along(x[g])"},
    }
    with pytest.raises(ValueError, match=r"at least one axis=var binding"):
        normalize_expr_rhs(spec)


def test_apply_along_requires_exactly_one_inner_expression() -> None:
    """apply_along with two non-binding args should raise."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "g", "coords": ["a", "b"]}],
        "state": ["x[g]"],
        "equations": {"x[g]": "apply_along(x[g:i], x[g:i], g=i)"},
    }
    with pytest.raises(ValueError, match=r"exactly one inner expression"):
        normalize_expr_rhs(spec)


def test_apply_along_unknown_kernel_errors() -> None:
    """An unknown kernel name should raise with the allowed list."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "g", "coords": ["a", "b"]}],
        "state": ["x[g]"],
        "equations": {"x[g]": "apply_along(x[g:i], g=i, kernel=median)"},
    }
    with pytest.raises(ValueError, match=r"kernel must be one of"):
        normalize_expr_rhs(spec)


def test_apply_along_axis_binding_must_be_identifier() -> None:
    """Axis binding to a non-identifier RHS should raise."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "g", "coords": ["a", "b"]}],
        "state": ["x[g]"],
        "equations": {"x[g]": "apply_along(x[g:i], g=1+2)"},
    }
    with pytest.raises(ValueError, match=r"must bind to an identifier"):
        normalize_expr_rhs(spec)


def test_apply_along_inside_arithmetic_expression() -> None:
    """apply_along should compose with surrounding arithmetic."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1", "p2"]}],
        "state": ["S[pop]", "I[pop]"],
        "equations": {
            "S[pop]": "-beta * S[pop] * apply_along(I[pop:j], pop=j)",
            "I[pop]": "beta * S[pop] * apply_along(I[pop:j], pop=j) - gamma * I[pop]",
        },
    }
    out = normalize_expr_rhs(spec)
    assert any("I__pop_p1" in eq and "I__pop_p2" in eq for eq in out.equations)


def test_apply_along_nested_inside_apply_along() -> None:
    """Nested apply_along calls should expand to canonical-order state names.

    Even though the inner ``apply_along`` binds ``vax`` first and the outer
    binds ``age``, the resulting bracket-collapsed names must follow the
    canonical axis order taken from the spec's ``axes:`` list, so they
    match what ``state: ["I[age, vax]"]`` produces directly.
    """
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
        ],
        "state": ["I[age, vax]", "N"],
        "equations": {
            "I[age, vax]": "0.0",
            "N": "apply_along(apply_along(I[age:a, vax:b], vax=b), age=a)",
        },
    }
    out = normalize_expr_rhs(spec)
    eq_n = out.equations[-1]
    for a in ("y", "o"):
        for v in ("u", "v"):
            assert f"I__age_{a}__vax_{v}" in eq_n
            # The reversed (binding-order) form should not appear.
            assert f"I__vax_{v}__age_{a}" not in eq_n


def test_apply_along_nested_three_axes_canonical_order() -> None:
    """Triple-nested apply_along should still produce canonical-order names."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
            {"name": "imm", "type": "ordinal", "coords": ["x0", "x1"]},
        ],
        "state": ["X[age, vax, imm]", "N"],
        "equations": {
            "X[age, vax, imm]": "0.0",
            # Inner-most binds imm, middle binds vax, outer binds age — the
            # exact pathological pattern that triggered the SMH R19 KeyError.
            "N": (
                "apply_along("
                "apply_along("
                "apply_along(X[age:a, vax:b, imm:k], imm=k), "
                "vax=b), "
                "age=a)"
            ),
        },
    }
    out = normalize_expr_rhs(spec)
    eq_n = out.equations[-1]
    for a in ("y", "o"):
        for v in ("u", "v"):
            for k in ("x0", "x1"):
                assert f"X__age_{a}__vax_{v}__imm_{k}" in eq_n


def test_apply_along_categorical_filter_subsets_coords() -> None:
    """`axis=var in [c1, c2]` should restrict expansion to the listed coords."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "vax", "coords": ["u", "v", "w"]}],
        "state": ["pop[vax]", "covered"],
        "equations": {
            "pop[vax]": "0.0",
            "covered": "apply_along(pop[vax:j], vax=j in [v, w])",
        },
    }
    out = normalize_expr_rhs(spec)
    eq = out.equations[-1]
    assert "pop__vax_v" in eq
    assert "pop__vax_w" in eq
    assert "pop__vax_u" not in eq


def test_apply_along_continuous_filter_recomputes_trapezoidal_deltas() -> None:
    """A `[lo, hi]` filter on a continuous axis should re-trapezoid weights."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "x", "type": "continuous", "coords": [0.0, 1.0, 3.0, 4.0]},
        ],
        "state": ["u[x]", "v"],
        "equations": {
            "u[x]": "0.0",
            "v": "apply_along(u[x:i], x=i in [1.0, 4.0])",
        },
    }
    out = normalize_expr_rhs(spec)
    eq = out.equations[-1]
    # Sub-interval coords [1.0, 3.0, 4.0] -> trapezoidal weights [1.0, 1.5, 0.5]
    assert "u__x_1_0" in eq
    assert "u__x_3_0" in eq
    assert "u__x_4_0" in eq
    assert "u__x_0_0" not in eq
    assert "1.5" in eq  # interior sub-interval weight
    assert "0.5" in eq  # right endpoint sub-interval weight


def test_apply_along_continuous_filter_requires_two_endpoints() -> None:
    """A continuous-axis filter must be exactly `[lo, hi]`."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "x", "type": "continuous", "coords": [0.0, 1.0, 3.0, 4.0]},
        ],
        "state": ["u[x]", "v"],
        "equations": {
            "u[x]": "0.0",
            "v": "apply_along(u[x:i], x=i in [0.0, 1.0, 3.0])",
        },
    }
    with pytest.raises(Exception, match="2-element"):
        normalize_expr_rhs(spec)


def test_apply_along_continuous_filter_empty_subinterval_errors() -> None:
    """A continuous-axis filter that selects no axis coords should be rejected."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "x", "type": "continuous", "coords": [0.0, 1.0, 3.0, 4.0]},
        ],
        "state": ["u[x]", "v"],
        "equations": {
            "u[x]": "0.0",
            "v": "apply_along(u[x:i], x=i in [1.5, 2.5])",
        },
    }
    with pytest.raises(Exception, match="selects no axis coords"):
        normalize_expr_rhs(spec)


def test_apply_along_filter_unknown_coord_errors() -> None:
    """A filter that lists an unknown coord should be rejected."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "vax", "coords": ["u", "v", "w"]}],
        "state": ["pop[vax]", "covered"],
        "equations": {
            "pop[vax]": "0.0",
            "covered": "apply_along(pop[vax:j], vax=j in [v, x])",
        },
    }
    with pytest.raises(Exception, match="unknown coords"):
        normalize_expr_rhs(spec)


def test_apply_along_filter_mixed_with_full_axis_in_multi_binding() -> None:
    """A multi-axis call may filter one axis and leave another unfiltered."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v", "w"]},
        ],
        "state": ["I[age, vax]", "N"],
        "equations": {
            "I[age, vax]": "0.0",
            "N": "apply_along(I[age:a, vax:b], age=a, vax=b in [v, w])",
        },
    }
    out = normalize_expr_rhs(spec)
    eq = out.equations[-1]
    for a in ("y", "o"):
        for v in ("v", "w"):
            assert f"I__age_{a}__vax_{v}" in eq
        assert f"I__age_{a}__vax_u" not in eq


def test_apply_along_ordinal_axis_full_expansion_uses_uniform_weights() -> None:
    """A bound ordinal axis (no filter) should expand like a categorical axis."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "imm", "type": "ordinal", "coords": ["X0", "X1", "X2", "X3"]},
        ],
        "state": ["pop[imm]", "total"],
        "equations": {
            "pop[imm]": "0.0",
            "total": "apply_along(pop[imm:k], imm=k)",
        },
    }
    out = normalize_expr_rhs(spec)
    eq = out.equations[-1]
    for c in ("X0", "X1", "X2", "X3"):
        assert f"pop__imm_{c}" in eq


def test_apply_along_ordinal_filter_inclusive_index_range() -> None:
    """An ordinal `in [lo_label, hi_label]` filter selects the inclusive index range."""
    spec = {
        "kind": "expr",
        "axes": [
            {
                "name": "imm",
                "type": "ordinal",
                "coords": ["X0", "X1", "X2", "X3", "X4"],
            },
        ],
        "state": ["pop[imm]", "covered"],
        "equations": {
            "pop[imm]": "0.0",
            "covered": "apply_along(pop[imm:k], imm=k in [X1, X3])",
        },
    }
    out = normalize_expr_rhs(spec)
    eq = out.equations[-1]
    for c in ("X1", "X2", "X3"):
        assert f"pop__imm_{c}" in eq
    for c in ("X0", "X4"):
        assert f"pop__imm_{c}" not in eq


def test_apply_along_ordinal_filter_requires_two_endpoints() -> None:
    """An ordinal-axis filter must be exactly `[lo_label, hi_label]`."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "imm", "type": "ordinal", "coords": ["X0", "X1", "X2", "X3"]},
        ],
        "state": ["pop[imm]", "covered"],
        "equations": {
            "pop[imm]": "0.0",
            "covered": "apply_along(pop[imm:k], imm=k in [X0, X1, X2])",
        },
    }
    with pytest.raises(Exception, match="2-element"):
        normalize_expr_rhs(spec)


def test_apply_along_ordinal_filter_unknown_endpoint_errors() -> None:
    """An ordinal filter referencing an undeclared coord label is rejected."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "imm", "type": "ordinal", "coords": ["X0", "X1", "X2", "X3"]},
        ],
        "state": ["pop[imm]", "covered"],
        "equations": {
            "pop[imm]": "0.0",
            "covered": "apply_along(pop[imm:k], imm=k in [X1, X9])",
        },
    }
    with pytest.raises(Exception, match="unknown coords"):
        normalize_expr_rhs(spec)


def test_apply_along_ordinal_filter_reversed_endpoints_errors() -> None:
    """An ordinal filter with index(lo) > index(hi) is rejected."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "imm", "type": "ordinal", "coords": ["X0", "X1", "X2", "X3"]},
        ],
        "state": ["pop[imm]", "covered"],
        "equations": {
            "pop[imm]": "0.0",
            "covered": "apply_along(pop[imm:k], imm=k in [X3, X1])",
        },
    }
    with pytest.raises(Exception, match="index"):
        normalize_expr_rhs(spec)
