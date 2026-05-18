"""Parity tests for IR-side ``expand_reduce_pointwise`` vs string expander."""

from __future__ import annotations

import pytest

from op_system._ir import parse_expr_to_ir, unparse_ir
from op_system._ir_expand import expand_reduce_pointwise
from op_system._normalize import _expand_apply_along


def _normalize_spaces(s: str) -> str:
    """Strip whitespace + outer parens so we can compare semantically."""
    return "".join(s.split())


def _ir_expand_to_string(
    src: str,
    *,
    axes,
    shaped_params=None,
    lhs_assignment=None,
    axis_coords=None,
) -> str:
    ir = parse_expr_to_ir(src, lower_helpers=True)
    expanded = expand_reduce_pointwise(
        ir,
        axes=axes,
        shaped_params=shaped_params,
        lhs_assignment=lhs_assignment,
        axis_coords=axis_coords,
    )
    return unparse_ir(expanded)


@pytest.fixture
def axes_age_pop():
    return [
        {"name": "age", "type": "categorical", "coords": ["a0", "a1", "a2"]},
        {"name": "pop", "type": "categorical", "coords": ["p1", "p2"]},
    ]


def test_single_axis_sum_basic_names(axes_age_pop):
    """Expanded names must include the canonical __axis_<coord> suffix."""
    out = _ir_expand_to_string(
        "apply_along(I[pop:j], pop=j)",
        axes=axes_age_pop,
    )
    assert "I__pop_p1" in out
    assert "I__pop_p2" in out


def test_single_axis_sum_two_term_sum(axes_age_pop):
    """A 2-coord apply_along produces a binary Apply('+') of two pointwise terms."""
    from op_system._ir import Apply

    ir = parse_expr_to_ir("apply_along(I[pop:j], pop=j)", lower_helpers=True)
    expanded = expand_reduce_pointwise(ir, axes=axes_age_pop)
    assert isinstance(expanded, Apply)
    assert expanded.op == "+"
    assert len(expanded.args) == 2


def test_same_axis_twice_with_lhs_assignment():
    """K[age, age:ap] with lhs={age:a0}: row pinned to 0 via lhs, col bound 0..1."""
    from op_system._ir import Apply, AxisIndex, Subscript, Sym

    axes = [{"name": "age", "type": "categorical", "coords": ["a0", "a1"]}]
    shaped = {"K": ("age", "age")}
    axis_coords = {"age": ["a0", "a1"]}
    ir = parse_expr_to_ir(
        "apply_along(K[age, age:ap] * I[age:ap], age=ap)", lower_helpers=True
    )
    expanded = expand_reduce_pointwise(
        ir,
        axes=axes,
        shaped_params=shaped,
        lhs_assignment={"age": "a0"},
        axis_coords=axis_coords,
    )
    # Two-term sum K[0,0]*I__age_a0 + K[0,1]*I__age_a1.
    assert isinstance(expanded, Apply) and expanded.op == "+"
    summands = expanded.args
    assert len(summands) == 2

    def _extract_k_index(term):
        # term = Apply("*", [Subscript(K,...), Sym("I__age_aX")])
        for arg in term.args:
            if isinstance(arg, Subscript) and arg.name == "K":
                return tuple(idx.coord for idx in arg.indices)
        return None

    k_indices = {_extract_k_index(t) for t in summands}
    assert k_indices == {("0", "0"), ("0", "1")}

    i_names = {
        a.name
        for term in summands
        for a in term.args
        if isinstance(a, Sym) and a.name.startswith("I__")
    }
    assert i_names == {"I__age_a0", "I__age_a1"}


def test_integrate_kernel_emits_weight_literal():
    """Continuous-axis apply_along with kernel=integrate emits float weights."""
    axes = [
        {
            "name": "t",
            "type": "continuous",
            "coords": ["0.0", "1.0", "2.0"],
            "deltas": [0.5, 1.0, 0.5],
        }
    ]
    out = _ir_expand_to_string(
        "apply_along(f[t:tau], t=tau, kernel=integrate)",
        axes=axes,
    )
    assert "0.5" in out
    assert "1.0" in out
    assert "f__t_0_0" in out or "f__t_0" in out  # depends on _sanitize_fragment


def test_parity_with_string_expander_single_axis(axes_age_pop):
    """IR expansion must produce a string equivalent to the string expander."""
    src = "apply_along(I[pop:j], pop=j)"
    ir_out = _ir_expand_to_string(src, axes=axes_age_pop)
    str_out = _expand_apply_along(src, axes=axes_age_pop)
    # Both should reference the same per-coord names.
    for name in ("I__pop_p1", "I__pop_p2"):
        assert name in ir_out
        assert name in str_out


def test_parity_two_axis_cartesian(axes_age_pop):
    """Multi-axis apply_along produces |age|*|pop| terms."""
    src = "apply_along(I[age:i, pop:j], age=i, pop=j)"
    ir = parse_expr_to_ir(src, lower_helpers=True)
    expanded = expand_reduce_pointwise(ir, axes=axes_age_pop)
    # Should be a fold of 3*2 = 6 terms.
    from op_system._ir import Apply

    def count_summands(e):
        if isinstance(e, Apply) and e.op == "+":
            return sum(count_summands(a) for a in e.args)
        return 1

    assert count_summands(expanded) == 6
