"""Tests for IR-level template expansion (issue #112)."""

from __future__ import annotations

from op_system._ir import (
    Apply,
    AxisIndex,
    Literal,
    Reduce,
    Subscript,
    Sym,
    parse_expr_to_ir,
)
from op_system._ir_templates import expand_inline_templates
from op_system._templates import _apply_template_substitutions

AXIS_LOOKUP = {"age": ["0_5", "5_18", "18_65", "65p"], "imm": ["x0", "x1"]}


def test_expand_renders_named_placeholder_to_sym() -> None:
    """A bare placeholder subscript becomes a Sym with the rendered name."""
    expr = Subscript(name="S", indices=(AxisIndex(axis="age"),))
    out = expand_inline_templates(expr, assignment={"age": "0_5"})
    assert out == Sym(name="S__age_0_5")


def test_expand_renders_multi_axis_subscript() -> None:
    """Multi-axis subscripts render placeholders in declared order."""
    expr = Subscript(
        name="X",
        indices=(AxisIndex(axis="age"), AxisIndex(axis="pop")),
    )
    out = expand_inline_templates(expr, assignment={"age": "0_5", "pop": "p1"})
    assert out == Sym(name="X__age_0_5__pop_p1")


def test_expand_leaves_subscript_when_axis_not_in_assignment() -> None:
    """Subscripts with unresolved axes are returned unchanged."""
    expr = Subscript(name="S", indices=(AxisIndex(axis="age"),))
    assert expand_inline_templates(expr, assignment={"pop": "p1"}) is expr


def test_expand_leaves_subscript_with_literal_coord() -> None:
    """Subscripts that already carry a literal coord are not eligible."""
    expr = Subscript(
        name="theta",
        indices=(AxisIndex(axis="", coord="0"),),
    )
    assert expand_inline_templates(expr, assignment={"age": "0_5"}) is expr


def test_expand_leaves_subscript_with_placeholder() -> None:
    """``$``-style placeholders are preserved, not expanded."""
    expr = Subscript(
        name="C",
        indices=(AxisIndex(axis="age", placeholder="age"),),
    )
    assert expand_inline_templates(expr, assignment={"age": "0_5"}) is expr


def test_expand_rewrites_shaped_param_to_literal_index() -> None:
    """Shaped-param subscripts become literal integer coord indices."""
    expr = Subscript(name="theta", indices=(AxisIndex(axis="imm"),))
    out = expand_inline_templates(
        expr,
        assignment={"imm": "x1"},
        shaped_params={"theta": ("imm",)},
        axis_lookup=AXIS_LOOKUP,
    )
    assert out == Subscript(name="theta", indices=(AxisIndex(axis="", coord="1"),))


def test_expand_shaped_param_axis_mismatch_falls_through() -> None:
    """If axes don't match the registered order, the subscript is unchanged."""
    expr = Subscript(
        name="theta",
        indices=(AxisIndex(axis="age"), AxisIndex(axis="imm")),
    )
    out = expand_inline_templates(
        expr,
        assignment={"age": "0_5", "imm": "x1"},
        shaped_params={"theta": ("imm",)},
        axis_lookup=AXIS_LOOKUP,
    )
    assert out is expr


def test_expand_recurses_into_apply() -> None:
    """Nested Apply args are expanded recursively."""
    expr = Apply(
        op="*",
        args=(
            Subscript(name="S", indices=(AxisIndex(axis="age"),)),
            Literal(value=2.0),
        ),
    )
    out = expand_inline_templates(expr, assignment={"age": "0_5"})
    assert out == Apply(op="*", args=(Sym(name="S__age_0_5"), Literal(value=2.0)))


def test_expand_respects_reduce_binding_shadowing() -> None:
    """A Reduce binding shadows the outer assignment for its bound name."""
    body = Subscript(
        name="C",
        indices=(AxisIndex(axis="age"), AxisIndex(axis="ap")),
    )
    expr = Reduce(kind="sum_over", bindings=(("age", "ap"),), body=body)
    out = expand_inline_templates(expr, assignment={"age": "0_5", "ap": "1_2"})
    # "ap" is bound -> not substituted; "age" only -> not all axes covered
    # so the subscript is left intact.
    assert out is expr


def test_expand_parity_with_string_path_named_only() -> None:
    """IR expansion matches the string-based expansion (named-only case)."""
    src = "beta * S[age] + I[age] * 0.5"
    assignment = {"age": "0_5"}

    string_out = _apply_template_substitutions(
        src, assignment=assignment, template_map={}
    )
    ir_out = expand_inline_templates(parse_expr_to_ir(src), assignment=assignment)

    # Parse the string-path result and compare structurally.
    assert ir_out == parse_expr_to_ir(string_out)


def test_expand_parity_with_string_path_shaped_param() -> None:
    """IR expansion matches the string-based expansion (shaped-param case)."""
    src = "theta[imm] * S[age]"
    assignment = {"age": "0_5", "imm": "x1"}
    shaped = {"theta": ("imm",)}

    string_out = _apply_template_substitutions(
        src,
        assignment=assignment,
        template_map={},
        shaped_params=shaped,
        axis_lookup=AXIS_LOOKUP,
    )
    ir_out = expand_inline_templates(
        parse_expr_to_ir(src),
        assignment=assignment,
        shaped_params=shaped,
        axis_lookup=AXIS_LOOKUP,
    )

    assert ir_out == parse_expr_to_ir(string_out)
