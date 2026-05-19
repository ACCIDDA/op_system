"""Tests for IR-level template expansion (issue #112)."""

from __future__ import annotations

import pytest

from op_system._ir import (
    Apply,
    AxisIndex,
    Literal,
    Reduce,
    Subscript,
    Sym,
    parse_expr_to_ir,
)
from op_system._ir_templates import (
    expand_inline_templates,
    expand_over_axes,
    inline_aliases,
    placeholder_axes,
)
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


def test_placeholder_axes_collects_in_first_seen_order() -> None:
    """Placeholder axes are returned deterministically by walk order."""
    expr = parse_expr_to_ir("S[age] * theta[imm] + I[pop] * S[age]")
    assert placeholder_axes(expr) == ("age", "imm", "pop")


def test_placeholder_axes_skips_shaped_param_subscripts() -> None:
    """Subscripts whose base is a shaped param do not contribute axes."""
    expr = parse_expr_to_ir("theta[imm] * S[age]")
    out = placeholder_axes(expr, shaped_params={"theta": ("imm",)})
    assert out == ("age",)


def test_placeholder_axes_ignores_literal_coords_and_placeholders() -> None:
    """Literal coords and ``$``-placeholders are not collected."""
    expr = Subscript(
        name="K",
        indices=(
            AxisIndex(axis="", coord="0"),
            AxisIndex(axis="age", placeholder="age"),
        ),
    )
    assert placeholder_axes(expr) == ()


def test_placeholder_axes_drops_reduce_bound_names() -> None:
    """Names bound by a Reduce do not appear in the result set."""
    body = Subscript(name="C", indices=(AxisIndex(axis="age"), AxisIndex(axis="ap")))
    expr = Reduce(kind="sum_over", bindings=(("age", "ap"),), body=body)
    # "age" is still a free placeholder; "ap" is bound and dropped.
    assert placeholder_axes(expr) == ("age",)


def test_expand_over_axes_empty_axes_returns_singleton() -> None:
    """With no axes, the input is returned as a single ({}, expr) entry."""
    expr = parse_expr_to_ir("beta * S")
    out = expand_over_axes(expr, axes=(), axis_lookup=AXIS_LOOKUP)
    assert out == [({}, expr)]


def test_expand_over_axes_cross_product_count() -> None:
    """The result length equals the product of coord-list lengths."""
    expr = parse_expr_to_ir("S[age] * theta[imm]")
    out = expand_over_axes(
        expr,
        axes=("age", "imm"),
        axis_lookup=AXIS_LOOKUP,
        shaped_params={"theta": ("imm",)},
    )
    assert len(out) == len(AXIS_LOOKUP["age"]) * len(AXIS_LOOKUP["imm"])


def test_expand_over_axes_matches_string_path_named_only() -> None:
    """Each IR expansion matches parsing the string-path expansion."""
    src = "beta * S[age] + I[age]"
    expr = parse_expr_to_ir(src)
    axes = placeholder_axes(expr)
    ir_results = expand_over_axes(expr, axes=axes, axis_lookup=AXIS_LOOKUP)

    for assignment, ir_expr in ir_results:
        string_expr = _apply_template_substitutions(
            src, assignment=assignment, template_map={}
        )
        assert ir_expr == parse_expr_to_ir(string_expr)


def test_expand_over_axes_matches_string_path_with_shaped_param() -> None:
    """Cross-product expansion matches the string path for shaped params."""
    src = "theta[imm] * S[age]"
    shaped = {"theta": ("imm",)}
    expr = parse_expr_to_ir(src)
    axes = placeholder_axes(expr, shaped_params=shaped)
    ir_results = expand_over_axes(
        expr,
        axes=("age", "imm"),  # include imm so theta gets rewritten
        axis_lookup=AXIS_LOOKUP,
        shaped_params=shaped,
    )

    assert axes == ("age",)
    for assignment, ir_expr in ir_results:
        string_expr = _apply_template_substitutions(
            src,
            assignment=assignment,
            template_map={},
            shaped_params=shaped,
            axis_lookup=AXIS_LOOKUP,
        )
        assert ir_expr == parse_expr_to_ir(string_expr)


def test_inline_aliases_basic_substitution() -> None:
    """A single alias reference is replaced with its body."""
    expr = parse_expr_to_ir("rho * S")
    aliases = {"rho": parse_expr_to_ir("beta * I")}
    out = inline_aliases(expr, aliases)
    assert out == parse_expr_to_ir("(beta * I) * S")


def test_inline_aliases_chained_references() -> None:
    """Aliases referencing other aliases fully expand to leaves."""
    aliases = {
        "rho": parse_expr_to_ir("beta * gamma"),
        "force": parse_expr_to_ir("rho * I"),
    }
    out = inline_aliases(parse_expr_to_ir("force * S"), aliases)
    assert out == parse_expr_to_ir("((beta * gamma) * I) * S")


def test_inline_aliases_returns_input_when_no_aliases() -> None:
    """An empty alias map returns the original expression unchanged."""
    expr = parse_expr_to_ir("a + b")
    assert inline_aliases(expr, {}) is expr


def test_inline_aliases_leaves_unrelated_symbols() -> None:
    """Symbols not in the alias map are preserved."""
    expr = parse_expr_to_ir("alpha + rho")
    out = inline_aliases(expr, {"rho": Literal(value=0.5)})
    assert out == Apply(op="+", args=(Sym(name="alpha"), Literal(value=0.5)))


def test_inline_aliases_rejects_self_cycle() -> None:
    """A self-referential alias raises immediately."""
    aliases = {"rho": parse_expr_to_ir("rho + 1")}
    with pytest.raises(ValueError, match="alias cycle"):
        inline_aliases(parse_expr_to_ir("rho"), aliases)


def test_inline_aliases_rejects_mutual_cycle() -> None:
    """Mutually recursive aliases raise."""
    aliases = {
        "a": parse_expr_to_ir("b + 1"),
        "b": parse_expr_to_ir("a * 2"),
    }
    with pytest.raises(ValueError, match="alias cycle"):
        inline_aliases(parse_expr_to_ir("a"), aliases)


def test_inline_aliases_respects_reduce_binding_shadowing() -> None:
    """An alias name shadowed by a Reduce binding is not substituted."""
    body = Apply(op="*", args=(Sym(name="rho"), Sym(name="S")))
    expr = Reduce(kind="sum_over", bindings=(("age", "rho"),), body=body)
    aliases = {"rho": Literal(value=0.5)}
    out = inline_aliases(expr, aliases)
    # The Reduce shadows "rho" -> body's "rho" stays a Sym.
    assert isinstance(out, Reduce)
    assert out.body == body


# ---------------------------------------------------------------------------
# Alias expansion IR: expand_over_axes + inline_aliases pipeline
# ---------------------------------------------------------------------------


def test_alias_template_expands_to_one_ir_per_coord() -> None:
    """expand_over_axes on a templated alias body yields one entry per coord."""
    src = "b0 * S[age]"
    expr = parse_expr_to_ir(src)
    axes = placeholder_axes(expr)
    results = expand_over_axes(expr, axes=axes, axis_lookup=AXIS_LOOKUP)

    assert axes == ("age",)
    assert len(results) == len(AXIS_LOOKUP["age"])
    for assignment, ir_expr in results:
        coord = assignment["age"]
        assert ir_expr == parse_expr_to_ir(f"b0 * S__age_{coord}")


def test_alias_pipeline_inlines_cross_alias_after_template_expansion() -> None:
    """After expand_over_axes, inline_aliases resolves chained alias refs."""
    # Suppose: k[age] = k_base * 2,  beta[age] = b0 * k[age]
    # After expansion over age=["0_5", "5_18", "18_65", "65p"]:
    #   k__age_<c>  maps to  k_base * 2
    #   beta__age_<c> = b0 * k[age] -> after inline -> b0 * (k_base * 2)
    k_body = parse_expr_to_ir("k_base * 2")
    aliases_ir = {f"k__age_{c}": k_body for c in AXIS_LOOKUP["age"]}

    src = "b0 * k[age]"
    expr = parse_expr_to_ir(src)
    for _assignment, ir_expr in expand_over_axes(
        expr, axes=("age",), axis_lookup=AXIS_LOOKUP
    ):
        inlined = inline_aliases(ir_expr, aliases_ir)
        assert inlined == parse_expr_to_ir("b0 * (k_base * 2)")


def test_alias_template_expansion_matches_string_path() -> None:
    """expand_over_axes on a templated alias matches parse of string expansion."""
    src = "b0 * S[age] + offset[imm]"
    shaped = {"offset": ("imm",)}
    expr = parse_expr_to_ir(src)
    ir_results = expand_over_axes(
        expr,
        axes=("age", "imm"),
        axis_lookup=AXIS_LOOKUP,
        shaped_params=shaped,
    )

    for assignment, ir_expr in ir_results:
        string_expr = _apply_template_substitutions(
            src,
            assignment=assignment,
            template_map={},
            shaped_params=shaped,
            axis_lookup=AXIS_LOOKUP,
        )
        assert ir_expr == parse_expr_to_ir(string_expr)
