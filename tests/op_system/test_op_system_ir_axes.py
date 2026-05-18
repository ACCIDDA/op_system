"""Tests for IR axis-resolution helpers (issue #112, paves the way for #88)."""

from __future__ import annotations

from op_system._ir import (
    Apply,
    AxisIndex,
    AxisKind,
    Literal,
    Reduce,
    Subscript,
    Sym,
    axis_kinds,
    classify_axis_index,
    iter_subscripts,
    parse_expr_to_ir,
)

AXES = frozenset({"age", "loc"})


def test_classify_free_axis() -> None:
    """A bare known axis identifier classifies as FREE."""
    assert classify_axis_index(AxisIndex(axis="age"), axis_names=AXES) == AxisKind.FREE


def test_classify_literal_coord() -> None:
    """A literal coord classifies as COORD regardless of axis name."""
    idx = AxisIndex(axis="", coord="0")
    assert classify_axis_index(idx, axis_names=AXES) == AxisKind.COORD


def test_classify_placeholder() -> None:
    """A placeholder classifies as PLACEHOLDER and takes priority over axis."""
    idx = AxisIndex(axis="age", placeholder="age")
    assert classify_axis_index(idx, axis_names=AXES) == AxisKind.PLACEHOLDER


def test_classify_unknown_symbol_is_coord_symbol() -> None:
    """An unknown identifier used as a subscript classifies as COORD_SYMBOL."""
    idx = AxisIndex(axis="ap")
    assert classify_axis_index(idx, axis_names=AXES) == AxisKind.COORD_SYMBOL


def test_iter_subscripts_walks_apply_and_reduce() -> None:
    """``iter_subscripts`` visits subscripts under Apply and Reduce nodes."""
    inner = Subscript(name="I", indices=(AxisIndex(axis="age"),))
    outer = Subscript(name="K", indices=(AxisIndex(axis="age"), AxisIndex(axis="ap")))
    expr = Apply(
        op="+",
        args=(
            outer,
            Reduce(kind="sum_over", bindings=(("age", "age"),), body=inner),
        ),
    )

    names = [s.name for s in iter_subscripts(expr)]
    assert names == ["K", "I"]


def test_axis_kinds_for_parsed_expression() -> None:
    """``axis_kinds`` classifies all positions in walk order."""
    expr = parse_expr_to_ir("K[age, ap] + I[age]")

    assert axis_kinds(expr, axis_names=AXES) == (
        AxisKind.FREE,
        AxisKind.COORD_SYMBOL,
        AxisKind.FREE,
    )


def test_axis_kinds_handles_placeholder_dollar_syntax() -> None:
    """``$``-prefixed indices are classified as PLACEHOLDER."""
    expr = Subscript(
        name="C",
        indices=(
            AxisIndex(axis="age"),
            AxisIndex(axis="age", placeholder="age"),
        ),
    )

    assert axis_kinds(expr, axis_names=AXES) == (
        AxisKind.FREE,
        AxisKind.PLACEHOLDER,
    )


def test_iter_subscripts_ignores_non_subscript_leaves() -> None:
    """Literal and Sym leaves yield no subscripts."""
    expr = Apply(op="*", args=(Literal(value=2.0), Sym(name="beta")))
    assert list(iter_subscripts(expr)) == []
