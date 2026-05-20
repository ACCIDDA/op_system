"""Tests for ``resolve_axis_kinds`` (issue #112)."""

from __future__ import annotations

from op_system._ir import (
    Apply,
    AxisIndex,
    AxisKind,
    Reduce,
    Subscript,
    Sym,
    parse_expr_to_ir,
    resolve_axis_kinds,
    unparse_ir,
)

AXES = frozenset({"age", "loc"})


def test_resolve_axis_kinds_sets_kind_on_every_index() -> None:
    """Each AxisIndex in a parsed expression gets its kind populated."""
    expr = parse_expr_to_ir("K[age, ap] + I[age]")
    out = resolve_axis_kinds(expr, axis_names=AXES)

    assert isinstance(out, Apply)
    k_sub, i_sub = out.args
    assert isinstance(k_sub, Subscript)
    assert isinstance(i_sub, Subscript)

    assert tuple(i.kind for i in k_sub.indices) == (
        AxisKind.FREE,
        AxisKind.COORD_SYMBOL,
    )
    assert tuple(i.kind for i in i_sub.indices) == (AxisKind.FREE,)


def test_resolve_axis_kinds_recurses_into_reduce_body() -> None:
    """Indices inside Reduce bodies are resolved too."""
    body = Subscript(name="K", indices=(AxisIndex(axis="age"),))
    expr = Reduce(kind="sum_over", bindings=(("age", "age"),), body=body)

    out = resolve_axis_kinds(expr, axis_names=AXES)

    assert isinstance(out, Reduce)
    assert isinstance(out.body, Subscript)
    assert out.body.indices[0].kind == AxisKind.FREE


def test_resolve_axis_kinds_is_idempotent() -> None:
    """A second resolution returns the same object."""
    expr = parse_expr_to_ir("K[age, 0]")
    once = resolve_axis_kinds(expr, axis_names=AXES)
    twice = resolve_axis_kinds(once, axis_names=AXES)

    assert twice is once


def test_resolve_axis_kinds_preserves_leaves() -> None:
    """Literal and Sym leaves are returned unchanged."""
    expr = Sym(name="beta")
    assert resolve_axis_kinds(expr, axis_names=AXES) is expr


def test_resolve_axis_kinds_does_not_change_unparse_output() -> None:
    """Round-trip through the unparser is unaffected by resolution."""
    expr = parse_expr_to_ir("K[age, ap] * I[age]")
    out = resolve_axis_kinds(expr, axis_names=AXES)

    assert unparse_ir(expr) == unparse_ir(out)
