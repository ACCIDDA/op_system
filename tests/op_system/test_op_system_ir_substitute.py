"""Tests for IR traversal / substitution helpers (issue #112)."""

from __future__ import annotations

from op_system._ir import (
    Apply,
    AxisIndex,
    Literal,
    Reduce,
    Subscript,
    Sym,
    free_symbols,
    parse_expr_to_ir,
    substitute,
    walk,
)


def test_walk_yields_all_nodes_in_preorder() -> None:
    """``walk`` visits root first, then children."""
    expr = Apply(op="+", args=(Sym(name="a"), Literal(value=1)))
    nodes = list(walk(expr))
    assert nodes[0] is expr
    assert Sym(name="a") in nodes
    assert Literal(value=1) in nodes


def test_walk_descends_into_reduce_body() -> None:
    """``walk`` enters Reduce bodies."""
    body = Apply(op="*", args=(Sym(name="x"), Sym(name="y")))
    expr = Reduce(kind="sum_over", bindings=(("age", "age"),), body=body)
    names = {n.name for n in walk(expr) if isinstance(n, Sym)}
    assert names == {"x", "y"}


def test_free_symbols_collects_sym_names() -> None:
    """``free_symbols`` returns every Sym name reachable from the root."""
    expr = parse_expr_to_ir("a * b + c")
    assert free_symbols(expr) == frozenset({"a", "b", "c"})


def test_free_symbols_ignores_subscript_axes() -> None:
    """Subscript axis tokens are not Sym leaves."""
    expr = Subscript(name="K", indices=(AxisIndex(axis="age"),))
    assert free_symbols(expr) == frozenset()


def test_free_symbols_respects_reduce_binding_shadowing() -> None:
    """Bound names inside a Reduce body do not appear as free."""
    body = Apply(op="*", args=(Sym(name="ap"), Sym(name="beta")))
    expr = Reduce(kind="sum_over", bindings=(("age", "ap"),), body=body)
    assert free_symbols(expr) == frozenset({"beta"})


def test_substitute_replaces_matching_sym() -> None:
    """``substitute`` swaps Sym leaves by name."""
    expr = Apply(op="+", args=(Sym(name="a"), Sym(name="b")))
    out = substitute(expr, {"a": Literal(value=2.0)})
    assert out == Apply(op="+", args=(Literal(value=2.0), Sym(name="b")))


def test_substitute_leaves_unmapped_symbols() -> None:
    """Symbols absent from the mapping are returned unchanged."""
    expr = Sym(name="z")
    assert substitute(expr, {"a": Literal(value=1.0)}) is expr


def test_substitute_recurses_into_apply_and_reduce_body() -> None:
    """``substitute`` rewrites nested Apply and Reduce subtrees."""
    body = Apply(op="*", args=(Sym(name="x"), Sym(name="beta")))
    expr = Reduce(kind="sum_over", bindings=(("age", "age"),), body=body)
    out = substitute(expr, {"beta": Literal(value=0.5)})
    assert isinstance(out, Reduce)
    assert out.body == Apply(op="*", args=(Sym(name="x"), Literal(value=0.5)))


def test_substitute_does_not_capture_bound_names() -> None:
    """A mapping for a Reduce-bound name is shadowed inside the body."""
    body = Apply(op="*", args=(Sym(name="ap"), Sym(name="beta")))
    expr = Reduce(kind="sum_over", bindings=(("age", "ap"),), body=body)

    out = substitute(
        expr,
        {"ap": Literal(value=99.0), "beta": Literal(value=0.5)},
    )

    assert isinstance(out, Reduce)
    assert out.body == Apply(op="*", args=(Sym(name="ap"), Literal(value=0.5)))


def test_substitute_is_identity_when_no_match() -> None:
    """Substituting an empty mapping yields a structurally equal tree."""
    expr = parse_expr_to_ir("a + b * c")
    assert substitute(expr, {}) == expr
