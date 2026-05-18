"""Tests for the typed expression IR parser foundation (issue #112)."""

from __future__ import annotations

import pytest

from op_system._errors import InvalidExpressionError
from op_system._ir import (
    Apply,
    AxisIndex,
    Literal,
    Reduce,
    Subscript,
    Sym,
    extract_common_subexpressions,
    lower_helper_calls,
    parse_expr_to_ir,
)


def test_parse_expr_to_ir_arithmetic_tree_shape() -> None:
    """Arithmetic expressions parse into nested ``Apply`` IR nodes."""
    expr = "beta * S * I / N - gamma * I"
    ir = parse_expr_to_ir(expr)

    assert isinstance(ir, Apply)
    assert ir.op == "-"
    left, right = ir.args
    assert isinstance(left, Apply)
    assert left.op == "/"
    assert isinstance(right, Apply)
    assert right.op == "*"


def test_parse_expr_to_ir_subscript_with_symbolic_indices() -> None:
    """Subscript expressions parse into ``Subscript`` and ``AxisIndex`` nodes."""
    ir = parse_expr_to_ir("K[age, ap]")

    assert isinstance(ir, Subscript)
    assert ir.name == "K"
    assert ir.indices == (AxisIndex(axis="age"), AxisIndex(axis="ap"))


def test_parse_expr_to_ir_function_calls_and_literals() -> None:
    """Calls and literals are represented as ``Apply``/``Literal`` nodes."""
    ir = parse_expr_to_ir("np.maximum(x, 0.0)")

    assert isinstance(ir, Apply)
    assert ir.op == "np.maximum"
    assert len(ir.args) == 2
    assert isinstance(ir.args[0], Sym)
    assert isinstance(ir.args[1], Literal)


def test_parse_expr_to_ir_rejects_invalid_syntax() -> None:
    """Invalid expression syntax raises ``InvalidExpressionError``."""
    with pytest.raises(InvalidExpressionError, match="invalid expression syntax"):
        parse_expr_to_ir("beta **")


def test_parse_expr_to_ir_rejects_unsupported_nodes() -> None:
    """Unsupported AST constructs raise ``InvalidExpressionError``."""
    with pytest.raises(InvalidExpressionError, match="unsupported"):
        parse_expr_to_ir("(lambda x: x)(1)")


def test_parse_expr_to_ir_captures_call_kwargs_as_ir_nodes() -> None:
    """Call kwargs are preserved in IR for later lowering passes."""
    ir = parse_expr_to_ir("apply_along(I[age], age=ap)")

    assert isinstance(ir, Apply)
    assert ir.op == "apply_along"
    assert len(ir.args) == 2
    assert isinstance(ir.args[0], Subscript)
    assert isinstance(ir.args[1], Apply)
    assert ir.args[1].op == "kwarg"


def test_helper_lowering_rewrites_apply_along_to_reduce() -> None:
    """Helper lowering converts apply_along call into a Reduce node."""
    ir = parse_expr_to_ir("apply_along(I[age], age=ap)", lower_helpers=True)

    assert isinstance(ir, Reduce)
    assert ir.kind == "apply_along"
    assert ir.bindings == (("age", "ap"),)
    assert isinstance(ir.body, Subscript)


def test_lower_helper_calls_preserves_non_helper_apply() -> None:
    """Lowering leaves non-helper Apply nodes unchanged."""
    ir = parse_expr_to_ir("np.maximum(x, 0.0)")
    lowered = lower_helper_calls(ir)

    assert isinstance(lowered, Apply)
    assert lowered.op == "np.maximum"


def test_extract_common_subexpressions_reuses_repeated_apply() -> None:
    """Repeated non-leaf subexpressions become generated symbol bindings."""
    bindings, rewritten = extract_common_subexpressions((
        parse_expr_to_ir("(a + b) * (a + b)"),
        parse_expr_to_ir("c + (a + b)"),
    ))

    assert bindings == (("_cse0", parse_expr_to_ir("a + b")),)
    assert rewritten == (
        parse_expr_to_ir("_cse0 * _cse0"),
        parse_expr_to_ir("c + _cse0"),
    )


def test_extract_common_subexpressions_orders_children_before_parents() -> None:
    """Nested CSE bindings should be emitted child-before-parent."""
    repeated = "(a + b) * (a + b)"
    bindings, rewritten = extract_common_subexpressions((
        parse_expr_to_ir(f"({repeated}) + ({repeated})"),
    ))

    assert bindings == (
        ("_cse0", parse_expr_to_ir("a + b")),
        ("_cse1", parse_expr_to_ir("_cse0 * _cse0")),
    )
    assert rewritten == (parse_expr_to_ir("_cse1 + _cse1"),)


def test_extract_common_subexpressions_ignores_repeated_leaves() -> None:
    """Bare symbol repetition alone should not introduce temporaries."""
    expr = parse_expr_to_ir("a + a")
    bindings, rewritten = extract_common_subexpressions((expr,))

    assert bindings == ()
    assert rewritten == (expr,)
