"""Tests for the typed expression IR parser foundation (issue #112)."""

from __future__ import annotations

import pytest

from op_system._errors import InvalidExpressionError
from op_system._ir import Apply, AxisIndex, Literal, Subscript, Sym, parse_expr_to_ir


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
