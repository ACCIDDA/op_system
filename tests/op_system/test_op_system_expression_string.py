"""Tests for ExpressionString parse wrapper (issue #22 groundwork)."""

from __future__ import annotations

import ast

import pytest

from op_system._errors import InvalidExpressionError
from op_system._ir import Apply, Reduce, Subscript
from op_system._symbols import (
    ExpressionString,
    _collect_names,
    _parse_expr,
    parse_expression_string,
)
from op_system.specs import ExpressionString as ExpressionStringFromSpecs


def test_expression_string_exposes_ast_and_names() -> None:
    """ExpressionString parses once and provides AST plus symbol set."""
    expr = ExpressionString("beta * S * I / N - gamma * I")

    assert isinstance(expr.ast, ast.AST)
    assert expr.names == frozenset({"beta", "S", "I", "N", "gamma"})


def test_expression_string_invalid_syntax_raises() -> None:
    """Invalid syntax raises InvalidExpressionError."""
    with pytest.raises(InvalidExpressionError, match="invalid expression syntax"):
        ExpressionString("beta **")


def test_collect_names_accepts_expression_string_wrapper() -> None:
    """_collect_names supports ExpressionString without re-walking callers."""
    expr = parse_expression_string("x + y + z")

    assert _collect_names(expr) == {"x", "y", "z"}


def test_parse_expr_backwards_compatible_with_ast() -> None:
    """Legacy _parse_expr API still returns an AST object."""
    tree = _parse_expr("a + b")

    assert isinstance(tree, ast.AST)
    assert _collect_names(tree) == {"a", "b"}


def test_expression_string_is_exported_from_specs() -> None:
    """ExpressionString is available from the public specs facade."""
    expr = ExpressionStringFromSpecs("r0 * I")

    assert expr.names == frozenset({"r0", "I"})


def test_expression_string_as_ir_arithmetic() -> None:
    """ExpressionString can project to the typed IR representation."""
    expr = ExpressionString("beta * I")

    ir = expr.as_ir()
    assert isinstance(ir, Apply)
    assert ir.op == "*"


def test_expression_string_as_ir_subscript() -> None:
    """IR projection preserves subscript structure for indexed expressions."""
    expr = ExpressionString("K[age, ap]")

    ir = expr.as_ir()
    assert isinstance(ir, Subscript)
    assert ir.name == "K"


def test_expression_string_as_lowered_ir_helpers() -> None:
    """ExpressionString can project directly to helper-lowered IR."""
    expr = ExpressionString("apply_along(I[age], age=ap)")

    ir = expr.as_lowered_ir()
    assert isinstance(ir, Reduce)
    assert ir.kind == "apply_along"
    assert ir.bindings == (("age", "ap"),)
