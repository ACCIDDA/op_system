"""Typed expression IR for op_system.

This module introduces a small, immutable intermediate representation (IR)
for expression terms. The initial scope is parse-only: convert Python AST
expressions into typed IR nodes without changing compile/normalize behavior.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import NoReturn

from ._errors import InvalidExpressionError


@dataclass(frozen=True, slots=True)
class Literal:
    """Literal scalar value in an expression."""

    value: float | int | bool | str


@dataclass(frozen=True, slots=True)
class Sym:
    """Bare identifier symbol (e.g. ``beta``, ``S``, ``np`` root names)."""

    name: str


@dataclass(frozen=True, slots=True)
class AxisIndex:
    """One subscript position for an indexed expression.

    For the initial parse-only phase, ``axis`` stores the token text for
    name-style indices (e.g. ``age`` in ``K[age, ap]``). Literal indices
    (e.g. ``K[0]`` or ``K['x']``) are represented by ``coord``.
    """

    axis: str
    coord: str | None = None
    placeholder: str | None = None


@dataclass(frozen=True, slots=True)
class Subscript:
    """Indexed reference (e.g. ``K[age, ap]``)."""

    name: str
    indices: tuple[AxisIndex, ...]


@dataclass(frozen=True, slots=True)
class Apply:
    """Generic operation/call node.

    The ``op`` field stores either an operator token (e.g. ``+``, ``*``) or
    function name (e.g. ``np.exp``, ``sum_state``).
    """

    op: str
    args: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Reduce:
    """Reduction primitive placeholder for future helper-lowering passes."""

    kind: str
    bindings: tuple[tuple[str, str], ...]
    body: Expr


Expr = Literal | Sym | Subscript | Apply | Reduce


_BIN_OP_NAMES: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "pow",
    ast.Mod: "%",
}

_CMP_OP_NAMES: dict[type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def _invalid(*, detail: str) -> NoReturn:
    raise InvalidExpressionError(detail=detail)


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = [func.attr]
        cur: ast.AST = func.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    _invalid(detail="unsupported call target in IR parser")


def _axis_index_from_expr(node: ast.expr) -> AxisIndex:
    if isinstance(node, ast.Name):
        if node.id.startswith("$"):
            ph = node.id[1:]
            return AxisIndex(axis=ph, placeholder=ph)
        return AxisIndex(axis=node.id)
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int, float, bool)
    ):
        return AxisIndex(axis="", coord=str(node.value))
    _invalid(detail=f"unsupported subscript index node: {type(node).__name__}")


def _parse_subscript_indices(slc: ast.expr) -> tuple[AxisIndex, ...]:
    if isinstance(slc, ast.Tuple):
        return tuple(_axis_index_from_expr(elt) for elt in slc.elts)
    return (_axis_index_from_expr(slc),)


def to_ir(node: ast.AST) -> Expr:  # noqa: C901, PLR0911, PLR0912
    """Convert a Python AST node (expression) into typed IR.

    Args:
        node: ``ast.Expression`` or ``ast.expr`` node.

    Returns:
        Parsed IR expression.

    """
    if isinstance(node, ast.Expression):
        return to_ir(node.body)

    if isinstance(node, ast.Name):
        return Sym(name=node.id)

    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int, float, bool)
    ):
        return Literal(value=node.value)

    if isinstance(node, ast.BinOp):
        op = _BIN_OP_NAMES.get(type(node.op))
        if op is None:
            _invalid(detail=f"unsupported binary operator: {type(node.op).__name__}")
        return Apply(op=op, args=(to_ir(node.left), to_ir(node.right)))

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return Apply(op="neg", args=(to_ir(node.operand),))
        if isinstance(node.op, ast.UAdd):
            return Apply(op="pos", args=(to_ir(node.operand),))
        _invalid(detail=f"unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BoolOp):
        op = "and" if isinstance(node.op, ast.And) else "or"
        return Apply(op=op, args=tuple(to_ir(v) for v in node.values))

    if isinstance(node, ast.Compare):
        if len(node.ops) != len(node.comparators):
            _invalid(detail="malformed comparison node")
        left = node.left
        parts: list[Expr] = []
        for op_node, right in zip(node.ops, node.comparators, strict=True):
            op = _CMP_OP_NAMES.get(type(op_node))
            if op is None:
                _invalid(
                    detail=f"unsupported comparison operator: {type(op_node).__name__}"
                )
            parts.append(Apply(op=op, args=(to_ir(left), to_ir(right))))
            left = right
        if len(parts) == 1:
            return parts[0]
        return Apply(op="and", args=tuple(parts))

    if isinstance(node, ast.IfExp):
        return Apply(
            op="ifelse",
            args=(to_ir(node.test), to_ir(node.body), to_ir(node.orelse)),
        )

    if isinstance(node, ast.Call):
        return Apply(
            op=_call_name(node.func),
            args=tuple(to_ir(arg) for arg in node.args),
        )

    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name):
            _invalid(detail="subscript base must be a simple name")
        return Subscript(
            name=node.value.id,
            indices=_parse_subscript_indices(node.slice),
        )

    _invalid(detail=f"unsupported AST node in IR parser: {type(node).__name__}")


def parse_expr_to_ir(expr: str) -> Expr:
    """Parse an expression string and convert it to typed IR.

    Args:
        expr: Expression source string.

    Returns:
        Parsed IR tree.

    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        _invalid(detail=f"invalid expression syntax: {exc.msg}")
    return to_ir(tree)


__all__ = [
    "Apply",
    "AxisIndex",
    "Expr",
    "Literal",
    "Reduce",
    "Subscript",
    "Sym",
    "parse_expr_to_ir",
    "to_ir",
]
