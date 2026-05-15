"""op_system._symbols.

AST-level symbol extraction utilities used by expression normalization.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from op_system._errors import InvalidExpressionError


@dataclass(frozen=True, slots=True)
class ExpressionString:
    """Validated expression wrapper with cached AST and symbol names."""

    source: str
    ast: ast.AST = field(init=False, repr=False)
    names: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        try:
            parsed = ast.parse(self.source, mode="eval")
        except SyntaxError as exc:
            raise InvalidExpressionError(
                detail=f"invalid expression syntax: {exc.msg}"
            ) from exc

        found_names: set[str] = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Name):
                found_names.add(str(node.id))

        object.__setattr__(self, "ast", parsed)
        object.__setattr__(self, "names", frozenset(found_names))


def parse_expression_string(expr: str) -> ExpressionString:
    """Parse ``expr`` into an :class:`ExpressionString`.

    Returns:
        Validated expression wrapper with parsed AST and names.
    """
    return ExpressionString(expr)


def _parse_expr(expr: str) -> ast.AST:
    """Parse a Python expression string and return the AST node.

    Returns:
        The parsed AST node.
    """
    return parse_expression_string(expr).ast


def _collect_names(tree: ast.AST | ExpressionString) -> set[str]:
    """Collect all Name identifiers referenced in an expression AST.

    Returns:
        Set of identifier names found in the expression.
    """
    if isinstance(tree, ExpressionString):
        return set(tree.names)

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(str(node.id))
    return names


__all__ = [
    "ExpressionString",
    "_collect_names",
    "_parse_expr",
    "parse_expression_string",
]
