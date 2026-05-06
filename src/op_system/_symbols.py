"""op_system._symbols.

AST-level symbol extraction utilities used by expression normalization.
"""

from __future__ import annotations

import ast

from op_system._errors import _raise_invalid_expression


def _parse_expr(expr: str) -> ast.AST:
    """Parse a Python expression string and return the AST node.

    Returns:
        The parsed AST node.
    """
    try:
        return ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        _raise_invalid_expression(detail=f"invalid expression syntax: {exc.msg}")


def _collect_names(tree: ast.AST) -> set[str]:
    """Collect all Name identifiers referenced in an expression AST.

    Returns:
        Set of identifier names found in the expression.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(str(node.id))
    return names
