"""Typed expression IR for op_system.

This module introduces a small, immutable intermediate representation (IR)
for expression terms. The initial scope is parse-only: convert Python AST
expressions into typed IR nodes without changing compile/normalize behavior.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NoReturn

from ._errors import InvalidExpressionError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence


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

    The optional ``kind`` field is populated by :func:`resolve_axis_kinds`
    once an axis registry is known. Parser output leaves ``kind`` unset.
    """

    axis: str
    coord: str | None = None
    kind: AxisKind | None = None


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
    """Reduction primitive placeholder for future helper-lowering passes.

    ``filters`` records optional per-axis coord subsets from helper
    invocations of the form ``axis=var in [c1, c2, ...]``. Each entry is
    ``(axis_name, (coord1, coord2, ...))``; absence of an entry for an
    axis means the reduction spans the full axis. For continuous axes
    with two numeric coords ``[lo, hi]`` the filter is interpreted as a
    sub-interval to integrate over.

    ``kernel`` records an explicit ``kernel=`` kwarg from
    ``apply_along(..., kernel=sum|integrate)``. ``None`` means auto-select
    based on the bound axis types (the same behavior as the string
    expander).
    """

    kind: str
    bindings: tuple[tuple[str, str], ...]
    body: Expr
    filters: tuple[tuple[str, tuple[str, ...]], ...] = ()
    kernel: str | None = None


@dataclass(frozen=True, slots=True)
class HistoryOp:
    """History-aware helper placeholder (issue #173 scaffolding).

    ``kind`` is one of ``history`` / ``delay`` / ``convolve_history``.
    ``body`` holds the signal expression to query over history and
    ``options`` stores helper keyword arguments as ``(name, Expr)`` pairs.
    """

    kind: str
    body: Expr
    options: tuple[tuple[str, Expr], ...] = ()


Expr = Literal | Sym | Subscript | Apply | Reduce | HistoryOp

_HELPER_REDUCE_OPS: frozenset[str] = frozenset({
    "apply_along",
    "sum_over",
    "integrate_over",
})

_HELPER_HISTORY_OPS: frozenset[str] = frozenset({
    "history",
    "delay",
    "convolve_history",
})


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

_BINARY_OPS: frozenset[str] = frozenset({
    "+",
    "-",
    "*",
    "/",
    "%",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "and",
    "or",
})


def _invalid(*, detail: str) -> NoReturn:
    raise InvalidExpressionError(detail=detail)


def _kwarg_name(arg: Expr) -> str:
    if isinstance(arg, Literal) and isinstance(arg.value, str):
        return arg.value
    _invalid(detail="kwarg marker must use a string key")


def _kwarg_value_as_binding(value: Expr, *, key: str) -> str:
    if isinstance(value, Sym):
        return value.name
    if isinstance(value, Literal):
        return str(value.value)
    _invalid(detail=f"apply_along axis {key!r} must bind to an identifier")


_APPLY_ALONG_KERNELS_IR: frozenset[str] = frozenset({"sum", "integrate"})


def _extract_filter_from_value(value: Expr) -> tuple[str, tuple[str, ...]] | None:
    """If ``value`` is ``var in [c1, c2, ...]``, return ``(var, coords)``.

    The coord list elements may be ``Sym`` (categorical/ordinal labels) or
    ``Literal`` (numeric / string coords for continuous filters). All
    other shapes return ``None`` so the caller falls back to plain
    binding interpretation.

    Returns:
        ``(var_name, coord_strs)`` when the value matches the ``var in
        [...]`` filter shape, otherwise ``None``.
    """
    if not (isinstance(value, Apply) and value.op == "in" and len(value.args) == 2):
        return None
    var_node, list_node = value.args
    if not isinstance(var_node, Sym):
        return None
    if not (isinstance(list_node, Apply) and list_node.op == "list"):
        return None
    coords: list[str] = []
    for elt in list_node.args:
        if isinstance(elt, Sym):
            coords.append(elt.name)
        elif isinstance(elt, Literal):
            coords.append(str(elt.value))
        else:
            return None
    return var_node.name, tuple(coords)


def _lower_single_helper(node: Apply) -> Expr:  # noqa: C901, PLR0912
    if node.op not in _HELPER_REDUCE_OPS | _HELPER_HISTORY_OPS:
        return node

    kw_nodes: list[Apply] = []
    positional: list[Expr] = []
    for arg in node.args:
        if isinstance(arg, Apply) and arg.op == "kwarg":
            kw_nodes.append(arg)
        else:
            positional.append(arg)

    if len(positional) != 1:
        _invalid(detail=f"{node.op} requires exactly one inner expression")

    if node.op in _HELPER_HISTORY_OPS:
        options: list[tuple[str, Expr]] = []
        for kw_node in kw_nodes:
            if len(kw_node.args) != 2:
                _invalid(detail="kwarg marker must contain (key, value)")
            key = _kwarg_name(kw_node.args[0])
            options.append((key, kw_node.args[1]))
        return HistoryOp(kind=node.op, body=positional[0], options=tuple(options))

    bindings: list[tuple[str, str]] = []
    filters: list[tuple[str, tuple[str, ...]]] = []
    kernel: str | None = None
    for kw_node in kw_nodes:
        if len(kw_node.args) != 2:
            _invalid(detail="kwarg marker must contain (key, value)")
        key = _kwarg_name(kw_node.args[0])
        value = kw_node.args[1]
        if key == "kernel":
            # ``kernel=sum`` / ``kernel=integrate`` selects the per-axis form
            # explicitly; recognize it specially so it doesn't get mistaken
            # for an axis binding.
            if isinstance(value, Sym):
                kernel_name = value.name
            elif isinstance(value, Literal):
                kernel_name = str(value.value)
            else:
                _invalid(detail=f"{node.op} kernel= must be 'sum' or 'integrate'")
            if kernel_name not in _APPLY_ALONG_KERNELS_IR:
                _invalid(
                    detail=(
                        f"{node.op} kernel must be one of "
                        f"{sorted(_APPLY_ALONG_KERNELS_IR)}, got {kernel_name!r}"
                    )
                )
            kernel = kernel_name
            continue
        flt = _extract_filter_from_value(value)
        if flt is not None:
            var_name, coords = flt
            bindings.append((key, var_name))
            filters.append((key, coords))
            continue
        val = _kwarg_value_as_binding(value, key=key)
        bindings.append((key, val))

    reduce_node = Reduce(
        kind=node.op,
        bindings=tuple(bindings),
        body=positional[0],
        filters=tuple(filters),
        kernel=kernel,
    )
    _validate_reduce(reduce_node)
    return reduce_node


def _validate_reduce(reduce_node: Reduce) -> None:
    if not reduce_node.bindings:
        _invalid(detail=(f"{reduce_node.kind} requires at least one axis=var binding"))


def lower_helper_calls(expr: Expr) -> Expr:
    """Lower helper-call ``Apply`` nodes into structured ``Reduce`` nodes.

    This is a no-runtime-impact scaffolding pass used by Option 1 migration.
    It rewrites helper calls in typed IR only; compile/normalize still consume
    string expressions today.

    Returns:
        Expression IR with helper calls lowered to ``Reduce`` nodes.
    """
    if isinstance(expr, Apply):
        lowered_args = tuple(lower_helper_calls(arg) for arg in expr.args)
        return _lower_single_helper(Apply(op=expr.op, args=lowered_args))

    if isinstance(expr, Reduce):
        return Reduce(
            kind=expr.kind,
            bindings=expr.bindings,
            body=lower_helper_calls(expr.body),
            filters=expr.filters,
            kernel=expr.kernel,
        )

    if isinstance(expr, HistoryOp):
        return HistoryOp(
            kind=expr.kind,
            body=lower_helper_calls(expr.body),
            options=tuple((k, lower_helper_calls(v)) for k, v in expr.options),
        )

    return expr


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
        return AxisIndex(axis=node.id)
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int, float, bool)
    ):
        return AxisIndex(axis="", coord=str(node.value))
    _invalid(detail=f"unsupported subscript index node: {type(node).__name__}")


def _bound_axis_index_from_slice(node: ast.Slice) -> AxisIndex:
    if node.step is not None:
        _invalid(detail="bound-axis subscript axis:binding must not have a step")
    if node.lower is None or node.upper is None:
        _invalid(detail="bound-axis subscript requires axis:binding with both names")
    if not isinstance(node.lower, ast.Name) or not isinstance(node.upper, ast.Name):
        _invalid(detail="bound-axis subscript axis:binding requires bare identifiers")
    return AxisIndex(axis=node.lower.id, coord=node.upper.id)


def _parse_subscript_index_element(node: ast.expr) -> AxisIndex:
    if isinstance(node, ast.Slice):
        return _bound_axis_index_from_slice(node)
    return _axis_index_from_expr(node)


def _parse_subscript_indices(slc: ast.expr) -> tuple[AxisIndex, ...]:
    if isinstance(slc, ast.Tuple):
        return tuple(_parse_subscript_index_element(elt) for elt in slc.elts)
    return (_parse_subscript_index_element(slc),)


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
        # Special-case ``x in [c1, c2, ...]`` (single ``In`` op) so helper
        # kwarg values like ``axis=var in [c1, c2]`` survive parsing and
        # can be recognized as filter specifications by
        # :func:`_lower_single_helper`.
        if (
            len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
            and isinstance(node.comparators[0], ast.List)
        ):
            return Apply(
                op="in",
                args=(
                    to_ir(node.left),
                    Apply(
                        op="list",
                        args=tuple(to_ir(e) for e in node.comparators[0].elts),
                    ),
                ),
            )
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
        call_args: list[Expr] = [to_ir(arg) for arg in node.args]
        for kw in node.keywords:
            if kw.arg is None:
                _invalid(detail="kwargs unpacking is not supported in IR parser")
            call_args.append(
                Apply(
                    op="kwarg",
                    args=(Literal(value=kw.arg), to_ir(kw.value)),
                )
            )
        return Apply(
            op=_call_name(node.func),
            args=tuple(call_args),
        )

    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name):
            _invalid(detail="subscript base must be a simple name")
        return Subscript(
            name=node.value.id,
            indices=_parse_subscript_indices(node.slice),
        )

    _invalid(detail=f"unsupported AST node in IR parser: {type(node).__name__}")


_PARSE_IR_CACHE: dict[tuple[str, bool], Expr] = {}


def parse_expr_to_ir(expr: str, *, lower_helpers: bool = False) -> Expr:
    """Parse an expression string and convert it to typed IR.

    Args:
        expr: Expression source string.
        lower_helpers: When ``True``, lower helper-call nodes to ``Reduce``.

    Returns:
        Parsed IR tree.

    """
    key = (expr, lower_helpers)
    cached = _PARSE_IR_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        _invalid(detail=f"invalid expression syntax: {exc.msg}")
    ir = to_ir(tree)
    if lower_helpers:
        ir = lower_helper_calls(ir)
    _PARSE_IR_CACHE[key] = ir
    return ir


def _literal_coord_value(coord: str) -> int | float | str:
    try:
        return int(coord)
    except ValueError:
        try:
            return float(coord)
        except ValueError:
            return coord


def _axis_index_to_ast(idx: AxisIndex) -> ast.expr:
    if idx.coord is not None:
        return ast.Constant(value=_literal_coord_value(idx.coord))
    return ast.Name(id=idx.axis, ctx=ast.Load())


def _call_func_ast(name: str) -> ast.expr:
    parts = name.split(".")
    node: ast.expr = ast.Name(id=parts[0], ctx=ast.Load())
    for part in parts[1:]:
        node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
    return node


def _binary_ast(op: str) -> ast.operator:
    if op == "+":
        return ast.Add()
    if op == "-":
        return ast.Sub()
    if op == "*":
        return ast.Mult()
    if op == "/":
        return ast.Div()
    if op == "%":
        return ast.Mod()
    if op == "pow":
        return ast.Pow()
    _invalid(detail=f"unsupported binary IR op: {op}")


def _compare_ast(op: str) -> ast.cmpop:
    if op == "==":
        return ast.Eq()
    if op == "!=":
        return ast.NotEq()
    if op == "<":
        return ast.Lt()
    if op == "<=":
        return ast.LtE()
    if op == ">":
        return ast.Gt()
    if op == ">=":
        return ast.GtE()
    _invalid(detail=f"unsupported comparison IR op: {op}")


def ir_to_ast_expr(expr: Expr) -> ast.expr:  # noqa: C901, PLR0911, PLR0912
    """Convert typed IR back to a Python AST expression node.

    Args:
        expr: Typed IR expression.

    Returns:
        Equivalent Python ``ast.expr`` node, suitable for validation,
        rewriting, and compilation by the existing runtime paths.
    """
    if isinstance(expr, Literal):
        return ast.Constant(value=expr.value)
    if isinstance(expr, Sym):
        return ast.Name(id=expr.name, ctx=ast.Load())
    if isinstance(expr, Subscript):
        parts = [_axis_index_to_ast(idx) for idx in expr.indices]
        slc: ast.expr = (
            parts[0] if len(parts) == 1 else ast.Tuple(parts, ctx=ast.Load())
        )
        return ast.Subscript(
            value=ast.Name(id=expr.name, ctx=ast.Load()),
            slice=slc,
            ctx=ast.Load(),
        )
    if isinstance(expr, Reduce):
        _invalid(detail="structured Reduce nodes cannot be converted to Python AST yet")
    if isinstance(expr, HistoryOp):
        if expr.kind in {"history", "delay"}:
            _invalid(
                detail=(
                    "history/delay operators are recognized but not yet "
                    f"implemented (helper={expr.kind!r}); see issue #173"
                )
            )
        _invalid(
            detail=(
                f"{expr.kind} requires the vectorized lowering path "
                "(this scalar-AST function is legacy and insufficient)"
            )
        )
    if isinstance(expr, Apply):
        if expr.op == "neg" and len(expr.args) == 1:
            return ast.UnaryOp(op=ast.USub(), operand=ir_to_ast_expr(expr.args[0]))
        if expr.op == "pos" and len(expr.args) == 1:
            return ast.UnaryOp(op=ast.UAdd(), operand=ir_to_ast_expr(expr.args[0]))
        if expr.op in {"+", "-", "*", "/"} and len(expr.args) >= 2:
            # Left-fold N-ary arithmetic into a chain of binary ops.
            result: ast.expr = ir_to_ast_expr(expr.args[0])
            for arg in expr.args[1:]:
                result = ast.BinOp(
                    left=result,
                    op=_binary_ast(expr.op),
                    right=ir_to_ast_expr(arg),
                )
            return result
        if expr.op in {"%", "pow"} and len(expr.args) == 2:
            return ast.BinOp(
                left=ir_to_ast_expr(expr.args[0]),
                op=_binary_ast(expr.op),
                right=ir_to_ast_expr(expr.args[1]),
            )
        if expr.op in {"==", "!=", "<", "<=", ">", ">="} and len(expr.args) == 2:
            return ast.Compare(
                left=ir_to_ast_expr(expr.args[0]),
                ops=[_compare_ast(expr.op)],
                comparators=[ir_to_ast_expr(expr.args[1])],
            )
        if expr.op in {"and", "or"}:
            bool_op: ast.boolop = ast.And() if expr.op == "and" else ast.Or()
            return ast.BoolOp(op=bool_op, values=[ir_to_ast_expr(a) for a in expr.args])
        if expr.op == "ifelse" and len(expr.args) == 3:
            return ast.IfExp(
                test=ir_to_ast_expr(expr.args[0]),
                body=ir_to_ast_expr(expr.args[1]),
                orelse=ir_to_ast_expr(expr.args[2]),
            )
        call_args: list[ast.expr] = []
        keywords: list[ast.keyword] = []
        for arg in expr.args:
            if isinstance(arg, Apply) and arg.op == "kwarg" and len(arg.args) == 2:
                key_node = arg.args[0]
                if not isinstance(key_node, Literal) or not isinstance(
                    key_node.value, str
                ):
                    _invalid(detail="malformed kwarg IR node")
                keywords.append(
                    ast.keyword(arg=key_node.value, value=ir_to_ast_expr(arg.args[1]))
                )
            else:
                call_args.append(ir_to_ast_expr(arg))
        return ast.Call(func=_call_func_ast(expr.op), args=call_args, keywords=keywords)
    _invalid(detail=f"unsupported IR node for AST conversion: {type(expr).__name__}")


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _render_coord(coord: object) -> str:  # noqa: PLR0911
    """Render an ``AxisIndex.coord`` value for source-text unparsing.

    Integer-like coords render as bare integers, identifier-like coords render
    bare, and anything else falls back to ``repr``.

    Args:
        coord: Coordinate value held on an :class:`AxisIndex` node.

    Returns:
        Source-text rendering of ``coord`` suitable for embedding in a
        subscript or bound-axis expression.
    """
    if isinstance(coord, bool):
        return repr(coord)
    if isinstance(coord, int):
        return str(coord)
    if isinstance(coord, float):
        return repr(coord)
    if isinstance(coord, str):
        if coord.lstrip("-").isdigit():
            return coord
        if _IDENT_RE.match(coord):
            return coord
        return repr(coord)
    return repr(coord)


def _unparse_axis_index(idx: AxisIndex) -> str:
    if idx.coord is not None:
        rendered = _render_coord(idx.coord)
        if idx.axis and idx.axis != idx.coord:
            return f"{idx.axis}:{rendered}"
        return rendered
    return idx.axis


def _unparse_call_args(
    args: tuple[Expr, ...],
    *,
    _memo: dict[tuple[int, int, bool], str] | None = None,
) -> str:
    parts: list[str] = []
    for arg in args:
        if isinstance(arg, Apply) and arg.op == "kwarg":
            key_node, value_node = arg.args
            if not isinstance(key_node, Literal) or not isinstance(key_node.value, str):
                _invalid(detail="malformed kwarg node in IR unparser")
            parts.append(
                f"{key_node.value}="
                f"{_unparse_ir(value_node, parent_prec=0, is_right=False, _memo=_memo)}"
            )
        else:
            parts.append(_unparse_ir(arg, parent_prec=0, is_right=False, _memo=_memo))
    return ", ".join(parts)


def unparse_ir(
    expr: Expr,
    *,
    _memo: dict[tuple[int, int, bool], str] | None = None,
) -> str:
    """Render a typed IR expression back to its Python source string.

    Args:
        expr: Typed IR expression.
        _memo: Optional identity-keyed cache of already-rendered
            subexpressions, used to amortize rendering of structurally
            shared IR across many top-level expressions (e.g. when many
            cells of one template share the same synthesized IR object;
            issue #145). When provided, the cache must not be reused
            across IR trees built with different ``parse_expr_to_ir``
            options.

    Returns:
        Python expression source string equivalent to ``expr``.
    """
    return _unparse_ir(expr, parent_prec=0, is_right=False, _memo=_memo)


# Operator precedence (higher binds tighter). Mirrors Python's grammar for
# the operator subset accepted by the IR.
_PREC_OR: int = 1
_PREC_AND: int = 2
_PREC_CMP: int = 4
_PREC_ADD: int = 9
_PREC_MUL: int = 10
_PREC_UNARY: int = 11
_PREC_POW: int = 12
_PREC_ATOM: int = 100

_BINARY_PREC: dict[str, int] = {
    "or": _PREC_OR,
    "and": _PREC_AND,
    "==": _PREC_CMP,
    "!=": _PREC_CMP,
    "<": _PREC_CMP,
    "<=": _PREC_CMP,
    ">": _PREC_CMP,
    ">=": _PREC_CMP,
    "+": _PREC_ADD,
    "-": _PREC_ADD,
    "*": _PREC_MUL,
    "/": _PREC_MUL,
    "%": _PREC_MUL,
}

# Right operand of these left-associative ops needs parens at equal precedence
# (e.g. ``a - (b + c)`` must not collapse to ``a - b + c``).
_LEFT_ASSOC_NEEDS_RIGHT_PARENS: frozenset[str] = frozenset({"-", "/", "%"})


def _expr_precedence(expr: Expr) -> int:
    if isinstance(expr, Apply):
        if expr.op in {"neg", "pos"}:
            return _PREC_UNARY
        if expr.op == "pow":
            return _PREC_POW
        if expr.op == "ifelse":
            # ternary binds looser than ``or``
            return 0
        if expr.op in _BINARY_PREC:
            return _BINARY_PREC[expr.op]
    return _PREC_ATOM


def _wrap(text: str, *, need: bool) -> str:
    return f"({text})" if need else text


def _unparse_binary(
    expr: Apply,
    *,
    op: str,
    prec: int,
    parent_prec: int,
    is_right: bool,
    _memo: dict[tuple[int, int, bool], str] | None = None,
) -> str:
    sep = f" {op} "
    # Fold args left-associatively so multi-arg flatten still renders correctly.
    args = expr.args
    left_str = _unparse_ir(args[0], parent_prec=prec, is_right=False, _memo=_memo)
    rendered = left_str
    for nxt in args[1:]:
        right_str = _unparse_ir(nxt, parent_prec=prec, is_right=True, _memo=_memo)
        rendered = f"{rendered}{sep}{right_str}"
    need_parens = prec < parent_prec or (
        prec == parent_prec and is_right and op in _LEFT_ASSOC_NEEDS_RIGHT_PARENS
    )
    return _wrap(rendered, need=need_parens)


def _unparse_ir(  # noqa: C901, PLR0911, PLR0912, PLR0914, PLR0915
    expr: Expr,
    *,
    parent_prec: int,
    is_right: bool,
    _memo: dict[tuple[int, int, bool], str] | None = None,
) -> str:
    key: tuple[int, int, bool] | None = None
    if _memo is not None:
        key = (id(expr), parent_prec, is_right)
        cached = _memo.get(key)
        if cached is not None:
            return cached

    result: str
    if isinstance(expr, Literal):
        result = repr(expr.value)
        if _memo is not None and key is not None:
            _memo[key] = result
        return result

    if isinstance(expr, Sym):
        if _memo is not None and key is not None:
            _memo[key] = expr.name
        return expr.name

    if isinstance(expr, Subscript):
        idx_str = ", ".join(_unparse_axis_index(i) for i in expr.indices)
        result = f"{expr.name}[{idx_str}]"
        if _memo is not None and key is not None:
            _memo[key] = result
        return result

    if isinstance(expr, Apply):
        if expr.op == "neg" and len(expr.args) == 1:
            inner = _unparse_ir(
                expr.args[0], parent_prec=_PREC_UNARY, is_right=False, _memo=_memo
            )
            need = parent_prec > _PREC_UNARY
            result = _wrap(f"-{inner}", need=need)
        elif expr.op == "pos" and len(expr.args) == 1:
            inner = _unparse_ir(
                expr.args[0], parent_prec=_PREC_UNARY, is_right=False, _memo=_memo
            )
            need = parent_prec > _PREC_UNARY
            result = _wrap(f"+{inner}", need=need)
        elif expr.op == "pow" and len(expr.args) == 2:
            left = _unparse_ir(
                expr.args[0], parent_prec=_PREC_POW + 1, is_right=False, _memo=_memo
            )
            right = _unparse_ir(
                expr.args[1], parent_prec=_PREC_POW, is_right=True, _memo=_memo
            )
            need = parent_prec > _PREC_POW
            result = _wrap(f"{left} ** {right}", need=need)
        elif expr.op == "ifelse" and len(expr.args) == 3:
            test, body, orelse = expr.args
            rendered = (
                f"{_unparse_ir(body, parent_prec=1, is_right=False, _memo=_memo)} if "
                f"{_unparse_ir(test, parent_prec=1, is_right=False, _memo=_memo)} else "
                f"{_unparse_ir(orelse, parent_prec=0, is_right=False, _memo=_memo)}"
            )
            result = _wrap(rendered, need=parent_prec > 0)
        elif expr.op in _BINARY_PREC and len(expr.args) >= 2:
            result = _unparse_binary(
                expr,
                op=expr.op,
                prec=_BINARY_PREC[expr.op],
                parent_prec=parent_prec,
                is_right=is_right,
                _memo=_memo,
            )
        else:
            result = f"{expr.op}({_unparse_call_args(expr.args, _memo=_memo)})"
        if _memo is not None and key is not None:
            _memo[key] = result
        return result

    if isinstance(expr, Reduce):
        binding_str = ", ".join(f"{k}={v}" for k, v in expr.bindings)
        suffix = f", {binding_str}" if binding_str else ""
        body_str = _unparse_ir(expr.body, parent_prec=0, is_right=False, _memo=_memo)
        result = f"{expr.kind}({body_str}{suffix})"
        if _memo is not None and key is not None:
            _memo[key] = result
        return result

    if isinstance(expr, HistoryOp):
        body_str = _unparse_ir(expr.body, parent_prec=0, is_right=False, _memo=_memo)
        kw_parts = [
            (f"{k}={_unparse_ir(v, parent_prec=0, is_right=False, _memo=_memo)}")
            for k, v in expr.options
        ]
        suffix = ""
        if kw_parts:
            suffix = ", " + ", ".join(kw_parts)
        result = f"{expr.kind}({body_str}{suffix})"
        if _memo is not None and key is not None:
            _memo[key] = result
        return result

    _invalid(detail=f"unsupported IR node in unparser: {type(expr).__name__}")


class AxisKind(StrEnum):
    """Classification of an ``AxisIndex`` after axis resolution.

    Categories:
        FREE: A known axis name with no coord
            (e.g. ``K[age]`` where ``age`` is a registered axis).
        COORD: A literal coord value (e.g. ``K[0]`` or ``K['x']``).
        COORD_SYMBOL: An identifier used in a subscript that is not a known
            axis - treated as a bound coord variable
            (e.g. the ``ap`` in ``K[age, ap]`` when only ``age`` is an axis).
    """

    FREE = "free"
    COORD = "coord"
    COORD_SYMBOL = "coord_symbol"


def classify_axis_index(idx: AxisIndex, *, axis_names: frozenset[str]) -> AxisKind:
    """Classify a single ``AxisIndex`` against a registry of known axes.

    Args:
        idx: The axis index node to classify.
        axis_names: Set of registered axis identifier strings.

    Returns:
        The :class:`AxisKind` describing this index position.
    """
    if idx.coord is not None:
        return AxisKind.COORD
    if idx.axis in axis_names:
        return AxisKind.FREE
    return AxisKind.COORD_SYMBOL


def iter_subscripts(expr: Expr) -> Iterator[Subscript]:
    """Yield every ``Subscript`` node reachable from ``expr`` in walk order.

    Args:
        expr: Root IR expression.

    Yields:
        Each :class:`Subscript` node encountered during a pre-order traversal.
    """
    if isinstance(expr, Subscript):
        yield expr
        return
    if isinstance(expr, Apply):
        for arg in expr.args:
            yield from iter_subscripts(arg)
        return
    if isinstance(expr, Reduce):
        yield from iter_subscripts(expr.body)
        return
    if isinstance(expr, HistoryOp):
        yield from iter_subscripts(expr.body)
        for _, opt_expr in expr.options:
            yield from iter_subscripts(opt_expr)


def walk(expr: Expr) -> Iterator[Expr]:
    """Yield every IR node reachable from ``expr`` in pre-order.

    Args:
        expr: Root IR expression.

    Yields:
        ``expr`` itself first, followed by every child node.
    """
    yield expr
    if isinstance(expr, Subscript):
        return
    if isinstance(expr, Apply):
        for arg in expr.args:
            yield from walk(arg)
        return
    if isinstance(expr, Reduce):
        yield from walk(expr.body)
        return
    if isinstance(expr, HistoryOp):
        yield from walk(expr.body)
        for _, opt_expr in expr.options:
            yield from walk(opt_expr)


def _free_symbols_push_children(
    stack: list[tuple[Expr, bool]],
    node: Expr,
    memo: Mapping[int, frozenset[str]],
) -> None:
    """Push unresolved children for ``free_symbols`` post-order traversal."""
    if isinstance(node, Apply):
        stack.append((node, True))
        stack.extend((arg, False) for arg in reversed(node.args) if id(arg) not in memo)
        return
    if isinstance(node, Reduce):
        stack.append((node, True))
        if id(node.body) not in memo:
            stack.append((node.body, False))
        return
    if isinstance(node, HistoryOp):
        stack.append((node, True))
        if id(node.body) not in memo:
            stack.append((node.body, False))
        stack.extend(
            (opt_expr, False)
            for _, opt_expr in reversed(node.options)
            if id(opt_expr) not in memo
        )
        return
    _invalid(detail=f"unsupported IR node in free_symbols: {type(node).__name__}")


def _free_symbols_finalize(
    node: Expr,
    memo: Mapping[int, frozenset[str]],
) -> frozenset[str]:
    """Combine already-computed child results for ``free_symbols``.

    Returns:
        The free-symbol set for ``node`` assembled from cached child results.
    """
    if isinstance(node, Apply):
        acc: set[str] = set()
        for arg in node.args:
            acc.update(memo[id(arg)])
        return frozenset(acc)
    if isinstance(node, Reduce):
        bound = {bind for _, bind in node.bindings}
        return frozenset(memo[id(node.body)] - bound)
    if isinstance(node, HistoryOp):
        hist_acc: set[str] = set(memo[id(node.body)])
        for _, opt_expr in node.options:
            hist_acc.update(memo[id(opt_expr)])
        return frozenset(hist_acc)
    _invalid(detail=f"unsupported IR node in free_symbols: {type(node).__name__}")


def free_symbols(
    expr: Expr,
    memo: dict[int, frozenset[str]] | None = None,
) -> frozenset[str]:
    """Return the set of ``Sym`` names referenced under ``expr``.

    ``Reduce`` bindings shadow outer symbols of the same name: a bound name
    appearing only inside a reduction body is *not* considered free.

    Args:
        expr: Root IR expression.
        memo: Optional identity-keyed cache (``id(node) -> frozenset``) used
            to share subtree results across calls on shared (frozen) IR.
            Callers that invoke ``free_symbols`` many times over the same
            substructure (e.g. inlining one alias body into many equations)
            should pass a single memo dict to avoid O(alias_size) per call.

    Returns:
        Frozen set of identifier strings that occur as free :class:`Sym`
        references.
    """
    if memo is None:
        memo = {}
    root_key = id(expr)
    cached = memo.get(root_key)
    if cached is not None:
        return cached
    stack: list[tuple[Expr, bool]] = [(expr, False)]
    while stack:
        node, expanded = stack.pop()
        key = id(node)
        if key in memo:
            continue
        if isinstance(node, Sym):
            memo[key] = frozenset({node.name})
            continue
        if isinstance(node, (Literal, Subscript)):
            memo[key] = frozenset()
            continue
        if expanded:
            memo[key] = _free_symbols_finalize(node, memo)
            continue
        _free_symbols_push_children(stack, node, memo)
    return memo[root_key]


def substitute(
    expr: Expr,
    mapping: Mapping[str, Expr],
    memo: dict[int, frozenset[str]] | None = None,
) -> Expr:
    """Replace named ``Sym`` leaves with their mapped IR expression.

    The substitution is capture-avoiding with respect to ``Reduce`` bindings:
    if a name is bound by an enclosing ``Reduce``, occurrences of that name
    in the body are left untouched (the binding shadows the outer mapping).

    Args:
        expr: Root IR expression.
        mapping: Substitution table from symbol name to replacement IR.
        memo: Optional identity-keyed cache shared with :func:`free_symbols`.
            When provided, subtrees with no free symbols intersecting
            ``mapping`` keys are returned unchanged in O(1), avoiding a full
            re-walk of large alias bodies that were just inlined.

    Returns:
        A new IR expression with substitutions applied; structurally equal
        to ``expr`` when no replacements occur.
    """
    keys = mapping.keys()
    if memo is not None and not (free_symbols(expr, memo) & keys):
        return expr
    if isinstance(expr, Sym):
        return mapping.get(expr.name, expr)
    if isinstance(expr, (Literal, Subscript)):
        return expr
    if isinstance(expr, Apply):
        return Apply(
            op=expr.op,
            args=tuple(substitute(arg, mapping, memo) for arg in expr.args),
        )
    if isinstance(expr, Reduce):
        bound = {bind for _, bind in expr.bindings}
        inner = (
            mapping
            if not bound
            else {k: v for k, v in mapping.items() if k not in bound}
        )
        return Reduce(
            kind=expr.kind,
            bindings=expr.bindings,
            body=substitute(expr.body, inner, memo),
            filters=expr.filters,
            kernel=expr.kernel,
        )
    if isinstance(expr, HistoryOp):
        return HistoryOp(
            kind=expr.kind,
            body=substitute(expr.body, mapping, memo),
            options=tuple((k, substitute(v, mapping, memo)) for k, v in expr.options),
        )
    _invalid(detail=f"unsupported IR node in substitute: {type(expr).__name__}")


def _map_children(expr: Expr, fn: Callable[[Expr], Expr]) -> Expr:  # noqa: PLR0911
    if isinstance(expr, Apply):
        new_args = tuple(fn(arg) for arg in expr.args)
        if new_args == expr.args:
            return expr
        return Apply(op=expr.op, args=new_args)
    if isinstance(expr, Reduce):
        new_body = fn(expr.body)
        if new_body == expr.body:
            return expr
        return Reduce(
            kind=expr.kind,
            bindings=expr.bindings,
            body=new_body,
            filters=expr.filters,
            kernel=expr.kernel,
        )
    if isinstance(expr, HistoryOp):
        new_body = fn(expr.body)
        new_options = tuple((k, fn(v)) for k, v in expr.options)
        if new_body == expr.body and new_options == expr.options:
            return expr
        return HistoryOp(kind=expr.kind, body=new_body, options=new_options)
    return expr


def _is_cse_candidate(expr: Expr) -> bool:
    return isinstance(expr, (Apply, Reduce, HistoryOp))


def _expr_cost(expr: Expr) -> int:
    if isinstance(expr, Apply):
        return 1 + sum(_expr_cost(arg) for arg in expr.args)
    if isinstance(expr, Reduce):
        return 1 + _expr_cost(expr.body)
    if isinstance(expr, HistoryOp):
        return 1 + _expr_cost(expr.body) + sum(_expr_cost(v) for _, v in expr.options)
    return 1


def _postorder(expr: Expr, out: dict[Expr, int]) -> None:
    if isinstance(expr, Apply):
        for arg in expr.args:
            _postorder(arg, out)
    elif isinstance(expr, Reduce):
        _postorder(expr.body, out)
    elif isinstance(expr, HistoryOp):
        _postorder(expr.body, out)
        for _, opt_expr in expr.options:
            _postorder(opt_expr, out)
    out[expr] = out.get(expr, 0) + 1


def extract_common_subexpressions(
    exprs: Sequence[Expr],
    *,
    prefix: str = "_cse",
    min_cost: int = 2,
    reserved_names: Iterable[str] = (),
) -> tuple[tuple[tuple[str, Expr], ...], tuple[Expr, ...]]:
    """Extract repeated IR subtrees into deterministic symbol bindings.

    This is a pure planning pass for the #112 IR migration. It does not
    perform code generation; callers receive a list of temporary bindings and
    rewritten root expressions that reference those temporaries. Bindings are
    emitted in child-before-parent order so later bindings may depend on
    earlier temporary names.

    Args:
        exprs: Root IR expressions to analyze together.
        prefix: Prefix used for generated temporary symbol names.
        min_cost: Minimum subtree cost eligible for extraction.
        reserved_names: Names that generated temporaries must not use.

    Returns:
        ``(bindings, rewritten_exprs)`` where ``bindings`` is a tuple of
        ``(name, expr)`` pairs and ``rewritten_exprs`` is positionally aligned
        with ``exprs``.
    """
    counts: dict[Expr, int] = {}
    for expr in exprs:
        _postorder(expr, counts)

    selected = tuple(
        expr
        for expr, count in counts.items()
        if count > 1 and _is_cse_candidate(expr) and _expr_cost(expr) >= min_cost
    )
    reserved = set(reserved_names)
    names: dict[Expr, str] = {}
    next_index = 0
    for expr in selected:
        while True:
            candidate = f"{prefix}{next_index}"
            next_index += 1
            if candidate not in reserved:
                break
        names[expr] = candidate
        reserved.add(candidate)

    def replace(expr: Expr) -> Expr:
        name = names.get(expr)
        if name is not None:
            return Sym(name=name)
        return _map_children(expr, replace)

    bindings = tuple((names[expr], _map_children(expr, replace)) for expr in selected)
    rewritten = tuple(replace(expr) for expr in exprs)
    return bindings, rewritten


def axis_kinds(expr: Expr, *, axis_names: frozenset[str]) -> tuple[AxisKind, ...]:
    """Classify every ``AxisIndex`` position in walk order.

    Args:
        expr: Root IR expression.
        axis_names: Set of registered axis identifier strings.

    Returns:
        Tuple of :class:`AxisKind` values in pre-order subscript / position
        traversal order.
    """
    return tuple(
        classify_axis_index(idx, axis_names=axis_names)
        for sub in iter_subscripts(expr)
        for idx in sub.indices
    )


def _resolve_index(idx: AxisIndex, *, axis_names: frozenset[str]) -> AxisIndex:
    kind = classify_axis_index(idx, axis_names=axis_names)
    if idx.kind is kind:
        return idx
    return AxisIndex(axis=idx.axis, coord=idx.coord, kind=kind)


def resolve_axis_kinds(expr: Expr, *, axis_names: frozenset[str]) -> Expr:  # noqa: PLR0911
    """Return a copy of ``expr`` with every ``AxisIndex.kind`` populated.

    Walks the IR tree and rewrites each :class:`AxisIndex` so its ``kind``
    field reflects classification against ``axis_names``. Nodes that already
    carry the correct kind are reused, so a resolved tree is idempotent.

    Args:
        expr: Root IR expression (typically straight from the parser).
        axis_names: Set of registered axis identifier strings.

    Returns:
        Structurally equivalent IR with ``AxisIndex.kind`` set on every
        subscript position.
    """
    if isinstance(expr, (Literal, Sym)):
        return expr
    if isinstance(expr, Subscript):
        new_indices = tuple(
            _resolve_index(i, axis_names=axis_names) for i in expr.indices
        )
        if new_indices == expr.indices:
            return expr
        return Subscript(name=expr.name, indices=new_indices)
    if isinstance(expr, Apply):
        new_args = tuple(
            resolve_axis_kinds(arg, axis_names=axis_names) for arg in expr.args
        )
        if new_args == expr.args:
            return expr
        return Apply(op=expr.op, args=new_args)
    if isinstance(expr, Reduce):
        new_body = resolve_axis_kinds(expr.body, axis_names=axis_names)
        if new_body is expr.body:
            return expr
        return Reduce(
            kind=expr.kind,
            bindings=expr.bindings,
            body=new_body,
            filters=expr.filters,
            kernel=expr.kernel,
        )
    if isinstance(expr, HistoryOp):
        new_body = resolve_axis_kinds(expr.body, axis_names=axis_names)
        new_options = tuple(
            (k, resolve_axis_kinds(v, axis_names=axis_names)) for k, v in expr.options
        )
        if new_body is expr.body and new_options == expr.options:
            return expr
        return HistoryOp(kind=expr.kind, body=new_body, options=new_options)
    _invalid(detail=f"unsupported IR node in resolve_axis_kinds: {type(expr).__name__}")


__all__ = [
    "Apply",
    "AxisIndex",
    "AxisKind",
    "Expr",
    "HistoryOp",
    "Literal",
    "Reduce",
    "Subscript",
    "Sym",
    "axis_kinds",
    "classify_axis_index",
    "extract_common_subexpressions",
    "free_symbols",
    "ir_to_ast_expr",
    "iter_subscripts",
    "lower_helper_calls",
    "parse_expr_to_ir",
    "resolve_axis_kinds",
    "substitute",
    "to_ir",
    "unparse_ir",
    "walk",
]
