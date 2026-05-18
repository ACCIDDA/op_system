"""Typed expression IR for op_system.

This module introduces a small, immutable intermediate representation (IR)
for expression terms. The initial scope is parse-only: convert Python AST
expressions into typed IR nodes without changing compile/normalize behavior.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NoReturn

from ._errors import InvalidExpressionError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


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
    placeholder: str | None = None
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
    """Reduction primitive placeholder for future helper-lowering passes."""

    kind: str
    bindings: tuple[tuple[str, str], ...]
    body: Expr


Expr = Literal | Sym | Subscript | Apply | Reduce

_HELPER_REDUCE_OPS: frozenset[str] = frozenset({
    "apply_along",
    "sum_over",
    "integrate_over",
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
    _invalid(detail=f"helper binding {key!r} must be a symbol or literal")


def _lower_single_helper(node: Apply) -> Expr:
    if node.op not in _HELPER_REDUCE_OPS:
        return node

    kw_nodes: list[Apply] = []
    positional: list[Expr] = []
    for arg in node.args:
        if isinstance(arg, Apply) and arg.op == "kwarg":
            kw_nodes.append(arg)
        else:
            positional.append(arg)

    if len(positional) != 1:
        _invalid(detail=f"{node.op} helper requires exactly one body expression")

    bindings: list[tuple[str, str]] = []
    for kw_node in kw_nodes:
        if len(kw_node.args) != 2:
            _invalid(detail="kwarg marker must contain (key, value)")
        key = _kwarg_name(kw_node.args[0])
        val = _kwarg_value_as_binding(kw_node.args[1], key=key)
        bindings.append((key, val))

    return Reduce(kind=node.op, bindings=tuple(bindings), body=positional[0])


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


def parse_expr_to_ir(expr: str, *, lower_helpers: bool = False) -> Expr:
    """Parse an expression string and convert it to typed IR.

    Args:
        expr: Expression source string.
        lower_helpers: When ``True``, lower helper-call nodes to ``Reduce``.

    Returns:
        Parsed IR tree.

    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        _invalid(detail=f"invalid expression syntax: {exc.msg}")
    ir = to_ir(tree)
    if lower_helpers:
        return lower_helper_calls(ir)
    return ir


def _unparse_axis_index(idx: AxisIndex) -> str:
    if idx.placeholder is not None:
        return f"${idx.placeholder}"
    if idx.coord is not None:
        return repr(idx.coord)
    return idx.axis


def _unparse_call_args(args: tuple[Expr, ...]) -> str:
    parts: list[str] = []
    for arg in args:
        if isinstance(arg, Apply) and arg.op == "kwarg":
            key_node, value_node = arg.args
            if not isinstance(key_node, Literal) or not isinstance(key_node.value, str):
                _invalid(detail="malformed kwarg node in IR unparser")
            parts.append(f"{key_node.value}={unparse_ir(value_node)}")
        else:
            parts.append(unparse_ir(arg))
    return ", ".join(parts)


def unparse_ir(expr: Expr) -> str:  # noqa: C901, PLR0911
    """Render a typed IR expression back to its Python source string.

    Args:
        expr: Typed IR expression.

    Returns:
        Python expression source string equivalent to ``expr``.
    """
    if isinstance(expr, Literal):
        return repr(expr.value)

    if isinstance(expr, Sym):
        return expr.name

    if isinstance(expr, Subscript):
        idx_str = ", ".join(_unparse_axis_index(i) for i in expr.indices)
        return f"{expr.name}[{idx_str}]"

    if isinstance(expr, Apply):
        if expr.op == "neg" and len(expr.args) == 1:
            return f"-({unparse_ir(expr.args[0])})"
        if expr.op == "pos" and len(expr.args) == 1:
            return f"+({unparse_ir(expr.args[0])})"
        if expr.op == "pow" and len(expr.args) == 2:
            left, right = expr.args
            return f"({unparse_ir(left)}) ** ({unparse_ir(right)})"
        if expr.op == "ifelse" and len(expr.args) == 3:
            test, body, orelse = expr.args
            return (
                f"({unparse_ir(body)}) if ({unparse_ir(test)})"
                f" else ({unparse_ir(orelse)})"
            )
        if expr.op in _BINARY_OPS and len(expr.args) >= 2:
            sep = f" {expr.op} "
            return "(" + sep.join(unparse_ir(a) for a in expr.args) + ")"
        return f"{expr.op}({_unparse_call_args(expr.args)})"

    if isinstance(expr, Reduce):
        binding_str = ", ".join(f"{k}={v}" for k, v in expr.bindings)
        suffix = f", {binding_str}" if binding_str else ""
        return f"{expr.kind}({unparse_ir(expr.body)}{suffix})"

    _invalid(detail=f"unsupported IR node in unparser: {type(expr).__name__}")


class AxisKind(StrEnum):
    """Classification of an ``AxisIndex`` after axis resolution.

    Categories:
        FREE: A known axis name with no coord or placeholder
            (e.g. ``K[age]`` where ``age`` is a registered axis).
        COORD: A literal coord value (e.g. ``K[0]`` or ``K['x']``).
        PLACEHOLDER: A ``$``-prefixed independent placeholder (#88).
        COORD_SYMBOL: An identifier used in a subscript that is not a known
            axis - treated as a bound coord variable
            (e.g. the ``ap`` in ``K[age, ap]`` when only ``age`` is an axis).
    """

    FREE = "free"
    COORD = "coord"
    PLACEHOLDER = "placeholder"
    COORD_SYMBOL = "coord_symbol"


def classify_axis_index(idx: AxisIndex, *, axis_names: frozenset[str]) -> AxisKind:
    """Classify a single ``AxisIndex`` against a registry of known axes.

    Args:
        idx: The axis index node to classify.
        axis_names: Set of registered axis identifier strings.

    Returns:
        The :class:`AxisKind` describing this index position.
    """
    if idx.placeholder is not None:
        return AxisKind.PLACEHOLDER
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


def free_symbols(expr: Expr) -> frozenset[str]:
    """Return the set of ``Sym`` names referenced under ``expr``.

    ``Reduce`` bindings shadow outer symbols of the same name: a bound name
    appearing only inside a reduction body is *not* considered free.

    Args:
        expr: Root IR expression.

    Returns:
        Frozen set of identifier strings that occur as free :class:`Sym`
        references.
    """
    if isinstance(expr, Sym):
        return frozenset({expr.name})
    if isinstance(expr, (Literal, Subscript)):
        return frozenset()
    if isinstance(expr, Apply):
        out: set[str] = set()
        for arg in expr.args:
            out |= free_symbols(arg)
        return frozenset(out)
    if isinstance(expr, Reduce):
        bound = {bind for _, bind in expr.bindings}
        return frozenset(free_symbols(expr.body) - bound)
    _invalid(detail=f"unsupported IR node in free_symbols: {type(expr).__name__}")


def substitute(expr: Expr, mapping: Mapping[str, Expr]) -> Expr:
    """Replace named ``Sym`` leaves with their mapped IR expression.

    The substitution is capture-avoiding with respect to ``Reduce`` bindings:
    if a name is bound by an enclosing ``Reduce``, occurrences of that name
    in the body are left untouched (the binding shadows the outer mapping).

    Args:
        expr: Root IR expression.
        mapping: Substitution table from symbol name to replacement IR.

    Returns:
        A new IR expression with substitutions applied; structurally equal
        to ``expr`` when no replacements occur.
    """
    if isinstance(expr, Sym):
        return mapping.get(expr.name, expr)
    if isinstance(expr, (Literal, Subscript)):
        return expr
    if isinstance(expr, Apply):
        return Apply(
            op=expr.op,
            args=tuple(substitute(arg, mapping) for arg in expr.args),
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
            body=substitute(expr.body, inner),
        )
    _invalid(detail=f"unsupported IR node in substitute: {type(expr).__name__}")


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
    return AxisIndex(
        axis=idx.axis, coord=idx.coord, placeholder=idx.placeholder, kind=kind
    )


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
        return Reduce(kind=expr.kind, bindings=expr.bindings, body=new_body)
    _invalid(detail=f"unsupported IR node in resolve_axis_kinds: {type(expr).__name__}")


__all__ = [
    "Apply",
    "AxisIndex",
    "AxisKind",
    "Expr",
    "Literal",
    "Reduce",
    "Subscript",
    "Sym",
    "axis_kinds",
    "classify_axis_index",
    "free_symbols",
    "iter_subscripts",
    "lower_helper_calls",
    "parse_expr_to_ir",
    "resolve_axis_kinds",
    "substitute",
    "to_ir",
    "unparse_ir",
    "walk",
]
