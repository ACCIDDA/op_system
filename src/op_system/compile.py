"""
op_system.compile.

Compile normalized RHS specifications into efficient, runnable callables.

This module is domain-agnostic and intentionally does not import flepimop2.

Contract
--------
- Accepts `NormalizedRhs` (from `op_system.specs`) and produces a `CompiledRhs`
  object containing an `eval_fn` suitable for numerical backends.
- Raises built-in exceptions and chains `OpSystemError` as the cause via helpers
  in `op_system.errors`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .errors import (
    raise_compilation_error,
    raise_invalid_expression,
    raise_parameter_error,
    raise_state_shape_error,
    raise_unsupported_feature,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import CodeType

    from .specs import NormalizedRhs

Float64Array = NDArray[np.float64]

# -----------------------------------------------------------------------------
# Public compiled RHS container
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledRhs:
    """Container for a compiled RHS evaluation function."""

    state_names: tuple[str, ...]
    param_names: tuple[str, ...]
    eval_fn: Callable[[np.float64, Float64Array], Float64Array]


# -----------------------------------------------------------------------------
# AST validation (very conservative v1)
# -----------------------------------------------------------------------------

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Load,
    ast.Name,
    ast.Constant,
    ast.Call,
    ast.Attribute,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.And,
    ast.Or,
    ast.BoolOp,
    ast.IfExp,
)


_ALLOWED_CALL_ROOTS: tuple[str, ...] = ("np",)
_ALLOWED_CALL_FUNCS: frozenset[str] = frozenset({
    # NumPy scalar math; keep small initially
    "abs",
    "exp",
    "log",
    "log1p",
    "sqrt",
    "maximum",
    "minimum",
    "clip",
    "where",
})


def _parse_expr(expr: str) -> ast.AST:
    """
    Parse a Python expression and return the AST.

    Args:
        expr: Expression string.

    Returns:
        Parsed AST for the expression.
    """
    try:
        return ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover
        raise_invalid_expression(detail=f"invalid expression syntax: {exc.msg}")


def _validate_ast(tree: ast.AST, *, expr: str) -> None:
    """Validate that an expression AST only contains allowed constructs.

    Args:
        tree: Parsed expression AST.
        expr: Original expression string, for diagnostics.
    """
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise_invalid_expression(
                detail=f"disallowed AST node {type(node).__name__} in {expr!r}"
            )

        # Explicitly reject constructs that could slip through via allowed nodes.
        if isinstance(node, ast.Name) and node.id in {"__import__", "__builtins__"}:
            raise_invalid_expression(detail=f"disallowed name: {node.id!r}")

        if isinstance(node, ast.Attribute) and (
            not isinstance(node.value, ast.Name) or node.value.id != "np"
        ):
            raise_invalid_expression(detail=f"disallowed attribute access in {expr!r}")

        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Attribute):
                raise_invalid_expression(
                    detail=f"only attribute calls allowed in {expr!r}"
                )
            if not isinstance(func.value, ast.Name):
                raise_invalid_expression(detail=f"invalid call root in {expr!r}")

            root = str(func.value.id)
            name = str(func.attr)
            if root not in _ALLOWED_CALL_ROOTS or name not in _ALLOWED_CALL_FUNCS:
                raise_invalid_expression(
                    detail=f"disallowed function call: {root}.{name}"
                )


def _compile_expr(expr: str) -> CodeType:
    """
    Parse, validate, and compile an expression into a code object.

    Args:
        expr: Expression string.

    Returns:
        Compiled code object.
    """
    tree = _parse_expr(expr)
    _validate_ast(tree, expr=expr)
    try:
        return compile(tree, filename="<op_system>", mode="eval")
    except (ValueError, TypeError, SyntaxError) as exc:  # pragma: no cover
        raise_compilation_error(
            detail=f"failed to compile expression {expr!r}: {exc!r}"
        )


def _collect_alias_code(aliases: Mapping[str, str]) -> dict[str, CodeType]:
    """
    Compile alias expressions into code objects.

    Args:
        aliases: Mapping from alias name to expression string.

    Returns:
        Dictionary mapping alias name to compiled code object.
    """
    out: dict[str, CodeType] = {}
    for name, expr in aliases.items():
        out[name] = _compile_expr(expr)
    return out


def _collect_eq_code(equations: tuple[str, ...]) -> list[CodeType]:
    """
    Compile equation expressions into code objects.

    Args:
        equations: Tuple of equation expression strings.

    Returns:
        List of compiled code objects.
    """
    return [_compile_expr(expr) for expr in equations]


def _resolve_aliases(
    alias_code: Mapping[str, CodeType],
    *,
    base_env: dict[str, object],
) -> dict[str, object]:
    """Evaluate aliases with simple dependency resolution.

    Supports aliases depending on state/params and on earlier aliases. Cycles or
    unresolved dependencies are reported as invalid expressions.

    Args:
        alias_code: Mapping from alias name to compiled code object.
        base_env: Base evaluation environment (e.g., state, params, numpy).

    Returns:
        A dictionary mapping alias name to its computed value.
    """
    pending = dict(alias_code)
    out: dict[str, object] = {}
    max_passes = max(1, len(pending))
    for _ in range(max_passes):
        progressed = False
        for name, codeobj in list(pending.items()):
            try:
                # Restricted eval; AST already validated.
                val = eval(codeobj, {"__builtins__": {}}, {**base_env, **out})  # noqa: S307
            except NameError:
                continue
            except (ValueError, TypeError, ArithmeticError) as exc:
                raise_invalid_expression(
                    detail=f"alias {name!r} evaluation failed: {exc!r}"
                )
            out[name] = val
            del pending[name]
            progressed = True
        if not pending:
            return out
        if not progressed:
            break

    # If anything remains, it's an unresolved dependency/cycle.
    raise_invalid_expression(
        detail=f"could not resolve alias dependencies: {sorted(pending.keys())}"
    )
    return {}  # pragma: no cover


# -----------------------------------------------------------------------------
# Public compile API
# -----------------------------------------------------------------------------


def compile_rhs(rhs: NormalizedRhs) -> CompiledRhs:
    """Compile a normalized RHS into a runnable evaluation function.

    Args:
        rhs: Normalized RHS produced by `op_system.specs.normalize_rhs`.

    Returns:
        A `CompiledRhs` containing an `eval_fn(t, y) -> dydt`.
    """
    if rhs.kind not in {"expr", "transitions"}:
        raise_unsupported_feature(
            feature=f"rhs.kind={rhs.kind}",
            detail="Only 'expr' and 'transitions' are supported in v1.",
        )

    state_names = rhs.state_names
    n_state = len(state_names)
    name_to_idx = {s: i for i, s in enumerate(state_names)}

    alias_code = _collect_alias_code(rhs.aliases)
    eq_code = _collect_eq_code(rhs.equations)

    def eval_fn(t: np.float64, y: Float64Array, **params: object) -> Float64Array:
        # Shape checks
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1:
            raise_state_shape_error(name="state", expected="1D array", got=y_arr.shape)
        if y_arr.size != n_state:
            raise_state_shape_error(
                name="state",
                expected=f"(n_state={n_state},)",
                got=y_arr.shape,
            )

        # Base environment: state vector mapped by name + numpy namespace + params
        env: dict[str, object] = {"np": np, "t": np.float64(t)}
        for s, i in name_to_idx.items():
            env[s] = np.float64(y_arr[i])
        env.update(params)

        # Evaluate aliases (may depend on state/params and earlier aliases)
        if alias_code:
            env.update(_resolve_aliases(alias_code, base_env=env))

        out = np.empty((n_state,), dtype=np.float64)
        for i, codeobj in enumerate(eq_code):
            try:
                # Restricted eval; AST already validated.
                val = eval(codeobj, {"__builtins__": {}}, env)  # noqa: S307
            except NameError as exc:
                raise_parameter_error(detail=f"unknown symbol in equation: {exc!s}")
            except (ValueError, TypeError, ArithmeticError) as exc:
                raise_invalid_expression(detail=f"equation evaluation failed: {exc!r}")
            out[i] = np.float64(val)

        return out

    return CompiledRhs(
        state_names=state_names,
        param_names=rhs.param_names,
        eval_fn=eval_fn,
    )
