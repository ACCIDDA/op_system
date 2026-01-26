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
from types import CodeType
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
    from collections.abc import Mapping

    from .specs import NormalizedRhs

Float64Array = NDArray[np.float64]
EvalFn = callable


# -----------------------------------------------------------------------------
# Public compiled object
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledRhs:
    """A compiled RHS with a backend-friendly evaluation function.

    Attributes:
        state_names: Ordered state names.
        param_names: Ordered parameter names inferred from the RHS.
        aliases: Alias expressions (string form).
        equations: Per-state RHS expressions (string form).
        eval_fn: Callable `f(t, y, **params) -> dydt` where `y` and `dydt` are 1D.
    """

    state_names: tuple[str, ...]
    param_names: tuple[str, ...]
    aliases: Mapping[str, str]
    equations: tuple[str, ...]
    eval_fn: object


# -----------------------------------------------------------------------------
# AST parsing / validation
# -----------------------------------------------------------------------------


_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.IfExp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Attribute,
    # operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _parse_expr(expr: str) -> ast.AST:
    """Parse a Python expression into an AST.

    Args:
        expr: Expression string.

    Returns:
        The parsed AST (root node).
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

    Raises:
        ValueError: If the AST contains disallowed constructs.
    """
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise_invalid_expression(
                detail=f"disallowed syntax node {type(node).__name__} in {expr!r}"
            )

        # Explicitly reject constructs that could slip through via allowed nodes.
        if isinstance(node, ast.Name) and node.id in {"__import__", "__builtins__"}:
            raise_invalid_expression(detail=f"disallowed name: {node.id!r}")

        if isinstance(node, ast.Attribute):
            # Avoid arbitrary attribute access, except whitelisted module-like roots.
            # In v1 we keep this strict; future versions can expand.
            if not isinstance(node.value, ast.Name) or node.value.id not in {"np"}:
                raise_invalid_expression(
                    detail=f"disallowed attribute access in {expr!r}"
                )

        if isinstance(node, ast.Call):
            # Only allow calls like np.<func>(...) to keep the surface area small.
            fn = node.func
            if not (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "np"
            ):
                raise_invalid_expression(
                    detail="only np.<func>(...) calls allowed in v1"
                )


def _compile_expr(expr: str) -> CodeType:
    """Compile a validated expression to a Python code object.

    Args:
        expr: Expression string.

    Returns:
        A compiled code object suitable for `eval`.
    """
    tree = _parse_expr(expr)
    _validate_ast(tree, expr=expr)

    try:
        return compile(tree, filename="<op_system>", mode="eval")
    except (SyntaxError, ValueError, TypeError) as exc:  # pragma: no cover
        raise_compilation_error(
            detail=f"failed to compile expression {expr!r}: {exc!r}"
        )


# -----------------------------------------------------------------------------
# Alias evaluation
# -----------------------------------------------------------------------------


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

    Raises:
        ValueError: If aliases cannot be resolved due to cycles or undefined names.
    """
    pending = dict(alias_code)
    out: dict[str, object] = {}

    max_passes = max(1, len(pending))
    for _ in range(max_passes):
        progressed = False
        for name, codeobj in list(pending.items()):
            try:
                # Restricted eval; AST already validated.
                val = eval(codeobj, {"__builtins__": {}}, {**base_env, **out})
            except NameError:
                continue
            except (ArithmeticError, TypeError, ValueError) as exc:
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

    raise_invalid_expression(
        detail=f"could not resolve alias dependencies: {sorted(pending.keys())}"
    )
    raise AssertionError("unreachable")  # pragma: no cover


# -----------------------------------------------------------------------------
# Public compile entrypoint
# -----------------------------------------------------------------------------


def compile_rhs(rhs: NormalizedRhs) -> CompiledRhs:
    """Compile a normalized RHS into a runnable evaluation function.

    Args:
        rhs: Normalized RHS produced by `op_system.specs.normalize_rhs`.

    Returns:
        A `CompiledRhs` containing an `eval_fn(t, y, **params) -> dydt`.

    Raises:
        NotImplementedError: If an unsupported RHS kind is provided.
        ValueError/TypeError: For invalid expressions, shapes, or parameters.
    """
    if rhs.kind not in {"expr", "transitions"}:
        raise_unsupported_feature(
            feature=f"rhs.kind={rhs.kind}",
            detail="Only 'expr' and 'transitions' are supported in v1.",
        )

    state_names = tuple(rhs.state_names)
    n_state = len(state_names)
    state_set = set(state_names)

    # Pre-compile alias and equation expressions.
    alias_code: dict[str, CodeType] = {
        k: _compile_expr(v) for k, v in rhs.aliases.items()
    }
    eq_code: list[CodeType] = [_compile_expr(expr) for expr in rhs.equations]

    def eval_fn(t: float, y: Float64Array, **params: object) -> Float64Array:
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1 or y_arr.shape[0] != n_state:
            raise_state_shape_error(
                name="state",
                expected=f"({n_state},)",
                got=y_arr.shape,
            )

        # Ensure required params exist (and allow extras).
        missing = [p for p in rhs.param_names if p not in params]
        if missing:
            raise_parameter_error(detail=f"missing parameter(s): {sorted(missing)}")

        # Environment: state variables + params + numpy utilities.
        env: dict[str, object] = {"np": np, "t": float(t)}
        env.update({name: y_arr[i] for i, name in enumerate(state_names)})
        env.update(params)

        # Resolve aliases (may depend on state/params and earlier aliases).
        if alias_code:
            env.update(_resolve_aliases(alias_code, base_env=env))

        out = np.empty((n_state,), dtype=np.float64)
        for i, codeobj in enumerate(eq_code):
            try:
                # Restricted eval; AST already validated.
                val = eval(codeobj, {"__builtins__": {}}, env)
            except NameError as exc:
                # Provide a clearer message for missing symbols.
                raise_parameter_error(detail=f"unknown symbol in equation: {exc!s}")
            except (ArithmeticError, TypeError, ValueError) as exc:
                raise_invalid_expression(detail=f"equation evaluation failed: {exc!r}")
            out[i] = np.float64(val)

        # Safety: ensure no state symbols were accidentally overwritten.
        if any(k in env and k not in state_set for k in state_set):  # pragma: no cover
            raise_invalid_expression(
                detail="internal evaluation environment corruption"
            )

        return out

    return CompiledRhs(
        state_names=state_names,
        param_names=tuple(rhs.param_names),
        aliases=rhs.aliases,
        equations=tuple(rhs.equations),
        eval_fn=eval_fn,
    )
