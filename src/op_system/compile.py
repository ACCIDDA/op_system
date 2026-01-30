"""op_system.compile.

Compile normalized RHS specifications into efficient, runnable callables.

This module is domain-agnostic and intentionally does not import flepimop2.

Contract
--------
- Accepts `NormalizedRhs` (from `op_system.specs`) and produces a `CompiledRhs`
  object containing an `eval_fn` suitable for numerical backends.
- Raises built-in exceptions with standardized messages.

Security note
-------------
This module uses `eval()` on code objects compiled from user-provided strings.
To reduce risk, expressions are parsed and validated with a conservative AST
whitelist, and evaluation runs with empty builtins.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import CodeType

    from .specs import NormalizedRhs

Float64Array = NDArray[np.float64]

# -----------------------------------------------------------------------------
# Error message constants
# -----------------------------------------------------------------------------

INVALID_EXPRESSION_PREFIX = "Invalid op_system expression."
COMPILATION_FAILED_PREFIX = "op_system compilation failed."
INVALID_STATE_SHAPE_PREFIX = "state has an invalid shape/value."
INVALID_PARAMETERS_PREFIX = "Invalid parameters for op_system."
UNSUPPORTED_FEATURE_PREFIX = "Unsupported op_system feature."

DISALLOWED_ATTRIBUTE_ACCESS = "disallowed attribute access"
ONLY_ATTRIBUTE_CALLS_ALLOWED = "only attribute calls allowed"
DISALLOWED_FUNCTION_CALL = "disallowed function call"


def _raise_invalid_expression(*, detail: str) -> NoReturn:
    """Raise a standardized expression error.

    Args:
        detail: Error detail.

    Raises:
        ValueError: Always.
    """
    msg = f"{INVALID_EXPRESSION_PREFIX} Detail: {detail}"
    raise ValueError(msg)


def _raise_compilation_error(*, detail: str) -> NoReturn:
    """Raise a standardized compilation error.

    Args:
        detail: Error detail.

    Raises:
        RuntimeError: Always.
    """
    msg = f"{COMPILATION_FAILED_PREFIX} Detail: {detail}"
    raise RuntimeError(msg)


def _raise_state_shape_error(*, expected: str, got: object) -> NoReturn:
    """Raise a standardized state shape/value error.

    Args:
        expected: Expected shape/format description.
        got: Observed shape/value.

    Raises:
        ValueError: Always.
    """
    msg = f"{INVALID_STATE_SHAPE_PREFIX} Expected {expected}. Got: {got!r}."
    raise ValueError(msg)


def _raise_parameter_error(*, detail: str) -> NoReturn:
    """Raise a standardized parameter/type error.

    Args:
        detail: Error detail.

    Raises:
        TypeError: Always.
    """
    msg = f"{INVALID_PARAMETERS_PREFIX} {detail}"
    raise TypeError(msg)


def _raise_unsupported_feature(*, feature: str, detail: str | None = None) -> NoReturn:
    """Raise a standardized unsupported feature error.

    Args:
        feature: Feature identifier.
        detail: Optional additional detail.

    Raises:
        NotImplementedError: Always.
    """
    msg = f"{UNSUPPORTED_FEATURE_PREFIX} Feature '{feature}' is not supported."
    if detail:
        msg = f"{msg} Detail: {detail}"
    raise NotImplementedError(msg)


# -----------------------------------------------------------------------------
# Public compiled RHS container
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledRhs:
    """Container for a compiled RHS evaluation function."""

    state_names: tuple[str, ...]
    param_names: tuple[str, ...]
    eval_fn: Callable[[np.float64, Float64Array], Float64Array]

    def bind(
        self, params: Mapping[str, object]
    ) -> Callable[[np.float64, Float64Array], Float64Array]:
        """Bind parameter values and return a 2-arg RHS: rhs(t, y) -> dydt.

        Args:
            params: Mapping of parameter names to values.

        Returns:
            A callable `rhs(t, y)` that evaluates the RHS with `params` fixed.
        """
        params_dict = dict(params)

        def rhs(t: np.float64, y: Float64Array) -> Float64Array:
            return self.eval_fn(t, y, **params_dict)

        return rhs


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
    # NumPy scalar math; keep small initially.
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
    """Parse a Python expression and return the AST.

    Args:
        expr: Expression string.

    Returns:
        Parsed AST for the expression.
    """
    try:
        return ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover
        _raise_invalid_expression(detail=f"invalid expression syntax: {exc.msg}")


def _validate_ast(tree: ast.AST, *, expr: str) -> None:
    """Validate that an expression AST only contains allowed constructs.

    Args:
        tree: Parsed expression AST.
        expr: Original expression string, for diagnostics.
    """
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            _raise_invalid_expression(
                detail=f"disallowed AST node {type(node).__name__} in {expr!r}"
            )

        # Explicitly reject constructs that could slip through via allowed nodes.
        if isinstance(node, ast.Name) and node.id in {"__import__", "__builtins__"}:
            _raise_invalid_expression(detail=f"disallowed name: {node.id!r}")

        if isinstance(node, ast.Attribute) and (
            not isinstance(node.value, ast.Name) or node.value.id != "np"
        ):
            _raise_invalid_expression(
                detail=f"{DISALLOWED_ATTRIBUTE_ACCESS} in {expr!r}"
            )

        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Attribute):
                _raise_invalid_expression(
                    detail=f"{ONLY_ATTRIBUTE_CALLS_ALLOWED} in {expr!r}"
                )
            if not isinstance(func.value, ast.Name):
                _raise_invalid_expression(detail=f"invalid call root in {expr!r}")

            root = str(func.value.id)
            name = str(func.attr)
            if root not in _ALLOWED_CALL_ROOTS or name not in _ALLOWED_CALL_FUNCS:
                _raise_invalid_expression(
                    detail=f"{DISALLOWED_FUNCTION_CALL}: {root}.{name}"
                )


def _compile_expr(expr: str) -> CodeType:
    """Parse, validate, and compile an expression into a code object.

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
        _raise_compilation_error(
            detail=f"failed to compile expression {expr!r}: {exc!r}"
        )


def _collect_alias_code(aliases: Mapping[str, str]) -> dict[str, CodeType]:
    """Compile alias expressions into code objects.

    Args:
        aliases: Mapping from alias name to expression string.

    Returns:
        Dictionary mapping alias name to compiled code object.
    """
    return {name: _compile_expr(expr) for name, expr in aliases.items()}


def _collect_eq_code(equations: tuple[str, ...]) -> list[CodeType]:
    """Compile equation expressions into code objects.

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
                val = eval(  # noqa: S307
                    codeobj, {"__builtins__": {}}, {**base_env, **out}
                )
            except NameError:
                continue
            except (ValueError, TypeError, ArithmeticError) as exc:
                _raise_invalid_expression(
                    detail=f"alias {name!r} evaluation failed: {exc!r}"
                )
            out[name] = val
            del pending[name]
            progressed = True
        if not pending:
            return out
        if not progressed:
            break

    _raise_invalid_expression(
        detail=f"could not resolve alias dependencies: {sorted(pending.keys())}"
    )


def _make_eval_fn(
    *,
    state_names: tuple[str, ...],
    aliases: Mapping[str, str],
    equations: tuple[str, ...],
) -> Callable[[np.float64, Float64Array], Float64Array]:
    n_state = len(state_names)
    name_to_idx = {s: i for i, s in enumerate(state_names)}
    alias_code = _collect_alias_code(aliases)
    eq_code = _collect_eq_code(equations)

    def eval_fn(t: np.float64, y: Float64Array, **params: object) -> Float64Array:
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1:
            _raise_state_shape_error(expected="1D array", got=y_arr.shape)
        if y_arr.size != n_state:
            _raise_state_shape_error(expected=f"(n_state={n_state},)", got=y_arr.shape)

        env: dict[str, object] = {"np": np, "t": np.float64(t)}
        for s, i in name_to_idx.items():
            env[s] = np.float64(y_arr[i])
        env.update(params)

        if alias_code:
            env.update(_resolve_aliases(alias_code, base_env=env))

        out = np.empty((n_state,), dtype=np.float64)
        for i, codeobj in enumerate(eq_code):
            try:
                val = eval(codeobj, {"__builtins__": {}}, env)  # noqa: S307
            except NameError as exc:
                _raise_parameter_error(detail=f"unknown symbol in equation: {exc!s}")
            except (ValueError, TypeError, ArithmeticError) as exc:
                _raise_invalid_expression(detail=f"equation evaluation failed: {exc!r}")
            out[i] = np.float64(val)

        return out

    return eval_fn


# -----------------------------------------------------------------------------
# Public compile API
# -----------------------------------------------------------------------------


def compile_rhs(rhs: NormalizedRhs) -> CompiledRhs:
    """Compile a normalized RHS into a runnable evaluation function.

    Args:
        rhs: Normalized RHS produced by `op_system.specs.normalize_rhs`.

    Returns:
        A `CompiledRhs` containing an `eval_fn(t, y, **params) -> dydt`.
    """
    if rhs.kind not in {"expr", "transitions"}:
        _raise_unsupported_feature(
            feature=f"rhs.kind={rhs.kind}",
            detail="Only 'expr' and 'transitions' are supported in v1.",
        )

    eval_fn = _make_eval_fn(
        state_names=rhs.state_names,
        aliases=rhs.aliases,
        equations=rhs.equations,
    )

    return CompiledRhs(
        state_names=rhs.state_names,
        param_names=rhs.param_names,
        eval_fn=eval_fn,
    )
