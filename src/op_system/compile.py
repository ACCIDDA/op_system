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
import importlib
import warnings
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    Protocol,
    SupportsFloat,
    SupportsIndex,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from op_system._block_axes import BlockAxisInfo, analyze_block_axes
from op_system._errors import InvalidExpressionError, UnsupportedFeatureError
from op_system._ir import (
    Expr,
    HistoryOp,
    extract_common_subexpressions,
    ir_to_ast_expr,
    unparse_ir,
    walk,
)
from op_system._normalize import ExprRhs, TransitionsRhs
from op_system._operators import OperatorDescriptor
from op_system._symbols import parse_expression_string

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from types import CodeType

    from .specs import NormalizedRhs

Float64Array = NDArray[np.float64]
StateDict = dict[str, Float64Array]
ScalarLike = SupportsFloat | SupportsIndex | str | bytes | None
_SAFE_BUILTINS = {"__import__": __import__}

_NUMERIC_DTYPE_KINDS = frozenset({"b", "i", "u", "f", "c"})


class _Indexable(Protocol):
    def __getitem__(self, idx: int | slice) -> object: ...


def _namespace_of(y: object) -> Any:  # noqa: ANN401
    """Return the Array-API namespace of ``y``.

    Raises:
        TypeError: If ``y`` does not implement ``__array_namespace__``.
            NumPy >= 2.0 arrays, JAX arrays (concrete and traced),
            and PyTorch tensors (via array-api compat) all qualify.
    """
    ns_fn = getattr(y, "__array_namespace__", None)
    if ns_fn is None:
        msg = (
            "op_system eval_fn requires Array-API arrays for `y` "
            "(NumPy >= 2.0, JAX, PyTorch). Got "
            f"{type(y).__name__!r} which does not implement "
            "__array_namespace__()."
        )
        raise TypeError(msg)
    return ns_fn()


def _check_numeric_dtype(xp: Any, dtype: object) -> None:  # noqa: ANN401
    """Validate that ``dtype`` is numeric in the Array-API sense.

    Mirrors :meth:`flepimop2.parameter.abc.ParameterValue.__post_init__`:
    use ``xp.isdtype(dtype, "numeric")`` and fall back to ``dtype.kind``
    for namespaces that predate ``isdtype``. Does *not* coerce.

    Raises:
        TypeError: If ``dtype`` is not a numeric Array-API dtype.
    """
    isdtype = getattr(xp, "isdtype", None)
    if isdtype is not None:
        try:
            if isdtype(dtype, "numeric"):
                return
        except (TypeError, ValueError):
            pass
    kind = getattr(dtype, "kind", None)
    if kind in _NUMERIC_DTYPE_KINDS:
        return
    msg = f"op_system requires numeric array dtype, got {dtype!r}"
    raise TypeError(msg)


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
        UnsupportedFeatureError: Always.
    """
    raise UnsupportedFeatureError(feature=feature, detail=detail)


# -----------------------------------------------------------------------------
# Public compiled RHS container
# -----------------------------------------------------------------------------
class EvalFn(Protocol):
    """Callable RHS evaluator supporting runtime parameter kwargs.

    Accepts a flat ``(n_state,)`` state array and returns a flat
    ``(n_state,)`` derivative array in the same array namespace.
    """

    def __call__(  # noqa: D102
        self, t: object, y: object, **params: object
    ) -> Float64Array: ...


class PytreeEvalFn(Protocol):
    """Callable RHS evaluator operating on shaped PyTree state dicts.

    Accepts ``y`` as a ``StateDict`` (mapping from state-template base name
    to a shaped array with the template's natural N-D shape) and returns a
    ``StateDict`` of the same structure containing the derivative.
    Enables the engine to skip the flatten/unflatten step entirely and
    expose the full tensor structure to JAX/XLA.
    """

    def __call__(  # noqa: D102
        self, t: object, y: StateDict, **params: object
    ) -> StateDict: ...


@dataclass(frozen=True, slots=True)
class CompiledRhs:
    """Container for a compiled RHS evaluation function.

    Instances produced by :func:`compile_rhs` retain a private reference to
    their source :class:`NormalizedRhs` so the container can be pickled and
    re-hydrated by re-running the compile pipeline on load. ``eval_fn``
    itself is a closure (and on the vectorized path captures compiled code
    objects), so it is dropped from the pickle and rebuilt by
    :func:`compile_rhs` in :meth:`__setstate__`. Round-tripping a
    ``CompiledRhs`` therefore costs one compile on load and yields a
    functionally equivalent instance whose ``eval_fn`` produces identical
    outputs for identical inputs.
    """

    state_names: tuple[str, ...]
    param_names: tuple[str, ...]
    eval_fn: EvalFn
    meta: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    operators: tuple[OperatorDescriptor, ...] = field(default_factory=tuple)
    factorize_axes: tuple[str, ...] = field(default_factory=tuple)
    # ``block_axes`` is set by :func:`compile_rhs` for every axis listed in
    # ``spec["factorize_axes"]`` that passes the IR separability check.
    # Excluded from equality and hash (it is derived from ``_rhs``) so that
    # two ``CompiledRhs`` instances compiled from the same ``NormalizedRhs``
    # continue to compare equal regardless of their ``block_axes`` content.
    block_axes: tuple[BlockAxisInfo, ...] = field(
        default_factory=tuple, compare=False, hash=False
    )
    # ``pytree_eval_fn`` is set when the vectorized compile path succeeds.
    # It accepts a ``StateDict`` (base → shaped array) and returns a
    # ``StateDict`` of derivatives, avoiding the flatten/unflatten step.
    # Excluded from equality, repr, and hash — it is a derived closure
    # rebuilt during ``__setstate__`` from ``_rhs``.
    pytree_eval_fn: PytreeEvalFn | None = field(
        default=None, repr=False, compare=False, hash=False
    )
    # ``template_shapes`` maps each state-template base name to its N-D shape.
    # Set when the vectorized compile path succeeds; ``None`` for the scalar
    # fallback.  Enables engines to build a PyTree y0 without knowing
    # vectorize internals.  Excluded from equality and hash (it is derived
    # from ``_rhs``).
    template_shapes: dict[str, tuple[int, ...]] | None = field(
        default=None, repr=False, compare=False, hash=False
    )
    # ``block_pytree_eval_fn`` is the PyTree eval fn compiled against the
    # block-stripped RHS (with the first declared factorize_axis removed).
    # Engines can vmap this over the block axis instead of calling the
    # monolithic ``pytree_eval_fn`` with axis-indexed state slices.
    # ``None`` when there are no factorize_axes or no pytree_eval_fn.
    block_pytree_eval_fn: PytreeEvalFn | None = field(
        default=None, repr=False, compare=False, hash=False
    )
    # ``block_template_shapes`` maps each state-template base name to its
    # per-block shape (block axis removed).  Mirrors ``template_shapes`` but
    # for the stripped compile.  ``None`` when ``block_pytree_eval_fn`` is
    # ``None``.
    block_template_shapes: dict[str, tuple[int, ...]] | None = field(
        default=None, repr=False, compare=False, hash=False
    )
    # Private: source spec retained for pickling. ``compile_rhs`` populates
    # this; direct constructions without ``_rhs`` are not picklable (the
    # ``eval_fn`` closure cannot be serialized) and will raise from
    # ``__getstate__``. Excluded from equality and repr so it doesn't leak
    # into the public surface.
    _rhs: NormalizedRhs | None = field(
        default=None, repr=False, compare=False, hash=False
    )

    def bind(
        self, params: Mapping[str, object]
    ) -> Callable[[object, object], Float64Array]:
        """Bind parameter values and return a 2-arg RHS: rhs(t, y) -> dydt.

        Args:
            params: Mapping of parameter names to values.

        Returns:
            A callable `rhs(t, y)` that evaluates the RHS with `params` fixed.
        """
        params_dict = dict(params)

        def rhs(t: object, y: object) -> Float64Array:
            return self.eval_fn(t, y, **params_dict)

        return rhs

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state.

        The compiled ``eval_fn`` is a closure (and on the vectorized path
        captures compiled :class:`types.CodeType` objects), which is not
        portably picklable. Instead we serialize just the source
        :class:`NormalizedRhs` and let :meth:`__setstate__` recompile.

        Raises:
            TypeError: If the source ``NormalizedRhs`` was not retained
                (i.e. the instance was constructed directly rather than
                via :func:`compile_rhs`).
        """
        if self._rhs is None:
            msg = (
                "CompiledRhs is not picklable: the source NormalizedRhs was "
                "not retained. Construct via compile_rhs() to produce a "
                "picklable CompiledRhs."
            )
            raise TypeError(msg)
        return {"_rhs": self._rhs}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore by recompiling from the pickled :class:`NormalizedRhs`."""
        rhs = state["_rhs"]
        rebuilt = compile_rhs(rhs)
        # frozen+slots dataclass: bypass __setattr__ via object.__setattr__
        object.__setattr__(self, "state_names", rebuilt.state_names)
        object.__setattr__(self, "param_names", rebuilt.param_names)
        object.__setattr__(self, "eval_fn", rebuilt.eval_fn)
        object.__setattr__(self, "meta", rebuilt.meta)
        object.__setattr__(self, "operators", rebuilt.operators)
        object.__setattr__(self, "factorize_axes", rebuilt.factorize_axes)
        object.__setattr__(self, "block_axes", rebuilt.block_axes)
        object.__setattr__(self, "pytree_eval_fn", rebuilt.pytree_eval_fn)
        object.__setattr__(self, "template_shapes", rebuilt.template_shapes)
        object.__setattr__(self, "block_pytree_eval_fn", rebuilt.block_pytree_eval_fn)
        object.__setattr__(self, "block_template_shapes", rebuilt.block_template_shapes)
        object.__setattr__(self, "_rhs", rhs)


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
    # Subscripts are needed for shaped-parameter buffer access (``theta[5]``)
    # and for ``np.array``-style index expressions emitted by template
    # substitution.
    ast.Subscript,
    ast.Index,
    ast.Tuple,
)

_ALLOWED_CALL_ROOTS: tuple[str, ...] = ("np",)
_ALLOWED_CALL_FUNCS: frozenset[str] = frozenset({
    # NumPy scalar math; keep small initially.
    "abs",
    "exp",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
    "sqrt",
    "maximum",
    "minimum",
    "clip",
    "where",
    # Trig and hyperbolic.
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    # Geometry-ish.
    "hypot",
    "arctan2",
})

_ALLOWED_HELPER_FUNCS: frozenset[str] = frozenset({"sum_state", "sum_prefix"})

_PLANNED_HISTORY_HELPERS: frozenset[str] = frozenset({
    "history",
    "delay",
    "convolve_history",
})


def _parse_expr(expr: str) -> ast.Expression:
    """Parse a Python expression and return the AST.

    Args:
        expr: Expression string.

    Returns:
        Parsed AST for the expression.
    """
    try:
        parsed = parse_expression_string(expr).ast
    except InvalidExpressionError as exc:
        _raise_invalid_expression(detail=exc.detail)
    return cast("ast.Expression", parsed)


def _validate_call(func: ast.AST, *, expr: str) -> None:
    if isinstance(func, ast.Attribute):
        if not isinstance(func.value, ast.Name):
            _raise_invalid_expression(detail=f"invalid call root in {expr!r}")
        root = str(func.value.id)
        name = str(func.attr)
        if root not in _ALLOWED_CALL_ROOTS or name not in _ALLOWED_CALL_FUNCS:
            _raise_invalid_expression(
                detail=f"{DISALLOWED_FUNCTION_CALL}: {root}.{name}"
            )
        return

    if isinstance(func, ast.Name):
        helper_name = str(func.id)
        if helper_name in _PLANNED_HISTORY_HELPERS:
            _raise_invalid_expression(
                detail=(
                    "history/delay operators are recognized but not yet "
                    f"implemented (helper={helper_name!r}); see issue #173"
                )
            )
        if helper_name not in _ALLOWED_HELPER_FUNCS:
            _raise_invalid_expression(
                detail=f"{DISALLOWED_FUNCTION_CALL}: {helper_name}"
            )
        return

    _raise_invalid_expression(detail=f"{ONLY_ATTRIBUTE_CALLS_ALLOWED} in {expr!r}")


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

        if isinstance(node, ast.Name) and node.id in {"__import__", "__builtins__"}:
            _raise_invalid_expression(detail=f"disallowed name: {node.id!r}")

        if isinstance(node, ast.Attribute) and (
            not isinstance(node.value, ast.Name) or node.value.id != "np"
        ):
            _raise_invalid_expression(
                detail=f"{DISALLOWED_ATTRIBUTE_ACCESS} in {expr!r}"
            )
        if isinstance(node, ast.Call):
            _validate_call(node.func, expr=expr)


def _compile_expr(expr: str, expr_ir: Expr | None = None) -> CodeType:
    """Parse, validate, and compile an expression into a code object.

    Args:
        expr: Expression string.
        expr_ir: Optional typed IR expression to compile directly.

    Returns:
        Compiled code object.
    """
    tree = (
        ast.Expression(body=ir_to_ast_expr(expr_ir))
        if expr_ir is not None
        else _parse_expr(expr)
    )
    ast.fix_missing_locations(tree)
    _validate_ast(tree, expr=expr)
    try:
        return compile(tree, filename="<op_system>", mode="eval")
    except (ValueError, TypeError, SyntaxError) as exc:  # pragma: no cover
        _raise_compilation_error(
            detail=f"failed to compile expression {expr!r}: {exc!r}"
        )


def _collect_alias_code(
    aliases: Mapping[str, str],
    aliases_ir: Mapping[str, Expr] | None = None,
) -> dict[str, CodeType]:
    """Compile alias expressions into code objects.

    Args:
        aliases: Mapping from alias name to expression string.
        aliases_ir: Optional mapping from alias name to typed IR.

    Returns:
        Dictionary mapping alias name to compiled code object.
    """
    ir_map = aliases_ir or {}
    return {
        name: _compile_expr(expr, ir_map.get(name)) for name, expr in aliases.items()
    }


def _collect_eq_code(
    equations: tuple[str, ...],
    equations_ir: tuple[Expr | None, ...] | None = None,
    reserved_names: Iterable[str] = (),
) -> tuple[tuple[tuple[str, CodeType], ...], list[CodeType]]:
    """Compile equation expressions into code objects.

    Args:
        equations: Tuple of equation expression strings.
        equations_ir: Optional tuple of typed IR expressions, positionally
            aligned with ``equations``.
        reserved_names: Runtime names that generated CSE temporaries must avoid.

    Returns:
        CSE binding code objects plus equation code objects.
    """
    if (
        equations_ir
        and len(equations_ir) == len(equations)
        and all(expr is not None for expr in equations_ir)
    ):
        concrete_ir = tuple(expr for expr in equations_ir if expr is not None)
        bindings, rewritten = extract_common_subexpressions(
            concrete_ir,
            prefix="__op_system_cse_",
            reserved_names=reserved_names,
        )
        cse_code = tuple((name, _compile_expr(name, expr)) for name, expr in bindings)
        eq_code = [  # noqa: FURB140
            _compile_expr(expr_s, expr_ir)
            for expr_s, expr_ir in zip(equations, rewritten, strict=True)
        ]
        return cse_code, eq_code
    return (), [_compile_expr(expr) for expr in equations]


def _evaluate_cse_bindings(
    cse_code: tuple[tuple[str, CodeType], ...],
    *,
    env: dict[str, object],
) -> None:
    """Evaluate CSE temporaries into ``env`` before equation evaluation."""
    for name, codeobj in cse_code:
        try:
            env[name] = eval(  # noqa: S307
                codeobj,
                {"__builtins__": _SAFE_BUILTINS},
                env,
            )
        except NameError as exc:
            _raise_parameter_error(detail=f"unknown symbol in CSE binding: {exc!s}")
        except (ValueError, TypeError, ArithmeticError) as exc:
            _raise_invalid_expression(detail=f"CSE binding evaluation failed: {exc!r}")


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
                    codeobj,
                    {"__builtins__": _SAFE_BUILTINS},
                    {**base_env, **out},
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


def _validate_state_vector(y_arr: object, *, n_state: int) -> object:
    """Validate shape of the state vector.

    Returns:
        State vector coerced to shape (n_state,).
    """
    expected_shape = (n_state,)
    shape = getattr(y_arr, "shape", None)
    shape_tuple = tuple(shape) if shape is not None else None
    if shape_tuple != expected_shape:
        _raise_state_shape_error(expected=f"(n_state={n_state},)", got=shape)
    return y_arr


def _evaluate_equations(
    *,
    eq_code: list[CodeType],
    env: Mapping[str, object],
    xp: Any,  # noqa: ANN401
) -> Float64Array:
    """Evaluate equation code objects against an environment.

    Returns:
        Derivative vector aligned to the provided state ordering, in the
        same array namespace as ``xp`` (i.e. as the input ``y``).
    """
    out_vals: list[object] = []
    for codeobj in eq_code:
        try:
            val = eval(codeobj, {"__builtins__": _SAFE_BUILTINS}, env)  # noqa: S307
        except NameError as exc:
            _raise_parameter_error(detail=f"unknown symbol in equation: {exc!s}")
        except (ValueError, TypeError, ArithmeticError) as exc:
            _raise_invalid_expression(detail=f"equation evaluation failed: {exc!r}")
        out_vals.append(val)

    return cast("Float64Array", xp.asarray(out_vals))


def _make_eval_fn(
    *,
    state_names: tuple[str, ...],
    aliases: Mapping[str, str],
    equations: tuple[str, ...],
    aliases_ir: Mapping[str, Expr] | None = None,
    equations_ir: tuple[Expr | None, ...] | None = None,
) -> EvalFn:
    """Build a namespace-polymorphic ``eval_fn(t, y, **params) -> dydt``.

    The compiled function infers its array namespace from the input ``y``
    via :meth:`y.__array_namespace__` at call time. No coercion is
    performed: producers hand in arrays of the desired backend and dtype,
    and outputs come back in that same namespace. This keeps the function
    natively callable under ``jax.jit`` / ``jax.vmap`` (tracers carry
    ``__array_namespace__``) without any wrapping for correctness.

    Returns:
        EvalFn: A callable ``(t, y, **params) -> dydt`` with the same
            namespace as ``y``.
    """
    n_state = len(state_names)
    name_to_idx = {s: i for i, s in enumerate(state_names)}
    alias_code = _collect_alias_code(aliases, aliases_ir)
    reserved_names = {*state_names, *aliases, "np", "t", "sum_state", "sum_prefix"}
    cse_code, eq_code = _collect_eq_code(
        equations,
        equations_ir,
        reserved_names=reserved_names,
    )

    def eval_fn(t: object, y: object, **params: object) -> Float64Array:
        xp = _namespace_of(y)
        _check_numeric_dtype(xp, getattr(y, "dtype", None))
        y_arr = _validate_state_vector(y, n_state=n_state)

        t_val = xp.asarray(t)
        env: dict[str, object] = {"np": xp, "t": t_val}
        for s, i in name_to_idx.items():
            env[s] = cast("_Indexable", y_arr)[i]
        env.update(params)

        def _sum_state() -> object:
            values = [v for k, v in env.items() if k in name_to_idx]
            if not values:
                return xp.asarray(0.0)
            return xp.sum(xp.stack(values))

        def _sum_prefix(prefix: object) -> object:
            pfx = str(prefix)
            values = [
                v for k, v in env.items() if k.startswith(pfx) and k in name_to_idx
            ]
            if not values:
                return xp.asarray(0.0)
            return xp.sum(xp.stack(values))

        env["sum_state"] = _sum_state
        env["sum_prefix"] = _sum_prefix

        if alias_code:
            env.update(_resolve_aliases(alias_code, base_env=env))
        if cse_code:
            _evaluate_cse_bindings(cse_code, env=env)

        return _evaluate_equations(eq_code=eq_code, env=env, xp=xp)

    return eval_fn


def _history_requirements_from_ir(
    *,
    aliases_ir: Mapping[str, Expr] | None,
    equations_ir: tuple[Expr | None, ...] | None,
) -> tuple[dict[str, object], ...]:
    """Collect structured history-operator requirements from typed IR.

    Returns:
        Tuple of requirement records, each containing helper kind, body,
        and normalized option expressions.
    """
    reqs: list[dict[str, object]] = []
    if aliases_ir:
        for alias_name, alias_expr in aliases_ir.items():
            for node in walk(alias_expr):
                if isinstance(node, HistoryOp):
                    reqs.append({
                        "scope": f"alias:{alias_name}",
                        "kind": node.kind,
                        "body": unparse_ir(node.body),
                        "options": {k: unparse_ir(v) for k, v in node.options},
                    })
    if equations_ir:
        for eq_idx, eq_expr in enumerate(equations_ir):
            if eq_expr is None:
                continue
            for node in walk(eq_expr):
                if isinstance(node, HistoryOp):
                    reqs.append({
                        "scope": f"equation:{eq_idx}",
                        "kind": node.kind,
                        "body": unparse_ir(node.body),
                        "options": {k: unparse_ir(v) for k, v in node.options},
                    })
    return tuple(reqs)


# -----------------------------------------------------------------------------
# Public compile API
# -----------------------------------------------------------------------------


def _interp_along_axis(
    t: object,
    ts: object,
    grid: object,
    *,
    axis: int,
    xp: Any,  # noqa: ANN401
) -> object:
    """Linearly interpolate ``grid`` along axis ``axis`` at scalar ``t``.

    For a 1-D ``grid`` of shape ``(N,)`` returns a scalar; for an N-D
    ``grid`` whose ``axis``-th dimension has length ``N`` returns an array
    with that dimension removed.  Boundary behaviour is constant
    extrapolation (clamp to the nearest end-point), matching
    :func:`numpy.interp`.

    Args:
        t: Scalar evaluation time.
        ts: 1-D monotonically non-decreasing array of grid times of length N.
        grid: Array whose ``axis``-th dimension indexes ``ts``.
        axis: Position of the time axis within ``grid``'s shape.
        xp: Array-API namespace, derived from the input ``y`` by the
            caller (``y.__array_namespace__()``).

    Returns:
        Interpolated value with ``grid``'s shape minus the ``axis`` slot.
    """
    ts_arr = xp.asarray(ts)
    grid_arr = xp.asarray(grid)
    if axis != 0:
        grid_arr = xp.moveaxis(grid_arr, axis, 0)
    n = ts_arr.shape[0]
    t_val = xp.asarray(t)
    raw_idx = xp.searchsorted(ts_arr, t_val, side="right")
    idx_right = xp.clip(raw_idx, 1, n - 1)
    idx_left = idx_right - 1
    t_left = ts_arr[idx_left]
    t_right = ts_arr[idx_right]
    span = t_right - t_left
    raw_w = (t_val - t_left) / xp.where(
        span == 0, xp.asarray(1.0, dtype=span.dtype), span
    )
    w = xp.clip(raw_w, 0.0, 1.0)
    g_left = grid_arr[idx_left]
    g_right = grid_arr[idx_right]
    if g_left.ndim > 0:
        w = xp.reshape(w, (1,) * g_left.ndim)
    return (1.0 - w) * g_left + w * g_right


def _wrap_eval_fn_for_time_varying(
    eval_fn: EvalFn,
    *,
    time_varying_params: tuple[tuple[str, tuple[str, ...]], ...],
    time_axis_name: str,
    axes_meta: tuple[Mapping[str, Any], ...],
) -> EvalFn:
    """Wrap ``eval_fn`` so each time-varying name is interpolated at runtime.

    For each ``(name, full_axes)`` in ``time_varying_params`` the wrapper
    expects a single kwarg ``name`` whose shape matches ``full_axes``.  At
    every call the wrapper interpolates that array along the time-axis
    position using the time axis's declared coordinates as the grid, and
    re-injects the reduced-shape result under ``name`` before delegating to
    ``eval_fn``.

    The array namespace used for interpolation is derived from the input
    ``y`` at call time, so the wrapped callable remains namespace-poly-
    morphic and trace-pure (works directly under ``jax.jit`` / ``jax.vmap``
    when called with JAX arrays).

    Args:
        eval_fn: The unwrapped evaluator returned by the vectorized or
            scalar compile path.
        time_varying_params: Pairs ``(name, full_axes)`` declared
            time-varying by the spec.  Each ``full_axes`` tuple must
            contain ``time_axis_name``.
        time_axis_name: The configured time-axis name (default ``"time"``).
        axes_meta: Normalized axes records (each a mapping with ``name``
            and ``coords``).  Used to look up the time-axis ``coords``
            array baked into the wrapper closure as the interpolation grid.

    Returns:
        A new `EvalFn` with the same shape contract as ``eval_fn``.
    """
    if not time_varying_params:
        return eval_fn
    # Bake the time-axis coords as plain numpy at compile time. ``xp.asarray``
    # at eval time will adopt them into the input namespace as needed.
    ts_lookup = {
        ax["name"]: np.asarray(ax["coords"], dtype=np.float64)
        for ax in axes_meta
        if ax.get("name") == time_axis_name
    }
    if time_axis_name not in ts_lookup:
        _raise_parameter_error(
            detail=(
                f"time-varying parameters declared but the configured time "
                f"axis {time_axis_name!r} is missing from the spec axes."
            )
        )
    ts = ts_lookup[time_axis_name]
    plan = tuple(
        (name, full_axes.index(time_axis_name))
        for name, full_axes in time_varying_params
    )

    def wrapped(t: object, y: object, **params: object) -> Float64Array:
        xp = _namespace_of(y)
        for name, axis_pos in plan:
            if name not in params:
                _raise_parameter_error(
                    detail=(
                        f"time-varying parameter {name!r} requires the bare "
                        f"{name!r} kwarg at call time."
                    )
                )
            grid = params.pop(name)
            params[name] = _interp_along_axis(t, ts, grid, axis=axis_pos, xp=xp)
        return eval_fn(t, y, **params)

    return wrapped


def _wrap_pytree_eval_fn_for_time_varying(
    eval_fn: PytreeEvalFn,
    *,
    time_varying_params: tuple[tuple[str, tuple[str, ...]], ...],
    time_axis_name: str,
    axes_meta: tuple[Mapping[str, Any], ...],
) -> PytreeEvalFn:
    """Like :func:`_wrap_eval_fn_for_time_varying` but for PyTree eval fns.

    The only difference from the flat version is that ``xp`` is obtained
    from the first value in the incoming state dict rather than from the
    state array directly.

    Args:
        eval_fn: Unwrapped PyTree evaluator.
        time_varying_params: Same as for the flat wrapper.
        time_axis_name: Configured time-axis name.
        axes_meta: Normalized axes metadata records.

    Returns:
        A new :class:`PytreeEvalFn` with the same PyTree shape contract.
    """
    if not time_varying_params:
        return eval_fn
    ts_lookup = {
        ax["name"]: np.asarray(ax["coords"], dtype=np.float64)
        for ax in axes_meta
        if ax.get("name") == time_axis_name
    }
    if time_axis_name not in ts_lookup:
        _raise_parameter_error(
            detail=(
                f"time-varying parameters declared but the configured time "
                f"axis {time_axis_name!r} is missing from the spec axes."
            )
        )
    ts = ts_lookup[time_axis_name]
    plan = tuple(
        (name, full_axes.index(time_axis_name))
        for name, full_axes in time_varying_params
    )

    def wrapped(t: object, y: StateDict, **params: object) -> StateDict:
        first_val = next(iter(y.values()))
        xp = _namespace_of(first_val)
        for name, axis_pos in plan:
            if name not in params:
                _raise_parameter_error(
                    detail=(
                        f"time-varying parameter {name!r} requires the bare "
                        f"{name!r} kwarg at call time."
                    )
                )
            grid = params.pop(name)
            params[name] = _interp_along_axis(t, ts, grid, axis=axis_pos, xp=xp)
        return eval_fn(t, y, **params)

    return wrapped


def _parse_operator_descriptors(
    meta: Mapping[str, Any],
) -> tuple[OperatorDescriptor, ...]:
    """Parse raw ``meta["operators"]`` into typed :class:`OperatorDescriptor` instances.

    Returns:
        Tuple of :class:`OperatorDescriptor` instances, empty if no operators.
    """
    ops = meta.get("operators")
    if not ops:
        return ()
    result: list[OperatorDescriptor] = []
    for op in ops:
        if not isinstance(op, _MappingABC):
            continue
        axis_raw = op.get("axis")
        if not isinstance(axis_raw, str) or not axis_raw:
            continue
        kind_raw = op.get("kind")
        bc_raw = op.get("bc")
        velocity_raw = op.get("velocity")
        rate_raw = op.get("rate")
        kernel_raw = op.get("kernel")
        kernel = (
            MappingProxyType(dict(kernel_raw))
            if isinstance(kernel_raw, _MappingABC)
            else None
        )
        result.append(
            OperatorDescriptor(
                axis=axis_raw,
                kind=kind_raw if isinstance(kind_raw, str) else None,
                bc=bc_raw if isinstance(bc_raw, str) else None,
                velocity=velocity_raw if isinstance(velocity_raw, str) else None,
                rate=rate_raw if isinstance(rate_raw, str) else None,
                kernel=kernel,
            )
        )
    return tuple(result)


def _parse_factorize_axes(meta: Mapping[str, Any]) -> tuple[str, ...]:
    """Parse ``meta["factorize_axes"]`` into a tuple of axis-name strings.

    Returns:
        Tuple of axis name strings; empty if not declared.
    """
    raw = meta.get("factorize_axes")
    if not raw:
        return ()
    return tuple(s for s in raw if isinstance(s, str))


def _wrap_eval_fn_for_synth_consts(
    eval_fn: EvalFn,
    synth_const_values: dict[str, Any],
) -> EvalFn:
    """Inject compile-time synthesized constants into every call to ``eval_fn``.

    Returns:
        A new :class:`EvalFn` that injects ``synth_const_values`` before
        delegating to the wrapped evaluator.
    """
    inner = eval_fn

    def wrapped(t: object, y: object, **params: object) -> Float64Array:
        # Cast synth-mask values to ``y``'s namespace and dtype so they
        # don't promote a float32 state buffer to float64 (or vice versa).
        xp = _namespace_of(y)
        y_dtype = getattr(y, "dtype", None)
        for k, v in synth_const_values.items():
            if k in params:
                continue
            if y_dtype is not None:
                params[k] = xp.asarray(v, dtype=y_dtype)
            else:
                params[k] = xp.asarray(v)
        return inner(t, y, **params)

    return wrapped


def _wrap_pytree_eval_fn_for_synth_consts(
    eval_fn: PytreeEvalFn,
    synth_const_values: dict[str, Any],
) -> PytreeEvalFn:
    """Like :func:`_wrap_eval_fn_for_synth_consts` but for PyTree eval fns.

    Returns:
        A new :class:`PytreeEvalFn` that injects ``synth_const_values`` before
        delegating to the wrapped evaluator.
    """
    inner = eval_fn

    def wrapped(t: object, y: StateDict, **params: object) -> StateDict:
        first_val = next(iter(y.values()))
        xp = _namespace_of(first_val)
        y_dtype = getattr(first_val, "dtype", None)
        for k, v in synth_const_values.items():
            if k in params:
                continue
            if y_dtype is not None:
                params[k] = xp.asarray(v, dtype=y_dtype)
            else:
                params[k] = xp.asarray(v)
        return inner(t, y, **params)

    return wrapped


def compile_rhs(rhs: NormalizedRhs, *, xp: object | None = None) -> CompiledRhs:  # noqa: C901, PLR0914
    """Compile a normalized RHS into a runnable evaluation function.

    Always uses the vectorized eval path that operates on shaped buffers
    (one tensor expression per state template) for specs that declare axes.
    Specs without axes (genuinely scalar models) fall back to the scalar path.
    Raising :class:`UnsupportedFeatureError` if an axis-indexed spec cannot be
    vectorized, rather than silently falling back to the catastrophically slow
    scalar path.

    The returned ``eval_fn`` is **namespace-polymorphic**: it infers its
    array namespace from the input ``y`` at call time
    (``y.__array_namespace__()``), and returns arrays in that same
    namespace. Calling it with JAX arrays (or tracers) yields a JAX-native
    computation suitable for ``jax.jit`` / ``jax.vmap`` without any
    correctness wrapping.

    Args:
        rhs: Normalized RHS produced by `op_system.specs.normalize_rhs`.
        xp: **Deprecated.** Formerly the compile-time array backend
            namespace. Now ignored — the namespace is resolved per call
            from the input ``y``. Will be removed in a future release.

    Returns:
        A `CompiledRhs` containing an `eval_fn(t, y, **params) -> dydt`.
        For axis-indexed specs the returned object also carries
        ``pytree_eval_fn`` and ``template_shapes``.  If the spec declares
        axes but the vectorizer cannot build a plan an
        ``UnsupportedFeatureError`` is raised (see bail reason in the detail
        message) rather than silently degrading to the scalar path.
    """
    if xp is not None:
        warnings.warn(
            "compile_rhs(xp=...) is deprecated and ignored. The compiled "
            "eval_fn now infers its array namespace from the input `y` at "
            "call time via __array_namespace__(); pass JAX arrays for a "
            "JAX-native call, NumPy arrays for a NumPy call. The `xp` "
            "kwarg will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not isinstance(rhs, (ExprRhs, TransitionsRhs)):
        _raise_unsupported_feature(
            feature=f"rhs type={type(rhs).__name__}",
            detail="Only ExprRhs and TransitionsRhs are supported in v1.",
        )

    history_requirements = _history_requirements_from_ir(
        aliases_ir=rhs.aliases_ir,
        equations_ir=rhs.equations_ir,
    )
    if history_requirements:
        _raise_unsupported_feature(
            feature="history operators",
            detail=(
                "History/delay operators require engine-managed state history "
                "buffers and are not implemented yet (issue #173). "
                f"history_requirements={history_requirements!r}"
            ),
        )

    # Lazy load via importlib to avoid a static circular import
    # (op_system._vectorize imports helpers from this module).
    vec = importlib.import_module("op_system._vectorize")
    plan = vec.build_vector_plan(rhs)

    if plan is None:
        # Specs that declare axes MUST vectorize.  Falling back to the scalar
        # path for an axis-indexed spec is catastrophically slow (O(N*M*...)
        # Python loop per eval call), breaks the PyTree interface
        # (pytree_eval_fn / template_shapes are absent), and silently hides
        # vectorizer regressions.  Raise loudly so the gap gets fixed.
        #
        # Exceptions:
        # - Genuinely scalar specs (no axes declared) have no tensor structure
        #   to exploit and the scalar path is correct for them.
        # - Specs that declare only a continuous time axis (for time-varying
        #   parameters) but have scalar state templates are functionally scalar.
        # - Mixed specs where some states lack axis wildcards (e.g. a scalar
        #   background compartment alongside axis-indexed states) bail with
        #   "scalar (non-wildcard) state template present"; this is a known
        #   limitation and the scalar path is the correct fallback.
        # Only fire the error for bail reasons that indicate an *unexpected*
        # vectorization failure (e.g. unsupported expression patterns).
        axes_meta_check = (
            rhs.meta.get("axes") if isinstance(rhs.meta, _MappingABC) else None
        )
        bail_reason = vec.last_vector_plan_bail_reason() or ""
        if (
            axes_meta_check
            and bail_reason != "scalar (non-wildcard) state template present"
        ):
            _raise_unsupported_feature(
                feature="vectorized eval path",
                detail=(
                    f"Spec declares axes but the vectorizer could not build a "
                    f"plan: {bail_reason}. All axis-indexed specs must use the "
                    f"vectorized path. If this expression pattern should be "
                    f"supported, please file an issue referencing issue #160."
                ),
            )

    eval_fn: EvalFn | None = (
        vec.make_vectorized_eval_fn(plan) if plan is not None else None
    )
    pytree_eval_fn: PytreeEvalFn | None = (
        vec.make_pytree_eval_fn(plan) if plan is not None else None
    )
    template_shapes: dict[str, tuple[int, ...]] | None = (
        {tpl.base: tpl.shape for tpl in plan.state_templates}
        if plan is not None
        else None
    )

    if eval_fn is None:
        eval_fn = _make_eval_fn(
            state_names=rhs.state_names,
            aliases=rhs.aliases,
            equations=rhs.equations,
            aliases_ir=rhs.aliases_ir,
            equations_ir=rhs.equations_ir,
        )

    time_axis_name = str(rhs.meta.get("time_axis", "time"))
    axes_meta = tuple(rhs.meta.get("axes") or ())
    eval_fn = _wrap_eval_fn_for_time_varying(
        eval_fn,
        time_varying_params=rhs.time_varying_params,
        time_axis_name=time_axis_name,
        axes_meta=axes_meta,
    )
    if pytree_eval_fn is not None:
        pytree_eval_fn = _wrap_pytree_eval_fn_for_time_varying(
            pytree_eval_fn,
            time_varying_params=rhs.time_varying_params,
            time_axis_name=time_axis_name,
            axes_meta=axes_meta,
        )

    # Inject normalize-time synthesized constants (e.g. one-hot masks for
    # pinned-coord transition selectors) into ``params`` so callers do not
    # need to supply them. ``setdefault`` lets user overrides win.
    synth_consts = rhs.meta.get("op_system_synth_constants")
    if isinstance(synth_consts, _MappingABC) and synth_consts:
        synth_const_values = dict(synth_consts)
        eval_fn = _wrap_eval_fn_for_synth_consts(eval_fn, synth_const_values)
        if pytree_eval_fn is not None:
            pytree_eval_fn = _wrap_pytree_eval_fn_for_synth_consts(
                pytree_eval_fn, synth_const_values
            )

    block_axes = analyze_block_axes(rhs)

    # ------------------------------------------------------------------
    # Block-stripped compile: produce a per-block-coord pytree_eval_fn by
    # stripping the first factorize axis from the RHS and re-running the
    # vectorizer.  Engines can jax.vmap this over the block axis instead
    # of baking literal axis indices that break under vmap.
    # ------------------------------------------------------------------
    block_pytree_eval_fn: PytreeEvalFn | None = None
    block_template_shapes: dict[str, tuple[int, ...]] | None = None
    if pytree_eval_fn is not None and block_axes:
        from op_system._normalize_block import strip_block_axis  # noqa: PLC0415

        stripped = strip_block_axis(rhs, block_axes[0].name)
        stripped_plan = vec.build_vector_plan(stripped)
        if stripped_plan is not None:
            raw_block_fn: PytreeEvalFn = vec.make_pytree_eval_fn(stripped_plan)
            block_template_shapes = {
                tpl.base: tpl.shape for tpl in stripped_plan.state_templates
            }
            stripped_time_axis = str(stripped.meta.get("time_axis", "time"))
            stripped_axes_meta = tuple(stripped.meta.get("axes") or ())
            raw_block_fn = _wrap_pytree_eval_fn_for_time_varying(
                raw_block_fn,
                time_varying_params=stripped.time_varying_params,
                time_axis_name=stripped_time_axis,
                axes_meta=stripped_axes_meta,
            )
            if isinstance(synth_consts, _MappingABC) and synth_consts:
                raw_block_fn = _wrap_pytree_eval_fn_for_synth_consts(
                    raw_block_fn, dict(synth_consts)
                )
            block_pytree_eval_fn = raw_block_fn

    return CompiledRhs(
        state_names=rhs.state_names,
        param_names=tuple(rhs.param_names),
        eval_fn=eval_fn,
        meta=rhs.meta,
        operators=_parse_operator_descriptors(rhs.meta),
        factorize_axes=_parse_factorize_axes(rhs.meta),
        block_axes=block_axes,
        pytree_eval_fn=pytree_eval_fn,
        template_shapes=template_shapes,
        block_pytree_eval_fn=block_pytree_eval_fn,
        block_template_shapes=block_template_shapes,
        _rhs=rhs,
    )
