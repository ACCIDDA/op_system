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
from dataclasses import dataclass, field
from itertools import starmap
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

from op_system._errors import InvalidExpressionError
from op_system._ir import Expr, extract_common_subexpressions, ir_to_ast_expr
from op_system._normalize import ExprRhs, TransitionsRhs
from op_system._symbols import parse_expression_string

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from types import CodeType

    from .specs import NormalizedRhs

Float64Array = NDArray[np.float64]
ScalarLike = SupportsFloat | SupportsIndex | str | bytes | None
_SAFE_BUILTINS = {"__import__": __import__}

_NUMERIC_DTYPE_KINDS = frozenset({"b", "i", "u", "f", "c"})


class _Indexable(Protocol):
    """Minimal indexable protocol used to type-check state-vector access."""

    def __getitem__(self, idx: int | slice) -> object:
        """Return the element at ``idx``."""
        ...


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
        NotImplementedError: Always.
    """
    msg = f"{UNSUPPORTED_FEATURE_PREFIX} Feature '{feature}' is not supported."
    if detail:
        msg = f"{msg} Detail: {detail}"
    raise NotImplementedError(msg)


# -----------------------------------------------------------------------------
# Public compiled RHS container
# -----------------------------------------------------------------------------
class EvalFn(Protocol):
    """Callable RHS evaluator supporting runtime parameter kwargs."""

    def __call__(self, t: object, y: object, **params: object) -> Float64Array:
        """Evaluate the RHS at time ``t`` and state ``y`` with bound parameters."""
        ...


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
            """Two-argument RHS with parameters bound by the enclosing call.

            Returns:
                ``dydt`` array in the namespace of ``y``.
            """
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
    """Validate that an AST call target is on the safe-eval allowlist.

    Permits attribute calls under whitelisted roots (currently ``np.``) and
    bare names that match registered helper functions.

    Args:
        func: ``func`` slot of an ``ast.Call`` node.
        expr: Original expression string, used in diagnostics.
    """
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
        eq_code = list(starmap(_compile_expr, zip(equations, rewritten, strict=True)))
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
        """Namespace-polymorphic compiled RHS body.

        Infers the array namespace from ``y`` at call time, builds the
        evaluation environment from state, parameters and aliases, and
        returns the equation outputs stacked in ``y``'s namespace.

        Returns:
            ``dydt`` array of shape ``(n_state,)`` in ``y``'s namespace.
        """
        xp = _namespace_of(y)
        _check_numeric_dtype(xp, getattr(y, "dtype", None))
        y_arr = _validate_state_vector(y, n_state=n_state)

        t_val = xp.asarray(t)
        env: dict[str, object] = {"np": xp, "t": t_val}
        for s, i in name_to_idx.items():
            env[s] = cast("_Indexable", y_arr)[i]
        env.update(params)

        def _sum_state() -> object:
            """Return the sum of all state variables (``sum_state()`` helper)."""
            values = [v for k, v in env.items() if k in name_to_idx]
            if not values:
                return xp.asarray(0.0)
            return xp.sum(xp.stack(values))

        def _sum_prefix(prefix: object) -> object:
            """Return the sum of state variables whose names start with ``prefix``."""
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
        """Interpolate each time-varying parameter at ``t`` then dispatch.

        Returns:
            ``dydt`` array produced by the wrapped ``eval_fn``.
        """
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


def compile_rhs(rhs: NormalizedRhs, *, xp: object | None = None) -> CompiledRhs:
    """Compile a normalized RHS into a runnable evaluation function.

    Always attempts the vectorized eval path that operates on shaped buffers
    (one tensor expression per state template) and falls back automatically
    to the scalar path when the spec falls outside the supported subset.

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

    Examples:
        >>> import numpy as np
        >>> from op_system.specs import normalize_rhs
        >>> rhs = normalize_rhs({
        ...     "kind": "expr",
        ...     "state": ["x"],
        ...     "equations": {"x": "2.0 * x"},
        ... })
        >>> compiled = compile_rhs(rhs)
        >>> y = np.array([3.0])
        >>> compiled.eval_fn(0.0, y)
        array([6.])
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

    # Lazy load via importlib to avoid a static circular import
    # (op_system._vectorize imports helpers from this module).
    vec = importlib.import_module("op_system._vectorize")
    plan = vec.build_vector_plan(rhs)
    eval_fn: EvalFn | None = (
        vec.make_vectorized_eval_fn(plan) if plan is not None else None
    )

    if eval_fn is None:
        eval_fn = _make_eval_fn(
            state_names=rhs.state_names,
            aliases=rhs.aliases,
            equations=rhs.equations,
            aliases_ir=rhs.aliases_ir,
            equations_ir=rhs.equations_ir,
        )

    eval_fn = _wrap_eval_fn_for_time_varying(
        eval_fn,
        time_varying_params=rhs.time_varying_params,
        time_axis_name=str(rhs.meta.get("time_axis", "time")),
        axes_meta=tuple(rhs.meta.get("axes") or ()),
    )

    return CompiledRhs(
        state_names=rhs.state_names,
        param_names=tuple(rhs.param_names),
        eval_fn=eval_fn,
        meta=rhs.meta,
        _rhs=rhs,
    )
