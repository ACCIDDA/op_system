"""Bounded scalar IR derivatives and safe Julia expression interoperability.

Derivatives describe the declared physical coordinates at regular domain points.
They do not include initial-state sensitivities, solve an ODE or establish global
identifiability. Native evaluation reuses the compiler's restricted evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from op_system._array import array_namespace
from op_system._ir import Apply, Expr, Literal, Sym, free_symbols, unparse_ir
from op_system._ir_templates import inline_aliases
from op_system.compile import _make_eval_fn
from op_system.identity import StructuralModelIdentity

if TYPE_CHECKING:
    from op_system.compile import CompiledRhs, EvalFn

__all__ = [
    "DerivativeValues",
    "JuliaExport",
    "SymbolicDerivatives",
    "export_julia",
    "julia_rhs",
    "symbolic_derivatives",
]

_ZERO = Literal(0)
_ONE = Literal(1)
_UNARY = {"np.exp", "np.log", "np.sqrt", "np.sin", "np.cos", "np.tanh"}


class DerivativeValues(NamedTuple):
    """Borrow four native matrices without copying arrays or detaching graphs."""

    state: object
    parameters: object
    observable_state: object
    observable_parameters: object


@dataclass(frozen=True, slots=True)
class SymbolicDerivatives:
    """Ordered dense derivative artifacts for a bounded scalar-expression subset.

    Shapes are (states, states), (states, parameters), (observables, states) and
    (observables, parameters). Structural dependency flags are independent of
    fitted parameter values. ``block_axes`` only repeats existing checked compiler
    declarations; no second separability analysis is performed.
    """

    model: StructuralModelIdentity
    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    expressions: tuple[tuple[str, ...], ...]
    state_dependencies: tuple[tuple[bool, ...], ...]
    block_axes: tuple[str, ...]
    _evaluate: EvalFn = field(repr=False, compare=False)

    def evaluate(self, t: object, y: object, **parameters: object) -> DerivativeValues:
        """Evaluate all four matrices on native NumPy/JAX/Torch scalar bindings.

        Returns:
            Native matrices in the declared state/parameter/observable order.

        Raises:
            ValueError: Parameter or output layout differs.
            TypeError: States do not have a native real floating dtype.
        """
        if set(parameters) != set(self.parameter_names):
            msg = "Derivative parameter names must match the declared order/layout"
            raise ValueError(msg)
        xp = array_namespace(y)
        if not xp.isdtype(getattr(y, "dtype", None), "real floating"):
            msg = "Derivative states require a native real floating array"
            raise TypeError(msg)
        values = self._evaluate(t, y, **parameters)
        if len(values.shape) != 1:
            msg = "Derivative parameters must be scalar; batch through native vmap"
            raise ValueError(msg)
        states, params, observed = (
            len(self.state_names),
            len(self.parameter_names),
            len(self.observable_names),
        )
        shapes = (
            (states, states),
            (states, params),
            (observed, states),
            (observed, params),
        )
        result = []
        offset = 0
        for shape in shapes:
            size = shape[0] * shape[1]
            result.append(xp.reshape(values[offset : offset + size], shape))
            offset += size
        return DerivativeValues(*result)

    def jvp(
        self, t: object, y: object, tangent: object, **parameters: object
    ) -> object:
        """Apply the bounded dense state Jacobian.

        Returns:
            Native state-direction action; storage is dense, not matrix-free.

        Raises:
            ValueError: The tangent shape differs from the state layout.
        """
        matrix: Any = self.evaluate(t, y, **parameters).state
        if getattr(tangent, "shape", None) != (len(self.state_names),):
            msg = "State tangent shape differs from derivative layout"
            raise ValueError(msg)
        if array_namespace(tangent) is not array_namespace(y) or getattr(
            tangent, "dtype", None
        ) != getattr(y, "dtype", None):
            msg = "Tangent namespace/precision differs from the state"
            raise ValueError(msg)
        return matrix @ tangent

    def vjp(
        self, t: object, y: object, cotangent: object, **parameters: object
    ) -> object:
        """Apply the transpose state Jacobian, retaining native derivative graphs.

        Returns:
            Native transpose action in the original state coordinates.

        Raises:
            ValueError: The cotangent shape differs from the state layout.
        """
        matrix: Any = self.evaluate(t, y, **parameters).state
        if getattr(cotangent, "shape", None) != (len(self.state_names),):
            msg = "State cotangent shape differs from derivative layout"
            raise ValueError(msg)
        xp = array_namespace(y)
        if array_namespace(cotangent) is not xp or getattr(
            cotangent, "dtype", None
        ) != getattr(y, "dtype", None):
            msg = "Cotangent namespace/precision differs from the state"
            raise ValueError(msg)
        return xp.matrix_transpose(matrix) @ cotangent


def _scalar_sources(
    compiled: CompiledRhs, *, max_states: int
) -> tuple[tuple[Expr, ...], tuple[tuple[str, Expr], ...], tuple[str, ...]]:
    if (
        type(max_states) is not int
        or max_states < 1
        or len(compiled.state_names) > max_states
    ):
        msg = "Symbolic export state limit exceeded or invalid"
        raise ValueError(msg)
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    if rhs is None:
        msg = "Symbolic export requires retained compiler source IR"
        raise NotImplementedError(msg)
    if (
        compiled.operators
        or compiled.history_requirements
        or compiled.gauge_fix is not None
        or rhs.shaped_params
        or rhs.time_varying_params
    ):
        msg = (
            "Scalar export requires natural IR without operators/history/shaped inputs"
        )
        raise NotImplementedError(msg)
    expressions = tuple(rhs.equations_ir)
    if len(expressions) != len(compiled.state_names) or any(
        expr is None for expr in expressions
    ):
        msg = "Symbolic scalar export requires complete expanded equation IR"
        raise NotImplementedError(msg)
    equations = tuple(
        inline_aliases(expr, rhs.aliases_ir) for expr in expressions if expr is not None
    )
    observed = []
    parameters = list(compiled.param_names)
    for artifact in rhs.observables_ir:
        if (
            artifact.definition.axes
            or artifact.definition.reduction
            or artifact.measures
            or any(axes for _, axes in artifact.parameter_axes)
        ):
            msg = (
                "Symbolic observable export currently requires scalar unreduced outputs"
            )
            raise NotImplementedError(msg)
        observed.append((
            artifact.definition.name,
            inline_aliases(artifact.expression, rhs.aliases_ir),
        ))
        parameters.extend(
            name for name, _ in artifact.parameter_axes if name not in parameters
        )
    allowed = {*compiled.state_names, *parameters, "t"}
    for expression in (*equations, *(expr for _, expr in observed)):
        _validate_expression(expression, allowed)
    return equations, tuple(observed), tuple(parameters)


def _validate_expression(expression: Expr, allowed: set[str]) -> None:
    stack = [(expression, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > 100_000 or depth > 64:
            msg = "Symbolic expression exceeds the bounded node/depth contract"
            raise ValueError(msg)
        if (
            isinstance(node, Literal)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and math.isfinite(node.value)
        ):
            continue
        if isinstance(node, Sym) and node.name in allowed:
            continue
        if isinstance(node, Apply):
            arity = 1 if node.op in {"neg", "pos", *_UNARY} else 2
            if (
                node.op in {"+", "-", "*", "/", "pow", "neg", "pos", *_UNARY}
                and len(node.args) == arity
            ):
                if node.op == "pow" and _exponent(node.args[1]) is None:
                    msg = "Symbolic powers require a literal real exponent"
                    raise NotImplementedError(msg)
                stack.extend((child, depth + 1) for child in node.args)
                continue
        msg = f"Unsupported symbolic scalar IR node/operation: {node!r}"
        raise NotImplementedError(msg)


def _op(  # ruff: ignore[too-many-return-statements]
    name: str, *arguments: Expr
) -> Expr:
    if name in {"+", "-"} and arguments[1] == _ZERO:
        return arguments[0]
    if name == "+" and arguments[0] == _ZERO:
        return arguments[1]
    if name == "*":
        if _ZERO in arguments:
            return _ZERO
        if arguments[0] == _ONE:
            return arguments[1]
        if arguments[1] == _ONE:
            return arguments[0]
    if name == "/" and arguments[0] == _ZERO:
        return _ZERO
    return Apply(name, arguments)


def _differentiate(  # ruff: ignore[complex-structure, too-many-return-statements]
    expression: Expr, name: str
) -> Expr:
    if name not in free_symbols(expression):
        return _ZERO
    if isinstance(expression, Sym):
        return _ONE
    if not isinstance(expression, Apply):
        msg = "Unsupported derivative node"
        raise NotImplementedError(msg)
    a = expression.args[0]
    da = _differentiate(a, name)
    op = expression.op
    if op in {"pos", "neg"}:
        return _op(op, da)
    if op in {"+", "-", "*", "/"}:
        b = expression.args[1]
        db = _differentiate(b, name)
        if op in {"+", "-"}:
            return _op(op, da, db)
        if op == "*":
            return _op("+", _op("*", da, b), _op("*", a, db))
        return _op(
            "/", _op("-", _op("*", da, b), _op("*", a, db)), _op("pow", b, Literal(2))
        )
    if op == "pow":
        exponent = _exponent(expression.args[1])
        if exponent is None:
            msg = "Unsupported symbolic exponent"
            raise NotImplementedError(msg)
        if exponent == 0:
            return _ZERO
        return _op(
            "*", _op("*", Literal(exponent), _op("pow", a, Literal(exponent - 1))), da
        )
    factor = {
        "np.exp": lambda: _op("np.exp", a),
        "np.log": lambda: _op("/", _ONE, a),
        "np.sqrt": lambda: _op("/", _ONE, _op("*", Literal(2), _op("np.sqrt", a))),
        "np.sin": lambda: _op("np.cos", a),
        "np.cos": lambda: _op("neg", _op("np.sin", a)),
        "np.tanh": lambda: _op("-", _ONE, _op("pow", _op("np.tanh", a), Literal(2))),
    }[op]()
    return _op("*", factor, da)


def symbolic_derivatives(
    compiled: CompiledRhs, *, max_states: int = 128, max_entries: int = 65_536
) -> SymbolicDerivatives:
    """Export full state/parameter/observable derivatives for the supported IR.

    Size checks precede dense derivative construction. Unsupported operations
    fail explicitly; numerical or AD fallbacks remain a caller choice.

    Returns:
        Immutable ordered expressions and a native evaluator.

    Raises:
        ValueError: A dense artifact exceeds its declared size limits.
    """
    equations, observed, parameters = _scalar_sources(compiled, max_states=max_states)
    states = compiled.state_names
    entries = (len(states) + len(observed)) * (len(states) + len(parameters))
    if type(max_entries) is not int or max_entries < 1 or entries > max_entries:
        msg = "Symbolic dense derivative entry limit exceeded or invalid"
        raise ValueError(msg)
    groups = (
        tuple(_differentiate(expr, name) for expr in equations for name in states),
        tuple(_differentiate(expr, name) for expr in equations for name in parameters),
        tuple(_differentiate(expr, name) for _, expr in observed for name in states),
        tuple(
            _differentiate(expr, name) for _, expr in observed for name in parameters
        ),
    )
    expressions = tuple(tuple(unparse_ir(expr) for expr in group) for group in groups)
    flat = tuple(expr for group in groups for expr in group)
    evaluator = _make_eval_fn(
        state_names=states,
        aliases={},
        equations=tuple(unparse_ir(expr) for expr in flat),
        equations_ir=flat,
    )
    dependencies = tuple(
        tuple(name in free_symbols(expr) for name in states) for expr in equations
    )
    return SymbolicDerivatives(
        StructuralModelIdentity.from_compiled(compiled),
        states,
        parameters,
        tuple(name for name, _ in observed),
        expressions,
        dependencies,
        tuple(info.name for info in compiled.block_axes),
        evaluator,
    )


def _julia(expression: Expr, bindings: dict[str, str]) -> str:
    if isinstance(expression, Literal) and type(expression.value) in {int, float}:
        return repr(expression.value)
    if isinstance(expression, Sym):
        return bindings[expression.name]
    if not isinstance(expression, Apply):
        msg = "Unsupported Julia expression"
        raise NotImplementedError(msg)
    args = tuple(_julia(arg, bindings) for arg in expression.args)
    if expression.op in {"pos", "neg"}:
        return f"({'+' if expression.op == 'pos' else '-'}{args[0]})"
    if expression.op.startswith("np."):
        return f"{expression.op[3:]}({args[0]})"
    operator = "^" if expression.op == "pow" else expression.op
    return f"({args[0]} {operator} {args[1]})"


@dataclass(frozen=True, slots=True)
class JuliaExport:
    """Bounded generated expressions and explicit external-runtime bindings.

    ``t`` is the known independent time input. Initial vectors are supplied by
    the consumer in ``state_names`` order. Complete declarations remain in the
    companion PortableIR document. ``rational`` describes only the supported
    expression operations, not an external-tool identifiability conclusion.
    """

    model: StructuralModelIdentity
    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    code: str
    rational: bool
    emitter_schema_version: int = field(default=1, init=False)


def export_julia(compiled: CompiledRhs, *, max_states: int = 128) -> JuliaExport:
    """Emit a bounded RHS function with explicit u/p/t binding order.

    Only compiler-validated scalar IR emits code; original names never become
    Julia identifiers or executable interpolation. Initial states remain explicit
    caller inputs. The structural identity and ordered bindings are retained in
    the separate PortableIR document; emission is not a rank/tool certificate.

    Returns:
        Generated code, source identity, ordered bindings and rational-subset flag.
    """
    equations, observed, parameters = _scalar_sources(compiled, max_states=max_states)
    bindings = {
        **{name: f"u[{i + 1}]" for i, name in enumerate(compiled.state_names)},
        **{name: f"p[{i + 1}]" for i, name in enumerate(parameters)},
        "t": "t",
    }
    rhs_text = (
        "function op_system_rhs(u, p, t)\n    return ["
        + ", ".join(_julia(expr, bindings) for expr in equations)
        + "]\nend\n"
    )
    observed_text = (
        "function op_system_observations(u, p, t)\n    return ["
        + ", ".join(_julia(expr, bindings) for _, expr in observed)
        + "]\nend\n"
    )
    rational = all(
        not (
            isinstance(node, Apply)
            and (
                node.op.startswith("np.")
                or (
                    node.op == "pow"
                    and (value := _exponent(node.args[1])) is not None
                    and value != int(value)
                )
            )
        )
        for expression in (*equations, *(expr for _, expr in observed))
        for node in _walk_scalar(expression)
    )
    return JuliaExport(
        StructuralModelIdentity.from_compiled(compiled),
        compiled.state_names,
        parameters,
        tuple(name for name, _ in observed),
        rhs_text + observed_text,
        rational,
    )


def julia_rhs(compiled: CompiledRhs, *, max_states: int = 128) -> str:
    """Return generated Julia code; prefer export_julia to retain its bindings.

    Returns:
        RHS and observation functions in the supported scalar subset.
    """
    return export_julia(compiled, max_states=max_states).code


def _walk_scalar(expression: Expr) -> tuple[Expr, ...]:
    result = []
    stack = [expression]
    while stack:
        node = stack.pop()
        result.append(node)
        if isinstance(node, Apply):
            stack.extend(node.args)
    return tuple(result)


def _exponent(expression: Expr) -> int | float | None:
    if (
        isinstance(expression, Literal)
        and isinstance(expression.value, (int, float))
        and not isinstance(expression.value, bool)
    ):
        return expression.value
    if (
        isinstance(expression, Apply)
        and expression.op in {"neg", "pos"}
        and len(expression.args) == 1
    ):
        value = _exponent(expression.args[0])
        if value is not None:
            return -value if expression.op == "neg" else value
    return None
