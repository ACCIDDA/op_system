"""Compile physical quotients for an explicit logarithmic state view."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING

from ._errors import UnsupportedFeatureError
from ._ir import Apply, Expr, Literal, Sym
from ._normalize import ExprRhs
from ._normalize_ir import _derive_equation_strings_lazy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .specs import NormalizedRhs


def _product(args: tuple[Expr, ...]) -> Expr:
    """Rebuild a product with exact unit factors removed.

    Returns:
        A product, its sole factor, or a literal unit.
    """
    selected = tuple(arg for arg in args if arg != Literal(1))
    if not selected:
        return Literal(1)
    return selected[0] if len(selected) == 1 else Apply("*", selected)


def _remove_state_factor(expr: Expr, target: Sym) -> Expr | None:
    """Cancel one syntactically exact factor on the positive-state domain.

    Returns:
        Residual expression, or None when no supported cancellation exists.
        Reduction bodies are kept intact; no algebraic identity is guessed.
    """
    if expr == target:
        return Literal(1)
    if expr == Literal(0):
        return Literal(0)
    if not isinstance(expr, Apply):
        return None
    return _remove_applied_factor(expr, target)


def _remove_applied_factor(expr: Apply, target: Sym) -> Expr | None:
    """Dispatch supported multiplicative identities.

    Returns:
        Residual expression, or None.
    """
    result: Expr | None = None
    if expr.op in {"neg", "pos"}:
        residual = _remove_state_factor(expr.args[0], target)
        if residual is not None:
            result = Apply(expr.op, (residual,))
    elif expr.op == "*":
        result = _remove_product_factor(expr.args, target)
    elif expr.op in {"+", "-"}:
        residuals = tuple(_remove_state_factor(arg, target) for arg in expr.args)
        if all(residual is not None for residual in residuals):
            result = Apply(
                expr.op, tuple(value for value in residuals if value is not None)
            )
    elif expr.op == "/":
        residual = _remove_state_factor(expr.args[0], target)
        if residual is not None:
            result = Apply("/", (residual, expr.args[1]))
    elif expr.op == "pow" and expr.args[0] == target:
        result = _remove_power_factor(expr.args[1], target)
    return result


def _remove_product_factor(args: tuple[Expr, ...], target: Sym) -> Expr | None:
    """Find one exact factor in a product without entering reductions.

    Returns:
        The residual product, or None.
    """
    for index, arg in enumerate(args):
        residual = _remove_state_factor(arg, target)
        if residual is not None:
            return _product((*args[:index], residual, *args[index + 1 :]))
    return None


def _remove_power_factor(exponent: Expr, target: Sym) -> Expr | None:
    """Cancel one factor from positive integer powers.

    Returns:
        A lower power, a literal unit, or None for unsupported powers.
    """
    if (
        not isinstance(exponent, Literal)
        or isinstance(exponent.value, bool)
        or not isinstance(exponent.value, int)
        or exponent.value < 1
    ):
        return None
    return (
        Literal(1)
        if exponent.value == 1
        else Apply("pow", (target, Literal(exponent.value - 1)))
    )


def _quotients(
    expressions: Sequence[Expr | None], names: tuple[str, ...], selected: frozenset[int]
) -> tuple[Expr, ...]:
    """Divide selected physical equations by their own physical state.

    Returns:
        Coordinate derivatives expressed in physical variables.

    Raises:
        UnsupportedFeatureError: If an equation has no typed expression.
    """
    result = []
    for index, (name, expression) in enumerate(zip(names, expressions, strict=True)):
        if expression is None:
            raise UnsupportedFeatureError(
                feature="log state coordinates",
                detail="Every coordinate equation needs typed IR",
            )
        current = expression
        if index in selected:
            target = Sym(name)
            residual = _remove_state_factor(expression, target)
            current = (
                residual if residual is not None else Apply("/", (expression, target))
            )
        result.append(current)
    return tuple(result)


def _quotient_source(rhs: NormalizedRhs, selected: frozenset[int]) -> ExprRhs:
    """Build an instantaneous deterministic source without jump artifacts.

    Returns:
        Physical-variable expressions G_i(y)=F_i(y)/y_i for log components.
        Evaluating them at decoded state gives the log-coordinate derivative.
    """
    source = ExprRhs(**{
        field.name: getattr(rhs, field.name) for field in fields(ExprRhs)
    })
    full = _quotients(rhs.equations_ir, rhs.state_names, selected)
    reduced = _quotients(rhs.equations_ir_reduce, rhs.state_names, selected)
    metadata = {
        name: value
        for name, value in rhs.meta.items()
        if not name.startswith("op_system_gauge")
        and name
        not in {"op_system_detect_gauges", "op_system_observables_ir", "initial_state"}
    }
    return replace(
        source,
        equations_ir=full,
        equations_ir_reduce=reduced,
        equations=_derive_equation_strings_lazy(full),
        meta=metadata,
    )
