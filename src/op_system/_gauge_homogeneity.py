"""Conservative rational homogeneity constraints for scaling generators.

A nullspace direction satisfies the supported expression structure on its
finite domain. Structural matching can miss identities that require algebraic
cancellation; it is not a complete structural-identifiability algorithm.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, TypeAlias

from ._errors import UnsupportedFeatureError
from ._ir import Apply, Expr, HistoryOp, Literal, Reduce, Subscript, Sym

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LinearForm: TypeAlias = tuple[Fraction, ...]


class _Homogeneity:
    """Collect linear constraints on shared state/parameter base exponents."""

    def __init__(
        self,
        variables: Sequence[str],
        symbol_bases: Mapping[str, str],
        aliases: Mapping[str, Expr],
        *,
        state_bases: Sequence[str],
    ) -> None:
        """Initialize an exact exponent space and detached constraint storage."""
        self.variables = tuple(variables)
        self.state_bases = tuple(state_bases)
        self.symbol_bases = symbol_bases
        self.aliases = aliases
        self.zero = tuple(Fraction(0) for _ in variables)
        self.index = {name: index for index, name in enumerate(variables)}
        self.rows: list[LinearForm] = []
        self._seen: set[LinearForm] = set()
        self._cache: dict[Expr, LinearForm | None] = {}

    def variable(self, name: str) -> LinearForm:
        """Return a unit exponent or a known fixed-time/constant exponent.

        Returns:
            Linear form of the referenced variable's scaling degree.

        Raises:
            UnsupportedFeatureError: If a symbol has no declared scaling role.
        """
        base = self.symbol_bases.get(name, name)
        if base in self.index:
            return tuple(
                Fraction(index == self.index[base])
                for index in range(len(self.variables))
            )
        if name in {"t", "np.pi", "np.e"} or name.startswith("__op_system_"):
            return self.zero
        raise UnsupportedFeatureError(
            feature="scaling gauges",
            detail=f"Cannot assign a scaling degree to symbol {name!r}",
        )

    def equate(
        self, left: LinearForm | None, right: LinearForm | None
    ) -> LinearForm | None:
        """Require equal nonzero-term degrees, treating literal zero neutrally.

        Returns:
            Common degree subject to the recorded constraints.
        """
        if left is None:
            return right
        if right is None:
            return left
        row = tuple(a - b for a, b in zip(left, right, strict=True))
        pivot = next((value for value in row if value), None)
        if pivot is not None:
            normalized = tuple(value / pivot for value in row)
            if normalized not in self._seen:
                self._seen.add(normalized)
                self.rows.append(normalized)
        return left

    def degree(self, expr: Expr) -> LinearForm | None:
        """Resolve a supported expression, with ``None`` denoting exact zero.

        Returns:
            Linear homogeneity degree and recorded compatibility constraints.

        """
        if expr in self._cache:
            return self._cache[expr]
        result = self._degree(expr)
        self._cache[expr] = result
        return result

    def _degree(self, expr: Expr) -> LinearForm | None:
        """Dispatch a single immutable expression node.

        Returns:
            Its supported scaling degree, with exact zero represented separately.

        Raises:
            UnsupportedFeatureError: If the node is outside this analysis.
        """
        if isinstance(expr, Literal):
            return (
                None
                if isinstance(expr.value, (float, int))
                and not isinstance(expr.value, bool)
                and expr.value == 0
                else self.zero
            )
        if isinstance(expr, Sym):
            return (
                self.degree(self.aliases[expr.name])
                if expr.name in self.aliases
                else self.variable(expr.name)
            )
        if isinstance(expr, Subscript):
            return self.variable(expr.name)
        if isinstance(expr, Reduce):
            return self.degree(expr.body)
        if isinstance(expr, HistoryOp):
            raise UnsupportedFeatureError(
                feature="scaling gauges",
                detail=(
                    "History operators need a scaling contract for the supplied history"
                ),
            )
        return self._apply(expr)

    def _agree(self, args: Sequence[Expr]) -> LinearForm | None:
        """Collect common-degree conditions on additive or selectable branches.

        Returns:
            A representative common degree, or exact zero.
        """
        result = None
        for arg in args:
            result = self.equate(result, self.degree(arg))
        return result

    def _product(self, args: Sequence[Expr]) -> LinearForm | None:
        """Return the sum of factor degrees, with literal-zero absorption."""
        factors = [self.degree(arg) for arg in args]
        if any(factor is None for factor in factors):
            return None
        return tuple(
            sum(
                (factor[index] for factor in factors if factor is not None), Fraction(0)
            )
            for index in range(len(self.variables))
        )

    def _power(self, base_expr: Expr, exponent: Expr) -> LinearForm | None:
        """Apply a literal exponent or require dimensionless dynamic powers.

        Returns:
            Supported degree, with exact zero represented by None.
        """
        base = self.degree(base_expr)
        if isinstance(exponent, Literal) and isinstance(exponent.value, (float, int)):
            if exponent.value == 0:
                return self.zero
            if base is None:
                return None
            coefficient = (
                Fraction(int(exponent.value))
                if isinstance(exponent.value, bool)
                else Fraction(str(exponent.value))
            )
            return tuple(value * coefficient for value in base)
        self.equate(base, self.zero)
        self.equate(self.degree(exponent), self.zero)
        return self.zero

    def _arithmetic(self, op: str, args: Sequence[Expr]) -> LinearForm | None:
        """Resolve supported arithmetic degrees and zero expressions.

        Returns:
            Degree subject to the collected equalities.

        Raises:
            UnsupportedFeatureError: If a denominator is identically zero.
        """
        if op in {"neg", "pos"}:
            result = self.degree(args[0])
        elif op in {"+", "-"}:
            result = (
                None
                if op == "-" and len(args) == 2 and args[0] == args[1]
                else self._agree(args)
            )
        elif op == "*":
            result = self._product(args)
        elif op == "pow":
            result = self._power(args[0], args[1])
        else:
            numerator, denominator = (self.degree(arg) for arg in args)
            if denominator is None:
                raise UnsupportedFeatureError(
                    feature="scaling gauges",
                    detail="Literal zero denominator has no finite scaling domain",
                )
            result = (
                tuple(a - b for a, b in zip(numerator, denominator, strict=True))
                if numerator is not None and denominator is not None
                else None
            )
        return result

    def _apply(self, expr: Apply) -> LinearForm | None:
        """Derive supported finite-domain homogeneity constraints.

        Returns:
            Degree subject to the recorded structural constraints.

        Raises:
            UnsupportedFeatureError: If an operation needs another analysis.
        """
        op, args = expr.op, expr.args
        if op in {"neg", "pos", "+", "-", "*", "/", "pow"}:
            result = self._arithmetic(op, args)
        elif op in {"np.where", "ifelse"}:
            self.equate(self.degree(args[0]), self.zero)
            result = self._agree(args[1:])
        elif op in {"==", "!=", "<", "<=", ">", ">=", "np.arctan2"}:
            self._agree(args)
            result = self.zero
        elif op in {"np.maximum", "np.minimum", "np.clip", "np.hypot", "%"}:
            result = self._agree(args)
        elif op in {"np.abs", "np.sqrt"}:
            degree = self.degree(args[0])
            factor = Fraction(1, 2) if op == "np.sqrt" else Fraction(1)
            result = (
                tuple(value * factor for value in degree)
                if degree is not None
                else None
            )
        elif op in {
            "np.exp",
            "np.expm1",
            "np.log",
            "np.log1p",
            "np.log2",
            "np.log10",
            "np.sin",
            "np.cos",
            "np.tan",
            "np.sinh",
            "np.cosh",
            "np.tanh",
            "and",
            "or",
        }:
            for arg in args:
                self.equate(self.degree(arg), self.zero)
            result = self.zero
        elif op == "sum_state" and not args:
            result = self._agree(tuple(Sym(name) for name in self.state_bases))
        else:
            raise UnsupportedFeatureError(
                feature="scaling gauges",
                detail=f"Structural homogeneity does not support {op!r}",
            )
        return result
