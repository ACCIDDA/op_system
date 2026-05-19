"""Parity and structure tests for IR-side ``expand_reduce_pointwise``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from op_system._ir import Apply, Subscript, Sym, parse_expr_to_ir, unparse_ir
from op_system._ir_expand import expand_reduce_pointwise

if TYPE_CHECKING:
    from collections.abc import Mapping

    from op_system._ir import Expr


def _ir_expand_to_string(
    src: str,
    *,
    axes: list[dict[str, Any]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    lhs_assignment: Mapping[str, str] | None = None,
    axis_coords: Mapping[str, list[str]] | None = None,
) -> str:
    """Helper: parse ``src``, expand, then unparse for substring checks.

    Returns:
        The unparsed pointwise expression string.
    """
    ir = parse_expr_to_ir(src, lower_helpers=True)
    expanded = expand_reduce_pointwise(
        ir,
        axes=axes,
        shaped_params=shaped_params,
        lhs_assignment=lhs_assignment,
        axis_coords=axis_coords,
    )
    return unparse_ir(expanded)


@pytest.fixture
def axes_age_pop() -> list[dict[str, Any]]:
    """Two categorical axes used by several tests.

    Returns:
        A list of two categorical axis dicts (age, pop).
    """
    return [
        {"name": "age", "type": "categorical", "coords": ["a0", "a1", "a2"]},
        {"name": "pop", "type": "categorical", "coords": ["p1", "p2"]},
    ]


def test_single_axis_sum_basic_names(axes_age_pop: list[dict[str, Any]]) -> None:
    """Expanded names include the canonical ``__axis_<coord>`` suffix."""
    out = _ir_expand_to_string(
        "apply_along(I[pop:j], pop=j)",
        axes=axes_age_pop,
    )
    assert "I__pop_p1" in out
    assert "I__pop_p2" in out


def test_single_axis_sum_two_term_sum(axes_age_pop: list[dict[str, Any]]) -> None:
    """A 2-coord apply_along produces a binary Apply('+') of two terms."""
    ir = parse_expr_to_ir("apply_along(I[pop:j], pop=j)", lower_helpers=True)
    expanded = expand_reduce_pointwise(ir, axes=axes_age_pop)
    assert isinstance(expanded, Apply)
    assert expanded.op == "+"
    assert len(expanded.args) == 2


def test_same_axis_twice_with_lhs_assignment() -> None:
    """K[age, age:ap] with lhs={age:a0}: row pinned to 0, col bound 0..1."""
    axes: list[dict[str, Any]] = [
        {"name": "age", "type": "categorical", "coords": ["a0", "a1"]},
    ]
    shaped = {"K": ("age", "age")}
    axis_coords = {"age": ["a0", "a1"]}
    ir = parse_expr_to_ir(
        "apply_along(K[age, age:ap] * I[age:ap], age=ap)", lower_helpers=True
    )
    expanded = expand_reduce_pointwise(
        ir,
        axes=axes,
        shaped_params=shaped,
        lhs_assignment={"age": "a0"},
        axis_coords=axis_coords,
    )
    assert isinstance(expanded, Apply)
    assert expanded.op == "+"
    summands = expanded.args
    assert len(summands) == 2

    def _extract_k_index(term: Expr) -> tuple[str | None, ...] | None:
        """Return K's index tuple if ``term`` is K * something, else None.

        Returns:
            Tuple of coord strings (one per axis), or ``None`` if K is
            not present in this term.
        """
        if not isinstance(term, Apply):
            return None
        for arg in term.args:
            if isinstance(arg, Subscript) and arg.name == "K":
                return tuple(idx.coord for idx in arg.indices)
        return None

    k_indices = {_extract_k_index(t) for t in summands}
    assert k_indices == {("0", "0"), ("0", "1")}

    i_names = {
        a.name
        for term in summands
        if isinstance(term, Apply)
        for a in term.args
        if isinstance(a, Sym) and a.name.startswith("I__")
    }
    assert i_names == {"I__age_a0", "I__age_a1"}


def test_integrate_kernel_emits_weight_literal() -> None:
    """Continuous-axis apply_along with kernel=integrate emits float weights."""
    axes: list[dict[str, Any]] = [
        {
            "name": "t",
            "type": "continuous",
            "coords": ["0.0", "1.0", "2.0"],
            "deltas": [0.5, 1.0, 0.5],
        },
    ]
    out = _ir_expand_to_string(
        "apply_along(f[t:tau], t=tau, kernel=integrate)",
        axes=axes,
    )
    assert "0.5" in out
    assert "1.0" in out


def test_parity_two_axis_cartesian(axes_age_pop: list[dict[str, Any]]) -> None:
    """Multi-axis apply_along produces ``|age|*|pop|`` summands."""
    src = "apply_along(I[age:i, pop:j], age=i, pop=j)"
    ir = parse_expr_to_ir(src, lower_helpers=True)
    expanded = expand_reduce_pointwise(ir, axes=axes_age_pop)

    def _count_summands(e: Expr) -> int:
        """Count leaf summands in a left-associative ``Apply('+')`` fold.

        Returns:
            Number of leaf terms.
        """
        if isinstance(e, Apply) and e.op == "+":
            return sum(_count_summands(a) for a in e.args)
        return 1

    assert _count_summands(expanded) == 6
