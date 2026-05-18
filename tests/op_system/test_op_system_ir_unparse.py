"""Tests for IR unparser round-trip behavior (issue #112)."""

from __future__ import annotations

import pytest

from op_system._ir import parse_expr_to_ir, unparse_ir


@pytest.mark.parametrize(
    "expr",
    [
        "beta * S * I / N - gamma * I",
        "np.maximum(x, 0.0)",
        "K[age, ap]",
        "x + y * z",
        "a ** 2",
        "-(x + 1)",
    ],
)
def test_unparse_ir_round_trip_value_equivalence(expr: str) -> None:
    """Round-tripping IR through unparser yields semantically equivalent source."""
    ir1 = parse_expr_to_ir(expr)
    rendered = unparse_ir(ir1)
    ir2 = parse_expr_to_ir(rendered)

    assert ir1 == ir2


def test_unparse_ir_renders_helper_reduce_node() -> None:
    """Unparser renders Reduce nodes back into helper call form."""
    ir = parse_expr_to_ir("apply_along(I[age], age=ap)", lower_helpers=True)

    rendered = unparse_ir(ir)
    assert rendered.startswith("apply_along(")
    assert "age=ap" in rendered
