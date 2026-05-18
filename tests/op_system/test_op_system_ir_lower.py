"""Tests for the IR → vector AST lowering (issue #112)."""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import pytest

from op_system._ir import (
    AxisIndex,
    AxisKind,
    Literal,
    Reduce,
    Subscript,
    Sym,
    parse_expr_to_ir,
    resolve_axis_kinds,
)
from op_system._ir_lower import (
    UnsupportedIRLoweringError,
    lower_subscript_to_buffer,
    lower_to_vector_ast,
)


def _eval(expr: ast.expr, env: dict[str, Any]) -> Any:  # noqa: ANN401
    tree = ast.Expression(body=expr)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<test>", "eval")
    return eval(code, {"__builtins__": {}}, env)  # noqa: S307


def _make_subscript(name: str, axes: tuple[str, ...]) -> Subscript:
    return Subscript(
        name=name,
        indices=tuple(AxisIndex(axis=a, kind=AxisKind.FREE) for a in axes),
    )


# ---------------------------------------------------------------------------
# lower_subscript_to_buffer
# ---------------------------------------------------------------------------


def test_lower_subscript_identity_when_axes_match_target() -> None:
    """Wildcard subscript matching target axes returns the bare buffer name."""
    sub = _make_subscript("S", ("age", "vax"))
    node = lower_subscript_to_buffer(
        sub,
        src_axes=("age", "vax"),
        target_axes=("age", "vax"),
        axis_names=frozenset({"age", "vax"}),
    )
    s_buf = np.arange(6).reshape(3, 2)
    assert np.array_equal(_eval(node, {"S_buf": s_buf, "np": np}), s_buf)


def test_lower_subscript_transposes_when_user_reorders_axes() -> None:
    """Subscript axes in non-declaration order trigger a transpose."""
    sub = _make_subscript("S", ("vax", "age"))
    node = lower_subscript_to_buffer(
        sub,
        src_axes=("age", "vax"),
        target_axes=("vax", "age"),
        axis_names=frozenset({"age", "vax"}),
    )
    s_buf = np.arange(6).reshape(3, 2)
    got = _eval(node, {"S_buf": s_buf, "np": np})
    assert np.array_equal(got, s_buf.T)


def test_lower_subscript_broadcasts_missing_target_axis() -> None:
    """Target axis not in subscript becomes a singleton (``None``) dim."""
    sub = _make_subscript("p", ("age",))
    node = lower_subscript_to_buffer(
        sub,
        src_axes=("age",),
        target_axes=("age", "vax"),
        axis_names=frozenset({"age", "vax"}),
    )
    p_buf = np.array([1.0, 2.0, 3.0])
    got = _eval(node, {"p_buf": p_buf, "np": np})
    assert got.shape == (3, 1)
    assert np.array_equal(got[:, 0], p_buf)


def test_lower_subscript_rejects_coord_literal() -> None:
    """Coord indices are out of scope for v1 and raise."""
    sub = Subscript(
        name="S",
        indices=(
            AxisIndex(axis="age", kind=AxisKind.FREE),
            AxisIndex(axis="vax", coord="unvac", kind=AxisKind.COORD),
        ),
    )
    with pytest.raises(UnsupportedIRLoweringError, match="non-FREE"):
        lower_subscript_to_buffer(
            sub,
            src_axes=("age", "vax"),
            target_axes=("age", "vax"),
            axis_names=frozenset({"age", "vax"}),
        )


def test_lower_subscript_rejects_arity_mismatch() -> None:
    """A subscript whose index count differs from buffer axes raises."""
    sub = _make_subscript("S", ("age",))
    with pytest.raises(UnsupportedIRLoweringError, match="indices"):
        lower_subscript_to_buffer(
            sub,
            src_axes=("age", "vax"),
            target_axes=("age", "vax"),
            axis_names=frozenset({"age", "vax"}),
        )


def test_lower_subscript_rejects_axes_not_subset_of_target() -> None:
    """If the buffer has an axis not in target_axes, lowering fails."""
    sub = _make_subscript("S", ("age", "vax"))
    with pytest.raises(UnsupportedIRLoweringError, match="subset"):
        lower_subscript_to_buffer(
            sub,
            src_axes=("age", "vax"),
            target_axes=("age",),
            axis_names=frozenset({"age", "vax"}),
        )


# ---------------------------------------------------------------------------
# lower_to_vector_ast
# ---------------------------------------------------------------------------


def test_lower_to_vector_ast_scalar_arithmetic() -> None:
    """Pure arithmetic over scalars lowers and evaluates as expected."""
    ir = parse_expr_to_ir("beta * 2.0 - gamma")
    node = lower_to_vector_ast(
        ir,
        target_axes=(),
        buffer_axes={},
        axis_names=frozenset(),
    )
    env = {"beta": 0.3, "gamma": 0.1}
    assert _eval(node, env) == pytest.approx(0.3 * 2.0 - 0.1)


def test_lower_to_vector_ast_mixes_scalar_and_buffer() -> None:
    """A scalar param multiplied by a templated state buffer broadcasts."""
    ir = resolve_axis_kinds(
        parse_expr_to_ir("beta * S[age, vax]"),
        axis_names=frozenset({"age", "vax"}),
    )
    node = lower_to_vector_ast(
        ir,
        target_axes=("age", "vax"),
        buffer_axes={"S": ("age", "vax")},
        axis_names=frozenset({"age", "vax"}),
    )
    s_buf = np.arange(6, dtype=float).reshape(3, 2)
    got = _eval(node, {"beta": 0.5, "S_buf": s_buf, "np": np})
    assert np.array_equal(got, 0.5 * s_buf)


def test_lower_to_vector_ast_unary_negation() -> None:
    """Unary minus lowers to ``ast.USub``."""
    ir = parse_expr_to_ir("-x")
    node = lower_to_vector_ast(
        ir,
        target_axes=(),
        buffer_axes={},
        axis_names=frozenset(),
    )
    assert _eval(node, {"x": 4.5}) == pytest.approx(-4.5)


def test_lower_to_vector_ast_comparison_and_ifelse() -> None:
    """Comparison and conditional expressions lower correctly."""
    ir = parse_expr_to_ir("a if a > b else b")
    node = lower_to_vector_ast(
        ir,
        target_axes=(),
        buffer_axes={},
        axis_names=frozenset(),
    )
    assert _eval(node, {"a": 3, "b": 7}) == 7
    assert _eval(node, {"a": 9, "b": 7}) == 9


def test_lower_to_vector_ast_rejects_reduce_node() -> None:
    """``Reduce`` nodes are out of scope for v1 lowering."""
    expr = Reduce(
        kind="sum_over",
        bindings=(("age", "a"),),
        body=Sym(name="x"),
    )
    with pytest.raises(UnsupportedIRLoweringError, match="Reduce"):
        lower_to_vector_ast(
            expr,
            target_axes=(),
            buffer_axes={},
            axis_names=frozenset({"age"}),
        )


def test_lower_to_vector_ast_rejects_function_call() -> None:
    """Generic function-call ``Apply`` nodes raise (only operators in v1)."""
    ir = parse_expr_to_ir("np.exp(x)")
    with pytest.raises(UnsupportedIRLoweringError, match=r"np\.exp"):
        lower_to_vector_ast(
            ir,
            target_axes=(),
            buffer_axes={},
            axis_names=frozenset(),
        )


def test_lower_to_vector_ast_rejects_unregistered_subscript() -> None:
    """A subscript whose name isn't a registered buffer raises."""
    ir = parse_expr_to_ir("K[age]")
    ir = resolve_axis_kinds(ir, axis_names=frozenset({"age"}))
    with pytest.raises(UnsupportedIRLoweringError, match="not a registered"):
        lower_to_vector_ast(
            ir,
            target_axes=("age",),
            buffer_axes={},
            axis_names=frozenset({"age"}),
        )


def test_lower_to_vector_ast_literal_passthrough() -> None:
    """Literals lower to ``ast.Constant`` and evaluate to themselves."""
    node = lower_to_vector_ast(
        Literal(value=math.pi),
        target_axes=(),
        buffer_axes={},
        axis_names=frozenset(),
    )
    assert _eval(node, {}) == pytest.approx(math.pi)


def test_lower_to_vector_ast_two_templated_buffers_broadcast() -> None:
    """Two templated buffers over a subset of target axes broadcast correctly."""
    ir = resolve_axis_kinds(
        parse_expr_to_ir("S[age] * p[vax]"),
        axis_names=frozenset({"age", "vax"}),
    )
    node = lower_to_vector_ast(
        ir,
        target_axes=("age", "vax"),
        buffer_axes={"S": ("age",), "p": ("vax",)},
        axis_names=frozenset({"age", "vax"}),
    )
    s_buf = np.array([1.0, 2.0, 3.0])
    p_buf = np.array([10.0, 20.0])
    got = _eval(node, {"S_buf": s_buf, "p_buf": p_buf, "np": np})
    assert got.shape == (3, 2)
    expected = np.outer(s_buf, p_buf)
    assert np.array_equal(got, expected)


def test_lower_subscript_resolves_axis_kinds_when_unset() -> None:
    """When ``AxisIndex.kind`` is unset, the lowering classifies on the fly."""
    sub = Subscript(
        name="S",
        indices=(AxisIndex(axis="age"), AxisIndex(axis="vax")),
    )
    node = lower_subscript_to_buffer(
        sub,
        src_axes=("age", "vax"),
        target_axes=("age", "vax"),
        axis_names=frozenset({"age", "vax"}),
    )
    s_buf = np.arange(6).reshape(3, 2)
    assert np.array_equal(_eval(node, {"S_buf": s_buf, "np": np}), s_buf)
