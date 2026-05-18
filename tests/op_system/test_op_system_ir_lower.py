"""Tests for the IR → vector AST lowering (issue #112)."""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import pytest

from op_system._ir import (
    Apply,
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
    lift_cell_ir_to_template,
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


# ---------------------------------------------------------------------------
# lift_cell_ir_to_template
# ---------------------------------------------------------------------------


def test_lift_replaces_templated_cell_syms_with_free_subscripts() -> None:
    """Per-cell ``Sym`` leaves lift to FREE-axis ``Subscript`` nodes."""
    ir = parse_expr_to_ir("beta * S__age_y__loc_a")
    lifted = lift_cell_ir_to_template(
        ir,
        cell_to_template={
            "S__age_y__loc_a": ("S", ("age", "loc")),
            "S__age_y__loc_b": ("S", ("age", "loc")),
            "S__age_o__loc_a": ("S", ("age", "loc")),
            "S__age_o__loc_b": ("S", ("age", "loc")),
        },
    )
    assert isinstance(lifted, Apply)
    assert lifted.op == "*"
    sub = lifted.args[1]
    assert isinstance(sub, Subscript)
    assert sub.name == "S"
    assert tuple(idx.axis for idx in sub.indices) == ("age", "loc")
    assert all(idx.kind == AxisKind.FREE for idx in sub.indices)


def test_lift_leaves_scalar_params_alone() -> None:
    """Symbols absent from the cell map (scalar params) survive intact."""
    ir = parse_expr_to_ir("beta + gamma * 2")
    lifted = lift_cell_ir_to_template(ir, cell_to_template={})
    assert lifted == ir


def test_lift_rewrites_scalar_aliases_to_buf_syms() -> None:
    """Empty-axes mapping rewrites the symbol to its ``<base>_buf`` form."""
    ir = parse_expr_to_ir("I_total * 0.5")
    lifted = lift_cell_ir_to_template(ir, cell_to_template={"I_total": ("I_total", ())})
    assert isinstance(lifted, Apply)
    assert lifted.op == "*"
    assert lifted.args[0] == Sym(name="I_total_buf")


def test_lift_refuses_when_multiple_cells_of_same_base_cooccur() -> None:
    """String-expanded axis reductions must be rejected, not silently merged."""
    # Mimics ``apply_along(pop=j, I[pop=j])`` after string expansion to
    # ``I__pop_p1 + I__pop_p2`` — collapsing both to ``I[pop]`` would
    # broadcast instead of reduce.
    ir = parse_expr_to_ir("I__pop_p1 + I__pop_p2")
    with pytest.raises(UnsupportedIRLoweringError, match="multiple"):
        lift_cell_ir_to_template(
            ir,
            cell_to_template={
                "I__pop_p1": ("I", ("pop",)),
                "I__pop_p2": ("I", ("pop",)),
            },
        )


def test_lift_then_lower_round_trip_evaluates_to_buffer_product() -> None:
    """Lift + lower of ``S * I`` produces a shaped element-wise product."""
    ir = parse_expr_to_ir("S__age_0__vax_0 * I__age_0__vax_0")
    cell_to_template = {
        "S__age_0__vax_0": ("S", ("age", "vax")),
        "I__age_0__vax_0": ("I", ("age", "vax")),
    }
    lifted = lift_cell_ir_to_template(ir, cell_to_template=cell_to_template)
    node = lower_to_vector_ast(
        lifted,
        target_axes=("age", "vax"),
        buffer_axes={"S": ("age", "vax"), "I": ("age", "vax")},
        axis_names=frozenset({"age", "vax"}),
    )
    s_buf = np.array([[1.0, 2.0], [3.0, 4.0]])
    i_buf = np.array([[5.0, 6.0], [7.0, 8.0]])
    out = _eval(node, {"S_buf": s_buf, "I_buf": i_buf, "np": np})
    assert np.array_equal(out, s_buf * i_buf)


# ---------------------------------------------------------------------------
# Weighted reductions (integrate_over / continuous-axis bindings)
# ---------------------------------------------------------------------------


def test_lower_integrate_over_emits_weighted_sum_one_axis() -> None:
    """``integrate_over`` over a single axis multiplies body by deltas then sums."""
    body = Subscript(
        name="f",
        indices=(AxisIndex(axis="x", kind=AxisKind.FREE),),
    )
    expr = Reduce(kind="integrate_over", bindings=(("x", "x"),), body=body)
    node = lower_to_vector_ast(
        expr,
        target_axes=(),
        buffer_axes={"f": ("x",)},
        axis_names=frozenset({"x"}),
        axis_weights={"x": (0.5, 1.5, 1.0)},
    )
    f_buf = np.array([2.0, 4.0, 6.0])
    got = _eval(node, {"f_buf": f_buf, "np": np})
    assert got == pytest.approx(0.5 * 2.0 + 1.5 * 4.0 + 1.0 * 6.0)


def test_lower_integrate_over_rejects_when_weights_missing() -> None:
    """Without ``axis_weights`` for the bound axis, integration cannot lower."""
    body = Subscript(
        name="f",
        indices=(AxisIndex(axis="x", kind=AxisKind.FREE),),
    )
    expr = Reduce(kind="integrate_over", bindings=(("x", "x"),), body=body)
    with pytest.raises(UnsupportedIRLoweringError, match="integration weights"):
        lower_to_vector_ast(
            expr,
            target_axes=(),
            buffer_axes={"f": ("x",)},
            axis_names=frozenset({"x"}),
        )


def test_lower_apply_along_with_weights_for_continuous_axis() -> None:
    """``apply_along`` over a non-reducible axis lowers when weights are provided."""
    body = Subscript(
        name="f",
        indices=(
            AxisIndex(axis="x", kind=AxisKind.FREE),
            AxisIndex(axis="age", kind=AxisKind.FREE),
        ),
    )
    expr = Reduce(kind="apply_along", bindings=(("x", "i"),), body=body)
    node = lower_to_vector_ast(
        expr,
        target_axes=("age",),
        buffer_axes={"f": ("x", "age")},
        axis_names=frozenset({"x", "age"}),
        reducible_axes=frozenset({"age"}),
        axis_weights={"x": (1.0, 2.0)},
    )
    f_buf = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])  # shape (x=2, age=3)
    got = _eval(node, {"f_buf": f_buf, "np": np})
    expected = 1.0 * f_buf[0] + 2.0 * f_buf[1]
    assert np.array_equal(got, expected)


def test_lower_uniform_apply_along_unchanged_when_weights_absent() -> None:
    """Plain ``apply_along`` over a reducible axis still lowers without weights."""
    body = Subscript(
        name="S",
        indices=(AxisIndex(axis="age", kind=AxisKind.FREE),),
    )
    expr = Reduce(kind="apply_along", bindings=(("age", "a"),), body=body)
    node = lower_to_vector_ast(
        expr,
        target_axes=(),
        buffer_axes={"S": ("age",)},
        axis_names=frozenset({"age"}),
        reducible_axes=frozenset({"age"}),
    )
    s_buf = np.array([1.0, 2.0, 3.0])
    got = _eval(node, {"S_buf": s_buf, "np": np})
    assert got == pytest.approx(6.0)


def test_lower_apply_along_with_categorical_filter() -> None:
    """A categorical filter restricts the sum to the listed coords."""
    body = Subscript(
        name="pop",
        indices=(AxisIndex(axis="vax", kind=AxisKind.FREE),),
    )
    expr = Reduce(
        kind="apply_along",
        bindings=(("vax", "j"),),
        body=body,
        filters=(("vax", ("v", "w")),),
    )
    node = lower_to_vector_ast(
        expr,
        target_axes=(),
        buffer_axes={"pop": ("vax",)},
        axis_names=frozenset({"vax"}),
        reducible_axes=frozenset({"vax"}),
        axis_coords={"vax": ("u", "v", "w")},
    )
    pop_buf = np.array([1.0, 2.0, 4.0])
    got = _eval(node, {"pop_buf": pop_buf, "np": np})
    # Sum only over indices [1, 2] -> 2.0 + 4.0
    assert got == pytest.approx(6.0)


def test_lower_apply_along_filter_requires_axis_coords() -> None:
    """Lowering refuses filters when ``axis_coords`` is missing."""
    body = Subscript(
        name="pop",
        indices=(AxisIndex(axis="vax", kind=AxisKind.FREE),),
    )
    expr = Reduce(
        kind="apply_along",
        bindings=(("vax", "j"),),
        body=body,
        filters=(("vax", ("v", "w")),),
    )
    with pytest.raises(UnsupportedIRLoweringError, match="axis_coords"):
        lower_to_vector_ast(
            expr,
            target_axes=(),
            buffer_axes={"pop": ("vax",)},
            axis_names=frozenset({"vax"}),
            reducible_axes=frozenset({"vax"}),
        )


def test_lower_apply_along_filter_rejects_unknown_coord() -> None:
    """A filter coord that is not declared on the axis is rejected."""
    body = Subscript(
        name="pop",
        indices=(AxisIndex(axis="vax", kind=AxisKind.FREE),),
    )
    expr = Reduce(
        kind="apply_along",
        bindings=(("vax", "j"),),
        body=body,
        filters=(("vax", ("v", "z")),),
    )
    with pytest.raises(UnsupportedIRLoweringError, match="not declared"):
        lower_to_vector_ast(
            expr,
            target_axes=(),
            buffer_axes={"pop": ("vax",)},
            axis_names=frozenset({"vax"}),
            reducible_axes=frozenset({"vax"}),
            axis_coords={"vax": ("u", "v", "w")},
        )


def test_lower_apply_along_kernel_integrate_requires_weights() -> None:
    """``kernel='integrate'`` forces weighted sum even on reducible axes."""
    body = Subscript(
        name="S",
        indices=(AxisIndex(axis="age", kind=AxisKind.FREE),),
    )
    expr = Reduce(
        kind="apply_along",
        bindings=(("age", "a"),),
        body=body,
        kernel="integrate",
    )
    with pytest.raises(UnsupportedIRLoweringError, match="integration weights"):
        lower_to_vector_ast(
            expr,
            target_axes=(),
            buffer_axes={"S": ("age",)},
            axis_names=frozenset({"age"}),
            reducible_axes=frozenset({"age"}),
        )


def test_lower_apply_along_kernel_sum_skips_weights_on_continuous_axis() -> None:
    """``kernel='sum'`` forces uniform reduction even when weights exist."""
    body = Subscript(
        name="f",
        indices=(AxisIndex(axis="x", kind=AxisKind.FREE),),
    )
    expr = Reduce(
        kind="apply_along",
        bindings=(("x", "i"),),
        body=body,
        kernel="sum",
    )
    node = lower_to_vector_ast(
        expr,
        target_axes=(),
        buffer_axes={"f": ("x",)},
        axis_names=frozenset({"x"}),
        reducible_axes=frozenset(),  # x is non-reducible
        axis_weights={"x": (1.0, 2.0, 3.0)},  # would normally be used
    )
    f_buf = np.array([10.0, 20.0, 30.0])
    got = _eval(node, {"f_buf": f_buf, "np": np})
    # Uniform sum: 10 + 20 + 30 = 60 (NOT 1*10 + 2*20 + 3*30 = 140)
    assert got == pytest.approx(60.0)


def test_lower_apply_along_categorical_filter_with_continuous_weighted_axis() -> None:
    """Filter on a continuous axis subsets weights as well as body."""
    body = Subscript(
        name="f",
        indices=(AxisIndex(axis="x", kind=AxisKind.FREE),),
    )
    expr = Reduce(
        kind="integrate_over",
        bindings=(("x", "i"),),
        body=body,
        filters=(("x", ("a", "c")),),
    )
    node = lower_to_vector_ast(
        expr,
        target_axes=(),
        buffer_axes={"f": ("x",)},
        axis_names=frozenset({"x"}),
        reducible_axes=frozenset(),
        axis_weights={"x": (1.0, 2.0, 3.0)},
        axis_coords={"x": ("a", "b", "c")},
    )
    f_buf = np.array([10.0, 20.0, 30.0])
    got = _eval(node, {"f_buf": f_buf, "np": np})
    # Take indices [0, 2]; weights become (1.0, 3.0); sum = 1*10 + 3*30 = 100
    assert got == pytest.approx(100.0)
