"""Conservative structural reads, selectors and grouped-observable coverage."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from op_system import compile_spec, dependency_structure
from op_system._errors import UnsupportedFeatureError
from op_system._normalize_ir import _LazyCells


def _spec(expression: str = "-rate*x[loc]") -> dict[str, object]:
    """Return a three-cell system whose exact couplings are known.

    Returns:
        A local, reduced or nonlocal expression fixture.
    """
    return {
        "kind": "expr",
        "state": ["x[loc]"],
        "axes": [{"name": "loc", "type": "categorical", "coords": ["a", "b", "c"]}],
        "equations": {"x[loc]": expression},
    }


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("coupling", ["local", "global", "pinned"])
def test_native_couplings_are_covered_by_structural_rows(
    backend: str, dtype: str, coupling: str
) -> None:
    """An independently specified linear map detects missed selector/reduction reads."""
    expression, matrix, expected = {
        "local": ("-2*x[loc]", -2 * np.eye(3), ((0,), (1,), (2,))),
        "global": ("sum_over(x[loc:l], loc=l)", np.ones((3, 3)), ((0, 1, 2),) * 3),
        "pinned": (
            "x[loc:a]-x[loc]",
            np.array([[0, 0, 0], [1, -1, 0], [1, 0, -1]]),
            ((0,), (0, 1), (0, 2)),
        ),
    }[coupling]
    compiled = compile_spec(_spec(expression))
    report = dependency_structure(compiled)
    assert report.status == "reported"
    assert report.state is not None
    assert report.state.rows == expected
    assert report.state.output_names == compiled.state_names
    assert report.block_axes == ()  # undeclared independence is never invented
    xp: Any = np
    precision = nullcontext()
    if backend == "jax":
        jax = pytest.importorskip("jax")
        xp = pytest.importorskip("jax.numpy")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        precision = enable()
    elif backend == "torch":
        xp = pytest.importorskip("torch")
    with precision:
        values: Any = compiled.eval_fn(
            0.0, xp.asarray([1.0, 2.0, 4.0], dtype=getattr(xp, dtype))
        )
        host = values.detach().numpy() if backend == "torch" else np.asarray(values)
        np.testing.assert_allclose(host, matrix @ np.array([1.0, 2.0, 4.0]), rtol=1e-6)
    for row, columns in enumerate(report.state.rows):
        assert set(np.flatnonzero(matrix[row])) <= set(columns)


def test_aliases_shaped_parameters_and_observable_groups_keep_layouts() -> None:
    """Buffer parameters and array outputs keep their declared grouping."""
    spec = _spec("-rate[loc]*x[loc]+offset")
    spec["observables"] = {
        "total": {"expr": "x[loc]", "reduction": {"kind": "sum", "axes": ["loc"]}},
        "cells": {"expr": "gain*x[loc]", "axes": ["loc"]},
    }
    compiled = compile_spec(spec)
    report = dependency_structure(compiled)
    assert report.parameter_axes == (("offset", ()), ("rate", ("loc",)), ("gain", ()))
    assert report.parameters is not None
    assert report.parameters.rows == ((0, 1),) * 3
    assert report.observable_state is not None
    assert report.observable_state.output_shapes == ((), (3,))
    assert report.observable_state.rows == ((0, 1, 2),) * 2
    assert report.observable_parameters is not None
    assert report.observable_parameters.rows == ((), (2,))
    scalar = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "aliases": {"first": "a*x", "second": "first+b*z"},
        "equations": {"x": "second", "z": "t"},
    })
    aliases = dependency_structure(scalar)
    assert aliases.state is not None
    assert aliases.state.rows == ((0, 1), ())
    assert aliases.state.time_dependent == (False, True)
    assert aliases.model != report.model


def test_zero_coefficients_and_cancellation_do_not_remove_structural_edges() -> None:
    """Potential reads remain even when their numerical derivatives vanish."""
    compiled = compile_spec(_spec("rate*x[loc]+x[loc:a]-x[loc:a]"))
    report = dependency_structure(compiled)
    assert report.state is not None
    assert report.state.rows == ((0,), (0, 1), (0, 2))
    zero = compiled.eval_fn(0.0, np.ones(3), rate=np.asarray(0.0))
    np.testing.assert_array_equal(zero, np.zeros(3))
    assert dependency_structure(compiled) == report


def test_checked_blocks_are_copied_with_no_new_independence_claim() -> None:
    """Checked block metadata survives independently of mutable compiler mappings."""
    spec = _spec("-rate[loc]*x[loc]")
    spec["factorize_axes"] = ["loc"]
    spec["observables"] = {
        "total": {"expr": "x[loc]", "reduction": {"kind": "sum", "axes": ["loc"]}}
    }
    compiled = compile_spec(spec)
    report = dependency_structure(compiled)
    assert len(report.block_axes) == 1
    block = report.block_axes[0]
    assert block.name == "loc"
    assert block.state_axis_positions == (("x", 0),)
    assert block.parameter_axis_positions == (("rate", 0),)
    assert report.observable_state is not None
    assert report.observable_state.rows == (
        (0, 1, 2),
    )  # observation couples RHS blocks
    compiled.block_axes[0].state_axis_pos.clear()
    assert block.state_axis_positions == (("x", 0),)


@pytest.mark.parametrize("expression", ["sum_over(x[loc:l], loc=l)", "x[loc:a]-x[loc]"])
def test_existing_proof_rejects_global_and_nonlocal_factorization(
    expression: str,
) -> None:
    """The original compiler proof, rather than the exporter, rejects coupling."""
    spec = _spec(expression)
    spec["factorize_axes"] = ["loc"]
    with pytest.raises(UnsupportedFeatureError, match="factorize_axes"):
        compile_spec(spec)


def test_unsupported_coordinates_and_missing_source_never_become_empty_patterns() -> (
    None
):
    """An unsupported report differs from a supported constant equation."""
    compiled = compile_spec(
        {
            "kind": "expr",
            "state": ["x"],
            "equations": {"x": "rate*x*(1-x/K)"},
            "observables": {"measurement": {"expr": "gain*x"}},
            "gauges": [
                {
                    "name": "scale",
                    "states": {"x": 1},
                    "parameters": {"K": 1, "gain": -1},
                }
            ],
        },
        gauge_fix={"K": 1.0},
    )
    unsupported = dependency_structure(compiled)
    assert unsupported.status == "unsupported"
    assert unsupported.state is None
    assert unsupported.reasons == ("gauge_fixed_execution",)
    constant = dependency_structure(compile_spec(_spec("1")))
    assert constant.status == "reported"
    assert constant.state is not None
    assert constant.state.rows == ((), (), ())
    with pytest.raises(ValueError, match="source/layout"):
        dependency_structure(replace(compiled, _rhs=None))


@pytest.mark.parametrize(
    "limits",
    [
        {"max_states": 2},
        {"max_edges": 2},
        {"max_nodes": 2},
        {"max_states": True},
        {"max_edges": 0},
    ],
)
def test_limits_stop_export_instead_of_returning_partial_zero_patterns(
    limits: dict[str, Any],
) -> None:
    """Explicit output and traversal bounds cannot silently truncate coverage."""
    with pytest.raises(ValueError, match="limit"):
        dependency_structure(compile_spec(_spec()), **limits)


def test_primitive_report_retains_context_without_mutating_the_artifact() -> None:
    """Stable primitive serialization retains the exact conservative source/layout."""
    report = dependency_structure(compile_spec(_spec()))
    payload = json.loads(report.to_json())
    assert payload["model"] == json.loads(report.model.to_json())
    assert payload["state"]["rows"] == [[0], [1], [2]]
    payload["state"]["rows"][0].clear()
    assert report.state is not None
    assert report.state.rows[0] == (0,)
    assert json.loads(report.to_json())["limits"]["states"] == 4096


def test_forcing_buffer_is_a_grouped_input_with_implicit_time_dependence() -> None:
    """Known forcing tables retain their complete input axes and time read."""
    spec = _spec("-rate[time]*x[loc]")
    axes = spec["axes"]
    assert isinstance(axes, list)
    axes.append({"name": "time", "type": "continuous", "coords": [0.0, 1.0]})
    report = dependency_structure(compile_spec(spec))
    assert report.parameter_axes == (("rate", ("time",)),)
    assert report.parameters is not None
    assert report.parameters.rows == ((0,),) * 3
    assert report.state is not None
    assert report.state.time_dependent == (True,) * 3


def test_edge_budget_preserves_lazy_rows_of_a_dense_global_reduction() -> None:
    """A small export budget must not expand the entire quadratic source first."""
    spec = _spec("sum_over(x[loc:l], loc=l)")
    spec["axes"] = [{"name": "loc", "coords": [f"c{i}" for i in range(24)]}]
    compiled = compile_spec(spec)
    assert compiled._rhs is not None  # ruff: ignore[private-member-access]
    rows = compiled._rhs.equations_ir  # ruff: ignore[private-member-access]
    assert isinstance(rows, _LazyCells)
    before = len(rows._cache)  # ruff: ignore[private-member-access]
    with pytest.raises(ValueError, match="edge limit"):
        dependency_structure(compiled, max_edges=2)
    after = len(rows._cache)  # ruff: ignore[private-member-access]
    assert after <= before + 1
    assert after < len(compiled.state_names)


def test_operator_and_history_coverage_is_explicit() -> None:
    """Current-state rows never omit an unexamined operator or history callback."""
    operator = compile_spec({
        "kind": "expr",
        "state": ["x[space]"],
        "axes": [{"name": "space", "type": "continuous", "coords": [0.0, 1.0]}],
        "equations": {"x[space]": "-rate*x[space]"},
        "operators": [
            {"kind": "diffusion", "axis": "space", "rate": "D", "bc": "dirichlet"}
        ],
    })
    history = compile_spec({
        "kind": "expr",
        "state": ["x"],
        "equations": {
            "x": "convolve_history(x, kernel=gamma, window=14, interpolation=linear)"
        },
    })
    for compiled, reason in ((operator, "numerical_operators"), (history, "history")):
        report = dependency_structure(compiled)
        assert report.status == "unsupported"
        assert reason in report.reasons
        assert report.state is report.parameters is None
        assert json.loads(report.to_json())["state"] is None
