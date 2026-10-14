"""Physical declarations preserve equations and distinguish intensive totals."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from array_api_compat import array_namespace  # type: ignore[import-untyped]

from op_system import AxisDeclaration, AxisMeasure, StateDeclaration, compile_spec
from op_system._errors import InvalidRhsSpecError

if TYPE_CHECKING:
    from collections.abc import MutableMapping


def _spec(kind: str = "expr") -> dict[str, object]:
    """Use cell centers whose boundary volumes differ from point quadrature.

    Returns:
        Nonuniform unit-interval state fixture with explicit physical measures.
    """
    spec: dict[str, object] = {
        "kind": kind,
        "axes": [
            {
                "name": "space",
                "type": "continuous",
                "coords": [0.05, 0.25, 0.7],
                "units": "m",
                "measure": {"kind": "cell", "weights": [0.1, 0.3, 0.6], "units": "m"},
            }
        ],
        "state": ["rho[space]", "mass[space]", "counter"],
        "state_metadata": {
            "rho": {
                "representation": "density",
                "units": "items/m",
                "measure_axes": ["space"],
            },
            "mass": {"representation": "counts", "units": "items", "measure_axes": []},
            "counter": {"representation": "cumulative", "units": "items"},
        },
    }
    if kind == "expr":
        spec["equations"] = {
            "rho[space]": "0",
            "mass[space]": "0",
            "counter": "sum_state()",
        }
    else:
        spec["transitions"] = [{"from": None, "to": "counter", "rate": "sum_state()"}]
    return spec


@pytest.mark.parametrize("kind", ["expr", "transitions"])
def test_physical_metadata_and_unweighted_sum_semantics(kind: str) -> None:
    """Density integrates with volumes; extensive counts sum without volumes."""
    compiled = compile_spec(_spec(kind))
    assert compiled.state_meta["rho"] == StateDeclaration(
        "density", "items/m", None, ("space",)
    )
    assert compiled.axis_meta["space"] == AxisDeclaration(
        "m", AxisMeasure("cell", (0.1, 0.3, 0.6), "m")
    )
    measure = compiled.axis_meta["space"].measure
    assert measure is not None
    # rho(x)=1+2*x has exact integral 2 on [0,1]. Its cell averages are center values.
    density = np.asarray([1.1, 1.5, 2.4])
    cell_counts = density * np.asarray(measure.weights)
    assert np.sum(cell_counts) == pytest.approx(2.0)
    assert np.sum(density) == pytest.approx(5.0)
    state = np.concatenate((density, cell_counts, [0.0]))
    np.testing.assert_allclose(compiled.eval_fn(0.0, state), [0, 0, 0, 0, 0, 0, 7])
    undeclared = _spec(kind)
    del undeclared["state_metadata"]
    np.testing.assert_array_equal(
        compile_spec(undeclared).eval_fn(0.0, state), compiled.eval_fn(0.0, state)
    )
    # Existing continuous-axis quadrature describes the span of center points,
    # not the full finite-volume domain. Metadata does not override it.
    assert compiled.meta["axes"][0]["deltas"] == pytest.approx([0.1, 0.325, 0.225])


def test_partial_and_undeclared_physical_metadata() -> None:
    """Unknown annotations stay unknown, while every named axis is exported."""
    compiled = compile_spec({
        "kind": "expr",
        "axes": [{"name": "label", "coords": ["a", "b"]}],
        "state": ["x[label]", "y"],
        "equations": {"x[label]": "0", "y": "0"},
        "state_metadata": {"x": {"description": "Unqualified representation"}},
    })
    assert compiled.state_meta == {
        "x": StateDeclaration(description="Unqualified representation")
    }
    assert compiled.axis_meta == {"label": AxisDeclaration()}
    spec = _spec()
    del spec["state_metadata"]
    assert compile_spec(spec).state_meta == {}


@pytest.mark.parametrize(
    "representation", ["counts", "cell_mass", "density", "concentration", "cumulative"]
)
def test_supported_state_representations(representation: str) -> None:
    """Representations are annotations and do not impose numerical constraints."""
    spec: dict[str, object] = {
        "state": ["x"],
        "equations": {"x": "-x"},
        "state_metadata": {"x": {"representation": representation}},
    }
    compiled = compile_spec(spec)
    assert compiled.state_meta["x"].representation == representation
    np.testing.assert_array_equal(compiled.eval_fn(0.0, np.array([-2.0])), [2.0])


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        1,
        {"unknown": "value"},
        {"representation": "unknown"},
        {"representation": []},
        {"representation": None},
        {"units": False},
        {"description": " "},
        {"measure_axes": "space"},
        {"measure_axes": ["space", "space"]},
        {"measure_axes": [[]]},
        {"measure_axes": ["missing"]},
        {"representation": "counts", "measure_axes": ["space"]},
    ],
)
def test_malformed_state_declarations(declaration: object) -> None:
    """Malformed fields fail with a domain-specific error before compilation."""
    spec = _spec()
    spec["state_metadata"] = {"rho": declaration}
    with pytest.raises(InvalidRhsSpecError):
        compile_spec(spec)


@pytest.mark.parametrize(
    "metadata", [[], "rho", {"unknown": {}}, {"rho__space_0": {}}, {1: {}}]
)
def test_invalid_state_metadata_names(metadata: object) -> None:
    """Declarations belong to declared state bases, never generated cell names."""
    spec = _spec()
    spec["state_metadata"] = metadata
    with pytest.raises(InvalidRhsSpecError):
        compile_spec(spec)


@pytest.mark.parametrize(
    "measure",
    [
        None,
        [],
        {"kind": []},
        {"kind": "unknown"},
        {"kind": "counting", "weights": [1, 1, 1]},
        {"kind": "cell"},
        {"kind": "cell", "weights": [1]},
        {"kind": "cell", "weights": [1, 1, 0]},
        {"kind": "cell", "weights": [1, -1, 1]},
        {"kind": "cell", "weights": [1, True, 1]},
        {"kind": "cell", "weights": [1, float("nan"), 1]},
        {"kind": "cell", "weights": [1, float("inf"), 1]},
        {"kind": "cell", "weights": [1, 10**400, 1]},
        {"kind": "cell", "weights": [1, "1", 1]},
        {"kind": "cell", "weights": [1, 1, 1], "units": " "},
        {"kind": "counting", "extra": 1},
    ],
)
def test_malformed_axis_measures(measure: object) -> None:
    """Measures require a recognized kind and finite positive aligned weights."""
    spec = _spec()
    axes = cast("list[dict[str, object]]", spec["axes"])
    axes[0]["measure"] = measure
    with pytest.raises(InvalidRhsSpecError):
        compile_spec(spec)


def test_state_requires_declared_measure_on_its_own_axis() -> None:
    """A density declaration cannot silently borrow automatic point quadrature."""
    spec = _spec()
    axes = cast("list[dict[str, object]]", spec["axes"])
    del axes[0]["measure"]
    with pytest.raises(InvalidRhsSpecError, match="requires a measure"):
        compile_spec(spec)
    axes[0]["measure"] = {"kind": "cell", "weights": [0.1, 0.3, 0.6]}
    spec["state_metadata"] = {
        "counter": {"representation": "density", "measure_axes": ["space"]}
    }
    with pytest.raises(InvalidRhsSpecError, match="not a state axis"):
        compile_spec(spec)


def test_counting_and_quadrature_measure_records() -> None:
    """Explicit counting is distinct from explicit point quadrature."""
    assert AxisMeasure.from_mapping({"kind": "counting"}, size=3).weights == (1, 1, 1)
    weights = AxisMeasure.from_mapping(
        {"kind": "quadrature", "weights": [0.5, 1.5, 1]}, size=3
    )
    # Integral of f(x)=x over [0,3] from nodes 0,1,3 is exactly 4.5.
    assert sum(
        w * x for w, x in zip(weights.weights, (0, 1, 3), strict=True)
    ) == pytest.approx(4.5)


def test_scientific_metadata_snapshot_and_pickle() -> None:
    """Scientific records detach inputs and survive trusted local serialization."""
    spec = _spec()
    compiled = compile_spec(spec)
    axes = cast("list[dict[str, object]]", spec["axes"])
    axes[0]["measure"] = {"kind": "counting"}
    measure = compiled.axis_meta["space"].measure
    assert measure is not None
    assert measure.weights == (0.1, 0.3, 0.6)
    with pytest.raises(TypeError):
        cast("MutableMapping[str, AxisDeclaration]", compiled.axis_meta)["space"] = (
            AxisDeclaration()
        )
    with pytest.raises(TypeError):
        cast("MutableMapping[str, StateDeclaration]", compiled.state_meta)["rho"] = (
            StateDeclaration()
        )
    restored = pickle.loads(pickle.dumps(compiled))  # ruff: ignore[suspicious-pickle-usage]
    assert restored.axis_meta == compiled.axis_meta
    assert restored.state_meta == compiled.state_meta
    np.testing.assert_array_equal(
        restored.eval_fn(0.0, np.ones(7)), [0, 0, 0, 0, 0, 0, 7]
    )


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_physical_measure_example_preserves_native_arrays(backend: str) -> None:
    """A consumer can use declared static weights without changing array type."""
    module = pytest.importorskip("jax.numpy" if backend == "jax" else backend)
    compiled = compile_spec(_spec())
    density = module.asarray([1.1, 1.5, 2.4])
    xp = array_namespace(density)
    measure = compiled.axis_meta["space"].measure
    assert measure is not None
    weights = xp.asarray(measure.weights, dtype=density.dtype)
    cell_counts = xp.multiply(density, weights)
    total = xp.sum(cell_counts)
    assert array_namespace(total) is xp
    assert total.dtype == density.dtype
    assert float(total) == pytest.approx(2.0, rel=1e-6)
    state = xp.concat((density, cell_counts, xp.asarray([0], dtype=density.dtype)))
    derivative = compiled.eval_fn(0.0, state)
    assert array_namespace(derivative) is xp
    assert derivative.dtype == state.dtype
    np.testing.assert_allclose(np.asarray(derivative), [0, 0, 0, 0, 0, 0, 7], rtol=1e-6)
