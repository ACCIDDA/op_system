"""Parameter declarations preserve numerical behavior and validate annotations."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from array_api_compat import array_namespace  # type: ignore[import-untyped]

from op_system import ParameterDeclaration, ParameterSupport, compile_spec
from op_system._errors import InvalidRhsSpecError

if TYPE_CHECKING:
    from collections.abc import MutableMapping


def _spec(kind: str) -> dict[str, object]:
    """Build equivalent scalar expression and transition fixtures.

    Returns:
        A model with inferred rate and offset parameters.
    """
    if kind == "expr":
        return {
            "kind": kind,
            "state": ["x", "loss"],
            "equations": {"x": "offset - rate*x", "loss": "rate*x"},
        }
    return {
        "kind": kind,
        "state": ["x", "loss"],
        "transitions": [
            {"from": "x", "to": "loss", "rate": "rate"},
            {"from": None, "to": "x", "rate": "offset"},
        ],
    }


@pytest.mark.parametrize("kind", ["expr", "transitions"])
def test_partial_declarations_preserve_inference_and_numerics(kind: str) -> None:
    """Fixed/support annotations do not bind or enforce runtime values."""
    spec = _spec(kind)
    spec["parameters"] = {
        "rate": {
            "support": "positive",
            "units": "day^-1",
            "description": "Loss rate",
            "fixed": True,
        }
    }
    compiled = compile_spec(spec)
    assert compiled.param_names == ("offset", "rate")
    assert compiled.param_meta == {
        "rate": ParameterDeclaration(
            ParameterSupport("positive"), "day^-1", "Loss rate", fixed=True
        )
    }
    state = np.array([3.0, 0.0])
    np.testing.assert_array_equal(
        compiled.eval_fn(0.0, state, rate=2.0, offset=1.0), [-5.0, 6.0]
    )
    np.testing.assert_array_equal(
        compiled.eval_fn(0.0, state, rate=-2.0, offset=1.0), [7.0, -6.0]
    )


@pytest.mark.parametrize("kind", ["expr", "transitions"])
def test_undeclared_specs_retain_empty_metadata(kind: str) -> None:
    """Annotations stay optional and undeclared parameter domains stay unknown."""
    assert compile_spec(_spec(kind)).param_meta == {}
    spec = _spec(kind)
    spec["parameters"] = {"rate": {}}
    assert compile_spec(spec).param_meta["rate"] == ParameterDeclaration()


@pytest.mark.parametrize("kind", ["expr", "transitions"])
@pytest.mark.parametrize("name", ["unknown", "x", 1])
def test_declarations_require_inferred_parameter_names(kind: str, name: object) -> None:
    """State names and unused declarations cannot create new parameters."""
    spec = _spec(kind)
    spec["parameters"] = {name: {"support": "real"}}
    with pytest.raises(InvalidRhsSpecError, match="inferred runtime parameter"):
        compile_spec(spec)


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        1,
        {"prior": "normal"},
        {"fixed": 1},
        {"units": " "},
        {"description": False},
        {"support": None},
        {"support": "unknown"},
        {"support": "bounded"},
        {"support": {"kind": "bounded", "lower": True, "upper": 1}},
        {"support": {"kind": "bounded", "lower": 1, "upper": 1}},
        {"support": {"kind": "bounded", "lower": 2, "upper": 1}},
        {"support": {"kind": "bounded", "lower": 0, "upper": float("inf")}},
        {"support": {"kind": "bounded", "lower": float("nan"), "upper": 1}},
        {"support": {"kind": "bounded", "lower": 0, "upper": 10**400}},
        {"support": {"kind": "bounded", "lower": 0}},
    ],
)
def test_malformed_parameter_annotations_fail(declaration: object) -> None:
    """Malformed supports and annotations produce domain-specific errors."""
    spec = _spec("expr")
    spec["parameters"] = {"rate": declaration}
    with pytest.raises(InvalidRhsSpecError):
        compile_spec(spec)


def test_bounded_support_and_snapshot_survive_pickle() -> None:
    """Compiled metadata is detached, read-only and restored by recompilation."""
    declaration: dict[str, object] = {
        "support": {"kind": "bounded", "lower": -1, "upper": 3},
    }
    spec = _spec("expr")
    spec["parameters"] = {"rate": declaration}
    compiled = compile_spec(spec)
    declaration["support"] = "real"
    assert compiled.param_meta["rate"].support == ParameterSupport("bounded", -1, 3)
    with pytest.raises(TypeError):
        cast("MutableMapping[str, ParameterDeclaration]", compiled.param_meta)[
            "rate"
        ] = ParameterDeclaration()
    restored = pickle.loads(pickle.dumps(compiled))  # ruff: ignore[suspicious-pickle-usage]
    assert restored.param_meta == compiled.param_meta
    np.testing.assert_array_equal(
        restored.eval_fn(0.0, np.array([2.0, 0.0]), rate=0.5, offset=3.0), [2.0, 1.0]
    )


def test_shaped_parameter_declaration_uses_runtime_base_name() -> None:
    """Shaped parameter metadata belongs to its buffer rather than each cell."""
    compiled = compile_spec({
        "kind": "expr",
        "axes": [{"name": "unit", "type": "categorical", "coords": ["a", "b"]}],
        "state": ["x[unit]"],
        "equations": {"x[unit]": "-rate[unit]*x[unit]"},
        "parameters": {"rate": {"support": "positive"}},
    })
    assert compiled.param_names == ()
    assert compiled.param_meta["rate"].support == ParameterSupport("positive")
    np.testing.assert_array_equal(
        compiled.eval_fn(0.0, np.array([2.0, 3.0]), rate=np.array([0.5, 2.0])),
        [-1.0, -6.0],
    )


def test_time_varying_parameter_and_alias_declarations() -> None:
    """Runtime buffers retain annotations after time/alias normalization."""
    compiled = compile_spec({
        "kind": "expr",
        "axes": [
            {"name": "unit", "coords": ["a", "b"]},
            {"name": "time", "type": "continuous", "coords": [0.0, 1.0]},
        ],
        "state": ["x[unit]"],
        "aliases": {"drift[unit]": "-rate[time,unit]*x[unit]"},
        "equations": {"x[unit]": "drift[unit]"},
        "parameters": {"rate": {"support": "positive"}},
    })
    assert compiled.param_meta["rate"].support == ParameterSupport("positive")
    np.testing.assert_array_equal(
        compiled.eval_fn(
            0.0,
            np.array([2.0, 4.0]),
            rate=np.array([[0.5, 1.0], [2.0, 3.0]]),
        ),
        [-1.0, -4.0],
    )


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_declarations_preserve_native_evaluation(backend: str) -> None:
    """Annotations do not cause native state/parameter arrays to be coerced."""
    module = pytest.importorskip("jax.numpy" if backend == "jax" else backend)
    spec = _spec("expr")
    spec["parameters"] = {"rate": {"support": "positive"}}
    compiled = compile_spec(spec)
    state = module.asarray([3.0, 0.0])
    rate = module.asarray(2.0)
    result = compiled.eval_fn(0.0, state, rate=rate, offset=1.0)
    assert array_namespace(result) is array_namespace(state)
    assert result.dtype == state.dtype
    np.testing.assert_array_equal(np.asarray(result), [-5.0, 6.0])


@pytest.mark.parametrize("kind", ["real", "positive", "unit_interval"])
def test_named_support_domains(kind: str) -> None:
    """All supported named domains are retained without inferred defaults."""
    spec = _spec("expr")
    spec["parameters"] = {"rate": {"support": kind}}
    support = compile_spec(spec).param_meta["rate"].support
    assert support is not None
    assert support.kind == kind
    assert support.lower is None
    assert support.upper is None


@pytest.mark.parametrize("declarations", [[], "rate", {"rate": {"typo": True}}])
def test_invalid_declaration_blocks(declarations: object) -> None:
    """A declarations block cannot be a value table or an untyped container."""
    spec = _spec("expr")
    spec["parameters"] = declarations
    with pytest.raises(InvalidRhsSpecError):
        compile_spec(spec)
