"""Independent state maps, static schedule validation and native derivatives."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import Any, cast

import numpy as np
import pytest

from op_system import StructuralModelIdentity, compile_spec
from op_system._errors import InvalidRhsSpecError, UnsupportedFeatureError


def _spec(kind: str = "expr") -> dict[str, object]:
    spec: dict[str, object] = {
        "kind": kind,
        "axes": [{"name": "unit", "coords": ["a", "b"]}],
        "state": ["x[unit]", "z[unit]"],
        "parameters": {"boost": {"support": "positive"}},
        "impulses": [
            {
                "time": 1.0,
                "updates": [
                    {"target": "x[unit=a]", "operation": "add", "expr": "boost[unit]"},
                    {"target": "z[unit]", "operation": "set", "expr": "x[unit]"},
                ],
            },
            {
                "time": 2.0,
                "updates": [
                    {"target": "x[unit]", "operation": "multiply", "expr": "scale"}
                ],
            },
        ],
    }
    if kind == "expr":
        spec["equations"] = {"x[unit]": "-k*x[unit]", "z[unit]": "0"}
    else:
        spec["transitions"] = [{"from": "x[unit]", "to": "z[unit]", "rate": "k"}]
    return spec


@pytest.mark.parametrize("kind", ["expr", "transitions"])
def test_numpy_simultaneous_selectors_and_parameter_separation(kind: str) -> None:
    """A selected addition and a full pre-state copy have independently known values."""
    model = compile_spec(_spec(kind))
    state = np.array([2.0, 3.0, 10.0, 20.0])
    before = state.copy()
    first, second = model.impulses
    assert model.param_names == ("k",)
    assert "boost" in model.param_meta
    assert model.impulse_times == (1.0, 2.0)
    assert first.parameter_axes == (("boost", ("unit",)),)
    assert first.parameter_shapes == (("boost", (2,)),)
    after = first.eval_fn(1.0, state, boost=np.array([5.0, 100.0]))
    np.testing.assert_array_equal(after, [7.0, 3.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        second.eval_fn(2.0, after, scale=3.0), [21.0, 9.0, 2.0, 3.0]
    )
    np.testing.assert_array_equal(state, before)
    assert getattr(after, "dtype", None) == state.dtype
    model.eval_fn(0.0, state, k=0.5)


def test_numpy_exact_decay_between_recorded_maps() -> None:
    """Analytic decay exposes pre/post semantics without a solver dependency."""
    model = compile_spec(_spec())
    pre = np.array([2.0, 3.0, 10.0, 20.0])
    pre[:2] *= np.exp(-0.5)
    first = np.asarray(model.impulses[0].eval_fn(1.0, pre, boost=np.array([5.0, 8.0])))
    first[:2] *= np.exp(-0.5)
    result = model.impulses[1].eval_fn(2.0, first, scale=3.0)
    np.testing.assert_allclose(
        np.asarray(result),
        [
            3 * (2 * np.exp(-1) + 5 * np.exp(-0.5)),
            9 * np.exp(-1),
            2 * np.exp(-0.5),
            3 * np.exp(-0.5),
        ],
    )


def test_numpy_transfer_preserves_total_and_reads_prestate() -> None:
    """Disjoint set maps exchange states simultaneously, preserving their total."""
    model = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "equations": {"x": "0", "z": "0"},
        "impulses": [
            {
                "time": 0.0,
                "updates": [
                    {"target": "x", "operation": "set", "expr": "z"},
                    {"target": "z", "operation": "set", "expr": "x"},
                ],
            }
        ],
    })
    np.testing.assert_array_equal(
        model.impulses[0].eval_fn(0.0, np.array([2.0, 7.0])), [7.0, 2.0]
    )


def test_numpy_metadata_pickle_identity_and_noevent_compatibility() -> None:
    """Event changes affect identity; source mutations cannot change maps."""
    spec = _spec()
    model = compile_spec(spec)
    identity = StructuralModelIdentity.from_compiled(model)
    restored = pickle.loads(pickle.dumps(model))  # ruff: ignore[suspicious-pickle-usage]
    assert StructuralModelIdentity.from_compiled(restored) == identity
    np.testing.assert_array_equal(
        restored.impulses[0].eval_fn(
            1.0, np.array([2.0, 3.0, 4.0, 5.0]), boost=np.array([5.0, 6.0])
        ),
        [7.0, 3.0, 2.0, 3.0],
    )
    events = cast("list[dict[str, object]]", spec["impulses"])
    events[0]["time"] = 0.5
    assert model.impulse_times == (1.0, 2.0)
    assert StructuralModelIdentity.from_compiled(compile_spec(spec)) != identity
    for field, value in [
        ("target", "x[unit=b]"),
        ("expr", "2*boost[unit]"),
        ("operation", "set"),
    ]:
        changed = _spec()
        updates = cast(
            "list[dict[str, str]]",
            cast("list[dict[str, object]]", changed["impulses"])[0]["updates"],
        )
        updates[0][field] = value
        assert StructuralModelIdentity.from_compiled(compile_spec(changed)) != identity
    empty = _spec()
    empty.pop("impulses")
    empty.pop("parameters")
    absent = compile_spec(empty)
    assert absent.impulses == ()
    assert absent.impulse_times == ()
    empty["impulses"] = []
    assert StructuralModelIdentity.from_compiled(
        absent
    ) == StructuralModelIdentity.from_compiled(compile_spec(empty))


@pytest.mark.parametrize(
    "times",
    [[1.0, 1.0], [2.0, 1.0], [float("nan")], [float("inf")], [True], ["1"], [None]],
)
def test_invalid_times_fail_before_compilation(times: list[object]) -> None:
    """Recorded times are real, finite and strictly increasing, without coercion."""
    spec = _spec()
    spec["impulses"] = [
        {
            "time": time,
            "updates": [{"target": "x[unit]", "operation": "add", "expr": "1"}],
        }
        for time in times
    ]
    with pytest.raises(InvalidRhsSpecError, match="finite and strictly increasing"):
        compile_spec(spec)


@pytest.mark.parametrize(
    "updates",
    [
        [{"target": "missing", "operation": "set", "expr": "1"}],
        [{"target": "x", "operation": "set", "expr": "1"}],
        [{"target": "x[unit=missing]", "operation": "set", "expr": "1"}],
        [{"target": "x[unit]", "operation": "divide", "expr": "1"}],
        [{"target": "x[unit]", "operation": "set", "expr": "1e999"}],
        [
            {"target": "x[unit]", "operation": "add", "expr": "1"},
            {"target": "x[unit=a]", "operation": "set", "expr": "2"},
        ],
        [],
    ],
)
def test_invalid_maps_fail_before_compilation(updates: object) -> None:
    """Unknown selections, overlapping writes and nonfinite literals are errors."""
    spec = _spec()
    spec["impulses"] = [{"time": 1.0, "updates": updates}]
    with pytest.raises(InvalidRhsSpecError):
        compile_spec(spec)


def test_unsupported_expression_and_inconsistent_parameter_axes() -> None:
    """Unsupported expression features and conflicting map parameter layouts fail."""
    spec = _spec()
    spec["impulses"] = [
        {
            "time": 1.0,
            "updates": [
                {"target": "x[unit]", "operation": "add", "expr": "np.sin(x[unit])"}
            ],
        }
    ]
    with pytest.raises(UnsupportedFeatureError):
        compile_spec(spec)
    spec["impulses"] = [
        {
            "time": 1.0,
            "updates": [
                {"target": "x[unit]", "operation": "add", "expr": "a[unit]"},
                {"target": "z[unit]", "operation": "set", "expr": "a"},
            ],
        }
    ]
    with pytest.raises(InvalidRhsSpecError, match="inconsistent axes"):
        compile_spec(spec)


def test_numpy_wrong_shapes_and_dtypes_are_explicit_errors() -> None:
    """Map inputs enforce the original state and parameter layout."""
    event = compile_spec(_spec()).impulses[0]
    with pytest.raises(ValueError, match="flat state shape"):
        event.eval_fn(1.0, np.ones((4, 1)), boost=np.ones(2))
    with pytest.raises(ValueError, match="must have shape"):
        event.eval_fn(1.0, np.ones(4), boost=np.ones(3))
    with pytest.raises(TypeError, match="real floating"):
        event.eval_fn(1.0, np.ones(4, dtype=int), boost=np.ones(2))
    with pytest.raises(ValueError, match="requires parameter"):
        event.eval_fn(1.0, np.ones(4))


def test_jax_jit_and_gradient_with_frozen_times() -> None:
    """A selected addition has unit gain derivative; copies read pre-values."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    event = compile_spec(_spec()).impulses[0]
    state = jnp.array([2.0, 3.0, 4.0, 5.0])
    evaluate = jax.jit(lambda gain: event.eval_fn(1.0, state, boost=gain))
    np.testing.assert_array_equal(
        np.asarray(evaluate(jnp.array([5.0, 8.0]))), [7.0, 3.0, 2.0, 3.0]
    )
    np.testing.assert_array_equal(
        np.asarray(jax.jacrev(evaluate)(jnp.array([5.0, 8.0]))),
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    assert evaluate(jnp.array([5.0, 8.0])).dtype == state.dtype


def test_torch_native_gradient_and_pytree() -> None:
    """Native Torch maps preserve autograd and the full named state layout."""
    torch = pytest.importorskip("torch")
    event = compile_spec(_spec()).impulses[0]
    state = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64, requires_grad=True)
    gain = torch.tensor([5.0, 8.0], dtype=torch.float64, requires_grad=True)
    value = cast("Any", event.eval_fn(1.0, state, boost=gain))
    np.testing.assert_array_equal(value.detach().numpy(), [7.0, 3.0, 2.0, 3.0])
    value.sum().backward()
    np.testing.assert_array_equal(gain.grad.numpy(), [1.0, 0.0])
    np.testing.assert_array_equal(state.grad.numpy(), [2.0, 2.0, 0.0, 0.0])
    assert value.dtype == state.dtype
    assert value.device == state.device
    assert value.shape == state.shape


def test_numpy_disjoint_selectors_on_two_axes() -> None:
    """Independent selections preserve unselected cells and original axis ordering."""
    model = compile_spec({
        "kind": "expr",
        "axes": [
            {"name": "unit", "coords": ["a", "b"]},
            {"name": "channel", "coords": ["low", "high"]},
        ],
        "state": ["x[unit,channel]"],
        "equations": {"x[unit,channel]": "0"},
        "impulses": [
            {
                "time": 1.0,
                "updates": [
                    {
                        "target": "x[unit=a,channel]",
                        "operation": "add",
                        "expr": "amount[unit]",
                    },
                    {
                        "target": "x[channel=high,unit=b]",
                        "operation": "set",
                        "expr": "2",
                    },
                ],
            }
        ],
    })
    state = np.array([1.0, 2.0, 3.0, 4.0])
    result = model.impulses[0].eval_fn(1.0, state, amount=np.array([10.0, 20.0]))
    np.testing.assert_array_equal(result, [11.0, 12.0, 3.0, 2.0])
    assert model.reaction_gaps[-1].reason == "scheduled_state_maps"
