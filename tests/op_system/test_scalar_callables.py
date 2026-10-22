"""Scalar and latent-observable callables bind fresh values without runtime eval."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pytest

from op_system import compile_spec

if TYPE_CHECKING:
    from collections.abc import Callable

    from op_system.compile import CompiledRhs


def _model() -> CompiledRhs:
    return compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "aliases": {"offset": "forcing+1", "nested": "offset*0.5"},
        "equations": {"x": "-rate*x*z+nested", "z": "rate*x*z-nested"},
        "observables": {"measurement": {"expr": "gain*x+z"}},
    })


def test_numpy_scalar_and_observable_do_not_eval_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aliases, shared subexpressions and observations use prebuilt safe callables."""
    model = _model()
    y = np.array([2.0, 4.0])

    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("Scalar/observable runtime unexpectedly called eval")

    with monkeypatch.context() as context:
        context.setattr(builtins, "eval", forbidden)
        first = model.eval_fn(0.0, y, rate=np.array(0.3), forcing=np.array(0.0))
        observed = model.observables["measurement"].eval_fn(0.0, y, gain=np.array(1.5))
        second = model.eval_fn(0.0, y, rate=np.array(0.7), forcing=np.array(0.2))
        changed = model.observables["measurement"].eval_fn(0.0, y, gain=np.array(3.0))
    np.testing.assert_allclose(first, [-1.9, 1.9], rtol=1e-14)
    np.testing.assert_allclose(second, [-5.0, 5.0], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(observed), 7.0)
    np.testing.assert_allclose(np.asarray(changed), 10.0)
    np.testing.assert_array_equal(y, [2.0, 4.0])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_torch_scalar_observable_values_gradients_and_current_capture(
    dtype: Literal["float32", "float64"],
) -> None:
    """Keep eager minimum support and qualify strict capture on current Torch."""
    torch = pytest.importorskip("torch")
    model = _model()
    graphs = []

    def capture(
        graph: Any,  # ruff: ignore[any-type]
        _inputs: object,
    ) -> Callable[..., object]:
        graphs.append(graph)
        return cast("Callable[..., object]", graph.forward)

    def operation(y: Any, parameters: Any) -> Any:  # ruff: ignore[any-type]
        derivative: Any = model.eval_fn(
            0.0, y, rate=parameters[0], forcing=parameters[1]
        )
        observed = model.observables["measurement"].eval_fn(0.0, y, gain=parameters[2])
        return derivative.square().sum() + observed

    current_capture = tuple(int(part) for part in torch.__version__.split(".")[:2]) >= (
        2,
        14,
    )
    # Older supported Torch remains an eager/AD contract. Its compiler fails
    # namespace-helper constructs before reaching this cached expression path.
    native = (
        torch.compile(operation, backend=capture, fullgraph=True, dynamic=False)
        if current_capture
        else operation
    )
    state = torch.tensor([2.0, 4.0], dtype=getattr(torch, dtype), requires_grad=True)
    for values in ([0.3, 0.0, 1.5], [0.7, 0.2, 3.0]):
        parameters = torch.tensor(
            values, dtype=getattr(torch, dtype), requires_grad=True
        )
        result = native(state, parameters)
        gradient_y, gradient_p = torch.autograd.grad(result, (state, parameters))
        rate, forcing, gain = values
        flow = rate * 2 * 4 - (forcing + 1) * 0.5
        expected = 2 * flow**2 + gain * 2 + 4
        np.testing.assert_allclose(
            result.detach().numpy(),
            expected,
            rtol=3e-6 if dtype == "float32" else 1e-13,
        )
        np.testing.assert_allclose(
            gradient_y.detach().numpy(),
            [4 * flow * rate * 4 + gain, 4 * flow * rate * 2 + 1],
            rtol=3e-6 if dtype == "float32" else 1e-13,
        )
        np.testing.assert_allclose(
            gradient_p.detach().numpy(),
            [4 * flow * 8, -2 * flow, 2],
            rtol=3e-6 if dtype == "float32" else 1e-13,
        )
    assert len(graphs) == (1 if current_capture else 0)
