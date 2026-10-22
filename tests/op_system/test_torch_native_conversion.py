"""Native shaped inputs and time retain gradients on the minimum Torch stack."""

from typing import Any, Literal

import numpy as np
import pytest

from op_system import compile_spec


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("layout", ["flat", "structured"])
def test_torch_shaped_parameter_and_time_gradients(
    dtype: str, layout: Literal["flat", "structured"]
) -> None:
    """A native conversion preserves state, field and scalar-time graph edges."""
    torch = pytest.importorskip("torch")
    model = compile_spec({
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["X[loc]"],
        "equations": {"X[loc]": "gain[loc]*t*X[loc]"},
    })
    state = torch.tensor([1.0, 2.0], dtype=getattr(torch, dtype), requires_grad=True)
    gain = torch.tensor([0.25, 0.75], dtype=getattr(torch, dtype), requires_grad=True)
    time = torch.tensor(0.5, dtype=getattr(torch, dtype), requires_grad=True)
    if layout == "flat":
        result: Any = model.eval_fn(time, state, gain=gain)
    else:
        assert model.pytree_eval_fn is not None
        result = model.pytree_eval_fn(time, {"X": state}, gain=gain)["X"]
    result.sum().backward()
    assert state.grad is not None
    assert gain.grad is not None
    assert time.grad is not None
    np.testing.assert_allclose(result.detach().numpy(), [0.125, 0.75], rtol=1e-6)
    np.testing.assert_allclose(state.grad.numpy(), [0.125, 0.375], rtol=1e-6)
    np.testing.assert_allclose(gain.grad.numpy(), [0.5, 1.0], rtol=1e-6)
    np.testing.assert_allclose(time.grad.numpy(), 1.75, rtol=1e-6)
