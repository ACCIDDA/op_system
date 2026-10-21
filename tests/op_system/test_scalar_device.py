"""Generated constants retain native devices; time inputs keep their graphs."""

from __future__ import annotations

from typing import Any

import pytest

from op_system import compile_spec, symbolic_derivatives


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_torch_scalar_constants_and_derivative_zeros_follow_meta_device(
    dtype: str,
) -> None:
    """Meta allocation catches mixed-device stack failures without requiring CUDA."""
    torch = pytest.importorskip("torch")
    compiled = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "equations": {"x": "-rate*x", "z": "0.0"},
        "observables": {"measurement": {"expr": "gain*x+t"}},
    })
    y = torch.ones(2, dtype=getattr(torch, dtype), device="meta")
    parameters = {
        name: torch.ones((), dtype=y.dtype, device="meta") for name in ("rate", "gain")
    }
    rhs = compiled.eval_fn(0.5, y, rate=parameters["rate"])
    assert str(getattr(rhs, "device", None)) == "meta"
    assert rhs.dtype == y.dtype
    for matrix in symbolic_derivatives(compiled).evaluate(0.5, y, **parameters):
        assert str(getattr(matrix, "device", None)) == "meta"
        assert getattr(matrix, "dtype", None) == y.dtype
    timed = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "equations": {"x": "t", "z": "x"},
    })
    assert str(getattr(timed.eval_fn(0.5, y), "device", None)) == "meta"


def test_torch_native_time_gradient_is_retained() -> None:
    """Supported old Torch must not detach a differentiable scalar time binding."""
    torch = pytest.importorskip("torch")
    compiled = compile_spec({"kind": "expr", "state": ["x"], "equations": {"x": "t*x"}})
    time = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    result: Any = compiled.eval_fn(time, torch.tensor([2.0], dtype=torch.float64))
    result.sum().backward()
    assert time.grad.item() == pytest.approx(2.0)
