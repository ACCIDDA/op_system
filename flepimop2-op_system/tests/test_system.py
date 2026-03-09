"""Tests for the flepimop2 op_system adapter System."""

import numpy as np
import pytest

from flepimop2.system.op_system import OpSystemSystem


@pytest.fixture
def sir_spec() -> dict[str, object]:
    """Minimal SIR RHS spec for testing.

    Returns:
        dict[str, object]: Spec dictionary for a simple SIR model.
    """
    return {
        "kind": "expr",
        "state": ["S", "I", "R"],
        "aliases": {"N": "S + I + R"},
        "equations": {
            "S": "-beta * S * I / N",
            "I": "beta * S * I / N - gamma * I",
            "R": "gamma * I",
        },
    }


def test_builds_and_steps(sir_spec: dict[str, object]) -> None:
    """System compiles spec and produces expected derivatives."""
    sys = OpSystemSystem(spec=sir_spec)
    y0 = np.array([0.999, 0.001, 0.0], dtype=np.float64)
    out = sys.step(np.float64(0.0), y0, beta=0.3, gamma=0.1)
    assert out.shape == (3,)
    # Expected derivatives for initial SIR
    np.testing.assert_allclose(
        out,
        np.array([-0.0002997, 0.0001997, 0.0001], dtype=np.float64),
        rtol=1e-12,
        atol=0.0,
    )


def test_invalid_state_shape_raises(sir_spec: dict[str, object]) -> None:
    """Reject non-1D or wrong-length state arrays."""
    sys = OpSystemSystem(spec=sir_spec)
    with pytest.raises(ValueError, match=r"state must be a 1D array"):
        sys.step(np.float64(0.0), np.array([[1.0, 2.0]]), beta=0.1, gamma=0.2)


def test_invalid_spec_raises() -> None:
    """Invalid spec fails during model validation/compile."""
    bad_spec: dict[str, object] = {"kind": "expr", "state": ["x"], "equations": {}}
    with pytest.raises(ValueError, match=r"Missing equation"):
        OpSystemSystem(spec=bad_spec)
