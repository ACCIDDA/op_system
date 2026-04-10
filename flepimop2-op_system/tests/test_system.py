"""Tests for the flepimop2 op_system adapter System."""

from types import MappingProxyType

import numpy as np
import pytest
from flepimop2.typing import SystemProtocol

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


def test_option_mixing_kernels_bare_spec(sir_spec: dict[str, object]) -> None:
    """A spec without kernels exposes an empty mixing_kernels option."""
    sys = OpSystemSystem(spec=sir_spec)
    mk = sys.option("mixing_kernels", None)
    assert isinstance(mk, dict)
    assert mk == {}


def test_option_mixing_kernels_with_kernel() -> None:
    """A spec with a kernel exposes it via the mixing_kernels option."""
    spec: dict[str, object] = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["0", "1"]}],
        "kernels": [
            {
                "name": "contact",
                "form": "gaussian",
                "params": {"scale": 1.0, "sigma": 0.5},
            },
        ],
        "state": ["S[loc]"],
        "equations": {"S[loc]": "-S[loc]"},
    }
    sys = OpSystemSystem(spec=spec)
    mk = sys.option("mixing_kernels", None)
    assert isinstance(mk, dict)
    assert "contact" in mk
    assert isinstance(mk["contact"], np.ndarray)


def test_bind_returns_system_protocol(sir_spec: dict[str, object]) -> None:
    """bind() returns a callable satisfying SystemProtocol."""
    sys = OpSystemSystem(spec=sir_spec)
    stepper = sys.bind()
    assert isinstance(stepper, SystemProtocol)


def test_bind_with_static_params(sir_spec: dict[str, object]) -> None:
    """bind(params=...) partially applies parameters to the stepper."""
    sys = OpSystemSystem(spec=sir_spec)
    y0 = np.array([0.999, 0.001, 0.0], dtype=np.float64)

    stepper = sys.bind(params={"beta": 0.3, "gamma": 0.1})
    out = stepper(time=np.float64(0.0), state=y0)

    expected = np.array([-0.0002997, 0.0001997, 0.0001], dtype=np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0.0)


def test_bind_delegates_to_step(sir_spec: dict[str, object]) -> None:
    """bind() without params produces the same result as step()."""
    sys = OpSystemSystem(spec=sir_spec)
    y0 = np.array([0.999, 0.001, 0.0], dtype=np.float64)

    via_step = sys.step(np.float64(0.0), y0, beta=0.3, gamma=0.1)
    via_bind = sys.bind()(time=np.float64(0.0), state=y0, beta=0.3, gamma=0.1)

    np.testing.assert_allclose(via_bind, via_step, rtol=0.0, atol=0.0)


# -- Integration: full SystemABC.bind() contract -----------------------------------


def test_bind_rejects_forbidden_param(sir_spec: dict[str, object]) -> None:
    """SystemABC.bind() rejects 'time' and 'state' in params."""
    sys = OpSystemSystem(spec=sir_spec)
    with pytest.raises(TypeError, match="time"):
        sys.bind(params={"time": 0.0})


def test_bind_roundtrip_multi_step(sir_spec: dict[str, object]) -> None:
    """Bound stepper can be called repeatedly (simulating an engine loop)."""
    sys = OpSystemSystem(spec=sir_spec)
    stepper = sys.bind(params={"beta": 0.3, "gamma": 0.1})
    state = np.array([0.999, 0.001, 0.0], dtype=np.float64)

    for _ in range(5):
        deriv = stepper(time=np.float64(0.0), state=state)
        state += 0.1 * deriv  # simple Euler step

    # Population is conserved: S + I + R == 1.0
    np.testing.assert_allclose(state.sum(), 1.0, atol=1e-14)


# -- Operator wiring (#8, #54) -------------------------------------------------


@pytest.fixture
def sir_with_operators_spec() -> dict[str, object]:
    """SIR spec with a single diffusion operator on the loc axis.

    Returns:
        dict[str, object]: Spec with one operator.
    """
    return {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["S[loc]", "I[loc]", "R[loc]"],
        "equations": {
            "S[loc]": "-beta * S[loc] * I[loc]",
            "I[loc]": "beta * S[loc] * I[loc] - gamma * I[loc]",
            "R[loc]": "gamma * I[loc]",
        },
        "operators": [
            {"kind": "diffusion", "axis": "loc", "bc": "neumann"},
        ],
    }


@pytest.fixture
def multi_operator_spec() -> dict[str, object]:
    """Spec with two operators on different axes.

    Returns:
        dict[str, object]: Spec with operators on loc and age axes.
    """
    return {
        "kind": "expr",
        "axes": [
            {"name": "loc", "coords": ["a", "b"]},
            {"name": "age", "coords": ["young", "old"]},
        ],
        "state": ["u[loc,age]"],
        "equations": {"u[loc,age]": "-u[loc,age]"},
        "operators": [
            {"kind": "diffusion", "axis": "loc"},
            {"kind": "advection", "axis": "age"},
        ],
    }


def test_option_operators_none_when_absent(sir_spec: dict[str, object]) -> None:
    """A spec without operators returns None for the operators option."""
    sys = OpSystemSystem(spec=sir_spec)
    assert sys.option("operators", "missing") is None


def test_option_operators_present(
    sir_with_operators_spec: dict[str, object],
) -> None:
    """A spec with operators exposes them via the operators option."""
    sys = OpSystemSystem(spec=sir_with_operators_spec)
    ops = sys.option("operators", None)
    assert ops is not None
    assert len(ops) == 1
    assert ops[0]["kind"] == "diffusion"
    assert ops[0]["axis"] == "loc"
    assert ops[0]["bc"] == "neumann"


def test_option_operators_returns_immutable(
    sir_with_operators_spec: dict[str, object],
) -> None:
    """Operator dicts returned by option() are immutable MappingProxy objects."""
    sys = OpSystemSystem(spec=sir_with_operators_spec)
    ops = sys.option("operators", None)
    assert ops is not None
    assert isinstance(ops, tuple)
    assert isinstance(ops[0], MappingProxyType)
    with pytest.raises(TypeError):
        ops[0]["kind"] = "advection"  # type: ignore[index]


def test_option_operator_axis_single(
    sir_with_operators_spec: dict[str, object],
) -> None:
    """When all operators share one axis, operator_axis returns that name."""
    sys = OpSystemSystem(spec=sir_with_operators_spec)
    assert sys.option("operator_axis", None) == "loc"


def test_option_operator_axis_none_when_absent(
    sir_spec: dict[str, object],
) -> None:
    """No operators means operator_axis is None."""
    sys = OpSystemSystem(spec=sir_spec)
    assert sys.option("operator_axis", None) is None


def test_option_operator_axis_none_when_mixed(
    multi_operator_spec: dict[str, object],
) -> None:
    """Multiple axes across operators means operator_axis is None."""
    sys = OpSystemSystem(spec=multi_operator_spec)
    assert sys.option("operator_axis", None) is None


def test_option_operators_multi_axis(
    multi_operator_spec: dict[str, object],
) -> None:
    """Multi-operator spec surfaces both operators."""
    sys = OpSystemSystem(spec=multi_operator_spec)
    ops = sys.option("operators", None)
    assert ops is not None
    assert len(ops) == 2
    kinds = {op["kind"] for op in ops}
    assert kinds == {"diffusion", "advection"}


def test_option_operators_independent_of_mixing_kernels(
    sir_with_operators_spec: dict[str, object],
) -> None:
    """Operators and mixing_kernels are independent options."""
    sys = OpSystemSystem(spec=sir_with_operators_spec)
    mk = sys.option("mixing_kernels", None)
    ops = sys.option("operators", None)
    assert isinstance(mk, dict)
    assert ops is not None
    assert len(ops) == 1
