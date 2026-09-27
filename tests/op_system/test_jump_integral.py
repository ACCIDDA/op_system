"""Tests for portable conservative jump-integral semantics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from op_system import (
    compile_spec,
    jump_integral_generator,
    jump_integral_rhs,
    normalize_rhs,
    validate_jump_integral_kernel,
)
from op_system._errors import InvalidRhsSpecError


def _spec(
    *,
    axis_type: str = "ordinal",
    direction: str | None = "both",
    boundary: str | None = None,
    kernel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operator: dict[str, Any] = {
        "kind": "jump_integral",
        "axis": "trait",
        "rate": "nu",
        "kernel": kernel
        or {
            "form": "matrix",
            "params": {"matrix": "J"},
            "param_axes": {"J": ["trait", "trait"]},
        },
    }
    if direction is not None:
        operator["direction"] = direction
    if boundary is not None:
        operator["bc"] = boundary
    return {
        "kind": "expr",
        "axes": [
            {
                "name": "trait",
                "type": axis_type,
                "coords": [0.0, 0.4, 1.0]
                if axis_type == "continuous"
                else ["low", "mid", "high"],
            }
        ],
        "state": ["X[trait]"],
        "equations": {"X[trait]": "0.0"},
        "operators": [operator],
    }


def test_jump_integral_normalizes_to_one_matrix_contract() -> None:
    """Normalized metadata makes direction, boundary, and matrix axes explicit."""
    operator = compile_spec(_spec(direction=None)).operators[0]

    assert operator.kind == "jump_integral"
    assert operator.direction == "both"
    assert operator.bc == "reflecting"
    assert operator.rate == "nu"
    assert operator.apply_to is None
    assert operator.kernel == {
        "form": "matrix",
        "params": {"matrix": "J"},
        "param_axes": {"J": ["trait", "trait"]},
    }


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"kernel": {"form": "gaussian", "params": {"sigma": 1.0}}}, "form"),
        ({"kernel": {"form": "matrix", "params": {}}}, r"kernel\.params"),
        (
            {
                "kernel": {
                    "form": "matrix",
                    "params": {"matrix": "J"},
                    "param_axes": {"J": ["trait"]},
                }
            },
            "param_axes",
        ),
        ({"boundary": "absorbing"}, "bc must be 'reflecting'"),
    ],
)
def test_jump_integral_rejects_ambiguous_schemas(
    changes: dict[str, Any],
    match: str,
) -> None:
    """Unknown forms, incomplete axes, and undefined boundaries fail early."""
    kwargs: dict[str, Any] = {}
    if "kernel" in changes:
        kwargs["kernel"] = changes["kernel"]
    if "boundary" in changes:
        kwargs["boundary"] = changes["boundary"]
    with pytest.raises(InvalidRhsSpecError, match=match):
        normalize_rhs(_spec(**kwargs))


def test_categorical_axes_have_no_up_or_down_direction() -> None:
    """Directional masks require an ordered axis."""
    with pytest.raises(InvalidRhsSpecError, match=r"both.*categorical"):
        normalize_rhs(_spec(axis_type="categorical", direction="up"))


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (
            "up",
            np.asarray([[-5.0, 2.0, 3.0], [0.0, -7.0, 7.0], [0.0, 0.0, 0.0]]),
        ),
        (
            "down",
            np.asarray([[0.0, 0.0, 0.0], [5.0, -5.0, 0.0], [11.0, 13.0, -24.0]]),
        ),
        (
            "both",
            np.asarray([[-5.0, 2.0, 3.0], [5.0, -12.0, 7.0], [11.0, 13.0, -24.0]]),
        ),
    ],
)
def test_ordinal_generator_has_exact_source_target_orientation(
    direction: str,
    expected: np.ndarray,
) -> None:
    """Rows are sources, columns targets, and diagonals balance outflow."""
    kernel = np.asarray([[0.0, 2.0, 3.0], [5.0, 0.0, 7.0], [11.0, 13.0, 0.0]])
    generator = jump_integral_generator(
        kernel,
        axis_type="ordinal",
        direction=direction,
    )

    np.testing.assert_allclose(generator, expected)
    np.testing.assert_allclose(generator.sum(axis=1), 0.0)


def test_continuous_generator_uses_target_quadrature() -> None:
    """A non-uniform continuous grid weights destination columns."""
    kernel = np.asarray([[0.0, 2.0, 3.0], [5.0, 0.0, 7.0], [11.0, 13.0, 0.0]])
    weights = np.asarray([0.2, 0.5, 0.3])
    generator = jump_integral_generator(
        kernel,
        axis_type="continuous",
        quadrature_weights=weights,
    )
    expected_rates = kernel * weights[np.newaxis, :]
    expected = expected_rates - np.diag(expected_rates.sum(axis=1))

    np.testing.assert_allclose(generator, expected)
    np.testing.assert_allclose(generator.sum(axis=1), 0.0, atol=1e-15)


def test_jump_rhs_conserves_mass_across_batches() -> None:
    """The reference contraction conserves every independent batch slice."""
    rng = np.random.default_rng(216)
    state = rng.uniform(0.1, 2.0, size=(4, 3, 2))
    kernel = rng.uniform(0.0, 1.0, size=(3, 3)) * (1.0 - np.eye(3))
    derivative = jump_integral_rhs(
        state,
        kernel,
        axis=1,
        axis_type="ordinal",
        rate=0.4,
        direction="both",
    )

    assert derivative.shape == state.shape
    np.testing.assert_allclose(derivative.sum(axis=1), 0.0, atol=1e-14)


def test_kernel_value_validation_reports_sign_diagonal_and_weights() -> None:
    """Concrete host validation reports each numerical contract violation."""
    kernel = np.asarray([[1.0, -0.2], [0.3, 0.0]])
    problems = validate_jump_integral_kernel(
        kernel,
        axis_type="continuous",
        quadrature_weights=np.asarray([1.0, 0.0]),
    )

    assert "kernel entries must be nonnegative" in problems
    assert "kernel diagonal entries must be zero" in problems
    assert "quadrature_weights must be positive" in problems


def test_jump_integral_supports_jax_jit_and_grad_when_available() -> None:
    """Reference assembly keeps jump-kernel values traceable and differentiable."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    state = jnp.asarray([[1.0, 2.0, 3.0]])
    kernel = jnp.asarray([[0.0, 0.2, 0.1], [0.3, 0.0, 0.4], [0.5, 0.6, 0.0]])

    def loss(matrix: Any) -> Any:  # ruff: ignore[any-type]
        derivative = jump_integral_rhs(
            state,
            matrix,
            axis=1,
            axis_type="ordinal",
        )
        return jnp.sum(derivative**2)

    value = jax.jit(loss)(kernel)
    gradient = jax.jit(jax.grad(loss))(kernel)

    assert np.isfinite(np.asarray(value)).all()
    assert np.isfinite(np.asarray(gradient)).all()


def test_jump_integral_supports_torch_autograd_when_available() -> None:
    """Reference assembly and contraction preserve raw Torch autograd."""
    torch = pytest.importorskip("torch")
    state = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    kernel = torch.tensor(
        [[0.0, 0.2, 0.1], [0.3, 0.0, 0.4], [0.5, 0.6, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    derivative = jump_integral_rhs(
        state,
        kernel,
        axis=1,
        axis_type="ordinal",
    )
    derivative.square().sum().backward()

    assert isinstance(derivative, torch.Tensor)
    assert state.grad is not None
    assert kernel.grad is not None
