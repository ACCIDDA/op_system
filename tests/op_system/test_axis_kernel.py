"""Tests for ``axis_kernel`` operator validation and reference semantics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from op_system import (
    axis_kernel_generator_rhs,
    axis_kernel_redistribute,
    compile_spec,
    normalize_rhs,
    validate_axis_kernel_matrix,
)
from op_system._errors import InvalidRhsSpecError


def _spec(kernel: dict[str, Any], **operator: object) -> dict[str, Any]:
    return {
        "kind": "transitions",
        "axes": [
            {"name": "vax", "coords": ["u", "v"]},
            {"name": "imm", "type": "ordinal", "coords": ["x0", "x1", "x2"]},
        ],
        "state": ["X[vax, imm]"],
        "transitions": [
            {"from": "X[vax=u, imm]", "to": "X[vax=v, imm]", "rate": "nu"},
        ],
        "operators": [
            {"kind": "axis_kernel", "axis": "imm", "kernel": kernel, **operator}
        ],
    }


GENERATOR_KERNEL: dict[str, Any] = {
    "form": "generator",
    "params": {"matrix": "G"},
    "param_axes": {"G": ["imm", "imm"]},
}
TRANSFER_KERNEL: dict[str, Any] = {
    "form": "stochastic",
    "params": {"matrix": "K", "nu": "nu"},
    "param_axes": {"K": ["imm", "imm"]},
    "transfer": {"axis": "vax", "source": "u", "target": "v", "rate": ["nu"]},
}


def test_valid_axis_kernel_specs_normalize_and_compile() -> None:
    """Generator and transfer kernels normalize, compile, and keep their form."""
    for kernel, extra in ((GENERATOR_KERNEL, {"velocity": "w"}), (TRANSFER_KERNEL, {})):
        compiled = compile_spec(_spec(kernel, **extra))
        (operator,) = compiled.operators
        assert operator.kind == "axis_kernel"
        assert operator.kernel is not None
        assert operator.kernel["form"] == kernel["form"]


@pytest.mark.parametrize(
    ("kernel", "match"),
    [
        ({**GENERATOR_KERNEL, "form": "dense"}, "kernel.form"),
        ({**GENERATOR_KERNEL, "params": {}}, "params.matrix"),
        ({**GENERATOR_KERNEL, "param_axes": {"G": ["imm"]}}, "param_axes"),
        ({**TRANSFER_KERNEL, "form": "generator"}, "requires form 'stochastic'"),
        (
            {
                **TRANSFER_KERNEL,
                "transfer": {**TRANSFER_KERNEL["transfer"], "axis": "imm"},
            },
            "different axis",
        ),
        (
            {
                **TRANSFER_KERNEL,
                "transfer": {**TRANSFER_KERNEL["transfer"], "target": "w"},
            },
            "not a coordinate",
        ),
        (
            {
                **TRANSFER_KERNEL,
                "transfer": {**TRANSFER_KERNEL["transfer"], "rate": ["eta"]},
            },
            "kernel.params",
        ),
    ],
)
def test_malformed_axis_kernels_fail_at_normalization(
    kernel: dict[str, Any], match: str
) -> None:
    """Malformed axis_kernel specs raise at normalization with a precise field."""
    with pytest.raises(InvalidRhsSpecError, match=match):
        normalize_rhs(_spec(kernel))


def _generator(rng: np.random.Generator, n: int) -> np.ndarray:
    rates = rng.uniform(0.0, 1.0, size=(n, n)) * (1.0 - np.eye(n))
    return rates - np.diag(rates.sum(axis=1))


def test_generator_rhs_conserves_mass_along_the_axis() -> None:
    """The generator term conserves mass and equals velocity * x @ G."""
    rng = np.random.default_rng(0)
    state = rng.uniform(1.0, 2.0, size=(4, 5, 3))
    generator = _generator(rng, 5)
    derivative = axis_kernel_generator_rhs(state, generator, axis=1, velocity=0.3)
    assert derivative.shape == state.shape
    np.testing.assert_allclose(derivative.sum(axis=1), 0.0, atol=1e-12)
    expected = 0.3 * np.einsum("aic,ij->ajc", state, generator)
    np.testing.assert_allclose(derivative, expected, rtol=1e-12)


def test_redistribution_conserves_the_transferred_flux() -> None:
    """Redistribution moves the transferred flux without creating mass."""
    rng = np.random.default_rng(1)
    flux = rng.uniform(0.0, 1.0, size=(6, 4))
    kernel = rng.uniform(0.0, 1.0, size=(4, 4))
    kernel /= kernel.sum(axis=1, keepdims=True)
    delta = axis_kernel_redistribute(flux, kernel)
    np.testing.assert_allclose(delta.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(flux + delta, flux @ kernel, rtol=1e-12)


def test_matrix_validation_reports_form_violations() -> None:
    """Matrix validation reports each violated form property."""
    rng = np.random.default_rng(2)
    assert validate_axis_kernel_matrix(_generator(rng, 3), form="generator") == []
    assert validate_axis_kernel_matrix(np.eye(3), form="stochastic") == []
    assert "generator rows must sum to zero" in validate_axis_kernel_matrix(
        np.ones((3, 3)), form="generator"
    )
    assert "stochastic rows must sum to one" in validate_axis_kernel_matrix(
        np.full((3, 3), 0.5), form="stochastic"
    )
    assert validate_axis_kernel_matrix(np.ones((2, 3)), form="stochastic") == [
        "matrix must be square"
    ]
    with pytest.raises(ValueError, match="unknown axis_kernel form"):
        validate_axis_kernel_matrix(np.eye(2), form="dense")
