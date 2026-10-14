# flepimop2-op_system: Operator-Partitioned System Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Independent scalar oracles for generated kernels and provider injection."""

import math

import numpy as np
import pytest

from flepimop2.system.op_system import OpSystemSystem


@pytest.mark.parametrize(
    "form", ["erfc", "gaussian", "exponential", "gamma", "power_law"]
)
def test_rectangular_kernel_and_rhs_injection(form: str) -> None:
    """Rows are source coordinates and columns are targets, including quadrature."""
    source = [0.0, 1.0, 3.0]
    target = [0.0, 2.0]
    params = {
        "scale": 1.7,
        "sigma": 1.3,
        "lambda": 0.8,
        "k": 2.5,
        "theta": 0.9,
        "p": 1.2,
    }
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "axes": [
                {"name": "source", "type": "continuous", "coords": source},
                {"name": "target", "type": "continuous", "coords": target},
            ],
            "state": ["S[source]", "T[target]"],
            "equations": {
                "S[source]": "S[source] - S[source]",
                "T[target]": "apply_along(C[source:s,target] * S[source:s], source=s)",
            },
            "kernels": [
                {
                    "name": "C",
                    "axes": ["source", "target"],
                    "form": form,
                    "params": params,
                }
            ],
        }
    )
    expected = np.empty((3, 2))
    for row, a in enumerate(source):
        for column, b in enumerate(target):
            distance = abs(a - b)
            unit = distance / 1.3
            formulas = {
                "erfc": math.erfc(unit),
                "gaussian": math.exp(-unit * unit / 2),
                "exponential": math.exp(-distance / 0.8),
                # This is the declared unnormalized shape, not a gamma PDF.
                "gamma": (max(distance, 1e-12) / 0.9) ** 1.5
                * math.exp(-max(distance, 1e-12) / 0.9),
                "power_law": (1 + unit) ** -1.2,
            }
            expected[row, column] = 1.7 * formulas[form]
    kernels = system.option("mixing_kernels", None)
    assert isinstance(kernels, dict)
    matrix = kernels["C"]
    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.float64
    np.testing.assert_allclose(matrix, expected, rtol=2e-14, atol=0)
    state = np.asarray([2.0, 5.0, 11.0, 0.0, 0.0])
    # Independent trapezoid cell contributions; source endpoint gaps are unequal.
    target_rhs = [
        sum(expected[i, j] * state[i] * w for i, w in enumerate([0.5, 1.5, 1.0]))
        for j in range(2)
    ]
    np.testing.assert_allclose(
        system.step(np.float64(0), state), [0, 0, 0, *target_rhs]
    )
    bound = system.bind()
    assert bound.pytree_stepper is not None
    structured = bound.pytree_stepper(np.float64(0), {"S": state[:3], "T": state[3:]})
    np.testing.assert_allclose(structured["T"], target_rhs)


def test_custom_asymmetric_kernel_orientation() -> None:
    """A supplied contact matrix distinguishes source rows from target columns."""
    matrix = [[1.0, 7.0], [3.0, 2.0]]
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "axes": [{"name": "age", "coords": ["young", "old"]}],
            "state": ["S[age]"],
            "equations": {"S[age]": "apply_along(C[age:s,age] * S[age:s], age=s)"},
            "kernels": [{"name": "C", "axes": ["age", "age"], "value": matrix}],
        }
    )
    np.testing.assert_allclose(
        system.step(np.float64(0), np.asarray([2.0, 5.0])), [17.0, 24.0]
    )


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (
            {
                "name": "C",
                "axes": ["missing"],
                "form": "gaussian",
                "params": {"scale": 1, "sigma": 1},
            },
            "unknown axis",
        ),
        (
            {"name": "C", "axes": ["age"], "form": "unknown", "params": {}},
            "not supported",
        ),
        (
            {"name": "C", "axes": ["age"], "form": "gaussian", "params": {"scale": 1}},
            "missing required key",
        ),
        (
            {"name": "C", "axes": ["age"], "form": "custom_value", "params": {}},
            "requires params.value",
        ),
    ],
)
def test_kernel_errors_at_public_construction(
    kernel: dict[str, object], message: str
) -> None:
    """Malformed declarations fail during construction with contextual errors."""
    with pytest.raises(ValueError, match=message):
        OpSystemSystem(
            spec={
                "kind": "expr",
                "axes": [{"name": "age", "coords": ["young", "old"]}],
                "state": ["S[age]"],
                "equations": {"S[age]": "-S[age]"},
                "kernels": [kernel],
            }
        )


def test_single_axis_kernel_reuses_coordinates() -> None:
    """One declared axis yields a square matrix over that coordinate grid."""
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "axes": [{"name": "age", "type": "continuous", "coords": [0.0, 2.0]}],
            "state": ["S[age]"],
            "equations": {"S[age]": "-S[age]"},
            "kernels": [
                {
                    "name": "C",
                    "axes": ["age"],
                    "form": "gaussian",
                    "params": {"scale": 2.0, "sigma": 2.0},
                }
            ],
        }
    )
    kernels = system.option("mixing_kernels", None)
    assert isinstance(kernels, dict)
    off_diagonal = 2 * math.exp(-0.5)
    np.testing.assert_allclose(kernels["C"], [[2, off_diagonal], [off_diagonal, 2]])


@pytest.mark.parametrize("value", [0.0, 2.5])
def test_kernel_without_axes_retains_scalar_value(value: float) -> None:
    """No-axis declarations use params.value and never generate a matrix."""
    params = {"scale": 1.0, "sigma": 2.0}
    if value:
        params["value"] = value
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "state": ["S"],
            "equations": {"S": "-S"},
            "kernels": [{"name": "C", "form": "gaussian", "params": params}],
        }
    )
    kernels = system.option("mixing_kernels", None)
    assert isinstance(kernels, dict)
    assert kernels["C"].shape == ()
    assert kernels["C"].dtype == np.float64
    assert kernels["C"] == value
