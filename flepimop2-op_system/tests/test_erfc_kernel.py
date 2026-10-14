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

"""Public erfc kernel construction, including nonzero complementary tails."""

import numpy as np
import pytest

from flepimop2.system.op_system import OpSystemSystem


@pytest.mark.parametrize("sigma", [1.0, 2.0])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_erfc_kernel_values(sigma: float, scale: float) -> None:
    """Known values independently checked with scipy.special.erfc stay positive."""
    distances = np.asarray([0.0, 0.5, 1.0, 2.0, 8.0])
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "axes": [
                {
                    "name": "age",
                    "type": "continuous",
                    "coords": (sigma * distances).tolist(),
                }
            ],
            "state": ["S[age]"],
            "equations": {"S[age]": "-S[age]"},
            "kernels": [
                {
                    "name": "C",
                    "axes": ["age", "age"],
                    "form": "erfc",
                    "params": {"scale": scale, "sigma": sigma},
                }
            ],
        }
    )
    kernels = system.option("mixing_kernels", None)
    assert isinstance(kernels, dict)
    matrix = kernels["C"]
    assert matrix.shape == (5, 5)
    assert matrix.dtype == np.float64
    assert np.isfinite(matrix).all()
    # Fixed numerical references avoid sharing the implementation's math.erfc.
    expected = scale * np.asarray([
        1.0,
        0.4795001221869535,
        0.15729920705028516,
        0.004677734981047266,
        1.1224297172982928e-29,
    ])
    np.testing.assert_allclose(matrix[0], expected, rtol=2e-14, atol=0)
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), np.full(5, scale))
    assert matrix[0, -1] > 0
