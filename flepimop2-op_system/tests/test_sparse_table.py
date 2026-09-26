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

"""Tests for the ``sparse_table`` parameter module and routing transitions (#88)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from flepimop2.axis import Axis, AxisCollection
from flepimop2.parameter.abc import ParameterRequest, build
from pydantic import ValidationError

from flepimop2.parameter.sparse_table import SparseTableParameter
from flepimop2.system.op_system import OpSystemSystem

IMM = ("x0", "x1", "x2", "x3")
AXES = AxisCollection({
    "imm": Axis(name="imm", kind="categorical", size=4, labels=IMM),
    "vax": Axis(name="vax", kind="categorical", size=2, labels=("u", "v")),
    "time": Axis(name="time", kind="continuous", size=4, domain=(0.0, 3.0)),
})


def _routing_spec() -> dict[str, Any]:
    return {
        "kind": "transitions",
        "axes": [
            {"name": "vax", "coords": ["u", "v"]},
            {"name": "imm", "type": "ordinal", "coords": list(IMM)},
            {
                "name": "time",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 3.0},
                "size": 4,
            },
        ],
        "state": ["X[vax, imm]"],
        "transitions": [
            {
                "from": "X[vax=u, imm:i]",
                "to": "X[vax=v, imm:j]",
                "rate": "nu * eta[time, imm:i, imm:j]",
            }
        ],
    }


def _table() -> SparseTableParameter:
    return SparseTableParameter(
        indices=("imm", "imm"),
        entries=[
            {
                "index": ["x0", "x2"],
                "value": {"module": "fixed", "value": [0.1, 0.2, 0.3, 0.4]},
            },
            {"index": ["x1", "x3"], "value": 0.5},
        ],
    )


def test_sparse_table_is_built_by_module_name() -> None:
    """``module: sparse_table`` resolves to this provider's module."""
    param = build({"module": "sparse_table", "indices": ["imm", "imm"], "entries": []})
    assert isinstance(param, SparseTableParameter)


def test_sparse_table_fills_support_and_nests_time_series() -> None:
    """Scalars broadcast over leading axes; nested configs get the leading axes."""
    request = ParameterRequest(name="eta", axes=("time", "imm", "imm"))
    value = np.asarray(_table().sample(axes=AXES, request=request).value)
    assert value.shape == (4, 4, 4)
    np.testing.assert_allclose(value[:, 0, 2], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(value[:, 1, 3], 0.5)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 2] = mask[1, 3] = False
    assert not value[:, mask].any()
    assert _table().support(AXES) == ((0, 2), (1, 3))


def test_sparse_table_feeds_a_routing_transition() -> None:
    """The provider requests ``eta`` over (time, imm, imm) and routes with it."""
    system = OpSystemSystem(spec=_routing_spec())
    requests = system.requested_parameters(AXES)
    assert requests["eta"].axes == ("time", "imm", "imm")
    eta = np.asarray(_table().sample(axes=AXES, request=requests["eta"]).value)
    stepper = system.bind(params={"nu": 2.0, "eta": eta})
    y = np.arange(1.0, 9.0).reshape(2, 4)
    out = np.asarray(stepper(time=np.float64(1.0), state=y.ravel())).reshape(2, 4)
    flow = 2.0 * y[0]
    expected = np.stack([-flow * eta[1].sum(1), flow @ eta[1]])
    np.testing.assert_allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"indices": [], "entries": []}, "at least one"),
        (
            {"indices": ["imm", "imm"], "entries": [{"index": ["x0"], "value": 1.0}]},
            "1 labels",
        ),
        (
            {
                "indices": ["imm", "imm"],
                "entries": [
                    {"index": ["x0", "x1"], "value": 1.0},
                    {"index": ["x0", "x1"], "value": 2.0},
                ],
            },
            "declared twice",
        ),
    ],
)
def test_sparse_table_rejects_malformed_tables(
    kwargs: dict[str, Any], message: str
) -> None:
    """Empty indices, wrong index lengths, and duplicates fail validation."""
    with pytest.raises(ValidationError, match=message):
        SparseTableParameter(**kwargs)


def test_sparse_table_rejects_bad_requests_and_labels() -> None:
    """Mismatched request axes, unknown labels, and bad entry shapes raise."""
    with pytest.raises(ValueError, match="request's axes"):
        _table().sample(
            axes=AXES, request=ParameterRequest(name="eta", axes=("vax", "imm"))
        )
    unknown = SparseTableParameter(
        indices=("imm", "imm"), entries=[{"index": ["x0", "x9"], "value": 1.0}]
    )
    with pytest.raises(ValueError, match="not a label"):
        unknown.sample(
            axes=AXES, request=ParameterRequest(name="eta", axes=("imm", "imm"))
        )
    wrong = SparseTableParameter(
        indices=("imm", "imm"),
        entries=[
            {"index": ["x0", "x1"], "value": {"module": "fixed", "value": [1.0, 2.0]}}
        ],
    )
    with pytest.raises(ValueError, match="compatible"):
        wrong.sample(
            axes=AXES, request=ParameterRequest(name="eta", axes=("time", "imm", "imm"))
        )
