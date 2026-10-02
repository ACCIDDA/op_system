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

"""Time-table policy and forcing metadata at the provider boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from flepimop2.axis import AxisCollection

from flepimop2.system.op_system import OpSystemSystem

if TYPE_CHECKING:
    from op_system import CompiledReaction, PytreeEvalFn


def _system(policy: str | None, *, singleton: bool = False) -> OpSystemSystem:
    """Compile a named transfer with a time-indexed per-group rate.

    Returns:
        A system exposing flat, PyTree, block, and reaction evaluators.
    """
    spec: dict[str, object] = {
        "kind": "transitions",
        "time_axis": "day",
        "axes": [
            {"name": "group", "coords": ["a", "b"]},
            {"name": "site", "coords": ["x"]},
            {
                "name": "day",
                "type": "continuous",
                "coords": [1.0] if singleton else [0.0, 1.0, 2.0],
            },
        ],
        "factorize_axes": ["group"],
        "state": ["A[group,site]", "B[group,site]"],
        "transitions": [
            {
                "name": "transfer",
                "from": "A[group,site]",
                "to": "B[group,site]",
                "rate": "rate[day,group]",
                "reactants": [{"state": "A[group,site]", "order": 1}],
            }
        ],
    }
    if policy is not None:
        spec["time_interpolation"] = policy
    return OpSystemSystem(spec=spec)


@pytest.mark.parametrize("policy", [None, "linear", "previous"])
def test_provider_forwards_policy_coordinates_and_parameter_axes(
    policy: str | None,
) -> None:
    """Consumers can discover forcing changes without consulting output times."""
    system = _system(policy)
    assert system.option("time_axis") == "day"
    assert system.option("time_interpolation") == (policy or "linear")
    assert system.option("time_coordinates") == (0.0, 1.0, 2.0)
    assert system.option("forcing_breakpoints") == (
        (1.0, 2.0) if policy == "previous" else ()
    )
    assert system.requested_parameters(AxisCollection())["rate"].axes == (
        "day",
        "group",
    )


@pytest.mark.parametrize(
    ("policy", "expected"), [("linear", [2.0, 4.0]), ("previous", [3.0, 6.0])]
)
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_provider_evaluators_share_time_table_policy(
    policy: str,
    expected: list[float],
    backend: str,
) -> None:
    """Flat drift, shaped drift, block drift, and propensity use the same rates."""
    system = _system(policy)
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    table = xp.asarray([[0.0, 0.0], [3.0, 6.0], [1.0, 2.0]])
    state = {"A": xp.ones((2, 1)), "B": xp.zeros((2, 1))}
    flat = system.step(np.float64(1.5), xp.asarray([1.0, 1.0, 0.0, 0.0]), rate=table)
    assert flat.__array_namespace__() is xp
    np.testing.assert_allclose(
        np.asarray(flat), [-expected[0], -expected[1], *expected]
    )
    tree_fn = cast("PytreeEvalFn", system.option("pytree_stepper_fn"))
    tree = tree_fn(1.5, state, rate=table)
    assert tree["B"].__array_namespace__() is xp
    np.testing.assert_allclose(np.asarray(tree["B"]).ravel(), expected)
    reaction = cast("tuple[CompiledReaction, ...]", system.option("reactions"))[0]
    np.testing.assert_allclose(
        np.asarray(reaction.propensity_fn(1.5, state, rate=table)).ravel(), expected
    )
    block_fn = cast("PytreeEvalFn", system.option("block_pytree_stepper_fn"))
    assert callable(block_fn)
    for group in range(2):
        block = block_fn(1.5, {"A": xp.ones(1), "B": xp.zeros(1)}, rate=table[:, group])
        assert block["B"].__array_namespace__() is xp
        np.testing.assert_allclose(np.asarray(block["B"]), expected[group])


@pytest.mark.parametrize("policy", ["linear", "previous"])
def test_provider_single_coordinate_table_has_no_forcing_changes(policy: str) -> None:
    """A constant table retains its full parameter request and endpoint value."""
    system = _system(policy, singleton=True)
    assert system.option("time_coordinates") == (1.0,)
    assert system.option("forcing_breakpoints") == ()
    assert system.requested_parameters(AxisCollection())["rate"].axes == (
        "day",
        "group",
    )
    for time in [-1.0, 1.0, 2.0]:
        result = system.step(
            np.float64(time),
            np.asarray([1.0, 1.0, 0.0, 0.0]),
            rate=np.asarray([[3.0, 6.0]]),
        )
        np.testing.assert_array_equal(result, [-3.0, -6.0, 3.0, 6.0])
