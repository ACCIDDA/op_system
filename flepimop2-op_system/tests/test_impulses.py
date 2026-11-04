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

"""The provider publishes explicit maps and refuses engines that ignore them."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from flepimop2.axis import AxisCollection

from flepimop2.system.op_system import OpSystemSystem

if TYPE_CHECKING:
    from op_system import CompiledImpulse
    from op_system.observables import ObservableEvalFn


def _system() -> OpSystemSystem:
    return OpSystemSystem(
        spec={
            "kind": "expr",
            "state": ["x[unit]"],
            "axes": [{"name": "unit", "coords": ["a", "b"]}],
            "equations": {"x[unit]": "-k*x[unit]"},
            "impulses": [
                {
                    "time": 1.0,
                    "updates": [
                        {
                            "target": "x[unit=a]",
                            "operation": "add",
                            "expr": "boost[unit]",
                        },
                    ],
                }
            ],
        }
    )


def test_provider_forwards_unbound_maps_and_event_parameters() -> None:
    """An engine can discover the full schedule and bind map-only shaped values."""
    system = _system()
    assert system.option("impulse_times") == (1.0,)
    requests = system.requested_parameters(AxisCollection())
    assert requests["boost"].axes == ("unit",)
    assert requests["k"].axes == ()
    events = cast("tuple[CompiledImpulse, ...]", system.option("impulses"))
    state = np.array([2.0, 3.0])
    np.testing.assert_array_equal(
        events[0].eval_fn(1.0, state, boost=np.array([5.0, 8.0])), [7.0, 3.0]
    )
    rhs = cast("ObservableEvalFn", system.option("impulse_rhs"))
    np.testing.assert_array_equal(rhs(0.0, state, k=0.5), [-1.0, -1.5])


def test_provider_requires_explicit_engine_schedule_support() -> None:
    """Ordinary and bound steppers fail instead of silently dropping state maps."""
    system = _system()
    with pytest.raises(NotImplementedError, match="impulse-aware engine"):
        system.step(np.float64(0.0), np.ones(2), k=0.5)
    bound = system.bind({"k": 0.5})
    with pytest.raises(NotImplementedError, match="impulse-aware engine"):
        bound(np.float64(0.0), np.ones(2))
    assert system.option("pytree_stepper_fn") is None
    assert system.option("block_pytree_stepper_fn") is None
