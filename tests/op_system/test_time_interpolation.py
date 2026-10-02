"""Contracts for explicit time-indexed parameter interpolation policies."""

from __future__ import annotations

import numpy as np
import pytest

from op_system import normalize_rhs
from op_system._errors import InvalidRhsSpecError


def _spec(kind: str = "expr", *, time_axis: str = "time") -> dict[str, object]:
    """Build a scalar-rate table over three declared times.

    Returns:
        An expression or named-transition specification.
    """
    spec: dict[str, object] = {
        "kind": kind,
        "time_axis": time_axis,
        "axes": [
            {"name": "group", "coords": ["a", "b"]},
            {"name": time_axis, "type": "continuous", "coords": [0.0, 1.0, 2.0]},
        ],
        "state": ["A[group]", "B[group]"],
    }
    if kind == "expr":
        spec["equations"] = {
            "A[group]": f"-rate[{time_axis}] * A[group]",
            "B[group]": f"rate[{time_axis}] * A[group]",
        }
    else:
        spec["transitions"] = [
            {
                "name": "transfer",
                "from": "A[group]",
                "to": "B[group]",
                "rate": f"rate[{time_axis}]",
                "reactants": [{"state": "A[group]", "order": 1}],
            }
        ]
    return spec


@pytest.mark.parametrize("kind", ["expr", "transitions"])
@pytest.mark.parametrize("policy", [None, "linear", "previous"])
def test_time_policy_metadata(kind: str, policy: str | None) -> None:
    """Linear is the default; only hold-mode tables publish forcing changes."""
    spec = _spec(kind)
    if policy is not None:
        spec["time_interpolation"] = policy
    rhs = normalize_rhs(spec)
    assert rhs.meta["time_interpolation"] == (policy or "linear")
    assert rhs.meta["time_coordinates"] == (0.0, 1.0, 2.0)
    assert rhs.meta["forcing_breakpoints"] == (
        (1.0, 2.0) if policy == "previous" else ()
    )
    assert rhs.time_varying_params == (("rate", ("time",)),)


@pytest.mark.parametrize("policy", [None, True, 1, {}, [], "hold", "LINEAR", ""])
@pytest.mark.parametrize("kind", ["expr", "transitions"])
def test_invalid_time_policy_is_rejected(kind: str, policy: object) -> None:
    """The new schema field rejects unsupported policies without coercion."""
    with pytest.raises(InvalidRhsSpecError, match="time_interpolation"):
        normalize_rhs(_spec(kind) | {"time_interpolation": policy})


@pytest.mark.parametrize(
    "coords",
    [
        [0.0, np.nan, 2.0],
        [0.0, 1.0, np.inf],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [False, 1.0, 2.0],
        [],
    ],
)
@pytest.mark.parametrize("policy", ["linear", "previous"])
def test_invalid_time_coordinates_are_rejected(
    coords: list[object], policy: str
) -> None:
    """Both policies require a finite strictly increasing active time grid."""
    spec = _spec() | {"time_interpolation": policy}
    spec["axes"] = [
        {"name": "group", "coords": ["a", "b"]},
        {"name": "time", "type": "continuous", "coords": coords},
    ]
    with pytest.raises(InvalidRhsSpecError, match=r"coords|coordinates"):
        normalize_rhs(spec)


def test_custom_time_axis_coordinates_are_snapshotted() -> None:
    """Metadata uses the configured axis and survives mutation of the raw spec."""
    spec = _spec(time_axis="day") | {"time_interpolation": "previous"}
    times = [0.0, 1.0, 2.0]
    spec["axes"] = [
        {"name": "group", "coords": ["a", "b"]},
        {"name": "day", "type": "continuous", "coords": times},
    ]
    rhs = normalize_rhs(spec)
    times[1] = 99.0
    assert rhs.meta["time_axis"] == "day"
    assert rhs.meta["time_coordinates"] == (0.0, 1.0, 2.0)
    assert rhs.meta["forcing_breakpoints"] == (1.0, 2.0)


def test_unused_time_axis_does_not_publish_forcing() -> None:
    """A grid unrelated to parameters cannot create spurious forcing changes."""
    spec = _spec() | {"time_interpolation": "previous"}
    spec["equations"] = {"A[group]": "-rate * A[group]", "B[group]": "rate * A[group]"}
    rhs = normalize_rhs(spec)
    assert rhs.meta["time_coordinates"] == ()
    assert rhs.meta["forcing_breakpoints"] == ()
