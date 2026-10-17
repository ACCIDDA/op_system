"""Literal numeric ``initial_state`` entries (constant initial values, #303)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_system import compile_spec
from op_system._errors import InvalidRhsSpecError
from op_system.specs import normalize_transitions_rhs

if TYPE_CHECKING:
    from collections.abc import Mapping


def _spec(initial_state: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "transitions",
        "axes": [
            {"name": "loc", "coords": ["a", "b"]},
            {"name": "age", "coords": ["y", "o"]},
        ],
        "state": ["E[loc]", "C[loc,age]", "D"],
        "transitions": [
            {"from": "E[loc]", "to": "D", "rate": "mu"},
            {"to": "C[loc,age]", "rate": "sigma * E[loc]"},
        ],
        "initial_state": initial_state,
    }


def test_literal_scalar_state() -> None:
    """A scalar state accepts a literal; ints become floats."""
    spec = _spec({"D": 0, "E[loc]": "E0[loc]"})
    ic = normalize_transitions_rhs(spec).meta["initial_state"]
    assert ic["D"] == pytest.approx(0.0)
    assert isinstance(ic["D"], float)
    assert ic["E__loc_a"] == "E0__loc_a"


def test_literal_broadcasts_over_state_axes() -> None:
    """A literal on a templated key reaches every selected cell."""
    ic = normalize_transitions_rhs(_spec({"C[loc,age]": 2.5})).meta["initial_state"]
    assert ic == {
        "C__loc_a__age_y": 2.5,
        "C__loc_a__age_o": 2.5,
        "C__loc_b__age_y": 2.5,
        "C__loc_b__age_o": 2.5,
    }


def test_literal_with_pinned_coordinate_and_mixed_entries() -> None:
    """Pinned keys select a subset; other entry forms are unchanged."""
    ic = normalize_transitions_rhs(
        _spec({
            "C[loc=a,age]": 1.0,
            "E[loc]": {"shaped": "E0", "axes": ["loc"]},
            "D": "D0",
        })
    ).meta["initial_state"]
    assert ic["C__loc_a__age_y"] == pytest.approx(1.0)
    assert ic["C__loc_a__age_o"] == pytest.approx(1.0)
    assert "C__loc_b__age_y" not in ic
    assert ic["E__loc_b"] == {"shaped": "E0", "axes": ("loc",), "coords": {"loc": "b"}}
    assert ic["D"] == "D0"


def test_existing_entries_unchanged() -> None:
    """String and shaped entries normalize exactly as before."""
    ic = normalize_transitions_rhs(
        _spec({"C[loc,age]": "C0[loc,age]", "D": "D0"})
    ).meta["initial_state"]
    assert ic["C__loc_b__age_o"] == "C0__loc_b__age_o"
    assert ic["D"] == "D0"


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -float("inf"), True, False]
)
def test_invalid_literals_rejected(bad: object) -> None:
    """Non-finite numbers and booleans are rejected, naming the entry."""
    with pytest.raises(InvalidRhsSpecError, match=r"initial_state\['D'\]"):
        normalize_transitions_rhs(_spec({"D": bad}))


def test_fixed_mapping_still_rejected() -> None:
    """Only the literal spelling is added; ``{fixed: ...}`` stays invalid."""
    with pytest.raises(InvalidRhsSpecError, match="unknown keys"):
        normalize_transitions_rhs(_spec({"D": {"fixed": 0.0}}))
    with pytest.raises(InvalidRhsSpecError, match="unknown keys"):
        normalize_transitions_rhs(_spec({"D": {"value": 0.0}}))


_COORDS = {"loc": ["a", "b"], "age": ["y", "o"]}


def _y0(initial_state: dict[str, Any], params: Mapping[str, Any]) -> np.ndarray:
    compiled = compile_spec(_spec(initial_state))
    ic = compiled.meta["initial_state"]
    values = []
    for name in compiled.state_names:
        entry = ic[name]
        if isinstance(entry, float):
            values.append(entry)
        elif isinstance(entry, str):
            values.append(float(params[entry]))
        else:
            raw = np.asarray(params[entry["shaped"]])
            idx = tuple(_COORDS[ax].index(entry["coords"][ax]) for ax in entry["axes"])
            values.append(float(raw[idx]))
    return np.asarray(values, dtype=np.float64)


def _integrate(initial_state: dict[str, Any], seeds: Mapping[str, Any]) -> np.ndarray:
    compiled = compile_spec(_spec(initial_state))
    rhs_params = {"mu": 0.3, "sigma": 0.2}
    y = _y0(initial_state, seeds)
    dt = 0.01

    def f(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(compiled.eval_fn(t, y, **rhs_params), dtype=np.float64)

    for k in range(500):
        t = k * dt
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt / 2 * k1)
        k3 = f(t + dt / 2, y + dt / 2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def test_literal_end_to_end_matches_zero_table_workaround() -> None:
    """Compile and integrate: literal zero equals the zero-table seed."""
    seeds = {"E0": np.asarray([10.0, 4.0])}
    literal = _integrate(
        {"E[loc]": {"shaped": "E0", "axes": ["loc"]}, "C[loc,age]": 0.0, "D": 0.0},
        seeds,
    )
    table = _integrate(
        {
            "E[loc]": {"shaped": "E0", "axes": ["loc"]},
            "C[loc,age]": {"shaped": "C0", "axes": ["loc", "age"]},
            "D": "D0",
        },
        {**seeds, "C0": np.zeros((2, 2)), "D0": 0.0},
    )
    np.testing.assert_array_equal(literal, table)
    # C accumulates sigma * integral of E = sigma * E0 * (1 - exp(-mu t)) / mu.
    expected = 0.2 * np.asarray([10.0, 4.0]) * (1 - np.exp(-0.3 * 5.0)) / 0.3
    got = literal[2:6].reshape(2, 2)
    np.testing.assert_allclose(got[:, 0], expected, rtol=1e-8)
    np.testing.assert_allclose(got[:, 1], expected, rtol=1e-8)
