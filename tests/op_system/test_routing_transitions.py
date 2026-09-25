"""Routing transitions: ``axis:alias`` in ``from``/``to`` (#88)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_system import CompiledRhs, compile_spec, validate_spec
from op_system._errors import InvalidRhsSpecError

if TYPE_CHECKING:
    from op_system.compile import PytreeEvalFn

TIME_AXIS = {
    "name": "time",
    "type": "continuous",
    "domain": {"lb": 0.0, "ub": 3.0},
    "size": 4,
}


def _spec(
    n_imm: int, transitions: list[dict[str, str]], *, time_axis: bool = False
) -> dict[str, Any]:
    axes: list[dict[str, Any]] = [
        {"name": "loc", "coords": ["CA", "NY", "TX"]},
        {"name": "vax", "coords": ["u", "v"]},
        {"name": "imm", "type": "ordinal", "coords": [f"x{i}" for i in range(n_imm)]},
    ]
    if time_axis:
        axes.append(TIME_AXIS)
    return {
        "kind": "transitions",
        "axes": axes,
        "state": ["X[loc, vax, imm]"],
        "transitions": transitions,
    }


WANE = {
    "from": "X[loc, vax, imm:i]",
    "to": "X[loc, vax, imm:j]",
    "rate": "w[loc] * G[imm:i, imm:j]",
}
VACC = {
    "from": "X[loc, vax=u, imm:i]",
    "to": "X[loc, vax=v, imm:j]",
    "rate": "nu[loc] * K[imm:i, imm:j]",
}


def _eval(compiled: CompiledRhs) -> PytreeEvalFn:
    assert compiled.pytree_eval_fn is not None
    return compiled.pytree_eval_fn


def _expected(y: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    # Self-routing drops the generator diagonal; the vax flip keeps it.
    g_off = p["G"] - np.diag(np.diag(p["G"]))
    out = p["w"][:, None, None] * (y @ g_off - y * g_off.sum(1))
    flow = p["nu"][:, None] * y[:, 0, :]
    out[:, 0] -= flow * p["K"].sum(1)
    out[:, 1] += flow @ p["K"]
    return np.asarray(out)


def _params(n: int, seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    params = {
        "G": rng.uniform(0, 1, (n, n)),
        "K": rng.uniform(0, 1, (n, n)),
        "nu": rng.uniform(0, 1, 3),
        "w": rng.uniform(0, 1, 3),
    }
    return rng.uniform(1, 2, (3, 2, n)), params


@pytest.mark.parametrize("factorize", [None, ["loc"]])
def test_routing_matches_matrix_flows_and_conserves_mass(
    factorize: list[str] | None,
) -> None:
    """Routing equals the dense matrix flows, with and without block axes."""
    spec = _spec(5, [WANE, VACC])
    if factorize:
        spec["factorize_axes"] = factorize
    compiled = compile_spec(spec)
    y, params = _params(5)
    got = np.asarray(_eval(compiled)(0.0, {"X": y}, **params)["X"])
    np.testing.assert_allclose(got, _expected(y, params), rtol=0, atol=1e-13)
    assert abs(got.sum()) < 1e-12


def test_time_varying_routing_matrix_interpolates() -> None:
    """A ``[time, imm, imm]`` routing matrix is interpolated like other rates."""
    n = 4
    spec = _spec(
        n,
        [
            {
                "from": "X[loc, vax=u, imm:i]",
                "to": "X[loc, vax=v, imm:j]",
                "rate": "nu[time, loc] * eta[time, imm:i, imm:j]",
            }
        ],
        time_axis=True,
    )
    compiled = compile_spec(spec)
    rng = np.random.default_rng(1)
    nu, eta = rng.uniform(0, 1, (4, 3)), rng.uniform(0, 1, (4, n, n))
    y = rng.uniform(1, 2, (3, 2, n))
    got = np.asarray(_eval(compiled)(np.asarray(1.5), {"X": y}, nu=nu, eta=eta)["X"])
    nu_t, eta_t = 0.5 * (nu[1] + nu[2]), 0.5 * (eta[1] + eta[2])
    flow = nu_t[:, None] * y[:, 0, :]
    expected = np.zeros_like(y)
    expected[:, 0] -= flow * eta_t.sum(1)
    expected[:, 1] += flow @ eta_t
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)
    report = validate_spec(spec)
    assert report.ok
    assert report.parameters == {"eta": ("time", "imm", "imm"), "nu": ("time", "loc")}
    assert report.cost["routing_transitions"] == 1


def test_normalization_leaves_the_input_spec_unchanged() -> None:
    """Time stripping works on a copy of the caller's transitions."""
    spec = _spec(
        3,
        [
            {
                "from": "X[loc, vax, imm]",
                "to": "X[loc, vax=v, imm]",
                "rate": "nu[time, loc]",
            }
        ],
        time_axis=True,
    )
    compile_spec(spec)
    assert spec["transitions"][0]["rate"] == "nu[time, loc]"
    assert validate_spec(spec).parameters["nu"] == ("time", "loc")


def test_routing_compiles_without_per_entry_expansion() -> None:
    """81 coordinates compile quickly and still match the matrix flows."""
    # One pinned transition per matrix entry would be 81 * 81 transitions.
    start = time.perf_counter()
    compiled = compile_spec(_spec(81, [WANE, VACC]))
    elapsed = time.perf_counter() - start
    y, params = _params(81, seed=3)
    got = np.asarray(_eval(compiled)(0.0, {"X": y}, **params)["X"])
    np.testing.assert_allclose(got, _expected(y, params), rtol=0, atol=1e-11)
    assert elapsed < 20.0


def test_routing_has_no_reaction_artifact() -> None:
    """Named routing transitions are outside the reaction-artifact scope."""
    compiled = compile_spec(_spec(3, [{**WANE, "name": "wane"}]))
    assert compiled.reactions == ()


@pytest.mark.parametrize(
    ("transition", "message"),
    [
        (
            {"from": None, "to": "X[loc, vax, imm:j]", "rate": "K[imm:j, imm:j]"},
            "requires a 'from'",
        ),
        (
            {
                "from": "X[loc, vax, imm:i]",
                "to": "X[loc, vax, imm]",
                "rate": "K[imm:i, imm]",
            },
            "exactly one axis:alias",
        ),
        (
            {
                "from": "X[loc:i, vax, imm]",
                "to": "X[loc, vax, imm:j]",
                "rate": "K[imm:i, imm:j]",
            },
            "'to' routes",
        ),
        (
            {
                "from": "X[loc, vax, imm:i]",
                "to": "X[loc, vax, imm:i]",
                "rate": "K[imm:i, imm:i]",
            },
            "must differ",
        ),
        (
            {
                "from": "X[loc, vax, imm:i]",
                "to": "X[loc, vax, imm:loc]",
                "rate": "K[imm:i, imm:loc]",
            },
            "not an axis name",
        ),
        (
            {
                "from": "X[loc, vax, imm:i]",
                "to": "X[loc, vax, imm:j]",
                "rate": "K[imm:i]",
            },
            "missing",
        ),
        (
            {
                "from": "X[loc, vax, imm:i]",
                "to": "X[loc, vax, imm:j]",
                "rate": "K[imm:i, loc:j]",
            },
            "belongs to axis",
        ),
    ],
)
def test_routing_validation_errors(transition: dict[str, Any], message: str) -> None:
    """Invalid alias usage raises with a message naming the problem."""
    with pytest.raises(InvalidRhsSpecError, match=message):
        compile_spec(_spec(3, [transition]))


def test_routing_along_block_axis_is_rejected() -> None:
    """The routed axis cannot be a ``factorize_axes`` block axis."""
    spec = _spec(3, [WANE])
    spec["factorize_axes"] = ["imm"]
    report = validate_spec(spec)
    assert not report.ok
