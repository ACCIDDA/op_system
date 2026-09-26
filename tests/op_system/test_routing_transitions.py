"""Routing transitions: ``axis:alias`` in ``from``/``to`` (#88)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_system import Array, CompiledRhs, compile_spec, validate_spec
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


def _fanout_spec(n_imm: int, transition: dict[str, str]) -> dict[str, Any]:
    """Build a structured source-to-larger-target fanout spec.

    Returns:
        Transition spec with source ``I[age,loc]`` and target
        ``X[age,loc,imm]``.
    """
    return {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["child", "adult"]},
            {"name": "loc", "coords": ["a", "b", "c"]},
            {
                "name": "imm",
                "type": "ordinal",
                "coords": [f"x{i}" for i in range(n_imm)],
            },
        ],
        "state": ["I[age,loc]", "X[age,loc,imm]"],
        "transitions": [transition],
    }


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


@pytest.mark.parametrize("factorize_axes", [(), ("loc",)])
def test_target_only_fanout_matches_explicit_pinned_transitions(
    factorize_axes: tuple[str, ...],
) -> None:
    """One target alias equals a coordinate-pinned transition family."""
    n_imm = 5
    fanout_spec = _fanout_spec(
        n_imm,
        {
            "from": "I[age,loc]",
            "to": "X[age,loc,imm:j]",
            "rate": "reset_rate * weights[imm:j]",
        },
    )
    explicit_spec = _fanout_spec(
        n_imm,
        {
            "from": "I[age,loc]",
            "to": "X[age,loc,imm=x0]",
            "rate": "reset_rate * w0",
        },
    )
    explicit_spec["transitions"] = [
        {
            "from": "I[age,loc]",
            "to": f"X[age,loc,imm=x{index}]",
            "rate": f"reset_rate * w{index}",
        }
        for index in range(n_imm)
    ]
    if factorize_axes:
        fanout_spec["factorize_axes"] = list(factorize_axes)
        explicit_spec["factorize_axes"] = list(factorize_axes)
    fanout = compile_spec(fanout_spec)
    explicit = compile_spec(explicit_spec)

    rng = np.random.default_rng(9)
    i_state = rng.uniform(1.0, 5.0, (2, 3))
    x_state = rng.uniform(0.0, 1.0, (2, 3, n_imm))
    weights = rng.uniform(0.0, 1.0, n_imm)
    reset_rate = np.asarray(0.2)
    fanout_result = _eval(fanout)(
        0.0,
        {"I": i_state, "X": x_state},
        reset_rate=reset_rate,
        weights=weights,
    )
    explicit_result = _eval(explicit)(
        0.0,
        {"I": i_state, "X": x_state},
        reset_rate=reset_rate,
        **{f"w{index}": weight for index, weight in enumerate(weights)},
    )

    np.testing.assert_allclose(
        fanout_result["I"], explicit_result["I"], rtol=0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        fanout_result["X"], explicit_result["X"], rtol=0.0, atol=1e-13
    )
    assert abs(float(fanout_result["I"].sum() + fanout_result["X"].sum())) < 1e-12


def test_target_only_fanout_numpy_jax_jit_and_gradient_agree() -> None:
    """The lazy fanout IR remains namespace-polymorphic and differentiable."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    n_imm = 4
    compiled = compile_spec(
        _fanout_spec(
            n_imm,
            {
                "from": "I[age,loc]",
                "to": "X[age,loc,imm:j]",
                "rate": "reset_rate * weights[imm:j]",
            },
        )
    )
    evaluator = _eval(compiled)
    i_state = np.arange(1.0, 7.0).reshape(2, 3)
    weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    reset_rate = 0.25
    numpy_result = evaluator(
        np.asarray(0.0),
        {"I": i_state, "X": np.zeros((2, 3, n_imm))},
        reset_rate=np.asarray(reset_rate),
        weights=weights,
    )

    def target_flow(source: Array, target_weights: Array) -> Array:
        return evaluator(
            jnp.asarray(0.0),
            {
                "I": source,  # type: ignore[dict-item]  # namespace-polymorphic
                "X": jnp.zeros((2, 3, n_imm)),
            },
            reset_rate=jnp.asarray(reset_rate),
            weights=target_weights,
        )["X"]

    jax_result = jax.jit(target_flow)(jnp.asarray(i_state), jnp.asarray(weights))
    gradient = jax.grad(
        lambda target_weights: jnp.sum(
            target_flow(jnp.asarray(i_state), target_weights)
        )
    )(jnp.asarray(weights))

    np.testing.assert_allclose(
        np.asarray(jax_result), numpy_result["X"], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(gradient),
        np.full(n_imm, reset_rate * i_state.sum()),
        rtol=1e-6,
        atol=1e-6,
    )


def test_target_only_fanout_stays_one_lazy_transition() -> None:
    """Target-axis size does not create one normalized transition per cell."""
    spec = _fanout_spec(
        101,
        {
            "from": "I[age,loc]",
            "to": "X[age,loc,imm:j]",
            "rate": "reset_rate * weights[imm:j]",
        },
    )
    start = time.perf_counter()
    compiled = compile_spec(spec)
    elapsed = time.perf_counter() - start
    transitions = compiled.meta.get("transitions")

    assert isinstance(transitions, list)
    assert len(transitions) == 1
    assert transitions[0]["routing"]["target_only"] is True
    assert elapsed < 20.0
    report = validate_spec(spec)
    assert report.cost["routing_transitions"] == 1


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


@pytest.mark.parametrize(
    ("transition", "message"),
    [
        (
            {
                "from": "I[age,loc]",
                "to": "X[age,loc,imm:j]",
                "rate": "weights[imm]",
            },
            "must reference imm:j",
        ),
        (
            {
                "from": "I[age,loc]",
                "to": "X[age,loc,imm:j]",
                "rate": "matrix[imm:i, imm:j]",
            },
            "unexpected aliases",
        ),
        (
            {
                "from": "I[age,loc]",
                "to": "X[age,imm:j]",
                "rate": "weights[imm:j]",
            },
            "same axes apart",
        ),
        (
            {
                "from": "X[age,loc,imm]",
                "to": "X[age,loc,imm:j]",
                "rate": "weights[imm:j]",
            },
            "also appears without an alias",
        ),
        (
            {
                "from": "I[age,loc]",
                "to": "X[age,loc:k,imm:j]",
                "rate": "weights[imm:j]",
            },
            "exactly one axis:alias",
        ),
    ],
)
def test_target_only_fanout_validation_errors(
    transition: dict[str, str], message: str
) -> None:
    """Malformed target fanout fails with a structural routing error."""
    with pytest.raises(InvalidRhsSpecError, match=message):
        compile_spec(_fanout_spec(3, transition))


def test_routing_along_block_axis_is_rejected() -> None:
    """The routed axis cannot be a ``factorize_axes`` block axis."""
    spec = _spec(3, [WANE])
    spec["factorize_axes"] = ["imm"]
    report = validate_spec(spec)
    assert not report.ok
