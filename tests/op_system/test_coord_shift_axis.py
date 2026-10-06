"""Axis-wide ``coord_shift`` aging chains (issue #238)."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_system import CompiledReaction, CompiledRhs, compile_spec
from op_system._errors import InvalidRhsSpecError
from op_system.specs import normalize_transitions_rhs

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from op_system.compile import StateDict

N_AGE = 5


def _axes(*, age_first: bool = True, n_age: int = N_AGE) -> list[dict[str, Any]]:
    age = {"name": "age", "type": "ordinal", "coords": [f"a{k}" for k in range(n_age)]}
    vax = {"name": "vax", "coords": ["u", "v"]}
    return [age, vax] if age_first else [vax, age]


def _spec(shifts: list[dict[str, Any]], *, age_first: bool = True) -> dict[str, Any]:
    axes = "age, vax" if age_first else "vax, age"
    return {
        "kind": "transitions",
        "axes": _axes(age_first=age_first),
        "state": [f"S[{axes}]", f"I[{axes}]"],
        "transitions": [
            {
                "name": "infection",
                "from": f"S[{axes}]",
                "to": f"I[{axes}]",
                "rate": f"beta * I[{axes}]",
            },
            *shifts,
        ],
    }


def _axis_wide(
    boundary: str, *, step: int = 1, name: str | None = "aging"
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "coord_shift": {
            "axis": "age",
            "step": step,
            "rate": "aging_rate[age]",
            "boundary": boundary,
        },
        "apply_to": ["S", "I"],
    }
    if name is not None:
        entry["name"] = name
    return entry


def _pairwise(*, step: int = 1) -> list[dict[str, Any]]:
    pairs = (
        [(k, k + step) for k in range(N_AGE - step)]
        if step > 0
        else [(k, k + step) for k in range(-step, N_AGE)]
    )
    return [
        {
            "coord_shift": {"age": f"a{src} -> a{dst}"},
            "rate": "aging_rate[age]",
            "apply_to": ["S", "I"],
        }
        for src, dst in pairs
    ]


def _random_inputs(
    compiled: CompiledRhs, seed: int, *, age_first: bool = True
) -> tuple[npt.NDArray[np.float64], StateDict, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, 10.0, size=len(compiled.state_names))
    shape = (N_AGE, 2) if age_first else (2, N_AGE)
    half = y.size // 2
    state = {"S": y[:half].reshape(shape), "I": y[half:].reshape(shape)}
    params = {"beta": 0.01, "aging_rate": rng.uniform(0.1, 1.0, size=N_AGE)}
    return y, state, params


@pytest.mark.parametrize("age_first", [True, False])
@pytest.mark.parametrize("step", [1, 2, -1])
def test_stay_rhs_equals_pairwise_entries(*, age_first: bool, step: int) -> None:
    """One axis-wide entry reproduces the n - |step| pairwise entries.

    Forward shifts match bit for bit. Backward pairwise entries contribute
    their terms in a different order, so those sums may differ by one ulp.
    """
    wide = compile_spec(_spec([_axis_wide("stay", step=step)], age_first=age_first))
    pairwise = compile_spec(_spec(_pairwise(step=step), age_first=age_first))
    assert wide.state_names == pairwise.state_names
    assert wide.pytree_eval_fn is not None
    assert pairwise.pytree_eval_fn is not None

    def check(actual: object, desired: object) -> None:
        actual, desired = np.asarray(actual), np.asarray(desired)
        if step > 0:
            np.testing.assert_array_equal(actual, desired)
        else:
            np.testing.assert_allclose(actual, desired, rtol=1e-14, atol=1e-13)

    for seed in range(3):
        y, state, params = _random_inputs(wide, seed, age_first=age_first)
        check(wide.eval_fn(0.0, y, **params), pairwise.eval_fn(0.0, y, **params))
        wide_tree = wide.pytree_eval_fn(0.0, state, **params)
        pairwise_tree = pairwise.pytree_eval_fn(0.0, state, **params)
        for base in ("S", "I"):
            check(wide_tree[base], pairwise_tree[base])


@pytest.mark.parametrize("step", [1, -2])
def test_absorb_removes_boundary_mass(step: int) -> None:
    """Absorbing sources lose their flux; nothing is deposited off the axis."""
    compiled = compile_spec(_spec([_axis_wide("absorb", step=step)]))
    assert compiled.pytree_eval_fn is not None
    _, state, params = _random_inputs(compiled, 7)
    params["beta"] = 0.0
    rhs = compiled.pytree_eval_fn(0.0, state, **params)
    for base in ("S", "I"):
        flow = params["aging_rate"][:, None] * state[base]
        expected = -flow
        for k in range(N_AGE):
            if 0 <= k + step < N_AGE:
                expected[k + step] += flow[k]
        np.testing.assert_allclose(rhs[base], expected, rtol=1e-14)
    boundary = [k for k in range(N_AGE) if not 0 <= k + step < N_AGE]
    lost = sum(
        (params["aging_rate"][:, None] * state[base])[boundary].sum()
        for base in ("S", "I")
    )
    total = sum(rhs[base].sum() for base in ("S", "I"))
    np.testing.assert_allclose(total, -lost, rtol=1e-12)


def test_rate_may_be_nested_or_top_level_without_mutating_spec() -> None:
    """A nested ``rate`` is hoisted onto the transition; the input is untouched."""
    nested = _spec([_axis_wide("stay")])
    original = copy.deepcopy(nested)
    top_level = _spec([_axis_wide("stay")])
    shift = top_level["transitions"][1]
    shift["rate"] = shift["coord_shift"].pop("rate")
    a, b = compile_spec(nested), compile_spec(top_level)
    assert nested == original
    y, _, params = _random_inputs(a, 3)
    np.testing.assert_array_equal(
        a.eval_fn(0.0, y, **params), b.eval_fn(0.0, y, **params)
    )


def test_one_template_transition_per_base() -> None:
    """The entry lowers once per state template, not once per coordinate pair."""
    rhs = normalize_transitions_rhs(_spec([_axis_wide("absorb")]))
    shifts = [tr for tr in rhs.meta["transitions"] if "coord_shift" in tr]
    assert [(tr["name"], tr["from"], tr["to"]) for tr in shifts] == [
        ("aging_S", "S[age, vax]", "S[age, vax]"),
        ("aging_I", "I[age, vax]", "I[age, vax]"),
    ]
    assert all(
        tr["coord_shift"] == {"axis": "age", "step": 1, "boundary": "absorb"}
        for tr in shifts
    )
    assert "aging_rate" in dict(rhs.shaped_params)
    assert not any(name.startswith("__op_system_") for name in rhs.param_names)


def test_axis_named_axis_keeps_pairwise_meaning() -> None:
    """``{axis: "p -> q"}`` still shifts between two coordinates of ``axis``."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [{"name": "axis", "coords": ["p", "q"]}],
        "state": ["X[axis]"],
        "transitions": [
            {"coord_shift": {"axis": "p -> q"}, "apply_to": ["X"], "rate": "nu"}
        ],
    }
    pairs = [
        (t["from"], t["to"])
        for t in normalize_transitions_rhs(spec).meta["transitions"]
    ]
    assert pairs == [("X__axis_p", "X__axis_q")]


def test_alias_and_state_dependent_rate_read_source_cell() -> None:
    """Free ``age`` indices in the rate, aliases included, use the source bin."""
    spec = _spec([
        {
            "coord_shift": {"axis": "age", "boundary": "stay"},
            "rate": "g[age] * (1 + S[age, vax])",
            "apply_to": ["S"],
        }
    ])
    spec["aliases"] = {"g[age]": "2 * aging_rate[age]"}
    compiled = compile_spec(spec)
    assert compiled.pytree_eval_fn is not None
    _, state, params = _random_inputs(compiled, 11)
    params["beta"] = 0.0
    flow = 2 * params["aging_rate"][:, None] * (1 + state["S"]) * state["S"]
    expected = np.zeros_like(flow)
    expected[:-1] -= flow[:-1]
    expected[1:] += flow[:-1]
    rhs = compiled.pytree_eval_fn(0.0, state, **params)
    np.testing.assert_allclose(rhs["S"], expected, rtol=1e-14)


def test_time_varying_rate() -> None:
    """A ``[time, age]`` rate is interpolated before the shift is applied."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "time_axis": "time",
        "axes": [
            {"name": "time", "type": "continuous", "coords": [0.0, 1.0]},
            _axes()[0],
        ],
        "state": ["S[age]"],
        "transitions": [
            {
                "coord_shift": {"axis": "age", "boundary": "absorb"},
                "rate": "rate[time, age]",
                "apply_to": ["S"],
            }
        ],
    }
    compiled = compile_spec(spec)
    grid = np.array([np.linspace(0.1, 0.5, N_AGE), np.linspace(0.3, 0.9, N_AGE)])
    s = np.arange(1.0, N_AGE + 1.0)
    flow = grid.mean(axis=0) * s
    expected = -flow
    expected[1:] += flow[:-1]
    np.testing.assert_allclose(compiled.eval_fn(0.5, s, rate=grid), expected)


def test_block_axis_survives_shift_on_other_axis() -> None:
    """Shifting ``age`` keeps ``vax`` separable for block-axis vmap."""
    spec = _spec([_axis_wide("absorb")], age_first=False)
    spec["factorize_axes"] = ["vax"]
    compiled = compile_spec(spec)
    assert [info.name for info in compiled.block_axes] == ["vax"]


def test_jax_jit_matches_numpy() -> None:
    """The lowered shift traces under JAX ``jit``."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    compiled = compile_spec(_spec([_axis_wide("stay")]))
    y, _, params = _random_inputs(compiled, 5)
    jitted = jax.jit(
        lambda yy, rate: compiled.eval_fn(0.0, yy, beta=0.01, aging_rate=rate)
    )
    np.testing.assert_allclose(
        np.asarray(jitted(jnp.asarray(y), jnp.asarray(params["aging_rate"]))),
        compiled.eval_fn(0.0, y, **params),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("shift", "match"),
    [
        ({"axis": "age"}, r"moves the last 1 coordinate.*absorb.*stay"),
        ({"axis": "age", "step": -2, "boundary": "error"}, r"first 2 coordinate"),
        ({"axis": "age", "boundary": "wrap"}, "boundary must be one of"),
        ({"axis": "age", "step": 0, "boundary": "stay"}, "nonzero integer"),
        ({"axis": "age", "step": True, "boundary": "stay"}, "nonzero integer"),
        ({"axis": "age", "step": 1.0, "boundary": "stay"}, "nonzero integer"),
        ({"axis": "age", "step": N_AGE, "boundary": "stay"}, "smaller in magnitude"),
        ({"axis": "age", "boundary": "stay", "rat": "x"}, r"unknown fields.*rat"),
        ({"axis": "missing", "boundary": "stay"}, "not defined"),
    ],
)
def test_invalid_shift_raises(shift: dict[str, Any], match: str) -> None:
    """Malformed axis-wide shifts fail during normalization."""
    entry = {"coord_shift": shift, "rate": "aging_rate[age]", "apply_to": ["S"]}
    with pytest.raises(InvalidRhsSpecError, match=match):
        normalize_transitions_rhs(_spec([entry]))


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"rate": "other"}, "rate both in coord_shift and on the transition"),
        ({"apply_to": []}, "apply_to"),
        ({"reactants": []}, "does not accept 'reactants'"),
    ],
)
def test_invalid_entry_raises(changes: dict[str, Any], match: str) -> None:
    """Entry-level fields are validated alongside the shift itself."""
    entry = {**_axis_wide("stay"), **changes}
    with pytest.raises(InvalidRhsSpecError, match=match):
        normalize_transitions_rhs(_spec([entry]))


def test_missing_rate_raises() -> None:
    """A rate is required somewhere."""
    entry = _axis_wide("stay")
    del entry["coord_shift"]["rate"]
    with pytest.raises(InvalidRhsSpecError, match="requires a 'rate'"):
        normalize_transitions_rhs(_spec([entry]))


def test_state_without_shift_axis_raises() -> None:
    """Every ``apply_to`` state must carry the shifted axis as a wildcard."""
    spec = _spec([{**_axis_wide("stay"), "apply_to": ["S", "R"]}])
    spec["state"].append("R[vax]")
    with pytest.raises(InvalidRhsSpecError, match=r"'R' needs one state template"):
        normalize_transitions_rhs(spec)


# ---------------------------------------------------------------------------
# Templated reaction artifacts
# ---------------------------------------------------------------------------


def _fire(
    state: StateDict, reaction: CompiledReaction, index: tuple[int, ...]
) -> StateDict:
    """Apply one firing using only the published ``CompiledReaction`` contract.

    Returns:
        A new state mapping with the firing applied.
    """
    assert reaction.from_base is not None
    out = {base: value.copy() for base, value in state.items()}
    channel = dict(zip(reaction.from_axes, index, strict=True))
    source = {**channel, **dict(reaction.from_pinned)}
    out[reaction.from_base][tuple(source[ax] for ax in reaction.full_axes)] -= 1
    offsets = dict(reaction.offsets)
    target = {ax: channel[ax] for ax in reaction.to_axes}
    target.update(dict(reaction.pinned))
    target.update({ax: channel[ax] + step for ax, step in offsets.items()})
    size = dict(zip(reaction.full_axes, out[reaction.to_base].shape, strict=True))
    if all(0 <= target[ax] < size[ax] for ax in reaction.full_axes):
        out[reaction.to_base][tuple(target[ax] for ax in reaction.full_axes)] += 1
    return out


@pytest.mark.parametrize("boundary", ["absorb", "stay"])
@pytest.mark.parametrize("step", [1, -2])
def test_offset_reaction_metadata(boundary: str, step: int) -> None:
    """One templated reaction per state; the shifted axis is an offset."""
    compiled = compile_spec(_spec([_axis_wide(boundary, step=step)]))
    by_name = {r.name: r for r in compiled.reactions}
    assert set(by_name) == {"infection", "aging_S", "aging_I"}
    for base in ("S", "I"):
        reaction = by_name[f"aging_{base}"]
        assert reaction.from_base == reaction.to_base == base
        assert reaction.from_axes == reaction.full_axes == ("age", "vax")
        assert reaction.to_axes == ("vax",)
        assert reaction.sum_axes == ()
        assert reaction.pinned == reaction.from_pinned == ()
        assert reaction.offsets == (("age", step),)
        # The aging rate reads no state, so the shifted source is complete.
        assert reaction.reactants_complete is True
    assert by_name["infection"].offsets == ()


@pytest.mark.parametrize("boundary", ["absorb", "stay"])
@pytest.mark.parametrize("step", [1, -2])
def test_single_firing_moves_one_unit(boundary: str, step: int) -> None:
    """A firing at age k decrements (S, k) and increments (S, k + step)."""
    compiled = compile_spec(_spec([_axis_wide(boundary, step=step)]))
    reaction = next(r for r in compiled.reactions if r.name == "aging_S")
    state = {"S": np.full((N_AGE, 2), 3.0), "I": np.zeros((N_AGE, 2))}
    rates = np.linspace(0.2, 1.0, N_AGE)
    propensity = np.asarray(reaction.propensity_fn(0.0, state, aging_rate=rates))
    fires = np.array([
        boundary == "absorb" or 0 <= k + step < N_AGE for k in range(N_AGE)
    ])
    expected_propensity = np.where(fires, 3 * rates, 0.0)[:, None]
    np.testing.assert_allclose(
        propensity, np.broadcast_to(expected_propensity, propensity.shape)
    )
    for k in range(N_AGE):
        if propensity[k, 1] == 0:
            continue
        after = _fire(state, reaction, (k, 1))["S"]
        delta = after - state["S"]
        expected = np.zeros_like(delta)
        expected[k, 1] = -1
        if 0 <= k + step < N_AGE:
            expected[k + step, 1] = 1
        np.testing.assert_array_equal(delta, expected)


@pytest.mark.parametrize("boundary", ["absorb", "stay"])
@pytest.mark.parametrize("step", [1, 2, -1])
def test_propensities_reconstruct_rhs(boundary: str, step: int) -> None:
    """Propensity-weighted offset stoichiometry equals the deterministic RHS."""
    compiled = compile_spec(_spec([_axis_wide(boundary, step=step)]))
    assert compiled.pytree_eval_fn is not None
    _, state, params = _random_inputs(compiled, 13)
    drift = {base: np.zeros_like(value) for base, value in state.items()}
    zero = {base: np.zeros_like(value) for base, value in state.items()}
    for reaction in compiled.reactions:
        propensity = np.asarray(reaction.propensity_fn(0.0, state, **params))
        for index in np.ndindex(propensity.shape):
            for base, change in _fire(zero, reaction, index).items():
                drift[base] += propensity[index] * change
    rhs = compiled.pytree_eval_fn(0.0, state, **params)
    for base, value in drift.items():
        np.testing.assert_allclose(value, rhs[base], rtol=1e-12, atol=1e-14)


def test_unnamed_shift_publishes_no_reaction() -> None:
    """Like other transitions, only named entries publish reactions."""
    compiled = compile_spec(_spec([_axis_wide("absorb", name=None)]))
    assert [r.name for r in compiled.reactions] == ["infection"]


def _ssa_exit_time(
    reaction: CompiledReaction,
    rates: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> float:
    """Run one individual from the first bin until it leaves the axis.

    Returns:
        The absorption time.
    """
    state = {"S": np.zeros(rates.size)}
    state["S"][0] = 1.0
    t = 0.0
    while state["S"].sum():
        propensity = np.asarray(reaction.propensity_fn(t, state, aging_rate=rates))
        total = propensity.sum()
        t += rng.exponential(1.0 / total)
        index = int(rng.choice(rates.size, p=propensity / total))
        state = _fire(state, reaction, (index,))
    return t


def _ks_statistic(
    samples: npt.NDArray[np.float64], cdf: Callable[[float], float]
) -> float:
    """Return the two-sided Kolmogorov-Smirnov distance to ``cdf``.

    Returns:
        ``sup |F_n - F|`` over the samples.
    """
    ordered = np.sort(samples)
    expected = np.array([cdf(float(x)) for x in ordered])
    n = ordered.size
    return float(
        max(
            np.max(np.arange(1, n + 1) / n - expected),
            np.max(expected - np.arange(n) / n),
        )
    )


def test_pure_aging_ssa_reproduces_erlang_exit_time() -> None:
    """An SSA over the artifact exits an n-bin chain after Erlang(n, r) time."""
    n_bins, rate, runs = 4, 2.0, 4000
    spec: dict[str, object] = {
        "kind": "transitions",
        "axes": [_axes(n_age=n_bins)[0]],
        "state": ["S[age]"],
        "transitions": [
            {
                "name": "aging",
                "coord_shift": {"axis": "age", "boundary": "absorb"},
                "rate": "aging_rate[age]",
                "apply_to": ["S"],
            }
        ],
    }
    (reaction,) = compile_spec(spec).reactions
    rng = np.random.default_rng(238)
    rates = np.full(n_bins, rate)
    exits = np.array([_ssa_exit_time(reaction, rates, rng) for _ in range(runs)])

    def erlang_cdf(x: float) -> float:
        return 1.0 - sum(
            math.exp(-rate * x) * (rate * x) ** k / math.factorial(k)
            for k in range(n_bins)
        )

    assert _ks_statistic(exits, erlang_cdf) < 1.36 / math.sqrt(runs)
    np.testing.assert_allclose(exits.mean(), n_bins / rate, rtol=0.05)
    np.testing.assert_allclose(exits.var(), n_bins / rate**2, rtol=0.1)
