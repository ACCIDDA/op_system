"""Contracts for explicit time-indexed parameter interpolation policies."""

from __future__ import annotations

import numpy as np
import pytest

from op_system import compile_spec, normalize_rhs
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


def test_nonnumeric_active_time_labels_fail_during_normalization() -> None:
    """A metadata-only model cannot postpone an unusable table until compile."""
    spec = _spec()
    spec["axes"] = [
        {"name": "group", "coords": ["a", "b"]},
        {"name": "time", "coords": ["t0", "t1"]},
    ]
    with pytest.raises(InvalidRhsSpecError, match="finite real times"):
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


@pytest.mark.parametrize("kind", ["expr", "transitions"])
@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("linear", [0.0, 0.0, 1.5, 3.0, 2.0, 1.0, 1.0]),
        ("previous", [0.0, 0.0, 0.0, 3.0, 3.0, 1.0, 1.0]),
    ],
)
def test_scalar_table_is_right_continuous_and_clamped(
    kind: str,
    policy: str,
    expected: list[float],
) -> None:
    """Flat, PyTree, and reaction paths share the chosen table semantics."""
    compiled = compile_spec(_spec(kind) | {"time_interpolation": policy})
    assert compiled.pytree_eval_fn is not None
    y = {"A": np.ones(2), "B": np.zeros(2)}
    flat = np.asarray([1.0, 1.0, 0.0, 0.0])
    params = {"rate": np.asarray([0.0, 3.0, 1.0])}
    for time, rate in zip([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0], expected, strict=True):
        np.testing.assert_allclose(
            compiled.eval_fn(time, flat, **params), [-rate, -rate, rate, rate]
        )
        drift = compiled.pytree_eval_fn(time, y, **params)
        np.testing.assert_allclose(drift["A"], [-rate, -rate])
        np.testing.assert_allclose(drift["B"], [rate, rate])
        if kind == "transitions":
            assert len(compiled.reactions) == 1
            np.testing.assert_allclose(
                np.asarray(compiled.reactions[0].propensity_fn(time, y, **params)),
                [rate, rate],
            )


@pytest.mark.parametrize("kind", ["expr", "transitions"])
@pytest.mark.parametrize("policy", ["linear", "previous"])
def test_single_coordinate_time_table_is_constant(kind: str, policy: str) -> None:
    """A one-entry table is constant at every time and has no forcing changes."""
    spec = _spec(kind, time_axis="day") | {"time_interpolation": policy}
    spec["axes"] = [
        {"name": "group", "coords": ["a", "b"]},
        {"name": "day", "type": "continuous", "coords": [1.0]},
    ]
    compiled = compile_spec(spec)
    assert compiled.meta["time_coordinates"] == (1.0,)
    assert compiled.meta["forcing_breakpoints"] == ()
    assert compiled.pytree_eval_fn is not None
    for time in [-1.0, 1.0, 2.0]:
        np.testing.assert_allclose(
            compiled.eval_fn(
                time, np.asarray([1.0, 1.0, 0.0, 0.0]), rate=np.asarray([3.0])
            ),
            [-3.0, -3.0, 3.0, 3.0],
        )
        out = compiled.pytree_eval_fn(
            time, {"A": np.ones(2), "B": np.zeros(2)}, rate=np.asarray([3.0])
        )
        np.testing.assert_array_equal(out["B"], [3.0, 3.0])
        if kind == "transitions":
            np.testing.assert_array_equal(
                compiled.reactions[0].propensity_fn(
                    time, {"A": np.ones(2), "B": np.zeros(2)}, rate=np.asarray([3.0])
                ),
                [3.0, 3.0],
            )


def test_singleton_spatial_axis_still_requires_integration_grid() -> None:
    """The time-table exception does not relax spatial quadrature validation."""
    spec: dict[str, object] = {
        "kind": "expr",
        "axes": [{"name": "x", "type": "continuous", "coords": [1.0]}],
        "state": ["A[x]"],
        "equations": {"A[x]": "-A[x]"},
    }
    with pytest.raises(InvalidRhsSpecError, match=">=2 coords"):
        normalize_rhs(spec)


def _shaped_spec(kind: str, axes: tuple[str, ...], policy: str) -> dict[str, object]:
    """Create a rate tensor whose time axis occupies any of three positions.

    Returns:
        A shaped expression or transition specification.
    """
    spec = _spec(kind) | {"time_interpolation": policy}
    spec["axes"] = [
        {"name": "group", "coords": ["a", "b"]},
        {"name": "site", "coords": ["x", "y"]},
        {"name": "time", "type": "continuous", "coords": [0.0, 1.0, 2.0]},
    ]
    spec["state"] = ["A[group,site]", "B[group,site]"]
    rate = f"rate[{','.join(axes)}]"
    if kind == "expr":
        spec["equations"] = {
            "A[group,site]": f"-{rate} * A[group,site]",
            "B[group,site]": f"{rate} * A[group,site]",
        }
    else:
        spec["transitions"] = [
            {
                "name": "transfer",
                "from": "A[group,site]",
                "to": "B[group,site]",
                "rate": rate,
                "reactants": [{"state": "A[group,site]", "order": 1}],
            }
        ]
    return spec


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("kind", ["expr", "transitions"])
@pytest.mark.parametrize("policy", ["linear", "previous"])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_shaped_tables_support_every_time_axis_position(
    backend: str,
    kind: str,
    policy: str,
    position: int,
) -> None:
    """Shaped rates retain the state namespace across all compiled paths."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    axes = ["group", "site"]
    axes.insert(position, "time")
    compiled = compile_spec(_shaped_spec(kind, tuple(axes), policy))
    assert compiled.pytree_eval_fn is not None
    scale = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    table = np.asarray([0.0, 3.0, 1.0])[:, None, None] * scale
    grid = xp.asarray(np.moveaxis(table, 0, position))
    state = {"A": xp.ones((2, 2)), "B": xp.zeros((2, 2))}
    flat = xp.asarray([1.0] * 4 + [0.0] * 4)
    rate = 3.0 if policy == "previous" else 2.0
    expected = rate * scale

    out = compiled.eval_fn(1.5, flat, rate=grid)
    assert out.__array_namespace__() is xp
    np.testing.assert_allclose(
        np.asarray(out), np.concatenate([-expected.ravel(), expected.ravel()])
    )
    drift = compiled.pytree_eval_fn(1.5, state, rate=grid)
    assert drift["B"].__array_namespace__() is xp
    np.testing.assert_allclose(np.asarray(drift["B"]), expected)
    if kind == "transitions":
        propensity = compiled.reactions[0].propensity_fn(1.5, state, rate=grid)
        assert hasattr(propensity, "__array_namespace__")
        assert propensity.__array_namespace__() is xp
        np.testing.assert_allclose(np.asarray(propensity), expected)


@pytest.mark.parametrize("policy", ["linear", "previous"])
def test_jax_tables_remain_trace_pure_under_jit_and_vmap(policy: str) -> None:
    """Runtime evaluation times and table values remain dynamic while tracing."""
    jax = pytest.importorskip("jax")
    xp = pytest.importorskip("jax.numpy")
    compiled = compile_spec(_spec("transitions") | {"time_interpolation": policy})
    assert compiled.pytree_eval_fn is not None
    times = xp.asarray([0.5, 1.0, 1.5])
    grid = xp.asarray([0.0, 3.0, 1.0])
    state = {"A": xp.ones(2), "B": xp.zeros(2)}
    flat = xp.asarray([1.0, 1.0, 0.0, 0.0])
    rates = np.asarray([0.0, 3.0, 3.0] if policy == "previous" else [1.5, 3.0, 2.0])

    flat_fn = jax.jit(jax.vmap(compiled.eval_fn, in_axes=(0, None)))
    flat_out = flat_fn(times, flat, rate=xp.broadcast_to(grid, (3, 3)))
    np.testing.assert_allclose(
        np.asarray(flat_out)[:, 2:], np.repeat(rates[:, None], 2, axis=1)
    )
    tree_fn = jax.jit(compiled.pytree_eval_fn)
    np.testing.assert_allclose(
        np.asarray(tree_fn(times[2], state, rate=grid)["B"]), [rates[2]] * 2
    )
    propensity_fn = jax.jit(compiled.reactions[0].propensity_fn)
    np.testing.assert_allclose(
        np.asarray(propensity_fn(times[2], state, rate=grid)), [rates[2]] * 2
    )
    np.testing.assert_allclose(
        np.asarray(propensity_fn(times[2], state, rate=2 * grid)), [2 * rates[2]] * 2
    )


@pytest.mark.parametrize("policy", ["linear", "previous"])
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_block_evaluation_matches_whole_tensor_rates(policy: str, backend: str) -> None:
    """Block stripping preserves the time policy and adjusts the time position."""
    compiled = compile_spec(
        _shaped_spec("expr", ("site", "time", "group"), policy)
        | {"factorize_axes": ["site"]}
    )
    assert compiled.pytree_eval_fn is not None
    assert compiled.block_pytree_eval_fn is not None
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    state = {"A": xp.ones((2, 2)), "B": xp.zeros((2, 2))}
    grid = xp.asarray([
        [[0.0, 0.0], [3.0, 6.0], [1.0, 2.0]],
        [[0.0, 0.0], [9.0, 12.0], [3.0, 4.0]],
    ])
    full = compiled.pytree_eval_fn(1.5, state, rate=grid)
    for site in range(2):
        block_state = {name: value[:, site] for name, value in state.items()}
        block = compiled.block_pytree_eval_fn(1.5, block_state, rate=grid[site])
        assert block["B"].__array_namespace__() is xp
        np.testing.assert_allclose(
            np.asarray(block["A"]), np.asarray(full["A"])[:, site]
        )
        np.testing.assert_allclose(
            np.asarray(block["B"]), np.asarray(full["B"])[:, site]
        )
    if backend == "jax":
        jax = pytest.importorskip("jax")
        block_fn = jax.jit(
            jax.vmap(compiled.block_pytree_eval_fn, in_axes=(None, 1), out_axes=1)
        )
        mapped = block_fn(1.5, state, rate=grid)
        np.testing.assert_allclose(np.asarray(mapped["B"]), np.asarray(full["B"]))
