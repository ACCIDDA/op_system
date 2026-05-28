"""Tests for op_system._block_axes (analyze_block_axes + BlockAxisInfo)."""

from __future__ import annotations

import pickle  # noqa: S403

import pytest

from op_system import BlockAxisInfo, compile_rhs
from op_system._block_axes import analyze_block_axes
from op_system._errors import UnsupportedFeatureError
from op_system.specs import normalize_rhs

# ---------------------------------------------------------------------------
# Shared spec builders
# ---------------------------------------------------------------------------


def _sep_spec(
    *,
    factorize: list[str] | None = None,
    extra_axes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return a minimal separable transitions spec over age x loc.

    All equations are loc-local: the FOI only uses same-loc compartments.
    No reductions or pinned coords over loc.
    """
    axes: list[dict[str, object]] = [
        {"name": "age", "type": "categorical", "coords": ["y", "o"]},
        {"name": "loc", "type": "categorical", "coords": ["a", "b", "c"]},
    ]
    if extra_axes:
        axes.extend(extra_axes)
    spec: dict[str, object] = {
        "kind": "transitions",
        "state": ["S[age,loc]", "I[age,loc]", "R[age,loc]"],
        "transitions": [
            {
                "from": "S[age,loc]",
                "to": "I[age,loc]",
                "rate": "beta * I[age,loc]",
            },
            {"from": "I[age,loc]", "to": "R[age,loc]", "rate": "gamma"},
        ],
        "axes": axes,
    }
    if factorize is not None:
        spec["factorize_axes"] = factorize
    return spec


# ---------------------------------------------------------------------------
# BlockAxisInfo: basic construction
# ---------------------------------------------------------------------------


def test_block_axis_info_fields() -> None:
    """BlockAxisInfo stores name, size, and position dicts correctly."""
    info = BlockAxisInfo(
        name="loc",
        size=3,
        state_axis_pos={"S": 1, "I": 1},
        param_axis_pos={"rho": 0, "beta": None},
    )
    assert info.name == "loc"
    assert info.size == 3
    assert info.state_axis_pos["S"] == 1
    assert info.param_axis_pos["rho"] == 0
    assert info.param_axis_pos["beta"] is None


def test_block_axis_info_is_picklable() -> None:
    """BlockAxisInfo must survive a pickle round-trip."""
    info = BlockAxisInfo(
        name="loc",
        size=2,
        state_axis_pos={"S": 0},
        param_axis_pos={},
    )
    restored = pickle.loads(pickle.dumps(info))  # noqa: S301
    assert restored.name == info.name
    assert restored.size == info.size
    assert restored.state_axis_pos == info.state_axis_pos
    assert restored.param_axis_pos == info.param_axis_pos


# ---------------------------------------------------------------------------
# analyze_block_axes: separable spec passes
# ---------------------------------------------------------------------------


def test_analyze_no_factorize_axes_returns_empty() -> None:
    """analyze_block_axes returns () when factorize_axes is absent."""
    rhs = normalize_rhs(_sep_spec())
    assert analyze_block_axes(rhs) == ()


def test_analyze_separable_basic_passes() -> None:
    """Fully separable loc-local spec produces correct BlockAxisInfo."""
    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    infos = analyze_block_axes(rhs)
    assert len(infos) == 1
    info = infos[0]
    assert info.name == "loc"
    assert info.size == 3
    # loc is axis 1 in [age, loc] templates
    assert info.state_axis_pos == {"S": 1, "I": 1, "R": 1}
    # No shaped params in this spec
    assert info.param_axis_pos == {}


def test_analyze_separable_with_shaped_param_has_axis() -> None:
    """Shaped param that carries the block axis gets correct position."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "state": ["S[age,loc]", "I[age,loc]", "R[age,loc]"],
        "transitions": [
            {
                "from": "S[age,loc]",
                "to": "I[age,loc]",
                "rate": "beta * I[age,loc] * rho[loc]",
            },
            {"from": "I[age,loc]", "to": "R[age,loc]", "rate": "gamma"},
        ],
        "axes": [
            {"name": "age", "type": "categorical", "coords": ["y", "o"]},
            {"name": "loc", "type": "categorical", "coords": ["a", "b"]},
        ],
        "factorize_axes": ["loc"],
    }
    rhs = normalize_rhs(spec)
    infos = analyze_block_axes(rhs)
    assert len(infos) == 1
    info = infos[0]
    # rho[loc] is a shaped param with only the loc axis
    assert info.param_axis_pos.get("rho") == 0


def test_analyze_separable_with_shaped_param_no_axis() -> None:
    """Shaped param without the block axis gets None in param_axis_pos."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "state": ["S[age,loc]", "I[age,loc]", "R[age,loc]"],
        "transitions": [
            {
                "from": "S[age,loc]",
                "to": "I[age,loc]",
                "rate": "beta * I[age,loc] * K[age,age]",
            },
            {"from": "I[age,loc]", "to": "R[age,loc]", "rate": "gamma"},
        ],
        "axes": [
            {"name": "age", "type": "categorical", "coords": ["y", "o"]},
            {"name": "loc", "type": "categorical", "coords": ["a", "b"]},
        ],
        "factorize_axes": ["loc"],
    }
    rhs = normalize_rhs(spec)
    infos = analyze_block_axes(rhs)
    assert len(infos) == 1
    info = infos[0]
    # K[age, age] does not carry loc -> None (broadcast)
    assert info.param_axis_pos.get("K") is None


def test_compile_rhs_attaches_block_axes() -> None:
    """compile_rhs attaches BlockAxisInfo to CompiledRhs.block_axes."""
    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    compiled = compile_rhs(rhs)
    assert len(compiled.block_axes) == 1
    assert compiled.block_axes[0].name == "loc"
    assert compiled.block_axes[0].size == 3


def test_compile_rhs_no_factorize_axes_empty_block_axes() -> None:
    """compile_rhs leaves block_axes empty when factorize_axes is absent."""
    rhs = normalize_rhs(_sep_spec())
    compiled = compile_rhs(rhs)
    assert compiled.block_axes == ()


def test_compiled_rhs_pickle_round_trip_preserves_block_axes() -> None:
    """CompiledRhs pickles and restores with correct block_axes."""
    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    compiled = compile_rhs(rhs)
    restored = pickle.loads(pickle.dumps(compiled))  # noqa: S301
    assert len(restored.block_axes) == 1
    info = restored.block_axes[0]
    assert info.name == "loc"
    assert info.size == 3
    assert info.state_axis_pos == {"S": 1, "I": 1, "R": 1}


# ---------------------------------------------------------------------------
# analyze_block_axes: rejection cases
# ---------------------------------------------------------------------------


def test_analyze_rejects_reduce_over_block_axis() -> None:
    """Alias with apply_along that reduces over the block axis is rejected."""
    spec: dict[str, object] = {
        "kind": "expr",
        "state": ["S[age,loc]", "I[age,loc]"],
        "aliases": {
            # Nfoi sums over all age and loc cells - the inner apply_along
            # reduces over loc (the block axis), triggering rejection.
            "Nfoi": (
                "apply_along("
                "apply_along(S[age:a, loc:l], loc=l, kernel=sum), "
                "age=a, kernel=sum)"
            ),
        },
        "equations": {
            "S[age,loc]": "-beta * S[age,loc] * Nfoi",
            "I[age,loc]": "beta * S[age,loc] * Nfoi",
        },
        "axes": [
            {"name": "age", "type": "categorical", "coords": ["y", "o"]},
            {"name": "loc", "type": "categorical", "coords": ["a", "b"]},
        ],
        "factorize_axes": ["loc"],
    }
    rhs = normalize_rhs(spec)
    with pytest.raises(UnsupportedFeatureError, match="reduces over axis 'loc'"):
        analyze_block_axes(rhs)


def test_compile_rhs_rejects_reduce_over_block_axis() -> None:
    """compile_rhs propagates the UnsupportedFeatureError from analyze_block_axes."""
    spec: dict[str, object] = {
        "kind": "expr",
        "state": ["S[age,loc]", "I[age,loc]"],
        "aliases": {
            "Nfoi": (
                "apply_along("
                "apply_along(S[age:a, loc:l], loc=l, kernel=sum), "
                "age=a, kernel=sum)"
            ),
        },
        "equations": {
            "S[age,loc]": "-beta * S[age,loc] * Nfoi",
            "I[age,loc]": "beta * S[age,loc] * Nfoi",
        },
        "axes": [
            {"name": "age", "type": "categorical", "coords": ["y", "o"]},
            {"name": "loc", "type": "categorical", "coords": ["a", "b"]},
        ],
        "factorize_axes": ["loc"],
    }
    rhs = normalize_rhs(spec)
    with pytest.raises(UnsupportedFeatureError, match="reduces over axis 'loc'"):
        compile_rhs(rhs)


def test_analyze_rejects_mixed_template_missing_axis() -> None:
    """A non-scalar state template that lacks the block axis causes rejection."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "state": ["S[age,loc]", "I[age]"],
        "transitions": [
            {
                "from": "S[age,loc]",
                "to": "S[age,loc]",
                "rate": "beta * I[age]",
            },
        ],
        "axes": [
            {"name": "age", "type": "categorical", "coords": ["y", "o"]},
            {"name": "loc", "type": "categorical", "coords": ["a", "b"]},
        ],
        "factorize_axes": ["loc"],
    }
    rhs = normalize_rhs(spec)
    with pytest.raises(
        UnsupportedFeatureError,
        match="does not include block axis 'loc'",
    ):
        analyze_block_axes(rhs)


def test_analyze_rejects_operator_on_block_axis() -> None:
    """A spatial operator acting on the block axis is rejected."""
    spec: dict[str, object] = {
        "kind": "expr",
        "state": ["S[loc]"],
        "equations": {"S[loc]": "-delta * S[loc]"},
        "axes": [
            {"name": "loc", "type": "categorical", "coords": ["a", "b", "c"]},
        ],
        "operators": [{"axis": "loc", "kind": "advection", "velocity": "v"}],
        "factorize_axes": ["loc"],
    }
    rhs = normalize_rhs(spec)
    with pytest.raises(
        UnsupportedFeatureError,
        match="Operator kind='advection' acts on axis 'loc'",
    ):
        analyze_block_axes(rhs)


def test_tv_param_axis_pos_uses_runtime_position() -> None:
    """Time-varying param's block-axis position accounts for the time dimension.

    A time-varying param ``beta[time, loc]`` has shape ``(n_time, n_loc)`` at
    runtime (time prepended).  The block axis ``loc`` is therefore at index 1
    in the actual array, not index 0 in the reduced axes.
    """
    spec: dict[str, object] = {
        "kind": "transitions",
        "state": ["S[age,loc]", "I[age,loc]", "R[age,loc]"],
        "transitions": [
            {
                "from": "S[age,loc]",
                "to": "I[age,loc]",
                # beta is time-varying: declared with time axis
                "rate": "beta[time, loc] * I[age,loc]",
            },
            {"from": "I[age,loc]", "to": "R[age,loc]", "rate": "gamma"},
        ],
        "axes": [
            {"name": "time", "type": "categorical", "coords": ["t0", "t1"]},
            {"name": "age", "type": "categorical", "coords": ["y", "o"]},
            {"name": "loc", "type": "categorical", "coords": ["a", "b"]},
        ],
        "factorize_axes": ["loc"],
    }
    rhs = normalize_rhs(spec)
    infos = analyze_block_axes(rhs)
    assert len(infos) == 1
    info = infos[0]
    # beta[time, loc] → runtime shape (n_time, n_loc); loc is at index 1
    assert info.param_axis_pos["beta"] == 1


# ---------------------------------------------------------------------------
# Axis position correctness
# ---------------------------------------------------------------------------


def test_state_axis_pos_when_loc_is_first_axis() -> None:
    """state_axis_pos is 0 when loc is the first template axis."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "state": ["S[loc,age]", "I[loc,age]", "R[loc,age]"],
        "transitions": [
            {
                "from": "S[loc,age]",
                "to": "I[loc,age]",
                "rate": "beta * I[loc,age]",
            },
            {"from": "I[loc,age]", "to": "R[loc,age]", "rate": "gamma"},
        ],
        "axes": [
            {"name": "loc", "type": "categorical", "coords": ["a", "b"]},
            {"name": "age", "type": "categorical", "coords": ["y", "o"]},
        ],
        "factorize_axes": ["loc"],
    }
    rhs = normalize_rhs(spec)
    infos = analyze_block_axes(rhs)
    assert infos[0].state_axis_pos == {"S": 0, "I": 0, "R": 0}


def test_analyze_multiple_factorize_axes() -> None:
    """analyze_block_axes handles two independent factorize axes."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "state": ["S[age,loc,vax]", "I[age,loc,vax]"],
        "transitions": [
            {
                "from": "S[age,loc,vax]",
                "to": "I[age,loc,vax]",
                "rate": "beta * I[age,loc,vax]",
            },
        ],
        "axes": [
            {"name": "age", "type": "categorical", "coords": ["y", "o"]},
            {"name": "loc", "type": "categorical", "coords": ["a", "b"]},
            {"name": "vax", "type": "categorical", "coords": ["u", "v"]},
        ],
        "factorize_axes": ["loc", "vax"],
    }
    rhs = normalize_rhs(spec)
    infos = analyze_block_axes(rhs)
    assert len(infos) == 2
    loc_info = next(i for i in infos if i.name == "loc")
    vax_info = next(i for i in infos if i.name == "vax")
    assert loc_info.state_axis_pos["S"] == 1
    assert vax_info.state_axis_pos["S"] == 2


# ---------------------------------------------------------------------------
# strip_block_axis: unit tests
# ---------------------------------------------------------------------------


from op_system._normalize_block import strip_block_axis  # noqa: E402


def test_strip_block_axis_removes_axis_from_template_shapes() -> None:
    """strip_block_axis removes the block axis from all state template shapes."""
    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    stripped = strip_block_axis(rhs, "loc")
    # Each template should now have only age axis (size 2)
    for tpl in stripped.state_templates:
        assert "loc" not in tpl.axes
        assert tpl.shape == (2,)
    # Only first-loc cells remain: 3 states x 2 age coords = 6
    assert len(stripped.state_names) == 6


def test_strip_block_axis_state_names_use_ref_coord() -> None:
    """Stripped state names correspond to the first loc coordinate ('a')."""
    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    stripped = strip_block_axis(rhs, "loc")
    # All expanded names should contain '__loc_a' (first coord)
    for name in stripped.state_names:
        assert "__loc_a" in name, f"Expected '__loc_a' in {name!r}"


def test_strip_block_axis_equation_count_matches_state_names() -> None:
    """Number of equations equals number of state names after stripping."""
    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    stripped = strip_block_axis(rhs, "loc")
    assert len(stripped.equations) == len(stripped.state_names)
    assert len(stripped.equations_ir) == len(stripped.state_names)
    assert len(stripped.equations_ir_reduce) == len(stripped.state_names)


def test_strip_block_axis_strips_tv_param_axes() -> None:
    """strip_block_axis removes the block axis from time_varying_params axes."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "state": ["S[age,loc]", "I[age,loc]", "R[age,loc]"],
        "transitions": [
            {
                "from": "S[age,loc]",
                "to": "I[age,loc]",
                "rate": "beta[loc] * I[age,loc]",
            },
            {"from": "I[age,loc]", "to": "R[age,loc]", "rate": "gamma"},
        ],
        "axes": [
            {"name": "age", "type": "categorical", "coords": ["y", "o"]},
            {"name": "loc", "type": "categorical", "coords": ["a", "b", "c"]},
            {"name": "time", "type": "categorical", "coords": ["0", "1"]},
        ],
        "factorize_axes": ["loc"],
    }
    rhs = normalize_rhs(spec)
    stripped = strip_block_axis(rhs, "loc")
    tv = dict(stripped.time_varying_params)
    # beta was time_varying with (time, loc) axes; after strip should be (time,)
    if "beta" in tv:
        assert "loc" not in tv["beta"]


def test_strip_block_axis_raises_on_unknown_axis() -> None:
    """strip_block_axis raises UnsupportedFeatureError for non-factorize axis."""
    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    with pytest.raises(UnsupportedFeatureError, match=r"not in rhs\.meta"):
        strip_block_axis(rhs, "age")


def test_strip_block_axis_raises_when_axis_not_in_factorize() -> None:
    """strip_block_axis raises when factorize_axes is absent."""
    rhs = normalize_rhs(_sep_spec())  # no factorize_axes
    with pytest.raises(UnsupportedFeatureError):
        strip_block_axis(rhs, "loc")


def test_strip_block_axis_meta_no_longer_contains_axis() -> None:
    """Stripped meta no longer lists the block axis in axes or factorize_axes."""
    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    stripped = strip_block_axis(rhs, "loc")
    axes_names = [ax["name"] for ax in (stripped.meta.get("axes") or [])]
    assert "loc" not in axes_names
    assert "loc" not in (stripped.meta.get("factorize_axes") or [])


# ---------------------------------------------------------------------------
# compile_rhs: block_pytree_eval_fn integration tests
# ---------------------------------------------------------------------------


def test_compile_rhs_emits_block_pytree_eval_fn_when_block_axes_nonempty() -> None:
    """compile_rhs sets block_pytree_eval_fn when factorize_axes is declared."""
    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    compiled = compile_rhs(rhs)
    assert compiled.block_pytree_eval_fn is not None
    assert compiled.block_template_shapes is not None


def test_compile_rhs_no_factorize_axes_leaves_block_fields_none() -> None:
    """compile_rhs sets block_pytree_eval_fn=None when no factorize_axes."""
    rhs = normalize_rhs(_sep_spec())
    compiled = compile_rhs(rhs)
    assert compiled.block_pytree_eval_fn is None
    assert compiled.block_template_shapes is None


def test_block_template_shapes_have_no_block_axis() -> None:
    """block_template_shapes reflect the per-block shape (loc axis removed)."""
    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    compiled = compile_rhs(rhs)
    assert compiled.block_template_shapes is not None
    assert compiled.template_shapes is not None
    # Each template in block_template_shapes should have one fewer axis
    for base, shape in compiled.block_template_shapes.items():
        full_shape = compiled.template_shapes[base]
        assert len(shape) == len(full_shape) - 1


def test_block_pytree_eval_fn_is_callable() -> None:
    """block_pytree_eval_fn accepts a per-block state dict and returns derivatives."""
    import numpy as np  # noqa: PLC0415

    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    compiled = compile_rhs(rhs)
    assert compiled.block_pytree_eval_fn is not None
    assert compiled.block_template_shapes is not None
    # Build a per-block state dict with age axis only
    y_block = {
        base: np.ones(shape, dtype=np.float64)
        for base, shape in compiled.block_template_shapes.items()
    }
    result = compiled.block_pytree_eval_fn(0.0, y_block, beta=0.3, gamma=0.1)
    assert set(result.keys()) == set(y_block.keys())
    for base, arr in result.items():
        assert arr.shape == y_block[base].shape


def test_block_pytree_eval_fn_pickle_round_trip() -> None:
    """CompiledRhs with block_pytree_eval_fn survives a pickle round-trip."""
    import numpy as np  # noqa: PLC0415

    rhs = normalize_rhs(_sep_spec(factorize=["loc"]))
    compiled = compile_rhs(rhs)
    restored = pickle.loads(pickle.dumps(compiled))  # noqa: S301
    assert restored.block_pytree_eval_fn is not None
    assert restored.block_template_shapes == compiled.block_template_shapes
    # Functional check
    assert compiled.block_template_shapes is not None
    assert compiled.block_pytree_eval_fn is not None
    y_block = {
        base: np.ones(shape, dtype=np.float64)
        for base, shape in compiled.block_template_shapes.items()
    }
    r1 = compiled.block_pytree_eval_fn(0.0, y_block, beta=0.3, gamma=0.1)
    r2 = restored.block_pytree_eval_fn(0.0, y_block, beta=0.3, gamma=0.1)
    for base in r1:
        np.testing.assert_array_equal(r1[base], r2[base])
