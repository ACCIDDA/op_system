"""Independent chain rules and explicit natural/log coordinate contracts."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
import runpy
from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_system import StateDeclaration, array_namespace, compile_spec
from op_system._errors import InvalidRhsSpecError, UnsupportedFeatureError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


@contextmanager
def _backend(name: str) -> Iterator[Any]:
    """Scope mandatory native-backend checks to double precision.

    Yields:
        A native namespace, skipping only outside required qualification jobs.
    """
    module = np if name == "numpy" else pytest.importorskip(name)
    if name == "jax":
        enable = getattr(module, "enable_x64", None)
        if enable is None:
            enable = module.experimental.enable_x64
        with enable():
            yield import_module("jax.numpy")
    else:
        yield module


def _host(value: object) -> NDArray[np.float64]:
    """Copy an evaluated test value after native differentiation.

    Returns:
        A host array for comparison to independently calculated equations.
    """
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _decay() -> dict[str, Any]:
    """Positive decay and a signed accumulator.

    Returns:
        A mixed-coordinate model with two distinct natural observations.
    """
    return {
        "kind": "expr",
        "state": ["x", "z"],
        "state_metadata": {"x": {"scale": "log"}},
        "equations": {"x": "-a*x", "z": "a*x"},
        "observables": {"total": {"expr": "x+z"}, "x": {"expr": "x"}},
    }


def _field() -> dict[str, Any]:
    """Two-cell nonlocal linear transport, represented by named fields.

    Returns:
        Equations whose log derivatives are x_b/x_a-1 and x_a/x_b-1.
    """
    return {
        "kind": "expr",
        "axes": [{"name": "cell", "coords": ["a", "b"]}],
        "state": ["x[cell]"],
        "state_metadata": {"x": {"scale": "log"}},
        "equations": {
            "x[cell]": "apply_along(M[cell,cell:c]*x[cell:c], cell=c, kernel=sum)"
        },
        "observables": {
            "total": {"expr": "x[cell]", "reduction": {"kind": "sum", "axes": ["cell"]}}
        },
    }


def test_explicit_view_preserves_physical_model_and_known_solution() -> None:
    """A single log Euler step is exact for exponential decay, a special case."""
    compiled = compile_spec(_decay())
    physical = np.array([1.0, 0.0])
    np.testing.assert_array_equal(compiled.eval_fn(0, physical, a=2.0), [-2, 2])
    view = compiled.coordinate_view()
    encoded = _host(view.encode(physical))
    np.testing.assert_array_equal(encoded, [0, 0])
    np.testing.assert_array_equal(view.eval_fn(0, encoded, a=2.0), [-2, 2])
    updated = _host(view.decode(encoded + view.bind({"a": 2.0})(0, encoded)))
    assert updated[0] == pytest.approx(np.exp(-2))
    assert (physical + compiled.eval_fn(0, physical, a=2.0))[0] == -1
    assert view.observables["total"].eval_fn(0, encoded) == 1
    assert view.observables["x"].eval_fn(0, view.encode(np.array([2.0, 3.0]))) == 2
    assert view.observables["total"].eval_fn(0, view.encode(np.array([2.0, 3.0]))) == 5
    assert view.natural is compiled
    assert view.state_names == compiled.state_names
    assert view.param_names == ("a",)
    assert not hasattr(view, "reactions")


@pytest.mark.parametrize(
    "expression", ["-a*x", "x*(-a)", "-a*x+2*x", "-a*x/2", "-a*x**2"]
)
def test_exact_factors_cancel_before_underflow(expression: str) -> None:
    """Multiplicative decay stays defined when exp(z) underflows to zero."""
    spec = _decay()
    spec["equations"]["x"] = expression
    view = compile_spec(spec).coordinate_view()
    expected = {"-a*x": -4, "x*(-a)": -4, "-a*x+2*x": -2, "-a*x/2": -2, "-a*x**2": 0}
    assert view.eval_fn(0, np.array([-1000.0, 0.0]), a=4)[0] == expected[expression]
    assert not bool(view.decode_native(np.array([-1000.0, 0.0])).valid)


def test_additive_inflow_and_alias_use_decoded_physical_values() -> None:
    """An uncancelled inflow requires b/x; a physical alias must not use z."""
    spec = _decay()
    spec["equations"] = {"x": "b-a*x", "z": "sum_state()"}
    view = compile_spec(spec).coordinate_view()
    np.testing.assert_allclose(
        view.eval_fn(0, np.array([np.log(2), 3.0]), a=0.5, b=3), [1, 5]
    )
    spec["aliases"] = {"loss": "a*x"}
    spec["equations"]["x"] = "-loss"
    view = compile_spec(spec).coordinate_view()
    assert view.eval_fn(0, np.array([np.log(2), 3.0]), a=0.5)[0] == pytest.approx(-0.5)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_initial_domain_errors_precede_logarithms(bad: float) -> None:
    """No floor or clamp changes initial scientific values."""
    view = compile_spec(_decay()).coordinate_view()
    with np.errstate(all="raise"), pytest.raises(ValueError, match="strictly positive"):
        view.encode(np.array([bad, 0.0]))
    with np.errstate(all="ignore"):
        assert not bool(view.encode_native(np.array([bad, 0.0])).valid)


def test_representation_flags_layout_and_trajectory_axis() -> None:
    """Encoding and trajectory decoding are explicit about component layout."""
    view = compile_spec(_decay()).coordinate_view()
    values = np.array([[2.0, -3.0], [4.0, -6.0]])
    encoded = view.encode(values, state_axis=1)
    np.testing.assert_allclose(
        _host(view.decode_checked(encoded, state_axis=1)), values
    )
    with pytest.raises(TypeError, match="real floating"):
        view.encode(np.array([1, 0]))
    with pytest.raises(ValueError, match="component axis"):
        view.encode(np.ones(3))
    with np.errstate(over="ignore"):
        assert not bool(view.decode_native(np.array([1000.0, 0.0])).valid)
        with pytest.raises(ValueError, match="finite states"):
            view.decode_checked(np.array([1000.0, 0.0]))


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_native_mixed_coordinates_and_inactive_branch_derivatives(backend: str) -> None:
    """Signed identity states never enter log/exp, including inactive AD branches."""
    view = compile_spec(_decay()).coordinate_view()
    with _backend(backend) as xp:
        make = xp.tensor if backend == "torch" else xp.asarray
        state = make([2.0, -1e300], dtype=xp.float64)
        encoded = view.encode_native(state)
        assert bool(encoded.valid)
        assert array_namespace(encoded.state) is array_namespace(state)
        assert encoded.state.dtype == state.dtype
        np.testing.assert_allclose(_host(view.decode(encoded.state)), [2, -1e300])
        np.testing.assert_allclose(
            _host(view.eval_fn(0, encoded.state, a=0.5)), [-0.5, 1]
        )
        positive = make([np.log(2.0), 1e300], dtype=xp.float64)
        assert bool(view.decode_native(positive).valid)
        if backend == "jax":
            jax = import_module("jax")
            jacobian = jax.jit(
                jax.jacfwd(lambda y: view.decode(view.encode_native(y).state))
            )(state)
            np.testing.assert_allclose(jacobian, np.eye(2))
            result = jax.jit(jax.vmap(view.encode_native))(xp.stack([state, state]))
            np.testing.assert_array_equal(result.valid, [True, True])
            assert not bool(
                jax.jit(view.encode_native)(make([0.0, 1.0], dtype=xp.float64)).valid
            )
            with pytest.raises(TypeError, match="encode_native"):
                jax.jit(view.encode)(state)
        elif backend == "torch":
            state.requires_grad_()
            restored = view.decode(view.encode_native(state).state)
            xp.sum(restored).backward()
            np.testing.assert_allclose(state.grad.numpy(), [1, 1])


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_nonlocal_field_chain_rule_native_jacobian_and_observation(
    backend: str,
) -> None:
    """Known nonlocal log Jacobian is [[-2,2],[.5,-.5]] at physical [2,4]."""
    view = compile_spec(_field()).coordinate_view()
    with _backend(backend) as xp:
        make = xp.tensor if backend == "torch" else xp.asarray
        state = make(np.log([2.0, 4.0]), dtype=xp.float64)
        matrix = make([[-1.0, 1.0], [1.0, -1.0]], dtype=xp.float64)

        def rhs(y: Any) -> Any:  # ruff: ignore[any-type]
            return view.eval_fn(0, y, M=matrix)

        def observation(y: Any) -> Any:  # ruff: ignore[any-type]
            return view.observables["total"].eval_fn(0, y)

        np.testing.assert_allclose(_host(rhs(state)), [1, -0.5])
        assert float(observation(state)) == pytest.approx(6)
        assert view.pytree_eval_fn is not None
        buffers = {"x": state}
        np.testing.assert_allclose(
            _host(view.pytree_eval_fn(0, buffers, M=matrix)["x"]), [1, -0.5]
        )
        np.testing.assert_allclose(
            _host(view.decode_pytree_native(buffers).state["x"]), [2, 4]
        )
        assert bool(
            view.encode_pytree_native({"x": make([2.0, 4.0], dtype=xp.float64)}).valid
        )
        assert _host(
            view.observables["total"].pytree_eval_fn(0, buffers)
        ).item() == pytest.approx(6)
        if backend == "jax":
            jax = import_module("jax")
            jacobian = jax.jit(jax.jacfwd(rhs))(state)
            gradient = jax.jit(jax.grad(observation))(state)
            np.testing.assert_allclose(
                jax.jit(jax.vmap(rhs))(xp.stack([state, state])), [[1, -0.5]] * 2
            )
        elif backend == "torch":
            torch = import_module("torch")
            state.requires_grad_()
            jacobian = torch.autograd.functional.jacobian(rhs, state)
            gradient = torch.autograd.functional.jacobian(observation, state)
        else:
            return
        np.testing.assert_allclose(
            _host(jacobian), np.array([[-2.0, 2.0], [0.5, -0.5]])
        )
        np.testing.assert_allclose(_host(gradient), [2, 4])


def test_transition_view_keeps_original_jumps_and_natural_balance() -> None:
    """Transform deterministic drift only; SSA still sees original integer jumps."""
    spec = _decay()
    spec["kind"] = "transitions"
    del spec["equations"]
    spec["transitions"] = [{"name": "transfer", "from": "x", "to": "z", "rate": "a"}]
    natural = compile_spec(spec)
    reaction = natural.reactions[0]
    view = natural.coordinate_view()
    state = _host(view.encode(np.array([2.0, 3.0])))
    np.testing.assert_allclose(view.eval_fn(0, state, a=0.5), [-0.5, 1])
    assert len(natural.reactions) == 1
    assert natural.reactions[0] is reaction
    assert view.observables["total"].eval_fn(0, state) == pytest.approx(5)
    # Nonlinear decoding changes a discretization's balance error.
    after = _host(view.decode(state + 0.1 * view.eval_fn(0, state, a=0.5)))
    assert np.sum(after) != pytest.approx(5, abs=1e-5)


def test_identity_view_and_positional_declaration_remain_compatible() -> None:
    """No annotation means no coordinate change, including signed states."""
    spec = _decay()
    del spec["state_metadata"]
    natural = compile_spec(spec)
    view = natural.coordinate_view()
    values = np.array([-3.0, 2.0])
    np.testing.assert_array_equal(view.encode(values), values)
    np.testing.assert_array_equal(
        view.eval_fn(0, values, a=0.5), natural.eval_fn(0, values, a=0.5)
    )
    assert view.log_indices == ()
    assert StateDeclaration(None, None, None, None).scale == "identity"


@pytest.mark.parametrize("scale", [True, "ln", None, 1])
def test_unknown_or_nontext_scales_rejected(scale: object) -> None:
    """Only the documented identity and log coordinates can be declared."""
    spec = _decay()
    spec["state_metadata"]["x"]["scale"] = scale
    with pytest.raises(InvalidRhsSpecError, match="identity or log"):
        compile_spec(spec)


def test_gauge_composition_and_view_pickle() -> None:
    """Gauge representative x/K is encoded before taking its logarithm."""
    spec: dict[str, Any] = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "a*x*(1-x/K)"},
        "state_metadata": {"x": {"scale": "log"}},
        "observables": {"ratio": {"expr": "x/K"}},
        "gauges": [{"name": "units", "states": {"x": 1}, "parameters": {"K": 1}}],
    }
    natural = compile_spec(spec, gauge_fix={"K": 1})
    assert natural.gauge_fix is not None
    mapped = natural.gauge_fix.encode(np.array([6.0]), {"K": 12.0, "a": 0.5})
    view = natural.coordinate_view()
    encoded = view.encode(mapped.state)
    np.testing.assert_allclose(_host(encoded), [np.log(0.5)])
    np.testing.assert_allclose(view.eval_fn(0, encoded, **mapped.parameters), [0.25])
    assert view.observables["ratio"].eval_fn(0, encoded) == pytest.approx(0.5)
    restored = pickle.loads(pickle.dumps(view))  # ruff: ignore[suspicious-pickle-usage]
    np.testing.assert_allclose(
        restored.eval_fn(0, encoded, **mapped.parameters), [0.25]
    )
    with pytest.raises(TypeError, match="override gauge-fixed"):
        view.eval_fn(0, encoded, K=1, a=0.5)


def test_independent_blocks_keep_their_stripped_buffer_layout() -> None:
    """Log conversion applies after selecting a block, without restoring its axis."""
    spec: dict[str, Any] = {
        "kind": "expr",
        "axes": [
            {"name": "region", "coords": ["left", "right"]},
            {"name": "cell", "coords": ["a", "b"]},
        ],
        "state": ["x[region,cell]"],
        "state_metadata": {"x": {"scale": "log"}},
        "equations": {"x[region,cell]": "-rate[cell]*x[region,cell]"},
        "factorize_axes": ["region"],
    }
    natural = compile_spec(spec)
    view = natural.coordinate_view()
    assert view.block_pytree_eval_fn is not None
    rate = np.array([0.2, 0.4])
    physical = {"x": np.array([2.0, 4.0])}
    encoded = view.encode_pytree_native(physical)
    np.testing.assert_allclose(
        view.block_pytree_eval_fn(0, {"x": _host(encoded.state["x"])}, rate=rate)["x"],
        -rate,
    )
    assert natural.block_template_shapes == {"x": (2,)}
    with pytest.raises(ValueError, match="every declared state base"):
        view.decode_pytree_native({"wrong": np.ones(2)})


def test_time_tables_still_interpolate_at_the_solver_time() -> None:
    """A held rate changes at its knot without changing the coordinate contract."""
    spec: dict[str, Any] = {
        "kind": "expr",
        "axes": [
            {"name": "cell", "coords": ["a", "b"]},
            {"name": "time", "type": "continuous", "coords": [0.0, 2.0]},
        ],
        "state": ["x[cell]"],
        "state_metadata": {"x": {"scale": "log"}},
        "equations": {"x[cell]": "-rate[time,cell]*x[cell]"},
        "time_interpolation": "previous",
    }
    view = compile_spec(spec).coordinate_view()
    encoded = view.encode(np.array([2.0, 4.0]))
    table = np.array([[0.2, 0.4], [0.6, 0.8]])
    np.testing.assert_allclose(view.eval_fn(1, encoded, rate=table), [-0.2, -0.4])
    np.testing.assert_allclose(view.eval_fn(2, encoded, rate=table), [-0.6, -0.8])


@pytest.mark.parametrize("contract", ["operator", "history"])
def test_separate_operator_and_history_contracts_require_explicit_composition(
    contract: str,
) -> None:
    """Reject partial log dynamics while preserving the original compiled model."""
    spec: dict[str, Any] = {
        "kind": "expr",
        "state": ["x"],
        "state_metadata": {"x": {"scale": "log"}},
        "equations": {
            "x": "convolve_history(x, kernel=gamma, window=14, interpolation=linear)"
        },
    }
    if contract == "operator":
        spec.update({
            "axes": [{"name": "cell", "coords": ["a", "b"]}],
            "state": ["x[cell]"],
            "equations": {"x[cell]": "-a*x[cell]"},
            "operators": [{"kind": "advection", "axis": "cell", "velocity": "v"}],
        })
    natural = compile_spec(spec)
    with pytest.raises(UnsupportedFeatureError, match="Separate operator/history"):
        natural.coordinate_view()


def test_public_example_executes() -> None:
    """The guide's independently checked table runs with the baseline package."""
    runpy.run_path("examples/log_state_coordinates.py")
