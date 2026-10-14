"""Independent fixed-mesh observation values and native derivative contracts."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from array_api_compat import array_namespace  # type: ignore[import-untyped]

from op_system import (
    ObservableDefinition,
    ObservableReduction,
    compile_rhs,
    compile_spec,
    normalize_expr_rhs,
    normalize_transitions_rhs,
)
from op_system._errors import InvalidRhsSpecError, UnsupportedFeatureError

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from op_system import CompiledObservable


def _spec(kind: str = "expr") -> dict[str, object]:
    spec: dict[str, object] = {
        "kind": kind,
        "axes": [
            {
                "name": "x",
                "type": "continuous",
                "coords": [0.05, 0.25, 0.7],
                "measure": {"kind": "cell", "weights": [0.1, 0.3, 0.6], "units": "m"},
            }
        ],
        "state": ["rho[x]", "mass[x]", "counter"],
        "state_metadata": {
            "rho": {
                "representation": "density",
                "measure_axes": ["x"],
                "units": "items/m",
            },
            "mass": {"representation": "cell_mass", "measure_axes": []},
            "counter": {"representation": "cumulative"},
        },
        "parameters": {"gain": {"support": "positive"}},
        "observables": {
            "total": {
                "expr": "gain*rho[x]",
                "reduction": {"kind": "integral", "axes": ["x"]},
                "units": "items",
            },
            "mass_total": {
                "expr": "mass[x]",
                "reduction": {"kind": "sum", "axes": ["x"]},
            },
            "counter_value": {"expr": "counter"},
            "point": {"expr": "rho[0.25]"},
            "field": {"expr": "gain*rho[x]+t", "axes": ["x"]},
            "constant_field": {"expr": "gain", "axes": ["x"]},
        },
    }
    if kind == "expr":
        spec["equations"] = {"rho[x]": "0", "mass[x]": "0", "counter": "-k*counter"}
    else:
        spec["transitions"] = [{"from": "counter", "to": "counter", "rate": "k"}]
    return spec


def _state() -> np.ndarray:
    return np.array([1.1, 1.5, 2.4, 0.11, 0.45, 1.44, 9.0])


@pytest.mark.parametrize("kind", ["expr", "transitions"])
def test_numpy_independent_totals_samples_and_counter(kind: str) -> None:
    """Cell averages of 1+2*x integrate to 2; mass sums and counters need no weights."""
    compiled = compile_spec(_spec(kind))
    y = _state()
    before = y.copy()
    assert compiled.param_names == ("k",)
    assert "gain" in compiled.param_meta
    # Observable-only gain is not needed to evaluate the dynamics.
    compiled.eval_fn(0.0, y, k=0.5)
    outputs = {
        name: obs.eval_fn(3.0, y, gain=2.0)
        for name, obs in compiled.observables.items()
    }
    assert outputs["total"] == pytest.approx(4.0)
    assert outputs["mass_total"] == pytest.approx(2.0)
    assert outputs["counter_value"] == pytest.approx(9.0)
    assert outputs["point"] == pytest.approx(1.5)
    np.testing.assert_allclose(np.asarray(outputs["field"]), [5.2, 6.0, 7.8])
    np.testing.assert_array_equal(outputs["constant_field"], [2, 2, 2])
    np.testing.assert_array_equal(y, before)
    for name, value in outputs.items():
        assert array_namespace(value) is array_namespace(y)
        assert getattr(value, "dtype", None) == y.dtype
        assert getattr(value, "shape", None) == compiled.observables[name].output_shape


def test_two_nonuniform_meshes_have_same_physical_definition() -> None:
    """Refinement preserves the analytic physical quantity."""
    totals = []
    for centers, widths in [
        ([0.25, 0.75], [0.5, 0.5]),
        ([0.25, 0.55, 0.8], [0.5, 0.1, 0.4]),
    ]:
        spec = _spec()
        spec["axes"] = [
            {
                "name": "x",
                "type": "continuous",
                "coords": centers,
                "measure": {"kind": "cell", "weights": widths},
            }
        ]
        density = 1 + 2 * np.array(centers)
        y = np.concatenate((density, density * widths, [0.0]))
        compiled = compile_spec(spec)
        totals.append(compiled.observables["total"].eval_fn(0.0, y, gain=3.0))
        assert compiled.observables["mass_total"].eval_fn(0.0, y) == pytest.approx(2.0)
    assert totals == pytest.approx([6.0, 6.0])


def test_numpy_metadata_is_detached_readonly_and_pickle_rebuilt() -> None:
    """Detached metadata survives trusted model round trips."""
    spec = _spec()
    compiled = compile_spec(spec)
    obs = compiled.observables["field"]
    assert obs.definition == ObservableDefinition("field", "gain*rho[x]+t", ("x",))
    assert obs.coordinates == (("x", ("0.05", "0.25", "0.7")),)
    assert obs.output_shape == (3,)
    assert obs.parameter_shapes == (("gain", ()),)
    assert obs.state_dependencies == ("rho",)
    assert obs.time_dependent
    assert compiled.observables["total"].definition.reduction == ObservableReduction(
        "integral", ("x",)
    )
    assert compiled.observables["total"].measure_requirements[0][1].weights == (
        0.1,
        0.3,
        0.6,
    )
    with pytest.raises(TypeError):
        cast("MutableMapping[str, CompiledObservable]", compiled.observables)["new"] = (
            obs
        )
    raw = cast("dict[str, dict[str, object]]", spec["observables"])
    raw["field"]["expr"] = "0"
    restored = pickle.loads(pickle.dumps(compiled))  # ruff: ignore[suspicious-pickle-usage]
    assert restored.observables["field"].definition == obs.definition
    np.testing.assert_allclose(
        restored.observables["field"].eval_fn(1.0, _state(), gain=2.0), [3.2, 4, 5.8]
    )


@pytest.mark.parametrize("normalizer", [normalize_expr_rhs, normalize_transitions_rhs])
def test_direct_normalization_exports_observables(normalizer: object) -> None:
    """Both direct normalization entry points preserve observation-only metadata."""
    kind = "expr" if normalizer is normalize_expr_rhs else "transitions"
    rhs = normalizer(_spec(kind))  # type: ignore[operator]
    assert len(rhs.observables_ir) == 6
    assert "gain" in compile_rhs(rhs).param_meta


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        [],
        {},
        {"expr": ""},
        {"expr": "rho[x]", "extra": 1},
        {"expr": "rho[x]", "axes": "x"},
        {"expr": "rho[x]", "axes": ["x", "x"]},
        {"expr": "rho[x]", "axes": ["missing"]},
        {"expr": "rho[x]", "units": None},
        {"expr": "rho[x]", "reduction": None},
        {"expr": "rho[x]", "reduction": {"kind": [], "axes": ["x"]}},
        {"expr": "rho[x]", "reduction": {"kind": "sum", "axes": []}},
        {"expr": "rho[x]", "axes": ["x"], "reduction": {"kind": "sum", "axes": ["x"]}},
    ],
)
def test_invalid_observable_schema(declaration: object) -> None:
    """Malformed declarations fail before a partial compiled artifact can escape."""
    spec = _spec()
    spec["observables"] = {"bad": declaration}
    with pytest.raises(InvalidRhsSpecError):
        compile_spec(spec)


@pytest.mark.parametrize(
    "expression",
    [
        "rho",
        "rho[missing]",
        "rho[x]+rho[0.9]",
        "gain[x]+gain",
        "k[x]",
        "rho_buf+rho[x]",
    ],
)
def test_invalid_observable_references(expression: str) -> None:
    """Invalid buffers, coordinates and parameter layouts fail."""
    spec = _spec()
    spec["observables"] = {"bad": {"expr": expression, "axes": ["x"]}}
    with pytest.raises(InvalidRhsSpecError):
        compile_spec(spec)


@pytest.mark.parametrize(
    "expression",
    [
        "sum_state()",
        "integrate_over(rho[x],x=x)",
        "history(counter)",
        "np.log(counter)",
        "np.exp(counter,counter)",
        "True",
        "'text'",
    ],
)
def test_unsupported_observables_fail_explicitly(expression: str) -> None:
    """Unsupported operations fail before artifact construction."""
    spec = _spec()
    spec["observables"] = {"bad": {"expr": expression}}
    with pytest.raises(UnsupportedFeatureError):
        compile_spec(spec)


@pytest.mark.parametrize("state", ["mass", "counter"])
def test_numpy_extensive_states_cannot_be_weighted_twice(state: str) -> None:
    """Physical integrals refuse extensive cell masses and accumulated counters."""
    spec = _spec()
    expression = "mass[x]" if state == "mass" else "counter"
    spec["observables"] = {
        "bad": {"expr": expression, "reduction": {"kind": "integral", "axes": ["x"]}}
    }
    with pytest.raises(InvalidRhsSpecError, match="intensive"):
        compile_spec(spec)


def test_numpy_integrals_require_declared_intensive_measures() -> None:
    """Legacy center quadrature cannot replace a physical measure."""
    spec = _spec()
    del spec["state_metadata"]
    with pytest.raises(InvalidRhsSpecError, match="intensive"):
        compile_spec(spec)
    spec = _spec()
    spec["state_metadata"] = {"rho": {"representation": "density"}}
    axes = cast("list[dict[str, object]]", spec["axes"])
    del axes[0]["measure"]
    with pytest.raises(InvalidRhsSpecError, match="explicit measure"):
        compile_spec(spec)


def test_numpy_shaped_parameters_axis_reordering_and_pytree() -> None:
    """Output ordering and reductions match an independent matrix oracle."""
    spec = _spec()
    axes = cast("list[dict[str, object]]", spec["axes"])
    axes.append({
        "name": "region",
        "coords": ["left", "right"],
        "measure": {"kind": "counting"},
    })
    spec["state"] = ["rho[region,x]"]
    spec["equations"] = {"rho[region,x]": "0"}
    spec["state_metadata"] = {
        "rho": {"representation": "density", "measure_axes": ["x"]}
    }
    spec["factorize_axes"] = ["region"]
    spec["observables"] = {
        "profile": {
            "expr": "gain[x]*rho[region,x]",
            "axes": ["region"],
            "reduction": {"kind": "integral", "axes": ["x"]},
        },
        "transposed": {"expr": "rho[region,x]", "axes": ["x", "region"]},
        "selected": {"expr": "rho[region:left,x]", "axes": ["x"]},
    }
    compiled = compile_spec(spec)
    state = np.array([[1.1, 1.5, 2.4], [2.2, 3, 4.8]])
    gain = np.array([1.0, 2.0, 3.0])
    expected = [0.11 + 0.9 + 4.32, 0.22 + 1.8 + 8.64]
    np.testing.assert_allclose(
        np.asarray(
            compiled.observables["profile"].eval_fn(0.0, state.ravel(), gain=gain)
        ),
        expected,
    )
    np.testing.assert_allclose(
        np.asarray(
            compiled.observables["profile"].pytree_eval_fn(
                0.0, {"rho": state}, gain=gain
            )
        ),
        expected,
    )
    np.testing.assert_array_equal(
        compiled.observables["transposed"].eval_fn(0.0, state.ravel()), state.T
    )
    np.testing.assert_array_equal(
        compiled.observables["selected"].eval_fn(0.0, state.ravel()), state[0]
    )
    assert compiled.observables["profile"].parameter_shapes == (("gain", (3,)),)
    # Global observations do not change dynamics-only block independence.
    assert compiled.block_axes[0].name == "region"
    assert compiled.block_pytree_eval_fn is not None


def test_numpy_runtime_shape_dtype_and_missing_parameters() -> None:
    """Input errors are reported at the observation boundary."""
    obs = compile_spec(_spec()).observables["total"]
    with pytest.raises(ValueError, match="flat state shape"):
        obs.eval_fn(0.0, np.zeros(3), gain=1.0)
    with pytest.raises(ValueError, match="requires parameter"):
        obs.eval_fn(0.0, _state())
    with pytest.raises(ValueError, match="must have shape"):
        obs.eval_fn(0.0, _state(), gain=np.ones(2))
    with pytest.raises(TypeError, match="real floating"):
        obs.eval_fn(0.0, np.ones(7, dtype=int), gain=1.0)
    with pytest.raises(ValueError, match="all compiled state bases"):
        obs.pytree_eval_fn(0.0, {"rho": np.ones(3)}, gain=1.0)


def _gradient_spec() -> dict[str, object]:
    spec = _spec()
    spec["observables"] = {
        "smooth": {
            "expr": "(+rho[x])**2/gain-(-rho[x])+gain*np.exp(rho[x])+t",
            "reduction": {"kind": "integral", "axes": ["x"]},
        }
    }
    return spec


def test_numpy_arithmetic_exp_and_finite_difference_reference() -> None:
    """Native values and finite differences match analytic weighted derivatives."""
    obs = compile_spec(_gradient_spec()).observables["smooth"]
    y = _state()
    u = y[:3]
    weights = np.array([0.1, 0.3, 0.6])
    value = obs.eval_fn(3.0, y, gain=2.0)
    expected = float(np.dot(weights, u**2 / 2 + u + 2 * np.exp(u) + 3))
    assert value == pytest.approx(expected)
    step = 1e-5
    derivative = (
        float(np.asarray(obs.eval_fn(3.0, y, gain=2.0 + step)))
        - float(np.asarray(obs.eval_fn(3.0, y, gain=2.0 - step)))
    ) / (2 * step)
    assert derivative == pytest.approx(float(np.dot(weights, -(u**2) / 4 + np.exp(u))))


def test_numpy_separable_partial_integral_and_point_quadrature() -> None:
    """Partial integration applies only the requested physical measures."""
    spec: dict[str, object] = {
        "axes": [
            {
                "name": "x",
                "type": "continuous",
                "coords": [0, 1, 3],
                "measure": {"kind": "quadrature", "weights": [0.5, 1.5, 1]},
            },
            {
                "name": "y",
                "type": "continuous",
                "coords": [0.125, 0.625],
                "measure": {"kind": "cell", "weights": [0.25, 0.75]},
            },
        ],
        "state": ["rho[x,y]"],
        "equations": {"rho[x,y]": "0"},
        "state_metadata": {
            "rho": {"representation": "density", "measure_axes": ["x", "y"]}
        },
        "observables": {
            "marginal": {
                "expr": "rho[x,y]",
                "axes": ["y"],
                "reduction": {"kind": "integral", "axes": ["x"]},
            },
            "total": {
                "expr": "rho[x,y]",
                "reduction": {"kind": "integral", "axes": ["x", "y"]},
            },
        },
    }
    compiled = compile_spec(spec)
    # rho(x,y)=x*(1+2*y): exact x-integral 4.5 and y-integral of the factor 2.
    state = np.outer([0.0, 1.0, 3.0], [1.25, 2.25]).ravel()
    np.testing.assert_allclose(
        np.asarray(compiled.observables["marginal"].eval_fn(0.0, state)),
        [5.625, 10.125],
    )
    assert compiled.observables["total"].eval_fn(0.0, state) == pytest.approx(9.0)
    assert tuple(
        axis for axis, _ in compiled.observables["marginal"].measure_requirements
    ) == ("x",)


def test_torch_foreign_array_parameters_are_rejected() -> None:
    """Cross-namespace arrays cannot silently sever a differentiable parameter path."""
    torch = pytest.importorskip("torch")
    obs = compile_spec(_spec()).observables["total"]
    y = torch.tensor(_state(), dtype=torch.float64)
    with pytest.raises(TypeError, match="state array namespace"):
        obs.eval_fn(0.0, y, gain=np.array(2.0))


def test_jax_jit_vmap_and_state_parameter_time_gradients() -> None:
    """Initial operations and weighted sums match independent derivatives."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)  # ruff: ignore[boolean-positional-value-in-call] -- JAX config API
    obs = compile_spec(_gradient_spec()).observables["smooth"]
    y = jnp.asarray(_state())
    gain = jnp.asarray(2.0)
    time = jnp.asarray(3.0)
    result = jax.jit(obs.eval_fn)(time, y, gain=gain)
    u = _state()[:3]
    w = np.array([0.1, 0.3, 0.6])
    assert float(result) == pytest.approx(
        float(np.dot(w, u**2 / 2 + u + 2 * np.exp(u) + 3))
    )
    gy, gp, gt = jax.grad(
        lambda yy, gg, tt: obs.eval_fn(tt, yy, gain=gg), argnums=(0, 1, 2)
    )(y, gain, time)
    np.testing.assert_allclose(
        gy, np.r_[w * (u + 1 + 2 * np.exp(u)), np.zeros(4)], rtol=1e-12
    )
    assert float(gp) == pytest.approx(float(np.dot(w, -(u**2) / 4 + np.exp(u))))
    assert float(gt) == pytest.approx(1.0)
    batched = jax.jit(jax.vmap(lambda yy: obs.eval_fn(time, yy, gain=gain)))(
        jnp.stack([y, y])
    )
    np.testing.assert_allclose(batched, [result, result])
    assert array_namespace(result) is array_namespace(y)
    assert getattr(result, "dtype", None) == y.dtype


@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test_torch_state_parameter_time_gradients_and_dtype(dtype_name: str) -> None:
    """Native arithmetic, casts and reductions preserve Torch autograd."""
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    obs = compile_spec(_gradient_spec()).observables["smooth"]
    y = torch.tensor(_state(), dtype=dtype, requires_grad=True)
    gain = torch.tensor(2.0, dtype=dtype, requires_grad=True)
    time = torch.tensor(3.0, dtype=dtype, requires_grad=True)
    result = obs.eval_fn(time, y, gain=gain)
    gy, gp, gt = torch.autograd.grad(result, (y, gain, time))
    u = _state()[:3]
    w = np.array([0.1, 0.3, 0.6])
    tolerance = 2e-6 if dtype_name == "float32" else 1e-12
    np.testing.assert_allclose(
        gy.detach().numpy(),
        np.r_[w * (u + 1 + 2 * np.exp(u)), np.zeros(4)],
        rtol=tolerance,
    )
    assert float(gp) == pytest.approx(
        float(np.dot(w, -(u**2) / 4 + np.exp(u))), rel=tolerance
    )
    assert float(gt) == pytest.approx(1.0, rel=tolerance)
    assert float(torch.Tensor.detach(result)) == pytest.approx(
        float(np.dot(w, u**2 / 2 + u + 2 * np.exp(u) + 3)), rel=tolerance
    )
    assert getattr(result, "dtype", None) == y.dtype
    assert array_namespace(result) is array_namespace(y)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_point_counter_and_shaped_parameter_gradients(backend: str) -> None:
    """Coordinate/counter selection and shaped fields retain exact gradients."""
    module = pytest.importorskip(backend)
    spec = _spec()
    spec["observables"] = {
        "field_total": {
            "expr": "gain[x]*rho[x]",
            "reduction": {"kind": "integral", "axes": ["x"]},
        },
        "point": {"expr": "rho[0.25]"},
        "counter_value": {"expr": "counter"},
    }
    c = compile_spec(spec)
    if backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        module.config.update("jax_enable_x64", True)  # ruff: ignore[boolean-positional-value-in-call] -- JAX config API
        y = jnp.asarray(_state())
        gain = jnp.asarray([1.0, 2.0, 3.0])
        gy, gp = module.grad(
            lambda yy, gg: c.observables["field_total"].eval_fn(0.0, yy, gain=gg),
            argnums=(0, 1),
        )(y, gain)
        point_grad = module.grad(lambda yy: c.observables["point"].eval_fn(0.0, yy))(y)
        counter_grad = module.grad(
            lambda yy: c.observables["counter_value"].eval_fn(0.0, yy)
        )(y)
        actual_counter = np.asarray(counter_grad)
        actual_y, actual_p, actual_point = (
            np.asarray(gy),
            np.asarray(gp),
            np.asarray(point_grad),
        )
    else:
        y = module.tensor(_state(), dtype=module.float64, requires_grad=True)
        gain = module.tensor([1.0, 2.0, 3.0], dtype=module.float64, requires_grad=True)
        gy, gp = module.autograd.grad(
            c.observables["field_total"].eval_fn(0.0, y, gain=gain), (y, gain)
        )
        point_grad = module.autograd.grad(c.observables["point"].eval_fn(0.0, y), y)[0]
        counter_grad = module.autograd.grad(
            c.observables["counter_value"].eval_fn(0.0, y), y
        )[0]
        actual_counter = counter_grad.detach().numpy()
        actual_y, actual_p, actual_point = (
            gy.detach().numpy(),
            gp.detach().numpy(),
            point_grad.detach().numpy(),
        )
    np.testing.assert_allclose(actual_y, [0.1, 0.6, 1.8, 0, 0, 0, 0])
    np.testing.assert_array_equal(actual_counter, [0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(actual_p, [0.11, 0.45, 1.44])
    np.testing.assert_array_equal(actual_point, [0, 1, 0, 0, 0, 0, 0])
