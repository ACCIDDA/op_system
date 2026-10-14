"""Required compiler qualification against independent hierarchical equations."""

from __future__ import annotations

from contextlib import contextmanager
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarks.hierarchical_model import (
    TEMPLATES,
    fixture,
    handwritten_rhs,
    model_spec,
    reference_rhs,
)
from op_system import array_namespace, compile_spec
from op_system._errors import UnsupportedFeatureError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from op_system import CompiledRhs

CASES = [
    (False, "flat"),
    (False, "structured"),
    (False, "block"),
    (True, "flat"),
    (True, "structured"),
]
BACKENDS = ("numpy", "jax", "torch")


def _total_spec(layout: str, *, alias: bool) -> dict[str, Any]:
    suffix = "" if layout == "scalar" else "[group,age]"
    population = "N" if alias else "sum_state()"
    spec: dict[str, Any] = {
        "kind": "expr",
        "state": [name + suffix for name in ["S", "I", "R"]],
        "equations": {
            "S" + suffix: f"-beta * S{suffix} * I{suffix} / {population}",
            "I" + suffix: (
                f"beta * S{suffix} * I{suffix} / {population} - gamma * I{suffix}"
            ),
            "R" + suffix: f"gamma * I{suffix}",
        },
    }
    if layout != "scalar":
        spec["axes"] = [
            {"name": "group", "coords": ["a", "b"]},
            {"name": "age", "type": "continuous", "coords": [0.0, 0.5, 2.0]},
        ]
    if layout == "mixed":
        spec["state"].append("Z")
        spec["equations"]["Z"] = "Z - Z"
    if alias:
        spec["aliases"] = {"N": "sum_state()"}
    return spec


def _total_reference(
    state: dict[str, Any], beta: float, gamma: float
) -> dict[str, np.ndarray]:
    values = {name: _host(value) for name, value in state.items() if name != "extra"}
    total = sum(float(value) for array in values.values() for value in array.ravel())
    flow = beta * values["S"] * values["I"] / total
    recovery = gamma * values["I"]
    result = {"S": -flow, "I": flow - recovery, "R": recovery}
    if "Z" in values:
        result["Z"] = np.zeros_like(values["Z"])
    return result


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("layout", ["scalar", "shaped", "mixed"])
@pytest.mark.parametrize("alias", [False, True])
def test_global_state_total_layouts(
    backend: str, dtype: str, layout: str, *, alias: bool
) -> None:
    """Global totals include every declared state exactly once and remain native."""
    compiled = compile_spec(_total_spec(layout, alias=alias))
    assert compiled.pytree_eval_fn is not None
    raw = {
        name: np.asarray(value, dtype=dtype)
        for name, value in {"S": 9.0, "I": 1.0, "R": 2.0}.items()
    }
    if layout != "scalar":
        raw = {
            name: value + np.arange(6, dtype=dtype).reshape(2, 3) / 10
            for name, value in raw.items()
        }
    if layout == "mixed":
        raw["Z"] = np.asarray(3.0, dtype=dtype)
    with _precision(backend):
        module = (
            np
            if backend == "numpy"
            else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
        )
        state = {name: module.asarray(value) for name, value in raw.items()}
        xp = array_namespace(state["S"])
        params = {
            "beta": module.asarray(np.asarray(0.4, dtype=dtype)),
            "gamma": module.asarray(np.asarray(0.1, dtype=dtype)),
        }
        operation = compiled.pytree_eval_fn
        if backend == "jax":
            operation = pytest.importorskip("jax").jit(operation)
        expected = _total_reference(state, 0.4, 0.1)
        result = operation(
            0.0,
            {**state, "extra": module.asarray(np.asarray(1000.0, dtype=dtype))},
            **params,
        )
        tolerance = 3e-6 if dtype == "float32" else 1e-12
        for name, value in result.items():
            assert value.shape == raw[name].shape
            assert value.dtype == state[name].dtype
            assert array_namespace(value) is xp
            np.testing.assert_allclose(_host(value), expected[name], rtol=tolerance)
        flat = compiled.eval_fn(
            0.0,
            xp.concat([xp.reshape(value, (-1,)) for value in state.values()]),
            **params,
        )
        np.testing.assert_allclose(
            _host(flat),
            np.concatenate([value.ravel() for value in expected.values()]),
            rtol=tolerance,
        )
        altered = {**state, "R": state["R"] * 2}
        refreshed = operation(0.0, altered, **params)
        for name, value in _total_reference(altered, 0.4, 0.1).items():
            np.testing.assert_allclose(_host(refreshed[name]), value, rtol=tolerance)


@pytest.mark.parametrize("backend", BACKENDS)
def test_global_total_prevents_block_factorization(backend: str) -> None:
    """A cached total remains a cross-group dependency rather than a local rate."""
    pytest.importorskip(backend)
    spec = _total_spec("shaped", alias=True)
    spec["factorize_axes"] = ["group"]
    with pytest.raises(UnsupportedFeatureError, match=r"sum_state.*group"):
        compile_spec(spec)


def _reaction_total_reference(
    state: dict[str, np.ndarray], gain: float, total: float
) -> float:
    flow = gain * state["S"].astype(np.float64) * state["I"] / total
    recovery = 0.1 * state["I"].astype(np.float64)
    return float(np.sum(flow**2) + np.sum(recovery**2))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_global_total_reaction_values_and_parameter_derivatives(
    backend: str, dtype: str
) -> None:
    """Exported propensities retain native global reductions and their AD graph."""
    spec = _total_spec("mixed", alias=True)
    spec["kind"] = "transitions"
    del spec["equations"]
    spec["transitions"] = [
        {
            "name": "inf",
            "from": "S[group,age]",
            "to": "I[group,age]",
            "rate": "beta * I[group,age] / N",
        },
        {"name": "rec", "from": "I[group,age]", "to": "R[group,age]", "rate": "gamma"},
    ]
    compiled = compile_spec(spec)
    assert compiled.reaction_gaps == ()
    assert len(compiled.reactions) == 2
    raw = {
        name: np.full((2, 3), value, dtype=dtype)
        for name, value in {"S": 9.0, "I": 1.0, "R": 2.0}.items()
    }
    raw["Z"] = np.asarray(3.0, dtype=dtype)
    total = sum(float(value) for array in raw.values() for value in array.ravel())
    with _precision(backend):
        module = (
            np
            if backend == "numpy"
            else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
        )
        state = {name: module.asarray(value) for name, value in raw.items()}
        beta = module.asarray(np.asarray(0.4, dtype=dtype))
        gamma = module.asarray(np.asarray(0.1, dtype=dtype))
        xp = array_namespace(state["S"])
        tolerance = 3e-6 if dtype == "float32" else 1e-12
        for reaction, expected in zip(
            compiled.reactions,
            [0.4 * raw["S"] * raw["I"] / total, 0.1 * raw["I"]],
            strict=True,
        ):
            operation = reaction.propensity_fn
            if backend == "jax":
                operation = pytest.importorskip("jax").jit(operation)
            actual = operation(0.0, state, beta=beta, gamma=gamma)
            assert array_namespace(actual) is xp
            assert getattr(actual, "dtype", None) == state["S"].dtype
            np.testing.assert_allclose(_host(actual), expected, rtol=tolerance)

        def objective(gain: Any) -> Any:  # ruff: ignore[any-type]
            return sum(
                xp.sum(
                    xp.square(
                        reaction.propensity_fn(0.0, state, beta=gain, gamma=gamma)
                    )
                )
                for reaction in compiled.reactions
            )

        if backend == "numpy":
            return
        if backend == "jax":
            derivative = float(pytest.importorskip("jax").grad(objective)(beta))
        else:
            gain = beta.detach().clone().requires_grad_()
            derivative = float(module.autograd.grad(objective(gain), gain)[0].detach())
        step = 2e-3 if dtype == "float32" else 1e-5
        np.testing.assert_allclose(
            derivative,
            (
                _reaction_total_reference(raw, 0.4 + step, total)
                - _reaction_total_reference(raw, 0.4 - step, total)
            )
            / (2 * step),
            rtol=3e-3 if dtype == "float32" else 1e-7,
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("constant", ["0", "1.5"])
@pytest.mark.parametrize("layout", ["scalar", "shaped", "alias"])
def test_literal_rhs_preserves_state_precision(
    backend: str, dtype: str, constant: str, layout: str
) -> None:
    """Untyped literals inherit the floating state precision in every layout."""
    spec: dict[str, Any] = {
        "kind": "expr",
        "state": ["X"],
        "equations": {"X": constant},
    }
    if layout != "scalar":
        spec.update({
            "axes": [{"name": "group", "coords": ["a", "b"]}],
            "state": ["X[group]"],
            "equations": {"X[group]": constant},
            "factorize_axes": ["group"],
        })
    if layout == "alias":
        spec["aliases"] = {"value[group]": constant}
        spec["equations"]["X[group]"] = "value[group]"
    compiled = compile_spec(spec)
    assert compiled.pytree_eval_fn is not None
    with _precision(backend):
        module = (
            np
            if backend == "numpy"
            else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
        )
        raw = np.asarray(2.0 if layout == "scalar" else [1.0, 2.0], dtype=dtype)
        state = {"X": module.asarray(raw)}
        xp = array_namespace(state["X"])
        operation = compiled.pytree_eval_fn
        if backend == "jax":
            operation = pytest.importorskip("jax").jit(operation)
        result = operation(0.0, state)["X"]
        flat = compiled.eval_fn(0.0, xp.reshape(state["X"], (-1,)))
        for actual in [result, flat]:
            assert actual.dtype == state["X"].dtype
            assert array_namespace(actual) is xp
            np.testing.assert_allclose(_host(actual), float(constant))
        assert result.shape == raw.shape
        if layout != "scalar":
            assert compiled.block_pytree_eval_fn is not None
            block = compiled.block_pytree_eval_fn
            if backend == "jax":
                block = pytest.importorskip("jax").jit(block)
            for index in range(2):
                actual = block(0.0, {"X": state["X"][index]})["X"]
                assert actual.dtype == state["X"].dtype
                assert actual.shape == ()
                np.testing.assert_allclose(_host(actual), float(constant))


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_constant_rhs_has_zero_state_jacobian(backend: str) -> None:
    """Native AD transformations retain the zero derivative of literal forcing."""
    module = pytest.importorskip(backend)
    compiled = compile_spec({
        "kind": "expr",
        "axes": [{"name": "group", "coords": ["a", "b"]}],
        "state": ["X[group]"],
        "equations": {"X[group]": "1.5"},
    })
    assert compiled.pytree_eval_fn is not None
    with _precision(backend):
        xp = pytest.importorskip("jax.numpy") if backend == "jax" else module
        state = xp.asarray(np.array([1.0, 2.0], dtype="float64"))

        def operation(values: Any) -> Any:  # ruff: ignore[any-type]
            assert compiled.pytree_eval_fn is not None
            return compiled.pytree_eval_fn(0.0, {"X": values})["X"]

        jacobian = (
            module.jacfwd(operation)(state)
            if backend == "jax"
            else module.func.jacrev(operation)(state)
        )
        assert jacobian.dtype == state.dtype
        np.testing.assert_allclose(_host(jacobian), np.zeros((2, 2)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_mixed_alias_reductions_preserve_all_state_axes(
    backend: str, dtype: str
) -> None:
    """Wildcard group selection and nested bound coordinates retain every cell."""
    spec: dict[str, Any] = {
        "kind": "expr",
        "axes": [
            {"name": axis, "coords": [f"{axis}0", f"{axis}1"]}
            for axis in ["loc", "age", "vax", "imm"]
        ],
        "factorize_axes": ["loc"],
        "state": ["X[loc,age,vax,imm]", "I1[loc,age,vax]", "E[loc,age,vax]"],
        "aliases": {
            "foi[loc]": (
                "apply_along(apply_along(I1[loc,age:ap,vax:v] + "
                "apply_along(X[loc,age:ap,vax:v,imm:k], imm=k, kernel=sum), "
                "vax=v, kernel=sum), age=ap, kernel=sum)"
            )
        },
        "equations": {
            "X[loc,age,vax,imm]": "-foi[loc] * X[loc,age,vax,imm]",
            "I1[loc,age,vax]": "0 * I1[loc,age,vax]",
            "E[loc,age,vax]": (
                "foi[loc] * apply_along(X[loc,age,vax,imm:k], imm=k, kernel=sum)"
            ),
        },
    }
    compiled = compile_spec(spec)
    assert compiled.param_names == ()
    assert compiled.pytree_eval_fn is not None
    assert compiled.block_pytree_eval_fn is not None
    raw = {
        "X": np.linspace(0.1, 1.6, 16, dtype=dtype).reshape(2, 2, 2, 2),
        "I1": np.linspace(0.3, 2.4, 8, dtype=dtype).reshape(2, 2, 2),
        "E": np.zeros((2, 2, 2), dtype=dtype),
    }
    expected = {name: np.zeros_like(value) for name, value in raw.items()}
    for loc in range(2):
        foi = sum(float(value) for value in raw["I1"][loc].ravel()) + sum(
            float(value) for value in raw["X"][loc].ravel()
        )
        for age in range(2):
            for vax in range(2):
                for imm in range(2):
                    flow = foi * float(raw["X"][loc, age, vax, imm])
                    expected["X"][loc, age, vax, imm] = -flow
                    expected["E"][loc, age, vax] += flow
    with _precision(backend):
        module = (
            np
            if backend == "numpy"
            else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
        )
        state = {name: module.asarray(value) for name, value in raw.items()}
        xp = array_namespace(state["X"])
        result = compiled.pytree_eval_fn(0.0, state)
        tolerance = 2e-6 if dtype == "float32" else 1e-12
        for name, host_value in raw.items():
            assert result[name].shape == host_value.shape
            assert result[name].dtype == state[name].dtype
            assert array_namespace(result[name]) is xp
            np.testing.assert_allclose(
                _host(result[name]), expected[name], rtol=tolerance
            )
        flat = compiled.eval_fn(
            0.0, xp.concat([xp.reshape(value, (-1,)) for value in state.values()])
        )
        np.testing.assert_allclose(
            _host(flat),
            np.concatenate([value.ravel() for value in expected.values()]),
            rtol=tolerance,
        )
        block = compiled.block_pytree_eval_fn
        if backend == "jax":
            block = pytest.importorskip("jax").jit(block)
        for loc in range(2):
            part = block(0.0, {name: value[loc] for name, value in state.items()})
            for name in raw:
                np.testing.assert_allclose(
                    _host(part[name]), expected[name][loc], rtol=tolerance
                )


@contextmanager
def _precision(backend: str) -> Iterator[None]:
    if backend != "jax":
        yield
        return
    jax = pytest.importorskip("jax")
    previous = jax.config.read("jax_enable_x64")
    enable_x64 = True
    jax.config.update("jax_enable_x64", enable_x64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _backend_arrays(
    backend: str, dtype: str, *, coupled: bool
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    xp = (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )
    state, params = fixture(dtype=dtype, coupled=coupled)
    # Native Torch tensors need the Array API compatibility namespace.
    convert = xp.asarray
    state = {
        name: convert(value, dtype=getattr(xp, dtype)) for name, value in state.items()
    }
    params = {
        name: convert(value, dtype=getattr(xp, dtype)) for name, value in params.items()
    }
    return array_namespace(state["A"]), state, params


def _evaluate(
    compiled: CompiledRhs, evaluator: str, state: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    xp = array_namespace(state["A"])
    if evaluator == "flat":
        values = xp.concat([xp.reshape(state[name], (-1,)) for name in TEMPLATES])
        derivative = compiled.eval_fn(0.0, values, **params)
        size = prod(state["A"].shape)
        return {
            name: xp.reshape(
                derivative[index * size : (index + 1) * size], state[name].shape
            )
            for index, name in enumerate(TEMPLATES)
        }
    if evaluator == "structured":
        assert compiled.pytree_eval_fn is not None
        return compiled.pytree_eval_fn(0.0, state, **params)
    assert compiled.block_pytree_eval_fn is not None
    blocks = [
        compiled.block_pytree_eval_fn(
            0.0,
            {name: value[index] for name, value in state.items()},
            **{**params, "local": params["local"][index]},
        )
        for index in range(state["A"].shape[0])
    ]
    return {name: xp.stack([block[name] for block in blocks]) for name in TEMPLATES}


def _host(value: Any) -> np.ndarray:  # ruff: ignore[any-type]
    return np.asarray(
        value.detach().cpu().numpy() if hasattr(value, "detach") else value
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(("coupled", "evaluator"), CASES)
def test_hierarchical_value_contract(
    backend: str, dtype: str, evaluator: str, *, coupled: bool
) -> None:
    """Every qualified layout preserves shape/dtype/namespace and matches the oracle."""
    with _precision(backend):
        xp, state, params = _backend_arrays(backend, dtype, coupled=coupled)
        compiled = compile_spec(model_spec(3, 2, coupled=coupled))
        expected = reference_rhs(
            {k: _host(v) for k, v in state.items()},
            {k: _host(v) for k, v in params.items()},
            coupled=coupled,
        )
        result = _evaluate(compiled, evaluator, state, params)
        handwritten = handwritten_rhs(state, params, xp, coupled=coupled)
        tolerance = 2e-5 if dtype == "float32" else 1e-12
        for name in TEMPLATES:
            assert array_namespace(result[name]) is xp
            assert result[name].shape == state[name].shape
            assert result[name].dtype == state[name].dtype
            assert np.isfinite(_host(result[name])).all()
            np.testing.assert_allclose(
                _host(result[name]), expected[name], rtol=tolerance, atol=tolerance
            )
            np.testing.assert_allclose(
                _host(handwritten[name]), expected[name], rtol=tolerance, atol=tolerance
            )
        if coupled:
            assert compiled.block_pytree_eval_fn is None
            changed = {
                **state,
                "B": state["B"]
                + xp.asarray(
                    [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=state["B"].dtype
                ),
            }
            perturbed = _evaluate(compiled, evaluator, changed, params)
            assert not np.allclose(_host(perturbed["A"])[0], _host(result["A"])[0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_coupled_groups_reject_block_factorization(backend: str) -> None:
    """A declared independence optimization cannot silently remove coupling."""
    with _precision(backend):
        _backend_arrays(backend, "float64", coupled=True)
    spec = model_spec(3, 2, coupled=True)
    spec["factorize_axes"] = ["group"]
    with pytest.raises(UnsupportedFeatureError, match="group"):
        compile_spec(spec)


@pytest.mark.parametrize(("coupled", "evaluator"), CASES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_jax_jit_vmap_and_directional_derivatives(
    dtype: str, evaluator: str, *, coupled: bool
) -> None:
    """Compiled layouts support jit/vmap and derivatives checked against the oracle."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    with _precision("jax"):
        _xp, state, params = _backend_arrays("jax", dtype, coupled=coupled)
        compiled = compile_spec(model_spec(3, 2, coupled=coupled))

        def objective(gain: Any, local: Any) -> Any:  # ruff: ignore[any-type]
            result = _evaluate(
                compiled, evaluator, state, {**params, "gain": gain, "local": local}
            )
            return sum(jnp.sum(result[name] ** 2) for name in TEMPLATES)

        gains = jnp.asarray([0.3, 0.4, 0.5], dtype=getattr(jnp, dtype))
        batched = jax.jit(jax.vmap(objective, in_axes=(0, None)))(
            gains, params["local"]
        )
        assert batched.shape == (3,)
        direction = jnp.asarray([0.1, -0.2, 0.3], dtype=getattr(jnp, dtype))
        _, tangent = jax.jit(
            lambda g, p: jax.jvp(objective, (g, p), (jnp.ones_like(g), direction))
        )(params["gain"], params["local"])
        host_state = {name: _host(value) for name, value in state.items()}
        host_params = {name: _host(value) for name, value in params.items()}
        step = 2e-3 if dtype == "float32" else 1e-5

        def reference_objective(sign: float) -> float:
            altered = {
                **host_params,
                "gain": host_params["gain"] + sign * step,
                "local": host_params["local"] + sign * step * _host(direction),
            }
            result = reference_rhs(host_state, altered, coupled=coupled)
            return float(
                sum(np.sum(value.astype(np.float64) ** 2) for value in result.values())
            )

        expected = (reference_objective(1) - reference_objective(-1)) / (2 * step)
        np.testing.assert_allclose(
            float(tangent), expected, rtol=2e-3 if dtype == "float32" else 1e-7
        )


@pytest.mark.parametrize(("coupled", "evaluator"), CASES)
def test_torch_autograd_matches_independent_finite_differences(
    evaluator: str, *, coupled: bool
) -> None:
    """CPU Torch gradients survive shared/local parameters and all supported layouts."""
    torch = pytest.importorskip("torch")
    _xp, state, params = _backend_arrays("torch", "float64", coupled=coupled)
    state = {name: value.requires_grad_() for name, value in state.items()}
    params["gain"].requires_grad_()
    params["local"].requires_grad_()
    compiled = compile_spec(model_spec(3, 2, coupled=coupled))
    result = _evaluate(compiled, evaluator, state, params)
    sum(value.square().sum() for value in result.values()).backward()
    assert all(value.grad is not None for value in state.values())
    assert params["gain"].grad is not None
    assert params["local"].grad is not None
    host_state = {name: _host(value) for name, value in state.items()}
    host_params = {name: _host(value) for name, value in params.items()}
    step = 1e-5
    objectives = []
    for sign in [1, -1]:
        result = reference_rhs(
            host_state,
            {**host_params, "gain": host_params["gain"] + sign * step},
            coupled=coupled,
        )
        objectives.append(float(sum(np.sum(value**2) for value in result.values())))
    assert float(params["gain"].grad) == pytest.approx(
        (objectives[0] - objectives[1]) / (2 * step), rel=1e-7
    )
    assert torch.isfinite(params["local"].grad).all()


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_block_vmap_matches_independent_groups(backend: str) -> None:
    """Native backend mapping consumes sliced local and shared global parameters."""
    module = pytest.importorskip(backend)
    with _precision(backend):
        _xp, state, params = _backend_arrays(backend, "float64", coupled=False)
        compiled = compile_spec(model_spec(3, 2))
        block = compiled.block_pytree_eval_fn
        assert block is not None

        def one_group(values: Any, local: Any) -> Any:  # ruff: ignore[any-type]
            return block(0.0, values, **{**params, "local": local})

        result = (
            module.jit(module.vmap(one_group))(state, params["local"])
            if backend == "jax"
            else module.func.vmap(one_group)(state, params["local"])
        )
        expected = reference_rhs(
            {k: _host(v) for k, v in state.items()},
            {k: _host(v) for k, v in params.items()},
        )
        for name in TEMPLATES:
            np.testing.assert_allclose(
                _host(result[name]), expected[name], rtol=1e-12, atol=1e-12
            )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["expr", "transitions"])
def test_block_alias_and_local_parameter_regression(
    backend: str, dtype: str, kind: str
) -> None:
    """Stripped alias symbols and sliced local rates match independent cell flows."""
    spec: dict[str, Any] = {
        "kind": kind,
        "axes": [{"name": "loc", "coords": ["north", "south", "west"]}],
        "factorize_axes": ["loc"],
        "state": ["S[loc]", "I[loc]"],
        "aliases": {"foi[loc]": "beta * r0_loc[loc] * I[loc]"},
    }
    if kind == "expr":
        spec["equations"] = {
            "S[loc]": "-foi[loc] * S[loc]",
            "I[loc]": "foi[loc] * S[loc]",
        }
    else:
        spec["transitions"] = [{"from": "S[loc]", "to": "I[loc]", "rate": "foi[loc]"}]
    compiled = compile_spec(spec)
    assert compiled.pytree_eval_fn is not None
    assert compiled.block_pytree_eval_fn is not None
    host_state = {
        "S": np.array([9.0, 8.0, 11.0], dtype=dtype),
        "I": np.array([1.0, 2.0, 3.0], dtype=dtype),
    }
    host_local = np.array([1.25, 0.5, 2.0], dtype=dtype)
    beta = np.asarray(0.2, dtype=dtype)
    expected = beta * host_local * host_state["S"] * host_state["I"]
    with _precision(backend):
        module = (
            np
            if backend == "numpy"
            else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
        )
        state = {name: module.asarray(value) for name, value in host_state.items()}
        params = {"beta": module.asarray(beta), "r0_loc": module.asarray(host_local)}
        xp = array_namespace(state["S"])
        structured = compiled.pytree_eval_fn(0.0, state, **params)
        flat = compiled.eval_fn(0.0, xp.concat([state["S"], state["I"]]), **params)
        tolerance = 2e-6 if dtype == "float32" else 1e-12
        np.testing.assert_allclose(
            _host(flat), np.concatenate([-expected, expected]), rtol=tolerance
        )
        block = compiled.block_pytree_eval_fn
        if backend == "jax":
            block = pytest.importorskip("jax").jit(block)
        for index in range(3):
            result = block(
                0.0,
                {name: value[index] for name, value in state.items()},
                beta=params["beta"],
                r0_loc=params["r0_loc"][index],
            )
            for name, sign in [("S", -1), ("I", 1)]:
                assert result[name].shape == ()
                assert result[name].dtype == state[name].dtype
                assert array_namespace(result[name]) is xp
                np.testing.assert_allclose(
                    _host(result[name]), sign * expected[index], rtol=tolerance
                )
                np.testing.assert_allclose(
                    _host(structured[name])[index],
                    sign * expected[index],
                    rtol=tolerance,
                )
