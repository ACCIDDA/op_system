"""Typed interchange, native derivative matrices and independent external values."""

from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from op_system import (
    CompiledRhs,
    PortableIR,
    compile_spec,
    export_ir,
    julia_rhs,
    symbolic_derivatives,
)


def _compiled() -> CompiledRhs:
    """Construct a conservative coupled resource transfer and scalar measurement.

    Returns:
        Native compiler artifact with transitive aliases and an observable-only input.
    """
    return compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "aliases": {"loss": "a*x", "net": "loss-b*z"},
        "equations": {"x": "-net", "z": "net"},
        "observables": {"measurement": {"expr": "gain*x+z+t"}},
    })


def test_portable_roundtrip_and_identity_reject_changed_data() -> None:
    """All expression, layout and metadata bytes remain tied to expected source."""
    compiled = _compiled()
    document = export_ir(compiled)
    decoded = document.decode()
    assert decoded["state_names"] == ("x", "z")
    assert decoded["param_names"] == ("a", "b")
    assert "metadata" in decoded
    assert "observables" in decoded
    assert (
        PortableIR.from_json(document.to_json(), expected=document.identity) == document
    )
    with pytest.raises(TypeError):
        decoded["state_names"] = ("z", "x")  # type: ignore[index]
    data = json.loads(document.to_json())
    data["manifest"]["nodes"][0][1] = "changed"
    with pytest.raises(ValueError, match=r"identity|scalar"):
        PortableIR.from_json(json.dumps(data), expected=document.identity)
    data = json.loads(document.to_json())
    data["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        PortableIR.from_json(json.dumps(data), expected=document.identity)
    with pytest.raises(ValueError, match="duplicate"):
        PortableIR.from_json('{"format":1,"format":2}', expected=document.identity)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_derivatives_actions_and_observable_only_parameter(
    backend: str, dtype: str
) -> None:
    """Independent linear matrices detect lost aliases, coordinates and measurements."""
    compiled = _compiled()
    artifact = symbolic_derivatives(compiled)
    assert artifact.parameter_names == ("a", "b", "gain")
    module: Any = np
    jax: Any = None
    if backend == "jax":
        jax = pytest.importorskip("jax")
        module = pytest.importorskip("jax.numpy")
    elif backend == "torch":
        module = pytest.importorskip("torch")
    precision = nullcontext()
    if jax is not None:
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        precision = enable()
    with precision:
        y = module.asarray([2.0, 4.0], dtype=getattr(module, dtype))
        parameters = {
            name: module.asarray(value, dtype=getattr(module, dtype))
            for name, value in zip(
                artifact.parameter_names, (0.3, 0.7, 3.0), strict=True
            )
        }
        values = (
            jax.jit(artifact.evaluate)(module.asarray(0.5), y, **parameters)
            if jax is not None
            else artifact.evaluate(0.5, y, **parameters)
        )

        def host(value: Any) -> Any:  # ruff: ignore[any-type]
            """Convert only at the independent assertion boundary.

            Returns:
                Host values after native evaluation and differentiation.
            """
            return value.detach().numpy() if backend == "torch" else np.asarray(value)

        expected = np.array([[-0.3, 0.7], [0.3, -0.7]])
        np.testing.assert_allclose(host(values.state), expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            host(values.parameters), [[-2, 4, 0], [2, -4, 0]], rtol=1e-6
        )
        np.testing.assert_allclose(host(values.observable_state), [[3, 1]], rtol=1e-6)
        np.testing.assert_allclose(
            host(values.observable_parameters), [[0, 0, 2]], rtol=1e-6
        )
        vector = module.asarray([0.5, -1.0], dtype=y.dtype)
        np.testing.assert_allclose(
            host(artifact.jvp(0.5, y, vector, **parameters)),
            expected @ np.array([0.5, -1.0]),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            host(artifact.vjp(0.5, y, vector, **parameters)),
            expected.T @ np.array([0.5, -1.0]),
            rtol=1e-6,
            atol=1e-7,
        )
        zero = {**parameters, "a": module.asarray(0.0, dtype=y.dtype)}
        artifact.evaluate(0.5, y, **zero)
        assert artifact.state_dependencies == ((True, True), (True, True))


@pytest.mark.parametrize(
    "expression",
    [
        "np.exp(a*x)",
        "np.log(x)",
        "np.sqrt(x)",
        "np.sin(x)",
        "np.cos(x)",
        "np.tanh(x)",
        "x**3+a*x",
        "x**0.5",
        "x**(-2)",
        "a/(1+x*x)",
    ],
)
def test_nonlinear_symbolic_matrix_matches_independent_finite_difference(
    expression: str,
) -> None:
    """Separate numerical differentiation checks every supported chain/product rule."""
    compiled = compile_spec({
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": expression},
    })
    artifact = symbolic_derivatives(compiled)
    parameters = {name: np.array(0.4) for name in artifact.parameter_names}
    y = np.array([0.7])
    epsilon = 1e-5
    reference = (
        compiled.eval_fn(0.2, y + epsilon, **parameters)
        - compiled.eval_fn(0.2, y - epsilon, **parameters)
    ) / (2 * epsilon)
    actual = artifact.evaluate(0.2, y, **parameters)
    assert isinstance(actual.state, np.ndarray)
    np.testing.assert_allclose(
        actual.state, reference.reshape(1, 1), rtol=1e-8, atol=1e-8
    )


def test_unsupported_scope_and_allocation_bounds_are_explicit() -> None:
    """Unsupported nodes are never reported as structural zeros or fake derivatives."""
    compiled = _compiled()
    with pytest.raises(ValueError, match="state limit"):
        symbolic_derivatives(compiled, max_states=1)
    with pytest.raises(ValueError, match="entry limit"):
        symbolic_derivatives(compiled, max_entries=1)
    nonsmooth = compile_spec({
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "np.abs(x)"},
    })
    with pytest.raises(NotImplementedError, match="Unsupported"):
        symbolic_derivatives(nonsmooth)
    with pytest.raises(NotImplementedError):
        julia_rhs(nonsmooth)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_symbolic_matrices_retain_nonlinear_native_gradient_graphs(
    backend: str,
) -> None:
    """Independent curvature of exp(a*x) detects a detached derivative evaluator."""
    compiled = compile_spec({
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "np.exp(a*x)"},
    })
    artifact = symbolic_derivatives(compiled)
    if backend == "jax":
        jax = pytest.importorskip("jax")
        xp = pytest.importorskip("jax.numpy")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        with enable():
            gradient = jax.jit(
                jax.grad(
                    lambda y: xp.sum(artifact.evaluate(0.0, y, a=xp.asarray(0.4)).state)
                )
            )(xp.asarray([0.7]))
    else:
        torch = pytest.importorskip("torch")
        value = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        matrix = artifact.evaluate(
            0.0, value, a=torch.asarray(0.4, dtype=torch.float64)
        ).state
        assert isinstance(matrix, torch.Tensor)
        matrix.sum().backward()
        gradient = value.grad.detach().numpy()
    np.testing.assert_allclose(gradient, [0.16 * np.exp(0.28)], rtol=1e-12, atol=1e-12)
