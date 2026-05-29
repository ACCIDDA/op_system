"""Unit tests for op_system.compile (pytest).

These tests cover:
- compile_rhs produces an eval_fn callable
- eval_fn evaluates expr correctly
- CompiledRhs.bind returns a 2-arg RHS with parameters fixed
- bind returns a strict 2-arg callable (no runtime kwargs)
- shape validation rejects non-1D and wrong-length states
- missing symbols raise parameter errors
- invalid syntax is rejected during normalization (specs) in v1
- AST whitelist rejects disallowed calls/attribute access
- alias dependency resolution and cycles
- transitions RHS compiles and evaluates with expected flow semantics
- compile_rhs threads NormalizedRhs.meta through to CompiledRhs.meta

Note:
- errors.py was removed, so we only assert built-in exception types and messages.
"""

from __future__ import annotations

import pickle  # noqa: S403
import re
from dataclasses import replace

import numpy as np
import pytest

import op_system._vectorize as _vec
from op_system import compile_spec
from op_system._errors import UnsupportedFeatureError
from op_system._operators import OperatorDescriptor
from op_system._vectorize import _bail
from op_system.compile import CompiledRhs, _collect_eq_code, compile_rhs
from op_system.specs import NormalizedRhs, normalize_rhs


@pytest.fixture
def expr_spec_xy() -> dict[str, object]:
    """Base 2-state expression RHS spec used by multiple tests.

    Returns:
        Base RHS spec dictionary.
    """
    return {
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {"x": "a * x", "y": "x + y + b"},
    }


@pytest.fixture
def rhs_xy(expr_spec_xy: dict[str, object]) -> NormalizedRhs:
    """Normalized RHS for the base 2-state spec.

    Args:
        expr_spec_xy: Base RHS spec dictionary.

    Returns:
        NormalizedRhs instance.
    """
    return normalize_rhs(expr_spec_xy)


@pytest.fixture
def compiled_xy(rhs_xy: NormalizedRhs) -> CompiledRhs:
    """Compiled RHS for the base 2-state spec.

    Args:
        rhs_xy: NormalizedRhs instance.

    Returns:
        CompiledRhs instance.
    """
    return compile_rhs(rhs_xy, xp=np)


def test_compile_rhs_expr_happy_path_and_eval(compiled_xy: CompiledRhs) -> None:
    """compile_rhs produces eval_fn that evaluates expressions correctly."""
    out = compiled_xy.eval_fn(
        np.float64(0.0),
        np.array([2.0, 3.0], dtype=np.float64),
        a=2.0,
        b=1.0,
    )
    assert out.dtype == np.float64
    assert out.shape == (2,)
    assert np.allclose(out, np.array([4.0, 6.0], dtype=np.float64))


def test_compiledrhs_bind_binds_params(compiled_xy: CompiledRhs) -> None:
    """CompiledRhs.bind returns a 2-arg callable with parameters fixed."""
    bound = compiled_xy.bind({"a": 2.0, "b": 1.0})
    assert callable(bound)

    out = bound(np.float64(0.0), np.array([2.0, 3.0], dtype=np.float64))
    assert out.dtype == np.float64
    assert out.shape == (2,)
    assert np.allclose(out, np.array([4.0, 6.0], dtype=np.float64))


def test_bind_is_strict_two_arg_callable(compiled_xy: CompiledRhs) -> None:
    """bind() returns a strict (t, y) callable and rejects runtime kwargs."""
    bound = compiled_xy.bind({"a": 2.0, "b": 1.0})
    with pytest.raises(TypeError, match=r"unexpected keyword argument"):
        # This call is intentionally invalid at the type level
        bound(
            np.float64(0.0),
            np.array([2.0, 3.0], dtype=np.float64),
            a=3.0,
        )  # type: ignore[call-arg]


def test_eval_rejects_non_1d_state_shape() -> None:
    """eval_fn rejects non-1D state arrays with standardized message."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "0.0"}}
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)

    y_bad = np.zeros((1, 1), dtype=np.float64)
    msg = "state has an invalid shape/value. Expected (n_state=1,)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        compiled.eval_fn(np.float64(0.0), y_bad)


def test_eval_rejects_wrong_state_length() -> None:
    """eval_fn rejects state arrays of incorrect length with standardized message."""
    spec = {
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {"x": "0.0", "y": "0.0"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)

    msg = "state has an invalid shape/value. Expected (n_state=2,)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        compiled.eval_fn(np.float64(0.0), np.zeros((3,), dtype=np.float64))


def test_eval_unknown_symbol_raises_parameter_error() -> None:
    """eval_fn raises parameter error for unknown symbols in expressions."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "beta * x"}}
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)

    with pytest.raises(TypeError, match=r"Invalid parameters for op_system"):
        compiled.eval_fn(np.float64(0.0), np.array([1.0], dtype=np.float64))


def test_invalid_expression_syntax_rejected_during_normalize() -> None:
    """In v1, expression syntax is validated during specs normalization."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "beta **"}}
    with pytest.raises(ValueError, match=r"Invalid op_system expression"):
        normalize_rhs(spec)


def test_compile_rejects_disallowed_function_calls() -> None:
    """Disallowed function calls (e.g., np.sin) should be rejected."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "np.floor(x)"}}
    rhs = normalize_rhs(spec)
    with pytest.raises(ValueError, match=r"disallowed function call"):
        compile_rhs(rhs, xp=np)


def test_compile_rejects_nonwhitelisted_helper() -> None:
    """Bare helper calls must be whitelisted; unknown helpers are rejected."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "foo(x)"}}
    rhs = normalize_rhs(spec)
    with pytest.raises(ValueError, match=r"disallowed function call"):
        compile_rhs(rhs, xp=np)


@pytest.mark.parametrize("helper", ["history", "delay"])
def test_compile_rejects_planned_history_helpers_with_targeted_message(
    helper: str,
) -> None:
    """#173 scaffold: planned history helpers have explicit unsupported diagnostics."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": f"{helper}(x)"}}
    rhs = normalize_rhs(spec)
    with pytest.raises(
        UnsupportedFeatureError,
        match=r"issue #173.*history_requirements=",
    ):
        compile_rhs(rhs, xp=np)


def test_history_requirements_payload_includes_missing_required_options() -> None:
    """history_requirements payload reports missing mandatory helper kwargs."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "delay(x)"}}
    rhs = normalize_rhs(spec)

    with pytest.raises(UnsupportedFeatureError) as exc_info:
        compile_rhs(rhs, xp=np)

    msg = str(exc_info.value)
    assert "'kind': 'delay'" in msg
    assert "'required_options': ('tau',)" in msg
    assert "'missing_required_options': ('tau',)" in msg


def test_history_requirements_payload_captures_provided_options() -> None:
    """#174 integration: signal_id and provided options exposed in requirements map."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "equations": {
            "x": "convolve_history(x, kernel=gamma, window=14, interpolation=linear)"
        },
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)
    assert compiled.history_requirements
    first_req = compiled.history_requirements[0]
    options = first_req["options"]
    assert isinstance(options, dict)
    assert first_req["signal_id"] == 0
    assert options["kernel"] == "gamma"
    assert options["window"] == "14"
    assert options["interpolation"] == "linear"


def test_convolve_history_compiles_and_calls_provider() -> None:
    """#175 integration: convolve_history lowers to provider hook invocation."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["x[loc]"],
        "equations": {"x[loc]": "convolve_history(x[loc], kernel=gamma, window=14)"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)
    assert compiled.pytree_eval_fn is not None
    assert compiled.history_eval_fn is not None
    # Per-cell scopes report one requirement per axis coord; the vectorized
    # lowering collapses to a single ``__hist_query`` call (signal_id == 0).
    assert len(compiled.history_requirements) >= 1
    assert any(req["signal_id"] == 0 for req in compiled.history_requirements)

    # Mock provider exposes a ``.query`` method matching the lowered call:
    # ``__hist_query(signal_id, body, **options)``.
    mock_queries: list[tuple[int, object, dict[str, object]]] = []

    class MockProvider:
        @staticmethod
        def query(signal_id: int, body: object, **options: object) -> object:
            mock_queries.append((signal_id, body, options))
            return np.zeros_like(body)

    state = {"x": np.array([1.5, 2.5])}
    result = compiled.history_eval_fn(0.0, state, history_provider=MockProvider())
    assert len(mock_queries) >= 1
    signal_id, _body, opts = mock_queries[0]
    assert signal_id == 0
    assert opts["kernel"] == "gamma"
    assert opts["window"] == 14
    assert "x" in result


def test_history_eval_fn_is_none_when_no_history_ops() -> None:
    """#175 integration: history_eval_fn is None when no history ops present."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "x + 1"}}
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)
    assert compiled.history_eval_fn is None


def test_compile_rejects_disallowed_attribute_access() -> None:
    """Disallowed attribute access should be rejected before evaluation."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "x.real"}}
    with pytest.raises(
        ValueError,
        match=r"unsupported AST node in IR parser: Attribute",
    ):
        normalize_rhs(spec)


def test_eval_allows_whitelisted_np_calls() -> None:
    """A small set of np.* calls is whitelisted and should work."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "np.maximum(x, 0.0)"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)
    out = compiled.eval_fn(np.float64(0.0), np.array([-2.0], dtype=np.float64))
    assert np.allclose(out, np.array([0.0], dtype=np.float64))


def test_eval_allows_expanded_np_whitelist() -> None:
    """Expanded whitelist functions should parse, compile, and evaluate."""
    expr = (
        "np.sin(x) + np.cos(x) + np.tan(x) + np.sinh(x) + np.cosh(x) + "
        "np.tanh(x) + np.expm1(x) + np.log2(x) + np.log10(x) + np.hypot(x, 2.0) + "
        "np.arctan2(x, 2.0)"
    )
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": expr}}
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)
    x_val = np.array([1.0], dtype=np.float64)
    out = compiled.eval_fn(np.float64(0.0), x_val)
    expected = np.array(
        [
            np.sin(1.0)
            + np.cos(1.0)
            + np.tan(1.0)
            + np.sinh(1.0)
            + np.cosh(1.0)
            + np.tanh(1.0)
            + np.expm1(1.0)
            + np.log2(1.0)
            + np.log10(1.0)
            + np.hypot(1.0, 2.0)
            + np.arctan2(1.0, 2.0)
        ],
        dtype=np.float64,
    )
    assert out.dtype == np.float64
    assert out.shape == (1,)
    assert np.allclose(out, expected)


def test_templates_and_apply_along_compile_and_eval() -> None:
    """Templated states with apply_along compile and evaluate end-to-end."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1", "p2"]}],
        "state": ["S[pop]", "I[pop]"],
        "equations": {
            "S[pop]": "-beta * S[pop] * apply_along(I[pop:j], pop=j)",
            "I[pop]": "beta * S[pop] * apply_along(I[pop:j], pop=j) - gamma * I[pop]",
        },
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)

    # State ordering after expansion: S__pop_p1, S__pop_p2, I__pop_p1, I__pop_p2
    y = np.array([10.0, 20.0, 1.0, 2.0], dtype=np.float64)
    out = compiled.eval_fn(np.float64(0.0), y, beta=0.5, gamma=0.1)

    inf_total = 1.0 + 2.0
    expected = np.array(
        [
            -0.5 * 10.0 * inf_total,
            -0.5 * 20.0 * inf_total,
            0.5 * 10.0 * inf_total - 0.1 * 1.0,
            0.5 * 20.0 * inf_total - 0.1 * 2.0,
        ],
        dtype=np.float64,
    )
    assert out.shape == (4,)
    assert out.dtype == np.float64
    assert np.allclose(out, expected)


def test_reducer_helpers_sum_state_and_prefix() -> None:
    """sum_state and sum_prefix are available in equations/aliases."""
    spec = {
        "kind": "expr",
        "state": ["x", "y"],
        "aliases": {"tot": "sum_state()"},
        "equations": {
            "x": "sum_state()",  # x+y
            "y": "sum_prefix('x')",  # just x
        },
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)
    out = compiled.eval_fn(np.float64(0.0), np.array([2.0, 3.0], dtype=np.float64))
    assert np.allclose(out, np.array([5.0, 2.0], dtype=np.float64))


def test_alias_dependency_resolution() -> None:
    """Aliases may depend on earlier aliases and should resolve."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "aliases": {"a": "x + 1", "b": "a + 2"},
        "equations": {"x": "b"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)
    out = compiled.eval_fn(np.float64(0.0), np.array([3.0], dtype=np.float64))
    assert np.allclose(out, np.array([6.0], dtype=np.float64))


def test_alias_cycle_is_rejected() -> None:
    """Alias cycles should be rejected during evaluation (cannot resolve)."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "aliases": {"a": "b + 1", "b": "a + 1"},
        "equations": {"x": "a"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)
    with pytest.raises(ValueError, match=r"could not resolve alias dependencies"):
        compiled.eval_fn(np.float64(0.0), np.array([1.0], dtype=np.float64))


def test_scalar_compile_uses_normalized_ir_when_available() -> None:
    """Scalar fallback should compile from IR instead of reparsing strings."""
    rhs = normalize_rhs({
        "kind": "expr",
        "state": ["x"],
        "aliases": {"double": "x * 2"},
        "equations": {"x": "double + 1"},
    })
    rhs = replace(
        rhs,
        aliases={"double": "not valid python !!!"},
        equations=("also not valid python !!!",),
    )

    compiled = compile_rhs(rhs, xp=np)
    out = compiled.eval_fn(np.float64(0.0), np.array([2.0], dtype=np.float64))
    assert np.allclose(out, np.array([5.0]))


def test_scalar_compile_extracts_equation_ir_cse() -> None:
    """Repeated equation IR subtrees should compile as scalar temporaries."""
    rhs = normalize_rhs({
        "kind": "expr",
        "state": ["x", "y"],
        "equations": {
            "x": "(a + b) * (a + b)",
            "y": "(a + b) + 1",
        },
    })

    cse_code, _ = _collect_eq_code(
        rhs.equations,
        rhs.equations_ir,
        reserved_names={"__op_system_cse_0"},
    )
    assert tuple(name for name, _ in cse_code) == ("__op_system_cse_1",)

    compiled = compile_rhs(rhs, xp=np)
    out = compiled.eval_fn(
        np.float64(0.0),
        np.array([0.0, 0.0], dtype=np.float64),
        a=2.0,
        b=3.0,
    )
    assert np.allclose(out, np.array([25.0, 6.0]))


def test_transitions_rhs_compiles_and_evaluates() -> None:
    """Transitions RHS should compile and evaluate with expected flow semantics."""
    spec = {
        "kind": "transitions",
        "state": ["S", "I", "R"],
        "aliases": {"N": "S + I + R"},
        "transitions": [
            {"from": "S", "to": "I", "rate": "beta * I / N"},
            {"from": "I", "to": "R", "rate": "gamma"},
        ],
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)

    y = np.array([999.0, 1.0, 0.0], dtype=np.float64)
    out = compiled.eval_fn(np.float64(0.0), y, beta=0.3, gamma=0.1)

    n_total = float(np.sum(y))
    inf = 0.3 * float(y[1]) / n_total * float(y[0])
    rec = 0.1 * float(y[1])
    expected = np.array([-inf, inf - rec, rec], dtype=np.float64)
    assert np.allclose(out, expected)


def test_transitions_source_only_compiles_and_evaluates() -> None:
    """Source-only transitions add to ``to`` without draining any donor."""
    spec = {
        "kind": "transitions",
        "state": ["I", "H_cum"],
        "transitions": [
            {"from": None, "to": "H_cum", "rate": "k * I"},
        ],
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)

    y = np.array([4.0, 10.0], dtype=np.float64)
    out = compiled.eval_fn(np.float64(0.0), y, k=2.5)
    np.testing.assert_allclose(out, np.array([0.0, 10.0], dtype=np.float64))


def test_transitions_source_only_templated_vectorized() -> None:
    """Templated source-only transitions work in the vectorized compile path."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "age", "coords": ["y", "o"]}],
        "state": ["I[age]", "H_cum[age]"],
        "transitions": [
            {"to": "H_cum[age]", "rate": "k[age] * I[age]"},
        ],
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)

    y = np.array([2.0, 3.0, 100.0, 200.0], dtype=np.float64)
    out = compiled.eval_fn(np.float64(0.0), y, k=np.array([1.0, 0.5]))
    # dI/dt = 0; dH_cum/dt = k[age] * I[age]
    np.testing.assert_allclose(out, np.array([0.0, 0.0, 2.0, 1.5], dtype=np.float64))


# ---------------------------------------------------------------------------
# Meta threading (#11)
# ---------------------------------------------------------------------------


def test_compiled_rhs_meta_from_spec_with_axes_and_kernels() -> None:
    """compile_rhs preserves axes and kernels metadata from NormalizedRhs."""
    spec: dict[str, object] = {
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["a0", "a1", "a2"]}],
        "state": ["S[age]", "I[age]"],
        "kernels": [
            {
                "name": "contact",
                "form": "gaussian",
                "params": {"scale": 1.0, "sigma": 0.5},
            },
        ],
        "equations": {"S[age]": "-beta * S[age]", "I[age]": "beta * S[age]"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=np)

    assert "axes" in compiled.meta
    assert len(compiled.meta["axes"]) == 1
    assert compiled.meta["axes"][0]["name"] == "age"
    assert "kernels" in compiled.meta
    assert len(compiled.meta["kernels"]) == 1
    assert compiled.meta["kernels"][0]["name"] == "contact"


def test_compiled_rhs_meta_empty_for_bare_spec(compiled_xy: CompiledRhs) -> None:
    """A spec with no axes/kernels/operators produces meta with empty values."""
    assert compiled_xy.meta.get("axes") == []
    assert compiled_xy.meta.get("kernels") == []


def test_compile_spec_preserves_meta() -> None:
    """compile_spec (public facade) round-trips meta through normalize+compile."""
    spec: dict[str, object] = {
        "kind": "expr",
        "axes": [{"name": "space", "coords": ["s0", "s1"]}],
        "state": ["x[space]"],
        "equations": {"x[space]": "a * x[space]"},
    }
    compiled = compile_spec(spec)

    assert "axes" in compiled.meta
    assert compiled.meta["axes"][0]["name"] == "space"


def test_compile_spec_owns_default_backend_policy() -> None:
    """compile_spec defaults to NumPy without requiring callers to pass xp."""
    spec: dict[str, object] = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "2.0 * x"},
    }
    compiled = compile_spec(spec)
    out = compiled.eval_fn(np.float64(0.0), np.array([3.0], dtype=np.float64))
    assert np.allclose(out, np.array([6.0], dtype=np.float64))


def test_compile_spec_rejects_conflicting_backend_and_xp() -> None:
    """compile_spec emits a DeprecationWarning when xp/backend are passed.

    The old "conflicting backend and xp" error is gone: namespace selection
    is now per-call from the input ``y``, so both kwargs are accepted (and
    ignored) under a single deprecation warning.
    """
    spec: dict[str, object] = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "x"},
    }
    with pytest.warns(DeprecationWarning, match="namespace from the input"):
        compiled = compile_spec(spec, xp=np, backend="jax")
    # Still produces a working compiled RHS.
    out = compiled.eval_fn(0.0, np.array([2.0], dtype=np.float64))
    assert np.isclose(float(out[0]), 2.0)


def test_compile_spec_with_backend_jax_is_jittable() -> None:
    """compile_spec produces a JAX-jittable eval_fn when called with a JAX state.

    The namespace is inferred from the input ``y`` via
    ``__array_namespace__``, so no compile-time backend selection is
    required (the deprecated ``backend`` kwarg is accepted and ignored).
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec: dict[str, object] = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "beta * x"},
    }
    with pytest.warns(DeprecationWarning, match="namespace from the input"):
        compiled = compile_spec(spec, backend="jax")

    y0 = jnp.asarray([2.0])
    out = jax.jit(lambda beta: compiled.eval_fn(0.0, y0, beta=beta))(1.5)
    assert np.allclose(np.asarray(out), np.asarray([3.0]))


def test_compile_rhs_requires_explicit_backend_namespace() -> None:
    """compile_rhs no longer requires (or honors) an explicit ``xp``.

    Backend selection is per-call from the input ``y`` via
    ``__array_namespace__``. Passing ``xp`` is accepted under a
    DeprecationWarning for one release; omitting it is the new default.
    """
    spec: dict[str, object] = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "x"},
    }
    rhs = normalize_rhs(spec)
    # Default (no xp) works.
    compiled = compile_rhs(rhs)
    out = compiled.eval_fn(0.0, np.array([2.5], dtype=np.float64))
    assert np.isclose(float(out[0]), 2.5)
    # Passing xp emits a DeprecationWarning but still produces a usable
    # CompiledRhs.
    with pytest.warns(DeprecationWarning, match="namespace from the input"):
        compiled_dep = compile_rhs(rhs, xp=np)
    out2 = compiled_dep.eval_fn(0.0, np.array([4.0], dtype=np.float64))
    assert np.isclose(float(out2[0]), 4.0)


def test_compile_rhs_with_jax_backend_is_jittable() -> None:
    """JAX backend preserves tracers and can be jitted/differentiated."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    spec = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "beta * x"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=jnp)

    y0 = jnp.asarray([2.0])
    eval_jit = jax.jit(lambda beta: compiled.eval_fn(0.0, y0, beta=beta))
    out = eval_jit(1.5)
    assert np.allclose(np.asarray(out), np.asarray([3.0]))

    grad_fn = jax.grad(
        lambda beta: compiled.eval_fn(0.0, y0, beta=beta)[0],
    )
    assert np.isclose(float(grad_fn(1.5)), 2.0)


def test_compile_rhs_with_jax_backend_diffrax_smoke() -> None:
    """Compiled JAX RHS can be consumed directly by diffrax."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    diffrax = pytest.importorskip("diffrax")

    spec = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "-beta * x"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=jnp)

    term = diffrax.ODETerm(lambda t, y, args: compiled.eval_fn(t, y, **args))
    solver = diffrax.Tsit5()

    def solve(beta: float) -> object:
        return diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.1,
            y0=jnp.asarray([1.0]),
            args={"beta": beta},
            saveat=diffrax.SaveAt(t1=True),
        ).ys[0, 0]

    out = jax.jit(solve)(0.5)
    expected = np.exp(-0.5)
    assert np.isclose(float(out), expected, rtol=1e-3)


def test_compile_rhs_traces_through_blackjax_nuts() -> None:
    """A representative NUTS step can trace gradients through RHS eval."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jr = pytest.importorskip("jax.random")
    blackjax = pytest.importorskip("blackjax")

    spec = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "-beta * x"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs, xp=jnp)

    y0 = jnp.asarray([1.0])
    observed_dydt = -0.7

    def logdensity(theta: np.ndarray) -> object:
        beta = jnp.exp(theta[0])
        pred = compiled.eval_fn(0.0, y0, beta=beta)[0]
        residual = pred - observed_dydt
        prior = -0.5 * theta[0] ** 2
        likelihood = -40.0 * residual**2
        return prior + likelihood

    nuts = blackjax.nuts(
        logdensity,
        step_size=0.1,
        inverse_mass_matrix=jnp.ones((1,)),
    )
    state = nuts.init(jnp.asarray([0.0]))
    key = jr.PRNGKey(0)

    next_state, _info = jax.jit(nuts.step)(key, state)
    assert np.isfinite(float(next_state.logdensity))


# -----------------------------------------------------------------------------
# Pickling round-trip
# -----------------------------------------------------------------------------


def test_compiledrhs_pickle_roundtrip_scalar_path(
    compiled_xy: CompiledRhs,
) -> None:
    """A CompiledRhs from the scalar path round-trips through pickle."""
    blob = pickle.dumps(compiled_xy)
    restored = pickle.loads(blob)  # noqa: S301

    assert isinstance(restored, CompiledRhs)
    assert restored.state_names == compiled_xy.state_names
    assert restored.param_names == compiled_xy.param_names

    y = np.array([2.0, 3.0], dtype=np.float64)
    expected = compiled_xy.eval_fn(np.float64(0.0), y, a=2.0, b=1.0)
    got = restored.eval_fn(np.float64(0.0), y, a=2.0, b=1.0)
    assert np.allclose(got, expected)


def test_compiledrhs_pickle_roundtrip_vectorized_path() -> None:
    """A CompiledRhs from the vectorized path round-trips through pickle."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["y", "o"]}],
        "state": ["S[age]", "I[age]"],
        "param": ["beta", "gamma"],
        "equations": {
            "S[age]": "-beta * S[age] * I[age]",
            "I[age]": "beta * S[age] * I[age] - gamma * I[age]",
        },
    }
    cr = compile_rhs(normalize_rhs(spec))
    blob = pickle.dumps(cr)
    restored = pickle.loads(blob)  # noqa: S301

    assert restored.state_names == cr.state_names
    y = np.array([99.0, 99.0, 1.0, 1.0], dtype=np.float64)
    expected = cr.eval_fn(0.0, y, beta=0.3, gamma=0.1)
    got = restored.eval_fn(0.0, y, beta=0.3, gamma=0.1)
    assert np.allclose(got, expected)


def test_compiledrhs_pickle_rejects_direct_construction(
    rhs_xy: NormalizedRhs,
) -> None:
    """A CompiledRhs built without ``_rhs`` raises a clear error on pickle."""
    cr = compile_rhs(rhs_xy)
    # Drop the retained source via dataclasses.replace and re-attach the
    # original eval_fn so the instance is still callable but no longer
    # carries a recipe for rebuilding itself.
    stripped = replace(cr, _rhs=None)
    with pytest.raises(TypeError, match=r"not picklable"):
        pickle.dumps(stripped)


# ---------------------------------------------------------------------------
# OperatorDescriptor enrichment (kind, bc) tests
# ---------------------------------------------------------------------------


def test_parse_operator_descriptor_kind_and_bc() -> None:
    """_parse_operator_descriptors maps kind and bc onto OperatorDescriptor."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["S[loc]"],
        "equations": {"S[loc]": "-S[loc]"},
        "operators": [
            {"kind": "advection", "axis": "loc", "bc": "absorbing", "velocity": "v"},
        ],
    }
    cr = compile_rhs(normalize_rhs(spec))
    assert len(cr.operators) == 1
    op = cr.operators[0]
    assert op.kind == "advection"
    assert op.bc == "absorbing"
    assert op.velocity == "v"
    assert op.axis == "loc"


def test_operator_descriptor_kind_bc_default_none() -> None:
    """OperatorDescriptor defaults kind and bc to None when constructed directly."""
    od = OperatorDescriptor(axis="loc")
    assert od.kind is None
    assert od.bc is None
    # When kind IS provided but bc is not, bc also defaults to None.
    od2 = OperatorDescriptor(axis="loc", kind="diffusion")
    assert od2.bc is None


# ---------------------------------------------------------------------------
# factorize_axes tests
# ---------------------------------------------------------------------------


def test_compiledrhs_factorize_axes_default_empty() -> None:
    """CompiledRhs.factorize_axes is an empty tuple when not declared in meta."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "-x"},
    }
    cr = compile_rhs(normalize_rhs(spec))
    assert cr.factorize_axes == ()


def test_compiledrhs_factorize_axes_from_meta() -> None:
    """CompiledRhs.factorize_axes is populated from meta['factorize_axes']."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "loc", "coords": ["a", "b", "c"]},
        ],
        "state": ["S[age, loc]"],
        "equations": {"S[age, loc]": "-S[age, loc]"},
        "factorize_axes": ["loc"],
    }
    cr = compile_rhs(normalize_rhs(spec))
    assert cr.factorize_axes == ("loc",)


# ---------------------------------------------------------------------------
# pytree_eval_fn tests
# ---------------------------------------------------------------------------


def test_pytree_eval_fn_set_for_vectorized_path() -> None:
    """pytree_eval_fn is not None when the vectorized compile path succeeds."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["S[loc]", "I[loc]"],
        "equations": {
            "S[loc]": "-beta * S[loc] * I[loc]",
            "I[loc]": "beta * S[loc] * I[loc] - gamma * I[loc]",
        },
    }
    cr = compile_rhs(normalize_rhs(spec))
    assert cr.pytree_eval_fn is not None


def test_pytree_eval_fn_produces_correct_result() -> None:
    """pytree_eval_fn accepts and returns StateDict with correct values."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["S[loc]", "I[loc]"],
        "equations": {
            "S[loc]": "-beta * S[loc] * I[loc]",
            "I[loc]": "beta * S[loc] * I[loc] - gamma * I[loc]",
        },
    }
    cr = compile_rhs(normalize_rhs(spec))
    assert cr.pytree_eval_fn is not None

    # Build a StateDict matching the template shapes.
    assert cr.template_shapes is not None
    y_dict = {
        base: np.ones(shape, dtype=np.float64)
        for base, shape in cr.template_shapes.items()
    }
    result = cr.pytree_eval_fn(0.0, y_dict, beta=0.3, gamma=0.1)

    # Verify structure: same keys, same shapes.
    assert set(result.keys()) == set(y_dict.keys())
    for base, arr in result.items():
        assert arr.shape == cr.template_shapes[base]

    # Flat eval and pytree eval must agree.
    y_flat = np.concatenate([y_dict[b].reshape(-1) for b in cr.template_shapes])
    flat_result = cr.eval_fn(0.0, y_flat, beta=0.3, gamma=0.1)
    pytree_flat = np.concatenate([result[b].reshape(-1) for b in cr.template_shapes])
    np.testing.assert_allclose(pytree_flat, flat_result)


def test_pytree_eval_fn_absent_for_scalar_path() -> None:
    """pytree_eval_fn is None when the scalar compile path is used.

    Only genuinely scalar specs (no axes declared) use the scalar path;
    axis-indexed specs that fail vectorization now raise instead of falling
    back.
    """
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "-x"}}
    cr = compile_rhs(normalize_rhs(spec))
    # Scalar spec → scalar path → pytree_eval_fn is None.
    assert cr.pytree_eval_fn is None


def test_compile_rhs_raises_for_axis_spec_that_fails_vectorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compile_rhs raises UnsupportedFeatureError for axis specs that cannot vectorize.

    Axis-indexed specs must not silently fall back to the slow scalar path.
    """
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["S[loc]"],
        "equations": {"S[loc]": "-S[loc]"},
    }
    rhs = normalize_rhs(spec)

    def _always_bail(_rhs: object) -> None:
        _bail("simulated vectorizer failure")

    monkeypatch.setattr(_vec, "build_vector_plan", _always_bail)

    with pytest.raises(UnsupportedFeatureError, match="vectorized eval path"):
        compile_rhs(rhs)


# ---------------------------------------------------------------------------
# template_shapes tests
# ---------------------------------------------------------------------------


def test_template_shapes_set_for_vectorized_path() -> None:
    """template_shapes maps each state base name to its N-D shape."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "loc", "coords": ["a", "b", "c"]},
        ],
        "state": ["S[age, loc]", "I[age, loc]", "R[age, loc]"],
        "equations": {
            "S[age, loc]": "-beta * S[age, loc]",
            "I[age, loc]": "beta * S[age, loc] - gamma * I[age, loc]",
            "R[age, loc]": "gamma * I[age, loc]",
        },
    }
    cr = compile_rhs(normalize_rhs(spec))
    assert cr.template_shapes is not None
    assert cr.template_shapes["S"] == (2, 3)
    assert cr.template_shapes["I"] == (2, 3)
    assert cr.template_shapes["R"] == (2, 3)


def test_template_shapes_preserved_after_pickle() -> None:
    """template_shapes survives a pickle round-trip."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["S[loc]", "I[loc]"],
        "equations": {
            "S[loc]": "-beta * S[loc]",
            "I[loc]": "beta * S[loc] - gamma * I[loc]",
        },
    }
    cr = compile_rhs(normalize_rhs(spec))
    restored = pickle.loads(pickle.dumps(cr))  # noqa: S301
    assert restored.template_shapes == cr.template_shapes
