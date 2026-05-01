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

import re

import numpy as np
import pytest

from op_system import compile_spec
from op_system.compile import CompiledRhs, compile_rhs
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
    return compile_rhs(rhs_xy)


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
    compiled = compile_rhs(rhs)

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
    compiled = compile_rhs(rhs)

    msg = "state has an invalid shape/value. Expected (n_state=2,)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        compiled.eval_fn(np.float64(0.0), np.zeros((3,), dtype=np.float64))


def test_eval_unknown_symbol_raises_parameter_error() -> None:
    """eval_fn raises parameter error for unknown symbols in expressions."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "beta * x"}}
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

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
        compile_rhs(rhs)


def test_compile_rejects_nonwhitelisted_helper() -> None:
    """Bare helper calls must be whitelisted; unknown helpers are rejected."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "foo(x)"}}
    rhs = normalize_rhs(spec)
    with pytest.raises(ValueError, match=r"disallowed function call"):
        compile_rhs(rhs)


def test_compile_rejects_disallowed_attribute_access() -> None:
    """Disallowed attribute access should be rejected."""
    spec = {"kind": "expr", "state": ["x"], "equations": {"x": "x.real"}}
    rhs = normalize_rhs(spec)
    with pytest.raises(ValueError, match=r"disallowed attribute access"):
        compile_rhs(rhs)


def test_eval_allows_whitelisted_np_calls() -> None:
    """A small set of np.* calls is whitelisted and should work."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "np.maximum(x, 0.0)"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)
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
    compiled = compile_rhs(rhs)
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


def test_templates_and_sum_over_compile_and_eval() -> None:
    """Templated states with sum_over compile and evaluate end-to-end."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1", "p2"]}],
        "state": ["S[pop]", "I[pop]"],
        "equations": {
            "S[pop]": "-beta * S[pop] * sum_over(pop=j, I[pop=j])",
            "I[pop]": "beta * S[pop] * sum_over(pop=j, I[pop=j]) - gamma * I[pop]",
        },
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

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
    compiled = compile_rhs(rhs)
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
    compiled = compile_rhs(rhs)
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
    compiled = compile_rhs(rhs)
    with pytest.raises(ValueError, match=r"could not resolve alias dependencies"):
        compiled.eval_fn(np.float64(0.0), np.array([1.0], dtype=np.float64))


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
    compiled = compile_rhs(rhs)

    y = np.array([999.0, 1.0, 0.0], dtype=np.float64)
    out = compiled.eval_fn(np.float64(0.0), y, beta=0.3, gamma=0.1)

    n_total = float(np.sum(y))
    inf = 0.3 * float(y[1]) / n_total * float(y[0])
    rec = 0.1 * float(y[1])
    expected = np.array([-inf, inf - rec, rec], dtype=np.float64)
    assert np.allclose(out, expected)


# ---------------------------------------------------------------------------
# Meta threading (#11)
# ---------------------------------------------------------------------------


def test_compiled_rhs_meta_from_spec_with_axes_and_kernels() -> None:
    """compile_rhs preserves axes and kernels metadata from NormalizedRhs."""
    spec: dict[str, object] = {
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["a0", "a1", "a2"]}],
        "state": ["S", "I"],
        "kernels": [
            {
                "name": "contact",
                "form": "gaussian",
                "params": {"scale": 1.0, "sigma": 0.5},
            },
        ],
        "equations": {"S": "-beta * S", "I": "beta * S"},
    }
    rhs = normalize_rhs(spec)
    compiled = compile_rhs(rhs)

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
        "state": ["x"],
        "equations": {"x": "a * x"},
    }
    compiled = compile_spec(spec)

    assert "axes" in compiled.meta
    assert compiled.meta["axes"][0]["name"] == "space"


def test_sum_over_in_filter_evaluates_correctly() -> None:
    """sum_over IN filter compiles and evaluates correctly end-to-end."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "vax", "coords": ["u", "v", "w"]}],
        "state": ["S[vax]"],
        "equations": {
            # Constant-rate decay so eval_fn output is predictable
            "S[vax]": "-S[vax]",
        },
        "aliases": {
            "covered": "sum_over(vax=j IN [v, w], S[vax=j])",
        },
    }
    rhs = normalize_rhs(spec)

    # NormalizedRhs.aliases holds the expanded string
    assert "S__vax_v" in rhs.aliases["covered"]
    assert "S__vax_w" in rhs.aliases["covered"]
    assert "S__vax_u" not in rhs.aliases["covered"]

    # Compile and run eval_fn: state = [u=10, v=3, w=7], dS/dt = -S
    compiled = compile_rhs(rhs)
    state = np.array([10.0, 3.0, 7.0], dtype=np.float64)
    derivs = compiled.eval_fn(np.float64(0.0), state)
    assert np.allclose(derivs, -state)


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
        ).ys[0]

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
