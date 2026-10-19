"""Independent scaling equations, invariant outputs and native quotient maps."""

from __future__ import annotations

import copy
import json
import pickle  # ruff: ignore[suspicious-pickle-import]
import runpy
from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_system import compile_spec, normalize_rhs, validate_spec
from op_system._errors import InvalidRhsSpecError, UnsupportedFeatureError
from op_system._gauge_algebra import _integer_nullspace, _rank

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray


@contextmanager
def _backend(name: str) -> Iterator[Any]:
    """Require the selected backend and scope JAX double precision.

    Yields:
        The selected native namespace.
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
    """Copy native test values after differentiation.

    Returns:
        A NumPy array for independent numerical assertions.
    """
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def model() -> dict[str, Any]:
    """Two independent logistic resources with a weighted observed sum.

    Returns:
        Synthetic equations, observed sum and two scaling generators.
    """
    return {
        "kind": "expr",
        "state": ["x", "z"],
        "equations": {"x": "a*x*(1-x/K)", "z": "b*z*(1-z/H)"},
        "observables": {"weighted_sum": {"expr": "c*x+d*z"}},
        "gauges": [
            {"name": "first", "states": {"x": 1}, "parameters": {"K": 1, "c": -1}},
            {
                "name": "joint",
                "states": {"x": 1, "z": 1},
                "parameters": {"K": 1, "H": 1, "c": -1, "d": -1},
            },
        ],
    }


def test_default_is_unanalyzed_and_declarations_are_detached() -> None:
    """Ordinary compilation pays no analysis cost and does not claim zero gauges."""
    spec = model()
    undeclared = copy.deepcopy(spec)
    del undeclared["gauges"]
    assert compile_spec(undeclared).gauges is None
    normalized = normalize_rhs(spec)
    spec["gauges"][0]["states"]["x"] = 9
    assert normalized.meta["op_system_gauge_declarations"][0].states == (("x", 1.0),)


def test_declared_detection_and_parameter_invariants() -> None:
    """The two group freedoms leave a,b,K*c,H*d invariant."""
    compiled = compile_spec(model(), detect_gauges=True)
    info = compiled.gauges
    assert info is not None
    assert info.rank == info.parameter_rank == len(info.detected) == 2
    assert info.verification is not None
    assert info.verification.structural
    assert info.verification.observables == ("weighted_sum",)
    assert info.verification.n_probes == 4
    assert info.verification.max_absolute_error < 1e-12
    params = {"a": 0.5, "b": 0.25, "K": 12.0, "H": 20.0, "c": 3.0, "d": 4.0}
    transformed = {"a": 0.5, "b": 0.25, "K": 72.0, "H": 60.0, "c": 0.5, "d": 4 / 3}
    assert len(info.invariant_coordinates) == 4
    for invariant in info.invariant_coordinates:
        assert invariant.evaluate(params) == pytest.approx(
            invariant.evaluate(transformed)
        )


def test_nonorthogonal_anchor_map_output_covariance_and_pickle() -> None:
    """Independent known representative: x/12,z/20,c*12,d*20."""
    spec = model()
    anchors = {"K": 1.0, "H": 1.0}
    compiled = compile_spec(spec, gauge_fix=anchors)
    anchors["K"] = 999.0
    assert compiled.gauge_fix is not None
    params = {"a": 0.5, "b": 0.25, "K": 12.0, "H": 20.0, "c": 3.0, "d": 4.0}
    state = np.array([6.0, 10.0])
    coordinates = compiled.gauge_fix.encode(state, params)
    np.testing.assert_allclose(_host(coordinates.state), [0.5, 0.5])
    assert coordinates.parameters == pytest.approx({
        "a": 0.5,
        "b": 0.25,
        "c": 36.0,
        "d": 80.0,
    })
    assert bool(coordinates.valid)
    assert set(compiled.param_names) == {"a", "b"}
    # c,d are observable-only parameters, not equation arguments.
    assert set(dict(compiled.observables["weighted_sum"].parameter_shapes)) == {
        "c",
        "d",
    }
    np.testing.assert_allclose(
        compiled.eval_fn(0, coordinates.state, **coordinates.parameters),
        [0.125, 0.0625],
    )
    assert compiled.observables["weighted_sum"].eval_fn(
        0, coordinates.state, **coordinates.parameters
    ) == pytest.approx(58.0)
    original_state, original_parameters = compiled.gauge_fix.decode(coordinates)
    np.testing.assert_allclose(_host(original_state), state)
    assert original_parameters == pytest.approx(params)
    assert params["K"] == 12
    np.testing.assert_array_equal(state, [6.0, 10.0])
    rebuilt = pickle.loads(pickle.dumps(compiled))  # ruff: ignore[suspicious-pickle-usage]
    assert rebuilt.gauge_fix == compiled.gauge_fix
    assert rebuilt.param_names == compiled.param_names
    np.testing.assert_allclose(
        rebuilt.eval_fn(0, coordinates.state, **coordinates.parameters), [0.125, 0.0625]
    )
    with pytest.raises(TypeError, match="override gauge-fixed"):
        rebuilt.eval_fn(0, coordinates.state, K=1, a=0.5, b=0.25)


@pytest.mark.parametrize("failure", ["rhs", "observable"])
def test_invalid_declared_covariance_or_output_invariance_fails(failure: str) -> None:
    """Covariance alone is insufficient when an observed output changes."""
    spec = model()
    if failure == "rhs":
        spec["equations"]["x"] += "+1"
    else:
        spec["observables"]["weighted_sum"]["expr"] = "x+z+c*x+d*z"
    with pytest.raises(InvalidRhsSpecError, match="Gauge failed"):
        compile_spec(spec)


def test_duplicate_generators_report_rank_not_declaration_count() -> None:
    """Repeating a valid generator does not remove another dimension."""
    spec = model()
    duplicate = copy.deepcopy(spec["gauges"][0])
    duplicate["name"] = "repeated"
    spec["gauges"].append(duplicate)
    info = compile_spec(spec).gauges
    assert info is not None
    assert len(info.declared) == 3
    assert info.rank == 2


@pytest.mark.parametrize("bad", [True, float("inf"), "1"])
def test_exponents_require_finite_real_numbers(bad: object) -> None:
    """Malformed exponent declarations fail before execution."""
    spec = model()
    spec["gauges"][0]["states"]["x"] = bad
    with pytest.raises(InvalidRhsSpecError, match="finite real"):
        compile_spec(spec)


@pytest.mark.parametrize(
    "anchors", [{"K": 1}, {"K": 0, "H": 1}, {"K": 1, "a": 1}, {"unknown": 1, "H": 1}]
)
def test_anchor_count_domain_and_independence(anchors: dict[str, float]) -> None:
    """A positive independent anchor must remove each reported direction."""
    with pytest.raises(InvalidRhsSpecError, match="Gauge"):
        compile_spec(model(), gauge_fix=anchors)


def test_zero_anchor_inputs_flag_invalid_without_clamping() -> None:
    """The native map flags an invalid domain instead of inventing a floor."""
    compiled = compile_spec(model(), gauge_fix={"K": 1, "H": 1})
    assert compiled.gauge_fix is not None
    with np.errstate(divide="ignore", invalid="ignore"):
        result = compiled.gauge_fix.encode(
            np.array([1.0, 1.0]),
            {"K": 0.0, "H": 2.0, "a": 0.5, "b": 0.25, "c": 1.0, "d": 1.0},
        )
    assert not bool(result.valid)
    with pytest.raises(TypeError, match="real floating"):
        compiled.gauge_fix.encode(np.array([1, 1]), {"K": 1, "H": 1})
    with pytest.raises(ValueError, match="scalar"):
        compiled.gauge_fix.encode(
            np.array([1.0, 1.0]), {"K": np.array([1.0, 2.0]), "H": np.array([1.0, 2.0])}
        )


def test_shared_template_scale_and_observable_only_anchor() -> None:
    """One shared scale covers both cells and a scalar output denominator."""
    spec: dict[str, Any] = {
        "kind": "expr",
        "axes": [{"name": "cell", "type": "categorical", "coords": ["a", "b"]}],
        "state": ["x[cell]"],
        "equations": {"x[cell]": "rate*x[cell]"},
        "observables": {"ratio": {"expr": "x[cell]/K", "axes": ["cell"]}},
        "gauges": [{"name": "units", "states": {"x": 1}, "parameters": {"K": 1}}],
    }
    compiled = compile_spec(spec, gauge_fix={"K": 1})
    assert compiled.gauge_fix is not None
    mapped = compiled.gauge_fix.encode(np.array([3.0, 6.0]), {"K": 3.0, "rate": 0.5})
    np.testing.assert_allclose(_host(mapped.state), [1.0, 2.0])
    assert compiled.pytree_eval_fn is not None
    np.testing.assert_allclose(
        compiled.pytree_eval_fn(0, {"x": _host(mapped.state)}, **mapped.parameters)[
            "x"
        ],
        [0.5, 1.0],
    )
    assert compiled.observables["ratio"].parameter_shapes == ()
    np.testing.assert_allclose(
        _host(
            compiled.observables["ratio"].pytree_eval_fn(0, {"x": _host(mapped.state)})
        ),
        [1.0, 2.0],
    )


def test_known_initial_conditions_are_not_silently_rescaled() -> None:
    """Without original parameter values compile cannot transform a preset."""
    spec = model()
    spec["initial_state"] = {"x": "seed_x", "z": "seed_z"}
    with pytest.raises(InvalidRhsSpecError, match="encode initial states"):
        compile_spec(spec, gauge_fix={"K": 1, "H": 1})


def test_stochastic_jumps_are_not_claimed_in_rescaled_units() -> None:
    """Deterministic state covariance does not rescale a CTMC's integer jumps."""
    spec: dict[str, Any] = {
        "kind": "transitions",
        "state": ["x", "z"],
        "transitions": [{"name": "transfer", "from": "x", "to": "z", "rate": "a"}],
        "observables": {"fraction": {"expr": "z/K"}},
        "gauges": [
            {"name": "units", "states": {"x": 1, "z": 1}, "parameters": {"K": 1}}
        ],
    }
    natural = compile_spec(spec)
    assert len(natural.reactions) == 1
    fixed = compile_spec(spec, gauge_fix={"K": 1})
    assert fixed.reactions == ()
    assert any(
        gap.reason == "gauge_fix_requires_jump_contract" for gap in fixed.reaction_gaps
    )
    np.testing.assert_allclose(
        fixed.eval_fn(0, np.array([0.75, 0.25]), a=2.0), [-1.5, 1.5]
    )


def test_exact_rank_and_nullspace_known_matrix() -> None:
    """The rows (1,2,0) and (0,1,1) annihilate (2,-1,1)."""
    assert _rank([[1, 2, 0], [0, 1, 1], [2, 4, 0]], 3) == 2
    assert _integer_nullspace([[1, 2, 0], [0, 1, 1]], 3) == ((2, -1, 1),)
    assert _integer_nullspace([], 2) == ((1, 0), (0, 1))
    assert _integer_nullspace([[]], 0) == ()


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_native_maps_output_and_derivatives(backend: str) -> None:
    """The observed sum 3*x+4*z and its derivatives survive the quotient."""
    with _backend(backend) as xp:
        make_array = xp.asarray if backend != "torch" else xp.tensor
        compiled = compile_spec(model(), gauge_fix={"K": 1.0, "H": 1.0})
        mapping = compiled.gauge_fix
        assert mapping is not None

        def output(state: Any) -> Any:  # ruff: ignore[any-type]
            coordinates = mapping.encode(
                state, {"K": 12.0, "H": 20.0, "a": 0.5, "b": 0.25, "c": 3.0, "d": 4.0}
            )
            return compiled.observables["weighted_sum"].eval_fn(
                0.0, coordinates.state, **coordinates.parameters
            )

        state = make_array([6.0, 10.0], dtype=xp.float64)
        assert float(output(state)) == pytest.approx(58.0, rel=1e-6)
        if backend == "jax":
            jax = import_module("jax")
            np.testing.assert_allclose(jax.jit(jax.grad(output))(state), [3.0, 4.0])
            np.testing.assert_allclose(
                jax.jit(jax.vmap(output))(xp.stack([state, 2 * state])), [58.0, 116.0]
            )
        elif backend == "torch":
            state.requires_grad_()
            output(state).backward()
            np.testing.assert_allclose(state.grad.numpy(), [3.0, 4.0])


def test_restricted_domains_are_not_reported_as_global_scales() -> None:
    """A fixed or bounded scale parameter removes that global direction."""
    spec = model()
    spec["parameters"] = {"K": {"support": "unit_interval"}}
    with pytest.raises(InvalidRhsSpecError, match="fixed/bounded"):
        compile_spec(spec)
    del spec["gauges"]
    info = compile_spec(spec, detect_gauges=True).gauges
    assert info is not None
    assert info.rank == 1


def test_unsupported_history_does_not_receive_a_partial_certificate() -> None:
    """The instantaneous evaluator alone cannot certify a history model."""
    spec = model()
    # Explicit compiler metadata cannot be ignored by verification.
    compiled = compile_spec(spec)
    assert compiled.gauges is not None
    spec["equations"]["x"] = "delay(x, lag=tau)"
    with pytest.raises((UnsupportedFeatureError, InvalidRhsSpecError)):
        compile_spec(spec)


def test_validation_counts_are_not_an_identifiability_claim() -> None:
    """Six scalar parameters lose two reported scaling freedoms."""
    report = validate_spec(model(), gauge_fix={"K": 1, "H": 1})
    assert report.ok
    assert report.gauges is not None
    assert report.gauges["rank"] == 2
    assert report.gauges["free_reported_scaling_directions"] == 0
    assert report.gauges["parameter_coordinate_count"] == 6
    assert report.gauges["parameter_coordinates_after_reported_action"] == 4
    assert report.gauges["identifiability_assessed"] is False
    assert set(report.parameters) == {"a", "b"}


def test_public_example_executes() -> None:
    """The guide's exact representative and output checks execute in CI."""
    runpy.run_path("examples/scaling_gauges.py")


def test_drift_cancellation_does_not_certify_jump_distribution() -> None:
    """Opposite balanced channels have zero drift but rate-dependent turnover."""
    spec: dict[str, Any] = {
        "kind": "transitions",
        "state": ["x", "z"],
        "transitions": [
            {"name": "forward", "from": "x", "to": "z", "rate": "a/x"},
            {"name": "backward", "from": "z", "to": "x", "rate": "a/z"},
        ],
        "observables": {"sum": {"expr": "x+z"}},
        "gauges": [{"name": "turnover", "parameters": {"a": 1}}],
    }
    natural = compile_spec(spec)
    assert len(natural.reactions) == 2
    np.testing.assert_allclose(
        natural.eval_fn(0, np.array([2.0, 3.0]), a=7.0), [0.0, 0.0], atol=1e-14
    )
    fixed = compile_spec(spec, gauge_fix={"a": 1})
    assert not fixed.reactions
    assert len(fixed.reaction_gaps) == 2


def test_ordinary_native_math_detection() -> None:
    """A ratio inside exp is scale invariant, not each individual argument."""
    spec: dict[str, Any] = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "rate*x*np.exp(-x/K)"},
        "observables": {"ratio": {"expr": "x/K"}},
    }
    info = compile_spec(spec, detect_gauges=True).gauges
    assert info is not None
    assert info.rank == 1


def test_cli_reports_counts_and_removes_fixed_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The JSON report keeps original gauge ranks and labels the fixed action."""
    from op_system.validate import main  # ruff: ignore[import-outside-top-level]

    source = tmp_path / "model.json"
    source.write_text(json.dumps(model()))
    assert (
        main([
            str(source),
            "--json",
            "--detect-gauges",
            "--gauge-fix",
            "K=1",
            "--gauge-fix",
            "H=1",
        ])
        == 0
    )
    report = next(iter(json.loads(capsys.readouterr().out).values()))
    assert report["gauges"]["rank"] == 2
    assert report["gauges"]["parameter_coordinates_after_reported_action"] == 4
    assert report["gauges"]["identifiability_assessed"] is False
    assert set(report["parameters"]) == {"a", "b"}


@pytest.mark.parametrize("value", ["K", "K=no", "=1"])
def test_cli_rejects_malformed_anchor(value: str) -> None:
    """Malformed anchors fail in argparse before any file is opened."""
    from op_system.validate import main  # ruff: ignore[import-outside-top-level]

    with pytest.raises(SystemExit) as error:
        main(["not-opened.yml", "--gauge-fix", value])
    assert error.value.code == 2
