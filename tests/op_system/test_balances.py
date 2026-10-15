"""Independent incidence/RHS references for conditional scalar balance artifacts."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_system import (
    AxisDeclaration,
    BalanceCoverageGap,
    ReactionBalanceLaw,
    ReactionBalanceReport,
    ReactionBalanceSource,
    StateDeclaration,
    array_namespace,
    compile_spec,
    reaction_balances,
)

if TYPE_CHECKING:
    from op_system import CompiledRhs


def _spec(*, sources: bool = False) -> dict[str, object]:
    transitions: list[dict[str, object]] = [
        {"name": "ab", "from": "a", "to": "b", "rate": "k"},
        {"name": "ba", "from": "b", "to": "a", "rate": "g"},
        {"name": "cd", "from": "c", "to": "d", "rate": "k * a"},
    ]
    if sources:
        transitions.extend([
            {"name": "in_a", "to": "a", "rate": "u"},
            {"name": "in_b", "to": "b", "rate": "v"},
        ])
    return {
        "kind": "transitions",
        "state": ["a", "b", "c", "d", "e"],
        "transitions": transitions,
    }


def test_exact_integer_basis_matches_independent_incidence_matrix() -> None:
    """Two transfer components and an isolated state span the full nullspace."""
    compiled = compile_spec(_spec())
    report = reaction_balances(compiled)
    expected = ((1, 1, 0, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 1))
    incidence = np.asarray([[-1, 1, 0], [1, -1, 0], [0, 0, -1], [0, 0, 1], [0, 0, 0]])
    assert report.fully_covered
    assert report.state_names == ("a", "b", "c", "d", "e")
    assert tuple(law.coefficients for law in report.laws) == expected
    np.testing.assert_array_equal(np.asarray(expected) @ incidence, np.zeros((3, 3)))
    assert np.linalg.matrix_rank(incidence) == 2
    assert np.linalg.matrix_rank(np.asarray(expected)) == 3
    assert all(
        type(weight) is int for law in report.laws for weight in law.coefficients
    )
    assert all(
        law.closed and law.origin == "compiled_unit_transfers" for law in report.laws
    )
    assert all(law.reaction_names == ("ab", "ba", "cd") for law in report.laws)
    # Reading another component in a propensity changes rates, not incidence.
    assert any(not reaction.reactants_complete for reaction in compiled.reactions)
    derivative = compiled.eval_fn(
        0.0, np.asarray([3.0, 1.0, 2.0, 0.5, 4.0]), k=0.2, g=0.3
    )
    np.testing.assert_allclose(derivative, [-0.3, 0.3, -1.2, 1.2, 0.0])
    np.testing.assert_allclose(np.asarray(expected) @ derivative, 0.0, atol=1e-15)


def test_open_sources_are_explicit_and_do_not_create_false_closed_totals() -> None:
    """A source breaks the old closed total and appears on the new balance RHS."""
    closed = reaction_balances(compile_spec(_spec()))
    compiled = compile_spec(_spec(sources=True))
    report = reaction_balances(compiled)
    assert report.fully_covered
    assert not report.laws[0].closed
    assert report.laws[0].sources == (
        ReactionBalanceSource("in_a", 1, "a"),
        ReactionBalanceSource("in_b", 1, "b"),
    )
    assert report.laws[1].closed
    assert report.laws[2].closed
    derivative = compiled.eval_fn(
        0.0, np.asarray([3.0, 1.0, 2.0, 0.5, 4.0]), k=0.2, g=0.3, u=0.5, v=0.4
    )
    assert np.asarray(closed.laws[0].coefficients) @ derivative == pytest.approx(0.9)
    np.testing.assert_allclose(
        np.asarray([law.coefficients for law in report.laws]) @ derivative,
        [0.9, 0.0, 0.0],
        atol=1e-15,
    )


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_balance_native_rhs_agreement(backend: str, dtype: str) -> None:
    """Exact static weights agree with independent source fluxes in native arrays."""
    module = (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )
    precision = nullcontext()
    if backend == "jax":
        jax = pytest.importorskip("jax")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        precision = enable()
    with precision:
        compiled = compile_spec(_spec(sources=True))
        report = reaction_balances(compiled)
        state = module.asarray([3.0, 1.0, 2.0, 0.5, 4.0], dtype=getattr(module, dtype))
        xp = array_namespace(state)
        params = {
            name: xp.asarray(value, dtype=state.dtype)
            for name, value in {"k": 0.2, "g": 0.3, "u": 0.5, "v": 0.4}.items()
        }
        derivative = compiled.eval_fn(0.0, state, **params)
        weights = xp.asarray(
            [law.coefficients for law in report.laws], dtype=state.dtype
        )
        totals = xp.matmul(weights, derivative)
        host = np.asarray(
            totals.detach().numpy() if hasattr(totals, "detach") else totals
        )
        np.testing.assert_allclose(
            host, [0.9, 0.0, 0.0], atol=2e-7 if dtype == "float32" else 1e-14
        )
        assert derivative.dtype == state.dtype
        assert array_namespace(derivative) is xp


@pytest.mark.parametrize("kind", ["expr", "unnamed", "templated", "routed"])
def test_unsupported_models_have_visible_coverage_and_no_laws(kind: str) -> None:
    """A covered reaction subset cannot stand in for the whole supplied model."""
    spec: dict[str, object]
    if kind == "expr":
        spec = {"kind": "expr", "state": ["x"], "equations": {"x": "-k * x"}}
    elif kind == "unnamed":
        spec = _spec() | {"transitions": [{"from": "a", "to": "b", "rate": "k"}]}
    else:
        spec = {
            "kind": "transitions",
            "axes": [{"name": "group", "coords": ["x", "y"]}],
            "state": ["a[group]", "b[group]"],
            "transitions": [
                {
                    "name": "move",
                    "from": "a[group:i]" if kind == "routed" else "a[group]",
                    "to": "b[group:j]" if kind == "routed" else "b[group]",
                    "rate": "k[group:i,group:j]" if kind == "routed" else "k",
                }
            ],
        }
    compiled = compile_spec(spec)
    report = reaction_balances(compiled)
    assert report.laws == ()
    assert not report.fully_covered
    assert report.coverage_gaps
    for gap in compiled.reaction_gaps:
        assert any(record.reaction_gap == gap for record in report.coverage_gaps)


@pytest.mark.parametrize("name", ["sources", "couplings", "constraints"])
def test_additional_declared_dynamics_prevent_issued_laws(name: str) -> None:
    """Even an empty unqualified reserved block is reported rather than ignored."""
    report = reaction_balances(compile_spec(_spec() | {name: []}))
    assert report.laws == ()
    assert (
        BalanceCoverageGap(name, "additional_declared_dynamics") in report.coverage_gaps
    )


def test_operator_boundary_flux_is_not_silently_set_to_zero() -> None:
    """An absorbing boundary invalidates a reaction-subset closed total."""
    compiled = compile_spec({
        "kind": "transitions",
        "axes": [{"name": "space", "type": "continuous", "coords": [0.0, 1.0]}],
        "state": ["a[space]", "b[space]"],
        "transitions": [
            {"name": "move", "from": "a[space]", "to": "b[space]", "rate": "k"}
        ],
        "operators": [
            {"kind": "diffusion", "axis": "space", "rate": "D", "bc": "dirichlet"}
        ],
    })
    report = reaction_balances(compiled)
    assert report.laws == ()
    assert any(gap.reason == "operator_dynamics" for gap in report.coverage_gaps)
    absorbing_generator = np.asarray([[-1.0, 1.0], [1.0, -2.0]])
    assert np.ones(2) @ absorbing_generator @ np.asarray([2.0, 3.0]) == pytest.approx(
        -3.0
    )


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "unknown_state", "non_scalar", "no_metadata", "no_layout"],
)
def test_inconsistent_artifacts_fail_closed(change: str) -> None:
    """Missing/changed identities cannot turn an incomplete model into an invariant."""
    compiled = compile_spec(_spec())
    if change == "missing":
        compiled = replace(compiled, reactions=compiled.reactions[:-1])
    elif change == "duplicate":
        compiled = replace(
            compiled, reactions=(*compiled.reactions, compiled.reactions[0])
        )
    elif change == "unknown_state":
        compiled = replace(
            compiled,
            reactions=(
                replace(compiled.reactions[0], to_base="absent"),
                *compiled.reactions[1:],
            ),
        )
    elif change == "non_scalar":
        compiled = replace(
            compiled,
            reactions=(
                replace(compiled.reactions[0], offsets=(("space", 1),)),
                *compiled.reactions[1:],
            ),
        )
    elif change == "no_metadata":
        compiled = replace(compiled, meta={})
    else:
        compiled = replace(compiled, template_shapes=None)
    report = reaction_balances(compiled)
    assert not report.fully_covered
    assert report.laws == ()


def test_scientific_metadata_preserves_compatible_and_unknown_units() -> None:
    """Separate components may have different units without inventing dimensions."""
    compiled = compile_spec(
        _spec()
        | {
            "state_metadata": {
                "a": {"representation": "counts", "units": "items"},
                "b": {"representation": "counts", "units": "items"},
                "c": {"representation": "cell_mass", "units": "kg"},
                "d": {"representation": "cell_mass", "units": "kg"},
            }
        }
    )
    report = reaction_balances(compiled)
    assert report.fully_covered
    assert dict(report.state_meta) == dict(compiled.state_meta)
    assert "e" not in dict(report.state_meta)
    assert report.assumptions
    assert all(law.assumptions == report.assumptions for law in report.laws)
    assert report.scope == "scalar_unit_transfers"


@pytest.mark.parametrize(
    ("declaration", "reason"),
    [
        (StateDeclaration("counts", "kg"), "incompatible_state_units"),
        (StateDeclaration("cumulative", "items"), "incompatible_state_representations"),
        (
            StateDeclaration("counts", "items", measure_axes=("space",)),
            "unsupported_state_measure",
        ),
    ],
)
def test_known_scientific_conflicts_prevent_laws(
    declaration: StateDeclaration, reason: str
) -> None:
    """Known label/measure conflicts cannot be hidden behind generic assumptions."""
    compiled = compile_spec(_spec())
    declarations = {"a": StateDeclaration("counts", "items"), "b": declaration}
    report = reaction_balances(replace(compiled, state_meta=declarations))
    assert report.laws == ()
    assert not report.fully_covered
    assert reason in {gap.reason for gap in report.coverage_gaps}


def test_report_is_a_detached_immutable_pickleable_record() -> None:
    """Artifacts carry no callbacks or mutable source dictionaries."""
    states = {"a": StateDeclaration("counts", "items")}
    axes = {"time": AxisDeclaration("day")}
    compiled = replace(compile_spec(_spec()), state_meta=states, axis_meta=axes)
    report = reaction_balances(compiled)
    states["a"] = StateDeclaration("cell_mass", "kg")
    axes.clear()
    assert dict(report.state_meta)["a"].units == "items"
    assert report.axis_meta == (("time", AxisDeclaration("day")),)
    with pytest.raises(AttributeError):
        report.laws = ()  # type: ignore[misc]
    restored = pickle.loads(pickle.dumps(report))  # ruff: ignore[suspicious-pickle-usage]
    assert isinstance(restored, ReactionBalanceReport)
    assert isinstance(restored.laws[0], ReactionBalanceLaw)
    assert restored == report
    assert hash(restored) == hash(report)


def test_invalid_analysis_input_has_a_specific_error() -> None:
    """The API expects a compiled container rather than a raw specification."""
    with pytest.raises(TypeError, match="CompiledRhs"):
        reaction_balances(cast("CompiledRhs", object()))
