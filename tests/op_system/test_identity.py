"""Structural identities preserve compiler meaning and bounded IR sharing."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle  # ruff: ignore[suspicious-pickle-import]
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from op_system import (
    StructuralModelIdentity,
    compile_rhs,
    compile_spec,
    normalize_rhs,
    structural_model_json,
)
from op_system._ir import Apply, Literal, Sym
from op_system.identity import _ManifestEncoder


def _model() -> dict[str, Any]:
    """Return independent linear resources on an ordered named axis.

    Returns:
        Public synthetic specification with declared latent observables.
    """
    return {
        "kind": "expr",
        "axes": [{"name": "site", "coords": ["west", "east"]}],
        "state": ["x[site]", "z[site]"],
        "shaped_params": {"rate": ["site"]},
        "equations": {"x[site]": "-rate[site]*x[site]", "z[site]": "2*x[site]"},
        "observables": {"twice": {"expr": "2*x[site]", "axes": ["site"]}},
    }


def test_golden_manifest_and_independent_compilations() -> None:
    """A schema change must deliberately update this public format fixture."""
    spec: dict[str, Any] = {"kind": "expr", "state": ["x"], "equations": {"x": "-k*x"}}
    compiled = compile_spec(spec)
    identity = StructuralModelIdentity.from_compiled(compiled)
    assert (
        identity.digest
        == "sha256:10de1c2ec77155c507cfe5290e73a03635bb29ed2963db800915ca7d4f007ebd"
    )
    manifest = structural_model_json(compiled)
    assert identity.digest == f"sha256:{hashlib.sha256(manifest.encode()).hexdigest()}"
    assert identity == StructuralModelIdentity.from_compiled(
        compile_spec(copy.deepcopy(spec))
    )
    spec["equations"]["x"] = " - k * (x) "
    assert identity == StructuralModelIdentity.from_compiled(compile_spec(spec))
    assert json.loads(manifest)["schema_version"] == 1


def test_roundtrips_runtime_binding_and_python_equality_are_separate() -> None:
    """No fitted value is captured, and compiler pickles retain their identity."""
    compiled = compile_spec(_model())
    expected = StructuralModelIdentity.from_compiled(compiled)
    assert StructuralModelIdentity.from_json(expected.to_json()) == expected
    rebuilt = pickle.loads(pickle.dumps(compiled))  # ruff: ignore[suspicious-pickle-usage]
    assert StructuralModelIdentity.from_compiled(rebuilt) == expected
    x = np.array([2.0, 4.0, 0.0, 0.0])
    np.testing.assert_allclose(
        compiled.bind({"rate": np.array([1.0, 2.0])})(0, x), [-2, -8, 4, 8]
    )
    np.testing.assert_allclose(
        compiled.bind({"rate": np.array([3.0, 4.0])})(0, x), [-6, -16, 4, 8]
    )
    assert StructuralModelIdentity.from_compiled(compiled) == expected
    assert compiled.eval_fn is not rebuilt.eval_fn
    with pytest.raises(FrozenInstanceError):
        expected.digest = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "change",
    [
        "state_order",
        "coordinate_order",
        "equation",
        "constant",
        "observable",
        "observable_order",
        "factorize",
        "initial_constant",
        "state_coordinate",
    ],
)
def test_meaningful_changes_have_different_identities(change: str) -> None:
    """Each case changes a real binding/layout, declared output or equation."""
    spec = _model()
    expected = StructuralModelIdentity.from_compiled(compile_spec(spec))
    if change == "state_order":
        spec["state"].reverse()
    elif change == "coordinate_order":
        spec["axes"][0]["coords"].reverse()
    elif change == "equation":
        spec["equations"]["z[site]"] = "x[site]+z[site]"
    elif change == "constant":
        spec["equations"]["z[site]"] = "3*x[site]"
    elif change == "observable":
        spec["observables"]["twice"]["expr"] = "3*x[site]"
    elif change == "observable_order":
        spec["observables"] = {
            "another": {"expr": "z[site]", "axes": ["site"]},
            **spec["observables"],
        }
    elif change == "factorize":
        spec["factorize_axes"] = ["site"]
    elif change == "initial_constant":
        spec["initial_state"] = {"x[site]": 1.0, "z[site]": 0.0}
    else:
        spec["state_metadata"] = {"x": {"scale": "log"}}
    assert StructuralModelIdentity.from_compiled(compile_spec(spec)) != expected


def test_parameter_binding_order_is_not_sorted_away() -> None:
    """Same named parameter set with a different ordered layout stays distinct."""
    rhs = normalize_rhs({"kind": "expr", "state": ["x"], "equations": {"x": "a*x+b"}})
    lhs = StructuralModelIdentity.from_compiled(compile_rhs(rhs))
    rhs_reordered = replace(rhs, param_names=tuple(reversed(rhs.param_names)))
    assert StructuralModelIdentity.from_compiled(compile_rhs(rhs_reordered)) != lhs


def test_operator_boundary_and_kernel_constants_are_structural() -> None:
    """Same reaction equation with a different transport contract cannot be reused."""
    spec: dict[str, Any] = {
        "kind": "expr",
        "axes": [{"name": "cell", "type": "ordinal", "coords": ["a", "b"]}],
        "state": ["x[cell]"],
        "equations": {"x[cell]": "0.0"},
        "operators": [
            {"axis": "cell", "kind": "advection", "velocity": 0.5, "bc": "absorbing"}
        ],
    }
    original = StructuralModelIdentity.from_compiled(compile_spec(spec))
    spec["operators"][0]["bc"] = "periodic"
    periodic = StructuralModelIdentity.from_compiled(compile_spec(spec))
    spec["operators"][0]["velocity"] = 0.75
    faster = StructuralModelIdentity.from_compiled(compile_spec(spec))
    assert len({original, periodic, faster}) == 3


def test_stochastic_channel_metadata_survives_identical_summed_equations() -> None:
    """Two named channels differ from one even when deterministic dynamics match."""
    spec: dict[str, Any] = {
        "kind": "transitions",
        "state": ["x", "z"],
        "transitions": [{"name": "move", "from": "x", "to": "z", "rate": "k"}],
    }
    first = compile_spec(spec)
    spec["transitions"][0]["name"] = "renamed_channel"
    second = compile_spec(spec)
    np.testing.assert_array_equal(
        first.eval_fn(0, np.array([2.0, 0.0]), k=3.0),
        second.eval_fn(0, np.array([2.0, 0.0]), k=3.0),
    )
    assert StructuralModelIdentity.from_compiled(
        first
    ) != StructuralModelIdentity.from_compiled(second)


def test_identity_replays_in_fresh_processes_with_different_hash_seeds() -> None:
    """No salted Python hash or object address affects the portable identity."""
    program = (
        "from op_system import compile_spec,StructuralModelIdentity; "
        "print(StructuralModelIdentity.from_compiled(compile_spec("
        + repr(_model())
        + ")).to_json())"
    )
    outputs = [
        subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, "-c", program],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for seed in ["7", "12345"]
    ]
    assert outputs[0] == outputs[1]


def test_mapping_insertion_order_is_not_a_layout() -> None:
    """Mapping key order cannot alter an explicitly recorded layout."""
    spec = _model()
    expected = StructuralModelIdentity.from_compiled(compile_spec(spec))
    spec["equations"] = dict(reversed(list(spec["equations"].items())))
    assert StructuralModelIdentity.from_compiled(compile_spec(spec)) == expected


def test_observable_order_survives_mapping_canonicalization() -> None:
    """Same two outputs in opposite declared order retain different identities."""
    spec = _model()
    spec["observables"]["another"] = {"expr": "z[site]", "axes": ["site"]}
    expected = StructuralModelIdentity.from_compiled(compile_spec(spec))
    spec["observables"] = dict(reversed(list(spec["observables"].items())))
    assert StructuralModelIdentity.from_compiled(compile_spec(spec)) != expected


def test_gauge_options_and_anchors_are_structural() -> None:
    """Gauge coordinates change parameter interpretation and must invalidate reuse."""
    spec: dict[str, Any] = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "a*x*(1-x/K)"},
        "observables": {"value": {"expr": "c*x"}},
        "gauges": [
            {"name": "scale", "states": {"x": 1}, "parameters": {"K": 1, "c": -1}}
        ],
    }
    models = [
        compile_spec(spec),
        compile_spec(spec, detect_gauges=True),
        compile_spec(spec, gauge_fix={"K": 1.0}),
        compile_spec(spec, gauge_fix={"K": 2.0}),
    ]
    assert len({StructuralModelIdentity.from_compiled(model) for model in models}) == 4


@pytest.mark.parametrize("version", [None, True, 1.0, "1", 0, 2])
def test_unknown_or_coerced_schema_is_refused(version: object) -> None:
    """Legacy and future identities are never silently assigned schema 1."""
    payload = json.dumps({"schema_version": version, "digest": "sha256:" + "0" * 64})
    with pytest.raises(ValueError, match="schema version"):
        StructuralModelIdentity.from_json(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        "[]",
        '{"digest":"sha256:abc"}',
        '{"schema_version":1,"digest":"sha256:abc"}',
        '{"schema_version":1,"digest":null}',
        '{"schema_version":1,"digest":"sha256:abc","extra":1}',
    ],
)
def test_malformed_reference_is_refused(payload: str) -> None:
    """Only a complete known reference can enter downstream expected provenance."""
    with pytest.raises(ValueError, match=r"Structural model identity"):
        StructuralModelIdentity.from_json(payload)


def test_manifest_preserves_typed_scalars_without_expanding_shared_trees() -> None:
    """A deep shared binary DAG stays linear and does not consume call-stack depth."""
    encoder = _ManifestEncoder()
    expression: Any = Sym("x")
    for _ in range(2000):
        expression = Apply("+", (expression, expression))
    encoder.encode(expression)
    assert len(encoder.nodes) < 5000
    identities = []
    values: list[bool | int | float | str] = [True, 1, 1.0, -0.0, 0.0, "1"]
    for value in values:
        current = _ManifestEncoder()
        current.encode(Literal(value))
        identities.append(json.dumps(current.nodes))
    assert len(set(identities)) == 6


def test_structural_interning_does_not_depend_on_object_sharing() -> None:
    """Independently allocated equal nodes produce the same canonical table."""
    shared = Sym("x")
    encoders = [_ManifestEncoder(), _ManifestEncoder()]
    roots = [
        encoder.encode(value)
        for encoder, value in zip(
            encoders,
            [Apply("+", (shared, shared)), Apply("+", (Sym("x"), Sym("x")))],
            strict=True,
        )
    ]
    assert roots[0] == roots[1]
    assert encoders[0].nodes == encoders[1].nodes


@pytest.mark.parametrize("bad", [object(), np.array([1.0]), {1: "bad"}])
def test_unsupported_metadata_is_not_silently_ignored(bad: object) -> None:
    """No implicit device/array snapshots or unstable object repr enter identity."""
    rhs = normalize_rhs({"kind": "expr", "state": ["x"], "equations": {"x": "x"}})
    compiled = compile_rhs(replace(rhs, meta={**rhs.meta, "extra": bad}))
    with pytest.raises(TypeError, match=r"structural|Structural"):
        StructuralModelIdentity.from_compiled(compiled)


def test_cyclic_metadata_fails_without_looping() -> None:
    """A user cycle is rejected rather than growing an unbounded traversal."""
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(TypeError, match="cycles"):
        _ManifestEncoder().encode(cycle)


def test_identity_does_not_materialize_lazy_scalar_ir() -> None:
    """Vector model identity leaves deferred per-cell scalar expansions alone."""
    rhs = normalize_rhs(_model())
    compiled = compile_rhs(rhs)
    cache = getattr(rhs.equations_ir, "_cache", None)
    assert isinstance(cache, dict)
    before = set(cache)
    StructuralModelIdentity.from_compiled(compiled)
    assert set(cache) == before


def test_pinned_reaction_constants_and_manual_container_policy() -> None:
    """Synthesized masks are recorded; missing legacy/manual source is refused."""
    spec: dict[str, Any] = {
        "kind": "transitions",
        "axes": [{"name": "site", "coords": ["a", "b"]}],
        "state": ["x[site]"],
        "transitions": [
            {"name": "move", "from": "x[site=a]", "to": "x[site=b]", "rate": "k"}
        ],
    }
    compiled = compile_spec(spec)
    assert StructuralModelIdentity.from_compiled(
        compiled
    ) == StructuralModelIdentity.from_compiled(compile_spec(copy.deepcopy(spec)))
    with pytest.raises(ValueError, match="typed normalized IR"):
        StructuralModelIdentity.from_compiled(replace(compiled, _rhs=None))
