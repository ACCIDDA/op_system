"""Execute the authoring guide's exact snippets against independent cell flows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml  # type: ignore[import-untyped]

from op_system import compile_spec, validate_spec

if TYPE_CHECKING:
    import pytest

    from op_system import CompiledRhs

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs/guides/getting-started.md"


def _snippet(marker: str, language: str) -> str:
    text = GUIDE.read_text()
    match = re.search(rf"<!-- {marker} -->\n```{language}\n(.*?)\n```", text, re.DOTALL)
    assert match is not None
    return match.group(1)


def test_complete_guide_against_independent_seir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual YAML/Python, both layouts and equivalent equations satisfy the oracle."""
    monkeypatch.chdir(ROOT)
    guide_spec = yaml.safe_load(_snippet("seir-spec", "yaml"))
    assert guide_spec == yaml.safe_load((ROOT / "examples/age-seir.yaml").read_text())
    namespace: dict[str, Any] = {}
    exec(_snippet("seir-evaluation", "python"), namespace)  # ruff: ignore[exec-builtin]
    compiled: CompiledRhs = namespace["compiled"]
    state = namespace["state"]
    params = namespace["params"]
    rhs = namespace["rhs"]
    population = [
        sum(float(state[name][age]) for name in ("S", "E", "I", "R"))
        for age in range(2)
    ]
    expected = {name: np.zeros(2) for name in state}
    for target in range(2):
        prevalence_mix = sum(
            float(params["C"][source, target])
            * float(state["I"][source])
            / population[source]
            for source in range(2)
        )
        infection = (
            0.3
            * float(params["susceptibility"][target])
            * prevalence_mix
            * float(state["S"][target])
        )
        progression = 0.2 * float(state["E"][target])
        recovery = 0.1 * float(state["I"][target])
        expected["S"][target] = -infection
        expected["E"][target] = infection - progression
        expected["I"][target] = progression - recovery
        expected["R"][target] = recovery
    for name in state:
        np.testing.assert_allclose(rhs[name], expected[name], rtol=1e-12)
    np.testing.assert_allclose(sum(expected.values()), np.zeros(2), atol=1e-12)
    assert compiled.template_shapes == dict.fromkeys(state, (2,))
    assert len(compiled.meta["initial_state"]) == 8
    assert validate_spec(guide_spec).parameters["C"] == ("age", "age")
    expression_spec = {
        **guide_spec,
        **yaml.safe_load(_snippet("seir-equations", "yaml")),
    }
    del expression_spec["transitions"]
    equations = compile_spec(expression_spec)
    np.testing.assert_allclose(
        equations.eval_fn(0.0, namespace["flat"], **params),
        namespace["flat_rhs"],
        rtol=1e-12,
    )


def test_constraint_metadata_does_not_mask_public_state_expansion() -> None:
    """The documented boundary is explicit: current reserved metadata is inert."""
    constraints = [
        {
            "axes": ["age", "vax"],
            "allow": [{"age": "young", "vax": ["none"]}, {"age": "old"}],
        }
    ]
    compiled = compile_spec({
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["young", "old"]},
            {"name": "vax", "coords": ["none", "dose1"]},
        ],
        "state": ["S[age,vax]"],
        "equations": {"S[age,vax]": "-S[age,vax]"},
        "constraints": constraints,
    })
    assert len(compiled.state_names) == 4
    assert compiled.meta["constraints"] == constraints
    np.testing.assert_array_equal(
        compiled.eval_fn(0.0, np.asarray([1.0, 2.0, 3.0, 4.0])),
        [-1.0, -2.0, -3.0, -4.0],
    )
