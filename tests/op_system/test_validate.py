"""Tests for ``op_system.validate_spec`` and the validation CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from op_system import validate_spec
from op_system.validate import main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _valid() -> dict[str, Any]:
    return {
        "kind": "transitions",
        "axes": [
            {"name": "vax", "coords": ["u", "v"]},
            {"name": "imm", "type": "ordinal", "coords": ["x0", "x1", "x2"]},
        ],
        "state": ["X[vax, imm]", "E[vax]"],
        "transitions": [
            {"from": "X[vax, imm]", "to": "E[vax]", "rate": "foi * theta[imm]"},
            {"from": "X[vax=u, imm]", "to": "X[vax=v, imm]", "rate": "nu"},
        ],
        "operators": [
            {
                "kind": "custom",
                "axis": "imm",
                "velocity": "w",
                "kernel": {
                    "form": "custom",
                    "params": {"generator": "G"},
                    "param_axes": {"G": ["imm", "imm"]},
                },
            }
        ],
    }


def _unvectorizable() -> dict[str, Any]:
    return {
        "kind": "transitions",
        "axes": [{"name": "imm", "type": "ordinal", "coords": ["x0", "x1"]}],
        "state": ["X[imm]"],
        "transitions": [{"from": "X[imm=x1]", "to": "X[imm=x0]", "rate": "k"}],
    }


def test_valid_spec_reports_cost_and_parameters() -> None:
    """A valid spec passes every stage and lists consumed parameters."""
    report = validate_spec(_valid())
    assert report.ok
    assert report.stages == {
        "normalize": "passed",
        "compile": "passed",
        "vectorize": "passed",
    }
    assert report.cost["expanded_states"] == 8
    assert report.cost["coordinate_pinned_transitions"] == 1
    assert report.cost["operators"] == 1
    assert report.parameters["theta"] == ("imm",)
    assert report.parameters["G"] == ("imm", "imm")
    assert report.parameters["w"] == ()
    assert report.parameters["nu"] == ()
    assert report.shape_groups == {}


def test_vectorization_failure_reports_expression_shapes() -> None:
    """A template whose cells differ in shape is diagnosed, not just rejected."""
    report = validate_spec(_unvectorizable())
    assert not report.ok
    assert report.stages["normalize"] == "passed"
    assert report.stages["vectorize"] == "failed"
    groups = report.shape_groups["X"]
    assert {group.example_expression for group in groups} == {
        "k * X__imm_x1",
        "-(k * X__imm_x1)",
    }
    assert "vectorized eval path" in report.errors[0]


def test_normalization_failure_is_reported() -> None:
    """Invalid specs report the normalization error and skip later stages."""
    spec = _valid()
    spec["transitions"][0]["from"] = "X[vax, age]"
    report = validate_spec(spec)
    assert report.stages == {
        "normalize": "failed",
        "compile": "skipped",
        "vectorize": "skipped",
    }
    assert report.errors


def test_cli_reads_flepimop2_configs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI validates op_system systems inside a flepimop2 config (JSON is YAML)."""
    good = tmp_path / "good.yml"
    good.write_text(
        json.dumps({"system": [{"module": "op_system", "spec": _valid()}]}),
        encoding="utf-8",
    )
    bad = tmp_path / "bad.yml"
    bad.write_text(json.dumps(_unvectorizable()), encoding="utf-8")

    assert main([str(good)]) == 0
    assert main([str(bad)]) == 1
    output = capsys.readouterr().out
    assert "OK" in output
    assert "FAIL" in output
    assert "expression shapes" in output

    assert main([str(good), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    (report,) = payload.values()
    assert report["stages"]["vectorize"] == "passed"
