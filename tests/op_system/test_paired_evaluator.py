"""Paired factory comparisons retain source guards and independent value checks."""

from pathlib import Path
from shutil import copytree, ignore_patterns

import pytest

import op_system
from benchmarks.paired_evaluator import run_pair


def test_numpy_paired_factories_qualify_and_alternate(tmp_path: Path) -> None:
    """Identical trusted source has zero oracle errors and two explicit panels."""
    baseline = tmp_path / "baseline"
    copytree(
        Path(op_system.__file__).resolve().parent,
        baseline / "src" / "op_system",
        ignore=ignore_patterns("__pycache__", "*.pyc"),
    )
    result = run_pair(
        baseline,
        2,
        2,
        coupled=True,
        repeats=2,
        inner_loops=2,
    )
    assert result["repeats"] == 2
    checks = result["max_absolute_errors"]
    assert isinstance(checks, dict)
    for values in checks.values():
        assert all(error < 1e-12 for error in values.values())
    samples = result["samples"]
    assert isinstance(samples, dict)
    for pairs in samples.values():
        assert pairs[0]["order"] == ("baseline", "candidate")
        assert pairs[1]["order"] == ("candidate", "baseline")
        assert all(value > 0 for pair in pairs for value in pair["seconds"].values())


def test_numpy_paired_source_guard_precedes_execution(tmp_path: Path) -> None:
    """An incompatible compiler tree cannot execute its supplied Python factory."""
    root = tmp_path / "src" / "op_system"
    root.mkdir(parents=True)
    (root / "_vectorize.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="identical compiler sources"):
        run_pair(tmp_path, 2, 2, coupled=False, repeats=1, inner_loops=1)
