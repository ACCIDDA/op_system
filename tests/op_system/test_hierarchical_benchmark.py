"""Keep benchmark correctness and result accounting independent of wall time."""

import json
from typing import Any

import pytest

from benchmarks.hierarchical_benchmark import BenchmarkCase, RepeatPolicy, run_case


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("coupled", [False, True], ids=["independent", "coupled"])
def test_benchmark_agreement(backend: str, dtype: str, *, coupled: bool) -> None:
    """Require oracle value/derivative agreement for every measured layout."""
    module = pytest.importorskip(backend)
    if backend == "jax":
        module.config.update("jax_enable_x64", True)  # ruff: ignore[boolean-positional-value-in-call]
    result: dict[str, Any] = run_case(
        BenchmarkCase(backend, 2, 3, dtype, coupled=coupled), measure=False
    )
    assert result["status"] == "qualified"
    assert result["state_elements"] == 18
    assert result["local_parameter_elements"] == 2
    assert result["shared_kernel_elements"] == 9
    assert result["normalize"] is None
    assert result["compile_rhs"] is None
    assert result["specification_json_bytes"] > 0
    assert result["fixture_bytes"] > 0
    assert set(result["evaluators"]) == {"handwritten", "flat", "structured"} | (
        set() if coupled else {"block"}
    )
    assert bool(result["block_unavailable_reason"]) == coupled
    for evaluator in result["evaluators"].values():
        assert evaluator["timings"] is None
        assert evaluator["max_abs_value_error"] < 1e-5
        assert (evaluator["gain_derivative"] is not None) == (backend != "numpy")
    json.dumps(result, allow_nan=False)


def test_timing_accounting() -> None:
    """A tiny run reports each compiler phase and every warm sample."""
    result: dict[str, Any] = run_case(
        BenchmarkCase("numpy", 2, 2), RepeatPolicy(2, 2, 1)
    )
    assert len(result["normalize"]["samples_seconds"]) == 2
    assert len(result["compile_rhs"]["samples_seconds"]) == 2
    for evaluator in result["evaluators"].values():
        timings = evaluator["timings"]
        assert timings["cold_jit_seconds"] is None
        assert len(timings["warm"]["samples_seconds"]) == 2
        assert timings["warm_calls_per_second"] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "case",
    [
        BenchmarkCase("missing", 2, 2),
        BenchmarkCase("numpy", 0, 2),
        BenchmarkCase("numpy", 2, 2, "int32"),
    ],
)
def test_invalid_case(case: BenchmarkCase) -> None:
    """Reject unavailable backends and dimensions rather than report timings."""
    with pytest.raises(ValueError, match=r"choose|positive"):
        run_case(case, measure=False)


def test_invalid_policy() -> None:
    """No timing run may silently reduce its repeat count to zero."""
    with pytest.raises(ValueError, match="positive"):
        run_case(BenchmarkCase("numpy", 2, 2), RepeatPolicy(0))


def test_numpy_profile_is_separate_from_unprofiled_timings() -> None:
    """Call accounting is opt-in, qualified and excludes local source paths."""
    result = run_case(BenchmarkCase("numpy", 2, 2), measure=False, profile_calls=3)
    assert result["status"] == "qualified"
    evaluators: Any = result["evaluators"]
    for evaluator in evaluators.values():
        assert evaluator["timings"] is None
        profile = evaluator["profile"]
        assert profile["evaluator_calls"] == 3
        assert profile["total_calls"] > 3
        assert profile["accounted_seconds"] > 0
        assert profile["top_self_time"]
        assert all(
            not row["module"].startswith("/") for row in profile["top_self_time"]
        )
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="correctness-only"):
        run_case(BenchmarkCase("numpy", 2, 2), profile_calls=3)
    with pytest.raises(ValueError, match="correctness-only"):
        run_case(BenchmarkCase("torch", 2, 2), measure=False, profile_calls=3)
