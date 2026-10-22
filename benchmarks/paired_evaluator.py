"""Alternate two trusted NumPy evaluator factories on one qualified tensor plan.

Only _vectorize.py may differ between compiler sources. This is a narrow RHS
factory comparison, not a general whole-package or cold-start benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.hierarchical_benchmark import _environment, _layouts
from benchmarks.hierarchical_model import fixture, model_spec, reference_rhs
from op_system import array_namespace, compile_spec
from op_system._vectorize import build_vector_plan  # ruff: ignore[import-private-name]

if TYPE_CHECKING:
    from types import ModuleType


def _load_factory(baseline: Path) -> tuple[ModuleType, dict[str, object]]:
    import op_system  # ruff: ignore[import-outside-top-level]

    current = Path(op_system.__file__).resolve().parent
    previous = baseline / "src" / "op_system"
    trees = [
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        }
        for root in (previous, current)
    ]
    if not trees[0] or {k: v for k, v in trees[0].items() if k != "_vectorize.py"} != {
        k: v for k, v in trees[1].items() if k != "_vectorize.py"
    }:
        msg = "Paired factories require identical compiler sources except _vectorize.py"
        raise ValueError(msg)
    name = f"_op_system_baseline_vectorize_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, previous / "_vectorize.py")
    if spec is None or spec.loader is None:
        msg = "Cannot load the trusted baseline evaluator source"
        raise ValueError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name)
    return module, {"baseline": trees[0], "candidate": trees[1]}


def run_pair(  # ruff: ignore[complex-structure, too-many-arguments, too-many-locals]
    baseline: Path,
    groups: int,
    stages: int,
    *,
    coupled: bool,
    repeats: int = 15,
    inner_loops: int = 200,
) -> dict[str, object]:
    """Qualify both factories, then alternate adjacent warm sample pairs.

    Trusted baseline Python is executed. The invariant source check keeps the
    normalized-plan layout fixed. Handwritten controls are the same function
    measured twice, exposing drift between adjacent sampling windows.

    Returns:
        Raw paired samples, complete source hashes and numerical checks.

    Raises:
        ValueError: Sources differ beyond factories or dimensions/budgets are invalid.
    """
    if any(type(v) is not int or v < 1 for v in (groups, stages, repeats, inner_loops)):
        msg = "Paired dimensions and budgets must be positive integers"
        raise ValueError(msg)
    old, sources = _load_factory(baseline)
    compiled = compile_spec(model_spec(groups, stages, coupled=coupled))
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    if rhs is None:
        msg = "Paired factories require retained normalized IR"
        raise ValueError(msg)
    plan = build_vector_plan(rhs)
    if plan is None:
        msg = "Paired fixture requires a vector plan"
        raise ValueError(msg)
    previous = replace(
        compiled,
        eval_fn=old.make_vectorized_eval_fn(plan),
        pytree_eval_fn=old.make_pytree_eval_fn(plan),
        block_pytree_eval_fn=None,
    )
    current = replace(compiled, block_pytree_eval_fn=None)
    state, params = fixture(groups, stages, coupled=coupled)
    expected = reference_rhs(state, params, coupled=coupled)
    functions = {
        "baseline": _layouts(
            previous, np, array_namespace(state["A"]), coupled=coupled
        ),
        "candidate": _layouts(
            current, np, array_namespace(state["A"]), coupled=coupled
        ),
    }
    checks: dict[str, dict[str, float]] = {}
    for variant, layouts in functions.items():
        checks[variant] = {}
        for layout, function in layouts.items():
            actual = function(state, params)
            errors = []
            for key, values in expected.items():
                np.testing.assert_allclose(actual[key], values, rtol=1e-12, atol=1e-13)
                errors.append(float(np.max(np.abs(actual[key] - values))))
            checks[variant][layout] = max(errors)
    results = {}
    for layout in functions["baseline"]:
        for _ in range(5):
            for variant in functions:
                functions[variant][layout](state, params)
        pairs = []
        for repeat in range(repeats):
            order = (
                ("baseline", "candidate")
                if repeat % 2 == 0
                else ("candidate", "baseline")
            )
            pair = {}
            for variant in order:
                function = functions[variant][layout]
                start = time.perf_counter()
                for _ in range(inner_loops):
                    function(state, params)
                pair[variant] = (time.perf_counter() - start) / inner_loops
            pairs.append({"order": order, "seconds": pair})
        results[layout] = pairs
    return {
        "schema_version": 1,
        "case": {
            "groups": groups,
            "stages": stages,
            "coupled": coupled,
            "dtype": "float64",
        },
        "repeats": repeats,
        "inner_loops": inner_loops,
        "sources": sources,
        "plan": "one vector plan; identical compiler sources except _vectorize.py",
        "max_absolute_errors": checks,
        "samples": results,
        "environment": _environment(np),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--coupled", action="store_true")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--inner-loops", type=int, default=200)
    args = parser.parse_args()
    print(  # ruff: ignore[print]
        json.dumps(
            run_pair(
                args.baseline,
                args.groups,
                args.stages,
                coupled=args.coupled,
                repeats=args.repeats,
                inner_loops=args.inner_loops,
            ),
            allow_nan=False,
        )
    )
