"""Compile independent reactions with held forcing and named observables.

Run from a checkout with ``python examples/independent_reactions.py``.
The compiler evaluates derivatives and hazards; an engine advances time.
"""

from __future__ import annotations

import json

import numpy as np

from op_system import compile_spec, validate_spec


def run_example() -> dict[str, object]:
    """Check independent equations, table boundaries and observer values.

    Returns:
        JSON-compatible summary of the checked compiler artifacts.

    Raises:
        ValueError: The example specification fails structural validation.
    """
    spec = {
        "kind": "transitions",
        "factorize_axes": ["rep"],
        "time_interpolation": "previous",
        "axes": [
            {"name": "rep", "coords": ["a", "b"]},
            {"name": "time", "type": "continuous", "coords": [0.0, 1.0]},
        ],
        "state": ["S[rep]", "I[rep]"],
        "transitions": [
            {
                "name": "move",
                "from": "S[rep]",
                "to": "I[rep]",
                "rate": "rate[time,rep]",
            },
        ],
        "observables": {"population": {"expr": "S[rep]+I[rep]", "axes": ["rep"]}},
    }
    report = validate_spec(spec)
    if not report.ok:
        raise ValueError(report.errors)
    compiled = compile_spec(spec)
    state = np.array([10.0, 20.0, 0.0, 0.0])
    table = np.array([[0.0, 0.5], [0.2, 0.1]])
    before = compiled.eval_fn(0.5, state, rate=table)
    after = compiled.eval_fn(1.0, state, rate=table)
    np.testing.assert_allclose(before, [0.0, -10.0, 0.0, 10.0])
    np.testing.assert_allclose(after, [-2.0, -2.0, 2.0, 2.0])
    population = compiled.observables["population"].eval_fn(1.0, state)
    np.testing.assert_array_equal(population, [10.0, 20.0])
    return {
        "state_names": compiled.state_names,
        "template_shapes": compiled.template_shapes,
        "forcing_breakpoints": compiled.meta["forcing_breakpoints"],
        "independent_axes": compiled.reactions[0].independent_axes,
        "drift_before": np.asarray(before).tolist(),
        "drift_after": np.asarray(after).tolist(),
        "population": np.asarray(population).tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2))  # ruff: ignore[print] - executable example output
