"""Coverage records for dynamics without reaction artifacts (issue #244)."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import Any

import pytest

from op_system import ReactionGap, compile_spec

AGE = {"name": "age", "coords": ["c", "a"]}
IMM = {"name": "imm", "coords": ["x0", "x1", "x2"]}


def _transitions(
    state: list[str], transitions: list[dict[str, Any]], *axes: dict[str, Any]
) -> dict[str, Any]:
    return {
        "kind": "transitions",
        "axes": list(axes),
        "state": state,
        "transitions": transitions,
    }


def _gaps(
    spec: dict[str, Any],
) -> list[tuple[str, str | None, str | None, str | None, str]]:
    return [
        (gap.origin, gap.name, gap.source, gap.target, gap.reason)
        for gap in compile_spec(spec).reaction_gaps
    ]


def test_fully_covered_model_has_no_gaps() -> None:
    """Every named, in-scope transition compiles, so nothing is reported."""
    spec = _transitions(
        ["S[age]", "I[age]", "R[age]"],
        [
            {"name": "infect", "from": "S[age]", "to": "I[age]", "rate": "b * I[age]"},
            {"name": "recover", "from": "I[age]", "to": "R[age]", "rate": "g"},
            {"name": "import", "to": "I[age]", "rate": "eta", "reactants": []},
        ],
        AGE,
    )
    compiled = compile_spec(spec)
    assert [r.name for r in compiled.reactions] == ["infect", "recover", "import"]
    assert compiled.reaction_gaps == ()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param(
            _transitions(
                ["S[age]", "I[age]"],
                [
                    {"name": "infect", "from": "S[age]", "to": "I[age]", "rate": "b"},
                    {"from": "I[age]", "to": "S[age]", "rate": "g"},
                ],
                AGE,
            ),
            [("transitions[1]", None, "I[age]", "S[age]", "unnamed")],
            id="unnamed",
        ),
        pytest.param(
            _transitions(
                ["S[age]"],
                [
                    {
                        "coord_shift": {"age": "c -> a"},
                        "rate": "b",
                        "apply_to": ["S"],
                    }
                ],
                AGE,
            ),
            [("transitions[0]", None, "S[age=c]", "S[age=a]", "unnamed")],
            id="pairwise-coord-shift",
        ),
        pytest.param(
            _transitions(
                ["S", "I[age]"],
                [{"name": "seed", "from": "S", "to": "I[age]", "rate": "b"}],
                AGE,
            ),
            [("transitions[0]", "seed", "S", "I[age]", "target_axis_not_on_source")],
            id="target-axis",
        ),
        pytest.param(
            _transitions(
                ["N[age]"],
                [
                    {
                        "name": "birth",
                        "to": "N[age=c]",
                        "rate": "B[age] * sum_over(N[age:a], age=a)",
                        "reactants": [],
                    }
                ],
                AGE,
            ),
            [("transitions[0]", "birth", None, "N[age=c]", "rate_axis_out_of_scope")],
            id="rate-axis",
        ),
        pytest.param(
            {"kind": "expr", "state": ["x"], "equations": {"x": "-k * x"}},
            [("equations", None, None, None, "expr_spec")],
            id="expr-spec",
        ),
    ],
)
def test_each_omission_is_reported_with_its_reason(
    spec: dict[str, Any],
    expected: list[tuple[str, str | None, str | None, str | None, str]],
) -> None:
    """Every transition without a compiled reaction names its origin and reason."""
    assert _gaps(spec) == expected


def test_axis_less_sir_is_fully_covered() -> None:
    """Axis-less states publish scalar reactions (issue #246)."""
    spec = _transitions(
        ["S", "I", "R"],
        [
            {"name": "infect", "from": "S", "to": "I", "rate": "b * I"},
            {"name": "recover", "from": "I", "to": "R", "rate": "g"},
        ],
    )
    compiled = compile_spec(spec)
    assert [r.name for r in compiled.reactions] == ["infect", "recover"]
    assert compiled.reaction_gaps == ()


def test_missing_vector_plan_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a state layout, every reaction is reported as unsupported."""
    import op_system._vectorize as vectorize  # ruff: ignore[import-outside-top-level]

    monkeypatch.setattr(vectorize, "build_vector_plan", lambda _rhs: None)
    spec = _transitions(
        ["S", "I"], [{"name": "infect", "from": "S", "to": "I", "rate": "b"}]
    )
    assert _gaps(spec) == [("transitions[0]", "infect", "S", "I", "unsupported_layout")]


def test_compile_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A propensity that cannot be lowered is reported instead of vanishing."""
    import op_system._vectorize as vectorize  # ruff: ignore[import-outside-top-level]

    monkeypatch.setattr(vectorize, "_compile_ir_expr", lambda *_a, **_k: None)
    spec = _transitions(
        ["S[age]", "I[age]"],
        [{"name": "infect", "from": "S[age]", "to": "I[age]", "rate": "b"}],
        AGE,
    )
    assert _gaps(spec) == [
        ("transitions[0]", "infect", "S[age]", "I[age]", "compile_failed")
    ]


def test_gaps_survive_pickling_and_stay_out_of_meta() -> None:
    """Pickled RHS rebuild their gaps; origin bookkeeping stays private."""
    spec = _transitions(
        ["S[age]", "I[age]"],
        [
            {"name": "infect", "from": "S[age]", "to": "I[age]", "rate": "b"},
            {"from": "I[age]", "to": "S[age]", "rate": "g"},
        ],
        AGE,
    )
    compiled = compile_spec(spec)
    restored = pickle.loads(pickle.dumps(compiled))  # ruff: ignore[suspicious-pickle-usage]
    assert restored.reaction_gaps == compiled.reaction_gaps
    assert all(isinstance(gap, ReactionGap) for gap in restored.reaction_gaps)
    assert not any(
        key.startswith("_op_system")
        for tr in compiled.meta["transitions"]
        for key in tr
    )
