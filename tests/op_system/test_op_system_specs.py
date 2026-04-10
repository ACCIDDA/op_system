"""Unit tests for op_system.specs (pytest).

These tests cover:
- expr-style RHS normalization
- transitions-style RHS normalization
- symbol/parameter extraction rules
- validation failures and error types/messages
- preservation of reserved future blocks via `meta`
"""

from __future__ import annotations

import re

import pytest

from op_system.specs import (
    ConstraintRule,
    NormalizedRhs,
    _normalize_constraints,
    normalize_expr_rhs,
    normalize_rhs,
    normalize_transitions_rhs,
)


def test_normalize_expr_rhs_happy_path() -> None:
    """Test expr-style RHS normalization happy path."""
    spec = {
        "kind": "expr",
        "state": ["S", "I", "R"],
        "aliases": {"N": "S + I + R"},
        "equations": {
            "S": "-(beta * S * I) / N",
            "I": "(beta * S * I) / N - gamma * I",
            "R": "gamma * I",
        },
    }

    out = normalize_expr_rhs(spec)
    assert isinstance(out, NormalizedRhs)
    assert out.kind == "expr"
    assert out.state_names == ("S", "I", "R")
    assert out.aliases["N"] == "S + I + R"
    assert out.equations[0].startswith("-(") or "beta" in out.equations[0]
    assert out.param_names == ("beta", "gamma")
    assert "S" in out.all_symbols
    assert "beta" in out.all_symbols
    assert "N" in out.all_symbols


def test_normalize_transitions_rhs_happy_path() -> None:
    """Test transitions-style RHS normalization happy path."""
    spec = {
        "kind": "transitions",
        "state": ["S", "I", "R"],
        "aliases": {"N": "S + I + R"},
        "transitions": [
            {"from": "S", "to": "I", "rate": "beta * I / N"},
            {"from": "I", "to": "R", "rate": "gamma"},
        ],
    }

    out = normalize_transitions_rhs(spec)
    assert out.kind == "transitions"
    assert out.state_names == ("S", "I", "R")
    assert out.param_names == ("beta", "gamma")

    # Ensure equations reflect flow conservation structure.
    eq_s, eq_i, eq_r = out.equations
    assert "beta" in eq_s
    assert "S" in eq_s
    assert "beta" in eq_i
    assert "gamma" in eq_i
    assert "gamma" in eq_r
    assert "I" in eq_r

    # S must lose infection flow; R must gain recovery flow.
    # The current normalization format is: -((rate_expr)*(from_state))
    assert eq_s.startswith("-(")
    assert ")*(S)" in eq_s or "*(S)" in eq_s

    assert "+(" in eq_r or eq_r.startswith("(") or "gamma" in eq_r

    # Meta should retain the original transitions list.
    assert "transitions" in out.meta
    assert isinstance(out.meta["transitions"], list)
    assert len(out.meta["transitions"]) == 2


def test_transitions_accepts_optional_name_and_preserves_meta() -> None:
    """Transitions may include an optional name that is preserved in meta."""
    spec = {
        "kind": "transitions",
        "state": ["S", "I", "R"],
        "transitions": [
            {"name": "infect", "from": "S", "to": "I", "rate": "beta"},
            {"from": "I", "to": "R", "rate": "gamma"},
        ],
    }

    out = normalize_transitions_rhs(spec)
    transitions_meta = out.meta.get("transitions")
    assert isinstance(transitions_meta, list)
    assert transitions_meta[0]["name"] == "infect"
    assert "name" not in transitions_meta[1]


def test_transitions_rejects_nonstring_name() -> None:
    """Transition names must be non-empty strings when provided."""
    spec = {
        "kind": "transitions",
        "state": ["S", "I"],
        "transitions": [
            {"name": 3, "from": "S", "to": "I", "rate": "beta"},
        ],
    }

    with pytest.raises(ValueError, match=r"name must be a non-empty string"):
        normalize_transitions_rhs(spec)


def test_normalize_rhs_preserves_reserved_blocks_in_meta() -> None:
    """Test that reserved future blocks are preserved in meta."""
    spec = {
        "kind": "transitions",
        "state": ["S", "I", "R"],
        "aliases": {"N": "S + I + R"},
        "transitions": [{"from": "S", "to": "I", "rate": "beta * I / N"}],
        "operators": {"default": {"scheme": "cn"}},  # reserved for future
        "sources": {"S": "0.0"},  # reserved for future
    }

    out = normalize_rhs(spec)
    assert out.kind == "transitions"
    assert out.meta.get("operators") == {"default": {"scheme": "cn"}}
    assert out.meta.get("sources") == {"S": "0.0"}


def test_expr_template_expansion_and_sum_over() -> None:
    """Templates over categorical axes expand state and equations; sum_over unrolls."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1", "p2"]}],
        "state": ["S[pop]", "I[pop]"],
        "equations": {
            "S[pop]": "-beta * S[pop] * sum_over(pop=j, I[pop=j])",
            "I[pop]": "beta * S[pop] * sum_over(pop=j, I[pop=j]) - gamma * I[pop]",
        },
    }

    out = normalize_expr_rhs(spec)
    assert set(out.state_names) == {"S__pop_p1", "S__pop_p2", "I__pop_p1", "I__pop_p2"}
    # sum_over should be unrolled to explicit sums
    assert any("I__pop_p1" in eq and "I__pop_p2" in eq for eq in out.equations)
    assert out.param_names == ("beta", "gamma")


def test_alias_and_param_templates_expand_over_axes() -> None:
    """Alias names and inline param placeholders expand over axis assignments."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["y", "o"]}],
        "state": ["S[age]", "I[age]"],
        "aliases": {
            "beta[age]": "b0 * k[age]",
            "k[age]": "k_base * (1 + offset[age])",
        },
        "equations": {
            "S[age]": "-beta[age] * S[age]",
            "I[age]": "beta[age] * S[age] - gamma * I[age]",
        },
    }

    out = normalize_expr_rhs(spec)

    expected_aliases = {"beta__age_y", "beta__age_o", "k__age_y", "k__age_o"}
    assert set(out.aliases) == expected_aliases

    assert any("beta__age_y" in eq for eq in out.equations)
    assert any("beta__age_o" in eq for eq in out.equations)

    expected_params = {
        "b0",
        "gamma",
        "k_base",
        "offset__age_y",
        "offset__age_o",
    }
    assert set(out.param_names) == expected_params


def test_sum_over_rejects_continuous_axis() -> None:
    """sum_over on a continuous axis should raise an error."""
    spec = {
        "kind": "expr",
        "axes": [
            {
                "name": "age",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 10.0},
                "size": 3,
            }
        ],
        "state": ["S[age]"],
        "equations": {"S[age]": "-sum_over(age=i, S[age])"},
    }
    with pytest.raises(ValueError, match=r"sum_over axis .* must be categorical"):
        normalize_expr_rhs(spec)


def test_integrate_over_expands_with_continuous_deltas() -> None:
    """integrate_over uses trapezoidal weights from continuous axis coords."""
    spec = {
        "kind": "expr",
        "axes": [
            {
                "name": "x",
                "type": "continuous",
                "coords": [0.0, 1.0, 3.0],
            }
        ],
        "state": ["u[x]", "v"],
        "equations": {
            "u[x]": "0.0",
            "v": "integrate_over(x=i, u[x=i])",
        },
    }

    out = normalize_expr_rhs(spec)
    axes_meta = out.meta["axes"]
    assert axes_meta[0]["deltas"] == [0.5, 1.5, 1.0]

    eq_v = out.equations[-1]
    assert "0.5" in eq_v
    assert "1.5" in eq_v
    assert "1.0" in eq_v
    assert "u__x_0_0" in eq_v
    assert "u__x_1_0" in eq_v
    assert "u__x_3_0" in eq_v


def test_integrate_over_rejects_categorical_axis() -> None:
    """integrate_over on a categorical axis should raise an error."""
    spec = {
        "kind": "expr",
        "axes": [
            {
                "name": "g",
                "coords": ["a", "b"],
            }
        ],
        "state": ["x"],
        "equations": {"x": "integrate_over(g=i, x)"},
    }

    with pytest.raises(ValueError, match=r"must be continuous"):
        normalize_expr_rhs(spec)


def test_axes_generate_continuous_linear_and_log() -> None:
    """Continuous axes generate coords from domain/size for linear and log spacing."""
    spec_linear = {
        "kind": "expr",
        "axes": [
            {
                "name": "x",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 10.0},
                "size": 3,
                "spacing": "linear",
            }
        ],
        "state": ["u"],
        "equations": {"u": "0.0"},
    }
    out_linear = normalize_expr_rhs(spec_linear)
    coords_linear = out_linear.meta["axes"][0]["coords"]
    assert coords_linear == [0.0, 5.0, 10.0]

    spec_log = {
        "kind": "expr",
        "axes": [
            {
                "name": "r",
                "type": "continuous",
                "domain": {"lb": 1.0, "ub": 100.0},
                "size": 3,
                "spacing": "log",
            }
        ],
        "state": ["v"],
        "equations": {"v": "0.0"},
    }
    out_log = normalize_expr_rhs(spec_log)
    coords_log = out_log.meta["axes"][0]["coords"]
    assert coords_log == pytest.approx([1.0, 10.0, 100.0])


def test_categorical_missing_coords_and_continuous_monotonicity() -> None:
    """Categorical axes require coords; continuous coords must be non-decreasing."""
    spec_missing_coords = {
        "kind": "expr",
        "axes": [{"name": "cat", "type": "categorical"}],
        "state": ["x"],
        "equations": {"x": "0.0"},
    }
    with pytest.raises(ValueError, match=r"categorical requires coords"):
        normalize_expr_rhs(spec_missing_coords)

    spec_bad_order = {
        "kind": "expr",
        "axes": [
            {
                "name": "z",
                "type": "continuous",
                "coords": [0.0, -1.0],
            }
        ],
        "state": ["x"],
        "equations": {"x": "0.0"},
    }
    with pytest.raises(ValueError, match=r"non-decreasing"):
        normalize_expr_rhs(spec_bad_order)


def test_kernels_and_operator_meta_preserved() -> None:
    """Kernel and operator metadata are normalized and preserved in meta."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["a", "b"]},
        ],
        "kernels": [
            {"name": "k_fixed", "value": 0.5, "axes": ["age"]},
            {
                "name": "k_gauss",
                "form": "gaussian",
                "params": {"scale": 1.0, "sigma": 0.5},
            },
        ],
        "operators": [
            {"name": "diff", "axis": "age", "kind": "laplacian"},
        ],
        "state": ["S"],
        "equations": {"S": "0.0"},
    }

    out = normalize_expr_rhs(spec)
    kernels_meta = out.meta.get("kernels")
    assert isinstance(kernels_meta, list)
    assert {mk["name"] for mk in kernels_meta} == {"k_fixed", "k_gauss"}
    assert any(
        isinstance(mk.get("value"), (int, float))
        and float(mk["value"]) == pytest.approx(0.5)
        for mk in kernels_meta
    )
    assert any(mk.get("form") == "gaussian" for mk in kernels_meta)

    operators_meta = out.meta.get("operators")
    assert isinstance(operators_meta, list)
    assert operators_meta[0].get("axis") == "age"


def test_operator_requires_kind_and_preserves_bc() -> None:
    """Operators must declare kind; optional bc is preserved as a string."""
    spec_missing_kind = {
        "kind": "expr",
        "axes": [
            {
                "name": "x",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 1.0},
                "size": 2,
            },
        ],
        "operators": [
            {"name": "op0", "axis": "x"},
        ],
        "state": ["u"],
        "equations": {"u": "0.0"},
    }

    with pytest.raises(
        ValueError, match=r"operators\[0\]\.kind must be a non-empty string"
    ):
        normalize_expr_rhs(spec_missing_kind)

    spec_with_bc = {
        "kind": "expr",
        "axes": [
            {
                "name": "x",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 1.0},
                "size": 2,
            },
        ],
        "operators": [
            {"name": "op0", "axis": "x", "kind": "advection", "bc": "periodic"},
        ],
        "state": ["u"],
        "equations": {"u": "0.0"},
    }

    out = normalize_expr_rhs(spec_with_bc)
    operators_meta = out.meta.get("operators")
    assert isinstance(operators_meta, list)
    assert operators_meta[0]["kind"] == "advection"
    assert operators_meta[0]["bc"] == "periodic"


def test_state_axes_validation_errors() -> None:
    """state_axes must reference known axes and avoid duplicates."""
    spec_unknown_axis = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1"]}],
        "state": ["S"],
        "state_axes": {"S": ["age"]},
        "equations": {"S": "0.0"},
    }
    with pytest.raises(ValueError, match=r"references unknown axis"):
        normalize_expr_rhs(spec_unknown_axis)

    spec_duplicate_axis = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1"]}],
        "state": ["S"],
        "state_axes": {"S": ["pop", "pop"]},
        "equations": {"S": "0.0"},
    }
    with pytest.raises(ValueError, match=r"duplicate axis"):
        normalize_expr_rhs(spec_duplicate_axis)


def test_transitions_chain_helper_appends_transitions() -> None:
    """Chain helper adds internal transitions for transitions kind."""
    spec = {
        "kind": "transitions",
        "state": ["S", "R"],
        "transitions": [],
        "chain": [
            {
                "name": "I",
                "length": 2,
                "entry": {"from": "S", "rate": "beta"},
                "forward": "gamma",
                "exit": {"to": "R"},
            }
        ],
    }

    out = normalize_transitions_rhs(spec)
    # Expect synthesized states and transitions S->I1, I1->I2, I2->R
    assert set(out.state_names) == {"S", "I1", "I2", "R"}
    assert len(out.meta["transitions"]) == 3
    eqs = out.equations
    assert any("beta" in eq and "S" in eq for eq in eqs)
    assert any("I1" in eq and "gamma" in eq for eq in eqs)
    assert any("I2" in eq and "gamma" in eq for eq in eqs)


def test_transitions_template_expansion_over_axes() -> None:
    """Transitions expand templated states and rates over categorical axes."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "age", "coords": ["a", "b"]}],
        "state": ["S[age]", "I[age]", "R[age]"],
        "aliases": {
            "beta[age]": "b0 * k[age]",
            "k[age]": "k_base",
        },
        "transitions": [
            {"from": "S[age]", "to": "I[age]", "rate": "beta[age]"},
            {"from": "I[age]", "to": "R[age]", "rate": "gamma"},
        ],
    }

    out = normalize_transitions_rhs(spec)

    assert set(out.state_names) == {
        "S__age_a",
        "S__age_b",
        "I__age_a",
        "I__age_b",
        "R__age_a",
        "R__age_b",
    }

    # Equations should reference expanded alias names
    assert any("beta__age_a" in eq for eq in out.equations)
    assert any("beta__age_b" in eq for eq in out.equations)

    expected_params = {"b0", "gamma", "k_base"}
    assert set(out.param_names) == expected_params


def test_expr_chain_helper_autofills_missing_equations() -> None:
    """Chain helper under expr fills missing stage equations when provided."""
    spec = {
        "kind": "expr",
        "state": ["S", "R"],
        "equations": {
            "S": "-beta * I1",
            "R": "gamma * I2",
        },
        "chain": [{"name": "I", "length": 2, "forward": "gamma", "exit": {"to": "R"}}],
    }

    out = normalize_expr_rhs(spec)
    assert set(out.state_names) == {"S", "I1", "I2", "R"}
    # Auto-generated equations for I1 and I2 should be present
    assert any(eq.startswith("-") and "I1" in eq for eq in out.equations)
    assert any("I2" in eq for eq in out.equations)
    assert "gamma" in " ".join(out.equations)


def test_expr_chain_helper_rejects_invalid_lengths_and_missing_states() -> None:
    """Chain helper should reject invalid lengths and invalid targets."""
    spec_short = {
        "kind": "expr",
        "state": ["S", "R"],
        "equations": {"S": "0.0", "R": "0.0"},
        "chain": [{"name": "I", "length": 1, "forward": "gamma"}],
    }
    with pytest.raises(ValueError, match=r"length must be >= 2"):
        normalize_expr_rhs(spec_short)

    spec_bad_exit = {
        "kind": "expr",
        "state": ["S", "R"],
        "equations": {"S": "0.0", "R": "0.0"},
        "chain": [{"name": "I", "length": 2, "forward": "gamma", "exit": {"to": "X"}}],
    }
    with pytest.raises(ValueError, match=r"exit.to='X' not in state"):
        normalize_expr_rhs(spec_bad_exit)


def test_transitions_requires_nonempty_list() -> None:
    """Transitions may be empty in input but not after chain expansion."""
    spec = {"kind": "transitions", "state": ["S", "I"], "transitions": []}
    with pytest.raises(
        ValueError,
        match=re.escape("transitions must be non-empty after applying chain expansion"),
    ):
        normalize_transitions_rhs(spec)


def test_transitions_rejects_unknown_state_names() -> None:
    """Test that transitions referencing unknown states raise StateShapeError."""
    spec = {
        "kind": "transitions",
        "state": ["S", "I"],
        "transitions": [{"from": "S", "to": "R", "rate": "beta"}],
    }
    with pytest.raises(ValueError, match=r"not in state"):
        normalize_transitions_rhs(spec)


def test_transitions_chain_invalid_sink_rejected() -> None:
    """Transitions chain helper should reject invalid sink targets."""
    spec = {
        "kind": "transitions",
        "state": ["S"],
        "transitions": [],
        "chain": [
            {
                "name": "I",
                "length": 2,
                "entry": {"from": "S", "rate": "beta"},
                "forward": "gamma",
                "exit": {"to": "R"},
            }
        ],
    }
    with pytest.raises(ValueError, match=r"exit.to='R' not in state"):
        normalize_transitions_rhs(spec)


def test_transitions_chain_supports_heterogeneous_rates() -> None:
    """Per-stage forward lists and exit rates are honored in transitions chains."""
    spec = {
        "kind": "transitions",
        "state": ["S", "R"],
        "transitions": [],
        "chain": [
            {
                "name": "I",
                "length": 3,
                "entry": {"from": "S", "rate": "beta"},
                "forward": ["g12", "g23"],
                "exit": {"to": "R", "rate": "g3r"},
            }
        ],
    }

    out = normalize_transitions_rhs(spec)
    transitions_meta = out.meta["transitions"]
    assert len(transitions_meta) == 4
    assert {tr["rate"] for tr in transitions_meta} == {"beta", "g12", "g23", "g3r"}


def test_chain_forward_list_length_must_match_internal_edges() -> None:
    """Forward list length must equal length-1 for chain definitions."""
    spec = {
        "kind": "transitions",
        "state": ["S", "R"],
        "transitions": [],
        "chain": [
            {
                "name": "I",
                "length": 3,
                "entry": {"from": "S", "rate": "beta"},
                "forward": ["g12"],
                "exit": {"to": "R"},
            }
        ],
    }

    with pytest.raises(ValueError, match=r"forward list length must be 2"):
        normalize_transitions_rhs(spec)


def test_normalize_rhs_unsupported_kind() -> None:
    """Test that unsupported RHS kinds raise UnsupportedFeatureError."""
    spec = {"kind": "pde", "state": ["u"], "equations": {"u": "0.0"}}
    with pytest.raises(NotImplementedError, match=r"Only 'expr' and 'transitions'"):
        normalize_rhs(spec)


# ---------------------------------------------------------------------------
# _normalize_constraints tests
# ---------------------------------------------------------------------------

_AXES_AGE_VAX: list[dict[str, object]] = [
    {"name": "age", "type": "categorical", "coords": ["u65", "o65"], "size": 2},
    {
        "name": "vax",
        "type": "categorical",
        "coords": ["none", "dose1", "dose2"],
        "size": 3,
    },
]


def test_normalize_constraints_none_returns_empty() -> None:
    """None constraints returns an empty list."""
    assert _normalize_constraints(None, axes=_AXES_AGE_VAX) == []


def test_normalize_constraints_empty_list_returns_empty() -> None:
    """An empty list returns an empty list."""
    assert _normalize_constraints([], axes=_AXES_AGE_VAX) == []


def test_normalize_constraints_allow_happy_path() -> None:
    """Allow-mode constraint parses correctly."""
    raw = [
        {
            "axes": ["age", "vax"],
            "allow": [
                {"age": "u65", "vax": ["none"]},
                {"age": "o65"},
            ],
        },
    ]
    result = _normalize_constraints(raw, axes=_AXES_AGE_VAX)
    assert result == [
        ConstraintRule(
            axes=("age", "vax"),
            mode="allow",
            rules=(
                {"age": ["u65"], "vax": ["none"]},
                {"age": ["o65"]},
            ),
        ),
    ]


def test_normalize_constraints_exclude_happy_path() -> None:
    """Exclude-mode constraint parses correctly."""
    raw = [
        {
            "axes": ["age", "vax"],
            "exclude": [
                {"age": "u65", "vax": ["dose1", "dose2"]},
            ],
        },
    ]
    result = _normalize_constraints(raw, axes=_AXES_AGE_VAX)
    assert result == [
        ConstraintRule(
            axes=("age", "vax"),
            mode="exclude",
            rules=({"age": ["u65"], "vax": ["dose1", "dose2"]},),
        ),
    ]


def test_normalize_constraints_rejects_not_a_list() -> None:
    """Constraints must be a list."""
    with pytest.raises(ValueError, match=r"constraints must be a list"):
        _normalize_constraints("bad", axes=_AXES_AGE_VAX)


def test_normalize_constraints_rejects_unknown_axis() -> None:
    """Reference to an undefined axis raises."""
    raw = [{"axes": ["age", "region"], "allow": [{"age": "u65"}]}]
    with pytest.raises(ValueError, match=r"unknown axis 'region'"):
        _normalize_constraints(raw, axes=_AXES_AGE_VAX)


def test_normalize_constraints_rejects_unknown_coord() -> None:
    """Reference to an undefined coordinate raises."""
    raw = [{"axes": ["age", "vax"], "allow": [{"age": "child"}]}]
    with pytest.raises(ValueError, match=r"unknown coord 'child'"):
        _normalize_constraints(raw, axes=_AXES_AGE_VAX)


def test_normalize_constraints_rejects_mixed_allow_exclude() -> None:
    """Providing both allow and exclude in one rule raises."""
    raw = [
        {
            "axes": ["age", "vax"],
            "allow": [{"age": "u65"}],
            "exclude": [{"age": "o65"}],
        },
    ]
    with pytest.raises(ValueError, match=r"either 'allow' or 'exclude', not both"):
        _normalize_constraints(raw, axes=_AXES_AGE_VAX)


def test_normalize_constraints_rejects_missing_mode() -> None:
    """A rule with neither allow nor exclude raises."""
    raw = [{"axes": ["age", "vax"]}]
    with pytest.raises(ValueError, match=r"must specify 'allow' or 'exclude'"):
        _normalize_constraints(raw, axes=_AXES_AGE_VAX)


def test_normalize_constraints_rejects_too_few_axes() -> None:
    """A rule with fewer than two axes raises."""
    raw = [{"axes": ["age"], "allow": [{"age": "u65"}]}]
    with pytest.raises(ValueError, match=r"at least two axis names"):
        _normalize_constraints(raw, axes=_AXES_AGE_VAX)


def test_normalize_constraints_rejects_duplicate_axes() -> None:
    """Duplicate axes in one rule raises."""
    raw = [{"axes": ["age", "age"], "allow": [{"age": "u65"}]}]
    with pytest.raises(ValueError, match=r"duplicate axis 'age'"):
        _normalize_constraints(raw, axes=_AXES_AGE_VAX)


def test_normalize_constraints_rejects_empty_allow_list() -> None:
    """An empty allow list raises."""
    raw = [{"axes": ["age", "vax"], "allow": []}]
    with pytest.raises(ValueError, match=r"must be a non-empty list"):
        _normalize_constraints(raw, axes=_AXES_AGE_VAX)


def test_normalize_constraints_rejects_empty_rule_mapping() -> None:
    """A rule entry with no axis keys raises."""
    raw = [{"axes": ["age", "vax"], "allow": [{}]}]
    with pytest.raises(ValueError, match=r"must specify at least one axis"):
        _normalize_constraints(raw, axes=_AXES_AGE_VAX)


def test_normalize_constraints_rejects_rule_axis_not_in_constraint_axes() -> None:
    """A rule referencing an axis not declared in the constraint axes raises."""
    axes = [
        *_AXES_AGE_VAX,
        {
            "name": "region",
            "type": "categorical",
            "coords": ["east", "west"],
            "size": 2,
        },
    ]
    raw = [{"axes": ["age", "vax"], "allow": [{"region": "east"}]}]
    with pytest.raises(ValueError, match=r"references axis 'region' not in"):
        _normalize_constraints(raw, axes=axes)


# -- Equation key whitespace normalisation --


def test_expr_equation_keys_with_spaces_accepted() -> None:
    """Equation keys with spaces inside brackets match their template."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "x", "coords": ["a", "b"]},
            {"name": "y", "coords": ["c", "d"]},
        ],
        "state": ["u[x,y]"],
        "equations": {"u[x, y]": "-u[x, y]"},
    }
    out = normalize_expr_rhs(spec)
    assert len(out.state_names) == 4
    assert len(out.equations) == 4


def test_expr_equation_keys_spaces_three_axes() -> None:
    """Equation keys with spaces work for 3+ axes."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "a", "coords": ["a1", "a2"]},
            {"name": "b", "coords": ["b1"]},
            {"name": "c", "coords": ["c1"]},
        ],
        "state": ["u[a,b,c]"],
        "equations": {"u[a, b, c]": "-u[a, b, c]"},
    }
    out = normalize_expr_rhs(spec)
    assert len(out.state_names) == 2
    assert len(out.equations) == 2


def test_expr_equation_keys_extra_spaces_normalised() -> None:
    """Extra whitespace around axis names in equation keys is tolerated."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["S[loc]"],
        "equations": {"S[ loc ]": "-S[ loc ]"},
    }
    out = normalize_expr_rhs(spec)
    assert len(out.state_names) == 2
