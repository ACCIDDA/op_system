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
    NormalizedRhs,
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
    assert any(mk.get("value") == 0.5 for mk in kernels_meta)
    assert any(mk.get("form") == "gaussian" for mk in kernels_meta)

    operators_meta = out.meta.get("operators")
    assert isinstance(operators_meta, list)
    assert operators_meta[0].get("axis") == "age"


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
        "state": ["S", "I1", "I2", "R"],
        "transitions": [
            {"from": "S", "to": "I1", "rate": "beta"},
        ],
        "chain": [{"name": "I", "length": 2, "forward": "gamma", "to": "R"}],
    }

    out = normalize_transitions_rhs(spec)
    # Expect added transitions I1->I2 and I2->R
    assert len(out.meta["transitions"]) == 3
    eqs = out.equations
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
        "state": ["S", "I1", "I2", "R"],
        "equations": {
            "S": "-beta * I1",
            "R": "gamma * I2",
        },
        "chain": [{"name": "I", "length": 2, "forward": "gamma", "to": "R"}],
    }

    out = normalize_expr_rhs(spec)
    assert set(out.state_names) == {"S", "I1", "I2", "R"}
    # Auto-generated equations for I1 and I2 should be present
    assert any(eq.startswith("-") and "I1" in eq for eq in out.equations)
    assert any("I2" in eq for eq in out.equations)
    assert "gamma" in " ".join(out.equations)


def test_expr_chain_helper_rejects_invalid_lengths_and_missing_states() -> None:
    """Chain helper should reject length < 2 and missing stage definitions."""
    spec_short = {
        "kind": "expr",
        "state": ["S", "R"],
        "equations": {"S": "0.0", "R": "0.0"},
        "chain": [{"name": "I", "length": 1, "forward": "gamma"}],
    }
    with pytest.raises(ValueError, match=r"length must be >= 2"):
        normalize_expr_rhs(spec_short)

    spec_missing = {
        "kind": "expr",
        "state": ["S", "R"],
        "equations": {"S": "0.0", "R": "0.0"},
        "chain": [{"name": "I", "length": 2, "forward": "gamma"}],
    }
    with pytest.raises(ValueError, match=r"references missing states"):
        normalize_expr_rhs(spec_missing)


def test_transitions_requires_nonempty_list() -> None:
    """Test that transitions kind requires non-empty transitions list."""
    spec = {"kind": "transitions", "state": ["S", "I"], "transitions": []}
    with pytest.raises(
        ValueError, match=re.escape("transitions must be a non-empty list")
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
        "state": ["S", "I1", "I2"],
        "transitions": [{"from": "S", "to": "I1", "rate": "beta"}],
        "chain": [{"name": "I", "length": 2, "forward": "gamma", "to": "R"}],
    }
    with pytest.raises(ValueError, match=r"to='R' not in state"):
        normalize_transitions_rhs(spec)


def test_normalize_rhs_unsupported_kind() -> None:
    """Test that unsupported RHS kinds raise UnsupportedFeatureError."""
    spec = {"kind": "pde", "state": ["u"], "equations": {"u": "0.0"}}
    with pytest.raises(NotImplementedError, match=r"Only 'expr' and 'transitions'"):
        normalize_rhs(spec)
