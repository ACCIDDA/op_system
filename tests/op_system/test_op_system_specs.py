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

import numpy as np
import pytest

from op_system import compile_rhs
from op_system._constraints import _ConstraintRule, _normalize_constraints
from op_system._errors import InvalidRhsSpecError
from op_system._ir import Apply, Reduce, free_symbols, parse_expr_to_ir, walk
from op_system._normalize import _derive_alias_strings, _derive_equation_strings
from op_system.specs import (
    NormalizedRhs,
    StateTemplate,
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


def test_derive_equation_strings_requires_typed_ir() -> None:
    """Equation rendering should fail fast when typed IR is missing."""
    with pytest.raises(InvalidRhsSpecError, match=r"equations\[0\] is missing typed IR"):
        _derive_equation_strings((None,))


def test_derive_alias_strings_requires_typed_ir() -> None:
    """Alias rendering should fail fast when typed IR is missing."""
    with pytest.raises(InvalidRhsSpecError, match="alias 'N' is missing typed IR"):
        _derive_alias_strings({}, {"N": "S + I + R"})


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


def test_expr_template_expansion_and_apply_along() -> None:
    """Templates over categorical axes expand state and equations."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1", "p2"]}],
        "state": ["S[pop]", "I[pop]"],
        "equations": {
            "S[pop]": "-beta * S[pop] * apply_along(I[pop:j], pop=j)",
            "I[pop]": "beta * S[pop] * apply_along(I[pop:j], pop=j) - gamma * I[pop]",
        },
    }

    out = normalize_expr_rhs(spec)
    assert set(out.state_names) == {"S__pop_p1", "S__pop_p2", "I__pop_p1", "I__pop_p2"}
    # apply_along should be unrolled to explicit sums
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

    # Aliases are inlined into equations (IR-derived strings); the templated
    # alias *bodies* — and the LHS cells they multiply — must appear per axis.
    assert any("offset[0]" in eq and "S__age_y" in eq for eq in out.equations)
    assert any("offset[1]" in eq and "S__age_o" in eq for eq in out.equations)

    # ``offset[age]`` is now a shaped parameter (not per-coord scalars and
    # not included in ``param_names`` — those list scalar params only).
    expected_params = {"b0", "gamma", "k_base"}
    assert set(out.param_names) == expected_params
    assert dict(out.shaped_params) == {"offset": ("age",)}


def test_shaped_param_single_axis_rewrites_to_literal_subscript() -> None:
    """A bare ``theta[ax]`` ref becomes a literal ``theta[idx]`` subscript."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "imm", "coords": ["x0", "x1", "x2"]}],
        "state": ["S[imm]"],
        "equations": {"S[imm]": "-theta[imm] * S[imm]"},
    }
    out = normalize_expr_rhs(spec)
    assert dict(out.shaped_params) == {"theta": ("imm",)}
    assert out.equations == (
        "-theta[0] * S__imm_x0",
        "-theta[1] * S__imm_x1",
        "-theta[2] * S__imm_x2",
    )
    # ``theta`` must not appear in the per-coord scalar param list.
    assert "theta" not in out.param_names


def test_shaped_param_multi_axis_rewrites_in_axis_order() -> None:
    """Multi-axis shaped params emit indices in the registered axis order."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
        ],
        "state": ["S[age, vax]"],
        "equations": {"S[age, vax]": "-phi[age, vax] * S[age, vax]"},
    }
    out = normalize_expr_rhs(spec)
    assert dict(out.shaped_params) == {"phi": ("age", "vax")}
    assert "-phi[0, 0] * S__age_y__vax_u" in out.equations
    assert "-phi[1, 1] * S__age_o__vax_v" in out.equations


def test_shaped_param_inconsistent_axes_rejected() -> None:
    """Inconsistent shaped-param axes across references must error."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
        ],
        "state": ["S[age, vax]"],
        "equations": {
            "S[age, vax]": "-phi[age, vax] * S[age, vax] + phi[vax, age]",
        },
    }
    with pytest.raises(InvalidRhsSpecError, match="inconsistent axes"):
        normalize_expr_rhs(spec)


def test_shaped_param_in_alias_body_propagates() -> None:
    """Shaped-param subscripts in alias bodies survive to expanded equations."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "imm", "coords": ["x0", "x1"]}],
        "state": ["S[imm]"],
        "aliases": {"k[imm]": "kappa[imm] * 2"},
        "equations": {"S[imm]": "-k[imm] * S[imm]"},
    }
    out = normalize_expr_rhs(spec)
    assert dict(out.shaped_params) == {"kappa": ("imm",)}
    assert out.aliases == {
        "k__imm_x0": "kappa[0] * 2",
        "k__imm_x1": "kappa[1] * 2",
    }


def test_shaped_param_in_transitions_rate() -> None:
    """Shaped params work the same way inside ``transitions``-kind specs."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "imm", "coords": ["x0", "x1", "x2"]}],
        "state": ["S[imm]", "I[imm]"],
        "transitions": [
            {"from": "S[imm]", "to": "I[imm]", "rate": "theta[imm] * I[imm]"},
        ],
    }
    out = normalize_rhs(spec)
    assert dict(out.shaped_params) == {"theta": ("imm",)}
    expanded = out.meta["transitions"]
    rates = sorted(tr["rate"] for tr in expanded)
    assert rates == [
        "theta[0] * I__imm_x0",
        "theta[1] * I__imm_x1",
        "theta[2] * I__imm_x2",
    ]


def test_shaped_param_eval_end_to_end_scalar_backend() -> None:
    """Compile + eval an RHS that uses a shaped parameter end-to-end."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "imm", "coords": ["x0", "x1", "x2"]}],
        "state": ["S[imm]"],
        "equations": {"S[imm]": "-theta[imm] * S[imm]"},
    }
    rhs = normalize_rhs(spec)
    assert "theta" not in rhs.param_names
    c = compile_rhs(rhs, xp=np)
    y = np.array([10.0, 20.0, 40.0])
    out = c.eval_fn(0.0, y, theta=np.array([0.1, 0.2, 0.5]))
    assert np.allclose(out, np.array([-1.0, -4.0, -20.0]))


def test_same_axis_twice_apply_along_per_row_contraction() -> None:
    """Regression for #107 — ``K[age, age=ap]`` inside an ``apply_along``.

    The bare ``age`` is the LHS-free axis (one row of ``foi[age]``); the
    ``age=ap`` is the bound apply_along coord. The expansion must produce
    a per-row contraction over the off-diagonal entries of K, not collapse
    both positions onto the bound coord (which prior to the fix produced
    only the diagonal of K, identical for every row of foi).
    """
    spec = {
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["a0", "a1", "a2"]}],
        "state": ["I[age]", "foi[age]"],
        "equations": {
            "I[age]": "0.0",
            "foi[age]": "apply_along(K[age, age:ap] * I[age:ap], age=ap)",
        },
    }
    rhs = normalize_rhs(spec)
    assert dict(rhs.shaped_params) == {"K": ("age", "age")}
    foi_eqs = rhs.equations[3:6]
    expected = [
        "K[0, 0] * I__age_a0 + K[0, 1] * I__age_a1 + K[0, 2] * I__age_a2",
        "K[1, 0] * I__age_a0 + K[1, 1] * I__age_a1 + K[1, 2] * I__age_a2",
        "K[2, 0] * I__age_a0 + K[2, 1] * I__age_a1 + K[2, 2] * I__age_a2",
    ]
    assert list(foi_eqs) == expected


def test_same_axis_twice_apply_along_eval_end_to_end() -> None:
    """Regression for #107 — compile + eval the same-axis-twice contraction.

    With ``K = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]`` and ``I = [10, 20, 30]``,
    the per-row contraction ``foi[i] = sum_j K[i, j] * I[j]`` should give
    ``[140, 320, 500]``.
    """
    spec = {
        "kind": "expr",
        "axes": [{"name": "age", "coords": ["a0", "a1", "a2"]}],
        "state": ["I[age]", "foi[age]"],
        "equations": {
            "I[age]": "0.0",
            "foi[age]": "apply_along(K[age, age:ap] * I[age:ap], age=ap)",
        },
    }
    rhs = normalize_rhs(spec)
    assert "K" not in rhs.param_names
    c = compile_rhs(rhs, xp=np)
    k_mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = np.array([10.0, 20.0, 30.0, 0.0, 0.0, 0.0])
    out = c.eval_fn(0.0, y, K=k_mat)
    # First three derivatives are I[age] = 0; last three are foi[age].
    assert np.allclose(out[:3], 0.0)
    assert np.allclose(out[3:], np.array([140.0, 320.0, 500.0]))


def test_same_axis_twice_in_templated_alias_substituted_into_transition() -> None:
    """Regression — same-axis-twice survives alias→transition substitution.

    Mirrors `test_same_axis_twice_apply_along_per_row_contraction` but the
    same-axis-twice contraction lives inside a *templated alias*
    (``foi[age]``) that is referenced from a transition rate
    (``foi[age] * theta[imm]``). Before the fix the alias's
    ``apply_along`` was helper-expanded once *without* an LHS row binding,
    which collapsed ``K[age, age=ap]`` onto the bound coord and leaked the
    full Cartesian product of K cells (e.g. ``K__age_a0__age_a0``) into
    ``param_names`` instead of preserving the per-row contraction.
    """
    spec = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["a0", "a1", "a2"]},
            {"name": "vax", "coords": ["u", "v"]},
            {"name": "imm", "type": "ordinal", "coords": ["x0", "x1"]},
        ],
        "state": ["I[age, vax]", "X[age, vax, imm]", "E[age, vax]"],
        "aliases": {
            "foi[age]": (
                "apply_along(K[age, age:ap]"
                " * apply_along(I[age:ap, vax:v], vax=v), age=ap)"
            ),
        },
        "transitions": [
            {
                "from": "X[age, vax, imm]",
                "to": "E[age, vax]",
                "rate": "foi[age] * theta[imm]",
            },
        ],
    }
    rhs = normalize_transitions_rhs(spec)
    assert dict(rhs.shaped_params) == {"K": ("age", "age"), "theta": ("imm",)}
    leaked = [n for n in rhs.param_names if "K" in n or n.startswith("foi")]
    assert leaked == [], leaked


def test_alias_subscript_axis_token_does_not_leak_into_param_names() -> None:
    """Axis names that appear only as subscripts in aliases are not params.

    Regression for the case where a non-time axis token (e.g. ``loc``)
    used in an alias like ``foi: r0_loc[loc] * ...`` was incorrectly
    surfaced in ``NormalizedRhs.param_names`` because
    ``_collect_alias_symbols`` walks every ``ast.Name`` (descending into
    ``Subscript.slice``) and the previous filter only excluded the time
    axis via ``time_varying_params``.
    """
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b", "c"]}],
        "state": ["S[loc]"],
        "aliases": {"foi": "beta * r0_loc[loc]"},
        "equations": {"S[loc]": "-foi * S[loc]"},
    }
    rhs = normalize_expr_rhs(spec)
    assert "loc" not in rhs.param_names
    assert dict(rhs.shaped_params) == {"r0_loc": ("loc",)}
    assert set(rhs.param_names) == {"beta"}


def test_transition_rate_axis_token_does_not_leak_into_param_names() -> None:
    """Same regression as above but for the ``transitions`` rhs kind."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["S[loc]", "I[loc]"],
        "transitions": [
            {"from": "S[loc]", "to": "I[loc]", "rate": "beta * r0_loc[loc]"}
        ],
    }
    rhs = normalize_transitions_rhs(spec)
    assert "loc" not in rhs.param_names
    assert dict(rhs.shaped_params) == {"r0_loc": ("loc",)}
    assert set(rhs.param_names) == {"beta"}


def test_unexpanded_alias_template_state_base_does_not_leak() -> None:
    """A bare templated-state base in a non-templated alias is not a param.

    ``_collect_alias_symbols`` walks every ``ast.Name`` and only the alias
    *names* matching a template are expanded per coord by
    ``_expand_alias_templates``.  An unexpanded alias like
    ``foi: beta * sum_state(S[loc])`` therefore yields the bare
    ``ast.Name('S')`` -- which previously fell through into ``param_names``
    because the filter only excluded the post-expansion state names
    (``S__loc_a`` etc.), not the template base.
    """
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["S[loc]"],
        "aliases": {"foi": "beta * sum_state(S[loc])"},
        "equations": {"S[loc]": "-foi"},
    }
    rhs = normalize_expr_rhs(spec)
    assert "S" not in rhs.param_names
    assert "sum_state" not in rhs.param_names
    assert set(rhs.param_names) == {"beta"}


def test_unexpanded_alias_template_alias_base_does_not_leak() -> None:
    """A bare templated-alias base in a non-templated alias is not a param."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["S[loc]"],
        "aliases": {
            "k[loc]": "k_base * mult[loc]",
            "foi": "beta * sum_state(k[loc])",
        },
        "equations": {"S[loc]": "-foi"},
    }
    rhs = normalize_expr_rhs(spec)
    assert "k" not in rhs.param_names
    assert "sum_state" not in rhs.param_names
    assert dict(rhs.shaped_params) == {"mult": ("loc",)}
    assert set(rhs.param_names) == {"beta", "k_base"}


def test_time_varying_scalar_records_meta_and_excludes_param() -> None:
    """A param subscripted with the time axis becomes time-varying."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "imm", "coords": ["x0", "x1"]},
            {
                "name": "time",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 2.0},
                "size": 3,
            },
        ],
        "state": ["S[imm]"],
        "equations": {"S[imm]": "-beta[time] * S[imm]"},
    }
    rhs = normalize_rhs(spec)
    assert rhs.time_varying_params == (("beta", ("time",)),)
    assert rhs.meta["time_varying_params"] == (("beta", ("time",)),)
    # Tv shaped names live in shaped_params with the time axis stripped.
    assert dict(rhs.shaped_params) == {"beta": ()}
    assert "beta" not in rhs.param_names
    c = compile_rhs(rhs, xp=np)
    # The compile contract is now a single bare-name kwarg per tv param.
    assert "beta_grid" not in c.param_names
    assert "beta_ts" not in c.param_names


def test_time_varying_scalar_eval_interpolates() -> None:
    """Compile + eval interpolates a time-varying scalar at run time."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "imm", "coords": ["x0", "x1"]},
            {
                "name": "time",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 2.0},
                "size": 3,
            },
        ],
        "state": ["S[imm]"],
        "equations": {"S[imm]": "-beta[time] * S[imm]"},
    }
    c = compile_rhs(normalize_rhs(spec), xp=np)
    grid = np.array([0.1, 0.3, 0.5])  # values at time=[0, 1, 2]
    y = np.array([10.0, 20.0])
    # At t=0.5 -> beta = 0.2
    out = c.eval_fn(0.5, y, beta=grid)
    assert np.allclose(out, np.array([-2.0, -4.0]))


def test_time_varying_shaped_eval_interpolates_per_axis() -> None:
    """A time-varying shaped param interpolates per axis cell at run time."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["young", "old"]},
            {
                "name": "time",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 1.0},
                "size": 2,
            },
        ],
        "state": ["S[age]"],
        "equations": {"S[age]": "-nu[time, age] * S[age]"},
    }
    rhs = normalize_rhs(spec)
    assert rhs.time_varying_params == (("nu", ("time", "age")),)
    # The reduced shape (axes minus time) is what shaped_params records.
    assert dict(rhs.shaped_params) == {"nu": ("age",)}
    c = compile_rhs(rhs, xp=np)
    grid = np.array([[0.1, 0.3], [0.5, 0.7]])  # shape (n_time=2, n_age=2)
    y = np.array([10.0, 20.0])
    # At t=0.5 -> nu = (0.3, 0.5)
    out = c.eval_fn(0.5, y, nu=grid)
    assert np.allclose(out, np.array([-3.0, -10.0]))


def test_time_varying_axis_can_be_trailing() -> None:
    """The time axis can appear in any position of a tv param's shape."""
    spec = {
        "kind": "expr",
        "axes": [
            {"name": "age", "coords": ["young", "old"]},
            {
                "name": "time",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 1.0},
                "size": 2,
            },
        ],
        "state": ["S[age]"],
        "equations": {"S[age]": "-nu[age, time] * S[age]"},
    }
    rhs = normalize_rhs(spec)
    assert rhs.time_varying_params == (("nu", ("age", "time")),)
    c = compile_rhs(rhs, xp=np)
    grid = np.array([[0.1, 0.5], [0.3, 0.7]])  # shape (n_age=2, n_time=2)
    y = np.array([10.0, 20.0])
    # At t=0.5 -> nu interp along trailing axis = (0.3, 0.5).
    out = c.eval_fn(0.5, y, nu=grid)
    assert np.allclose(out, np.array([-3.0, -10.0]))


def test_legacy_time_varying_field_rejected() -> None:
    """The legacy ``time_varying`` field is rejected with a migration hint."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "imm", "coords": ["x0"]}],
        "state": ["S[imm]"],
        "equations": {"S[imm]": "-beta * S[imm]"},
        "time_varying": ["beta"],
    }
    with pytest.raises(InvalidRhsSpecError, match=r"time_varying.*has been removed"):
        normalize_rhs(spec)


def test_time_varying_in_transitions_kind() -> None:
    """The implicit time-axis subscript also works for the transitions kind."""
    spec = {
        "kind": "transitions",
        "axes": [
            {"name": "imm", "coords": ["x0", "x1"]},
            {
                "name": "time",
                "type": "continuous",
                "domain": {"lb": 0.0, "ub": 1.0},
                "size": 2,
            },
        ],
        "state": ["S[imm]", "I[imm]"],
        "transitions": [
            {"from": "S[imm]", "to": "I[imm]", "rate": "beta[time]"},
        ],
    }
    rhs = normalize_rhs(spec)
    assert rhs.time_varying_params == (("beta", ("time",)),)
    c = compile_rhs(rhs, xp=np)
    grid = np.array([0.0, 1.0])  # values at time=[0, 1]
    # State order is (S[x0], S[x1], I[x0], I[x1]).
    y = np.array([10.0, 5.0, 0.0, 0.0])
    out = c.eval_fn(0.5, y, beta=grid)
    # beta=0.5; flow=beta*source -> dS=[-5,-2.5], dI=[5,2.5].
    assert np.allclose(out, np.array([-5.0, -2.5, 5.0, 2.5]))


def test_apply_along_kernel_sum_rejects_continuous_axis() -> None:
    """apply_along(kernel=sum) on a continuous axis should raise an error."""
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
        "equations": {"S[age]": "-apply_along(S[age:i], age=i, kernel=sum)"},
    }
    with pytest.raises(ValueError, match=r"requires categorical or ordinal axes"):
        normalize_expr_rhs(spec)


def test_apply_along_integrate_expands_with_continuous_deltas() -> None:
    """apply_along uses trapezoidal weights from continuous axis coords."""
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
            "v": "apply_along(u[x:i], x=i)",
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


def test_apply_along_kernel_integrate_rejects_categorical_axis() -> None:
    """apply_along(kernel=integrate) on a categorical axis should raise an error."""
    spec = {
        "kind": "expr",
        "axes": [
            {
                "name": "g",
                "coords": ["a", "b"],
            }
        ],
        "state": ["x[g]"],
        "equations": {"x[g]": "apply_along(x[g:i], g=i, kernel=integrate)"},
    }

    with pytest.raises(ValueError, match=r"requires continuous axes"):
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
            {
                "name": "op0",
                "axis": "x",
                "kind": "advection",
                "bc": "periodic",
                "velocity": "v",
            },
        ],
        "state": ["u"],
        "equations": {"u": "0.0"},
    }

    out = normalize_expr_rhs(spec_with_bc)
    operators_meta = out.meta.get("operators")
    assert isinstance(operators_meta, list)
    assert operators_meta[0]["kind"] == "advection"
    assert operators_meta[0]["bc"] == "periodic"


def test_operator_apply_to_validation() -> None:
    """Operator apply_to should be validated against known top-level states."""
    spec_bad_apply_to = {
        "kind": "expr",
        "axes": [{"name": "x", "coords": ["a"]}],
        "state": ["S", "I"],
        "operators": [
            {
                "name": "adv",
                "axis": "x",
                "kind": "advection",
                "velocity": "v",
                "apply_to": ["X"],
            }
        ],
        "equations": {
            "S": "0.0",
            "I": "0.0",
        },
    }
    with pytest.raises(ValueError, match=r"apply_to entry 'X' not in state"):
        normalize_expr_rhs(spec_bad_apply_to)


def test_operator_advection_requires_velocity() -> None:
    """Advection/transport operators must include a velocity field."""
    spec_missing_velocity = {
        "kind": "expr",
        "axes": [{"name": "x", "coords": ["a"]}],
        "state": ["S"],
        "operators": [
            {
                "name": "adv",
                "axis": "x",
                "kind": "advection",
                "apply_to": ["S"],
            }
        ],
        "equations": {"S": "0.0"},
    }
    with pytest.raises(ValueError, match=r"velocity is required"):
        normalize_expr_rhs(spec_missing_velocity)


def test_operator_jump_integral_requires_kernel_and_normalizes_direction() -> None:
    """jump_integral operators require kernel/rate and normalize direction."""
    spec_bad_jump = {
        "kind": "expr",
        "axes": [{"name": "x", "coords": ["a"]}],
        "state": ["S"],
        "operators": [
            {
                "name": "jump",
                "axis": "x",
                "kind": "jump_integral",
                "rate": "nu",
                "apply_to": ["S"],
            }
        ],
        "equations": {"S": "0.0"},
    }
    with pytest.raises(ValueError, match=r"kernel must be a mapping"):
        normalize_expr_rhs(spec_bad_jump)

    spec_ok_jump = {
        "kind": "expr",
        "axes": [{"name": "x", "coords": ["a"]}],
        "state": ["S"],
        "operators": [
            {
                "name": "jump",
                "axis": "x",
                "kind": "jump_integral",
                "rate": "nu",
                "kernel": {"form": "gaussian", "params": {"sigma": 0.1}},
                "direction": "UP",
                "apply_to": ["S"],
            }
        ],
        "equations": {"S": "0.0"},
    }
    out = normalize_expr_rhs(spec_ok_jump)
    ops = out.meta.get("operators")
    assert isinstance(ops, list)
    assert ops[0]["kind"] == "jump_integral"
    assert ops[0]["direction"] == "up"


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


def test_transitions_template_endpoint_pinning_expands_correctly() -> None:
    """Endpoint pinning keeps pinned axis fixed while expanding other axes."""
    spec = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["u65", "o65"]},
            {"name": "imm", "coords": ["X0", "X1"]},
        ],
        "state": ["I[age,imm]", "X[age,imm]"],
        "transitions": [
            {
                "from": "I[age,imm]",
                "to": "X[age,imm=X1]",
                "rate": "waning[imm]",
            }
        ],
    }

    out = normalize_transitions_rhs(spec)

    assert "X__age_u65__imm_X1" in out.state_names
    assert "X__age_o65__imm_X1" in out.state_names
    assert any("-" in eq and "I__age_u65__imm_X0" in eq for eq in out.equations)
    assert any(
        "I__age_u65__imm_X0" in eq and "I__age_u65__imm_X1" in eq
        for eq in out.equations
    )


def test_transitions_chain_helper_supports_templated_chain_names() -> None:
    """Transitions chains accept template names and expand stages over axes."""
    spec = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["a", "b"]},
            {"name": "vax", "coords": ["u", "v"]},
        ],
        "state": ["E[age,vax]", "R[age,vax]"],
        "transitions": [],
        "chain": [
            {
                "name": "I[age,vax]",
                "length": 3,
                "entry": {"from": "E[age,vax]", "rate": "sigma"},
                "forward": ["g12", "g23"],
                "exit": {"to": "R[age,vax]", "rate": "g3r"},
            }
        ],
    }

    out = normalize_transitions_rhs(spec)

    assert "I1__age_a__vax_u" in out.state_names
    assert "I3__age_b__vax_v" in out.state_names
    # 4 axis combinations * (entry + 2 internal + exit) transitions per combo.
    assert len(out.meta["transitions"]) == 16
    rates = {tr["rate"] for tr in out.meta["transitions"]}
    assert rates == {"sigma", "g12", "g23", "g3r"}


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
        _ConstraintRule(
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
        _ConstraintRule(
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


# -- initial_state template expansion ------------------------------------------


def test_transitions_no_initial_state_omits_meta_key() -> None:
    """When initial_state is absent, meta has no initial_state key."""
    spec = {
        "kind": "transitions",
        "state": ["S", "I", "R"],
        "transitions": [
            {"from": "S", "to": "I", "rate": "beta"},
            {"from": "I", "to": "R", "rate": "gamma"},
        ],
    }
    out = normalize_transitions_rhs(spec)
    assert "initial_state" not in out.meta


def test_transitions_scalar_initial_state() -> None:
    """Scalar (non-templated) initial_state passes through unchanged."""
    spec = {
        "kind": "transitions",
        "state": ["S", "I", "R"],
        "transitions": [
            {"from": "S", "to": "I", "rate": "beta"},
            {"from": "I", "to": "R", "rate": "gamma"},
        ],
        "initial_state": {"S": "S0", "I": "I0", "R": "R0"},
    }
    out = normalize_transitions_rhs(spec)
    assert out.meta["initial_state"] == {"S": "S0", "I": "I0", "R": "R0"}


def test_transitions_templated_initial_state_expands() -> None:
    """Templated keys and values expand across axis coordinates."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "vax", "coords": ["u", "v"]}],
        "state": ["S[vax]", "D"],
        "transitions": [
            {"from": "S[vax]", "to": "D", "rate": "mu"},
        ],
        "initial_state": {"S[vax]": "S0[vax]", "D": "D0"},
    }
    out = normalize_transitions_rhs(spec)
    expected = {
        "S__vax_u": "S0__vax_u",
        "S__vax_v": "S0__vax_v",
        "D": "D0",
    }
    assert out.meta["initial_state"] == expected


def test_transitions_multi_axis_initial_state() -> None:
    """Initial state expands over multiple axes."""
    spec = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
        ],
        "state": ["S[age,vax]"],
        "transitions": [
            {"from": "S[age,vax]", "to": "S[age,vax]", "rate": "alpha"},
        ],
        "initial_state": {"S[age,vax]": "S0[age,vax]"},
    }
    out = normalize_transitions_rhs(spec)
    ic = out.meta["initial_state"]
    assert len(ic) == 4
    assert ic["S__age_y__vax_u"] == "S0__age_y__vax_u"
    assert ic["S__age_o__vax_v"] == "S0__age_o__vax_v"


def test_expr_no_initial_state_omits_meta_key() -> None:
    """When initial_state is absent, meta has no initial_state key (expr)."""
    spec = {
        "kind": "expr",
        "state": ["S", "I", "R"],
        "equations": {"S": "-beta*S", "I": "beta*S - gamma*I", "R": "gamma*I"},
    }
    out = normalize_expr_rhs(spec)
    assert "initial_state" not in out.meta


def test_expr_scalar_initial_state() -> None:
    """Scalar initial_state passes through in expr specs."""
    spec = {
        "kind": "expr",
        "state": ["S", "I"],
        "equations": {"S": "-S", "I": "S"},
        "initial_state": {"S": "S0", "I": "I0"},
    }
    out = normalize_expr_rhs(spec)
    assert out.meta["initial_state"] == {"S": "S0", "I": "I0"}


def test_expr_templated_initial_state_expands() -> None:
    """Templated initial_state expands in expr specs."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["u[loc]"],
        "equations": {"u[loc]": "-u[loc]"},
        "initial_state": {"u[loc]": "u0[loc]"},
    }
    out = normalize_expr_rhs(spec)
    expected = {"u__loc_a": "u0__loc_a", "u__loc_b": "u0__loc_b"}
    assert out.meta["initial_state"] == expected


# ---------------------------------------------------------------------------
# coord_shift tests
# ---------------------------------------------------------------------------


def test_coord_shift_single_axis() -> None:
    """A single-axis coord_shift expands to concrete transitions."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "vax", "coords": ["u", "v"]}],
        "state": ["S[vax]", "R[vax]"],
        "transitions": [
            {"from": "S[vax]", "to": "R[vax]", "rate": "gamma"},
            {
                "coord_shift": {"vax": "u -> v"},
                "apply_to": ["S", "R"],
                "rate": "nu",
            },
        ],
    }
    out = normalize_transitions_rhs(spec)

    tr = out.meta["transitions"]
    # Original template expands: 2 states x 1 transition = 2.
    # coord_shift: 2 apply_to bases x 1 from_coord = 2.
    # Total: 4 expanded transitions.
    assert len(tr) == 4

    shift_pairs = [(t["from"], t["to"]) for t in tr if t["rate"] == "nu"]
    assert ("S__vax_u", "S__vax_v") in shift_pairs
    assert ("R__vax_u", "R__vax_v") in shift_pairs
    assert len(shift_pairs) == 2


def test_coord_shift_multi_axis_expands_over_non_shifted() -> None:
    """coord_shift expands over non-shifted axis coords."""
    spec = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
        ],
        "state": ["S[age,vax]"],
        "transitions": [
            {
                "coord_shift": {"vax": "u -> v"},
                "apply_to": ["S"],
                "rate": "nu",
            },
        ],
    }
    out = normalize_transitions_rhs(spec)

    tr = out.meta["transitions"]
    shift_pairs = [(t["from"], t["to"]) for t in tr]
    assert ("S__age_y__vax_u", "S__age_y__vax_v") in shift_pairs
    assert ("S__age_o__vax_u", "S__age_o__vax_v") in shift_pairs
    assert len(shift_pairs) == 2


def test_coord_shift_preserves_regular_transitions() -> None:
    """Regular transitions coexist with coord_shift entries."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "vax", "coords": ["u", "v"]}],
        "state": ["S[vax]", "I[vax]", "R[vax]"],
        "aliases": {"N": "S__vax_u + S__vax_v + I__vax_u + I__vax_v"},
        "transitions": [
            {"from": "S[vax]", "to": "I[vax]", "rate": "beta"},
            {"from": "I[vax]", "to": "R[vax]", "rate": "gamma"},
            {
                "coord_shift": {"vax": "u -> v"},
                "apply_to": ["S", "R"],
                "rate": "nu",
            },
        ],
    }
    out = normalize_transitions_rhs(spec)

    tr = out.meta["transitions"]
    # 2 template transitions * 2 coords + 2 coord_shift = 6
    assert len(tr) == 6


def test_coord_shift_rejects_missing_axis() -> None:
    """coord_shift with an unknown axis raises."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "vax", "coords": ["u", "v"]}],
        "state": ["S[vax]"],
        "transitions": [
            {
                "coord_shift": {"age": "y -> o"},
                "apply_to": ["S"],
                "rate": "nu",
            },
        ],
    }
    with pytest.raises(ValueError, match=r"axis.*age.*not defined"):
        normalize_transitions_rhs(spec)


def test_coord_shift_rejects_bad_coord() -> None:
    """coord_shift with a coordinate not in the axis raises."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "vax", "coords": ["u", "v"]}],
        "state": ["S[vax]"],
        "transitions": [
            {
                "coord_shift": {"vax": "u -> z"},
                "apply_to": ["S"],
                "rate": "nu",
            },
        ],
    }
    with pytest.raises(ValueError, match=r"coordinate.*z.*not in"):
        normalize_transitions_rhs(spec)


def test_coord_shift_rejects_missing_apply_to() -> None:
    """coord_shift without apply_to raises."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "vax", "coords": ["u", "v"]}],
        "state": ["S[vax]"],
        "transitions": [
            {
                "coord_shift": {"vax": "u -> v"},
                "rate": "nu",
            },
        ],
    }
    with pytest.raises(ValueError, match="apply_to"):
        normalize_transitions_rhs(spec)


def test_coord_shift_rejects_missing_rate() -> None:
    """coord_shift without rate raises."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "vax", "coords": ["u", "v"]}],
        "state": ["S[vax]"],
        "transitions": [
            {
                "coord_shift": {"vax": "u -> v"},
                "apply_to": ["S"],
            },
        ],
    }
    with pytest.raises(ValueError, match="rate"):
        normalize_transitions_rhs(spec)


def test_coord_shift_rejects_bad_arrow_syntax() -> None:
    """coord_shift with malformed arrow raises."""
    spec = {
        "kind": "transitions",
        "axes": [{"name": "vax", "coords": ["u", "v"]}],
        "state": ["S[vax]"],
        "transitions": [
            {
                "coord_shift": {"vax": "u to v"},
                "apply_to": ["S"],
                "rate": "nu",
            },
        ],
    }
    with pytest.raises(ValueError, match="from_coord -> to_coord"):
        normalize_transitions_rhs(spec)


# ---------------------------------------------------------------------------
# Pinned selector integration tests (#82 acceptance criteria)
# ---------------------------------------------------------------------------


def test_initial_state_pinned_key_expands_correctly() -> None:
    """initial_state key with pinned axis generates only fixed-coord entries."""
    spec = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
            {"name": "imm", "coords": ["X0", "X1"]},
        ],
        "state": ["S[age,vax,imm]"],
        "transitions": [
            {"from": "S[age,vax,imm]", "to": "S[age,vax,imm]", "rate": "alpha"},
        ],
        # Pin imm=X0; should expand age x vax but not imm.
        "initial_state": {"S[age,vax,imm=X0]": "S0[age,vax,imm=X0]"},
    }
    out = normalize_transitions_rhs(spec)
    ic = out.meta["initial_state"]
    # 2 age x 2 vax x 1 pinned imm = 4 entries.
    assert len(ic) == 4
    for key in ic:
        assert "imm_X0" in key
        assert "imm_X1" not in key
    assert "S__age_y__vax_u__imm_X0" in ic
    assert "S__age_o__vax_v__imm_X0" in ic


def test_initial_state_pinned_key_invalid_coord_rejected() -> None:
    """initial_state key with an unknown pinned coord raises."""
    spec = {
        "kind": "transitions",
        "axes": [
            {"name": "imm", "coords": ["X0", "X1"]},
        ],
        "state": ["S[imm]"],
        "transitions": [{"from": "S[imm]", "to": "S[imm]", "rate": "alpha"}],
        "initial_state": {"S[imm=X99]": "S0"},
    }
    with pytest.raises(ValueError, match=r"pinned coord"):
        normalize_transitions_rhs(spec)


# -- shaped initial_state values -----------------------------------------------


def _shaped_ic_axes_spec() -> dict[str, object]:
    return {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
            {"name": "imm", "coords": ["X0", "X1"]},
        ],
        "state": ["X[age,vax,imm]"],
        "transitions": [
            {"from": "X[age,vax,imm]", "to": "X[age,vax,imm]", "rate": "alpha"},
        ],
    }


def test_initial_state_shaped_full_wildcards() -> None:
    """Shaped IC over all-wildcard LHS records (name, axes, coords) per cell."""
    spec = _shaped_ic_axes_spec()
    spec["initial_state"] = {
        "X[age,vax,imm]": {"shaped": "x_init", "axes": ["age", "vax", "imm"]},
    }
    out = normalize_transitions_rhs(spec)
    ic = out.meta["initial_state"]
    # 2 * 2 * 2 = 8 cells.
    assert len(ic) == 8
    entry = ic["X__age_y__vax_u__imm_X0"]
    assert entry == {
        "shaped": "x_init",
        "axes": ("age", "vax", "imm"),
        "coords": {"age": "y", "vax": "u", "imm": "X0"},
    }
    # Distinct cells get distinct coord assignments but share name + axes.
    other = ic["X__age_o__vax_v__imm_X1"]
    assert other["shaped"] == "x_init"
    assert other["axes"] == ("age", "vax", "imm")
    assert other["coords"] == {"age": "o", "vax": "v", "imm": "X1"}


def test_initial_state_shaped_partial_with_pinned_axis() -> None:
    """Shaped IC works when LHS pins one axis (covers a slice only)."""
    spec = _shaped_ic_axes_spec()
    spec["initial_state"] = {
        "X[age,vax=u,imm]": {"shaped": "x_unvax_init", "axes": ["age", "imm"]},
        "X[age,vax=v,imm]": {"shaped": "x_vax_init", "axes": ["age", "imm"]},
    }
    out = normalize_transitions_rhs(spec)
    ic = out.meta["initial_state"]
    assert len(ic) == 8
    unvax = ic["X__age_y__vax_u__imm_X0"]
    assert unvax["shaped"] == "x_unvax_init"
    assert unvax["axes"] == ("age", "imm")
    assert unvax["coords"] == {"age": "y", "imm": "X0"}
    vax = ic["X__age_o__vax_v__imm_X1"]
    assert vax["shaped"] == "x_vax_init"
    assert vax["coords"] == {"age": "o", "imm": "X1"}


def test_initial_state_shaped_axis_order_independent_of_lhs() -> None:
    """Shaped axes preserve the user's declared order, not the LHS order."""
    spec = _shaped_ic_axes_spec()
    spec["initial_state"] = {
        "X[age,vax,imm]": {"shaped": "x_init", "axes": ["imm", "age", "vax"]},
    }
    out = normalize_transitions_rhs(spec)
    ic = out.meta["initial_state"]
    entry = ic["X__age_y__vax_u__imm_X0"]
    assert entry["axes"] == ("imm", "age", "vax")
    assert entry["coords"] == {"age": "y", "vax": "u", "imm": "X0"}


def test_initial_state_shaped_axis_not_bound_by_lhs_rejected() -> None:
    """Shaped axes must all be wildcards or pinned coords on the LHS."""
    spec = _shaped_ic_axes_spec()
    spec["initial_state"] = {
        # LHS has no `vax` token at all (ill-formed selector for the X
        # state, but exercised here only to confirm the missing-axis error
        # path; expand_selector will reject the LHS first if X requires vax.)
        "X[age,imm]": {"shaped": "x_init", "axes": ["age", "vax", "imm"]},
    }
    with pytest.raises(ValueError, match=r"shaped"):
        normalize_transitions_rhs(spec)


def test_initial_state_shaped_unknown_axis_rejected() -> None:
    """A shaped IC entry naming an axis absent from the spec is rejected."""
    spec = _shaped_ic_axes_spec()
    spec["initial_state"] = {
        "X[age,vax,imm]": {"shaped": "x_init", "axes": ["age", "bogus"]},
    }
    with pytest.raises(ValueError, match=r"shaped axis 'bogus' not defined"):
        normalize_transitions_rhs(spec)


def test_initial_state_shaped_missing_name_rejected() -> None:
    """A shaped IC entry without a ``shaped`` key is rejected."""
    spec = _shaped_ic_axes_spec()
    spec["initial_state"] = {
        "X[age,vax,imm]": {"axes": ["age", "vax", "imm"]},
    }
    with pytest.raises(ValueError, match=r"shaped"):
        normalize_transitions_rhs(spec)


def test_initial_state_shaped_unknown_key_rejected() -> None:
    """A shaped IC entry carrying an unknown key is rejected."""
    spec = _shaped_ic_axes_spec()
    spec["initial_state"] = {
        "X[age,vax,imm]": {
            "shaped": "x_init",
            "axes": ["age", "vax", "imm"],
            "shape": [2, 2, 2],
        },
    }
    with pytest.raises(ValueError, match=r"unknown"):
        normalize_transitions_rhs(spec)


def test_initial_state_shaped_and_scalar_mixed() -> None:
    """A spec may freely mix scalar and shaped IC entries across compartments."""
    spec = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
        ],
        "state": ["S[age,vax]", "I[age,vax]"],
        "transitions": [
            {"from": "S[age,vax]", "to": "I[age,vax]", "rate": "alpha"},
        ],
        "initial_state": {
            "S[age,vax]": {"shaped": "s_init", "axes": ["age", "vax"]},
            "I[age,vax]": "i_init",
        },
    }
    out = normalize_transitions_rhs(spec)
    ic = out.meta["initial_state"]
    assert ic["S__age_y__vax_u"]["shaped"] == "s_init"
    assert ic["I__age_y__vax_u"] == "i_init"


def test_transitions_pinned_from_endpoint_expands_correctly() -> None:
    """Pinned FROM endpoint keeps that axis fixed while expanding others."""
    spec = {
        "kind": "transitions",
        "axes": [
            {"name": "age", "coords": ["y", "o"]},
            {"name": "vax", "coords": ["u", "v"]},
        ],
        "state": ["S[age,vax]", "R[age,vax]"],
        "transitions": [
            # Pin age=y; only young age group transitions.
            {
                "from": "S[age=y, vax]",
                "to": "R[age=y, vax]",
                "rate": "gamma",
            }
        ],
    }
    out = normalize_transitions_rhs(spec)
    trs = out.meta["transitions"]
    # 2 vax combinations, age pinned to y.
    assert len(trs) == 2
    frm_states = {tr["from"] for tr in trs}
    assert frm_states == {"S__age_y__vax_u", "S__age_y__vax_v"}
    to_states = {tr["to"] for tr in trs}
    assert to_states == {"R__age_y__vax_u", "R__age_y__vax_v"}
    # Old-age states exist in state_names but are untouched by this transition.
    assert "S__age_o__vax_u" in out.state_names


def test_state_templates_wildcard_only_records_shape_and_offset() -> None:
    """Pure-wildcard state entries produce a single shaped StateTemplate."""
    spec = {
        "kind": "transitions",
        "state": ["S[age, vax]", "I[age, vax]", "R[age, vax]"],
        "axes": [
            {"name": "age", "kind": "categorical", "coords": ["y", "o"]},
            {"name": "vax", "kind": "categorical", "coords": ["u", "v"]},
        ],
        "transitions": [
            {"from": "S[age, vax]", "to": "I[age, vax]", "rate": "beta"},
        ],
    }
    out = normalize_transitions_rhs(spec)
    assert len(out.state_templates) == 3
    s_tpl = out.state_templates[0]
    assert isinstance(s_tpl, StateTemplate)
    assert s_tpl.base == "S"
    assert s_tpl.axes == ("age", "vax")
    assert s_tpl.shape == (2, 2)
    assert len(s_tpl.expanded_names) == 4
    assert s_tpl.offset == out.state_names.index(s_tpl.expanded_names[0])
    # Templates cover the full state vector contiguously in declared order.
    flat = tuple(n for tpl in out.state_templates for n in tpl.expanded_names)
    assert flat == out.state_names


def test_state_templates_mixed_scalar_and_wildcard() -> None:
    """A bare scalar state next to a wildcard template yields scalar+shaped."""
    spec = {
        "kind": "transitions",
        "state": ["S[age]", "D"],
        "axes": [
            {"name": "age", "kind": "categorical", "coords": ["y", "o"]},
        ],
        "transitions": [
            {"from": "S[age]", "to": "D", "rate": "mu"},
        ],
    }
    out = normalize_transitions_rhs(spec)
    assert len(out.state_templates) == 2
    s_tpl, d_tpl = out.state_templates
    assert s_tpl.axes == ("age",)
    assert s_tpl.shape == (2,)
    assert d_tpl.axes == ()
    assert d_tpl.shape == ()
    assert d_tpl.expanded_names == ("D",)
    assert d_tpl.offset == out.state_names.index("D")


def test_state_templates_pinned_only_treated_as_scalar() -> None:
    """Pinned-only selectors expand to a single scalar StateTemplate."""
    spec = {
        "kind": "transitions",
        "state": ["S[age=y]", "S[age=o]", "I[age]"],
        "axes": [
            {"name": "age", "kind": "categorical", "coords": ["y", "o"]},
        ],
        "transitions": [
            {"from": "S[age]", "to": "I[age]", "rate": "beta"},
        ],
    }
    out = normalize_transitions_rhs(spec)
    assert len(out.state_templates) == 3
    assert out.state_templates[0].axes == ()
    assert out.state_templates[0].shape == ()
    assert out.state_templates[0].expanded_names == ("S__age_y",)
    assert out.state_templates[1].expanded_names == ("S__age_o",)
    assert out.state_templates[2].axes == ("age",)
    assert out.state_templates[2].shape == (2,)


def test_state_templates_expr_kind_populated() -> None:
    """expr-kind RHS also exposes state_templates aligned with state_names."""
    spec = {
        "kind": "expr",
        "state": ["S[age]", "I[age]"],
        "axes": [
            {"name": "age", "kind": "categorical", "coords": ["y", "o"]},
        ],
        "equations": {
            "S[age]": "0",
            "I[age]": "0",
        },
    }
    out = normalize_expr_rhs(spec)
    assert len(out.state_templates) == 2
    flat = tuple(n for tpl in out.state_templates for n in tpl.expanded_names)
    assert flat == out.state_names


# ---------------------------------------------------------------------------
# aliases_ir field (typed IR exposed alongside string aliases)
# ---------------------------------------------------------------------------


def test_aliases_ir_exposes_parsed_ir_for_simple_alias() -> None:
    """``aliases_ir`` should contain a parsed IR Expr for each string alias."""
    spec = {
        "kind": "expr",
        "state": ["S", "I", "R"],
        "aliases": {"N": "S + I + R"},
        "equations": {"S": "0", "I": "0", "R": "0"},
    }
    out = normalize_expr_rhs(spec)
    assert "N" in out.aliases_ir
    assert out.aliases_ir["N"] == parse_expr_to_ir("S + I + R")
    assert isinstance(out.aliases_ir["N"], Apply)
    assert free_symbols(out.aliases_ir["N"]) == {"S", "I", "R"}


def test_aliases_ir_inlines_chained_aliases() -> None:
    """Alias bodies that reference other aliases should be flattened in IR."""
    spec = {
        "kind": "expr",
        "state": ["S", "I"],
        "aliases": {"N": "S + I", "frac": "S / N"},
        "equations": {"S": "0", "I": "0"},
    }
    out = normalize_expr_rhs(spec)
    # ``frac`` should no longer reference ``N`` after inlining.
    assert "N" not in free_symbols(out.aliases_ir["frac"])
    assert {"S", "I"} <= free_symbols(out.aliases_ir["frac"])


def test_aliases_ir_falls_back_for_cyclic_aliases() -> None:
    """Cyclic aliases must not abort normalization; cycle survives at eval time."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "aliases": {"a": "b + 1", "b": "a + 1"},
        "equations": {"x": "a"},
    }
    out = normalize_expr_rhs(spec)
    # Both entries are present (un-inlined) so existing string-based consumers
    # remain authoritative; lazy cycle detection still fires at eval time.
    assert set(out.aliases_ir) == {"a", "b"}


def test_aliases_ir_present_for_transitions_kind() -> None:
    """``transitions`` kind should also populate the typed IR alias map."""
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
    assert "N" in out.aliases_ir


# ---------------------------------------------------------------------------
# equations_ir field (typed IR exposed alongside string equations)
# ---------------------------------------------------------------------------


def test_equations_ir_aligned_with_equations_expr_kind() -> None:
    """``equations_ir`` should be positionally aligned with ``equations``."""
    spec = {
        "kind": "expr",
        "state": ["S", "I", "R"],
        "aliases": {"N": "S + I + R"},
        "equations": {
            "S": "-beta * S * I / N",
            "I": "beta * S * I / N - gamma * I",
            "R": "gamma * I",
        },
    }
    out = normalize_expr_rhs(spec)
    assert len(out.equations_ir) == len(out.equations)
    assert len(out.equations_ir_raw) == len(out.equations)
    assert all(expr is not None for expr in out.equations_ir)
    # Aliases inlined: ``N`` should not appear in any equation IR.
    for expr in out.equations_ir:
        assert expr is not None
        assert "N" not in free_symbols(expr)
    # Raw equation IR is available for consumers that still want alias-buffer
    # references rather than inlined alias bodies.
    assert any(
        expr is not None and "N" in free_symbols(expr) for expr in out.equations_ir_raw
    )


def test_equations_ir_present_for_transitions_kind() -> None:
    """``transitions`` kind should also populate ``equations_ir`` with IR."""
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
    assert len(out.equations_ir) == len(out.equations)
    assert len(out.equations_ir_raw) == len(out.equations)
    assert all(expr is not None for expr in out.equations_ir)
    for expr in out.equations_ir:
        assert expr is not None
        assert "N" not in free_symbols(expr)


def test_equations_ir_round_trip_matches_parse_expr_to_ir() -> None:
    """For simple equations, ``equations_ir`` entries equal direct IR parse."""
    spec = {
        "kind": "expr",
        "state": ["x"],
        "equations": {"x": "a + b"},
    }
    out = normalize_expr_rhs(spec)
    assert out.equations_ir[0] == parse_expr_to_ir("a + b")


# ---------------------------------------------------------------------------
# Helper-bearing IR fields (aliases_ir_reduce / equations_ir_reduce)
# ---------------------------------------------------------------------------


def test_equations_ir_reduce_is_positionally_aligned() -> None:
    """``equations_ir_reduce`` aligns with ``equations`` when populated."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1", "p2"]}],
        "state": ["S[pop]", "I[pop]"],
        "equations": {
            "S[pop]": "-beta * S[pop] * apply_along(I[pop:j], pop=j)",
            "I[pop]": ("beta * S[pop] * apply_along(I[pop:j], pop=j) - gamma * I[pop]"),
        },
    }
    out = normalize_expr_rhs(spec)
    assert len(out.equations_ir_reduce) == len(out.equations)


def test_equations_ir_reduce_contains_reduce_nodes_for_apply_along() -> None:
    """Pre-expansion IR exposes ``Reduce`` nodes for ``apply_along`` calls."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1", "p2"]}],
        "state": ["S[pop]", "I[pop]"],
        "equations": {
            "S[pop]": "-beta * S[pop] * apply_along(I[pop:j], pop=j)",
            "I[pop]": ("beta * S[pop] * apply_along(I[pop:j], pop=j) - gamma * I[pop]"),
        },
    }
    out = normalize_expr_rhs(spec)
    expr = out.equations_ir_reduce[0]
    assert expr is not None
    reduces = [n for n in walk(expr) if isinstance(n, Reduce)]
    assert len(reduces) == 1
    r = reduces[0]
    assert r.kind == "apply_along"
    assert r.bindings == (("pop", "j"),)


def test_aliases_ir_reduce_contains_reduce_for_aliased_apply_along() -> None:
    """Aliases whose body is an ``apply_along`` produce ``Reduce`` IR."""
    spec = {
        "kind": "expr",
        "axes": [{"name": "pop", "coords": ["p1", "p2"]}],
        "state": ["S[pop]", "I[pop]"],
        "aliases": {"I_total": "apply_along(I[pop:j], pop=j)"},
        "equations": {
            "S[pop]": "-beta * S[pop] * I_total",
            "I[pop]": "beta * S[pop] * I_total - gamma * I[pop]",
        },
    }
    out = normalize_expr_rhs(spec)
    assert "I_total" in out.aliases_ir_reduce
    body = out.aliases_ir_reduce["I_total"]
    reduces = [n for n in walk(body) if isinstance(n, Reduce)]
    assert len(reduces) == 1
    assert reduces[0].kind == "apply_along"


def test_equations_ir_reduce_empty_for_transitions_kind() -> None:
    """``transitions`` kind leaves the reduce-bearing IR fields empty."""
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
    assert out.aliases_ir_reduce == {}
    assert out.equations_ir_reduce == ()


def test_equations_ir_reduce_matches_equations_ir_when_no_helpers() -> None:
    """Without helpers, reduce-IR parses to the same tree as standard IR."""
    spec = {
        "kind": "expr",
        "state": ["S", "I"],
        "equations": {"S": "-beta * S * I", "I": "beta * S * I - gamma * I"},
    }
    out = normalize_expr_rhs(spec)
    assert len(out.equations_ir_reduce) == len(out.equations_ir)
    for ir_reduce, ir_raw in zip(
        out.equations_ir_reduce, out.equations_ir_raw, strict=True
    ):
        assert ir_reduce == ir_raw
