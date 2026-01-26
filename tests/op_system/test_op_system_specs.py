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


def test_normalize_rhs_unsupported_kind() -> None:
    """Test that unsupported RHS kinds raise UnsupportedFeatureError."""
    spec = {"kind": "pde", "state": ["u"], "equations": {"u": "0.0"}}
    with pytest.raises(NotImplementedError, match=r"Only 'expr' and 'transitions'"):
        normalize_rhs(spec)
