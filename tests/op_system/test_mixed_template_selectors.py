"""Keep compatibility string selectors intact alongside the typed-IR grammar."""

import pytest

from op_system import compile_spec
from op_system._errors import InvalidRhsSpecError
from op_system._templates import _apply_template_substitutions


def test_mixed_bound_entries_survive_compatibility_substitution() -> None:
    """Binding one wildcard must retain every other explicitly bound axis."""
    expression = "X[loc, age=ap, vax = v, imm=k]"
    result = _apply_template_substitutions(
        expression, template_map={}, assignment={"loc": "AK"}
    )
    assert result == "X[loc=AK, age=ap, vax = v, imm=k]"
    assert (
        _apply_template_substitutions(expression, template_map={}, assignment={})
        == expression
    )


def test_mixed_shaped_parameter_compatibility_is_preserved() -> None:
    """The established shaped-parameter branch is separate from state expansion."""
    expression = "theta[loc, age=ap]"
    result = _apply_template_substitutions(
        expression,
        template_map={},
        assignment={"loc": "AK"},
        shaped_params={"theta": ("loc", "age")},
        axis_lookup={"loc": ["AK"], "age": ["young", "old"]},
    )
    assert result == expression


def test_legacy_equals_state_selectors_fail_with_alias_context() -> None:
    """Legacy equals selectors fail rather than silently rewriting an alias."""
    with pytest.raises(
        InvalidRhsSpecError, match=r"aliases\['total\[loc\]'\].*invalid expression"
    ):
        compile_spec({
            "kind": "expr",
            "axes": [{"name": "loc", "coords": ["AK"]}],
            "state": ["X[loc]"],
            "aliases": {"total[loc]": "X[loc=AK]"},
            "equations": {"X[loc]": "total[loc]"},
        })
