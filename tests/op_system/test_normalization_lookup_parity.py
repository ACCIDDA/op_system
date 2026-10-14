"""Behavioral regressions for cached normalization lookups (#267)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from op_system import compile_rhs, normalize_rhs
from op_system._errors import InvalidRhsSpecError
from op_system._ir import parse_expr_to_ir


def _axes() -> list[dict[str, object]]:
    return [
        {"name": "group", "coords": ["west", "east", "north"]},
        {"name": "stage", "coords": ["older", "younger"]},
    ]


@pytest.mark.parametrize("kind", ["expr", "transitions"])
def test_cached_coordinates_preserve_shaped_rates_and_aliases(kind: str) -> None:
    """Cartesian coordinate order and scalar parameters match independent flows."""
    spec: dict[str, Any] = {
        "kind": kind,
        "axes": _axes(),
        "state": ["A[group,stage]", "B[group,stage]"],
        "aliases": {"forward[group]": "shared * rate[group]"},
    }
    if kind == "expr":
        spec["equations"] = {
            "A[group,stage]": (
                "-forward[group] * A[group,stage] + back[group,stage] * B[group,stage]"
            ),
            "B[group,stage]": (
                "forward[group] * A[group,stage] - back[group,stage] * B[group,stage]"
            ),
        }
    else:
        spec["transitions"] = [
            {
                "from": "A[group,stage]",
                "to": "B[group,stage]",
                "rate": "forward[group]",
            },
            {
                "from": "B[group,stage]",
                "to": "A[group,stage]",
                "rate": "back[group,stage]",
            },
        ]
    rhs = normalize_rhs(spec)
    assert rhs.param_names == ("shared",)
    assert dict(rhs.meta["shaped_params"]) == {
        "back": ("group", "stage"),
        "rate": ("group",),
    }
    compiled = compile_rhs(rhs)
    state = np.arange(1.0, 13.0).reshape(2, 3, 2)
    rates = np.array([0.1, 0.2, 0.4])
    back = np.arange(1.0, 7.0).reshape(3, 2) / 100
    forward_flow = 0.3 * rates[:, None] * state[0]
    reverse_flow = back * state[1]
    balance = forward_flow - reverse_flow
    expected = np.stack([-balance, balance])
    params = {"shared": 0.3, "rate": rates, "back": back}
    np.testing.assert_allclose(
        compiled.eval_fn(0.0, state.ravel(), **params), expected.ravel()
    )
    assert compiled.pytree_eval_fn is not None
    tree = compiled.pytree_eval_fn(0.0, {"A": state[0], "B": state[1]}, **params)
    np.testing.assert_allclose(tree["A"], expected[0])
    np.testing.assert_allclose(tree["B"], expected[1])


def test_explicit_cell_equation_precedes_template_equation() -> None:
    """The cell lookup retains the existing explicit-equation precedence."""
    rhs = normalize_rhs({
        "kind": "expr",
        "state": ["X[group]"],
        "axes": [{"name": "group", "coords": ["west", "east"]}],
        "equations": {
            "X[group]": "-k * X[group]",
            "X__group_west": "-override * X__group_west",
        },
    })
    assert rhs.equations_ir[0] == parse_expr_to_ir("-override * X__group_west")
    assert rhs.equations_ir[1] == parse_expr_to_ir("-k * X__group_east")
    assert rhs.param_names == ("k", "override")


@pytest.mark.parametrize("equation", [None, "", "  "])
def test_invalid_template_equation_still_names_the_template(equation: object) -> None:
    """Cached cell lookup preserves normalization's error context."""
    with pytest.raises(InvalidRhsSpecError, match=r"equations\['X\[group\]'\]"):
        normalize_rhs({
            "kind": "expr",
            "state": ["X[group]"],
            "axes": [{"name": "group", "coords": ["west", "east"]}],
            "equations": {"X[group]": equation},
        })
