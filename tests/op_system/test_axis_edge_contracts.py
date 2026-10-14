"""Public axis boundary contracts uncovered by the readiness inventory."""

import numpy as np
import pytest

from op_system import compile_spec
from op_system.specs import normalize_rhs


@pytest.mark.parametrize("axis_type", ["categorical", "ordinal", "continuous"])
def test_empty_coordinate_list_is_rejected(axis_type: str) -> None:
    """An explicitly empty coordinate domain cannot silently erase a state."""
    with pytest.raises(ValueError, match="coords must be a non-empty list"):
        normalize_rhs({
            "kind": "expr",
            "axes": [{"name": "age", "type": axis_type, "coords": []}],
            "state": ["S[age]"],
            "equations": {"S[age]": "-S[age]"},
        })


def test_no_axes_retains_scalar_model() -> None:
    """An empty axis declaration is valid for scalar compartments."""
    compiled = compile_spec({
        "kind": "expr",
        "axes": [],
        "state": ["S"],
        "equations": {"S": "-2 * S"},
    })
    np.testing.assert_array_equal(compiled.eval_fn(0.0, np.asarray([3.0])), [-6.0])
