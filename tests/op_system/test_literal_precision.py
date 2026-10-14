"""Literal construction follows state precision without coercing typed inputs."""

import numpy as np

from op_system import compile_spec


def test_direct_literals_follow_each_mixed_state_dtype() -> None:
    """Each direct literal uses its own target buffer when state precisions differ."""
    compiled = compile_spec({
        "kind": "expr",
        "state": ["X", "Y"],
        "equations": {"X": "0", "Y": "1.5"},
    })
    assert compiled.pytree_eval_fn is not None
    state: dict[str, np.ndarray] = {
        "X": np.asarray(1.0, dtype=np.float32),
        "Y": np.asarray(2.0, dtype=np.float64),
    }
    result = compiled.pytree_eval_fn(0.0, state)
    assert result["X"].dtype == np.float32
    assert result["Y"].dtype == np.float64


def test_explicitly_typed_parameter_is_not_coerced_to_state_precision() -> None:
    """The literal fix does not hide deliberate mixed-precision array inputs."""
    compiled = compile_spec({
        "kind": "expr",
        "state": ["X"],
        "equations": {"X": "forcing"},
    })
    assert compiled.pytree_eval_fn is not None
    result = compiled.pytree_eval_fn(
        0.0,
        {"X": np.asarray(1.0, dtype=np.float32)},
        forcing=np.asarray(0.5, dtype=np.float64),
    )
    assert result["X"].dtype == np.float64
