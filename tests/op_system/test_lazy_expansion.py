"""Lazy per-cell expansion keeps large reductions cheap (#88)."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import Any

import numpy as np

from op_system import compile_rhs, normalize_rhs
from op_system._normalize_ir import _LazyCells


def _routing_spec(n_imm: int) -> dict[str, Any]:
    return {
        "kind": "expr",
        "axes": [
            {"name": "vax", "coords": ["u", "v"]},
            {
                "name": "imm",
                "type": "ordinal",
                "coords": [f"x{i}" for i in range(n_imm)],
            },
        ],
        "state": ["X[vax, imm]"],
        "equations": {
            "X[vax, imm]": (
                "apply_along(K[imm:i, imm] * X[vax, imm:i], imm=i, kernel=sum)"
                " - X[vax, imm] * apply_along(K[imm, imm:j], imm=j, kernel=sum)"
            )
        },
    }


def _square(index: int, value: int) -> int:
    return value * value + index


def test_lazy_cells_behave_like_tuples() -> None:
    """Lazy cells index, slice, compare, hash, and pickle like tuples."""
    cells = _LazyCells((1, 2, 3, 4), _square)
    assert len(cells) == 4
    assert cells[1] == 5
    assert cells[-1] == 19
    assert isinstance(cells[1:3], _LazyCells)
    assert tuple(cells[1:3]) == (5, 11)
    assert cells == (1, 5, 11, 19)
    assert hash(cells) == hash((1, 5, 11, 19))
    restored = pickle.loads(pickle.dumps(cells))  # ruff: ignore[suspicious-pickle-usage]
    assert restored == (1, 5, 11, 19)
    assert isinstance(restored, tuple)


def test_routing_reduction_expands_only_cells_that_are_read() -> None:
    """Normalization and vectorized compile leave most cells unexpanded."""
    rhs = normalize_rhs(_routing_spec(30))
    compiled = compile_rhs(rhs)
    assert isinstance(rhs.equations_ir, _LazyCells)
    built = len(rhs.equations_ir._cache)  # ruff: ignore[private-member-access]
    assert built < len(rhs.state_names) // 4

    rng = np.random.default_rng(0)
    kernel = rng.uniform(0.0, 1.0, size=(30, 30))
    state = rng.uniform(1.0, 2.0, size=(2, 30))
    assert compiled.pytree_eval_fn is not None
    derivative = compiled.pytree_eval_fn(0.0, {"X": state}, K=kernel)["X"]
    expected = state @ kernel - state * kernel.sum(axis=1)
    np.testing.assert_allclose(derivative, expected, rtol=1e-12, atol=1e-12)


def test_history_flag_follows_the_raw_spec() -> None:
    """Specs without history helpers are flagged so compile skips the scan."""
    plain = normalize_rhs(_routing_spec(3))
    assert plain.meta["op_system_may_have_history"] is False
    delayed = {
        "kind": "expr",
        "axes": [{"name": "loc", "coords": ["a", "b"]}],
        "state": ["x[loc]", "I[loc]"],
        "equations": {
            "x[loc]": (
                "convolve_history(I[loc] * beta, kernel=gamma, window=14, "
                "kernel_shape=3, kernel_scale=2)"
            ),
            "I[loc]": "0.0",
        },
    }
    history_rhs = normalize_rhs(delayed)
    assert history_rhs.meta["op_system_may_have_history"] is True
    assert compile_rhs(history_rhs).history_requirements
