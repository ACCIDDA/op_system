"""Native state-buffer helpers shared by RHS and reaction evaluation."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _bind_quadrature_weights(
    env: dict[str, object],
    state: object,
    xp: Any,  # ruff: ignore[any-type]
) -> None:
    """Construct constant weights in the floating state's dtype and device."""
    dtype = getattr(state, "dtype", None)
    if dtype is not None and not xp.isdtype(dtype, "real floating"):
        dtype = None
    env["_op_system_weights"] = partial(
        xp.asarray, dtype=dtype, device=getattr(state, "device", None)
    )


def _bind_sum_state(
    env: dict[str, object],
    state_bases: Sequence[str],
    xp: Any,  # ruff: ignore[any-type]
) -> None:
    """Bind a lazy per-call total of declared state buffers, excluding aliases."""
    values = {base: env[f"{base}_buf"] for base in state_bases if f"{base}_buf" in env}
    total: object | None = None

    def sum_state() -> object:
        nonlocal total
        if total is None:
            missing = next((base for base in state_bases if base not in values), None)
            if missing is not None:
                msg = f"sum_state requires the declared state buffer {missing!r}"
                raise ValueError(msg)
            total = sum(xp.sum(values[base]) for base in state_bases)
        return total

    env["sum_state"] = sum_state
