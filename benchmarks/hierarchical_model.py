"""Synthetic closed relaxation network for compiler qualification.

The handwritten NumPy oracle deliberately uses cell loops, whereas the
specification uses aliases and axis reductions. No compiler expressions or
generated source are used to construct the reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

TEMPLATES = ("A", "B", "C")


def model_spec(groups: int, stages: int, *, coupled: bool = False) -> dict[str, Any]:
    """Return a nonlinear model with shared and group-local parameters.

    Returns:
        An expression spec with aliases, reducers, and stage transport.

    Raises:
        ValueError: If either dimension is less than one.
    """
    if groups < 1 or stages < 1:
        msg = "groups and stages must be positive"
        raise ValueError(msg)
    forcing = (
        "sum_over(B[group:g,stage], group=g) / count" if coupled else "B[group,stage]"
    )
    spec: dict[str, Any] = {
        "kind": "expr",
        "axes": [
            {"name": "group", "coords": [f"g{i}" for i in range(groups)]},
            {"name": "stage", "coords": [f"a{i}" for i in range(stages)]},
        ],
        "state": [f"{name}[group,stage]" for name in TEMPLATES],
        "aliases": {
            "total[group]": (
                "sum_over(A[group,stage:s] + B[group,stage:s]"
                " + C[group,stage:s], stage=s)"
            ),
            "flux[group,stage]": (
                "gain * local[group] * A[group,stage]"
                f" * (1 + ({forcing}) / total[group])"
            ),
            "transport[group,stage]": (
                "apply_along(K[stage:s,stage] * A[group,stage:s], stage=s, kernel=sum)"
                " - A[group,stage] * sum_over(K[stage,stage:s], stage=s)"
            ),
        },
        "equations": {
            "A[group,stage]": (
                "-flux[group,stage] + recovery[stage] * C[group,stage]"
                " + transport[group,stage]"
            ),
            "B[group,stage]": ("flux[group,stage] - recovery[stage] * B[group,stage]"),
            "C[group,stage]": ("recovery[stage] * (B[group,stage] - C[group,stage])"),
        },
    }
    if not coupled:
        spec["factorize_axes"] = ["group"]
    return spec


def fixture(
    groups: int = 3, stages: int = 2, *, dtype: str = "float64", coupled: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return deterministic positive state and parameter fixtures.

    Returns:
        A structured state and parameters, all with the requested dtype.
    """
    values = np.arange(1, 3 * groups * stages + 1, dtype=dtype).reshape(
        3, groups, stages
    )
    values /= 7
    state = dict(zip(TEMPLATES, values, strict=True))
    indices = np.arange(stages, dtype=dtype)
    kernel = (indices[:, None] + indices[None, :] + 1) / (100 * stages)
    np.fill_diagonal(kernel, 0)
    params = {
        "gain": np.asarray(0.4, dtype=dtype),
        "local": np.linspace(0.2, 0.8, groups, dtype=dtype),
        "recovery": np.linspace(0.1, 0.3, stages, dtype=dtype),
        "K": kernel,
    }
    if coupled:
        params["count"] = np.asarray(groups, dtype=dtype)
    return state, params


def reference_rhs(
    state: Mapping[str, object], params: Mapping[str, object], *, coupled: bool = False
) -> dict[str, np.ndarray]:
    """Evaluate independent balance equations with explicit cell loops.

    Returns:
        NumPy derivatives in the state's dtype and shape.
    """
    values = np.stack([np.asarray(state[name]) for name in TEMPLATES])
    groups, stages = values.shape[1:]
    derivative = np.zeros_like(values)
    local = np.asarray(params["local"])
    recovery = np.asarray(params["recovery"])
    kernel = np.asarray(params["K"])
    for group in range(groups):
        total = sum(float(value) for value in values[:, group, :].ravel())
        for stage in range(stages):
            a, b, c = (float(value) for value in values[:, group, stage])
            forcing = (
                sum(float(value) for value in values[1, :, stage]) / groups
                if coupled
                else b
            )
            flow = (
                float(np.asarray(params["gain"]))
                * float(local[group])
                * a
                * (1 + forcing / total)
            )
            transport = sum(
                float(values[0, group, source]) * float(kernel[source, stage])
                - a * float(kernel[stage, source])
                for source in range(stages)
            )
            rate = float(recovery[stage])
            derivative[:, group, stage] = (
                -flow + rate * c + transport,
                flow - rate * b,
                rate * (b - c),
            )
    return dict(zip(TEMPLATES, derivative, strict=True))


def handwritten_rhs(
    state: Mapping[str, Any],
    params: Mapping[str, Any],
    xp: Any,  # ruff: ignore[any-type]
    *,
    coupled: bool = False,
) -> dict[str, Any]:
    """Evaluate native vectorized equations independently of compiler code.

    Returns:
        Derivatives in the supplied numerical namespace.
    """
    a, b, c = (state[name] for name in TEMPLATES)
    totals = xp.sum(a + b + c, axis=-1, keepdims=True)
    source = xp.mean(b, axis=0, keepdims=True) if coupled else b
    rate = xp.reshape(params["local"], (-1, 1)) * params["gain"]
    flow = rate * a * (1 + source / totals)
    transport = xp.matmul(a, params["K"]) - a * xp.sum(params["K"], axis=1)
    return {
        "A": -flow + params["recovery"] * c + transport,
        "B": flow - params["recovery"] * b,
        "C": params["recovery"] * (b - c),
    }
