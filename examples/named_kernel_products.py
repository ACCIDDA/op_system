"""Two distinct named axes, rectangular interactions and conditional losses.

Run ``uv run python examples/named_kernel_products.py``. All inputs are
synthetic. The independent reference uses ordinary matrix products.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from op_system import compile_spec


def model_spec(resources: int = 3, consumers: int = 2) -> dict[str, Any]:
    """Describe interaction between two differently sized groups of variables.

    Returns:
        A specification with resource and consumer axes and a rectangular table.
    """
    return {
        "kind": "expr",
        "axes": [
            {"name": "resource", "coords": [f"r{i}" for i in range(resources)]},
            {"name": "consumer", "coords": [f"c{i}" for i in range(consumers)]},
        ],
        "state": ["P[resource]", "Z[consumer]"],
        "aliases": {
            "exposure[consumer]": (
                "apply_along(K[consumer,resource:r]*P[resource:r], "
                "resource=r, kernel=sum)"
            ),
            "removal[resource]": (
                "apply_along(g[consumer:c]*K[consumer:c,resource]*Z[consumer:c], "
                "consumer=c, kernel=sum)"
            ),
        },
        "equations": {
            "P[resource]": "-P[resource]*removal[resource]",
            "Z[consumer]": (
                "efficiency[consumer]*Z[consumer]*exposure[consumer] - "
                "np.where(linear[consumer] > 0, loss[consumer]*Z[consumer], "
                "loss[consumer]*Z[consumer]*Z[consumer])"
            ),
        },
    }


def inputs(resources: int = 3, consumers: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    """Return seeded states and a table ordered as consumer by resource."""
    rng = np.random.default_rng(312)
    state = rng.uniform(0.3, 2.0, size=resources + consumers)
    params = {
        "K": rng.uniform(0.0, 1.0, size=(consumers, resources)) / resources,
        "g": rng.uniform(0.2, 1.0, size=consumers),
        "efficiency": rng.uniform(0.2, 0.8, size=consumers),
        "loss": rng.uniform(0.05, 0.3, size=consumers),
        "linear": np.arange(consumers) % 2,
    }
    return state, params


def reference(state: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Compute the same equations directly, without the specification compiler.

    Returns:
        Flat resource and consumer derivatives in declared state order.
    """
    consumers, resources = params["K"].shape
    p, z = state[:resources], state[resources : resources + consumers]
    resource_loss = -p * (params["K"].T @ (params["g"] * z))
    consumer_change = params["efficiency"] * z * (params["K"] @ p) - np.where(
        params["linear"] > 0, params["loss"] * z, params["loss"] * z**2
    )
    return np.concatenate([resource_loss, consumer_change])


def main() -> None:
    """Check the compiled equations against the independent rectangular product."""
    state, params = inputs()
    compiled = compile_spec(model_spec())
    result = compiled.eval_fn(0.0, state, **params)
    np.testing.assert_allclose(result, reference(state, params), rtol=1e-13, atol=1e-14)
    print(result)  # ruff: ignore[print] - command-line example output


if __name__ == "__main__":
    main()
