"""Check log-coordinate decay against an analytic solution and a balance error.

Run ``uv run python examples/log_state_coordinates.py``. These synthetic
examples use explicit Euler to expose the coordinate choice; they do not
establish accuracy or performance for general equations or solver methods.
"""

import numpy as np

from op_system import compile_spec

SPEC = {
    "kind": "transitions",
    "state": ["x", "collected"],
    "state_metadata": {"x": {"scale": "log"}},
    "transitions": [
        {"name": "transfer", "from": "x", "to": "collected", "rate": "rate"}
    ],
    "observables": {"total": {"expr": "x+collected"}},
}
model = compile_spec(SPEC)
view = model.coordinate_view()
physical = np.array([1.0, 0.0])
initial = view.encode(physical)
rate = 2.0

# One Euler step of length one: d(log x)/dt=-rate exactly for this model.
direct = physical + model.eval_fn(0, physical, rate=rate)
encoded = initial + view.eval_fn(0, initial, rate=rate)
decoded = view.decode_checked(encoded)
np.testing.assert_allclose(direct, [-1.0, 2.0])
np.testing.assert_allclose(decoded, [np.exp(-2), 2.0])
np.testing.assert_allclose(
    view.observables["total"].eval_fn(1, encoded), np.exp(-2) + 2
)
# Natural-coordinate conservation does not imply a conserved log Euler update.
assert not np.isclose(decoded.sum(), 1)  # ruff: ignore[assert]
assert len(model.reactions) == 1  # ruff: ignore[assert]

print("quantity          direct Euler      log Euler        exact")  # ruff: ignore[print]
for name, index in [("x", 0), ("collected", 1)]:
    exact = np.exp(-2) if index == 0 else 1 - np.exp(-2)
    print(  # ruff: ignore[print] - executable table output
        f"{name}(1) {direct[index]:.6f}   {decoded[index]:.6f}   {exact:.6f}"
    )
print(  # ruff: ignore[print] - executable table output
    f"total(1) {direct.sum():.6f}   {decoded.sum():.6f}   1.000000"
)
