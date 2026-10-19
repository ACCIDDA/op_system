"""Verify a shared scaling and map to a fixed representative."""

import numpy as np

from op_system import compile_spec

SPEC = {
    "kind": "expr",
    "state": ["x"],
    "equations": {"x": "rate*x*(1-x/K)"},
    "observables": {"fraction": {"expr": "x/K"}},
    "gauges": [{"name": "units", "states": {"x": 1}, "parameters": {"K": 1}}],
}
model = compile_spec(SPEC, detect_gauges=True, gauge_fix={"K": 1.0})
assert model.gauges is not None  # ruff: ignore[assert]
assert model.gauges.rank == 1  # ruff: ignore[assert]
assert model.gauge_fix is not None  # ruff: ignore[assert]
natural = np.array([6.0])
coordinates = model.gauge_fix.encode(natural, {"rate": 0.5, "K": 12.0})
assert coordinates.valid  # ruff: ignore[assert]
np.testing.assert_allclose(coordinates.state, [0.5])
assert coordinates.parameters == {"rate": 0.5}  # ruff: ignore[assert]
np.testing.assert_allclose(
    model.eval_fn(0, coordinates.state, **coordinates.parameters), [0.125]
)
np.testing.assert_allclose(
    model.observables["fraction"].eval_fn(0, coordinates.state), 0.5
)
state, parameters = model.gauge_fix.decode(coordinates)
np.testing.assert_allclose(state, natural)
assert np.isclose(parameters["K"], 12)  # ruff: ignore[assert]
