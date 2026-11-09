# Conditional within-host kernel artifacts

`op_system.kernels` publishes an eager, immutable NumPy snapshot, not an automatic
binding between solvers. The caller solves the source model and evaluates its
named observable before constructing `ConditionalKernel`. Record the source model
identity, parameter/solver/precision settings identity, ordered infection-age grid
and units, conditional axes and coordinates, observable units and conditioning.
No host, antigen, strain, contact or shedding axis is reduced automatically.

```python
import numpy as np
from op_system.kernels import ConditionalKernel, KernelAxis, KernelReduction

kernel = ConditionalKernel(
    infection_age=(0.0, 1.0),
    axes=(KernelAxis("host", ("low", "high")),),
    values=np.array([[1.0, 2.0], [0.5, 1.5]]),
    source_identity="source-model-content-identity",
    source_settings_identity="parameters-and-solver-content-identity",
    observable="shedding-times-participation",
    units="transmission/day",
    age_units="day",
)
population = kernel.reduce(
    KernelReduction(("host",), (0.75, 0.25), "host-joint-distribution-identity")
)
assert np.allclose(population.values, [1.25, 0.75])
```

`KernelReduction` uses explicitly normalized joint probability weights in its
declared axis order. Partial reductions retain all other axes. Infection age cannot
be reduced by this API. General age integration, unnormalized quadrature and
weights conditioned on retained axes or infection age are not implemented.
When a distribution or its parameters change, regenerate the artifact: static
weights must not masquerade as inferred frailty. A full joint distribution can
preserve correlations that products of marginal weights destroy.

Apply nonlinear scientific mappings **before** population averaging when the
estimand is `E[f(X)]`. Thresholding the mean activity does not estimate
seropositivity. Likewise, averaging shedding and contact separately loses their
covariance. Publish the mapped observable with its own identity/settings and
units; do not relabel an unrelated mean as a transmission kernel.

Arrays are copied into immutable byte-backed float64 snapshots; input mutation
cannot change a table. `content_digest()` covers the ordered grid, axes, values,
units, source/settings, conditioning and the complete reduction weights/history.
It does not authenticate source identities or qualify a scientific model.
Callers own serialization and persistence of referenced settings and arrays.

`tests/op_system/test_conditional_kernels.py` qualifies a two-trait source ODE
against an independent exponential solution and explicitly couples its averaged
kernel to a discrete renewal recursion. It also demonstrates a nonlinear
counterexample and joint weights in reversed named-axis order.

This bounded increment toward #360 does not implement automatic source evaluation,
compiled target-history binding, changing conditional distributions, interpolation,
backend-native gradients, or two-way population/within-host feedback.
