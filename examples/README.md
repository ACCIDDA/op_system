# Examples

These examples exercise `op_system` as a model-specification compiler. The
compiled artifacts evaluate declared derivatives, hazards, and observables,
and the compiler can produce export artifacts. It does not integrate a
trajectory or scientifically validate the model; numerical integration and
scientific qualification belong to downstream consumers and independent
checks.

Run commands from the repository root after installing the development
environment with `uv sync --dev`.

## Start with a rectangular consumer--resource kernel

[`named_kernel_products.py`](named_kernel_products.py) is the general example
for interactions between two differently sized named axes. Its `P[resource]`
and `Z[consumer]` arrays can be read as resource and consumer classes--for
example, as a plankton-style resource--consumer interaction--while the same
rectangular contraction applies to other domains. `K[consumer,resource]`
maps resources to consumer exposure, and its transpose maps consumer-weighted
losses back to resources. Per-consumer conditions select linear or quadratic
loss terms.

The example uses synthetic inputs, evaluates the declared derivative at one
fixed state, and compares it with independently written NumPy matrix products.
It does not advance the state through time, select an integration method, or
establish that the equations are an adequate ecological or other scientific
model. The [named-axis kernel guide](../docs/guides/named-kernel-products.md)
explains the axis orientation, conditional selection, lowering boundary, and
backend checks. [`illustrate_named_kernels.py`](illustrate_named_kernels.py)
regenerates or checks the guide's fixed-state figure and retained data.

Run the core example with:

```bash
uv run python examples/named_kernel_products.py
```

## Other specification examples

- [`age-seir.yaml`](age-seir.yaml) is the checked age-structured transition
  specification used by the [model-writing guide](../docs/guides/getting-started.md).
  It declares compartments, categorical mixing, and initial-state metadata and
  references runtime parameters that the compiler infers; a caller still
  supplies state and parameter values and an engine for time integration.
- [`independent_reactions.py`](independent_reactions.py) validates and compiles
  one transfer reaction across two independent replicates, with held forcing
  and a named population observable. It evaluates the same fixed state before
  and at a forcing change; it does not simulate events or integrate an ODE. See
  [Capabilities and executable examples](../docs/guides/capabilities.md).
- [`log_state_coordinates.py`](log_state_coordinates.py) compares one explicit
  Euler update in physical and log coordinates with an analytic decay value.
  It exposes the coordinate choice on a synthetic fixture and does not
  establish accuracy for general equations or solver methods. See
  [Log state coordinates](../docs/guides/log-state-coordinates.md).
- [`scaling_gauges.py`](scaling_gauges.py) detects one declared shared scaling,
  maps a state and its parameters to a fixed representative, evaluates the
  compiled derivative and observable, and decodes the result. Gauge detection
  is not a structural-identifiability proof. See
  [Scaling gauges](../docs/guides/scaling-gauges.md).
- [`portable_julia.py`](portable_julia.py) emits Julia RHS and observation code
  for a bounded synthetic specification and checks their values at fixed points.
  It also compares Python symbolic derivatives with Julia finite differences.
  It requires an existing Julia executable and makes no integration,
  external-rank, or inference-accuracy claim. See
  [Portable IR](../docs/guides/portable-ir.md).
- [`benchmark_symbolic.py`](benchmark_symbolic.py) measures bounded synthetic
  derivative construction/evaluation and derivative actions with independent
  analytic checks. Its fresh-process timings are not whole-application
  performance claims.

[`illustrate_reactions.py`](illustrate_reactions.py) regenerates or checks the
fixed-state transfer figure, while
[`illustration_artifacts.py`](illustration_artifacts.py) is the shared artifact
helper rather than a standalone model example.
