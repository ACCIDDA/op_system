# Log state coordinates

Available in OP System 0.9.0. This feature applies to deterministic expression
and transition RHS evaluation, including supported named fields and nonlocal
expression reductions.

Declare `state_metadata: {x: {scale: log}}` to request `z = log(x)` for every
expanded component of state base `x`. Other states use identity coordinates.
Model expressions, initial physical values and observable expressions continue
to describe physical quantities. Build an explicit `model.coordinate_view()`
once, outside the solver callback. The original compiled model remains in its
natural coordinates.

`StateDeclaration.scale` defaults to `identity`. The record now has five fields;
existing four-argument construction remains valid, but code unpacking four tuple
items needs to use named attributes or include the new field.

```python
import numpy as np
from op_system import compile_spec

model = compile_spec({
    "kind": "expr",
    "state": ["x", "collected"],
    "state_metadata": {"x": {"scale": "log"}},
    "equations": {"x": "-rate*x", "collected": "rate*x"},
    "observables": {"total": {"expr": "x+collected"}},
})
view = model.coordinate_view()
z0 = view.encode(np.array([1.0, 0.0]))
rhs = view.bind({"rate": 2.0})
np.testing.assert_allclose(rhs(0, z0), [-2.0, 2.0])
np.testing.assert_allclose(view.observables["total"].eval_fn(0, z0), 1.0)
```

Pass `rhs` and `z0` to a deterministic solver's ordinary RHS interface. For OP
Engine use a deterministic core solver directly; the flepimop2 providers do not
automatically select the coordinate view from metadata. Decode its result
before reporting physical trajectories or applying output-state diagnostics.
For an `(output_time, state, ...)` array use
`view.decode_native(trajectory, state_axis=1)` and inspect its `valid` flag.
Initial-state expressions and provider initialization remain in natural units;
encode the evaluated initial state explicitly. The view exposes shaped
`pytree_eval_fn` and independent `block_pytree_eval_fn` when the source model
supports them. Its PyTree encoders accept complete state dictionaries, including
buffers with an independent block axis already stripped.

## Equation and domain

For a selected component, the view compiles `F_i(x)/x_i` and evaluates it at
`x = exp(z)`. It cancels syntactically exact multiplicative factors before
division where supported. For `dx/dt = -rate*x`, this gives `dz/dt = -rate`
even if `exp(z)` underflows. This simplification is not a general symbolic
limit calculation: an additive source `b-rate*x` requires `b/x-rate` and can
overflow when `x` is extremely small. Aliases, total-state helpers, time tables
and observables receive decoded physical state.

`view.encode` rejects nonfinite state values and zero or negative selected
components before evaluating logarithms. There is no implicit floor or clamp.
Select log coordinates only for states whose intended deterministic trajectory
stays strictly positive; a cumulative counter starting at zero usually needs
identity coordinates. The metadata is a coordinate request, not a proof that
the equations preserve positivity.

NumPy, JAX and CPU Torch transforms retain the source namespace and floating
dtype. Use `encode_native`, `decode_native`, `encode_pytree_native` and
`decode_pytree_native` inside JIT, vmap or automatic differentiation. They return
`CoordinateResult(state, valid)` without extracting host scalars. Invalid native
inputs can produce nonfinite outputs. Decoding can underflow to zero or overflow;
the scalar native flag reports either failure, and `decode_checked` checks it
eagerly. `decode` is the unchecked native value operation used in callbacks.
Flags describe domain and representability, not solver accuracy. Signed identity
states are masked before logarithms/exponentials so inactive branches do not
contaminate derivatives.

## Numerical behavior and exported structure

The checked [example](https://github.com/ACCIDDA/op_system/blob/main/examples/log_state_coordinates.py)
uses a deliberately large Euler step for `dx/dt = -2*x`,
`d(collected)/dt = 2*x`, starting at `(1, 0)`. Its independently known solution
at time one gives the following values.

| Quantity at time one | Direct Euler | Log Euler, decoded | Exact |
| --- | ---: | ---: | ---: |
| Remaining `x` | -1.000000 | 0.135335 | 0.135335 |
| Collected | 2.000000 | 2.000000 | 0.864665 |
| Total | 1.000000 | 2.135335 | 1.000000 |

Log Euler is exact for this particular remaining-state equation. The collected
state is inaccurate, and the decoded discrete total is not conserved. Smaller
steps, error controls and independent solution checks still matter. A change of
coordinates also changes the meaning of a solver's error norm and tolerances.

Natural-scale [reaction balance laws](reaction-balances.md) remain applicable to
the continuous physical equations after decoding. Their original linear
coefficients do not become linear laws of log coordinates. The view therefore
does not export reaction artifacts, balances or the source model's derivative
and affinity metadata as coordinate-space claims. Obtain coordinate Jacobians
by differentiating the view evaluator; nonlinear changes can alter sparsity and
affine dependence. The retained `view.natural` model still exports its original
artifacts. Stochastic SSA consumes those original physical states and integer
jumps; the log view transforms deterministic drift only.

Separate operator descriptors and history callbacks require a transformed
composition contract and currently reject log-view creation. This restriction
does not exclude named-field or nonlocal dynamics already represented in the
instantaneous expression RHS.

For [gauge-fixed models](scaling-gauges.md), first encode original parameters and
initial physical state with `model.gauge_fix.encode`. Then log-encode that
representative with the coordinate view. To recover original units, decode log
coordinates first and then apply the retained gauge inverse. Gauge anchors stay
bound across coordinate RHS and observable callbacks.
