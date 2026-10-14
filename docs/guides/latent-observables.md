# Named latent observables

Optional `observables` define instantaneous quantities separately from the RHS.
They work with expression and transition models. `CompiledRhs.observables` is a
read-only mapping of names to `CompiledObservable` records. Each contains a
definition, concrete output shape and coordinate labels, direct state and
parameter dependencies, explicit time use, applied physical measures, and native
flat/PyTree evaluators. Records and evaluator protocols are also exported from
`op_system.observables` and the package root.

```yaml
kind: expr
axes:
  - name: x
    type: continuous
    coords: [0.05, 0.25, 0.7]
    measure: {kind: cell, weights: [0.1, 0.3, 0.6], units: m}
state: ["rho[x]", "mass[x]", counter]
equations:
  rho[x]: "0"
  mass[x]: "0"
  counter: "-k*counter"
state_metadata:
  rho: {representation: density, units: items/m, measure_axes: [x]}
  mass: {representation: cell_mass, measure_axes: []}
  counter: {representation: cumulative}
parameters:
  gain: {support: positive}
observables:
  physical_total:
    expr: "gain*rho[x]"
    reduction: {kind: integral, axes: [x]}
    units: items
  extensive_total:
    expr: "mass[x]"
    reduction: {kind: sum, axes: [x]}
  counter_value:
    expr: counter
  center_sample:
    expr: "rho[0.25]"
  field:
    expr: "gain*rho[x]"
    axes: [x]
```

The counter equation here illustrates that its representation annotation does
not establish monotonicity. Choose the scientific counter dynamics explicitly.

Definitions require `expr`. Optional `axes` give output axes **after** reduction;
omission means scalar output. Optional `reduction` has exactly `kind` (`sum` or
`integral`) and a nonempty, unique list of `axes`. Output and reduction axes must
be disjoint and declared by the model. `units` and `description` are optional
nonempty strings. Unknown fields, unresolved axes/coordinates and incompatible
parameter layouts fail before returning a compiled artifact. An empty mapping
or omitted `observables` leaves the existing RHS interface unchanged.

The initial pointwise subset is `+`, `-`, `*`, `/`, `**`, unary `+`/`-`, and unary
`np.exp`. State arrays require named subscripts, such as `rho[x]`; scalar states
use bare names. A literal coordinate selects an **existing declared node**, such
as `rho[0.25]`. Named categorical selection also works, such as
`rho[region:left,x]`. This selection does not interpolate. Bare nonstate symbols
are inferred as scalar parameters; subscripted nonstate symbols define shaped
parameter buffers. Shaped parameters use complete arrays under their base names
and must have exactly their exported shapes. Expanded scalar parameter keywords
are not an observation-buffer interface.

`sum` applies unit weights even on continuous axes. `integral` uses only the
explicit axis `measure` from [state representations](state-representations.md),
once per reduced axis. Intensive state dependencies must declare `density` or
`concentration` and `measure_axes`; extensive counts, cell masses and cumulative
counters reject physical weighting. Partial spatial integration retains output
axes while applying only the requested measures. Multiple physical axes use
separable products of the declared weights. Extra noncounting measure axes must
be required by at least one referenced intensive state. Parameter-only constant
integrands are allowed with explicitly declared measures.

These checks distinguish known extensive states and physical measures. They do
not establish dimensions, geometry, or the units of arbitrary products and
parameter fields; supplied output units remain annotations. Legacy equation
reducers and their continuous-axis `deltas` are unchanged. Nested reducers,
aliases, observer-to-observer references, history, time-varying parameter buffers,
partial/duplicate state-base layouts, and other pointwise functions are outside
this first subset and fail explicitly.

For the example, `rho(x)=1+2*x` has exact cell-average values `[1.1,1.5,2.4]`.
Declared widths give integral `2`, while the unweighted density sum is `5`.
The extensive masses `[0.11,0.45,1.44]` sum directly to `2`.

```python
import numpy as np
from op_system import compile_spec

compiled = compile_spec({
    "state": ["x"],
    "equations": {"x": "-k*x"},
    "parameters": {"gain": {"support": "positive"}},
    "observables": {"scaled": {"expr": "gain*x+t", "units": "declared-output"}},
})
y = np.array([2.0])
rhs_value = compiled.eval_fn(3.0, y, k=0.5)  # gain is not an RHS requirement
observable = compiled.observables["scaled"]
observed = observable.eval_fn(3.0, y, gain=4.0)  # scalar native value: 11
assert np.asarray(observed).shape == observable.output_shape == ()
assert observable.parameter_shapes == (("gain", ()),)
```

Observable-only parameter declarations are accepted in `parameters`, without
adding them to the RHS `param_names` or equation binding requirements.
`validate_spec`'s consumed parameter report continues to describe dynamics;
observable requirements are exported independently in `parameter_shapes`.
State/parameter dependencies describe the instantaneous expression, not all
indirect solution sensitivities. `time_dependent` specifically indicates an
explicit `t` reference; a state-dependent observable can evolve in time even
without one.

`eval_fn(t,y,**params)` consumes the complete flat `(len(state_names),)` state.
`pytree_eval_fn` consumes all state bases with their natural shapes, including
zero-dimensional scalar arrays. States must share a namespace and real floating
dtype. Array parameters must use that namespace; host scalar/list constants are
converted and native arrays are cast without detaching derivatives. Results use
the state dtype and declared shape. Evaluators perform functional operations and
may return views, so treat inputs and outputs as immutable. Trusted model pickle
round trips rebuild these evaluators with their detached definitions.

The required qualification runner includes independent nonuniform-grid values,
axis reordering, shaped parameters, counters and declared-node samples. NumPy
checks analytic values; JAX checks float64 JIT/vmap and state/parameter/time
derivatives; CPU Torch checks those derivatives in float64 and float32. Gradient
checks use the initial arithmetic/exp subset and static measures. They do not
qualify mesh-coordinate/measure derivatives, Torch compiler transforms, GPUs,
discontinuities or generic future operations.

Numerical interpolation, time windows and integrals, trajectory storage and mesh
transfer belong to consuming engines. Noise, censoring and missingness belong to
statistical measurement models. Global observations stay on the full compiled
model even when dynamics have a per-block evaluator; summing independent dynamic
blocks can still couple a statistical objective.

Point quadrature integrates point samples over their coordinate span, whereas
cell measures apply the discretization owner's widths/volumes to cell averages.
For the point-rule reference see [SciPy trapezoid](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html).
The compiler evaluates the supplied measures with native products and reductions
so it does not introduce a SciPy runtime or assume a geometry from cell centers.
