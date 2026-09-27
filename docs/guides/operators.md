# Operator metadata

`op_system` validates operators as model-level metadata and publishes them as
typed `OperatorDescriptor` values. Engines own spatial discretization and
solver staging, but they must agree on the meaning of the descriptor.

## Advection and transport

Advection acts along the declared coordinate order. With no `direction`,
`velocity` is signed: positive moves toward increasing indices and negative
moves toward decreasing indices.

Use `direction` when the orientation is structural but the non-negative
coefficient remains dynamic:

```yaml
operators:
  - kind: advection
    axis: imm
    velocity: waning_rate
    direction: decreasing
    bc: reflecting
```

Providers multiply an `increasing` coefficient by `+1` and a `decreasing`
coefficient by `-1`. Numeric or traced coefficients supplied with a direction
should therefore be non-negative.

Boundary conditions are relative to the resolved direction:

- `absorbing`: zero upstream inflow and free downstream outflow;
- `reflecting`: zero upstream inflow and zero downstream flux, with mass
  accumulating in the terminal cell;
- `periodic`: downstream outflow wraps to the upstream cell.

These rules apply identically for increasing and decreasing transport. The
descriptor keeps direction separate from a parameter name so engines never
need to parse backend-specific scalar expressions such as `-waning_rate`.

## Jump integrals

`jump_integral` is a conservative gain-minus-loss operator along one axis. Its
matrix uses **rows as sources and columns as targets**. Given a non-negative
off-diagonal rate-density matrix `J`, op_system defines a generator `Q` by
masking disallowed destinations and setting each diagonal to the negative
off-diagonal row sum. Engines apply

\[
  \frac{d x}{d t}\bigg|_{jump} = r\,xQ
\]

along the declared axis. The scalar `rate` `r` is separate so it can remain a
dynamic inference parameter.

```yaml
operators:
  - kind: jump_integral
    axis: trait
    rate: jump_rate
    direction: up
    bc: reflecting
    kernel:
      form: matrix
      params: {matrix: trait_jump_density}
      param_axes: {trait_jump_density: [trait, trait]}
```

The current contract intentionally has one kernel form, `matrix`. Unknown
forms are rejected. The matrix diagonal must be zero and off-diagonal values
must be finite and non-negative. `kernel.param_axes` is mandatory and must
declare `[axis, axis]`; this fixes orientation and lets providers request the
parameter with the correct shape.

Directions are defined in declared coordinate order:

- `up` retains source-to-target entries with `target > source`;
- `down` retains entries with `target < source`;
- `both` retains every off-diagonal entry and is the default.

Categorical axes are unordered and therefore accept only `both`. Ordinal axes
use their listed order. Continuous axes require strictly increasing numeric
coordinates and multiply each target column by the axis's trapezoidal
quadrature weight. Thus a continuous kernel has rate-density units inverse to
the axis units, while `rate * J * target_weight` has inverse-time units.

Only `reflecting` boundaries are defined. This means truncation at the declared
domain: no jump outside the coordinate set occurs, the remaining in-domain
off-diagonal rates are not renormalized, and a terminal source with no allowed
target has a zero generator row. Every row sums to zero, so mass is conserved
for every independent slice of the other state axes. `absorbing` and
`periodic` are rejected until their non-local geometry is specified.

`jump_integral_generator`, `jump_integral_rhs`, and
`validate_jump_integral_kernel` are public Array-API reference functions.
Engines must match them exactly. Validate concrete parameter values at an eager
orchestration boundary; the reference assembly itself keeps matrix values
dynamic for JAX, Torch, and other differentiable array backends.

The older project-local Diffrax behavior that treated `kernel.params` as
per-coordinate diagonal outflow is not this operator: it omitted direction and
required a separately duplicated inflow. Model that behavior as an explicit
source transition, or migrate it to the conservative matrix contract.
