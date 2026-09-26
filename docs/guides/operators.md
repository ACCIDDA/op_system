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
