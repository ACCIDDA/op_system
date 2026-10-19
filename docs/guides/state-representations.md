# State representations and physical measures

An optional `state_metadata` block describes state values. Declarations use
state **base names**, such as `rho` for `rho[space]`, and work in expression and
transition specifications. Partial declarations are allowed. Metadata leaves
equations, runtime values, constraints and existing reductions unchanged.

The optional `scale: log` field requests a separate deterministic
[coordinate view](log-state-coordinates.md). It leaves the primary evaluator
and stochastic reactions in physical coordinates; callers explicitly encode
initial states and select the view's callbacks.

```yaml
kind: expr
axes:
  - name: space
    type: continuous
    coords: [0.05, 0.25, 0.7]
    units: m
    measure:
      kind: cell
      weights: [0.1, 0.3, 0.6]
      units: m
state: ["rho[space]", "mass[space]", counter]
equations:
  rho[space]: "0"
  mass[space]: "0"
  counter: "sum_state()"
state_metadata:
  rho:
    representation: density
    units: items/m
    description: Cell-average density on the unit interval
    measure_axes: [space]
  mass:
    representation: counts
    units: items
    measure_axes: []
  counter:
    representation: cumulative
    units: items
```

The illustrative counter derivative exposes the existing total reduction;
choose scientific counter equations explicitly in an actual model.

| Representation | Stored value | Physical total over represented cells |
| --- | --- | --- |
| `counts` | Extensive count per cell/category | Unweighted sum |
| `cell_mass` | Extensive amount or mass per cell | Unweighted sum |
| `density` | Amount per declared spatial measure | Sum after multiplying by that measure |
| `concentration` | Amount per declared volume or reference measure | Sum after multiplying by that measure |
| `cumulative` | Extensive accumulated counter | Unweighted spatial sum; time differences require a separately defined observation |

Counts are annotations, rather than an integer execution constraint.
`cumulative` neither proves monotonic equations nor turns an instantaneous
derivative into an interval count. Missing `representation` remains unknown.
For intensive states, omitted `measure_axes` means integration requirements
are undeclared; `[]` explicitly requires no measure axes. Listed axes must
belong to the state template or its legacy `state_axes` declaration and have
explicit measures. Extensive representations reject nonempty `measure_axes`
to avoid weighting an amount twice.

`CompiledRhs.state_meta` is a read-only mapping from declared bases to exported
immutable `StateDeclaration` records; undeclared bases are absent.
`CompiledRhs.axis_meta` contains all declared axes as `AxisDeclaration` records
with coordinate `units` and optional `AxisMeasure`. Undeclared units/measures
are `None`, including on continuous axes. Trusted pickle round trips rebuild
the detached snapshots alongside the evaluator.

Supported measure kinds are:

- `counting`: unit weight per coordinate; omit `weights`.
- `cell`: explicit cell widths, areas or volumes aligned with coordinates.
- `quadrature`: explicit point weights aligned with coordinates.

Cell/quadrature weights must match the coordinate count and be finite, strictly
positive real numbers. Boolean, zero, negative and nonfinite weights fail.
Signed quadrature, zero-measure masks, runtime measure buffers and nonseparable
multidimensional geometry are outside this initial interface. Coordinate and
measure units are independent: coordinates may use `m` while volumes use
`m^3`. Annotations do not verify dimensions or geometry.

The discretization owner supplies the measure. It is never inferred from
centers or substituted for existing continuous-axis `deltas`. Those deltas
retain point-trapezoidal semantics over the coordinate span. Centers
`[0.05, 0.25, 0.7]` span `0.65`; the declared volumes `[0.1, 0.3, 0.6]` cover
the unit interval. These describe different discretizations.

This runnable example explicitly reduces a density using the exported measure:

```python
import numpy as np
from op_system import array_namespace, compile_spec

compiled = compile_spec({
    "axes": [
        {
            "name": "space",
            "type": "continuous",
            "coords": [0.05, 0.25, 0.7],
            "measure": {"kind": "cell", "weights": [0.1, 0.3, 0.6], "units": "m"},
        }
    ],
    "state": ["rho[space]"],
    "equations": {"rho[space]": "0"},
    "state_metadata": {
        "rho": {
            "representation": "density",
            "units": "items/m",
            "measure_axes": ["space"],
        }
    },
})
density = np.asarray([1.1, 1.5, 2.4])  # Cell averages of 1+2*x on [0,1].
xp = array_namespace(density)
measure = compiled.axis_meta["space"].measure
assert measure is not None
weights = xp.asarray(measure.weights, dtype=density.dtype)
cell_counts = xp.multiply(density, weights)
physical_total = xp.sum(cell_counts)  # 2: exact integral over [0,1].
unweighted_density_total = xp.sum(density)  # 5: a different quantity.
```

These operations preserve NumPy/JAX/Torch namespace and floating dtype;
required backend qualification executes their numerical expectations.
Metadata supplies no new expression helper: [`sum_state()`](global-state-totals.md)
remains unweighted, and existing integration helpers retain their point weights.
Named physical observations and mesh-transfer consumers are separate work in
[op_system #294](https://github.com/ACCIDDA/op_system/issues/294),
[op_engine #225](https://github.com/ACCIDDA/op_engine/issues/225) and
[op_engine #226](https://github.com/ACCIDDA/op_engine/issues/226).
Those consumers must check representation, measure coverage, unit assumptions
and actual discretization accuracy before claiming an integral or conservation.

Existing specs need no migration. When annotating a state, review whether its
values are extensive or intensive and supply the measure explicitly. This
interface does not convert representations, constrain signs or counters, or
certify scientific validity.

Use [named latent observables](latent-observables.md) to compile these explicit
physical reductions into independently discoverable fixed-mesh outputs.


[Conditional scalar reaction balances](reaction-balances.md) retain these
declarations without inferring physical mass or spatial weights. Known unit or
representation conflicts within a proposed total remain visible coverage gaps;
unknown labels remain assumptions.
