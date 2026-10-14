# Parameter declarations

An optional `parameters` block records scientific annotations for parameters
already inferred from the RHS. It does not introduce symbols, supply values,
enforce runtime bounds, select an inference library or change the equations.
Specs without this block keep their existing behavior.

```yaml
kind: expr
state: [x]
equations:
  x: "offset - rate*x"
parameters:
  rate:
    support: positive
    units: "day^-1"
    description: "Loss rate"
    fixed: false
  offset:
    support: {kind: bounded, lower: 0, upper: 10}
    units: "amount/day"
    fixed: true
```

Each entry is optional, as is each field within an entry. An omitted support
stays unknown; it is not inferred as `real`. Undeclared RHS parameters remain
valid. Unknown declaration fields and names that are not inferred runtime
parameters are errors. Values and priors belong in the caller or statistical
layer, outside this block.

| Field | Meaning |
| --- | --- |
| `support` | `real`, strict `positive` (`x > 0`), closed `unit_interval` (`0 <= x <= 1`), or a bounded mapping |
| `support.kind: bounded` | Requires finite numeric `lower < upper`; both endpoints are included |
| `units` | Nonempty text annotation; no dimensional-analysis proof is implied |
| `description` | Nonempty human-readable text |
| `fixed` | Boolean inference annotation, default `false`; it neither binds a value nor removes a runtime argument |

The same block is supported for expression and transition specs. Declare a
shaped or time-varying buffer by its base name, such as `rate`, rather than
`rate[unit]` or a coordinate-expanded cell name. Alias-only parameter
dependencies are included. States, axes and compiler-synthesized masks are
not parameters that this block can declare. Operator-only coefficients and
initial-state-only inputs are outside this RHS declaration interface.

Compilation exposes a detached read-only mapping, `CompiledRhs.param_meta`,
whose values are immutable `ParameterDeclaration` records. Their optional
`support` values are `ParameterSupport` records. Both types are exported by
`op_system` and `op_system.specs`. Metadata survives compiled-object pickle
round trips by recompiling its normalized source.

```python
import numpy as np
from op_system import compile_spec

compiled = compile_spec({
    "kind": "expr",
    "state": ["x"],
    "equations": {"x": "-rate*x"},
    "parameters": {"rate": {"support": "positive", "fixed": True}},
})
assert compiled.param_meta["rate"].fixed
support = compiled.param_meta["rate"].support
assert support is not None and support.kind == "positive"
np.testing.assert_allclose(compiled.eval_fn(0.0, np.array([3.0]), rate=0.5), [-1.5])
```

Downstream tools decide how to use these declarations, including support
validation, fixed-value binding, prior selection and coordinate transforms.
For example, a logit transform covers the interior of a closed unit interval;
endpoint atoms require their own statistical treatment. Array namespace,
dtype and differentiation behavior remain properties of the numerical
evaluation, independent of these static annotations.

[Named latent observables](latent-observables.md) can infer additional scalar or
shaped parameters and annotate them in this same `parameters` block. Those names
appear in `param_meta` and in each observable's `parameter_shapes`, while the
existing RHS binding requirements and `param_names` remain unchanged.
