# Writing a compartmental model

This SEIR example introduces the syntax through one deterministic compartmental
model. The same specification layer also describes spatial/trait fields,
nonlocal kernel interactions and stochastic channels. See the
[model-class map](capabilities.md#model-classes-and-numerical-representations) and
[operator contracts](operators.md) for the corresponding interfaces and boundaries.

Install the core compiler and a YAML reader:

```shell
python -m pip install op-system PyYAML
```

The compiler accepts a Python mapping. YAML/JSON loading is a caller concern;
`compile_spec(spec)` validates, normalizes, and compiles it. `normalize_rhs(spec)`
returns a normalized model without compilation, and `validate_spec(spec)`
returns a structural report. There is no public `load_spec` file loader.

Optional [parameter declarations](parameter-declarations.md) attach support,
units, descriptions and fixed/free status to inferred RHS parameters for
downstream consumers, without changing numerical evaluation.

## A complete age-structured SEIR model

Save the following as `examples/age-seir.yaml` (create the `examples` directory
first), or use the [checked example](https://github.com/ACCIDDA/op_system/blob/main/examples/age-seir.yaml).
The groups are categorical population bins, so mixing uses an unweighted sum.
Rates are per unit time. This synthetic example demonstrates authoring and
balance checks; choose and independently qualify your scientific equations.

<!-- seir-spec -->
```yaml
# Shared rate beta; susceptibility varies by target age group.
kind: transitions
axes:
  - name: age
    type: categorical
    coords: [young, old]
state: ["S[age]", "E[age]", "I[age]", "R[age]"]
aliases:
  "N[age]": "S[age] + E[age] + I[age] + R[age]"
  "foi[age]": "beta * susceptibility[age] * apply_along(C[age:s,age] * I[age:s] / N[age:s], age=s, kernel=sum)"
transitions:
  - name: infection
    from: "S[age]"
    to: "E[age]"
    rate: "foi[age]"
  - name: progression
    from: "E[age]"
    to: "I[age]"
    rate: "kappa"
  - name: recovery
    from: "I[age]"
    to: "R[age]"
    rate: "gamma"
# Optional provider seed metadata; runtime values are supplied separately.
initial_state:
  "S[age]": {shaped: S0, axes: [age]}
  "E[age]": {shaped: E0, axes: [age]}
  "I[age]": {shaped: I0, axes: [age]}
  "R[age]": {shaped: R0, axes: [age]}
```

`state` order fixes the flat layout: all young/old S cells, then E, I, and R.
Each bracketed state carries only its listed axes. Bare states are scalar;
there is no automatic inheritance of every declared axis. Quote selectors in
YAML flow lists, because their brackets and commas are part of the string.

An alias is an expression, not another compartment. `N[age]` is the current
population in that age bin. `C[source_age, target_age]` has source rows and target
columns: `C[age:s,age]` binds the first index to source `s`, keeps the second at
the current target age, and sums the source prevalence `I/N`. Use an asymmetric
matrix while checking orientation; a symmetric matrix can hide a transpose.

A transition with `from` uses a **per-capita hazard**: the infection flow is
`foi[age] * S[age]`, not just `foi[age]`. Do not multiply by the source a second
time in its rate. `from` omitted/null creates an absolute source inflow.
`to` must name a declared state: omitted/null sink destinations are currently
unsupported and fail validation. Use an expression equation for an open loss,
or track the loss in an explicit compartment. The latter retains the lost
amount in the declared state total and exports a closed transfer reaction.
A closed transition subtracts and adds the same flow, preserving population.

These standalone alternatives both give `x' = -k*x`; the explicit loss
compartment additionally gives `loss' = k*x`:

<!-- loss-alternatives -->
```python
import numpy as np
from op_system import compile_spec

open_loss = compile_spec({
    "kind": "expr",
    "state": ["x"],
    "equations": {"x": "-k*x"},
})
tracked_loss = compile_spec({
    "kind": "transitions",
    "state": ["x", "loss"],
    "transitions": [{"name": "loss", "from": "x", "to": "loss", "rate": "k"}],
})
open_derivative = open_loss.eval_fn(0.0, np.asarray([3.0]), k=2.0)  # [-6]
tracked_derivative = tracked_loss.eval_fn(0.0, np.asarray([3.0, 0.0]), k=2.0)  # [-6, 6]
```

Expression equations do not currently export executable jump reactions; choose
the representation required by the consuming deterministic/stochastic engine.

`initial_state` records provider seed parameter names and constants. It does not
allocate or initialize core arrays. A consuming framework resolves `S0`, `E0`, `I0`,
and `R0`; core callers supply the state themselves as below. The shaped form
states the parameter's axis order explicitly. A scalar mapping such as `S: S0` or
a per-cell template such as `"S[age]": "S0[age]"` is also supported.

A state that starts at a known constant (typically `0` for a cumulative counter
fed by a source-only transition) needs no seed parameter: give the key a numeric
literal, for example `"C[age]": 0.0` or `D: 0`. The literal is broadcast to every
cell the key selects (a pinned coordinate such as `"C[age=young]": 0.0` selects
one), is stored as a `float` in `compiled.meta["initial_state"]`, and requests no
provider parameter. It must be a finite real number; booleans, `nan`, and `inf`
are rejected. There is no `{fixed: ...}`/`{value: ...}` mapping spelling, and a
quoted `"0.0"` is still a parameter name. States omitted from `initial_state`
are left to the consuming framework (the op_engine provider uses zero, but a
literal records the intent explicitly).

## Evaluate the example

Run from the directory containing `examples/age-seir.yaml`. The scalar parameters
`beta`, `kappa`, and `gamma` are shared, `susceptibility` varies over the target
age axis, and `C` is shared as a two-axis matrix. Arrays use float64 throughout.
`compiled.param_names` lists inputs; `validate_spec(spec).parameters` reports
scalar and shaped parameter axes;
`compiled.template_shapes` describes the structured layout.

<!-- seir-evaluation -->
```python
from pathlib import Path

import numpy as np
import yaml

from op_system import compile_spec

spec = yaml.safe_load(Path("examples/age-seir.yaml").read_text())
compiled = compile_spec(spec)
state = {
    "S": np.asarray([900.0, 600.0]),
    "E": np.asarray([20.0, 10.0]),
    "I": np.asarray([30.0, 40.0]),
    "R": np.asarray([50.0, 50.0]),
}
params = {
    "beta": 0.3,
    "kappa": 0.2,
    "gamma": 0.1,
    "susceptibility": np.asarray([1.0, 0.7]),
    "C": np.asarray([[4.0, 1.0], [2.0, 3.0]]),
}
flat = np.concatenate([state[name] for name in ("S", "E", "I", "R")])
flat_rhs = compiled.eval_fn(0.0, flat, **params)
assert compiled.pytree_eval_fn is not None
rhs = compiled.pytree_eval_fn(0.0, state, **params)
np.testing.assert_allclose(
    flat_rhs, np.concatenate([rhs[name] for name in ("S", "E", "I", "R")])
)
np.testing.assert_allclose(sum(rhs.values()), np.zeros(2), atol=1e-12)
print(np.round(np.stack(list(rhs.values())), 3))
```

Expected rows are S, E, I, R, with columns young, old:

```text
[[-63.257 -25.38 ]
 [ 59.257  23.38 ]
 [  1.     -2.   ]
 [  3.      4.   ]]
```

This evaluates a derivative. Use a numerical engine to integrate it in time.
For scientific checks beyond structural validation, use
[`qualify_rhs`](numerical-qualification.md) with independently written equations
and fixtures. The guide's CI test executes these exact YAML/Python fences and
checks cell flows and per-age conservation against an independent loop oracle.

## Equations and expression syntax

Use `kind: expr` with one equation for each declared state to write derivatives
directly. The following equations are equivalent to the transitions above when
the same axes, aliases, and state are retained and `transitions` is removed:

<!-- seir-equations -->
```yaml
kind: expr
equations:
  "S[age]": "-foi[age] * S[age]"
  "E[age]": "foi[age] * S[age] - kappa * E[age]"
  "I[age]": "kappa * E[age] - gamma * I[age]"
  "R[age]": "gamma * I[age]"
```

Expressions support allowlisted arithmetic, named state/parameter references,
aliases, and documented helper calls. They are not arbitrary Python programs:
imports, attribute traversal, and arbitrary function calls are rejected.
Reference a templated state as `I[age]`; a bare `I` is invalid unless a scalar
state `I` is declared. Expression bindings use `age:s` inside brackets and
`age=s` in reducer arguments. Endpoint pinning uses `"S[age=young]"` in transition
selectors. See [expression selectors](expression-selectors.md),
[global totals](global-state-totals.md), and [aging chains](aging-chains.md).

## Axes and mixing choices

| Axis type | Coordinates and reduction |
| --- | --- |
| Categorical (default) | Unique nonempty labels; `sum_over` / `apply_along(..., kernel=sum)` use unweighted sums |
| Ordinal | Unique labels in declared order; reductions use sums while directional operators use the order |
| Continuous | Strictly increasing numeric coordinates; `integrate_over` / default `apply_along` use trapezoidal weights, requiring at least two points except the special constant time-table axis |

Nonuniform continuous grids carry quadrature weights; interpreting population
counts as density values changes units. `sum_state()` totals declared cells
without quadrature. For example, `sum_over(I[age:a], age=a)` sums discrete age
bins, while `integrate_over(u[x:i], x=i)` integrates a field over continuous `x`.
Time is a separate forcing axis; see [time-indexed parameters](time-indexed-parameters.md).

Add spatial/risk axes explicitly to states and use local parameters such as
`beta[loc]`. An independent `loc` can be declared in `factorize_axes: [loc]` only
when no RHS, alias, or operator couples locations. The compiler rejects invalid
factorization; see [independent batches](batched-models.md). Full and block
layouts are separate choices from the runtime [array namespace](array-namespaces.md).

Core callers supply matrix `C` as a parameter. Provider users can alternatively
declare `kernels` with an explicit `value` matrix or a supported analytic form
and declared numeric axes; the provider constructs and injects those values.
Core compilation preserves these declarations as metadata and does not itself
materialize analytic kernels. See the
[provider README](https://github.com/ACCIDDA/op_system/blob/main/flepimop2-op_system/README.md)
and [coverage audit](../development/coverage-audit.md) for generation contracts.

## Operators and splitting

An `operators` block describes processes for a consuming numerical engine,
separately from equations/transitions. For example, an ordinal immune-state axis
can declare advection with a rate parameter and direction, while a continuous
trait can declare conservative jumps. Compilation exposes typed descriptors in
`compiled.operators`; `eval_fn` alone does not advance or apply these metadata
operators. The engine must resolve coefficients, discretization, boundaries,
and splitting stages. Follow the [operator guide](operators.md) for supported
schemas and conservative reference functions; do not count a process twice by
also including it in the RHS.

## Cross-axis constraints: current boundary

`constraints` is currently a reserved metadata block. Public normalization
preserves it verbatim; the private constraint parser's validation does not
provide public masking. State expansion and reductions still cover the full
Cartesian product. Filtering and constraint-aware reductions are tracked in
[#37](https://github.com/ACCIDDA/op_system/issues/37) and
[#38](https://github.com/ACCIDDA/op_system/issues/38).

The planned schema uses at least two axes and either `allow` or `exclude`:

```yaml
# Planned masking metadata; this does not remove current compartments.
constraints:
  - axes: [age, vax]
    allow:
      - {age: young, vax: [none]}
      - {age: old}
```

An allow rule would retain its matching combinations; an omitted axis in a
rule would match all its coordinates. An exclude rule would remove matches.
Until filtering is implemented, represent asymmetric populations explicitly,
for example `state: ["S_young", "S_old[vax]"]`, with equations/transitions for
those states and independently checked totals. Do not rely on this reserved
block to prevent invalid flows or omit state cells. Broader constraint examples
remain tracked in [#39](https://github.com/ACCIDDA/op_system/issues/39).
