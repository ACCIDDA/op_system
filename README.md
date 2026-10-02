# op_system

Domain-agnostic specification and compilation of right-hand sides (RHS) for
ODE, PDE, and multi-physics / multi-scale compartmental systems.  `op_system`
takes a YAML/JSON-friendly spec, validates and normalizes it, then compiles
it into a fast, **array-API-polymorphic** callable whose namespace is selected
from the inputs at call time. NumPy, JAX (concrete and traced), and raw PyTorch
tensors are covered by the test suite. Other Array-API implementations may
work through `array-api-compat`, but should be qualified before production use.

- Docs: <https://accidda.github.io/op_system/>
- License: MIT
- Python: 3.11 – 3.13

## Why op_system?

Modelers often combine compartment hazards, templated populations, and rich
metadata (axes, kernels, operators) that must be validated and preserved for
downstream solvers.  `op_system` provides:

- **Two equivalent surfaces** — `expr` (explicit equations) and `transitions`
  (hazard / flow style) — that share the same axis, alias, template, and
  reducer machinery.
- **Validated, restricted expression parsing** with a small allowlist of
  NumPy ops and helpers; no arbitrary code execution.
- **A typed intermediate representation (IR)** that handles template
  expansion, alias inlining, and `apply_along`/`sum_over` reductions
  symbolically before code generation.
- **Vectorized compilation** that operates on shaped state buffers (one
  tensor expression per template) rather than per-cell scalar code, with
  template-level common-subexpression elimination.
- **Backend polymorphism at call time** — `array-api-compat` selects the
  namespace from each input, so the same compiled artifact serves NumPy,
  JAX `jit`/`vmap`/`grad`, and raw PyTorch tensors with autograd.
- **First-class PyTree interface** (`pytree_eval_fn`) for engines that want
  to keep state as a dict of shaped arrays rather than a flat vector.
- **Block-axis vmap support** (`block_pytree_eval_fn`) for hierarchical
  models — declare `factorize_axes` and the engine can vmap a stripped
  per-block RHS over a block axis instead of evaluating a monolithic flat
  state.
- **Picklable `CompiledRhs`** — round-trips through `pickle.dumps`/`loads`
  by retaining the source spec and recompiling on load.

## Installation

```bash
pip install op-system
# or, from a checkout, using uv:
uv pip install .
```

Optional extras:

```bash
pip install "op-system[jax]"            # JAX runtime support
pip install "op-system[jax-inference]"  # adds diffrax + blackjax
pip install "op-system[torch]"          # PyTorch runtime support
pip install "op-system[data]"           # pandas + pyarrow helpers
```

## Time-indexed parameters

Time-indexed parameter tables use linear interpolation by default. Set
`time_interpolation: previous` in a specification to hold each table value
until the next coordinate, with right-continuous changes and constant endpoint
extrapolation. The same policy applies to flat, PyTree, block, and reaction
evaluators. Compiled metadata and the flepimop2 provider publish immutable
forcing coordinates for numerical engines. See the
[time-indexed parameter guide](https://accidda.github.io/op_system/guides/time-indexed-parameters/)
for the schema, examples, and exactness conditions.

## Quick start

```python
import numpy as np

from op_system import compile_spec

spec = {
    "kind": "expr",
    "state": ["S", "I", "R"],
    "aliases": {"N": "S + I + R"},
    "equations": {
        "S": "-beta * S * I / N",
        "I": "beta * S * I / N - gamma * I",
        "R": "gamma * I",
    },
}

compiled = compile_spec(spec)
dydt = compiled.eval_fn(0.0, np.asarray([999.0, 1.0, 0.0]), beta=0.3, gamma=0.1)
```

The compiled object exposes:

| Attribute | Description |
|---|---|
| `eval_fn(t, y, **params) -> dydt` | Flat-vector RHS; array namespace inferred from `y`. |
| `pytree_eval_fn(t, state_dict, **params) -> dict` | PyTree RHS keyed by state template base name (axis-indexed specs). |
| `template_shapes` | `{base: shape}` for each state template. |
| `state_names`, `param_names` | Tuples of expanded state cells and parameter names. |
| `factorize_axes`, `block_axes` | Axes the IR proved separable for block vmap. |
| `block_pytree_eval_fn`, `block_template_shapes` | Per-block PyTree RHS with the first factorize axis stripped. |
| `meta` | Normalized metadata (axes, state_axes, kernels, operators, reserved blocks). |
| `operators` | Tuple of `OperatorDescriptor` preserving normalized names, state selectors, coefficients, directions, boundary conditions, and kernel metadata. |
| `reactions`, `reaction_gaps` | One `CompiledReaction` per named transition with a reaction artifact, and one `ReactionGap` per transition without one (empty when every transition is covered). |

### Advection contract

Advection and transport act along the declared coordinate order. A signed
`velocity` without `direction` is used directly: positive moves toward
increasing indices and negative moves toward decreasing indices. An optional
direction makes the orientation explicit while keeping a dynamic coefficient:

```yaml
operators:
  - kind: advection
    axis: imm
    velocity: waning_rate
    direction: decreasing
    bc: reflecting
```

Providers multiply an `increasing` coefficient by `+1` and a `decreasing`
coefficient by `-1`. Coefficients used with explicit direction should
therefore be non-negative; producers of traced dynamic values are responsible
for that invariant.

Boundary conditions are defined relative to the resolved direction:

- `absorbing` uses zero upstream inflow and permits downstream outflow;
- `reflecting` uses zero upstream inflow and zero downstream flux, so mass
  accumulates in the terminal cell;
- `periodic` wraps downstream outflow to the upstream cell.

Engines must apply these semantics identically for either velocity sign.

### Jump-integral contract

`jump_integral` metadata defines a conservative row-source, column-target
matrix generator along an axis. `direction: up|down|both` masks destinations;
continuous axes use target trapezoidal weights; and the currently supported
`reflecting` boundary truncates out-of-domain jumps without renormalizing or
losing mass. See the [operator guide](https://accidda.github.io/op_system/guides/operators/)
for the exact schema, units, and Array-API reference functions.

`compile_spec` accepts legacy `backend=` / `xp=` keyword arguments but they
are deprecated and ignored — the compiled callable infers its array
namespace from the input `y` on every call.

## JAX usage

```python
import jax, jax.numpy as jnp
from op_system import compile_spec

compiled = compile_spec(spec)
y0 = jnp.asarray([999.0, 1.0, 0.0])

# Native JAX call — eval_fn returns a jnp array.
dydt = compiled.eval_fn(0.0, y0, beta=0.3, gamma=0.1)

# Works inside jit / vmap / grad without recompilation.
solve = jax.jit(lambda y: compiled.eval_fn(0.0, y, beta=0.3, gamma=0.1))
```

For diffrax-based ODE solves and NUTS / HMC inference, install the
`jax-inference` extra above.

## YAML examples

The full guide of YAML patterns — including templates, axis asymmetry,
chains, continuous axes with kernels, and block-axis hierarchical models —
lives at <https://accidda.github.io/op_system/guides/getting-started/>.
A few highlights:

### Baseline SIR (two pathways)

```yaml
# expr
spec:
  kind: expr
  state: [S, I, R]
  equations:
    S: -beta * S * I / sum_state()
    I:  beta * S * I / sum_state() - gamma * I
    R:  gamma * I
```

```yaml
# transitions
spec:
  kind: transitions
  state: [S, I, R]
  transitions:
    - {from: S, to: I, rate: beta * I / sum_state()}
    - {from: I, to: R, rate: gamma}
```

Source-only tracking transitions are also supported (``from: null`` or omitted):

```yaml
spec:
  kind: transitions
  state: [I, H_cum]
  transitions:
    - {to: H_cum, rate: k * I}  # equivalent to {from: null, ...}
```

This pattern is useful for cumulative trackers (e.g., weekly admissions via
``diff(H_cum)``) without introducing a dummy donor compartment.

Named transitions may also declare the molecular reactants needed by
stochastic solvers. The list is independent of net source/target
stoichiometry, so it must include the consumed source as well as catalysts:

```yaml
spec:
  kind: transitions
  axes:
    - {name: age, coords: [child, adult]}
    - {name: vax, coords: [u, v]}
  state: [S[age,vax], E[age,vax], I[age]]
  transitions:
    - name: infection
      from: S[age,vax]
      to: E[age,vax]
      rate: beta * I[age]
      reactants:
        - {state: S[age,vax], order: 1}
        - {state: I[age], order: 1}  # catalytic: not consumed
```

The compiled reaction exposes these entries as array-neutral structural
metadata. If `reactants` is omitted, op_system preserves compatibility by
publishing the consumed source at order one with `reactants_complete=false`;
adaptive stochastic consumers should require complete metadata rather than
try to infer catalysts from the rate expression. An explicit empty list marks
a source-only zero-order reaction as complete.

Not every transition publishes a reaction. `CompiledRhs.reaction_gaps` (and
the provider's `reaction_gaps` option) lists each one that does not, with its
spec origin (`transitions[1]`, `chain[0].forward[0]`), selectors, and a
reason such as `unnamed`, `routing`, or `unsupported_layout`. An `expr` spec
reports a single `expr_spec` gap. A consumer that executes only the reactions,
such as a pure stochastic simulation, should reject a non-empty value rather
than silently drop those dynamics.

Source-only rates may also depend on population through a bound reduction,
such as `sum_over(B[age:a] * N[age:a], age=a)`, while their destination pins
`age=a0`. This produces one total birth hazard into that cell, without donor
depletion. See the [renewal births guide](docs/guides/renewal-births.md) for
reaction metadata, retained group axes, and a stationary age-population example.

### Templated states with `apply_along`

```yaml
spec:
  kind: expr
  axes:
    - {name: age,  coords: [child, adult]}
    - {name: vax,  coords: [u, v]}
  state: [S[age,vax], I[age,vax], R[age,vax]]
  aliases:
    lambda[age]: beta * apply_along(vax=j, I[age,vax=j]) / sum_state()
  equations:
    S[age,vax]: -lambda[age] * S[age,vax]
    I[age,vax]:  lambda[age] * S[age,vax] - gamma * I[age,vax]
    R[age,vax]:  gamma * I[age,vax]
```

`apply_along(axis=var, expr)` contracts `expr` along one or more axes in a
single call.  Categorical / ordinal axes use uniform weights of 1;
continuous axes use trapezoidal weights derived from axis spacing
(non-uniform supported).  Bindings can be restricted with
`axis=var in [...]` for sub-range integration.

### Routing transitions with `axis:alias`

```yaml
spec:
  kind: transitions
  axes:
    - {name: vax, coords: [u, v]}
    - {name: imm, type: ordinal, coords: [x0, x1, x2, x3]}
  state: [X[vax, imm]]
  transitions:
    - from: X[vax, imm:i]            # waning along a generator G
      to:   X[vax, imm:j]
      rate: waning_rate * G[imm:i, imm:j]
    - from: X[vax=u, imm:i]          # vaccination with routing weights eta
      to:   X[vax=v, imm:j]
      rate: nu * eta[time, imm:i, imm:j]
```

Binding the same axis under one alias in `from` and another in `to` moves
mass along that axis with a matrix-valued per-capita rate:
`dX_from[i] -= r X_from[i] sum_j K[i, j]` and
`dX_to[j] += r sum_i K[i, j] X_from[i]`. The rate must reference both
aliases on that axis; other axes are shared or pinned as usual. When
`from` and `to` are otherwise the same slice, the diagonal `K[i, i]` is a
no-op. One routed axis per transition; it cannot be a `factorize_axes`
block axis. Routing is lowered once per template, so its cost does not grow
with the number of matrix entries. It has no per-transition `reactions`
artifact yet.

A target-only alias fans one source cell into a target axis the source does
not own:

```yaml
spec:
  kind: transitions
  axes:
    - {name: age, coords: [child, adult]}
    - {name: imm, type: ordinal, coords: [x0, x1, x2]}
  state: [I3[age], X[age,imm]]
  transitions:
    - from: I3[age]
      to: X[age,imm:j]
      rate: reset_rate * reset_kernel[imm:j]
```

This compiles as one lazy transition. Each target receives
`reset_rate * reset_kernel[j] * I3`, while the source loses
`reset_rate * sum_j(reset_kernel[j]) * I3` exactly once. The weights are
arbitrary per-target rates; op_system does not force normalization. When they
sum to one, `reset_rate` is the total departure hazard. In every case the
generated source loss equals the summed target inflow, so the transition is
mass-conserving algebraically. Physical rate non-negativity remains a model
input responsibility, consistent with other transition rates.

### Chain helper

```yaml
spec:
  kind: transitions
  state: [S, I, R]
  chain:
    - name: I
      length: 3
      entry:   {from: S, rate: beta * S / sum_state()}
      forward: [gamma12, gamma23]
      exit:    {to: R, rate: gamma3r}
  transitions: []
```

`chain` synthesizes the staged compartments (`I1..I3`) and the internal
forward / exit transitions; declare only the base `I` in `state`.

### Axis-wide aging with `coord_shift`

```yaml
spec:
  kind: transitions
  axes:
    - {name: age, type: ordinal, coords: [a0, a1, a2, a3]}
  state: [S[age], I[age]]
  transitions:
    - name: aging
      coord_shift: {axis: age, step: 1, rate: "aging_rate[age]", boundary: absorb}
      apply_to: [S, I]
```

Every bin `k` moves to `k + step` at the source bin's rate. `boundary: absorb`
removes mass shifted off the axis, and `stay` keeps it in the terminal bin.
The entry lowers once per state, and named entries publish one templated
reaction per state with an `offsets` field. See the
[aging-chain guide](https://accidda.github.io/op_system/guides/aging-chains/).

### Continuous axis + kernel

```yaml
spec:
  kind: expr
  axes:
    - name: x
      type: continuous
      domain: {lb: 0.0, ub: 10.0}
      size: 5
      spacing: linear
  state: [u[x]]
  state_axes: {u: [x]}
  kernels:
    - {name: K, axes: [x], form: gaussian, params: {scale: 1.0, sigma: 0.5}}
  equations:
    u[x]: apply_along(x=xi, K[x=xi] * u[x=xi]) - decay * u[x]
```

## Public API

```python
from op_system import (
    compile_spec,  # validate + normalize + compile
    compile_rhs,  # compile a pre-normalized NormalizedRhs
    normalize_rhs,  # validate + normalize only
    normalize_expr_rhs,
    normalize_transitions_rhs,
    CompiledRhs,
    NormalizedRhs,
    ExprRhs,
    TransitionsRhs,
    BodyEvalFn,
    EvalFn,
    PytreeEvalFn,
    StateDict,
    OperatorDescriptor,
    BlockAxisInfo,
)
```

`NormalizedRhs` is a discriminated union of `ExprRhs | TransitionsRhs`; use
`isinstance` to dispatch.

## Expression guardrails

Expressions are parsed with `ast` and restricted to:

- Arithmetic, comparisons, ternary, boolean ops, names and constants.
- A NumPy allowlist under the `np.` root: `abs`, `exp`, `expm1`, `log`,
  `log1p`, `log2`, `log10`, `sqrt`, `maximum`, `minimum`, `clip`, `where`,
  `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `hypot`, `arctan2`.
- Helpers: `sum_state()`, `sum_prefix(prefix)`, `apply_along(...)`,
  `sum_over(...)`.

`convolve_history(...)` is available via the history-provider runtime hook
(`CompiledRhs.history_eval_fn` and `OpSystemSystem`'s
`options["history_stepper_fn"]`). `history(...)` and `delay(...)` remain
reserved for issue #173 and still raise a targeted unsupported-feature error
with `history_requirements=...` payloads.

For adaptive ring-buffer engines, use `CompiledRhs.body_eval_fn` (or
`OpSystemSystem`'s `options["body_eval_fn"]`) to evaluate each history
signal body exactly once at a known outer-step boundary. This complements
`history_eval_fn`, which is still responsible for in-RHS history queries.

Each history requirement record currently includes: `scope`, `kind`,
`signal_expr`, `options`, `required_options`, `missing_required_options`, and
`unknown_options`.

## Runnable convolve_history example

```python
import numpy as np

from op_system import compile_spec

spec = {
    "kind": "expr",
    "axes": [{"name": "loc", "coords": ["a", "b"]}],
    "state": ["x[loc]"],
    "equations": {"x[loc]": "convolve_history(inflow[loc], kernel=gamma, window=14)"},
}
compiled = compile_spec(spec)

# history_eval_fn is available for axis-indexed convolve_history specs.
assert compiled.history_eval_fn is not None
print(compiled.history_requirements)


class ZeroHistoryProvider:
    def query(self, signal_id: int, body: object, **options: object) -> object:
        # Runtime contract from lowering: __hist_query(signal_id, body, **options)
        return np.zeros_like(body)


state = {"x": np.array([1.0, 2.0], dtype=np.float64)}
out = compiled.history_eval_fn(
    0.0,
    state,
    history_provider=ZeroHistoryProvider(),
    inflow=np.array([0.2, 0.4], dtype=np.float64),
)
print(out["x"])  # [0. 0.]
```

Anything else — non-`np` attribute access, imports, lambdas, comprehensions,
other AST nodes — raises `ValueError` / `TypeError` /
`UnsupportedFeatureError` at normalize time.

## Development

```bash
just ci      # ruff + pytest + mypy (core + flepimop2-op_system mirror) + docs
just test    # pytest only
just ruff
just mypy
just docs    # mkdocs build
```

See [docs/development/](docs/development/) for the IR architecture, block
axis plan, and code-style guide.

## Repository layout

| Path | Purpose |
|---|---|
| `src/op_system/` | Library source (specs, IR, normalize, vectorize, compile). |
| `flepimop2-op_system/` | Thin adapter package exposing `op_system` to flepimop2. |
| `tests/op_system/` | Pytest suite (~430 tests). |
| `docs/` | mkdocs sources; built site published to GitHub Pages. |
| `scripts/` | Release validation and API-reference generation helpers. |
