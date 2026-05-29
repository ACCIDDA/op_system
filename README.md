# op_system

Domain-agnostic specification and compilation of right-hand sides (RHS) for
ODE, PDE, and multi-physics / multi-scale compartmental systems.  `op_system`
takes a YAML/JSON-friendly spec, validates and normalizes it, then compiles
it into a fast, **array-API-polymorphic** callable that runs identically on
NumPy, JAX (concrete and traced), or any other Array-API backend — without
recompiling.

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
- **Backend polymorphism at call time** — the compiled `eval_fn` reads
  `y.__array_namespace__()` on every call, so the same compiled artifact
  serves NumPy hosts, JAX `jit`/`vmap`/`grad`, and traced inference loops.
- **First-class PyTree interface** (`pytree_eval_fn`) for engines that want
  to keep state as a dict of shaped arrays rather than a flat vector.
- **Block-axis vmap support** (`block_pytree_eval_fn`) for hierarchical
  models — declare a `factorize_axis` and the engine can vmap a stripped
  per-block RHS over the block axis instead of evaluating a monolithic
  flat state.
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
pip install "op-system[data]"           # pandas + pyarrow helpers
```

## Quick start

```python
from op_system import compile_spec

spec = {
    "kind": "expr",
    "state": ["S", "I", "R"],
    "aliases": {"N": "S + I + R"},
    "equations": {
        "S": "-beta * S * I / N",
        "I":  "beta * S * I / N - gamma * I",
        "R":  "gamma * I",
    },
}

compiled = compile_spec(spec)
dydt = compiled.eval_fn(0.0, [999.0, 1.0, 0.0], beta=0.3, gamma=0.1)
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
| `operators` | Tuple of `OperatorDescriptor` (e.g. advection terms). |

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
    compile_spec,            # validate + normalize + compile
    compile_rhs,             # compile a pre-normalized NormalizedRhs
    normalize_rhs,           # validate + normalize only
    normalize_expr_rhs,
    normalize_transitions_rhs,
    CompiledRhs,
    NormalizedRhs, ExprRhs, TransitionsRhs,
    EvalFn, PytreeEvalFn, StateDict,
    OperatorDescriptor, BlockAxisInfo,
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

Planned history helpers (`history(...)`, `delay(...)`, `convolve_history(...)`)
are reserved for issue #173 and currently raise a targeted "not yet
implemented" validation error.

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
