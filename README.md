# op_system

Domain-agnostic RHS specification and compilation for ODE-style models with templates, helpers, and preserved metadata.

## Statement of Need

Modelers often combine compartment-style hazards, templated populations, and metadata such as axes, kernels, and operators that must be validated and preserved for downstream solvers. `op_system` provides:
- YAML/JSON-friendly specs for RHS definitions (expr and transitions styles).
- Normalization with strict validation and metadata preservation.
- Compilation to safe, fast callables usable by engines like `op_engine` or custom integrators.

## Quick start (Python dict)

```python
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
dydt = compiled.eval_fn(0.0, [999.0, 1.0, 0.0], beta=0.3, gamma=0.1)
```
## YAML examples (feature coverage)

### Classic vs linear-chain SIR (reducers + chain helper)

```yaml
system:
  sir_classic:
    module: op_system
    spec:
      kind: expr
      state: [S, I, R]
      equations:
        S: -beta * S * I / sum_state()
        I: beta * S * I / sum_state() - gamma * I
        R: gamma * I

  sir_linear_chain:
    module: op_system
    spec:
      kind: expr
      state: [S, I1, I2, I3, R]
      chain:
        - name: I
          length: 3
          forward: gamma
          to: R
      equations:
        S: -beta * S * sum_prefix('I') / sum_state()
        I: beta * S * sum_prefix('I') / sum_state() - gamma * I3
        R: gamma * I3
```

### Two-population SIR (templates + sum_over + contact matrix params)

```yaml
system:
  - module: op_system
    spec:
      kind: expr
      axes:
        - name: pop
          coords: [p1, p2]
      state: [S[pop], I[pop], R[pop]]
      aliases:
        N: sum_state()
        c[pop,pop2]: c_base * m[pop,pop2]  # params expand to m__pop_p1__pop2_p2, etc.
      equations:
        S[pop]: -beta * S[pop] * sum_over(pop2=j, c[pop,j] * I[pop=j] / N)
        I[pop]:  beta * S[pop] * sum_over(pop2=j, c[pop,j] * I[pop=j] / N) - gamma * I[pop]
        R[pop]:  gamma * I[pop]
```

### Templated aliases/parameters (per-axis)

```yaml
system:
  - module: op_system
    spec:
      kind: expr
      axes:
        - name: age
          coords: [child, adult]
      state: [S[age], I[age]]
      aliases:
        beta[age]: b0 * k[age]
        k[age]: k_base * (1 + offset[age])
      equations:
        S[age]: -beta[age] * S[age]
        I[age]: beta[age] * S[age] - gamma * I[age]
```
Expands to `beta__age_child`, `beta__age_adult`, `offset__age_child`, etc.; you bind only the concrete parameter names.

Use `aliases` for readable reusable symbolic groups (e.g., force of infection terms, axis-specific coefficients), not for introducing extra runtime state variables.

### Multi-axis templates (age, vaccination, strain)

```yaml
system:
  - module: op_system
    spec:
      kind: expr
      axes:
        - name: age
          coords: [child, adult]
        - name: vax
          coords: [u, v]
        - name: strain
          coords: [wt, var]
      state: [S[age,vax], I[age,vax,strain], R[age,vax]]
      aliases:
        lambda[age,vax,strain]: beta[strain] * I[age,vax,strain] / sum_state()
      equations:
        S[age,vax]: -sum_over(strain=s, lambda[age,vax,s] * S[age,vax])
        I[age,vax,strain]: lambda[age,vax,strain] * S[age,vax] - gamma[strain] * I[age,vax,strain]
        R[age,vax]: sum_over(strain=s, gamma[strain] * I[age,vax,s])
```

Each template expands over the listed axes only, so different states can legitimately use different axis sets.

### Continuous axis + integrate_over (with kernel meta)

```yaml
system:
  - module: op_system
    spec:
      kind: expr
      axes:
        - name: x
          type: continuous
          domain:
            lb: 0.0
            ub: 10.0
          size: 5
          spacing: linear
      state: [u[x]]
      state_axes:
        u: [x]
      kernels:
        - name: K
          axes: [x]
          form: gaussian
          params:
            scale: 1.0
            sigma: 0.5
      equations:
        u[x]: integrate_over(x=xi, K[xi] * u[xi]) - decay * u[x]
```

`integrate_over` uses trapezoidal weights derived from `axes[0].coords`; non-uniform spacing is respected. Kernel/operator metadata is preserved in `meta` for downstream solvers.

### Transitions (hazard/flow) with chain helper

```yaml
system:
  - module: op_system
    spec:
      kind: transitions
      state: [S, R]
      aliases:
        N: sum_state()
      chain:
        - name: I
          length: 3
          entry:
            from: S
            rate: beta * sum_prefix('I') / N
          forward: gamma
          exit:
            to: R
      transitions: []
      # chain fills S->I1 plus I1->I2->I3->R
```

### Transitions with heterogeneous chain stage rates

```yaml
system:
  - module: op_system
    spec:
      kind: transitions
      state: [S, R]
      chain:
        - name: I
          length: 3
          entry:
            from: S
            rate: beta
          forward: [gamma12, gamma23]
          exit:
            to: R
            rate: gamma3r
      transitions: []
```

### Biogeochemical network (templated transitions + kernels + mass balance)

```yaml
system:
  - module: op_system
    spec:
      kind: transitions
      axes:
        - name: P_size
          coords: [p0, p1]
        - name: Z_size
          coords: [z0, z1]
      kernels:
        - name: feed
          axes: [P_size, Z_size]
          form: gaussian
          params:
            scale: 1.0
            sigma: 0.5
      state: [N, P[P_size], Z[Z_size], Source]
      aliases:
        season: 1 + sea * np.sin(2*np.pi*t/365)
      transitions:
        - name: N_in
          from: Source
          to: N
          rate: n_t
        - name: uptake[p]
          from: N
          to: P[P_size=p]
          rate: season * mu_max_P[p] * P[P_size=p] / (kn_P[p] + N)
        - name: graze[p,z]
          from: P[P_size=p]
          to: Z[Z_size=z]
          rate: gamma * g_Z[z] * feed[P_size=p,Z_size=z] * Z[Z_size=z]
        - name: P_loss[p]
          from: P[P_size=p]
          to: N
          rate: lam
        - name: Z_loss[z]
          from: Z[Z_size=z]
          to: N
          rate: delta_Z[z]
```


---

## Installation

```bash
pip install op-system
# or locally
uv pip install .
```

Supports Python >= 3.11.

## Testing

```bash
just ci        # ruff + pytest + mypy (core + provider mirror)
# or individually
just pytest
just ruff
just mypy
```

## Features

- RHS kinds: `expr` (explicit equations) and `transitions` (hazard/flow style).
- Transitions support optional `name` metadata preserved in `meta.transitions`; templated transitions expand before hazard assembly with metadata intact.
- Templates: `State[axis,...]` expand over categorical axes; equations may be written once per template.
- Aliases and inline placeholders like `theta[age]` expand over categorical axes using the same assignments, removing per-axis parameter duplication.
- Transitions now accept templated states and rates over categorical axes; templated `from`/`to`/`rate` are expanded before hazard assembly.
- `sum_over(axis=var, expr)`: unrolls over categorical coords; continuous axes are rejected.
- `integrate_over(axis=var, expr)`: trapezoidal integrate along continuous axes using axis-derived deltas (non-uniform spacing supported).
- Chain helper: `chain` block auto-fills equations/transitions for staged compartments (expr or transitions kinds), and staged states (`I1..Ik`) are synthesized automatically when omitted from `state`.
- Reducers in expressions: `sum_state()`, `sum_prefix(prefix)`.
- Axes: categorical or continuous; continuous can be generated via `domain` + `size` + `spacing` (linear/log/geom).
- Metadata passthrough: axes, state_axes, kernels, operators, reserved blocks (`sources`, `couplings`, `constraints`) in `NormalizedRhs.meta`.

## Specification behavior notes

- Alias usage: aliases are expanded expressions evaluated against state, params, and earlier aliases; they are best used to reduce repeated symbolic expressions.
- Multi-axis grouping: forms like `S[age,vax,strain]` are supported when all listed axes are defined in `axes`.
- Axis asymmetry (`expr`): states/equations may use different axis subsets (for example `S[age]`, `I[age,strain]`), and substitutions only apply where placeholders are present.
- Axis asymmetry (`transitions`): placeholder expansion uses all placeholders appearing in `from`, `to`, `rate`, and optional `name`, then renders each field with its own placeholders.
- Chain helper state behavior: `chain` synthesizes staged names from `name` and `length`, and can generate the first infection transition via `entry` so transitions specs do not need a manual `S -> I1` edge.

## Chain schema

```yaml
chain:
  - name: I
    length: 3
    entry: {from: S, rate: beta * force}  # optional
    forward: gamma                         # scalar broadcast
    # or forward: [gamma12, gamma23]       # per internal edge, length-1 values
    exit: {to: R, rate: gamma3r}           # optional; rate defaults to last forward rate
```

- `forward` may be scalar or a list of `length - 1` rates.
- `entry` is optional and, when provided, generates `entry.from -> I1`.
- `exit` is optional and, when provided, generates `I_last -> exit.to`.

## Validation & AST guardrails

Expressions are parsed with `ast` and restricted to:
- Arithmetic, comparisons, ternary, bool ops; names/constants; calls/attributes on an allowlist.
- NumPy calls with `np.` root: `abs`, `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `sqrt`, `maximum`, `minimum`, `clip`, `where`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `hypot`, `arctan2`.
- Helpers: `sum_state()`, `sum_prefix(prefix)`.

Disallowed: non-`np` attribute access, non-whitelisted helpers, imports, or other AST node types. Errors are raised as `ValueError` / `TypeError` / `NotImplementedError`.

## API (brief)

- `compile_spec(spec) -> CompiledRhs`: normalize + compile in one step.
- `normalize_rhs(spec) -> NormalizedRhs`: validation + metadata preservation.
- `compile_rhs(rhs) -> CompiledRhs`: compile a pre-normalized RHS.
- `NormalizedRhs.meta`: access normalized metadata (axes, state_axes, kernels, operators, reserved blocks).

## License & support

- License: GPL-3.0 (see LICENSE).
- Issues/feedback: use the GitHub issue tracker.
