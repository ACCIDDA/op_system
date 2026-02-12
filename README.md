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

---

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
        I: -beta * S * sum_prefix('I') / sum_state() - gamma * I3
        R: gamma * I3
```

### Two-population SIR (templates + sum_over)

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
      equations:
        S[pop]: -beta * S[pop] * sum_over(pop=j, c_pop_j * I[pop=j] / N)
        I[pop]: beta * S[pop] * sum_over(pop=j, c_pop_j * I[pop=j] / N) - gamma * I[pop]
        R[pop]: gamma * I[pop]
```

Parameter names for contacts expand as you choose (e.g., `c_pop_j` → `c_p1_p1`, `c_p1_p2`, ... when binding params).

### Templated aliases/parameters (per-axis)

Write aliases and parameters once with inline placeholders; they expand to concrete names for each categorical coordinate.

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
Expands to `beta__age_child`, `beta__age_adult`, `offset__age_child`, etc., so you only bind concrete parameters when running.

### Continuous axis + kernel/operator meta + state_axes

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
      state: [u]
      state_axes:
        u: [x]
      kernels:
        - name: K
          axes: [x]
          form: gaussian
          params:
            scale: 1.0
            sigma: 0.5
      operators:
        - name: diff
          axis: x
          kind: diffusion  # physical operator (solver chooses discretization)
          bc: dirichlet
      equations:
        u: "0.0"
```

Integrate along continuous axes with `integrate_over(axis=var, expr)`, which uses trapezoidal weights derived from the axis `coords` (non-uniform spacing respected).
Operators describe the physical intent (`kind` such as diffusion/advection and `bc` such as dirichlet/neumann/periodic); numerical discretization is left to backends like `op_engine`.
Normalized metadata preserves kernels in `meta["kernels"]` and operators in `meta["operators"]` alongside axes and state axes.

### Transitions (hazard/flow style)

```yaml
system:
  - module: op_system
    spec:
      kind: transitions
      state: [S, I, R]
      aliases:
        N: S + I + R
      transitions:
        - name: infect
          from: S
          to: I
          rate: beta * I / N
        - from: I
          to: R
          rate: gamma
```

### Complex structure 
```yaml
system:
  usa_flu:
    module: op_system
    spec:
      kind: expr
      axes:
        - name: vacc
          coords: [unvaccinated, dose1, waned]
        - name: age
          coords: [age0to4, age5to17, age18to49, age50to64, age65to100]

      state:
        - S[vacc,age]
        - E[vacc,age]
        - I1[vacc,age]
        - I2[vacc,age]
        - I3[vacc,age]
        - R[vacc,age]

      aliases:
        # Per-age vaccination intensity (timeseries params you bind)
        nu1[age]: nu1_path[age]

        # Per-age VE (set ve1[...] and veW[...] params)
        theta1[age]: 1 - ve1[age]
        thetaW[age]: 1 - veW[age]

        # Force of infection with age-/vacc-specific VE
        lambda[vacc,age]: r0 * gamma * (
          sum_over(age=j, I1[vacc=j, age=j] + I2[vacc=j, age=j] + I3[vacc=j, age=j])
        ) * (
          theta1[age] if vacc == 'dose1'
          else thetaW[age] if vacc == 'waned'
          else 1
        )

      equations:
        # Infection and progression
        S[vacc,age]: -lambda[vacc,age] * S[vacc,age]
        E[vacc,age]: lambda[vacc,age] * S[vacc,age] - sigma_AllFlu * E[vacc,age]
        I1[vacc,age]: sigma_AllFlu * E[vacc,age] - 3*gamma * I1[vacc,age]
        I2[vacc,age]: 3*gamma * I1[vacc,age] - 3*gamma * I2[vacc,age]
        I3[vacc,age]: 3*gamma * I2[vacc,age] - 3*gamma * I3[vacc,age]
        R[vacc,age]: 3*gamma * I3[vacc,age]

        # Vaccination flow: unvaccinated -> dose1
        S[unvaccinated,age]: -nu1[age] * S[unvaccinated,age]
        E[unvaccinated,age]: -nu1[age] * E[unvaccinated,age]
        R[unvaccinated,age]: -nu1[age] * R[unvaccinated,age]
        S[dose1,age]: nu1[age] * S[unvaccinated,age] - epsilon * S[dose1,age]
        E[dose1,age]: nu1[age] * E[unvaccinated,age] - epsilon * E[dose1,age]
        R[dose1,age]: nu1[age] * R[unvaccinated,age] - epsilon * R[dose1,age]

        # Waning: dose1 -> waned
        S[waned,age]: epsilon * S[dose1,age]
        E[waned,age]: epsilon * E[dose1,age]
        R[waned,age]: epsilon * R[dose1,age]

        # External seeding for unvaccinated age18to49 (match original intent)
        E[unvaccinated,age18to49]: lambda_ext  # add to existing E flow
```
### Transitions version: 
```yaml
system:
  usa_flu:
    module: op_system
    spec:
      kind: transitions
      axes:
        - name: vacc
          coords: [unvaccinated, dose1, waned]
        - name: age
          coords: [age0to4, age5to17, age18to49, age50to64, age65to100]

      state: [S[vacc,age], E[vacc,age], I1[vacc,age], I2[vacc,age], I3[vacc,age], R[vacc,age]]

      aliases:
        nu1[age]: nu1_path[age]
        theta1[age]: 1 - ve1[age]
        thetaW[age]: 1 - veW[age]
        lambda[vacc,age]: r0 * gamma * (
          sum_over(age=j, I1[vacc=j, age=j] + I2[vacc=j, age=j] + I3[vacc=j, age=j])
        ) * (
          theta1[age] if vacc == 'dose1'
          else thetaW[age] if vacc == 'waned'
          else 1
        )

      transitions:
        # Infection S -> E (per vacc, age)
        - from: S[vacc,age]
          to: E[vacc,age]
          rate: lambda[vacc,age]

        # Progression E -> I1 -> I2 -> I3 -> R
        - from: E[vacc,age]
          to: I1[vacc,age]
          rate: sigma_AllFlu
        - from: I1[vacc,age]
          to: I2[vacc,age]
          rate: 3*gamma
        - from: I2[vacc,age]
          to: I3[vacc,age]
          rate: 3*gamma
        - from: I3[vacc,age]
          to: R[vacc,age]
          rate: 3*gamma

        # Vaccination: unvaccinated -> dose1 (apply to S/E/R)
        - from: S[unvaccinated,age]
          to: S[dose1,age]
          rate: nu1[age]
        - from: E[unvaccinated,age]
          to: E[dose1,age]
          rate: nu1[age]
        - from: R[unvaccinated,age]
          to: R[dose1,age]
          rate: nu1[age]

        # Waning: dose1 -> waned (apply to S/E/R)
        - from: S[dose1,age]
          to: S[waned,age]
          rate: epsilon
        - from: E[dose1,age]
          to: E[waned,age]
          rate: epsilon
        - from: R[dose1,age]
          to: R[waned,age]
          rate: epsilon

        # External seeding: unvaccinated, age18to49
        - from: S[unvaccinated,age18to49]
          to: E[unvaccinated,age18to49]
          rate: lambda_ext
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
- Chain helper: `chain` block auto-fills staged compartments (expr or transitions kinds).
- Reducers in expressions: `sum_state()`, `sum_prefix(prefix)`.
- Axes: categorical or continuous; continuous can be generated via `domain` + `size` + `spacing` (linear/log/geom).
- Metadata passthrough: axes, state_axes, kernels, operators, reserved blocks (`sources`, `couplings`, `constraints`) in `NormalizedRhs.meta`.

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
