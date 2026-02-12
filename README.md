# op_system

Domain-agnostic RHS specification and compilation for ODE-style models with templates, helpers, and preserved metadata.

## Statement of Need

Modelers often mix compartment-style hazards, templated populations, and metadata (axes, mixing, operators) that must be validated and preserved for downstream solvers. `op_system` provides:
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
        S: "-beta * S * I / sum_state()"
        I: "beta * S * I / sum_state() - gamma * I"
        R: "gamma * I"

  sir_linear_chain:
    module: op_system
    spec:
      kind: expr
      state: [S, I1, I2, I3, R]
      chain:
        - {name: I, length: 3, forward: gamma, to: R}
      equations:
        S: "-beta * S * sum_prefix('I') / sum_state()"
```

### Two-population SIR (templates + sum_over)

```yaml
system:
  - module: op_system
    spec:
      kind: expr
      axes:
        - {name: pop, coords: [p1, p2]}
      state: [S[pop], I[pop], R[pop]]
      aliases:
        N: "sum_state()"
      equations:
        S[pop]: "-beta * S[pop] * sum_over(pop=j, c_pop_j * I[pop=j] / N)"
        I[pop]: "beta * S[pop] * sum_over(pop=j, c_pop_j * I[pop=j] / N) - gamma * I[pop]"
        R[pop]: "gamma * I[pop]"
```

Parameter names for contacts expand as you choose (e.g., `c_pop_j` → `c_p1_p1`, `c_p1_p2`, ... when binding params).

### Continuous axis + mixing/operator meta + state_axes

```yaml
system:
  - module: op_system
    spec:
      kind: expr
      axes:
        - {name: x, type: continuous, domain: {lb: 0.0, ub: 10.0}, size: 5, spacing: linear}
      state: [u]
      state_axes:
        u: [x]
      mixing:
        - {name: K, axes: [x], form: gaussian, params: {scale: 1.0, sigma: 0.5}}
      operators:
        - {name: diff, axis: x, kind: laplacian}
      equations:
        u: "0.0"
```

### Transitions (hazard/flow style)

```yaml
system:
  - module: op_system
    spec:
      kind: transitions
      state: [S, I, R]
      aliases:
        N: "S + I + R"
      transitions:
        - {from: S, to: I, rate: "beta * I / N"}
        - {from: I, to: R, rate: "gamma"}
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
- Templates: `State[axis,...]` expand over categorical axes; equations may be written once per template.
- `sum_over(axis=var, expr)`: unrolls over categorical coords; continuous axes are rejected.
- Chain helper: `chain` block auto-fills staged compartments (expr or transitions kinds).
- Reducers in expressions: `sum_state()`, `sum_prefix(prefix)`.
- Axes: categorical or continuous; continuous can be generated via `domain` + `size` + `spacing` (linear/log/geom).
- Metadata passthrough: axes, state_axes, mixing, operators, reserved blocks (`sources`, `couplings`, `constraints`) in `NormalizedRhs.meta`.

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
- `NormalizedRhs.meta`: access normalized metadata (axes, state_axes, mixing, operators, reserved blocks).

## License & support

- License: GPL-3.0 (see LICENSE).
- Issues/feedback: use the GitHub issue tracker.
