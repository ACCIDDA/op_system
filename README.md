# op_system

Lightweight, domain-agnostic RHS specification and compilation utilities. Write your model once in a YAML/JSON-friendly form; `op_system` validates it, normalizes it, and compiles it into a fast NumPy-friendly RHS callable for solvers.

## Why use it?
- Clean separation of model specification from numerical engines (works with `op_engine`, SciPy, or custom solvers).
- Two spec styles built in: explicit expressions (`kind: expr`) and transition diagrams (`kind: transitions`).
- Conservative AST validation and NumPy-only dependency.
- Forward-compatible IR that preserves reserved blocks for multiphysics (sources/operators/couplings/constraints).

## Installation

```bash
pip install op_system
```

## Quickstart

Expression RHS:

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

rhs = compile_spec(spec)
dydt = rhs.eval_fn(0.0, [999.0, 1.0, 0.0], beta=0.3, gamma=0.1)
```

Transition RHS (hazard/flow style):

```python
from op_system import compile_spec

spec = {
    "kind": "transitions",
    "state": ["S", "I", "R"],
    "aliases": {"N": "S + I + R"},
    "transitions": [
        {"from": "S", "to": "I", "rate": "beta * I / N"},
        {"from": "I", "to": "R", "rate": "gamma"},
    ],
}

rhs = compile_spec(spec)
```

## Public API
- `compile_spec(spec)` — validate, normalize, and compile in one step.
- `normalize_rhs(spec)` — spec → `NormalizedRhs` (backend-facing IR).
- `compile_rhs(rhs)` — `NormalizedRhs` → `CompiledRhs` with `eval_fn(t, y, **params)`.
- Convenience: `normalize_expr_rhs`, `normalize_transitions_rhs`.

Core data:
- `NormalizedRhs`: `kind`, `state_names`, `equations`, `aliases`, `param_names`, `all_symbols`, `meta` (carries reserved fields like `sources`, `operators`, `couplings`, `constraints`, `transitions`).
- `CompiledRhs`: `state_names`, `param_names`, `eval_fn`, plus `bind(params)` for solver-friendly `rhs(t, y)`.

## Safety and validation
- Expressions parsed with `ast` and validated against a small allowlist (arithmetic, comparisons, boolean ops, ternary, selected `np.*` scalar math).
- Evaluation runs with empty builtins; only `np` and provided symbols are available.
- Shapes checked at runtime for `y` and equation counts; descriptive `ValueError`/`TypeError`/`NotImplementedError` messages on invalid input.

## IR and forward compatibility
- Current kinds: `expr`, `transitions`.
- Normalization preserves reserved blocks in `meta` so future multiphysics/PDE/operator specs can be added without breaking existing callers.
- Works out of the box for ODE-style RHS; adapters can add operator blocks for IMEX/PDE backends.

## Development

Clone and install dev tools:

```bash
uv sync --dev
```

Run checks:

```bash
just ci
```

## License

GPL-3.0
