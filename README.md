# op_system

**Domain-agnostic RHS specification and compilation utilities for scientific simulation systems.**

`op_system` provides a lightweight, backend-neutral way to:

1. Define system dynamics in a YAML/JSON-friendly format
2. Normalize those specifications into a structured representation
3. Compile them into fast callable right-hand-side (RHS) functions usable by numerical engines

It is designed to integrate cleanly with `op_engine` and external orchestration systems (e.g. flepimop2) while remaining standalone and dependency-minimal.

---

## Core Concepts

### 1. RHS Specification

Users define system dynamics using a dictionary (or YAML equivalent). Specs can be provided inline or loaded from `spec_file` (YAML/JSON) when using the flepimop2 adapter.

Spec sources:
- Inline dicts (shown below)
- External files via `spec_file` (YAML/JSON)

Two RHS styles are supported in v1:

---

### Expression Style (`kind: expr`)

Directly specify derivatives:

```python
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
```

---

### Transition Style (`kind: transitions`)

Diagram-oriented hazard formulation:

```python
spec = {
    "kind": "transitions",
    "state": ["S", "I", "R"],
    "aliases": {"N": "S + I + R"},
    "transitions": [
        {"from": "S", "to": "I", "rate": "beta * I / N"},
        {"from": "I", "to": "R", "rate": "gamma"},
    ],
}
```

This is internally converted to conservation-law equations:

```
dS/dt -= (beta * I / N) * S
dI/dt += (beta * I / N) * S - gamma * I
dR/dt += gamma * I
```

---

## Axes & Mixing (preserved metadata)

- Axes: categorical or continuous. Continuous axes can be specified via explicit `coords` or via `domain`+`size`+`spacing` (linear/log/geom). Resolved axis names, coords, and sizes are placed in `meta["axes"]`.
- Mixing kernels: optional `mixing` blocks support `form` values `erfc`, `gaussian`, `exponential`, `gamma`, `power_law`, and `custom_value`, with validated parameters. Normalized kernels are placed in `meta["mixing"]` for adapters to consume (e.g., flepimop2 builds `mixing_kernels`).
- Operators: `operators` metadata is normalized and preserved in `meta["operators"]`; wiring into solvers is future-facing (not yet consumed by op_engine).

---

## Basic Usage

### Option A — One-step API (recommended)

```python
from op_system import compile_spec

compiled = compile_spec(spec)

dydt = compiled.eval_fn(
    t=0.0,
    y=[999.0, 1.0, 0.0],
    beta=0.3,
    gamma=0.1,
)
```

---

### Option B — Two-step API (advanced control)

```python
from op_system import normalize_rhs, compile_rhs

rhs = normalize_rhs(spec)
compiled = compile_rhs(rhs)
dydt = compiled.eval_fn(0.0, [999, 1, 0], beta=0.3, gamma=0.1)
```

---

### Option C — Load from file (YAML/JSON)

```yaml
# sir.yaml
kind: expr
state: [S, I, R]
aliases:
  N: "S + I + R"
# op_system

**Domain-agnostic RHS specification and compilation utilities for scientific simulation systems.**

`op_system` lets you:

1. Define system dynamics in YAML/JSON-friendly form (inline or from file).
2. Normalize specs into a structured representation.
3. Compile fast RHS callables for downstream engines (e.g., op_engine, scipy). 

It integrates cleanly with `op_engine` and orchestration layers (e.g., flepimop2) while remaining standalone and dependency-minimal.

---

## Core Concepts

- Specs: `kind: expr` or `kind: transitions`.
- Sources: inline dicts or `spec_file` (YAML/JSON) when using the flepimop2 adapter.
- Metadata: axes, mixing, operators are normalized and preserved in `CompiledRhs.meta`.

### Expression Style (`kind: expr`)

```python
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
```

### Transition Style (`kind: transitions`)

```python
spec = {
    "kind": "transitions",
    "state": ["S", "I", "R"],
    "aliases": {"N": "S + I + R"},
    "transitions": [
        {"from": "S", "to": "I", "rate": "beta * I / N"},
        {"from": "I", "to": "R", "rate": "gamma"},
    ],
}
```

Conservation form:

```
dS/dt -= (beta * I / N) * S
dI/dt += (beta * I / N) * S - gamma * I
dR/dt += gamma * I
```

---

## Axes & Mixing (preserved metadata)

- **Axes:** categorical or continuous. Continuous axes via explicit `coords` or `domain` + `size` + `spacing` (linear/log/geom). Resolved names/coords/sizes are stored in `meta["axes"]`.
- **Mixing kernels:** `mixing` entries with `form` in `{erfc, gaussian, exponential, gamma, power_law, custom_value}` plus validated params. Normalized kernels land in `meta["mixing"]`; the flepimop2 adapter builds `mixing_kernels` from these.
- **Operators:** `operators` metadata is preserved in `meta["operators"]`; wiring into op_engine IMEX is forward-facing (not yet consumed).

### Usage with Mixing (inline)

```python
from op_system import compile_spec

spec = {
    "kind": "expr",
    "state": ["S", "I"],
    "equations": {
        "S": "-beta * S * I / N",
        "I": "beta * S * I / N - gamma * I",
    },
    "aliases": {"N": "S + I"},
    "axes": [
        {"name": "space", "type": "continuous", "domain": {"lb": 0, "ub": 10}, "size": 5},
    ],
    "mixing": [
        {"name": "K", "axes": ["space", "space"], "form": "gaussian", "params": {"sigma": 1.5, "scale": 1.0}},
    ],
}

compiled = compile_spec(spec)

# metadata available to adapters
axes_meta = compiled.meta["axes"]
mixing_meta = compiled.meta["mixing"]
```

### Usage from file (YAML)

```yaml
# sir.yaml
kind: expr
state: [S, I, R]
aliases:
  N: "S + I + R"
equations:
  S: "-beta * S * I / N"
  I: "beta * S * I / N - gamma * I"
  R: "gamma * I"
mixing:
  - name: K
    axes: [space, space]
    form: gaussian
    params: {sigma: 1.0, scale: 1.0}
axes:
  - name: space
    type: continuous
    domain: {lb: 0, ub: 10}
    size: 5
```

```python
from op_system import compile_spec

compiled = compile_spec(None, spec_file="sir.yaml")
dydt = compiled.eval_fn(0.0, [999, 1, 0], beta=0.3, gamma=0.1)
```

---

## Basic API

- `compile_spec(spec | None, spec_file: str | None = None) -> CompiledRhs`
- `normalize_rhs(spec) -> NormalizedRhs`
- `compile_rhs(rhs: NormalizedRhs) -> CompiledRhs`

`CompiledRhs` fields:
- `state_names`, `param_names`, `eval_fn`
- `meta` with `axes`, `mixing`, `operators`, plus passthrough reserved blocks (sources, couplings, constraints)

---

## Safety & Validation

- Expressions parsed with `ast`; allowed: arithmetic, names, selected NumPy math (`np.exp`, `np.log`, etc.).
- Validates shapes, required fields, mixing forms/params, axis references.
- Errors raise descriptive `ValueError` / `TypeError` (no custom wrapper).

---

## Design & Integration

- Domain-agnostic; no dependency on op_engine or flepimop2.
- Backends: works with op_engine, scipy.integrate, custom solvers; GPU-friendly via wrapper.
- Adapters: flepimop2 system adapter consumes inline or `spec_file`, builds `mixing_kernels` from `meta["mixing"]`, exposes them to downstream engines; operator metadata preserved but not yet wired into op_engine IMEX.

---

## Status

Version: `0.1.0`

Current scope: ODE RHS, expr and transitions styles, axes/mixing metadata preservation.

Planned:
- Wire operator metadata into op_engine IMEX paths
- PDE operators / operator splitting
- Multiphysics coupling and constraints/couplings blocks
- Explicit compatibility/guardrail checks with op_engine adapters

  S: "-beta * S * I / N"
  I: "beta * S * I / N - gamma * I"
  R: "gamma * I"
```

```python
from op_system import compile_spec

compiled = compile_spec(None, spec_file="sir.yaml")
dydt = compiled.eval_fn(0.0, [999, 1, 0], beta=0.3, gamma=0.1)
```

---

## Output Object

`compile_rhs` returns a `CompiledRhs` container:

```python
CompiledRhs(
    state_names = ("S", "I", "R"),
    param_names = ("beta", "gamma"),
    eval_fn = callable,
    meta = {"axes": [...], "mixing": [...], "operators": [...]}
)
```

This matches the function signature expected by most ODE solvers:

```
rhs(t, y) -> dydt
```

`CompiledRhs.meta` preserves optional blocks (axes, mixing, operators, sources, couplings, constraints) for adapters to consume.

---

## Safety & Validation

### Expression Security

Expressions are parsed with Python `ast` and restricted to:

- Arithmetic operations
- Named variables
- Selected NumPy math functions (`np.exp`, `np.log`, etc)

Disallowed operations (imports, attribute access, function injection) are rejected.

Validation errors raise descriptive built-in exceptions (`ValueError`, `TypeError`); no custom error wrapper is shipped.

---

## Design Principles

### Domain Agnostic

`op_system`:

- Does NOT import flepimop2
- Does NOT depend on op_engine
- Does NOT impose solver semantics

It only defines RHS structure and compilation.

---

### Backend Friendly

Compiled RHS functions are compatible with:

- op_engine
- scipy.integrate
- custom solvers
- GPU backends (via wrapping)

---

### Forward Compatible

Reserved fields are preserved during normalization and exposed via `CompiledRhs.meta` so future multiphysics extensions and adapters can read axes, mixing kernels, and operator metadata without breaking the API.

### Adapters

- The flepimop2 system adapter can consume inline specs or `spec_file`, builds `mixing_kernels` from `meta["mixing"]`, and exposes them for downstream engines (e.g., op_engine adapter). Operator metadata is preserved in `meta` but not yet wired into op_engine IMEX.

---

## Public API Summary

```python
compile_spec(spec)         # one-step entrypoint
normalize_rhs(spec)        # spec → NormalizedRhs
compile_rhs(rhs)           # NormalizedRhs → CompiledRhs
```

Data types:

```python
NormalizedRhs
CompiledRhs
CompiledRhs.meta  # axes, mixing, operators, and passthrough reserved blocks
```

---

## Status

Version: `0.1.0`

Current scope:

- ODE RHS
- Algebraic expressions
- Compartment transitions

Planned:

- Wire operator metadata into op_engine IMEX paths
- PDE operators / operator splitting
- Multiphysics coupling and constraints/couplings blocks
- Explicit compatibility/guardrail checks with op_engine adapters
