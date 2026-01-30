# op_system

**Domain-agnostic RHS specification and compilation utilities for scientific simulation systems.**

`op_system` provides a lightweight, backend-neutral way to:

1. Define system dynamics in a YAML/JSON-friendly format  
2. Normalize those specifications into a structured representation  
3. Compile them into fast callable right-hand-side (RHS) functions usable by numerical engines  

It is designed to integrate cleanly with `op_engine` and external orchestration systems (eg flepimop2) while remaining standalone and dependency-minimal.

---

## Core Concepts

### 1. RHS Specification

Users define system dynamics using a dictionary (or YAML equivalent).

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

## Output Object

`compile_rhs` returns a `CompiledRhs` container:

```python
CompiledRhs(
    state_names = ("S", "I", "R"),
    param_names = ("beta", "gamma"),
    eval_fn = callable
)
```

This matches the function signature expected by most ODE solvers:

```
rhs(t, y) -> dydt
```

---

## Safety & Validation

### Expression Security

Expressions are parsed with Python `ast` and restricted to:

- Arithmetic operations
- Named variables
- Selected NumPy math functions (`np.exp`, `np.log`, etc)

Disallowed operations (imports, attribute access, function injection) are rejected.

---

### Error Handling

All user-facing errors:

- Raise built-in Python exceptions (`ValueError`, `TypeError`, etc)
- Chain an `OpSystemError` as the cause with a machine-readable `ErrorCode`

Example:

```python
try:
    compiled.eval_fn(0.0, [1, 2])
except ValueError as exc:
    code = exc.__cause__.code
    print(code)
```

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

Reserved fields are preserved during normalization:

```yaml
sources:
operators:
couplings:
constraints:
```

This allows future multiphysics extensions without breaking the API.

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
ErrorCode
OpSystemError
```

---

## Status

Version: `0.1.0`

Current scope:

- ODE RHS
- Algebraic expressions
- Compartment transitions

Planned:

- Operator splitting
- PDE operators
- Multiphysics coupling
- Engine adapters
