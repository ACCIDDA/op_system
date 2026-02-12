# op_system

**Domain-agnostic RHS specification and compilation utilities for scientific simulation systems.**

`op_system` provides a lightweight, backend-neutral way to:

1. Define system dynamics in a YAML/JSON-friendly format (with axes, templates, helpers)
2. Normalize those specifications into a structured representation (with preserved metadata)
3. Compile them into fast callable right-hand-side (RHS) functions usable by numerical engines

It is designed to integrate cleanly with `op_engine` and orchestration systems (e.g. flepimop2) while remaining standalone and dependency-minimal.

---

## Core Concepts

### 1. RHS Specification

Users define system dynamics using a dictionary. Inline specifications are passed directly to the API functions.

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

### Templated State Names and `sum_over`

- **Templated states**: use `State[axis_a, axis_b]` in `state` and provide categorical axes; they expand into concrete names like `State__axis_a_a0__axis_b_b1`.
- **`sum_over(axis=var, expr)`**: unrolls over categorical axes at normalization time, replacing template references accordingly.
- Continuous axes are rejected in `sum_over`.

Example:

```python
spec = {
  "kind": "expr",
  "axes": [{"name": "pop", "coords": ["p1", "p2"]}],
  "state": ["S[pop]", "I[pop]"],
  "equations": {
    "S[pop]": "-beta * S[pop] * sum_over(pop=j, I[pop=j])",
    "I[pop]": "beta * S[pop] * sum_over(pop=j, I[pop=j]) - gamma * I[pop]",
  },
}
```

### Chain Helper (expr + transitions)

Optional `chain` blocks auto-fill missing equations or transitions for staged compartments:

```python
"chain": [{"name": "I", "length": 3, "forward": "gamma", "to": "R"}]
```

- For `expr`, missing stage equations are generated.
- For `transitions`, missing internal transitions and optional sink transitions are appended.

---

## Axes & Mixing (preserved metadata)

- Axes: categorical or continuous. Continuous axes can be specified via explicit `coords` or via `domain`+`size`+`spacing` (linear/log/geom). Resolved axis names, coords, and sizes are placed in `NormalizedRhs.meta["axes"]`.
- Mixing kernels: optional `mixing` blocks support `form` values `erfc`, `gaussian`, `exponential`, `gamma`, `power_law`, and `custom_value`, with validated parameters. Normalized kernels are placed in `NormalizedRhs.meta["mixing"]` for adapters to consume (e.g., flepimop2 builds `mixing_kernels`).
- Operators: `operators` metadata is normalized and preserved in `NormalizedRhs.meta["operators"]`; wiring into solvers is future-facing (not yet consumed by op_engine).

State-axis mapping: optional `state_axes` maps state names to axes; validation ensures referenced axes exist and are not duplicated.

---

## Basic Usage

### Option A — One-step API 

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

### Option C — Working with YAML specifications

While `compile_spec` doesn't directly load files, you can easily load YAML/JSON files using standard Python libraries:

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
```

```python
import yaml
from op_system import compile_spec

with open("sir.yaml") as f:
    spec = yaml.safe_load(f)

compiled = compile_spec(spec)
dydt = compiled.eval_fn(0.0, [999, 1, 0], beta=0.3, gamma=0.1)
```

---

## Metadata Preservation

The `NormalizedRhs` object (returned by `normalize_rhs`) includes a `meta` field that contains:

- `axes`: Normalized axis definitions with resolved coords and sizes
- `state_axes`: Mapping of state variables to their axis dependencies
- `mixing`: Normalized mixing kernel specifications
- `operators`: Operator metadata for future IMEX integration
- Reserved blocks: `sources`, `couplings`, `constraints` (passthrough for future use)

This metadata is preserved for consumption by adapters and orchestration layers.

### Example: Accessing Metadata

```python
from op_system import normalize_rhs

spec = {
    "kind": "expr",
    "state": ["S", "I"],
    "equations": {"S": "-beta * S * I", "I": "beta * S * I - gamma * I"},
    "axes": [
        {"name": "space", "type": "continuous", "domain": {"lb": 0, "ub": 10}, "size": 5},
    ],
    "mixing": [
        {"name": "K", "axes": ["space", "space"], "form": "gaussian", "params": {"sigma": 1.5, "scale": 1.0}},
    ],
}

rhs = normalize_rhs(spec)

# Access normalized metadata
axes_meta = rhs.meta["axes"]        # List of normalized axis definitions
mixing_meta = rhs.meta["mixing"]    # List of normalized mixing kernels
```

---

## Safety & Validation

### Expression Security

Expressions are parsed with Python `ast` and restricted to:

- Arithmetic operations, comparisons, ternary expressions, bool ops
- Named variables and constants
- Selected NumPy math functions with `np.` root: `abs`, `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `sqrt`, `maximum`, `minimum`, `clip`, `where`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `hypot`, `arctan2`
- Built-in helpers: `sum_state()`, `sum_prefix(prefix)`

Disallowed operations include non-`np` attribute access, non-whitelisted helpers, imports, and other AST node types. Validation errors raise descriptive built-in exceptions (`ValueError`, `TypeError`).

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

Reserved fields are preserved during normalization and exposed via `NormalizedRhs.meta` so future multiphysics extensions and adapters can read axes, mixing kernels, and operator metadata without breaking the API.

### Adapters

- The flepimop2 system adapter can consume inline specs, builds `mixing_kernels` from `NormalizedRhs.meta["mixing"]`, and exposes them for downstream engines (e.g., op_engine adapter). Operator metadata is preserved in meta but not yet wired into op_engine IMEX.

---

## API Reference

### Primary Functions

**`compile_spec(spec: dict) -> CompiledRhs`**

One-step entrypoint that validates, normalizes, and compiles a RHS specification. Recommended for most users.

- **Args:** `spec` - Raw RHS specification mapping (YAML/JSON friendly)
- **Returns:** `CompiledRhs` object with compiled evaluation function

**`normalize_rhs(spec: dict) -> NormalizedRhs`**

Validates and normalizes a RHS specification into a structured representation.

- **Args:** `spec` - Raw RHS specification mapping
- **Returns:** `NormalizedRhs` object with normalized equations and metadata

**`compile_rhs(rhs: NormalizedRhs) -> CompiledRhs`**

Compiles a normalized RHS into an efficient callable evaluation function.

- **Args:** `rhs` - Normalized RHS from `normalize_rhs`
- **Returns:** `CompiledRhs` object with compiled evaluation function

**`normalize_expr_rhs(spec: dict) -> NormalizedRhs`**

Specialized normalization for expression-style (`kind: expr`) specifications.

**`normalize_transitions_rhs(spec: dict) -> NormalizedRhs`**

Specialized normalization for transition-style (`kind: transitions`) specifications.

### Data Types

**`CompiledRhs`**

Container for a compiled RHS evaluation function.

- **Fields:**
  - `state_names: tuple[str, ...]` - State variable names
  - `param_names: tuple[str, ...]` - Parameter names
  - `eval_fn: Callable` - Compiled RHS function with signature `(t, y, **params) -> dydt`

- **Methods:**
  - `bind(params: dict) -> Callable` - Returns a 2-arg RHS function `rhs(t, y) -> dydt` with parameters bound

**`NormalizedRhs`**

Normalized RHS representation suitable for compilation.

- **Fields:**
  - `kind: str` - RHS kind (`"expr"` or `"transitions"`)
  - `state_names: tuple[str, ...]` - State variable names
  - `equations: tuple[str, ...]` - Normalized equation expressions
  - `aliases: Mapping[str, str]` - Alias definitions
  - `param_names: tuple[str, ...]` - Sorted parameter names
  - `all_symbols: frozenset[str]` - All symbols used in equations
  - `meta: Mapping` - Metadata (axes, mixing, operators, reserved blocks)

### Module Constants

**`__version__`** - Current version string (`"0.1.0"`)

**`SUPPORTED_RHS_KINDS`** - Tuple of supported RHS kinds: `("expr", "transitions")`

**`EXPERIMENTAL_FEATURES`** - Frozenset of experimental features (currently empty)

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
