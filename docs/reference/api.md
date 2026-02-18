## API Reference

<p align="center"> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT6ojFFBRart4a2lcwM5_3B2zs_ZzI_lspOhA&s" width="400" alt="mkdocs"> 
</p>

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
