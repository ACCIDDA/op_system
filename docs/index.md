# `op_system`

`op_system` validates YAML/JSON-friendly model specifications and compiles them
into vectorized right-hand-side callables for ODE, PDE, and compartmental
systems. It is domain-agnostic: [`flepimop2`](https://github.com/ACCIDDA/flepimop2)
is an important consumer, not the package's only intended use.

The test suite covers NumPy, concrete and traced JAX arrays, and raw PyTorch
tensors. Other Array API implementations may work through
`array-api-compat`, but should be qualified before production use.

## Installation

```bash
pip install op-system
```

## Main interfaces

- `compile_spec(...)` validates, normalizes, and compiles a loaded mapping.
- `normalize_rhs(...)` normalizes without compilation.
- `validate_spec(...)` returns a structural validation report.
- `compile_rhs(...)` creates a reusable `CompiledRhs` object.
- `CompiledRhs.eval_fn(...)` evaluates a flat state vector.
- `CompiledRhs.pytree_eval_fn(...)` evaluates shaped state mappings.
- `CompiledRhs.block_pytree_eval_fn(...)` supports hierarchical block-axis
  vectorization using the specification's `factorize_axes` declaration.

Start with the [model specification guide](guides/getting-started.md) and its
executable age-structured SEIR example. See the [repository README](https://github.com/ACCIDDA/op_system#readme) for a
worked example and the navigation for the specification and API references.

## Array namespace boundary

The array namespace is selected from input arrays at call time. A numerical
engine selects integration methods, while a provider adapts framework interfaces.
See [array namespace terminology](guides/array-namespaces.md). Keep model
expressions within the documented allowlist, and validate an additional backend
against the package's conformance tests before treating it as supported in a
production workflow.

## Contributing and citation

The [contributor guide](development/contributing.md) covers setup, focused PRs,
required checks, and support. [Citation and archiving](development/citation-and-archiving.md)
explains attribution, release identifiers, and the software-paper archive plan.

See [capabilities and executable examples](guides/capabilities.md) for current interfaces,
optional dependencies, limitations and the checks that run in CI.
