# Array namespaces, engines, and providers

Use **array namespace** for the numerical array API resolved from input arrays
by `array_api_compat.array_namespace`. “Array backend” is an equivalent term
when discussing NumPy, JAX, or Torch. Use the qualified phrase when an engine
or provider is also in the discussion.

| Term | Meaning in this repository |
| --- | --- |
| Array namespace / array backend | Array operations, dtype, and device selected by runtime state arrays |
| Numerical engine | A downstream integrator that advances the model in time, such as `op_engine` |
| Provider / adapter | Integration package exposing the core compiler through another framework, such as `flepimop2-op_system` |
| Compiler | `op_system`, which validates a specification and constructs reusable RHS evaluators |

Compile a specification with `compile_spec(spec)`, then provide NumPy arrays,
JAX arrays/tracers, or Torch tensors when evaluating it. The specification and
provider configuration do not select an array namespace. Parameters should use
the qualified state's namespace and floating precision; see the
[qualification matrix](backend-qualification.md) for tested combinations.
A JAX input selects JAX array operations; the caller still chooses whether to
apply `jax.jit`, and the numerical engine chooses its integration method.

For flepimop2, `system.module: op_system` selects the **system provider**, while
an engine module selects the **numerical engine**. Neither setting replaces the
array namespace selected from the runtime state. The provider passes state
arrays through without namespace coercion. Generated mixing matrices are
constructed on the host during provider setup; their construction is a separate
boundary from the runtime state namespace.

Existing `compile_spec(backend=..., xp=...)` and `compile_rhs(xp=...)` keyword
arguments remain accepted, deprecated, and ignored. Their names are retained
for compatibility, with the existing deprecation warnings. Remove these
arguments and supply arrays of the desired namespace. No replacement
`array_backend` configuration field is introduced: adding such a field would
suggest compile-time selection that the current API does not perform.
The provider has no namespace-selection `backend` setting; remove a legacy
setting if present and configure runtime arrays through the consuming engine.
