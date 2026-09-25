# Changelog

All notable changes to `op_system` and `flepimop2-op_system` are documented
in this file. The two packages are released together under one shared
version; format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Operators can declare the axes of array parameters they consume in
  `kernel.param_axes` (for example an `[imm, imm]` generator or a `[time]`
  routing series). `flepimop2-op_system` requests those names with the
  declared axes; undeclared names stay scalar requests (#204).
- `axis_kernel` operators move mass along one axis with a matrix-valued
  parameter instead of one coordinate-pinned transition per coordinate pair,
  whose compile cost grows with states times transitions. `kernel.form` is
  `generator` (rows sum to zero, scaled by `velocity`) or `stochastic`
  (row-stochastic redistribution of a flux, optionally tied to a `transfer`
  between two coordinates of another axis). Specs are validated at
  normalization, and `axis_kernel_generator_rhs`,
  `axis_kernel_redistribute`, and `validate_axis_kernel_matrix` provide
  shared backend-agnostic reference semantics for engines (#206).
- `op_system.validate_spec` returns a `ValidationReport` instead of raising:
  per-stage status (normalize, compile, vectorize), errors, compile-cost
  drivers (expanded states, coordinate-pinned transitions, operators),
  consumed parameters with their axes (including operator
  `kernel.param_axes`), and, for templates whose cells differ, the distinct
  expression shapes with an example cell. `python -m op_system.validate`
  applies it to bare specs or flepimop2 configurations (#205).

## [0.2.0] - 2026-08-17

### Added

- **Block-axis / PyTree state interface**: `analyze_block_axes` and
  `BlockAxisInfo` for detecting block-structured axes, `block_axes`
  forwarded to connector options, block-stripped RHS compilation, and a
  shape-polymorphic `block_pytree_eval_fn` for `vmap`-friendly
  block-structured state.
- **`OperatorDescriptor` / PyTree state interface**: `kind`/`bc` metadata,
  `factorize_axes`, `pytree_eval_fn`, and `template_shapes` for
  PyTree-native compilation.
- **History and delay operators**: `convolve_history(...)`, wired through
  `CompiledRhs.history_eval_fn` and a provider `history_stepper_fn` hook;
  `CompiledRhs.body_eval_fn` for evaluating history signal bodies once per
  outer-step boundary on adaptive/ring-buffer engines. `history(...)` and
  `delay(...)` remain reserved and still raise a targeted
  unsupported-feature error with `history_requirements=...` payloads.
- Block-structured variants of the history hooks —
  `CompiledRhs.block_history_eval_fn` / `block_body_eval_fn`, exposed by
  the provider as `block_history_stepper_fn` / `block_body_eval_fn` — for
  history/delay signals over block-axis (`vmap`-friendly) state.
- **`null` state support in `transitions` specs**, allowing transitions
  that originate from or terminate outside the tracked state.
- Shaped and time-varying parameters, including parameters that use the
  same axis twice (`apply_along` same-axis-twice support).
- Single unified `apply_along(...)` primitive, replacing `sum_over(...)`
  and `integrate_over(...)`.
- Enhanced schema validation for operator specs.
- Typed expression IR (`ExpressionString`, axis resolution, template/alias
  expansion, common-subexpression elimination) underlying the compiler —
  replaces the previous string-surgery based expansion pipeline.

### Changed

- **JAX-native, namespace-polymorphic evaluation**: compiled `eval_fn`
  now infers its array namespace from the input `y` at call time via
  `y.__array_namespace__()` instead of a compile-time backend selection.
  A single compiled callable now works with NumPy, JAX (concrete and
  traced), and other Array-API backends, and is trace-pure under
  `jax.make_jaxpr`, `jax.jit`, and `jax.vmap`.
- Large internal performance improvements to specification normalization
  and vectorization (e.g. `normalize_transitions_rhs` on a representative
  continuum spec dropped from ~25s to ~2.45s), plus a fix for an alias
  expansion out-of-memory issue.

### Deprecated

- `compile_spec(xp=..., backend=...)` and `compile_rhs(rhs, xp=...)`: the
  `xp`/`backend` kwargs are now ignored and emit a `DeprecationWarning`.
  Pass JAX arrays for a JAX-native call, or NumPy arrays for a NumPy call —
  the backend is inferred from `y` at call time. These kwargs will be
  removed in a future release.

### Fixed

- `OperatorDescriptor` now preserves normalized operator names, expanded
  `apply_to` state selections, jump directions, and symbolic or numeric
  velocity/rate coefficients instead of silently dropping them (#210).
- Several axis-handling edge cases: continuous-axis coordinate lookups,
  shaped-parameter subscript vectorization, bare axis-label binding
  variables in `apply_along` bodies, and reduced-target axis preservation
  through IR lowering.
- Excluded axis names, template bases, and builtins from parameter-name
  inference in `normalize`.

## [0.1.2] and earlier

Released before this changelog was introduced. See the
[GitHub releases](https://github.com/ACCIDDA/op_system/releases) for the
auto-generated PR history of those versions.
