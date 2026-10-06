# Changelog

All notable changes to `op_system` and `flepimop2-op_system` are documented
in this file. The two packages are released together under one shared
version; format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.7.0] - 2026-10-06

### Added

- `reactants: auto` on a named transition, and `catalysts: auto` on `chain`
  and `coord_shift` entries, infer a reaction's molecular reactants from its
  alias-inlined rate. The result is the consumed source at order one plus
  each state factor at its integer power, with `reactants_complete=True`.
  Inference covers rates that are a single product of states (mass action,
  density dependence, first-order flows) (#255).
- For a `reactants: auto` rate that is not a single product of states, such
  as a frequency-dependent force of infection, `CompiledReaction` publishes
  `dependencies` (every state selection the propensity reads, with one pinned
  entry per coordinate of a reduction), `propensity_order` (a structural
  bound on `sum_i |d log a / d log x_i|`), and `dependencies_complete=True`.
  `reactants_complete` stays false on these reactions, so consumers that
  predate the fields refuse adaptive tau-leaping. Subtraction, negation,
  other functions of a state, symbolic powers, and history operators raise
  `InvalidRhsSpecError` naming the construct (#256).

### Changed

- When `reactants` (or `catalysts`) is omitted and the rate reads no state,
  the consumed-source reactant is complete: `reactants_complete=True`.
  Nothing else can then be a reactant, so first-order flows and constant
  aging or chain rates no longer block adaptive tau-leaping. Rates that read
  state keep `reactants_complete=False`. Parameters are assumed not to depend
  on the state (#255).

### Fixed

- A transition rate that names an axis-less alias by bare name (`rate: lam`
  with `aliases: {lam: ...}`) now inlines it into the reaction propensity, as
  bracketed aliases already were. This includes axis-less aliases used inside
  other aliases' bodies. Previously the reaction compiled with no gap and its
  `propensity_fn` raised `NameError` when evaluated. A rate that still names
  an alias after inlining (one on a reference cycle, or a templated alias
  referenced by bare name) is now reported as an `unresolved_alias` reaction
  gap instead of publishing a propensity that cannot be evaluated (#254).

## [0.6.0] - 2026-10-02

### Added

- `coord_shift` accepts an axis-wide form,
  `{axis: age, step: 1, rate: ..., boundary: absorb|stay}`. It shifts every
  coordinate of one axis in a single entry instead of `n - 1` pairwise
  entries. Rates read the source coordinate. Each `apply_to` state lowers
  once through matrix routing, with cell values equal to the pairwise form;
  a 240-bin, four-state chain compiles in tens of milliseconds instead of
  seconds. Named entries publish one templated `CompiledReaction` per state.
  The new `offsets` field gives the shifted axis and its step, and
  off-axis destinations either leave the system (`absorb`) or never fire
  (`stay`). `boundary` has no default (#238).
- Time-indexed parameters support opt-in right-continuous hold interpolation
  with `time_interpolation: previous`, retaining linear interpolation by
  default. Flat, PyTree, block, and reaction evaluators share the policy;
  compiled metadata and provider options expose the immutable time coordinates
  and forcing breakpoints. Single-coordinate time tables remain constant (#240).
- `CompiledRhs.reaction_gaps` and the provider's `reaction_gaps` option list
  every transition that has no compiled reaction artifact. Each `ReactionGap`
  gives its spec origin (including `chain:` stages), name, selectors, and a
  reason (`unnamed`, `target_axis_not_on_source`, `rate_axis_out_of_scope`,
  `unsupported_layout`, `compile_failed`, or `expr_spec`), so reaction-only
  consumers can refuse to drop dynamics silently (#244).
- Axis-less states compile on the vectorized path as 0-d templates. Scalar
  and mixed scalar/templated transitions models publish reaction artifacts,
  `template_shapes` (with `()` for axis-less states), and a PyTree evaluator;
  axis-less history specs gain `history_eval_fn`. `CompiledReaction` gains
  `to_full_axes`, the target template's axis order, for reactions between
  templates with different axes (#246).
- `chain:` transitions publish reactions named `{base}_entry`,
  `{base}_advance_{k}`, and `{base}_exit`, and named pairwise `coord_shift`
  entries publish `{name}_{state}`. Chains (`entry.catalysts`, `catalysts`)
  and both `coord_shift` forms (`catalysts`) declare reactants beyond the
  consumed source, so the generated reactions can be complete for adaptive
  tau-leaping (#247).
- Routing (`X[imm:i] -> X[imm:j]`) and target-only fan-out
  (`I[age] -> X[age, imm:j]`) transitions publish reaction artifacts. The
  routed target coordinate is a trailing propensity dimension listed in the new
  `CompiledReaction.routed_axes`; each channel moves one unit from a source
  cell to one target coordinate. Same-slice routing masks its no-op diagonal,
  so a generator's negative diagonal never becomes a hazard (#248).

### Changed

- Specs with no axes now have a vector plan: `template_shapes` (all `()`) and
  `pytree_eval_fn` are populated instead of `None`. Their `eval_fn` keeps the
  scalar evaluator and its diagnostics (#246).
- `coord_shift` entries reject `reactants`, which pairwise entries previously
  ignored; declare extra reactants as `catalysts` (#247).
- Models using `chain:` or named pairwise `coord_shift` entries publish more
  reactions than before, because the generated transitions are now named
  (#247).

### Fixed

- A target-only fan-out from an axis-less source (`from: I`,
  `to: X[imm:j]`) compiled but failed at evaluation with an undefined axis
  name, and had no PyTree evaluator (#245).

## [0.5.0] - 2026-09-27

### Added

- The Flepimop provider now publishes normalized `axis_types` alongside axis
  labels and coordinates, so engine providers can distinguish categorical,
  ordinal, and continuous numerical semantics without guessing (#234).
- `jump_integral` now has portable conservative semantics: a mandatory
  row-source/column-target matrix rate density, explicit `up`/`down`/`both`
  masking, target quadrature on continuous axes, and reflecting/truncated
  boundaries. Public Array-API reference assembly, RHS, and eager value
  validation functions give engine providers one conformance target (#216).
- Advection and transport operators accept a normalized optional `direction`
  (`increasing` or `decreasing`) so a non-negative dynamic coefficient can
  carry explicit orientation without provider-side expression parsing. The
  public contract now defines velocity sign relative to coordinate order and
  upstream/downstream behavior for absorbing, reflecting, and periodic
  boundaries (#225).
- Numerical namespace discovery now uses `array-api-compat` through one
  public `op_system.array_namespace` helper. Compiled flat and PyTree RHS
  functions and the reference axis kernels therefore accept raw PyTorch
  tensors while preserving autograd, alongside existing NumPy and JAX
  concrete/traced behavior. PyTorch remains optional through the `torch`
  extra (#227).

### Fixed

- `flepimop2.system.op_system.__version__` now comes from installed
  distribution metadata instead of a stale hard-coded value. Direct imports
  from an uninstalled source tree use the explicit `0+unknown` sentinel (#231).

## [0.4.0] - 2026-09-26

### Added

- Named transitions publish provider-neutral `CompiledReaction` artifacts
  through `CompiledRhs.reactions` and the flepimop2 `reactions` system option.
  Each artifact carries a namespace-preserving propensity callback plus typed
  source, destination, free-axis, summed-axis, destination-pin, and source-pin
  metadata. This covers ordinary, source-only, point-to-point pinned, and
  collapse-to-fixed-target reactions; propensity expressions support mixing
  kernels, time-varying parameters, reductions, and nested aliases (#185,
  #187, #188, #191, #192, #194, #195, #196, #201).
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
- Routing transitions: a transition that binds one axis under an alias in
  `from` (`X[vax=u, imm:i]`) and another in `to` (`X[vax=v, imm:j]`) moves
  mass along that axis with a matrix-valued rate such as
  `nu * eta[time, imm:i, imm:j]`. It is lowered once per template to a
  contraction, so one transition replaces one coordinate-pinned transition
  per matrix entry and compiles in milliseconds at hundreds of coordinates.
  Self-routing diagonals are no-ops, the routed axis cannot be a block axis,
  and routing transitions have no `reactions` artifact yet. `validate_spec`
  counts them under `routing_transitions` (#88, step 2).
- `flepimop2-op_system` provides a `sparse_table` parameter module. It builds
  a routing matrix such as `eta[time, imm, imm]` from a declared support
  whose entries are numbers or nested parameter configurations (for example
  one per-day CSV series per `(imm_from, imm_to)` pair), and it exposes the
  support positions for inference. A full matrix remains a `fixed` parameter
  with a shape. On the COVID loc3 production spec, one routing transition
  plus a `sparse_table` reproduces the 17 coordinate-pinned vaccination
  transitions to 1e-17 (#88, step 3).

### Changed

- `flepimop2-op_system` now types flat stepper state and results with
  Flepimop2's backend-neutral `Array` protocol, matching its existing
  namespace-preserving NumPy and JAX behavior (#229).
- Normalization no longer expands every cell's reductions or renders every
  cell's equation string eagerly. `NormalizedRhs.equations`,
  `equations_ir`, and `equations_ir_reduce` may be lazy sequences, built
  on first access, that index, iterate, compare, hash, and pickle like
  tuples; their annotations are now `Sequence`. The first cell of each
  distinct equation is still expanded during normalization, so template
  errors surface as before. Block-axis stripping is lazy, and compile skips
  the history-operator scan when the spec calls no history helpers
  (`meta["op_system_may_have_history"]`). A routing reduction over 61 immune
  coordinates now normalizes and compiles in 1.5 s instead of 15.1 s, and
  the production COVID loc3 spec in 0.8 s instead of 15.2 s, with an
  identical compiled right-hand side (#88, step 1).

### Fixed

- The flepimop2 provider now declares its actual `flepimop2>=0.3.0`
  compatibility floor, and clean-wheel release validation exercises the
  lowest published direct dependencies instead of installing flepimop2 from
  its development branch.
- Normalizing a transitions spec no longer rewrites the caller's
  `transitions` entries in place when stripping the time axis from rates,
  so normalizing the same spec twice keeps time-varying parameters
  time-varying. Time stripping also matches `axis:alias` subscripts.
- `validate_spec` no longer lists synthesized coordinate masks
  (`__op_system_mask__*`) among consumed parameters.
- `validate_spec` computes `shape_groups` only when vectorization fails.
  Comparing every cell's expanded equation took 77 s on the COVID loc3 spec
  (131 million characters); a passing spec now validates in under a second.

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
