# Changelog

All notable changes to `op_system` and `flepimop2-op_system` are documented
in this file. The two packages are released together under one shared
version; format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- Preserve gradients of already-native shaped Torch parameters, vectorized time
  and state-coordinate buffers on minimum supported Torch. Avoid the older
  compatibility conversion that detaches existing tensors; preserve dtype/device.
  Require the complete minimum-Torch backend checks with zero skips.

- Scope prepared template substitutions to their normalization batch, releasing
  expanded model mappings and respecting shaped-parameter exclusions. Preserve
  batch reuse while removing the process-global identity cache.

- Avoid general NumPy broadcasting for ordinary buffers already at the required
  shape, preserving read-only views, aliasing and input mutability. Keep other
  shapes, subclasses and namespaces on the existing broadcast path.

- Keep generated scalar equation/derivative constants on the state's device and
  retain differentiable native time bindings, including minimum supported Torch.

### Added

- Add explicit optional JAX/Torch state JVP and transpose/VJP adapters for
  compiled RHS and scalar/array observables, preserving source and input layouts.
  Qualify independent nonlinear products, native transforms and outer gradients;
  reject operator/history/gauge execution. Retain CPU startup, warm and RSS
  comparisons without changing defaults or claiming universal memory savings.

- Export bounded conservative dependency rows from retained compiler IR without
  numerical derivative construction. Retain pinned selectors, reductions,
  buffer-parameter and grouped-observable layouts, source identity and immutable
  existing block proofs. Unsupported coverage remains distinct from empty rows.

- Cache validated scalar RHS and latent-observable expression functions with
  explicit runtime bindings, sharing the vector evaluator helper and preserving
  inspection code. Qualify strict current-Torch FX capture alongside the existing
  minimum eager/AD contract; retain paired eager scalar-latency costs.

- Add an explicit optional CuPy CUDA 12 installation extra and ten device-gated
  compiler contract cases. The required CuPy command checks CUDA execution and
  rejects skips; standard CPU-suite skips do not constitute GPU qualification.
  Allow retaining required-backend JUnit reports for device evidence.

- Add explicit optional NumPy CPU lowering with one eager homogeneous native
  signature, reusable packed state/parameter layouts and bounded source emission.
  Qualify float32/64 values, input/error contracts and independent/coupled models
  against the handwritten oracle; retain cold compilation, packing and memory costs.

- Add opt-in NumPy Python-call profiling to the existing correctness-gated
  hierarchical harness. Keep profiled workers separate from latency samples and
  retain reviewed effective affinity/thread-pool records.

- Export canonical typed IR envelopes with guarded readonly decoding and the
  existing structural model identity. Add bounded scalar state/parameter and
  observable derivative matrices with native NumPy/JAX/CPU Torch evaluation,
  conservative dependencies, dense actions and Julia expression exports with
  explicit bindings. Keep unsupported tensor/operator/history/gauge compositions
  and external-tool rank qualification explicit.

- Add opt-in schema-1 structural model manifests and SHA-256 references for
  guarded downstream provenance. Preserve ordered layouts, typed IR, constants,
  observables, operators and gauge/coordinate options; intern shared trees
  without materializing deferred scalar IR. Keep runtime values/settings and
  Python equality/hash behavior separate (#328).

### Documentation

- Record compiler/backend execution choices, structural identity and derivative/export
  boundaries, representative baseline requirements and the deferred Rust IR experiment.

- Replace issue and pull-request breadcrumbs in guides and API docstrings with
  current behavior, named interfaces and explicit limitations.

- Show an executable flat-state/subgroup-axis adapter connecting the log-state
  view to OP Engine's deterministic core interface. Keep callback layout and
  decoded output-axis requirements explicit; runtime APIs are unchanged (#314).

## [0.9.0] - 2026-10-19

### Added

- Request mixed identity/log deterministic state coordinates with
  `state_metadata` and an explicit `coordinate_view()`. Compile the chain rule,
  cancel supported exact multiplicative factors before division, encode initial
  states without floors, decode natural observations and retain native domain
  flags. Flat, shaped and independent block evaluators preserve physical-model
  contracts; separate operator/history composition is explicitly unsupported.
  Checked examples explain balance errors, representability, tolerance changes,
  derivative metadata and natural stochastic jumps (#314).

- Declare shared positive scaling gauges; report seeded finite covariance/output
  checks, optional rational homogeneity detection, ranks and parameter invariant
  monomials. Scalar anchor maps bind parameters across evaluators, record native
  domain flags and retained group coordinates, and survive pickling. CLI summaries
  distinguish the reported action from structural identifiability. Preset initial
  states and unsupported operator/history/table contracts are rejected during
  fixing/analysis as documented; gauge-fixed stochastic exports fail closed
  until a jump-process scaling contract exists (#313).

- Lower single-axis unweighted, unfiltered rectangular kernel products to
  native matrix-vector operations, including vector products and named-axis
  transposes. Add a two-axis example, independent NumPy/JAX/CPU Torch checks,
  a generated figure/table and an optional size-sweep benchmark (#312).

### Changed

- `StateDeclaration` appends the `scale` field, defaulting to `identity`. Existing
  four-argument construction still works; consumers unpacking the record as a
  four-item tuple must use attributes or handle the added field (#314).

- Cache expression functions for vectorized flat/PyTree evaluation, binding
  fresh arguments without runtime `eval` per alias or equation. Preserve the
  scalar fallback and expose the original generated code to internal plan
  inspection tests. Lower elementwise `np.where` through typed IR and document
  its eager branch evaluation (#312).

### Documentation

- Render the complete kernel-product and scaling-gauge example includes, using
  attributed Python fences so formatting preserves the include directives.

- Align package summaries with spatial, nonlocal and reaction support, and state
  the alpha development stage separately from public release availability.

- Simplify introductions and feature descriptions, removing promotional phrasing
  and separating supported interfaces from numerical and scientific claims.

- Clarify the scope across PDE/structured fields, nonlocal operators, ODEs and
  stochastic reaction contracts. Map model classes to numerical consumers and
  label the compartmental examples as individual cases.

- Illustrate the executable transfer-model vignette with a numerical figure,
  generated drift/population table and retained plotting inputs. Check source,
  data and export identities in the docs build; keep plotting optional.
- Audit current capabilities against public APIs; correct scope and limitations,
  add executable workflow examples and public API doctests, and document their
  CI and optional-dependency checks. No runtime API or dependency changes.


## [0.8.0] - 2026-10-19

### Added

- Reactions expose `resolved_propensity_fn` for consumers that have explicitly
  replaced time tables with current-time slices, their consumed table axes,
  explicit-time dependence, and per-reaction verified `independent_axes`.
  Certificates require declared `factorize_axes` and preserve source/target
  coordinates and propensity locality before deterministic cancellation.
  The provider merges mixing kernels into both callbacks. Ordinary callbacks
  retain their scalar-time interpolation contract (op_engine #241).


- `initial_state` accepts a literal finite number (`"C[loc,age]": 0.0`) as a
  constant initial value broadcast over the selected cells, so cumulative
  counters need no zero seed table. Entries are stored as floats in
  `meta["initial_state"]`, the provider requests no parameter for them, and the
  op_engine engine consumes them unchanged. Booleans, `nan` and `inf` are
  rejected; string and `shaped` entries are unchanged (#303).

- `reaction_balances` exports exact integer coordinate-sum identities for fully
  represented named scalar unit transfers, with explicit external source terms,
  scientific declarations and applicability assumptions. Immutable reports
  retain original omissions and withhold all laws for unsupported dynamics,
  layouts or known metadata conflicts. Native RHS checks cover NumPy/JAX/CPU
  Torch in both precisions; weighted spatial/operator balance extensions remain
  outstanding on #295. Compiler storage and runtime evaluators are unchanged.

- Optional named `observables` compile independent state/parameter/time
  expressions, declared-node samples, counters, unweighted sums and explicit
  physical integrals into native flat/PyTree evaluators. Read-only compiled
  metadata exports output shapes/coordinates, direct dependencies and applied
  measures. Observable-only parameters leave RHS binding requirements unchanged.
  NumPy/JAX/CPU Torch qualification covers the initial arithmetic/exp, indexing,
  broadcasting and reduction paths; numerical time/mesh adapters remain consumer
  responsibilities (#294).

- Optional `state_metadata` declares counts, cell mass, densities/concentrations
  and extensive cumulative counters. Axes can declare aligned counting, cell
  or point-quadrature measures. Read-only `CompiledRhs.state_meta`/`axis_meta`
  export immutable records with unknown annotations retained as unknown.
  Nonuniform cell-volume examples distinguish physical integrals from existing
  point quadrature and unweighted `sum_state()`; metadata leaves equations and
  reductions unchanged (#293).

- Optional `parameters` declarations validate support, units, descriptions and
  fixed/free annotations for inferred scalar, shaped and time-varying RHS
  parameters. Compiled `param_meta` exposes immutable records without binding
  values or changing numerical behavior (#292).

- Readiness coverage audit maps axis/reducer/transition/provider/error contracts
  to existing regression suites, adds independent values for every generated
  kernel form and provider injection, and checks empty-axis boundaries (#29).

- Terminology guidance distinguishes runtime array namespaces from numerical
  engines and framework providers. Compiler keyword compatibility is preserved;
  no compile-time `array_backend` configuration is introduced (#79).

- Runnable YAML model guide covers states/seeds, hazards versus absolute
  inflows, categorical/ordinal/continuous axes, mixing orientation, syntax,
  operators and the reserved constraint boundary. Exact SEIR snippets run
  against independent cell flows in source and installed-wheel tests (#45).

- Required installed-provider typing qualification verifies the shipped
  `py.typed` marker, public state/stepper types, and rejection of invalid state
  arguments. Clean-wheel tests and a dedicated CI job run the downstream check;
  provider source typing remains isolated from the core environment (#60).

- Required backend regressions verify local alias and shaped-parameter behavior
  after block stripping in expression and transition models. Independent cell
  flows check flat, structured, and block evaluation on NumPy, JAX, and CPU Torch
  in both precisions; positional parameter-index regressions cover either block
  axis position and protect the previously implemented fixes (#170).

- A correctness-gated hierarchical benchmark compares handwritten and generated
  NumPy/JAX/CPU Torch RHS layouts over independent and coupled group variants.
  Isolated serial workers separate normalization, code generation, JAX
  compilation, and warm evaluation, emitting dimensions, agreement, timings,
  memory, and provenance as JSON. Small correctness cases are mandatory backend
  checks; larger timings remain opt-in through `just benchmark` (#265).

- Root contribution and citation guidance covers clean development setup,
  compiler/provider boundaries, baseline and required backend checks, installed
  wheels, support, and shared releases. `CITATION.cff` preserves existing package
  attribution and the verified release URL; `just citation` checks its schema
  and metadata locally, in CI, and before releases. The archive guide records
  the identifier audit and software-paper deposit plan without inventing a DOI
  or preferred paper citation (#266).

- `qualify_rhs` compares named state/parameter fixtures against independently
  supplied equations through flat, structured, and available block evaluators.
  Reports include tolerances, finite values, shapes/dtypes, and agreement
  diagnostics; conditional `InvariantCheck` callbacks and optional JAX state
  directions add balance and derivative checks. Unavailable block layouts,
  wrong references, and callback failures are reported explicitly without
  changing structural validation or normal RHS evaluation (#264).

- Required core compiler qualification jobs install NumPy, JAX, and CPU
  Torch explicitly and reject missing or skipped backend tests. A synthetic
  hierarchical model checks flat, structured, and valid block evaluation
  against independent equations, dtype/shape/namespace preservation, JAX
  jit/vmap/directional derivatives, and CPU Torch autograd. Coupled groups
  are rejected when declared independent. See `just qualify` and the backend
  capability guide; optional runtime extras remain optional (#263).

### Fixed

- Reaction callbacks reject undeclared state dimensions with a reaction/state
  shape diagnostic rather than expression-dependent broadcasting or a misleading
  missing-name error. Explicit replicate axes remain supported (#306).
- Each reaction binds only the shaped and time-varying parameters referenced by
  its lowered, alias-expanded expression. Unrelated reactions no longer
  interpolate time tables; interpolation policies and endpoint semantics are
  unchanged, and no mutable parameter cache is introduced (#305).


- The authoring guide now states that omitted/null transition destinations are
  unsupported. Executed examples qualify open-loss expression equations and
  explicit loss compartments, including their different total/reaction-export
  semantics; the compiler's supported schema is unchanged (#297).

- README YAML flow collections quote bracketed state/reactant selectors, so
  examples load with a standard YAML parser before compiler validation. Source
  and installed-wheel suites parse every README YAML fence (#286).

- Shared citation metadata records `GPL-3.0-or-later` for the provider, matching
  its existing source grants, alongside core MIT. Citation validation rejects
  license drift; this corrects metadata without changing either grant (#275).

- Vectorized continuous reductions construct quadrature weights through the
  runtime Array API, using the floating state dtype/device. This fixes CPU
  Torch evaluation and float32 promotion on NumPy/JAX; analytic mixed-axis
  integration regressions are required backend qualification (#283).

- `sum_state()` lowers on the vectorized path and totals declared buffers
  through native reductions in flat, structured, and reaction evaluation.
  Totals exclude aliases/parameters and refresh per call. Global totals reject
  invalid block factorization; README chain examples distinguish the generated
  prefix from a separately declared base compartment (#260).

- Untyped RHS literals and constant aliases inherit the floating state dtype
  in flat, structured, and block assembly instead of using backend integer or
  default-float precision. Native typed values remain unchanged; required CPU
  regressions cover both precisions and zero state Jacobians (#277).

- Provider `erfc` mixing kernels construct float64 matrices with the standard
  complementary error function instead of calling the unavailable `numpy.erfc`.
  Independent value regressions cover scaled distances and positive tails (#274).

- Mixed selectors in compatibility template substitution retain explicitly
  bound coordinates when substituting wildcard axes, instead of collapsing a
  partial selector to a cell name. Required CPU backend regressions qualify
  nested typed-IR alias reductions over all state axes in both precisions;
  selector guidance distinguishes modern colon expressions from legacy string
  substitution (#113).

- Raw equations, transition rates, and alias bodies reject bare references to
  templated state bases during normalization, with context and an indexed-form
  suggestion. This prevents compile-success/runtime-NameError failures without
  changing valid scalar/indexed references or generated block symbols. Cached
  IR is inspected once per distinct raw expression before expansion (#259).

- Generated axis reordering and vectorized assembly use Array API
  `permute_dims`, fixing Torch evaluation of reordered axes and reducers
  that previously called NumPy-style `transpose` (#263).
- Axis-indexed expression normalization builds state membership, expanded
  equation lookup, and coordinate-position maps once rather than repeating
  full scans for each cell. Alias inlining uses the mapping's keys view
  instead of rebuilding the expanded alias set. Coordinate order, explicit
  cell equation precedence, shaped parameters, and validation errors are
  preserved; transition normalization shares the coordinate maps (#267).

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
