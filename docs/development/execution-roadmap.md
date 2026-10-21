# Compiler execution and export roadmap

Engineering decisions recorded on 2026-10-21. Planned work does not expand the
[qualified backend surface](../guides/backend-qualification.md). The shared
[execution roadmap](https://github.com/ACCIDDA/op_engine/blob/main/docs/development/execution-roadmap.md)
and [baseline protocol](https://github.com/ACCIDDA/op_engine/blob/main/docs/development/execution-baselines.md)
describe cross-package ownership and measurements.

## Keep the public numerical contract

Retain Python model interfaces and namespace discovery from input arrays. Keep
NumPy, JAX and CPU Torch qualification. CuPy/CUDA is the next explicit end-to-end
target, not an implication of Array API compatibility. Sparse operators,
compilation, differentiation and device support have separate contracts.

Generated vectorized expression functions are already cached without per-expression
runtime `eval`. Measure normalization, compilation and evaluator layouts before
choosing an acceleration target. Optional Numba needs a supported generated
function signature; decorating an arbitrary dynamic Python closure is insufficient.
An explicitly requested unavailable/unsupported accelerator must produce a clear
result, rather than silently appearing to use that accelerator.

## Compiler-owned work

| Work | Scope and dependency |
| --- | --- |
| Structural model identity | Versioned canonical content serialization with ordered state/parameter/coordinate layouts, expressions, observables, constants and relevant compilation/view options. Separate structural identity from fitted values and runtime identity. Do not change object equality/hash behavior incidentally. |
| Derivative exports | Reuse checked dependency structure. State and parameter derivatives and observable derivatives have distinct shapes and meanings. Unsupported IR operations fail explicitly; qualify small dense values/actions independently before general sparse or matrix-free claims. |
| IR export | Versioned typed serialization and a bounded Julia emitter with explicit supported rational/expression semantics. Export remains separate from normal evaluation. |
| Numba experiment | Optional NumPy-only lowering, chosen after representative baseline measurements; report compilation and amortized execution costs and preserve reference behavior. |
| Rust/PyO3 IR experiment | Deferred private implementation of normalization/tree passes, preserving Python IR/public contracts; enter after compile-time profiles or external-runtime consumers justify it. |

Julia export serves optional external structural-identifiability tools; it does
not start a full Julia port. A future C/LLVM emitter is another separately scoped
consumer, not an automatic consequence of serializing the IR. No Rust rewrite of
the small Python array protocol is planned.

## Reuse existing artifacts

Scaling-gauge declarations, bounded detection, verified outputs and anchored
coordinate maps already exist. Inference consumers should reuse them rather than
reimplement detection. Reported gauge rank or remaining parameter-coordinate count
does not establish structural identifiability. Initial conditions, observation and
noise parameters can change the statistical dimension.

Latent observables already compile separately from statistical measurement noise.
OP Engine owns numerical observation evaluation/reduction; OP Infer owns shared
sampling/log-density, censoring and prior semantics. Keep this boundary when
qualifying a complete compiled simulation-to-observation workflow.

## Sequence

First reconcile open work and run the existing hierarchical compiler baseline.
Develop structural identity and forward-consumer contracts early. Derivative and
IR export work can proceed alongside shared observation/prior interfaces. Qualify
JAX execution and explicit GPU cases before selecting additional native kernels.
Revisit Rust normalization only after those measurements and contracts stabilize.

## External-tool reproducibility

The downstream evidence on [#326](https://github.com/ACCIDDA/op_system/issues/326)
includes rank disagreements and timeouts from manually transcribed external
models. Their cause remains unresolved. A generated export should remove the
transcription ambiguity, not assert that an external result is correct.

Preserve stable axis-expanded names and aliases, ordered coordinates, initial
conditions, known time-varying inputs versus unknown parameters/constants, and
observables with their measures. Unsupported operations must have explicit
diagnostics. Bind the schema/emitter version and supported-subset contract to
the structural identity from [#328](https://github.com/ACCIDDA/op_system/issues/328).

On independent synthetic fixtures, compare exported expression values and
state/parameter/observable derivatives against the same compiled IR. Retain tool
versions, assumptions, exact/numerical reference definitions, witness points,
tolerances, disagreements and censored timeouts. Different rank notions or known
input assumptions must not be treated as the same calculation. The optional Julia
consumer remains interoperability work alongside forward/provenance interfaces.
