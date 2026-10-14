# Hierarchical compiler benchmarks

The synthetic workload in `benchmarks/hierarchical_model.py` describes three
nonlinear relaxation compartments over group and stage axes. Its aliases,
`sum_over` reducers, and `apply_along` stage transport preserve readable model
structure. One shared gain and transport matrix coexist with group-local rates
and initial conditions. Increasing groups grows local parameters and states;
increasing stages also grows the shared kernel quadratically.

The independent variant declares `factorize_axes: [group]`. The coupled variant
uses the mean of a state across groups and omits that declaration. It has no
block result: treating its groups as independent would change the equations.
Required qualification tests also reject a forced independence declaration for
that coupled model.

## Reproduce a run

Use a supported Python version, sync the checkout, and install desired backends.
These commands select CPU Torch and enable JAX's double precision inside each
worker. NumPy alone is enough for the default benchmark command.

```shell
uv sync --dev --extra jax
uv pip install 'torch>=2.6' --index-url https://download.pytorch.org/whl/cpu
just benchmark --backends numpy jax torch --groups 4 16 --stages 2 4 \
  --coupling both --dtype float64 --repeats 5 --inner-loops 100 \
  --warmups 3 --output benchmark.json
```

Repeat with `--dtype float32` to qualify that precision separately. For a quick
correctness-only sweep, add `--check-only`. `just qualify` executes small
correctness cases in the required NumPy/JAX/Torch jobs and rejects skipped
backend cases. Larger timing runs remain opt-in; CI has no wall-time thresholds.
The runner aborts if a requested backend is missing or a case fails agreement.

Each case runs serially in a fresh process. Workers request one thread through
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS`; JAX uses CPU.
These settings do not pin CPU affinity or guarantee that every runtime uses
exactly one thread. Keep other heavy workloads idle when collecting evidence.
The command is also available as `uv run --no-sync python -m
scripts.benchmark_hierarchical` with the same arguments.

## What is measured

Every generated and handwritten layout must first agree with independent
cell-wise NumPy equations. Checks cover shape, dtype, namespace, and RHS values.
JAX and Torch also compare the derivative of the squared RHS norm with respect
to the shared gain against central differences of the independent equations.
NumPy records that AD comparison as unavailable. The reference finite-difference
step, tolerances, values, and errors are retained in the result. This checks one
fixture and one parameter direction, not all derivatives or scientific regimes.

| Result | Meaning |
| --- | --- |
| `normalize` | Specification validation, alias/template processing, and normalized IR construction |
| `compile_rhs` | Compilation from already normalized IR, including lowering and generated Python callables |
| `backend_setup_seconds` | Backend import and conversion of the resident input arrays |
| `cold_jit_seconds` | JAX tracing/lowering/native compilation of a new evaluator wrapper; null for NumPy and eager Torch |
| `first_evaluation_seconds` | First synchronized call after that explicit JAX compilation, or after eager correctness checks |
| `warm` | Samples of average synchronized RHS latency, with median, minimum, and maximum |
| `warm_calls_per_second` | Reciprocal of median warm latency |
| `fixture_bytes` | Host state/parameter array payload, excluding Python objects and runtimes |
| `process_peak_rss_bytes` | Worker peak resident memory on Linux/macOS, including imports, correctness/AD checks, and every layout |

Normalization and compiler samples repeat independently of warm evaluation.
Each warm sample averages `inner_loops` calls after `warmups` calls. JAX outputs
are synchronized on every call. Inputs are already resident; device transfers,
ODE integration, and likelihood calculations are outside the measurement.
Correctness runs before explicit JIT timings, so this is function compilation
cost in an initialized backend, not whole-process startup. Shapes are fixed per
case. No Torch compilation or GPU timing claim is made.

All layouts accept the same structured input and return structured derivatives.
The flat wrapper includes packing/unpacking. NumPy block evaluation uses a
Python loop; JAX and Torch use native `vmap`. These different execution policies
are part of the recorded layout comparison, not interchangeable timing paths.
The handwritten vectorized implementation uses a matrix multiply for transport;
the independent oracle uses cell loops and is never timed as the handwritten
implementation.

JSON includes the fixture definition, dimensions, specification size, array
payload, full samples, repeat policy, dtype, Python/compiler/backend versions,
CPU/platform information, thread settings, command arguments, and source commit
with dirty-checkout status. Keep the JSON with any published timing claim. Peak
RSS is a process high-water mark, not an incremental tensor or per-layout memory
measurement. JAX workers may use more threads than the requested BLAS policy.

## Interpreting the tradeoff

The structured specification supplies named compartments, alias relationships,
parameter locality, and independence metadata from one model description.
Generated code avoids maintaining separate backend equations. That authoring
benefit carries normalization/code-generation overhead and does not imply a
universal speedup over well-written native code.

A preliminary serial CPU sweep at 4 and 16 groups with 3 stages, float64, Python
3.13.13, NumPy 2.5.4, JAX 0.11.2, and CPU Torch 2.14.1 used an Intel Core Ultra 7
265U, three repeats of 30 calls after three warmups. All 12 independent/coupled
backend cases agreed with their references. Handwritten NumPy was faster than
generated evaluation in those cases; NumPy's Python block loop grew with group
count. JAX layouts were often close at these small shapes, while eager Torch
results varied substantially. This small sweep motivates larger, repeated
measurements rather than a ranking or performance threshold.

## Related tools and a fair comparison

Other tools already connect model authoring to execution. The
[pyodesys project](https://github.com/bjodah/pyodesys) offers symbolic ODE systems,
automatic Jacobians, native code generation, and integration interfaces; see its
[software paper](https://joss.theoj.org/papers/10.21105/joss.00490).
[MCSimMod](https://usepa.github.io/MCSimMod/) wraps GNU MCSim models for use with
R's deSolve and native C compilation; see its
[software paper](https://joss.theoj.org/papers/10.21105/joss.08492).

| Tool | Authoring/execution focus relevant to this workload |
| --- | --- |
| `op_system` | Restricted specifications, explicit axes/aliases/operators, runtime array namespace, structured/block RHS metadata |
| pyodesys | Symbolic/numerical ODE definitions, Jacobians and integration, including native code-generation options |
| MCSimMod | MCSim model files, R/deSolve integration, native C model execution |

This runner measures only `op_system` and handwritten native functions. It does
not report cross-tool timings. A fair future comparison must implement identical
equations, fixtures, precision, and coupling; verify values first; and separate
model preprocessing, native compilation, and warm RHS evaluation. Compare
integration only with matching solvers, tolerances, output points, and error
criteria. Report unavailable AD/block features explicitly rather than assigning
them artificial timings. The tools' authoring and integration scope should
accompany any speed comparison.
