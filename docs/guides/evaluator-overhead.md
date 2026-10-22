# Profiled NumPy evaluator overhead

The hierarchical evaluator now avoids general NumPy broadcasting when an
ordinary ndarray already has the target shape. It creates a read-only view,
preserving shared storage and the caller's input flags. Scalars, shapes that
require broadcasting, ndarray subclasses and other native namespaces keep the
existing broadcast operation.

This applies to alias and equation buffers in flat, structured and independently
factored block evaluators. It does not fuse equations or remove required
broadcasts. A NumPy parameter cannot bypass native namespace conversion because
the fast path requires the NumPy namespace.

## Results and shared-host check

The 80 separate-process workers all qualified (72 timing and eight profiling),
but their unchanged handwritten controls also varied substantially. For one
case the baseline/candidate control medians differed by about 44%. Retain those
rows as descriptive observations; they do not establish precise causal speedups
or JAX/Torch regressions.

The [paired follow-up plan](../assets/evaluator-overhead-2026-10-21/paired-plan.json)
and [12 complete workers](../assets/evaluator-overhead-2026-10-21/paired-receipt.json)
alternate the original and candidate NumPy factories within each process. All
other compiler sources must match byte-for-byte. Both use one normalized vector
plan and independently check the actual flat/structured results against the
public cell-loop oracle. Handwritten controls use the same function twice.

| Shape/coupling | Flat baseline/candidate | Structured baseline/candidate |
| --- | ---: | ---: |
| g4-s2 independent | 92.75 / 76.20 us | 67.75 / 50.58 us |
| g4-s2 coupled | 94.17 / 78.52 us | 67.69 / 52.88 us |
| g16-s8 independent | 98.88 / 82.58 us | 72.74 / 57.14 us |
| g16-s8 coupled | 103.08 / 86.80 us | 76.21 / 61.15 us |

The table reports medians across three fresh processes, each summarized from
15 adjacent sample pairs of 200 calls. Median paired ratios imply 16.5–18.0%
lower flat latency and 19.5–24.7% lower structured latency. The 12 unchanged
control process-median ratios remain within 3.4% of one. Pairs inside a process
are not independent process replicates. This follow-up measures warm factories,
not compiler construction, full solvers or block-layout speedups.

Separate profiles attribute about 14–15% of baseline structured self time to
NumPy's general `_broadcast_to`, with 6,000 calls per 1,000 RHS evaluations.
The matching candidate uses 6,000 lightweight checked view calls instead. The
complete raw profiles retain other overhead. Process RSS is retained separately;
no memory reduction or GPU/training speedup is established by this change.

## Comparison protocol

The [plan](../assets/evaluator-overhead-2026-10-21/plan.json),
[complete records](../assets/evaluator-overhead-2026-10-21/receipt.json) and
[summary](../assets/evaluator-overhead-2026-10-21/summary.json) retain the public
fixture, source hashes, versions and raw samples. Baseline compiler source is
`ad3c28ac63e9af3b1210413a764b44a9c9bd2074`; changed candidate source is identified
by content hashes. Both variants use the same benchmark harness.

Timing cases use two shapes (four groups/two stages and 16 groups/eight stages),
independent and coupled equations, float64 and NumPy/JAX/CPU Torch. Every case
runs in three fresh serial processes per variant. Adjacent pairs alternate
baseline/candidate order, and case order reverses on the middle repetition.
Each evaluator has five warm samples of 200 calls. Normalization, compilation,
backend setup, selected JAX compilation and first/warm execution remain separate.
Compiler samples within a process can reuse caches; only the first samples
represent that worker's initial construction.

Separate fresh NumPy workers profile 1,000 calls per layout after qualification.
Profiler times identify Python overhead and must not substitute for ordinary
latency. Whole-worker RSS includes imports, compilation and, for profile workers,
profiler storage. It does not measure solver state or estimator bank memory.

Completed workers pass the existing independent cell-loop value oracle and
native gain-derivative checks before reporting measurements. The shared host has
singleton CPU affinity and observed one-thread native pools. Results are bounded,
descriptive measurements, without CI timing thresholds or application, solver,
training or GPU speedup claims.

## Reproduce and inspect

Install the optional benchmark reporter and selected CPU backends. From the
candidate checkout, select each immutable compiler's `src` directory through
`PYTHONPATH` while keeping the benchmark harness fixed:

```sh
uv sync --dev --extra benchmark --extra jax --extra torch
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=false \
  uv run --no-sync python -m scripts.benchmark_hierarchical \
  --backends numpy jax torch --groups 4 16 --stages 2 8 \
  --coupling both --repeats 5 --inner-loops 200 --warmups 3 \
  --output /tmp/hierarchical-timing.json
```

This command performs the Cartesian shape sweep; the receipt uses the two paired
shapes in its plan. Apply the plan's singleton affinity and outer worker timeout,
retain every attempt and reverse the order for equivalent comparisons. Use fresh
output paths. Collect Python attribution separately:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python -m scripts.benchmark_hierarchical \
  --backends numpy --groups 16 --stages 8 --coupling coupled \
  --check-only --profile-calls 1000 --output /tmp/hierarchical-profile.json
```

Profiling requires NumPy correctness-only mode. Environment records include
affinity and reviewed thread-pool fields; without the optional reporter, observed
pools are explicitly unavailable. Local source/library paths are excluded from
public profiles and pool metadata.

Reproduce the adjacent factory comparison in each fresh NumPy worker using the
trusted baseline checkout. The program checks source compatibility before loading
its Python factory; unrelated compiler differences are explicitly rejected.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python -m benchmarks.paired_evaluator \
  --baseline /path/to/ad3c28a-checkout --groups 16 --stages 8 --coupled \
  --repeats 15 --inner-loops 200 > /tmp/paired-evaluator.json
```

Repeat three fresh workers per paired shape/coupling, with the plan's affinity
and reversed case order. The main comparison separately retains whole-package
construction/first-call observations; this narrow comparison isolates warm RHS
factories and makes no construction-cost claim.

Remaining overhead includes environment binding, alias/reduction evaluation and
layout handling. Optional Numba needs a generated flat-array signature and
measured generation/JIT/warm break-even; decorating a dynamic closure is
insufficient. Torch sequential compilation retains its separate version, carry,
gradient and fallback contract. This change supplies no evidence for a Rust rewrite.
