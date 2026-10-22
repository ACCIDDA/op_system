# Optional NumPy CPU lowering

Install `op_system[numba]` and explicitly call `op_system.numba.compile_numba`.
This leaves normal `compile_spec` evaluators and namespace discovery unchanged.
It generates a single native `(t, y, p)` function from the compiler's validated
expression plan. No dynamic evaluator closure or per-step environment dictionary
is sent through Numba. Numba is imported only when compilation is requested.

```python
import numpy as np
from op_system import compile_spec
from op_system.numba import compile_numba

model = compile_spec({
    "kind": "expr",
    "state": ["x"],
    "equations": {"x": "-rate*x"},
})
native = compile_numba(model, dtype="float64")
y = native.pack_state({"x": np.array(2.0)})
p = native.pack_parameters({"rate": np.array(0.5)})
assert np.allclose(native.evaluate(0.0, y, p), [-1.0])
```

Compilation is eager, including the homogeneous native signature. Inspect
`native.compilation_seconds` for source-generation, optional-import and native
compilation wall times. `native.source`, ordered `state_names`, `state_layout`,
`parameter_layout` and the existing structural model identity describe what was
compiled. Runtime parameter arrays are not captured in the artifact. There is no
process-global artifact cache, disk cache, automatic accelerator selection or
Python fallback.

## Input contract

Choose one dtype, `float32` or `float64`, per artifact. `pack_state` and
`pack_parameters` require exactly the declared names and shapes. They explicitly
copy C-order values into owned read-only flat arrays. Scalars can be Python real
numbers; explicitly typed NumPy scalars/arrays must already have the chosen dtype.
Convert other precision explicitly before packing. Lists, array subclasses and
foreign namespaces are rejected before attempting host conversion.

Reuse packed parameters while their values stay fixed; repack when parameters
change. `evaluate(t, y, p)` validates scalar time, exact flat lengths, dtype and
C contiguity, then returns a fresh flat NumPy derivative in `state_names` order.
A normal flat NumPy state vector in that layout needs no additional state packing.
Inputs can be writable or read-only and are never mutated. Output assembly and
expression temporaries allocate on every call.

`native.kernel` exposes the one compiled native signature for a Numba consumer.
It checks lengths and enables bounds checking. Its dispatcher permits native
signature conversions such as Python scalar time; use `evaluate` for strict
Python-side dtype validation. Callers of `kernel` must supply native C-contiguous
one-dimensional arrays in the documented dtype/layout. No solver adapter or
compiled integration loop is selected by this API.

## Supported subset and qualification

Fully vectorized template equations retain shaped parameters, scalar/templated
aliases, common subexpressions, static indexing, broadcasting and transposes.
Arithmetic, `abs`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `maximum`, `minimum`,
`where` and sums over one static nonnegative axis are supported. `keepdims` sums
require an explicit axis. Scalar source IR has a separate bounded lowering of
at most 128 states when the tensor planner does not apply. Tensor equations
requiring partial boundary unrolling, history, operators, gauge fixing and
time-indexed parameter tables fail explicitly. Other native typing failures
raise compilation errors. Natural-coordinate evaluators are compiled; use a
separately qualified coordinate view when transformed coordinates are required.

State and packed parameter counts are limited to one million each. Each captured
expression has at most 10000 AST nodes and depth 64; generated source has at most
100000 characters. This avoids expanding tensor IR into unbounded scalar code.
Arithmetic literals follow the requested native precision, with `fastmath=False`
and NumPy floating division behavior. NumPy-equivalent rounding is checked with
precision-specific tolerances rather than promised bitwise across implementations.

The optional dependency range is Numba 0.67–0.68. Required qualification includes
Numba 0.67/NumPy 2.0.2 and the current resolving stack. It checks the independent
hierarchical balance oracle in both precisions and coupling modes, scalar time,
input/parameter reuse, output order, overflow, mutation, bounds, unsupported calls,
missing dependencies and disabled JIT. It must execute without skips:

```sh
just qualify-numba
```

This is CPU NumPy qualification. It does not add CUDA, CuPy, automatic gradients,
Torch compilation or support for a JAX tracer. Keep existing JAX whole-loop
execution and GPU measurements separate from these RHS measurements.

## Performance evidence

The [final receipt](../assets/numba-cpu/receipt.json.gz) contains source
snapshots/hashes, environment and effective thread pools, commands, phase times,
raw sample orders/times, numerical checks and worker logs. The
[summary](../assets/numba-cpu/summary.json) is uncompressed. All 48 final workers
passed, with no timeouts. Earlier campaigns remain separately archived:
[initial source](../assets/numba-cpu/initial-receipt.json.gz) and
[flat packing output](../assets/numba-cpu/flat-packing-receipt.json.gz).
Those 96 prior workers also passed their numerical gates. The table uses only the
final campaign, which includes converting the native flat result back into a
structured dictionary in the full-packing comparison. Earlier flat-output packing
measurements are not treated as structured-return timings.

The final CPU campaign uses NumPy 2.5.4, Numba 0.68.0 and llvmlite 0.50.0 on the
recorded shared Linux host, relative to the post-broadcast-cleanup compiler at
`cb57783d2820cafdd43400b1aeb442498e8deccd`. Two shapes, both precisions and coupling
modes have three fresh workers per NumPy-only/native mode. Workers run serially
on one fixed logical CPU with one BLAS thread. Each native worker rotates seven
methods over 15 samples of 200 calls, including the unchanged NumPy control. Its
process-median ratios span 0.993–1.014. Process repetitions are the replication
unit; inner calls are not independent machine repetitions.

| Case | NumPy flat / validated native (us) | Raw kernel gain | Validated gain | Repack all gain | Break-even validated |
| --- | --- | --- | --- | --- | --- |
| g4-s2 independent float32 | 64.09 / 6.58 | 24.3x | 10.0x | 2.3x | 171,123 |
| g4-s2 independent float64 | 61.66 / 6.63 | 24.3x | 9.3x | 2.2x | 185,788 |
| g4-s2 coupled float32 | 65.60 / 6.77 | 23.5x | 9.7x | 2.2x | 177,979 |
| g4-s2 coupled float64 | 63.73 / 6.80 | 24.4x | 9.6x | 2.2x | 184,154 |
| g16-s8 independent float32 | 67.94 / 10.62 | 11.1x | 6.5x | 2.1x | 173,858 |
| g16-s8 independent float64 | 69.15 / 10.77 | 10.4x | 6.4x | 2.1x | 169,870 |
| g16-s8 coupled float32 | 72.15 / 10.82 | 11.4x | 6.7x | 2.1x | 170,544 |
| g16-s8 coupled float64 | 72.08 / 11.06 | 10.8x | 6.5x | 2.1x | 173,601 |

Latency columns are medians of worker medians; gains are medians of ratios
computed within each native worker. Raw-kernel and validated gains compare with
flat NumPy evaluation on an already-flat state. “Repack all” includes state and
parameter packing and reshaping the derivative into a structured dictionary on
every call; it compares with structured NumPy evaluation. Validated calls reuse
packed parameters. Native output allocation/assembly is included in every method.

Across case medians, source generation takes 0.011–0.069 s, optional import
0.108–0.160 s and eager native compilation 9.72–10.43 s. Total optional setup is
9.89–10.60 s. Its estimated break-even is roughly 170000–186000 validated RHS
calls; repacking all inputs and returning shaped derivatives raises it to
329,133–377,769 calls. These estimates divide measured optional setup by measured per-call
savings. Common model normalization/compiler setup is recorded separately and
excluded from both optional overhead and savings. First calls are separately
recorded after eager compilation.

NumPy-only peak process RSS is 54.7–58.8 MiB across case medians; native-worker
peak RSS is 264.0–270.7 MiB. This path increases peak process memory by roughly
210 MiB on this host. It is unsuitable as a default memory reduction. These are
process high-water marks, not allocator counts or a complete application memory
profile. Oracle errors across all layouts remain below 1.91e-6 in float32;
precision-specific checks and the independent finite-difference gain direction
pass before timing.

Keep this optional for sufficiently repeated CPU RHS use. The receipt does not
measure a compiled solver loop, complete integration, automatic differentiation,
GPU execution or SBI training. JAX whole-loop execution is a separate comparison.

To reproduce one fresh native worker after installing `.[numba,benchmark]`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0 python -m benchmarks.numba_rhs \
  --groups 16 --stages 8 --dtype float64 --coupled --native \
  --output native-g16-s8.json
```

Choose an allowed CPU from your affinity set. Omit `--native` for the NumPy-only
worker. Use the receipt's outer-worker plan, alternating mode/case order, three
fresh repetitions and 180 s worker timeout to reproduce the campaign. Timing and
hardware variation belong in an application adoption decision.
