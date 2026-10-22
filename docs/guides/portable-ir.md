# Portable IR and bounded symbolic derivatives

`export_ir(compiled)` returns a `PortableIR` envelope tied to the existing
`StructuralModelIdentity`. It preserves the normalized typed DAG, ordered states
and parameters, axes/coordinates, constants, aliases, observables, known-input
metadata and declarations. It does not expand lazy per-cell equations or serialize
fitted arrays, executable closures or a Python pickle.

```python
import numpy as np
from op_system import (
    PortableIR,
    compile_spec,
    export_ir,
    export_julia,
    symbolic_derivatives,
)

compiled = compile_spec({
    "kind": "expr",
    "state": ["x", "z"],
    "aliases": {"loss": "a*x", "net": "loss-b*z"},
    "equations": {"x": "-net", "z": "net"},
    "observables": {"measurement": {"expr": "gain*x+z+t"}},
})
interchange = export_ir(compiled)
restored = PortableIR.from_json(
    interchange.to_json(),
    expected=interchange.identity,
)
assert restored == interchange
assert restored.decode()["state_names"] == ("x", "z")

artifact = symbolic_derivatives(compiled)
assert artifact.parameter_names == ("a", "b", "gain")
matrices = artifact.evaluate(0.5, np.array([2.0, 4.0]), a=0.3, b=0.7, gain=3.0)
assert np.allclose(matrices.state, [[-0.3, 0.7], [0.3, -0.7]])
assert np.allclose(matrices.observable_parameters, [[0.0, 0.0, 2.0]])

julia = export_julia(compiled)
assert julia.parameter_names == artifact.parameter_names
assert julia.observable_names == ("measurement",)
assert julia.rational
```

Supply an independently expected identity when reading a saved export. Import
checks the canonical manifest's bytes against that identity and rejects unknown
schemas, extra envelope fields, duplicate JSON keys and malformed/cyclic node
references. Decoding returns readonly mappings, tuples and `IRRecord` data; it
does not import or instantiate the named record classes. A content identity does
not authenticate a source or establish an external tool's mathematical result.

## Interchange schema

The outer schema has exactly `format="op-system-external-ir"`, integer
`schema_version=1`, `source` and `manifest`. `source` is the existing schema-1
structural identity. `manifest` is the canonical `op-system-structural-model`
DAG with `nodes` and a zero-based `root`. Each node contains a type tag, scalar or
field labels, and references to earlier nodes. This preserves sharing and gives
non-Python consumers an explicit data format.

| Tag | Meaning |
| --- | --- |
| `none`, `bool`, `str` | Typed scalar values, without coercing booleans into integers. |
| `int` | Decimal integer text. |
| `float` | Exact Python hexadecimal floating-point text, including explicit metadata infinities where declared. |
| `sequence` | Ordered child references. |
| `mapping` | Named children in canonical key order. |
| `record` | Compiler record type name and ordered field names, interpreted as data. |

Expression records include `Literal`, `Sym`, `Apply`, `Subscript`, `Reduce` and
`HistoryOp`; consumers declare which operations they implement. Complete
observable definitions and measures remain in the normalized metadata, beside
the ordered observable names. `t` is the known independent time variable;
time-varying inputs and unknown scalar parameters have distinct declarations.
Initial-condition declarations remain metadata. A runtime consumer must supply
its initial vector in the exported state order and retain any initial-condition
parameterization separately.

Import has a 32 MiB UTF-8 envelope limit. Full compact-IR export is separate from
bounded scalar evaluation/emission: preserving an operator or reduction as data
does not claim that the scalar Julia emitter implements it.

## Native derivative artifacts

`symbolic_derivatives` supplies ordered expressions and native evaluation of
`df/dy`, `df/dtheta`, `dg/dy` and `dg/dtheta`. Observable-only parameters are
included explicitly. Values are a native named tuple, suitable for JAX JIT; no
host snapshot or graph detachment occurs. NumPy, JAX and CPU Torch float32/64 are
the qualified array cases. CUDA and other devices need separate qualification.

The initial supported subset is bounded scalar real arithmetic, literal powers,
exp/log/sqrt, sin/cos and tanh, including transitive aliases. Evaluate at points
where the original expression is defined and differentiable. Shaped parameter
buffers, history, numerical operators, gauge-fixed execution, variable powers,
nonsmooth operations and reduced/vector observables fail explicitly. General
compact IR can still preserve those constructs for another consumer.

Default limits are 128 expanded states and 65,536 dense derivative entries.
`jvp` and `vjp` apply that bounded dense state Jacobian; they do not claim
matrix-free storage. Tangent layouts, namespace and precision must match the
states. General sparse and operator-aware derivative exports remain separate work.

`state_dependencies` is conservative IR dependency metadata, independent of
current fitted values. A coefficient becoming zero does not remove its declared
state dependency. `block_axes` repeats the compiler's existing checked block
axes; no second separability test is introduced. Neither field proves that a
statistical objective factorizes or that parameters are globally identifiable.

A parameter-sensitivity solve additionally needs initial-condition derivatives
and the observation map. Its measured-output sensitivity is `dg/dy @ S +
dg/dtheta`; an RHS state Jacobian alone is not that map or a Fisher matrix.

### Representative construction and evaluation costs

`examples/benchmark_symbolic.py --backend jax --states 32` measures a linear
ring with three ordered parameters and one scalar measurement, checking all four
matrices against an independent analytic reference. Run each backend/size in a
fresh process; the retained [raw receipt](../assets/symbolic-costs-2026-10-21.json)
covers 2, 16 and 32 states on a shared Intel Core Ultra 7 265U, one CPU affinity
and native thread limits of one. These are advisory measurements, with one worker
per case and the median of seven synchronized warm evaluations.

| 32-state float64 case | Derivative construction | First evaluation | Warm evaluation |
| --- | ---: | ---: | ---: |
| NumPy | 42 ms | 91 ms | 2.31 ms |
| JAX | 113 ms | 1.41 s | 0.291 ms |
| CPU Torch | 54 ms | 13 ms | 3.90 ms |

The artifact contains 1,155 dense entries. First evaluation includes lazy
initialization and, for JAX, JIT compilation; construction follows source model
compilation, which is recorded separately. Whole-worker peak RSS includes imports
and source compilation. This fixture neither selects a native compiler rewrite
nor predicts application performance or the cost of a sensitivity solve.

Optional [native state actions](native-actions.md) provide a separate direct
JAX/Torch product boundary, including shaped parameters and array observables.
They preserve the dense API above; their transform, storage and performance
qualification is documented separately.

## Julia expression export

`export_julia` returns a `JuliaExport` record with source identity, emitter schema
version, ordered state/parameter/observable names, a rational-operation flag and
`code`. The code defines out-of-place `op_system_rhs(u, p, t)` and
`op_system_observations(u, p, t)`. State/parameter names become explicit array
indices, not executable identifier interpolation. The caller supplies `u0` and
`p` in the retained orders; there is no implicit initialization, prior or forcing
conversion. `julia_rhs` is the code-only convenience.

The same bounded scalar IR supplies the Python derivative artifacts and Julia
expressions. Non-rational supported functions retain their actual semantics and
set `rational=False`; no rational-tool compatibility is inferred. This is optional
interoperability, with no Julia runtime dependency or full Julia package port.

External structural tools still need their exact versions, rank definitions,
known-input assumptions, witness points/tolerances, disagreements and censored
timeouts retained. Shared-source export removes a hand-transcription step; it
does not settle a disputed rank or make different rank notions equivalent.

## Execute the independent Julia check

From a source checkout with a Julia executable already available:

```shell
uv run --no-sync python examples/portable_julia.py --julia julia
```

The standalone example emits RHS and measurement code from one compiled model,
then compares both with Python evaluation at three independently specified
points. It also compares all four derivative matrices with independent central
differences executed in Julia (absolute step `1e-5`, tolerance `1e-8`). Julia
1.12.6 matched all six value rows exactly and passed every derivative comparison
in the local qualification. This is numerical expression conformance, not
external structural-rank evidence.
The program retains model/emitter/runtime identities and uses a temporary depot
when none is explicitly supplied.
