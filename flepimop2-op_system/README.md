# flepimop2-op_system

`flepimop2-op_system` provides the `flepimop2` system adapter for `op_system`.

It packages the `flepimop2.system.op_system` provider so `flepimop2` can load and execute RHS specifications compiled by the core `op_system` package.

Bound system steppers accept and return Flepimop2's namespace-neutral `Array`
protocol. The provider performs no state conversion: NumPy, JAX, and other
supported array namespaces are selected by the state supplied by the numerical
engine. `module: op_system` selects this system provider; it does not select an
array namespace. There is no `array_backend` configuration field. See the
[terminology guide](../docs/guides/array-namespaces.md) for legacy compiler
keyword compatibility.

The wheel ships `py.typed` for downstream type checking. Repository quality
checks run strict mypy on the adapter in its separate environment; the root
compiler's mypy exclusion does not bypass that check. `just build-all` and CI's
`provider-typing` job additionally check the installed wheel's public types and
verify that an invalid state argument fails strict downstream typing.

System options preserve normalized axis metadata for engine providers:
`axis_order`, `axis_sizes`, numeric `axis_coords`, original `axis_labels`, and
declared `axis_types`. Consumers can therefore distinguish categorical,
ordinal, and continuous semantics without inferring them from coordinate values.

Time-indexed parameter tables select interpolation through the inline spec's
`time_interpolation` field: `linear` is the default, and `previous` holds the
value at each coordinate until the next, including the new value at an exact
boundary. Both policies clamp beyond the endpoints. All bound flat, PyTree,
block, and reaction steppers share the policy and preserve the state's array
namespace.

The provider also exposes `time_axis`, `time_interpolation`, `time_coordinates`,
and `forcing_breakpoints` through `system.option(...)`. Coordinates and
breakpoints are immutable tuples. Hold-mode tables publish all coordinates
after the first as possible forcing changes; linear tables and unused time axes
publish no breakpoints. A single-coordinate time table is constant and has no
forcing changes. Parameter requests still include the complete declared time
axis, so the parameter producer supplies the full table at run time.

An engine can use these options to configure forcing boundaries independently
of its output grid. Exact breakpoint SSA also requires every other external
time dependency in the propensity to be constant between those boundaries.
The provider exposes the schedule; the engine chooses how to consume it.

It also provides the `sparse_table` parameter module, which assembles a dense
array for a routing transition's matrix parameter (for example
`eta[time, imm:i, imm:j]`) from a declared support. Each entry is a number or
any nested parameter configuration, sampled with the request's leading axes:

```yaml
parameter:
  eta:
    module: sparse_table
    indices: [imm, imm]
    entries:
      - index: [x0, x3]
        value: {module: fixed, value: 0.2}
      - index: [x5, x7]
        value: 1.0
```

Off-support entries are zero. `SparseTableParameter.support(axes)` returns
the support positions for inference.

For an `erfc` mixing kernel on numeric coordinates, declare `axes: [age, age]`
and `params: {scale: 1.0, sigma: 2.0}`. The provider constructs the float64 host
matrix `C[i, j] = scale * erfc(abs(age[i] - age[j]) / sigma)` once when the
system is built. It uses Python's [complementary error function](https://docs.python.org/3/library/math.html#math.erfc)
directly to preserve small positive tail values; SciPy is not a runtime
dependency. Kernel declarations without axes retain their existing scalar
`params.value` behavior.

The provider source headers grant GPL version 3 **or any later version**
(`GPL-3.0-or-later`). The standalone core uses MIT. The shared citation records
both existing grants; preserve the relevant license when reusing code.
