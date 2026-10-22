# Native state derivative actions

`state_actions(compiled, backend="jax" | "torch", observable=None)` adapts a
compiled callable to native forward JVP and reverse/transpose VJP operations.
It requests products directly through
[JAX JVP](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) and
[VJP](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html), or
[Torch JVP](https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html) and
[VJP](https://docs.pytorch.org/docs/stable/generated/torch.func.vjp.html).
There is no requested numerical Jacobian matrix and no automatic backend choice.
Install the selected optional extra, `op_system[jax]` or `op_system[torch]`.

```python
import jax
import jax.numpy as jnp
from op_system import compile_spec, state_actions

compiled = compile_spec({
    "kind": "expr",
    "state": ["x", "z"],
    "equations": {"x": "-a*x+b*z", "z": "a*x-b*z"},
    "observables": {"measurement": {"expr": "gain*x+z"}},
})
rhs = state_actions(compiled, backend="jax")
measurement = state_actions(compiled, backend="jax", observable="measurement")
y = jnp.asarray([0.7, 1.2], dtype=jnp.float32)
v = jnp.asarray([0.2, -0.3], dtype=y.dtype)
a, b, gain = (jnp.asarray(value, dtype=y.dtype) for value in (0.4, 0.6, 1.5))
forward = jax.jit(rhs.jvp)(0.0, y, v, a=a, b=b)
reverse = rhs.vjp(0.0, y, jnp.asarray([-0.4, 0.6], dtype=y.dtype), a=a, b=b)
measured = measurement.jvp(0.0, y, v, gain=gain)
backprojected = measurement.vjp(0.0, y, jnp.asarray(0.6, dtype=y.dtype), gain=gain)
assert forward.shape == reverse.shape == y.shape
assert measured.shape == ()
assert backprojected.shape == y.shape
```

For an RHS `f`, `jvp` returns `J_f v` and `vjp` returns `J_f.T w`. A named
observable uses its own Jacobian: its tangent output and reverse cotangent have
the exact declared observable shape, including `()` for a scalar and the full
axis shape for an array. Reverse output always has the flat current-state shape.
Parameters and time remain fixed within each state derivative. These products
are distinct from parameter sensitivities and differentiation through an ODE solve.

The adapter retains structural source identity, natural `state_names`, exact
target `parameter_names` and `output_shape`. The RHS requires its own inputs;
an observable requires its declared inputs, including observer-only parameters.
Shaped buffers and linear/previous time-table interpolation use the existing
compiled evaluator. No parameter values, states or reverse tapes are cached by
the adapter. Native compiler caches and caller-retained derivative graphs have
their own lifetimes.

## Qualified scope

CPU tests cover float32/64 nonlinear arithmetic/exp, aliases, shaped buffers,
global reductions, scalar/array/constant observables, time-table state actions,
transpose duality and invalid layouts. Independent analytic derivatives and
the existing bounded dense artifact provide reference values. JAX `jit`/`vmap`,
Torch `vmap`, and outer state/parameter gradients through nonlinear JVP and VJP
are checked. These fixtures do not qualify arbitrary higher-order derivatives
or every operation accepted by the compiler.

State, direction and parameter inputs must be native float32/64 arrays with
matching precision, and Torch devices must match. Python real scalar time is
accepted, including weak scalar time under JAX `jit`; explicitly typed native
time must otherwise match precision. Parameter shapes retain compiler checks.
JAX float64 requires the caller's x64 setting. The factory imports an optional
backend only when explicitly requested. Unsupported native transforms propagate
backend errors. Nonfinite/domain failures retain native propagation; successful
construction does not certify a valid differentiation point.

Operators, history and gauge-fixed execution are explicitly rejected, including
when targeting an observable. The adapter does not silently omit their possible
contributions or substitute a dense implementation. GPU execution and Torch
compiler capture remain outside this action qualification. Existing
[`symbolic_derivatives`](portable-ir.md) and its dense actions remain available
with their original contract, including NumPy use.

## Storage and measured CPU costs

Direct products avoid asking for a state-by-state numerical matrix. AD tapes,
native compiler graphs, source expansion and runtime caches still consume
memory. The nonlinear 16-cell local-ring graph probe has only scalar/vector
intermediates in the inspected JAXPR and Torch FX graphs; this is a fixture-level
check, not a universal storage or process-memory bound. Global reductions and
nonlocal expressions can have different costs.

The existing `examples/benchmark_symbolic.py` harness compares four equal state
products on the same linear ring and scalar observable. The dense baseline
evaluates its four existing matrices **once**, then applies them. Its API also
computes parameter matrices which the direct state actions do not request. JAX
JIT is used for both implementations; Torch uses eager native execution. This
linear fixture permits constant-derivative compiler optimizations and does not
predict nonlinear or full-solver performance.

[Worker receipts](../assets/native-actions-cpu/workers.json) retain 16 successful
fresh serial CPU workers: two per backend/size/implementation, reversing order
in the second repetition, with affinity and numerical thread budgets of one.
Each worker validates independent analytic products before reporting a median
of 11 warm evaluations. The table takes medians across the two workers.
The earlier campaign is retained in
[a separate receipt](../assets/native-actions-cpu/workers-before-integer-time-fix.json);
the table uses only the final source/harness campaign.

| Backend | States | Path | Construction ms | First evaluation ms | Warm ms | Peak RSS MiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| JAX 0.12.0 | 32 | Dense | 201.0 | 576.9 | 0.049 | 243.2 |
| JAX 0.12.0 | 32 | Direct | 94.9 | 1087.4 | 0.058 | 266.8 |
| JAX 0.12.0 | 128 | Dense | 1277.5 | 12086.8 | 0.086 | 299.3 |
| JAX 0.12.0 | 128 | Direct | 100.4 | 5366.1 | 0.157 | 393.3 |
| Torch 2.14.1 CPU | 32 | Dense | 166.8 | 12.5 | 4.398 | 245.4 |
| Torch 2.14.1 CPU | 32 | Direct | 144.4 | 2172.6 | 26.454 | 319.0 |
| Torch 2.14.1 CPU | 128 | Dense | 1180.2 | 73.9 | 64.195 | 275.6 |
| Torch 2.14.1 CPU | 128 | Direct | 68.8 | 2087.6 | 88.874 | 320.6 |

Construction excludes source compilation, which is retained separately. First
evaluation includes lazy AD initialization and JAX compilation. Peak RSS covers
the entire worker, including imports and native compilation. Both paths return
`3*n+1` entries; the dense API constructs `(n+1)*(n+3)` matrix entries before
native compiler optimization. This is logical cardinality, not measured allocated
bytes. Warm timings vary across workers, particularly Torch dense evaluation;
inspect the retained samples before extrapolating.

Direct construction is cheaper at 128 states in this experiment, but direct
actions have **higher peak RSS and slower median warm evaluation in every final
comparison**.
They are an explicit building block for consumers requiring derivative products,
not a general CPU speed or memory improvement. No defaults change. A consumer
must measure its complete solve, derivative requirements and retained graphs
before adoption. These advisory measurements impose no CI timing thresholds.

Reproduce individual cases in fresh processes with explicit affinity/thread
settings, for example:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORM_NAME=cpu \
  taskset -c 0 python examples/benchmark_symbolic.py \
  --backend jax --states 128 --operation actions --implementation native
# Repeat with --implementation symbolic, and separately --backend torch.
# Select a logical CPU permitted by your process affinity.
```
