# Compiler backend qualification

`op_system` selects its numerical namespace from the input arrays. Optional
runtime extras install JAX or Torch; installing the compiler alone requires
neither. Root compiler qualification is a separate required CI matrix from the
provider's tests and the baseline compiler test suite.

The required jobs explicitly install NumPy, JAX, or CPU Torch and execute the
matching tests through `scripts/qualify_backend.py`. The runner fails if the
backend is missing, if no tests run, or if a selected test is skipped. Ordinary
optional-backend skips in the baseline suite therefore cannot substitute for
these jobs.

## Qualified capabilities

The CI jobs run on Python 3.12 with the dependencies resolved for that run;
their logs record backend and Python versions. Baseline package tests separately
cover supported Python 3.11–3.13. Qualification covers float32 and float64
values, with JAX x64 enabled explicitly for double-precision checks. State and
floating parameters use the same dtype; mixed-precision combinations remain
unqualified.

| Capability | NumPy | JAX | CPU Torch |
| --- | --- | --- | --- |
| Flat and structured/PyTree RHS parity, shapes, dtypes, namespace | Qualified | Qualified | Qualified |
| Independent-group block RHS | Qualified, eager loop | Qualified, eager loop and native vmap | Qualified, eager loop and `torch.func.vmap` |
| Coupled-group flat/structured RHS | Qualified | Qualified | Qualified |
| Treating coupled groups as independent blocks | Rejected | Rejected | Rejected |
| Native JIT and batched parameter evaluation | No AD/JIT contract | Qualified with `jit`/`vmap` | Torch compiler integration unqualified |
| Directional derivatives against independent finite differences | No AD contract | Qualified | Autograd qualified in float64 |
| CUDA/CuPy and other GPU execution | Unqualified here | Unqualified here | Unqualified here |

GPU qualification remains tracked separately in #182. The matrix describes
the tested workload and paths, not every combination of compiler operators,
backend versions, devices, or automatic differentiation transforms.

## Independent equations and structural checks

`benchmarks/hierarchical_model.py` defines a synthetic closed three-compartment
relaxation network. State and a local rate vary by group, while a global gain,
stage recovery rates, and a stage transport matrix are shared. Aliases,
stage reductions, and an `apply_along` transport expression exercise axis
reordering and broadcasting. The reference uses explicit cell loops, not
generated code or expressions copied from the compiler IR.

The coupled variant uses other groups' state through a group reduction. It
matches the same independently written equations, and an attempted
`factorize_axes: [group]` fails. Native block mapping is tested only for the
independent variant, slicing local parameters while sharing global ones.

Transport is part of this fixture's equations. Operator descriptors intended
for a downstream integration engine are a separate interface; compilation
does not automatically apply every metadata operator to an RHS.

JAX checks JIT, batch evaluation, and shared/local parameter directional
derivatives against finite differences of the independent reference. Torch
checks autograd through state and shared/local parameters, with a reference
finite-difference check. A closed RHS has conserved totals, so a squared-RHS
objective is used to avoid an identically zero gradient from summing all flows.

## Local commands

From a clean checkout with `uv` and `just` installed:

```sh
just qualify-numpy
just qualify-jax
just qualify-torch
# Or run all three required core qualifications:
just qualify
```

The Torch target installs CPU wheels explicitly. These commands change the
selected extras in the local environment; use separate worktrees/environments
when retaining multiple dependency selections matters. They run the same
qualification runner as CI. Normal quality, documentation, baseline tests,
provider checks, and wheel validation remain required alongside qualification.

Scientific validity, agreement across untested parameter regimes, and solution
accuracy of a downstream ODE/PDE integrator require separate evidence.

Literal RHS values, including zero and aliased constants, inherit the qualified
floating state's precision during assembly. Flat, structured, and available
block evaluation preserve dtype and namespace. Explicitly typed array-valued
parameters retain their own precision; this rule does not coerce deliberately
mixed-precision input arrays into the state dtype. Constant forcing has a zero
state Jacobian under the tested JAX and Torch functional AD transforms.
