# CuPy device qualification

CuPy is an optional runtime array namespace. Normal compiled evaluators discover
it from CuPy state arrays; there is no new compiler backend flag or provider
setting. The compiler's required CPU NumPy/JAX/Torch matrix and the optional
Numba CPU lowering remain separate contracts.

## Install one CUDA family explicitly

From this checkout, `uv sync --dev --extra cupy` installs
`cupy-cuda12x>=14.2,<15`. This extra deliberately selects the CUDA 12 wheel family.
It leaves CUDA runtime/toolkit installation to the environment, keeping those
large components optional. If the system does not supply them, explicitly install
`cupy-cuda12x[ctk]` in the same environment. Use one CuPy distribution per
environment; the CUDA 13 wheel family is a separate installation and qualification
choice. See the [official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

A GPU and compatible NVIDIA driver are required for execution. Package import or
successful dependency installation alone does not qualify a device. Runtime
state and floating parameter inputs in these tests are native CuPy arrays with
one declared dtype. Use `cupy.asarray` to construct them; CuPy scalar dtype aliases
are NumPy scalar types, so a dtype constructor alone does not establish GPU
storage. See [CuPy's documented differences](https://docs.cupy.dev/en/stable/user_guide/difference.html).

## Required device command and optional CPU skips

```sh
just qualify-cupy
# In an already prepared environment, retain the executed-case report:
python scripts/qualify_backend.py cupy --junitxml cupy-results.xml
```

The required command checks CUDA initialization and a real device operation before
pytest, then requires executed cases with zero skips. Missing CuPy, missing or
broken CUDA, missing devices, failed tests and skipped cases are failures. It
cannot report CPU fallback as qualified CuPy execution.

The baseline suite instead uses an explicitly optional device fixture. Without
CuPy or hardware it reports ten skips, with reasons. Standard hosted jobs do not
install this optional GPU namespace or require CUDA devices; their green status
is not CuPy coverage. Run the required command on suitable hardware and retain
its versions/device metadata, source/wheel hashes, logs and JUnit records.

## Bounded contract

Ten cases exercise both float32 and float64. They reuse a compiled artifact for
NumPy and CuPy calls and check native GPU result storage, dtype, flat ordering,
structured shapes and input immutability. The hierarchical balance fixture has
independently written equations, shaped shared/local parameters, aliases, stage
transport/reductions and both independent and coupled groups. A strided GPU state
also follows the same flat layout. Additional cases check direct constants and
native scalar time, layout errors, and bounded symbolic state/parameter and
observable derivative values.

Host transfer is explicit in numerical assertions via `cupy.asnumpy`, after
native device/dtype checks. This checks compiler values and layouts on the
recorded device/runtime. It does not qualify every compiler operator, independent
block vectorization, sparse downstream solvers, multiple devices, CuPy automatic
differentiation, statistical distributions, SBI training or CUDA 13/ROCm. It makes
no GPU performance or memory-reduction claim.

## Device evidence

Device receipts are added only after the required command completes on actual
hardware. CPU skips cannot satisfy this section's device acceptance gate.
