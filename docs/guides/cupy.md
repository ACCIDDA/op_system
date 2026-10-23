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

The maintainer-authorized workstation run on 23 October 2026 used an NVIDIA
GeForce RTX 5070 Ti Laptop GPU (Blackwell, compute capability 12.0), driver
580.173.02 and Python 3.12.12. Its exact pre-evidence-update compiler head was
`0c1306e5f3276902d2d75e76a12032a23797d103`; the pinned engine and inference
heads were `76edf594f5d4462f9c359482378e662e58668234` and
`713a2b76119e174dab75290eaf5bff9ffce63af6`. All sources were acquired from
their GitHub remotes, checked out detached and clean, built as wheels and
imported only from their isolated environments.

CuPy 14.2.0 passed all ten cases with zero skips or failures in both the current
and minimum dependency lanes. Each CuPy lane and the separate JAX 0.12.0 and
Torch 2.14.1+cu130 lanes passed all four float32/float64 and retained-history
core cases. JAX and Torch each passed 14 observation density/replay cases, ten
prior-density cases and the masked/censored gradient check. Torch additionally
preserved local/global RNG isolation and rejected a CPU generator for CUDA
sampling. The recorded validator confirmed every required case and all 40
recorded commands exited successfully.

The workstation was compute-idle rather than headless. Only the display server,
desktop shell and active editor held graphics contexts; there was no competing
compute context and about 11.1 GiB of device memory was free. Instantaneous GPU
utilization was 51% before and 10% after the run. These functional checks have
no timing threshold. Whole-process peak RSS was 450.4/452.0 MiB for the two
CuPy core lanes, 1001.9/1224.1 MiB for JAX core/statistics and 732.8/1353.6 MiB
for Torch core/statistics. Torch statistics peaked at 0.752 MiB allocated and
2.0 MiB reserved device memory. Those process and allocator measures have
different scopes and are not application or training-memory results.

This evidence update changes the PR head, so its results are not the final merge
receipt. The [workstation handoff](gpu-workstation.md) requires a fresh run and
independent review of the final head; the sanitized final receipt belongs on
PR #339 and issue #182, while raw machine records remain private. A repeated,
nonfatal JAX `hwloc` topology warning is retained with those raw records. CPU
skips cannot satisfy this device gate, and neither this bounded receipt nor its
rerun qualifies performance, estimator training, compiled Torch GPU loops,
Diffrax GPU execution, multiple devices, CUDA 13 CuPy wheels or consumer
adoption.
