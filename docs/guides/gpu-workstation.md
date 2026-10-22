# GPU workstation handoff

**Device execution is on hold at the maintainer's request.** These commands are
prepared for a later workstation session. Do not run GPU jobs, including CuPy,
Torch or JAX device tests, until that hold is lifted. The new CuPy contract in
[PR #339](https://github.com/ACCIDDA/op_system/pull/339) remains a draft; green CPU
CI does not qualify it. No package publication or consumer adoption is part of
this procedure.

## Source and isolated environments

Use a new Linux workstation directory, Python 3.12, `uv`, Git and authenticated
GitHub CLI. Leave existing application environments and frozen runs alone. The
workstation's GPU model, architecture, driver and available memory must be
recorded before choosing CUDA wheels. The earlier A100 receipts do not qualify a
different GPU. Installation success alone is insufficient.

| Repository | Reviewed source for this handoff |
| --- | --- |
| op_system | Exact current head of draft PR #339, including its new contract and this guide; record the full head before execution. |
| op_engine | `2b8d46a2b7a6c3ee2645c4d595be7d224c52433a` (merged sensitivities/native eager state fixes). |
| op_infer | `fc11f1a2e8388547dc9cd873cc4203c99d19f287` (merged CUDA statistics and policy support). |

```sh
set -euo pipefail
mkdir op-gpu-qualification
cd op-gpu-qualification
OP_GPU_ROOT="$PWD"
mkdir source wheels evidence
nvidia-smi -q > evidence/nvidia-smi-before.txt
uname -a > evidence/platform.txt
uv --version > evidence/uv-version.txt

gh repo clone ACCIDDA/op_system source/op_system
(cd source/op_system && gh pr checkout 339)
gh repo clone ACCIDDA/op_engine source/op_engine
git -C source/op_engine checkout --detach 2b8d46a2b7a6c3ee2645c4d595be7d224c52433a
gh repo clone ACCIDDA/op_infer source/op_infer
git -C source/op_infer checkout --detach fc11f1a2e8388547dc9cd873cc4203c99d19f287
for repo in op_system op_engine op_infer; do
    git -C "source/$repo" rev-parse HEAD > "evidence/$repo-commit.txt"
    git -C "source/$repo" status --porcelain > "evidence/$repo-status.txt"
    uv build --directory "source/$repo" --wheel --out-dir "$OP_GPU_ROOT/wheels"
done
sha256sum wheels/*.whl > evidence/wheels.sha256
```

Verify that all three status files are empty. Save the PR head in the final
receipt; a moving branch name is not an immutable result. Use the built wheels,
not editable installs. The tests/examples below come from those exact checkouts.
Set `PYTHONPATH` to an empty value for their execution, keeping the installed
wheel ahead of any checkout's source directory. Record each environment with
`uv pip freeze --python <environment>/bin/python` and `python -VV`.

Use separate environments for CuPy, JAX CUDA and Torch CUDA. The commands below
select the previously exercised CUDA 12 families for CuPy/JAX. If the workstation
needs another CUDA family or wheel architecture, select it using the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html),
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html), or
[Torch installer](https://pytorch.org/get-started/locally/), and retain that
change as a separate qualification lane. Do not mix CuPy distribution families.
Do not reuse the CPU Torch wheel index for CUDA qualification.

## CuPy: required current and minimum dependency lanes

```sh
uv venv --python 3.12 env-cupy-current
uv pip install --python env-cupy-current/bin/python \
    'cupy-cuda12x[ctk]==14.2.0' 'numpy==2.5.4' \
    'array-api-compat==1.15.0' 'pytest>=8.4.2' wheels/*.whl
uv venv --python 3.12 env-cupy-minimum
uv pip install --python env-cupy-minimum/bin/python \
    'cupy-cuda12x[ctk]==14.2.0' 'numpy==2.0.2' \
    'array-api-compat==1.12.0' 'pydantic==2.12.5' 'pytest>=8.4.2' wheels/*.whl
```

`[ctk]` explicitly installs the optional user-space toolkit; omit that extra only
when the workstation supplies a compatible toolkit, recording the difference.
Each lane needs **10 passed, zero skipped, zero failures**:

```sh
for lane in current minimum; do
    uv pip freeze --python "env-cupy-$lane/bin/python" > "evidence/cupy-$lane-freeze.txt"
    "env-cupy-$lane/bin/python" -VV > "evidence/cupy-$lane-python.txt"
    (cd source/op_system && PYTHONPATH= CUDA_VISIBLE_DEVICES=0 \
        "$OP_GPU_ROOT/env-cupy-$lane/bin/python" scripts/qualify_backend.py cupy \
        --junitxml "$OP_GPU_ROOT/evidence/cupy-$lane.xml") \
        > "evidence/cupy-$lane.log" 2>&1
    # A failed required command stops this shell; preserve its failed attempt.
    PYTHONPATH= CUDA_VISIBLE_DEVICES=0 "env-cupy-$lane/bin/python" \
        source/op_system/examples/cuda_core_contract.py --backend cupy \
        > "evidence/cupy-$lane-core.json" 2> "evidence/cupy-$lane-core.stderr"
done
```

The companion core program requires **4 precision/history cases** and explicitly
records CuPy autodiff as unsupported. Save `cupy.show_config()` and
`cupy.cuda.runtime.getDeviceProperties(0)` output for each lane as well as
`driverGetVersion()` and `runtimeGetVersion()`. No CPU fallback or optional test
skip counts as success. Keep PR #339 draft and issue #182 open until both new
10-case contracts pass and the device receipt is reviewed.

## JAX CUDA and Torch CUDA: separate reruns

The committed statistical fixture is in op_infer; its
[existing A100 receipts and scope](https://github.com/ACCIDDA/op_infer/blob/fc11f1a2e8388547dc9cd873cc4203c99d19f287/docs/cuda-qualification.md)
remain historical evidence, not workstation results.

```sh
uv venv --python 3.12 env-jax-cuda
uv pip install --python env-jax-cuda/bin/python \
    'jax[cuda12]==0.12.0' 'numpyro==0.22.0' 'numpy==2.5.4' \
    'scipy==1.18.1' wheels/*.whl
PYTHONPATH= CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false env-jax-cuda/bin/python \
    source/op_system/examples/cuda_core_contract.py --backend jax \
    > evidence/jax-core.json 2> evidence/jax-core.stderr
PYTHONPATH= CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false env-jax-cuda/bin/python \
    source/op_infer/examples/cuda_statistics.py --backend jax \
    > evidence/jax-statistics.json 2> evidence/jax-statistics.stderr

uv venv --python 3.12 env-torch-cuda
# The previously exercised Linux PyPI distribution supplied its CUDA runtime.
# Verify the workstation architecture/driver against the official installer first.
uv pip install --python env-torch-cuda/bin/python 'torch==2.14.1' \
    'numpy==2.5.4' 'scipy==1.18.1' wheels/*.whl
PYTHONPATH= CUDA_VISIBLE_DEVICES=0 env-torch-cuda/bin/python \
    source/op_system/examples/cuda_core_contract.py --backend torch \
    > evidence/torch-core.json 2> evidence/torch-core.stderr
PYTHONPATH= CUDA_VISIBLE_DEVICES=0 env-torch-cuda/bin/python \
    source/op_infer/examples/cuda_statistics.py --backend torch \
    > evidence/torch-statistics.json 2> evidence/torch-statistics.stderr
```

Each core receipt must contain **4 passed float32/64/history cases**; final-only
JAX/Torch cases include native first gradients. Each statistical receipt needs
**24 passed family/precision cases plus the masked/censored gradient check**.
The fixture checks 8,192 addressed draws per observation case, independent
references and replay. Torch also checks local/global RNG behavior. CuPy
statistical interfaces remain unsupported. These commands do not train an
estimator and do not qualify the separate CPU-only Torch compiled-loop adapter
on GPUs.

Before running cases, retain the native configuration in each separate lane:

```sh
for lane in current minimum; do
    CUDA_VISIBLE_DEVICES=0 "env-cupy-$lane/bin/python" -c \
        'import cupy as c; c.show_config(); print(c.cuda.runtime.getDeviceProperties(0)); print(c.cuda.runtime.driverGetVersion(), c.cuda.runtime.runtimeGetVersion())' \
        > "evidence/cupy-$lane-runtime.txt" 2>&1
done
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
    env-jax-cuda/bin/python -c \
    'import jax; print(jax.__version__, jax.default_backend(), jax.devices())' \
    > evidence/jax-runtime.txt 2>&1
CUDA_VISIBLE_DEVICES=0 env-torch-cuda/bin/python -c \
    'import torch as t; assert t.cuda.is_available(); print(t.__version__, t.version.cuda, t.cuda.get_device_name(0), t.cuda.get_device_capability(0), t.cuda.get_arch_list())' \
    > evidence/torch-runtime.txt 2>&1
for lane in jax-cuda torch-cuda; do
    uv pip freeze --python "env-$lane/bin/python" > "evidence/$lane-freeze.txt"
    "env-$lane/bin/python" -VV > "evidence/$lane-python.txt"
done
nvidia-smi -q > evidence/nvidia-smi-after.txt
```

## Evidence, failures and completion

Save exit status, complete stdout/stderr, Python/frozen dependency versions,
actual device/driver/runtime/build information, Git commits, wheel/fixture
hashes, JUnit records and JSON receipts. Record Torch `version.cuda`, device
capability/architecture support and peak allocated/reserved bytes. Save JAX
backend/devices and preallocation settings. Process RSS includes imports and CUDA
contexts and must remain distinct from device allocator memory.

Run lanes sequentially on an otherwise idle device. These are correctness and
compatibility checks; timing/default-selection studies require a separate
protocol with synchronized cold/first/warm phases and memory measurements. Keep
every failed installation or test attempt, label any corrected rerun and retain
its exact source/environment delta. Do not overwrite failed evidence or weaken
numerical thresholds after seeing a result. Native import checks and CPU CI alone
cannot close a device gate.

After the hold is lifted and results are reviewed, attach a sanitized receipt to
the relevant issues and update the qualification guides. Omit local account,
filesystem and environment-secret details from public artifacts. All release,
application-adoption and frozen-run decisions remain separate.
