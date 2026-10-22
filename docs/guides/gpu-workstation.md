# GPU workstation handoff

## New agent: start here

**Device execution is on hold at the maintainer's request.** Read and prepare
without running GPU jobs or probes. A maintainer request to run this procedure
in a GPU workstation session lifts that hold for this qualification. Otherwise,
report readiness and wait. This includes CuPy, Torch and JAX device tests.

The goal is a reproducible device receipt for draft
[op_system PR #339](https://github.com/ACCIDDA/op_system/pull/339) /
[issue #182](https://github.com/ACCIDDA/op_system/issues/182), plus separate
JAX/Torch compiler, core-solver and statistical reruns. Keep #339 draft and #182
open until both CuPy lanes pass and an independent reviewer approves the exact
source and receipt. Green CPU CI cannot close this device gate.

Use a new Linux directory, Bash, Python 3.12, `uv`, Git and authenticated `gh`.
Run lanes sequentially on an otherwise idle NVIDIA GPU. Leave application
configurations, environments, frozen runs, simulation banks and checkpoints
alone. Do not publish packages, tags or releases; **op_infer must not be uploaded
to PyPI or TestPyPI**. This procedure does not train an estimator.

## 1. Capture exact source and preserve every command

After the device hold is lifted, run the following in one Bash session. Keep
that shell open: later sections use its variables and `record` function. The
function records the command, stdout, stderr and exit status separately; it
refuses to overwrite an attempt. A failing command stops the procedure. Keep
its evidence before diagnosing it and use a new attempt directory for a rerun.

```bash
set -euo pipefail
mkdir op-gpu-qualification
cd op-gpu-qualification
OP_GPU_ROOT="$PWD"
OP_GPU_ATTEMPT="$OP_GPU_ROOT/evidence/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir source wheels evidence
mkdir "$OP_GPU_ATTEMPT"

record() {
    local label="$1" status
    shift
    local prefix="$OP_GPU_ATTEMPT/$label"
    if compgen -G "$prefix.*" > /dev/null; then
        echo "Refusing to overwrite evidence: $prefix" >&2
        return 1
    fi
    printf '%q ' "$@" > "$prefix.command"
    printf '\n' >> "$prefix.command"
    if "$@" > "$prefix.stdout" 2> "$prefix.stderr"; then
        status=0
    else
        status=$?
    fi
    printf '%s\n' "$status" > "$prefix.exitcode"
    return "$status"
}

record hardware-before nvidia-smi -q
record platform uname -a
record uv-version uv --version
OP_GPU_SYSTEM_SHA="$(gh pr view 339 --repo ACCIDDA/op_system --json headRefOid --jq .headRefOid)"
printf '%s\n' "$OP_GPU_SYSTEM_SHA" > "$OP_GPU_ATTEMPT/requested-system-head.txt"
gh repo clone ACCIDDA/op_system source/op_system
git -C source/op_system fetch origin "$OP_GPU_SYSTEM_SHA"
git -C source/op_system checkout --detach "$OP_GPU_SYSTEM_SHA"
gh repo clone ACCIDDA/op_engine source/op_engine
git -C source/op_engine checkout --detach 76edf594f5d4462f9c359482378e662e58668234
gh repo clone ACCIDDA/op_infer source/op_infer
git -C source/op_infer checkout --detach 713a2b76119e174dab75290eaf5bff9ffce63af6
for repo in op_system op_engine op_infer; do
    git -C "source/$repo" rev-parse HEAD > "$OP_GPU_ATTEMPT/$repo-commit.txt"
    git -C "source/$repo" status --porcelain > "$OP_GPU_ATTEMPT/$repo-status.txt"
    test ! -s "$OP_GPU_ATTEMPT/$repo-status.txt"
    record "build-$repo" uv build --directory "source/$repo" --wheel --out-dir "$OP_GPU_ROOT/wheels"
done
test "$(git -C source/op_system rev-parse HEAD)" = "$OP_GPU_SYSTEM_SHA"
sha256sum wheels/*.whl > "$OP_GPU_ATTEMPT/wheels.sha256"
sha256sum source/op_system/examples/cuda_core_contract.py \
    source/op_system/scripts/qualify_backend.py \
    source/op_system/tests/op_system/test_cupy_contract.py \
    source/op_system/docs/guides/gpu-workstation.md \
    source/op_infer/examples/cuda_statistics.py > "$OP_GPU_ATTEMPT/fixtures.sha256"
OP_GPU_ENV=(env PYTHONPATH= PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0)
```

| Repository | Source boundary |
| --- | --- |
| op_system | The full PR #339 head captured above; never substitute a moving branch in the receipt. |
| op_engine | `76edf594f5d4462f9c359482378e662e58668234`: includes current fixed loops, Diffrax bridge and checked operator adapters. |
| op_infer | `713a2b76119e174dab75290eaf5bff9ffce63af6`: includes empty-bootstrap handling and committed CUDA statistical fixtures. |

Inspect `hardware-before.stdout`: GPU model, architecture, driver and available
memory determine wheel compatibility. Historical A100 receipts do not qualify a
different GPU. These commands select CUDA 12 wheels. Check the official
[CuPy](https://docs.cupy.dev/en/stable/install.html),
[JAX](https://docs.jax.dev/en/latest/installation.html) and
[Torch](https://pytorch.org/get-started/locally/) installation requirements before
installing. If the pinned wheels cannot support this hardware/driver, stop and
record the incompatibility; propose a separately recorded compatible lane.
Do not silently substitute CPU wheels or mix CuPy distribution families.

Use wheel installs, with no editable installs. Verify installed import origins
for **each** environment using this helper; both the environment prefix and the
source file locations are preserved. Run commands from `$OP_GPU_ROOT` except
where a subshell explicitly changes directory.

```bash
installed_imports() {
    record "$1-imports" "${OP_GPU_ENV[@]}" "$OP_GPU_ROOT/$1/bin/python" -c '
import importlib, pathlib, sys
prefix = pathlib.Path(sys.prefix).resolve()
for name in ("op_system", "op_engine", "op_infer"):
    path = pathlib.Path(importlib.import_module(name).__file__).resolve()
    print(name, path)
    assert prefix in path.parents, (name, path, prefix)
'
    record "$1-freeze" uv pip freeze --python "$1/bin/python"
    record "$1-python" "$1/bin/python" -VV
}
```

## 2. CuPy: current and minimum dependency lanes

```bash
record install-env-cupy-current uv venv --python 3.12 env-cupy-current
record install-cupy-current uv pip install --python env-cupy-current/bin/python \
    'cupy-cuda12x[ctk]==14.2.0' 'numpy==2.5.4' \
    'array-api-compat==1.15.0' 'pytest==8.4.2' wheels/*.whl
record install-env-cupy-minimum uv venv --python 3.12 env-cupy-minimum
record install-cupy-minimum uv pip install --python env-cupy-minimum/bin/python \
    'cupy-cuda12x[ctk]==14.2.0' 'numpy==2.0.2' \
    'array-api-compat==1.12.0' 'pydantic==2.12.5' 'pytest==8.4.2' wheels/*.whl
for lane in current minimum; do
    installed_imports "env-cupy-$lane"
    record "cupy-$lane-runtime" "${OP_GPU_ENV[@]}" "env-cupy-$lane/bin/python" -c \
        'import cupy as c; c.show_config(); assert c.cuda.runtime.getDeviceCount() > 0; print(c.cuda.runtime.getDeviceProperties(0)); print(c.cuda.runtime.driverGetVersion(), c.cuda.runtime.runtimeGetVersion())'
    (cd source/op_system && record "cupy-$lane-tests" "${OP_GPU_ENV[@]}" \
        "$OP_GPU_ROOT/env-cupy-$lane/bin/python" scripts/qualify_backend.py cupy \
        --junitxml "$OP_GPU_ATTEMPT/cupy-$lane.xml")
    record "cupy-$lane-core" "${OP_GPU_ENV[@]}" "env-cupy-$lane/bin/python" \
        source/op_system/examples/cuda_core_contract.py --backend cupy
done
```

`[ctk]` installs the user-space CUDA toolkit; a compatible NVIDIA driver is still
required. Omitting `[ctk]` for a workstation-provided toolkit changes the lane
and must be recorded. The current NumPy pin is a qualification candidate, not a
claim about CuPy's upstream supported-version matrix. The minimum lane fixes
the declared NumPy, array-api-compat and Pydantic floors, with current CuPy.

Each required contract must report **10 passed, zero skipped, zero failures**.
Each companion core receipt must contain **4 precision/history cases**. CuPy
native autodiff and op_infer statistical interfaces remain unsupported. Keep
PR #339 draft and issue #182 open pending independent device receipt review.

## 3. JAX CUDA and Torch CUDA reruns

The statistical fixture and its
[historical A100 qualification](https://github.com/ACCIDDA/op_infer/blob/713a2b76119e174dab75290eaf5bff9ffce63af6/docs/cuda-qualification.md)
are separate from this workstation's results. Configure the runtime **before**
the fixtures. JAX must select a GPU, and Torch must expose a compatible CUDA
build and architecture. Do not use the CPU Torch wheel index.

```bash
record install-env-jax uv venv --python 3.12 env-jax-cuda
record install-jax uv pip install --python env-jax-cuda/bin/python \
    'jax[cuda12]==0.12.0' 'numpyro==0.22.0' 'numpy==2.5.4' \
    'scipy==1.18.1' wheels/*.whl
installed_imports env-jax-cuda
OP_GPU_JAX=("${OP_GPU_ENV[@]}" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false)
record jax-runtime "${OP_GPU_JAX[@]}" env-jax-cuda/bin/python -c \
    'import jax; print(jax.__version__, jax.default_backend(), jax.devices()); assert jax.default_backend() == "gpu"'
record jax-core "${OP_GPU_JAX[@]}" env-jax-cuda/bin/python \
    source/op_system/examples/cuda_core_contract.py --backend jax
record jax-statistics "${OP_GPU_JAX[@]}" env-jax-cuda/bin/python \
    source/op_infer/examples/cuda_statistics.py --backend jax

record install-env-torch uv venv --python 3.12 env-torch-cuda
# This pinned Linux PyPI distribution previously supplied its CUDA runtime.
# Check the actual build and supported architecture; installation alone is insufficient.
record install-torch uv pip install --python env-torch-cuda/bin/python \
    'torch==2.14.1' 'numpy==2.5.4' 'scipy==1.18.1' wheels/*.whl
installed_imports env-torch-cuda
record torch-runtime "${OP_GPU_ENV[@]}" env-torch-cuda/bin/python -c \
    'import torch as t; assert t.cuda.is_available(); print(t.__version__, t.version.cuda, t.cuda.get_device_name(0), t.cuda.get_device_capability(0), t.cuda.get_arch_list())'
record torch-core "${OP_GPU_ENV[@]}" env-torch-cuda/bin/python \
    source/op_system/examples/cuda_core_contract.py --backend torch
record torch-statistics "${OP_GPU_ENV[@]}" env-torch-cuda/bin/python \
    source/op_infer/examples/cuda_statistics.py --backend torch
record hardware-after nvidia-smi -q
```

Each core receipt must contain **4 passed float32/64/history cases**. Final-only
JAX/Torch cases include native first gradients. Each statistical receipt must
contain **24 passed family/precision cases plus the masked/censored gradient
check**. The fixture checks 8,192 addressed draws per observation case, independent
references and replay. Torch additionally checks local/global RNG isolation and
reports peak allocated/reserved device bytes. Linux process peak RSS includes
imports and CUDA contexts; keep it distinct from device allocator memory.

## 4. Check receipts and report

The fixtures themselves enforce native GPU arrays and numerical checks. This
additional summary rejects incomplete receipts and checks exact case counts:

```bash
record receipt-check env OP_GPU_ATTEMPT="$OP_GPU_ATTEMPT" env-jax-cuda/bin/python - <<'PY'
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(os.environ["OP_GPU_ATTEMPT"])
for path in root.glob("*.exitcode"):
    assert path.read_text().strip() == "0", path
for lane in ("current", "minimum"):
    cases = ET.parse(root / f"cupy-{lane}.xml").findall(".//testcase")
    assert len(cases) == 10, (lane, len(cases))
    assert all(not any(c.tag in {"skipped", "failure", "error"} for c in case) for case in cases)
for label, backend in (("cupy-current", "cupy"), ("cupy-minimum", "cupy"), ("jax", "jax"), ("torch", "torch")):
    result = json.loads((root / f"{label}-core.stdout").read_text())
    assert result["status"] == "passed" and result["backend"] == backend
    assert {(c["dtype"], c["history"]) for c in result["cases"]} == {(d, h) for d in ("float32", "float64") for h in (False, True)}
    assert len(result["cases"]) == 4
    for case in result["cases"]:
        assert case["values"] == case["matrices"] == "passed"
        expected = "not_supported_cupy_autodiff" if backend == "cupy" else ("not_requested_stored_history" if case["history"] else "passed")
        assert case["autograd"] == expected
for backend in ("jax", "torch"):
    result = json.loads((root / f"{backend}-statistics.stdout").read_text())
    assert result["status"] == result["masked_censored_gradient"] == "passed"
    assert result["backend"] == backend and len(result["cases"]) == 24
    observations = [c for c in result["cases"] if "family" in c]
    priors = [c for c in result["cases"] if "prior_family" in c]
    assert len(observations) == 14 and len(priors) == 10
    assert all(c["samples"] == 8192 and c["density_and_replay"] == "passed" for c in observations)
    assert all(c["density"] == "passed" for c in priors)
print("Required receipts complete: CuPy 10+10; core 4 per lane; JAX/Torch statistics 24+gradient each")
PY
```

Retain the entire evidence directory, wheels and exact source hashes. Use this
report template for the maintainer and independent reviewer:

- Exact three commit SHAs, PR head, wheel/fixture hashes, Python and frozen dependencies.
- GPU model/architecture, driver, runtime/build versions, supported wheel architectures and idle-device conditions.
- CuPy current/minimum: executed/passed/skipped/failed counts and JUnit paths.
- Each backend core: four cases, precision/history coverage, gradient status and JSON receipt path.
- JAX/Torch statistics: 24 cases, extra gradient/replay/RNG results and JSON receipt paths.
- Host RSS and available device-memory receipts, labeled by measurement scope.
- Every failed attempt, diagnosis, corrected rerun and exact source/environment delta.
- Qualification scope, remaining gaps and independent reviewer verdict for the exact tested head.

For driver/runtime mismatch, missing architectures, failed imports, missing GPU,
OOM, skips or numerical failures, preserve the failure and stop that lane. Do not
weaken thresholds or substitute CPU execution. Diagnose before making a new,
clearly labeled attempt. Run the remaining independent lanes only when safe;
the combined receipt check will intentionally fail until every required lane
has complete passing evidence. Never overwrite a failed attempt.

After review, attach a sanitized receipt to the issues and update qualification
docs. Omit account names, local paths and environment secrets from public
artifacts. If the PR head changes, qualify the new head before approval/merge.
No speedup or application calibration claim follows from these checks. This
scope excludes training, Torch compiled GPU loops, Diffrax GPU execution,
new native state-action helpers, and SciPy's eager CPU banded/GMRES adapters.
Those need their own qualification. Timing/default studies also need a separate
protocol with synchronized cold/first/warm phases and memory measurements.
