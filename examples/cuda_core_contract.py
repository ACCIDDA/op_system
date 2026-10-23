"""Require real CUDA arrays for a bounded compiler/solver value and AD contract.

This workstation handoff is pending device execution on the recorded candidate.
Use separate backend environments and retain stdout JSON plus stderr. It does
not qualify Torch compiled loops, training, application calibration or speedups.
"""

# Optional native namespaces are intentionally dynamic at this example boundary.
# ruff: file-ignore[any-type, print]
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import resource
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from op_system import compile_spec, symbolic_derivatives


def _backend(name: str) -> Any:
    """Require the selected native runtime, with CPU fallback rejected.

    Returns:
        Actual selected GPU array module.

    Raises:
        RuntimeError: If actual selected CUDA execution is unavailable.
    """
    if name == "jax":
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)  # ruff: ignore[boolean-positional-value-in-call]
        if jax.default_backend() != "gpu":
            msg = "Required JAX CUDA execution is unavailable"
            raise RuntimeError(msg)
        return importlib.import_module("jax.numpy")
    if name == "torch":
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            msg = "Required Torch CUDA execution is unavailable"
            raise RuntimeError(msg)
        return torch
    cupy = importlib.import_module("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        msg = "Required CuPy CUDA execution is unavailable"
        raise RuntimeError(msg)
    cupy.cuda.Device(0).use()
    return cupy


def _native(module: Any, name: str, value: object, dtype: str) -> Any:
    if name == "torch":
        return module.as_tensor(value, dtype=getattr(module, dtype), device="cuda:0")
    return module.asarray(value, dtype=getattr(module, dtype))


def _host(value: Any, name: str, module: Any) -> Any:
    if name == "torch":
        return value.detach().cpu().numpy()
    if name == "cupy":
        return module.asnumpy(value)
    return np.asarray(value)


def _device(value: Any, name: str, module: Any, dtype: str) -> None:
    if name == "jax":
        native = all(device.platform == "gpu" for device in value.devices())
    elif name == "torch":
        native = value.device.type == "cuda"
    else:
        native = isinstance(value, module.ndarray) and value.device.id == 0
    if not native or value.dtype != getattr(module, dtype):
        msg = "Required native GPU output/device or floating dtype was lost"
        raise RuntimeError(msg)


def _case(name: str, module: Any, dtype: str, *, history: bool) -> dict[str, object]:
    """Check independent exponential values and analytic derivative matrices.

    Returns:
        One fully executed precision/history case.
    """
    engine = importlib.import_module("op_engine")
    config = importlib.import_module("op_engine.core_solver")
    options = importlib.import_module("op_engine.model_core")
    jax = importlib.import_module("jax") if name == "jax" else None
    compiled = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "equations": {"x": "-rate*x", "z": "-rate*z"},
        "observables": {"measurement": {"expr": "gain*(x+2*z)"}},
    })
    derivatives = symbolic_derivatives(compiled)
    parameters = _native(module, name, [0.7, 1.5, 2.0], dtype)
    xp = engine.array_namespace(parameters)

    def solve(values: Any) -> Any:
        state = xp.stack((values[2], 2 * values[2]))
        core = engine.ModelCore(
            2,
            1,
            np.array([0.0, 0.5, 1.0]),
            options=options.ModelCoreOptions(
                dtype=getattr(np, dtype), store_history=history
            ),
        )
        core.set_initial_state(xp.reshape(state, (2, 1)))

        def rhs(time: Any, current: Any) -> Any:
            return xp.reshape(
                compiled.eval_fn(time, xp.reshape(current, (2,)), rate=values[0]),
                (2, 1),
            )

        engine.CoreSolver(core).run(
            rhs,
            config=config.RunConfig(
                method="rk4",
                fixed_max_step=1 / 32,
                fixed_loop="scan" if name == "jax" else "unroll",
            ),
        )
        if history:
            return xp.reshape(core.state_array, (3, 2))
        return xp.reshape(core.get_current_state(), (2,))

    actual = jax.jit(solve)(parameters) if jax is not None else solve(parameters)
    _device(actual, name, module, dtype)
    times = np.array([0.0, 0.5, 1.0]) if history else np.array([1.0])
    oracle = np.exp(-0.7 * times[:, None]) * np.array([2.0, 4.0])
    np.testing.assert_allclose(
        _host(actual, name, module).reshape(-1, 2), oracle, rtol=3e-6, atol=1e-7
    )
    matrices = derivatives.evaluate(
        _native(module, name, 0.5, dtype),
        _native(module, name, [2.0, 4.0], dtype),
        rate=parameters[0],
        gain=parameters[1],
    )
    for matrix, expected in zip(
        matrices,
        ([[-0.7, 0], [0, -0.7]], [[-2, 0], [-4, 0]], [[1.5, 3]], [[0, 10]]),
        strict=True,
    ):
        _device(matrix, name, module, dtype)
        np.testing.assert_allclose(
            _host(matrix, name, module), expected, rtol=1e-6, atol=1e-7
        )
    gradient_status = "not_requested_stored_history"
    if not history and name != "cupy":

        def measurement(values: Any) -> Any:
            final = solve(values)
            return values[1] * (final[0] + 2 * final[1])

        if jax is not None:
            gradient = jax.jit(jax.grad(measurement))(parameters)
        else:
            leaf = parameters.detach().requires_grad_()
            gradient = module.autograd.grad(measurement(leaf), leaf)[0]
        _device(gradient, name, module, dtype)
        np.testing.assert_allclose(
            _host(gradient, name, module),
            np.exp(-0.7) * np.array([-15.0, 10.0, 7.5]),
            rtol=3e-6,
            atol=1e-6,
        )
        gradient_status = "passed"
    elif name == "cupy":
        gradient_status = "not_supported_cupy_autodiff"
    return {
        "dtype": dtype,
        "history": history,
        "values": "passed",
        "matrices": "passed",
        "autograd": gradient_status,
    }


def main() -> None:
    """Execute four native cases or fail; retain a portable JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("jax", "torch", "cupy"))
    args = parser.parse_args()
    module = _backend(args.backend)
    cases = [
        _case(args.backend, module, dtype, history=history)
        for dtype in ("float32", "float64")
        for history in (False, True)
    ]
    if args.backend == "jax":
        device = str(importlib.import_module("jax").devices()[0])
    elif args.backend == "torch":
        device = module.cuda.get_device_name(0)
    else:
        device = str(module.cuda.runtime.getDeviceProperties(0)["name"])
    print(
        json.dumps(
            {
                "status": "passed",
                "backend": args.backend,
                "device": device,
                "cases": cases,
                "versions": {
                    name: version(name)
                    for name in (
                        "numpy",
                        "op-system",
                        "op-engine",
                        "cupy-cuda12x" if args.backend == "cupy" else args.backend,
                    )
                },
                "fixture_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "compiler_sha256": hashlib.sha256(
                    inspect.getsource(compile_spec).encode()
                ).hexdigest(),
                "process_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
