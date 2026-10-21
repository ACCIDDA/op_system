"""Check generated Julia RHS/observations against the same compiled synthetic IR.

Requires an explicitly available Julia executable, not a downloaded Julia package.
No external structural-rank result or inference accuracy claim is made.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from op_system import compile_spec, export_julia, symbolic_derivatives

_FINITE_DIFFERENCE = """
function independent_jacobian(f, u, p, t, argument)
    input = argument == 1 ? u : p
    columns = map(eachindex(input)) do j
        plus, minus = copy(input), copy(input)
        plus[j] += 1e-5
        minus[j] -= 1e-5
        left = argument == 1 ? f(plus, p, t) : f(u, plus, t)
        right = argument == 1 ? f(minus, p, t) : f(u, minus, t)
        (left - right) / 2e-5
    end
    return vec(permutedims(hcat(columns...)))
end
"""


def main() -> None:
    """Emit and execute the bounded model, retaining source and runtime identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    compiled = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "aliases": {"loss": "a*x", "net": "loss-b*z"},
        "equations": {"x": "-net", "z": "net"},
        "observables": {"measurement": {"expr": "gain*x+z+t"}},
    })
    artifact = export_julia(compiled)
    points = (
        ([2.0, 4.0], {"a": 0.3, "b": 0.7, "gain": 3.0}, 0.5),
        ([0.2, 1.5], {"a": 0.9, "b": 0.2, "gain": 0.5}, 1.2),
        ([3.0, 0.1], {"a": 0.1, "b": 1.3, "gain": 2.0}, 0.0),
    )
    commands = [artifact.code, _FINITE_DIFFERENCE]
    expected: list[NDArray[np.float64]] = []
    for state, parameters, time in points:
        values = [parameters[name] for name in artifact.parameter_names]
        invocation = f"{json.dumps(state)}, {json.dumps(values)}, {time!r}"
        commands.extend((
            f"println(join(op_system_rhs({invocation}), ','))",
            f"println(join(op_system_observations({invocation}), ','))",
        ))
        expected.extend((
            np.asarray(
                compiled.eval_fn(
                    time,
                    np.asarray(state),
                    **{name: parameters[name] for name in compiled.param_names},
                )
            ),
            np.asarray([
                compiled.observables[name].eval_fn(
                    time, np.asarray(state), **parameters
                )
                for name in artifact.observable_names
            ]),
        ))
        for function in ("op_system_rhs", "op_system_observations"):
            commands.extend(
                f"println(join(independent_jacobian({function}, {invocation}, "
                f"{argument}), ','))"
                for argument in (1, 2)
            )
        expected.extend(
            np.asarray(matrix).ravel()
            for matrix in symbolic_derivatives(compiled).evaluate(
                time, np.asarray(state), **parameters
            )
        )
    output, runtime = _execute(args.julia, "\n".join(commands))
    errors = _validate(output, expected)
    print(  # ruff: ignore[print]
        json.dumps({
            "runtime": runtime.strip(),
            "model": artifact.model.digest,
            "emitter_schema_version": artifact.emitter_schema_version,
            "state_names": artifact.state_names,
            "parameter_names": artifact.parameter_names,
            "observable_names": artifact.observable_names,
            "rational": artifact.rational,
            "independent_points": len(points),
            "maximum_value_error": errors[0],
            "maximum_derivative_error": errors[1],
            "derivative_reference": "Julia central difference, absolute step=1e-5",
        })
    )


def _validate(output: str, expected: list[NDArray[np.float64]]) -> list[float]:
    """Check generated values and independent Julia numerical derivatives.

    Returns:
        Largest absolute errors for values and derivative entries respectively.

    Raises:
        RuntimeError: Julia output rows differ from the emitted layout.
    """
    lines = output.splitlines()
    if len(lines) != len(expected):
        msg = "Julia output rows differ from the emitted model layout"
        raise RuntimeError(msg)
    errors = [0.0, 0.0]
    for index, (line, reference) in enumerate(zip(lines, expected, strict=True)):
        value = np.asarray([float(token) for token in line.split(",")])
        derivative = index % 6 >= 2
        tolerance = 1e-8 if derivative else 1e-12
        np.testing.assert_allclose(value, reference, rtol=tolerance, atol=tolerance)
        errors[derivative] = max(
            errors[derivative], float(np.max(np.abs(value - reference)))
        )
    return errors


def _execute(executable: str, code: str) -> tuple[str, str]:
    """Run compiler-generated code with a bounded explicit executable.

    Returns:
        Captured numeric output and Julia runtime version.
    """
    environment = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix="op-system-julia-") as directory:
        path = Path(directory) / "generated.jl"
        path.write_text(code)
        environment.setdefault("JULIA_DEPOT_PATH", str(Path(directory) / "depot"))
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [executable, "--startup-file=no", str(path)],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        runtime = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [executable, "--version"],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    return result.stdout, runtime.stdout


if __name__ == "__main__":
    main()
