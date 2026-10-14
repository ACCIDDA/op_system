"""Verify installed provider typing and rejection of an invalid state argument."""

from __future__ import annotations

import importlib
import importlib.resources
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import tempfile
from pathlib import Path

_CONSUMER = """
from typing import assert_type
import numpy as np
import numpy.typing as npt
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import Array, StateChangeEnum, SystemProtocol

def consume(
    provider: OpSystemSystem, state: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    assert_type(provider.state_change, StateChangeEnum)
    assert_type(provider.bind(), SystemProtocol)
    return assert_type(provider.step(np.float64(0.0), state), npt.NDArray[np.float64])
"""
_INVALID_CONSUMER = """
def invalid(provider: OpSystemSystem) -> Array:
    return provider.step(np.float64(0.0), "not an array")
"""


def _check(directory: Path, *, valid: bool) -> bool:
    path = directory / "consumer.py"
    path.write_text(_CONSUMER + ("" if valid else _INVALID_CONSUMER), encoding="utf-8")
    process = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-m", "mypy", "--strict", "--no-incremental", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    passed = (
        process.returncode == 0
        if valid
        else (process.returncode != 0 and "[arg-type]" in process.stdout)
    )
    if not passed:
        print(process.stdout + process.stderr, file=sys.stderr)
    return passed


def main() -> int:
    """Require shipped type metadata and a typed public consumer boundary.

    Returns:
        Zero only when the installed provider accepts and rejects expected types.
    """
    module = importlib.import_module("flepimop2.system.op_system")
    if module.__file__ is None or not Path(module.__file__).resolve().is_relative_to(
        Path(sys.prefix).resolve()
    ):
        print(
            "Provider typing must be qualified against the installed wheel",
            file=sys.stderr,
        )
        return 1
    marker = importlib.resources.files(module).joinpath("py.typed")
    if not marker.is_file():
        print("Installed provider is missing its py.typed marker", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="op-system-provider-types-") as directory:
        if not _check(Path(directory), valid=True) or not _check(
            Path(directory), valid=False
        ):
            return 1
    print(
        "Installed provider typing qualified: valid consumer and rejected invalid state"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
