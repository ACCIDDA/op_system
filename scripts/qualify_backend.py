"""Run required core compiler tests and reject absent or skipped qualification."""

from __future__ import annotations

import argparse
import importlib
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import tempfile
import xml.etree.ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]
from pathlib import Path


def main() -> int:
    """Run a selected backend and check its executed test records.

    Returns:
        Zero only when selected tests execute successfully without skips.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["numpy", "jax", "torch", "numba"])
    args = parser.parse_args()
    # This is a hard dependency check, not pytest.importorskip: missing
    # backends must fail their required job before test collection.
    try:
        module = importlib.import_module(args.backend)
    except ImportError:
        print(
            f"Required backend {args.backend!r} is missing; "
            f"run just qualify-{args.backend}",
            file=sys.stderr,
        )
        return 1
    print(
        f"Qualifying {args.backend} {module.__version__} on {sys.version.split()[0]}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="op-system-qualification-") as directory:
        report = Path(directory) / "results.xml"
        selected = (
            ["tests/op_system/test_numba.py"]
            if args.backend == "numba"
            else [
                "tests/op_system/test_backend_qualification.py",
                "tests/op_system/test_hierarchical_benchmark.py",
                "tests/op_system/test_parameter_metadata.py",
                "tests/op_system/test_scientific_metadata.py",
                "tests/op_system/test_balances.py",
                "tests/op_system/test_observables.py",
                "tests/op_system/test_reaction_callbacks.py",
                "tests/op_system/test_replicate_contract.py",
                "tests/op_system/test_named_kernel_products.py",
                "tests/op_system/test_gauges.py",
                "tests/op_system/test_state_coordinates.py",
                "tests/op_system/test_portable_symbolic.py",
                "tests/op_system/test_scalar_device.py",
                "tests/op_system/test_torch_native_conversion.py",
                "tests/op_system/test_scalar_callables.py",
                "tests/op_system/test_vector_buffer_semantics.py",
                "tests/op_system/test_paired_evaluator.py",
            ]
        )
        process = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [
                sys.executable,
                "-m",
                "pytest",
                *selected,
                *([] if args.backend == "numba" else ["-k", args.backend]),
                "-q",
                "--tb=short",
                f"--junitxml={report}",
            ],
            check=False,
        )
        if process.returncode:
            return process.returncode
        # Read only the locally generated pytest report.
        cases = ET.parse(report).findall(".//testcase")  # ruff: ignore[suspicious-xml-element-tree-usage]
        if not cases or any(case.find("skipped") is not None for case in cases):
            print(
                "Backend qualification must execute tests without skips",
                file=sys.stderr,
            )
            return 1
        print(f"Qualified {args.backend}: {len(cases)} tests, zero skips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
