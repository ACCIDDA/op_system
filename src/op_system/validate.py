"""Command-line validation of op_system specifications.

Usage::

    python -m op_system.validate CONFIG.yml [CONFIG.yml ...] [--json]

Each file may be a bare op_system spec or a flepimop2 configuration whose
``system`` entries use ``module: op_system``. The exit code is ``0`` when every
spec passes and ``1`` otherwise. Reading YAML requires PyYAML.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from op_system._validate import ValidationReport, validate_spec

if TYPE_CHECKING:
    from collections.abc import Sequence


def specs_from_file(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(label, spec)`` pairs from a spec or flepimop2 config file.

    Raises:
        RuntimeError: If PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore[import-untyped]  # ruff: ignore[import-outside-top-level]
    except ImportError as exc:  # pragma: no cover - environment dependent
        msg = "op_system.validate needs PyYAML to read configuration files"
        raise RuntimeError(msg) from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    systems = loaded.get("system") if isinstance(loaded, dict) else None
    if isinstance(systems, list):
        return [
            (f"{path}[system {index}]", dict(entry["spec"]))
            for index, entry in enumerate(systems)
            if isinstance(entry, dict) and entry.get("module") == "op_system"
        ]
    return [(str(path), dict(loaded))]


def _describe(label: str, report: ValidationReport) -> list[str]:
    lines = [
        f"{'OK  ' if report.ok else 'FAIL'} {label}: "
        + ", ".join(f"{stage} {status}" for stage, status in report.stages.items())
    ]
    lines.append(
        "     cost: "
        + ", ".join(f"{key}={value}" for key, value in report.cost.items())
    )
    lines.extend(f"     error: {error}" for error in report.errors)
    for template, groups in report.shape_groups.items():
        lines.append(f"     template {template!r} has {len(groups)} expression shapes:")
        lines.extend(
            f"       {group.cells} cell(s), e.g. {group.example_cell}: "
            f"{group.example_expression}"
            for group in groups
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Validate every op_system spec in the given files.

    Returns:
        ``0`` when every spec passes, otherwise ``1``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = [
        (label, validate_spec(spec))
        for path in args.paths
        for label, spec in specs_from_file(path)
    ]
    if args.json:
        print(json.dumps({label: asdict(r) for label, r in results}, indent=2))  # ruff: ignore[print]
    else:
        for label, report in results:
            print("\n".join(_describe(label, report)))  # ruff: ignore[print]
    return 0 if all(report.ok for _, report in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
