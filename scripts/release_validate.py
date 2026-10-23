"""Validate release version consistency for both distributable packages."""

from __future__ import annotations

import argparse
import os
import pathlib
import tomllib
from typing import Final

from packaging.version import InvalidVersion, Version

REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[1]
PROVIDER_PYPROJECT: Final[pathlib.Path] = (
    REPO_ROOT / "flepimop2-op_system" / "pyproject.toml"
)
DEVELOPMENT_STATUS_PREFIX: Final[str] = "Development Status :: "


def get_declared_versions() -> dict[str, str]:
    """Read each release version declaration used by this repository.

    Returns:
        A mapping from file path to declared release version.
    """
    core_version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))[
        "project"
    ]["version"]
    provider_version = tomllib.loads(PROVIDER_PYPROJECT.read_text("utf-8"))["project"][
        "version"
    ]

    return {
        "pyproject.toml": str(core_version),
        "flepimop2-op_system/pyproject.toml": str(provider_version),
    }


def parse_release_version(version: str) -> Version:
    """Parse one supported canonical release version.

    Args:
        version: Candidate PEP 440 version.

    Returns:
        The parsed version.

    Raises:
        SystemExit: If the version is invalid or is not a supported release form.
    """
    try:
        parsed = Version(version)
    except InvalidVersion as exc:
        msg = f"Release version is not valid PEP 440: {version!r}."
        raise SystemExit(msg) from exc

    unsupported = (
        parsed.epoch != 0
        or len(parsed.release) != 3
        or parsed.dev is not None
        or parsed.post is not None
        or parsed.local is not None
        or str(parsed) != version
    )
    if unsupported:
        msg = (
            "Release version must be canonical X.Y.Z or X.Y.ZaN/bN/rcN; "
            f"got {version!r}."
        )
        raise SystemExit(msg)

    return parsed


def validate_project_maturity() -> str:
    """Validate one matching maturity classifier for the release unit.

    Returns:
        The normalized shared maturity label.

    Raises:
        SystemExit: If either distribution does not declare exactly one matching
            development-status classifier.
    """
    projects = {
        "pyproject.toml": tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text("utf-8")
        )["project"],
        "flepimop2-op_system/pyproject.toml": tomllib.loads(
            PROVIDER_PYPROJECT.read_text("utf-8")
        )["project"],
    }
    maturities: dict[str, str] = {}
    for path, project in projects.items():
        matches = [
            str(value)
            for value in project.get("classifiers", [])
            if str(value).startswith(DEVELOPMENT_STATUS_PREFIX)
        ]
        if len(matches) != 1:
            msg = f"{path} must declare exactly one Development Status classifier."
            raise SystemExit(msg)
        maturities[path] = matches[0]

    if len(set(maturities.values())) != 1:
        rendered = ", ".join(f"{path}={value}" for path, value in maturities.items())
        msg = f"Project maturity classifiers do not match: {rendered}"
        raise SystemExit(msg)
    return next(iter(maturities.values())).split(" - ", maxsplit=1)[-1].lower()


def validate_release_tag(version: str) -> None:
    """Require an exact version tag when validation runs in a tag context.

    Args:
        version: Validated shared release version.

    Raises:
        SystemExit: If a tag-triggered workflow uses a different tag.
    """
    if os.environ.get("GITHUB_REF_TYPE") != "tag":
        return
    expected = f"v{version}"
    actual = os.environ.get("GITHUB_REF_NAME", "")
    if actual != expected:
        msg = f"Release tag must be {expected!r}, got {actual!r}."
        raise SystemExit(msg)


def validate_release_version() -> str:
    """Validate that all release version declarations agree.

    Returns:
        The shared release version.

    Raises:
        SystemExit: If declared versions differ or are not supported PEP 440
            release versions.
    """
    versions = get_declared_versions()
    distinct_versions = sorted(set(versions.values()))
    if len(distinct_versions) != 1:
        rendered = ", ".join(f"{path}={version}" for path, version in versions.items())
        msg = f"Release versions do not match across packages: {rendered}"
        raise SystemExit(msg)

    version = distinct_versions[0]
    parse_release_version(version)
    validate_project_maturity()
    validate_release_tag(version)

    return version


def write_github_outputs(version: str, output_path: pathlib.Path) -> None:
    """Write workflow outputs for downstream GitHub Actions jobs."""
    parsed = parse_release_version(version)
    prerelease = str(parsed.is_prerelease).lower()
    docs_alias = "pre" if parsed.is_prerelease else "latest"
    release_phase = "prerelease" if parsed.is_prerelease else "final"
    maturity = validate_project_maturity()
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"version={version}\n")
        fh.write(f"docs-version={version}\n")
        fh.write(f"docs-alias={docs_alias}\n")
        fh.write(f"prerelease={prerelease}\n")
        fh.write(f"release-phase={release_phase}\n")
        fh.write(f"maturity={maturity}\n")


def main() -> None:
    """Run release validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--github-output",
        type=pathlib.Path,
        default=None,
        help="Optional path to a GitHub Actions output file.",
    )
    args = parser.parse_args()

    version = validate_release_version()
    print(f"Validated release version: {version}")

    output_path = args.github_output
    if output_path is None:
        raw_output = os.environ.get("GITHUB_OUTPUT")
        if raw_output:
            output_path = pathlib.Path(raw_output)

    if output_path is not None:
        write_github_outputs(version, output_path)


if __name__ == "__main__":
    main()
