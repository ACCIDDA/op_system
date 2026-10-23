"""Tests for release-version classification and workflow outputs."""

from __future__ import annotations

import pathlib

import pytest

from scripts.release_validate import (
    parse_release_version,
    validate_project_maturity,
    validate_release_tag,
    write_github_outputs,
)


@pytest.mark.parametrize(
    "version", ["0.9.0", "1.0.0", "0.10.0a1", "0.10.0b2", "0.10.0rc3"]
)
def test_parse_release_version_accepts_supported_versions(version: str) -> None:
    """Final and explicit pre-release versions are valid release inputs."""
    assert str(parse_release_version(version)) == version


@pytest.mark.parametrize(
    "version",
    ["0.9", "0.9.0.dev1", "0.9.0.post1", "0.9.0+local", "1!0.9.0", "0.9.0-alpha.1"],
)
def test_parse_release_version_rejects_unsupported_versions(version: str) -> None:
    """Development, post, local, epoch, and noncanonical versions are rejected."""
    with pytest.raises(SystemExit, match="Release version"):
        parse_release_version(version)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (
            "0.9.0",
            {
                "docs-alias": "latest",
                "docs-version": "0.9.0",
                "prerelease": "false",
                "release-phase": "final",
            },
        ),
        (
            "0.10.0b1",
            {
                "docs-alias": "pre",
                "docs-version": "0.10.0b1",
                "prerelease": "true",
                "release-phase": "prerelease",
            },
        ),
    ],
)
def test_write_github_outputs_separates_release_phase_from_maturity(
    tmp_path: pathlib.Path,
    version: str,
    expected: dict[str, str],
) -> None:
    """A zero major version is not itself evidence of a pre-release."""
    output_path = tmp_path / "github-output.txt"
    write_github_outputs(version, output_path)
    outputs = dict(
        line.split("=", maxsplit=1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )
    assert outputs["version"] == version
    for key, value in expected.items():
        assert outputs[key] == value
    assert outputs["maturity"] == "alpha"


def test_release_tag_must_match_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tag-context validation cannot publish a differently named version."""
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")
    monkeypatch.setenv("GITHUB_REF_NAME", "v0.9.1")
    with pytest.raises(SystemExit, match="Release tag"):
        validate_release_tag("0.9.0")


def test_release_unit_has_one_matching_maturity() -> None:
    """Core and provider explicitly share a separate Alpha maturity."""
    assert validate_project_maturity() == "alpha"


def test_pages_writers_are_serialized_without_cancellation() -> None:
    """Development and release deployments cannot race or cancel each other."""
    workflow = pathlib.Path(".github/workflows/gh-pages.yaml").read_text("utf-8")
    assert "group: 'op-system-pages'" in workflow
    assert "cancel-in-progress: false" in workflow


def test_release_docs_require_exact_ref_before_checkout() -> None:
    """Release deployments reject an empty tag before mutating Pages state."""
    workflow = pathlib.Path(".github/workflows/gh-pages.yaml").read_text("utf-8")
    resolve_index = workflow.index("Resolve deployment inputs")
    checkout_index = workflow.index("Checkout repository")
    assert resolve_index < checkout_index
    assert "Release mode requires an exact v-prefixed release-ref." in workflow
    assert "git tag --points-at HEAD" in workflow
    assert "already exists; exact versions are immutable" in workflow
    assert "mike set-default" not in workflow
