# flepimop2-op_system: Operator-Partitioned System Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tests for provider package version metadata."""

from importlib import metadata

import pytest

import flepimop2.system.op_system as provider


def test_version_matches_distribution_metadata() -> None:
    """Expose the installed provider distribution version."""
    assert provider.__version__ == metadata.version("flepimop2-op-system")


def test_version_has_source_tree_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use an explicit sentinel when distribution metadata is unavailable."""

    def missing_distribution(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing_distribution)

    assert provider._distribution_version() == "0+unknown"  # ruff: ignore[private-member-access]
