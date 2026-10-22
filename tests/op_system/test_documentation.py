"""Execute the published quickstart and independent-reaction example."""

from __future__ import annotations

import re
import runpy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_dependency_guide_keeps_pinned_and_global_reads() -> None:
    """The complete published workflow retains nonlocal state and output reads."""
    blocks = re.findall(
        r"```python\n(.*?)\n```",
        (ROOT / "docs/guides/dependency-structure.md").read_text(),
        re.DOTALL,
    )
    namespace: dict[str, Any] = {}
    for block in blocks:
        exec(block, namespace)  # ruff: ignore[exec-builtin] - trusted guide
    assert namespace["record"]["state"]["rows"] == [[0], [0, 1], [0, 2]]
    assert namespace["record"]["observable_state"]["rows"] == [[0, 1, 2]]


def test_readme_quickstart_matches_sir_equations() -> None:
    """The README's actual Python block has an independently computed RHS."""
    blocks = re.findall(
        r"```python\n(.*?)\n```", (ROOT / "README.md").read_text(), re.DOTALL
    )
    namespace: dict[str, Any] = {}
    exec(blocks[0], namespace)  # ruff: ignore[exec-builtin] - trusted repository documentation
    np.testing.assert_allclose(namespace["dydt"], [-0.2997, 0.1997, 0.1])


def test_independent_reaction_example_checks_named_axes_and_forcing() -> None:
    """The standalone example runs and retains explicit independent coordinates."""
    namespace = runpy.run_path(str(ROOT / "examples/independent_reactions.py"))
    result = namespace["run_example"]()
    assert result["forcing_breakpoints"] == (1.0,)
    assert result["independent_axes"] == ("rep",)
    assert result["population"] == [10.0, 20.0]


def test_readme_history_provider_example_is_executable() -> None:
    """The actual history-hook workflow returns the provider's zero signal."""
    blocks = re.findall(
        r"```python\n(.*?)\n```", (ROOT / "README.md").read_text(), re.DOTALL
    )
    namespace: dict[str, Any] = {}
    exec(blocks[-1], namespace)  # ruff: ignore[exec-builtin] - trusted repository documentation
    np.testing.assert_array_equal(namespace["out"]["x"], [0.0, 0.0])
