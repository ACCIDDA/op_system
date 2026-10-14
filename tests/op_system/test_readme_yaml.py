"""Keep the public YAML cookbook loadable before model validation."""

import re
from pathlib import Path

import yaml  # type: ignore[import-untyped]


def test_readme_yaml_fences_parse() -> None:
    """Bracketed selectors remain strings rather than YAML collection syntax."""
    readme = Path(__file__).resolve().parents[2] / "README.md"
    snippets = re.findall(r"```yaml\n(.*?)\n```", readme.read_text(), re.DOTALL)
    assert snippets
    for snippet in snippets:
        assert isinstance(yaml.safe_load(snippet), dict)
