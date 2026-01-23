"""Unit tests for `op_system.__version__`."""

from op_system import __version__


def test_version_is_string() -> None:
    """Test that `__version__` is a string."""
    assert isinstance(__version__, str)
