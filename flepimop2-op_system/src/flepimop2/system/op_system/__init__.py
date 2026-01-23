"""flepimop2 system integration for op_system.

This package intentionally defines the public System class in this module so that
flepimop2's dynamic loader can auto-inject a default `build()` function.

Why:
- flepimop2 resolves `module: op_system` to `flepimop2.system.op_system`
- if that module has no `build`, it looks for a pydantic BaseModel subclass
  defined *in this module* and generates `build()` automatically.
"""

from __future__ import annotations

from .system import _OpSystemFlepimop2SystemImpl


class OpSystemFlepimop2System(_OpSystemFlepimop2SystemImpl):  # noqa: RUF067
    """Public op_system-backed flepimop2 System (default-build enabled)."""


__all__ = ["OpSystemeFlepimop2System"]
