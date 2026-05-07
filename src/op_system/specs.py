"""op_system.specs.

Public façade for RHS specification models and normalization.

The implementation lives in ``op_system._normalize``.  This module re-exports
the public types and functions under the stable ``op_system.specs`` namespace
for backward compatibility.

Supported RHS kinds
-------------------
1) kind: "expr"        - explicit d(state)/dt equations per state variable.
2) kind: "transitions" - diagram-style per-capita hazard transitions.

See ``op_system._normalize`` for the full implementation.
"""

from __future__ import annotations

from op_system._normalize import (
    NormalizedRhs,
    StateTemplate,
    normalize_expr_rhs,
    normalize_rhs,
    normalize_transitions_rhs,
)
from op_system._templates import PinnedToken, SelectorToken, WildcardToken

__all__ = [
    "NormalizedRhs",
    "PinnedToken",
    "SelectorToken",
    "StateTemplate",
    "WildcardToken",
    "normalize_expr_rhs",
    "normalize_rhs",
    "normalize_transitions_rhs",
]
