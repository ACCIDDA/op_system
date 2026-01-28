"""op_system.

Domain-agnostic RHS specification + compilation utilities.

Public API (v1)
--------------
Primary user entrypoints:
- `compile_spec`: Validate, normalize, and compile a RHS specification in one step.
- `normalize_rhs`: Validate and normalize a YAML-friendly RHS specification.
- `compile_rhs`: Compile a `NormalizedRhs` into an efficient callable RHS.

Core data structures:
- `NormalizedRhs`
- `CompiledRhs`

Design guarantees:
- No dependency on provider/adapters (eg flepimop2).
- Stable interface for downstream engines.
- Forward-compatible with multiphysics extensions.
"""

from __future__ import annotations

from .compile import CompiledRhs, compile_rhs
from .errors import (
    ErrorCode,
    OpSystemError,
    raise_compilation_error,
    raise_invalid_expression,
    raise_invalid_rhs_spec,
    raise_parameter_error,
    raise_state_shape_error,
    raise_unsupported_feature,
)
from .specs import (
    NormalizedRhs,
    normalize_expr_rhs,
    normalize_rhs,
    normalize_transitions_rhs,
)

# -----------------------------------------------------------------------------
# Versioning & capability metadata
# -----------------------------------------------------------------------------

__version__ = "0.1.0"

SUPPORTED_RHS_KINDS: tuple[str, ...] = ("expr", "transitions")  # noqa: RUF067

# Reserved for forward compatibility
EXPERIMENTAL_FEATURES: frozenset[str] = frozenset()  # noqa: RUF067

# -----------------------------------------------------------------------------
# High-level public façade
# -----------------------------------------------------------------------------


def compile_spec(spec: dict) -> CompiledRhs:  # noqa: RUF067
    """
    Validate, normalize, and compile a RHS specification in one call.

    This is the recommended public entrypoint for most users and adapters.

    Args:
        spec: Raw RHS specification mapping (YAML/JSON friendly).

    Returns:
        CompiledRhs: Runnable RHS callable container.
    """
    rhs = normalize_rhs(spec)
    return compile_rhs(rhs)


# -----------------------------------------------------------------------------
# Public export surface
# -----------------------------------------------------------------------------

__all__ = [
    "EXPERIMENTAL_FEATURES",
    "SUPPORTED_RHS_KINDS",
    "CompiledRhs",
    "NormalizedRhs",
    "__version__",
    "compile_rhs",
    "compile_spec",
    "normalize_expr_rhs",
    "normalize_rhs",
    "normalize_transitions_rhs",
]
