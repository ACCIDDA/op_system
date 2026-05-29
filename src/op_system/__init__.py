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
- `OperatorDescriptor`

Design guarantees:
- No dependency on provider/adapters (eg flepimop2).
- Stable interface for downstream engines.
- Forward-compatible with multiphysics extensions.
"""

from __future__ import annotations

import warnings
from importlib.metadata import version
from typing import Final, Literal

from op_system._block_axes import BlockAxisInfo
from op_system._identifer_string import IdentifierString
from op_system._operators import OperatorDescriptor
from op_system._state_string import StateString
from op_system._typing import Array

from .compile import (
    BodyEvalFn,
    CompiledRhs,
    EvalFn,
    PytreeEvalFn,
    StateDict,
    compile_rhs,
)
from .specs import (
    ExpressionString,
    ExprRhs,
    NormalizedRhs,
    TransitionsRhs,
    normalize_expr_rhs,
    normalize_rhs,
    normalize_transitions_rhs,
)

# -----------------------------------------------------------------------------
# Versioning & capability metadata
# -----------------------------------------------------------------------------

__version__ = version("op_system")

SUPPORTED_RHS_KINDS: tuple[str, ...] = ("expr", "transitions")  # noqa: RUF067
DEFAULT_ARRAY_BACKEND: Final[Literal["numpy", "jax"]] = "numpy"  # noqa: RUF067

# Reserved for forward compatibility
EXPERIMENTAL_FEATURES: frozenset[str] = frozenset()  # noqa: RUF067

# -----------------------------------------------------------------------------
# High-level public façade
# -----------------------------------------------------------------------------


def compile_spec(  # noqa: RUF067
    spec: dict[str, object],
    *,
    xp: object | None = None,
    backend: Literal["numpy", "jax"] = DEFAULT_ARRAY_BACKEND,
) -> CompiledRhs:
    """
    Validate, normalize, and compile a RHS specification in one call.

    This is the recommended public entrypoint for most users and adapters.

    The compiled ``eval_fn`` is **namespace-polymorphic**: it infers its
    array namespace from the input ``y`` at call time
    (``y.__array_namespace__()``), so a single compiled callable handles
    NumPy, JAX (concrete and traced), and any other Array-API backend
    natively. No compile-time backend selection is required.

    Args:
        spec: Raw RHS specification mapping (YAML/JSON friendly).
        xp: **Deprecated.** Formerly the compile-time array backend
            namespace. Now ignored — see ``compile_rhs`` for details.
            Will be removed in a future release.
        backend: **Deprecated.** Formerly selected the compile-time
            backend (``"numpy"`` or ``"jax"``). Now ignored. Will be
            removed in a future release.

    Returns:
        CompiledRhs: Runnable RHS callable container.
    """
    if xp is not None or backend != DEFAULT_ARRAY_BACKEND:
        warnings.warn(
            "compile_spec(xp=..., backend=...) is deprecated and ignored. "
            "The compiled eval_fn now infers its array namespace from the "
            "input `y` at call time via __array_namespace__(); pass JAX "
            "arrays for a JAX-native call, NumPy arrays for a NumPy call. "
            "These kwargs will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

    rhs = normalize_rhs(spec)
    return compile_rhs(rhs)


# -----------------------------------------------------------------------------
# Public export surface
# -----------------------------------------------------------------------------

__all__ = [
    "DEFAULT_ARRAY_BACKEND",
    "EXPERIMENTAL_FEATURES",
    "SUPPORTED_RHS_KINDS",
    "Array",
    "BlockAxisInfo",
    "BodyEvalFn",
    "CompiledRhs",
    "EvalFn",
    "ExprRhs",
    "ExpressionString",
    "IdentifierString",
    "NormalizedRhs",
    "OperatorDescriptor",
    "PytreeEvalFn",
    "StateDict",
    "StateString",
    "TransitionsRhs",
    "__version__",
    "compile_rhs",
    "compile_spec",
    "normalize_expr_rhs",
    "normalize_rhs",
    "normalize_transitions_rhs",
]
