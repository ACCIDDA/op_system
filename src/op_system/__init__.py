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

import importlib
from importlib.metadata import version
from typing import Final, Literal

import numpy as np

from op_system._identifer_string import IdentifierString
from op_system._state_string import StateString

from .compile import CompiledRhs, EvalFn, compile_rhs
from .specs import (
    NormalizedRhs,
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


def _resolve_xp(backend: Literal["numpy", "jax"]) -> object:  # noqa: RUF067
    if backend == "numpy":
        return np
    try:
        return importlib.import_module("jax.numpy")
    except ImportError as exc:
        msg = (
            "backend='jax' requires jax to be installed "
            '(pip install "op_system[jax]")'
        )
        raise ImportError(msg) from exc


def compile_spec(  # noqa: RUF067
    spec: dict[str, object],
    *,
    xp: object | None = None,
    backend: Literal["numpy", "jax"] = DEFAULT_ARRAY_BACKEND,
) -> CompiledRhs:
    """
    Validate, normalize, and compile a RHS specification in one call.

    This is the recommended public entrypoint for most users and adapters.

    Args:
        spec: Raw RHS specification mapping (YAML/JSON friendly).
        xp: Optional array backend namespace. If provided, takes precedence over
            backend selection.
        backend: Backend selector used when xp is not provided.

    Returns:
        CompiledRhs: Runnable RHS callable container.

    Raises:
        ValueError: If both `xp` and a non-default backend are provided.
    """
    if xp is not None and backend != DEFAULT_ARRAY_BACKEND:
        msg = "Pass either xp or backend='jax', not both."
        raise ValueError(msg)

    xp_ns = _resolve_xp(backend) if xp is None else xp
    rhs = normalize_rhs(spec)
    return compile_rhs(rhs, xp=xp_ns)


# -----------------------------------------------------------------------------
# Public export surface
# -----------------------------------------------------------------------------

__all__ = [
    "DEFAULT_ARRAY_BACKEND",
    "EXPERIMENTAL_FEATURES",
    "SUPPORTED_RHS_KINDS",
    "CompiledRhs",
    "EvalFn",
    "IdentifierString",
    "NormalizedRhs",
    "StateString",
    "__version__",
    "compile_rhs",
    "compile_spec",
    "normalize_expr_rhs",
    "normalize_rhs",
    "normalize_transitions_rhs",
]
