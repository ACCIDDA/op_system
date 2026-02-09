"""flepimop2 System integration for op_system (minimal, single-file).

This module exposes a pydantic-powered System that compiles an op_system spec
once during validation and presents it as a flepimop2-compatible stepper.

flepimop2 resolves `module: op_system` to `flepimop2.system.op_system`; when a
pydantic BaseModel subclass is defined here, flepimop2 auto-generates a
`build()` function for config-driven construction.
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from flepimop2.configuration import ModuleModel
from flepimop2.system.abc import SystemABC
from pydantic import ConfigDict, Field, model_validator

from op_system import compile_spec

__version__ = "0.1.0"


class OpSystemSystem(ModuleModel, SystemABC):  # noqa: D101
    module: Literal["flepimop2.system.op_system"] = "flepimop2.system.op_system"
    spec: dict = Field(description="op_system RHS specification")

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _compile_and_bind(self) -> Self:
        compiled = compile_spec(self.spec)
        n_state = len(compiled.state_names)

        def _stepper(t: np.float64, state: np.ndarray, **params: object) -> np.ndarray:
            state_arr = np.asarray(state, dtype=np.float64)
            if state_arr.ndim != 1 or state_arr.size != n_state:
                msg = (
                    "state must be a 1D array matching the spec state length; "
                    f"expected ({n_state},), got {state_arr.shape}."
                )
                raise ValueError(msg)
            return np.asarray(
                compiled.eval_fn(np.float64(t), state_arr, **params),
                dtype=np.float64,
            )

        self._stepper = _stepper
        self._compiled_rhs = compiled  # handy for debugging/adapters
        return self


__all__ = ["OpSystemSystem", "__version__"]
