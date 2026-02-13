"""flepimop2 System integration for op_system (thin, single-file).

Compiles an op_system spec once, exposes the stepper, and attaches optional
metadata hints (state names, param names, meta) without deriving extra data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from flepimop2.configuration import ModuleModel
from flepimop2.system.abc import SystemABC
from pydantic import ConfigDict, Field, model_validator

from op_system import compile_spec  # type: ignore[attr-defined]

__version__ = "0.1.0"

if TYPE_CHECKING:
    from flepimop2.typing import Float64NDArray


class OpSystemSystem(ModuleModel, SystemABC):  # noqa: D101
    module: Literal["flepimop2.system.op_system"] = "flepimop2.system.op_system"
    spec: dict[str, object] | None = Field(
        default=None, description="Inline op_system RHS specification (already loaded)"
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _compile_and_bind(self) -> Self:
        if self.spec is None:
            message = "spec must be provided as an already loaded mapping"
            raise ValueError(message)
        if not isinstance(self.spec, dict):
            message = "spec must be a mapping (dict)"
            raise TypeError(message)

        spec_obj = self.spec
        compiled = compile_spec(spec_obj)
        n_state = len(compiled.state_names)

        self.state_size = n_state
        self.state_names = compiled.state_names
        self.param_names = compiled.param_names
        # Pass through raw meta for optional downstream use; do not derive kernels here.
        self.meta = getattr(compiled, "meta", None)

        def _stepper(
            time: np.float64,
            state: Float64NDArray,
            **kwargs: Any,  # noqa: ANN401
        ) -> Float64NDArray:
            state_arr = np.asarray(state, dtype=np.float64)
            if state_arr.ndim != 1 or state_arr.size != n_state:
                msg = (
                    "state must be a 1D array matching the spec state length; "
                    f"expected ({n_state},), got {state_arr.shape}."
                )
                raise ValueError(msg)
            return np.asarray(
                compiled.eval_fn(np.float64(time), state_arr, **kwargs),
                dtype=np.float64,
            )

        self._stepper = _stepper
        self._compiled_rhs = compiled  # handy for debugging/adapters
        return self


__all__ = ["OpSystemSystem", "__version__"]
