# flepimop2-op_system: Operator-Partitioned System Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""flepimop2 System integration for op_system (minimal, single-file).

This module exposes a pydantic-powered System that compiles an op_system spec
once during validation and presents it as a flepimop2-compatible stepper.

flepimop2 resolves `module: op_system` to `flepimop2.system.op_system`; when a
pydantic BaseModel subclass is defined here, flepimop2 auto-generates a
`build()` function for config-driven construction.
"""

from __future__ import annotations

import functools
import importlib
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from flepimop2.configuration import ModuleModel
from flepimop2.system.abc import SystemABC
from flepimop2.typing import (
    IdentifierString,
    StateChangeEnum,
    SystemProtocol,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
from pydantic import ConfigDict, Field

from op_system import CompiledRhs, compile_spec

__version__ = "0.1.0"

if TYPE_CHECKING:
    from collections.abc import Callable

    from flepimop2.typing import Float64NDArray


class _AxesMeta(NamedTuple):
    axis_order: tuple[str, ...]
    axis_sizes: dict[str, int]
    axis_coords: dict[str, np.ndarray]


class OpSystemSystem(ModuleModel, SystemABC):  # noqa: D101
    module: Literal["flepimop2.system.op_system"] = "flepimop2.system.op_system"
    state_change: StateChangeEnum = StateChangeEnum.FLOW
    backend: Literal["numpy", "jax"] = "numpy"

    spec: dict[str, object] = Field(
        default=..., description="Inline op_system RHS specification (already loaded)"
    )

    model_config = ConfigDict(extra="allow")

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        """Compile `op_system` specification and prepare stepper and shape helpers."""
        del context

        spec_obj = self.spec
        xp = self._get_backend_namespace(self.backend)
        compiled = compile_spec(spec_obj, xp=xp)
        n_state = len(compiled.state_names)

        axes_meta = self._extract_axes_meta(compiled)
        mixing_kernels = self._build_mixing_kernels(compiled, axes_meta.axis_coords)
        shape_dims, flatten_fn, unflatten_fn = self._make_shape_helpers(
            n_state, axes_meta
        )

        operators = self._extract_operators(compiled)
        operator_axis = self._extract_operator_axis(compiled)
        self.options = {
            "axis_order": axes_meta.axis_order,
            "axis_sizes": axes_meta.axis_sizes,
            "axis_coords": axes_meta.axis_coords,
            "state_names": compiled.state_names,
            "initial_state": compiled.meta.get("initial_state"),
            "state_shape": shape_dims,
            "flatten": flatten_fn,
            "unflatten": unflatten_fn,
            "mixing_kernels": mixing_kernels,
            "operators": operators,
            "operator_axis": operator_axis,
        }

        def _stepper(
            time: np.float64,
            state: Float64NDArray,
            **kwargs: Any,  # noqa: ANN401
        ) -> Float64NDArray:
            if self.backend == "numpy":
                state_arr = np.asarray(state, dtype=np.float64)
            else:
                state_arr = xp.asarray(state)
            if state_arr.ndim != 1 or state_arr.size != n_state:
                msg = (
                    "state must be a 1D array matching the spec state length; "
                    f"expected ({n_state},), got {state_arr.shape}."
                )
                raise ValueError(msg)
            params = dict(mixing_kernels)
            params.update(kwargs)
            if self.backend == "numpy":
                return np.asarray(
                    compiled.eval_fn(np.float64(time), state_arr, **params),
                    dtype=np.float64,
                )
            result = compiled.eval_fn(time, state_arr, **params)
            if TYPE_CHECKING:
                return np.asarray(result, dtype=np.float64)
            return result

        self._stepper = _stepper
        self._compiled_rhs = compiled  # handy for debugging/adapters

    @override
    def _bind_impl(
        self, params: dict[IdentifierString, Any] | None = None
    ) -> SystemProtocol:
        """Return a stepper with any static parameters partially applied."""
        return functools.partial(self._stepper, **(params or {}))

    @staticmethod
    def _extract_axes_meta(compiled: CompiledRhs) -> _AxesMeta:
        axes_meta = compiled.meta.get("axes", [])
        axis_sizes: dict[str, int] = {"subgroup": 1}
        axis_coords: dict[str, np.ndarray] = {
            "subgroup": np.asarray([0], dtype=np.float64)
        }
        axis_order: list[str] = ["state", "subgroup"]
        for ax in axes_meta:
            name = str(ax.get("name"))
            size = int(ax.get("size", 0) or len(ax.get("coords", []) or []))
            raw_coords = ax.get("coords", None)
            try:
                coords = (
                    np.asarray(raw_coords, dtype=np.float64)
                    if raw_coords is not None
                    else np.arange(size, dtype=np.float64)
                )
            except (ValueError, TypeError):
                coords = np.arange(size, dtype=np.float64)
            axis_sizes[name] = size
            axis_coords[name] = coords
            axis_order.append(name)
        return _AxesMeta(
            axis_order=tuple(axis_order), axis_sizes=axis_sizes, axis_coords=axis_coords
        )

    @staticmethod
    def _get_backend_namespace(backend: Literal["numpy", "jax"]) -> Any:  # noqa: ANN401
        if backend == "numpy":
            return np
        try:
            jnp = importlib.import_module("jax.numpy")
        except ImportError as exc:
            msg = "backend='jax' requires jax to be installed"
            raise ImportError(msg) from exc
        return jnp

    @staticmethod
    def _extract_operators(
        compiled: CompiledRhs,
    ) -> tuple[MappingProxyType[str, Any], ...] | None:
        """Extract operator metadata from compiled spec as immutable mappings.

        Returns:
            Tuple of read-only operator dicts, or ``None`` if no operators.
        """
        ops = compiled.meta.get("operators")
        if not ops:
            return None
        return tuple(MappingProxyType(op) for op in ops)

    @staticmethod
    def _extract_operator_axis(compiled: CompiledRhs) -> str | None:
        """Return the single shared axis name if all operators act on the same axis.

        Returns:
            Shared axis name, or ``None`` if there are zero or multiple axes.
        """
        ops = compiled.meta.get("operators")
        if not ops:
            return None
        axes = {op.get("axis") for op in ops}
        if len(axes) == 1:
            return str(next(iter(axes)))
        return None

    def _build_mixing_kernels(
        self, compiled: CompiledRhs, axis_coords: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        mixing_meta = compiled.meta.get("kernels", []) or []
        kernels: dict[str, np.ndarray] = {}
        for mk in mixing_meta:
            name = str(mk.get("name"))
            if not name or name in kernels:
                continue
            value = mk.get("value")
            axes = tuple(mk.get("axes", ()))
            form = str(mk.get("form", "custom_value"))
            params = mk.get("params") or {}
            if value is not None:
                kernels[name] = np.asarray(value, dtype=np.float64)
                continue
            kernels[name] = self._generate_kernel(form, params, axes, axis_coords)
        return kernels

    @staticmethod
    def _as_float(val: object) -> float:
        return float(val)  # type: ignore[arg-type]

    @staticmethod
    def _generate_kernel(
        form: str,
        params: dict[str, object],
        axes: tuple[str, ...],
        axis_coords: dict[str, np.ndarray],
    ) -> np.ndarray:
        def _axis_coords(ax_name: str) -> np.ndarray:
            if ax_name not in axis_coords:
                msg = f"mixing references unknown axis {ax_name!r}"
                raise ValueError(msg)
            return np.asarray(axis_coords[ax_name], dtype=np.float64)

        if not axes:
            return np.asarray(params.get("value", 0.0), dtype=np.float64)
        coords0 = _axis_coords(axes[0])
        coords1 = _axis_coords(axes[1]) if len(axes) > 1 else coords0
        dx = np.abs(coords0[:, None] - coords1[None, :])
        form_l = form.lower()
        scale = OpSystemSystem._as_float(params.get("scale", 1.0))
        if form_l == "custom_value":
            val = params.get("value")
            if val is None:
                message = "custom_value mixing requires params.value"
                raise ValueError(message)
            return np.asarray(val, dtype=np.float64)
        handlers = {
            "erfc": OpSystemSystem._kernel_erfc,
            "gaussian": OpSystemSystem._kernel_gaussian,
            "exponential": OpSystemSystem._kernel_exponential,
            "gamma": OpSystemSystem._kernel_gamma,
            "power_law": OpSystemSystem._kernel_power_law,
        }
        if form_l not in handlers:
            msg = f"Unsupported mixing form {form}"
            raise ValueError(msg)
        kernel_fn = handlers[form_l]
        return kernel_fn(dx, params, scale)

    @staticmethod
    def _kernel_erfc(
        dx: np.ndarray, params: dict[str, object], scale: float
    ) -> np.ndarray:
        sigma = OpSystemSystem._as_float(params.get("sigma", 1.0))
        result = scale * np.erfc(dx / sigma)  # type: ignore[attr-defined]
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _kernel_gaussian(
        dx: np.ndarray, params: dict[str, object], scale: float
    ) -> np.ndarray:
        sigma = OpSystemSystem._as_float(params.get("sigma", 1.0))
        denom = 2.0 * sigma * sigma
        result = scale * np.exp(-(dx**2) / denom)
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _kernel_exponential(
        dx: np.ndarray, params: dict[str, object], scale: float
    ) -> np.ndarray:
        lambda_param = OpSystemSystem._as_float(params.get("lambda", 1.0))
        result = scale * np.exp(-dx / lambda_param)
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _kernel_gamma(
        dx: np.ndarray, params: dict[str, object], scale: float
    ) -> np.ndarray:
        k_param = OpSystemSystem._as_float(params.get("k", 1.0))
        theta = OpSystemSystem._as_float(params.get("theta", 1.0))
        dx_safe = np.clip(dx, 1e-12, None)
        result = scale * (dx_safe / theta) ** (k_param - 1.0) * np.exp(-dx_safe / theta)
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _kernel_power_law(
        dx: np.ndarray, params: dict[str, object], scale: float
    ) -> np.ndarray:
        sigma = OpSystemSystem._as_float(params.get("sigma", 1.0))
        p_param = OpSystemSystem._as_float(params.get("p", 1.0))
        result = scale * (1.0 + dx / sigma) ** (-p_param)
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _make_shape_helpers(
        n_state: int, axes_meta: _AxesMeta
    ) -> tuple[
        tuple[int, ...],
        Callable[[np.ndarray], np.ndarray],
        Callable[[np.ndarray], np.ndarray],
    ]:
        shape_dims = (
            n_state,
            axes_meta.axis_sizes.get("subgroup", 1),
            *[
                axes_meta.axis_sizes[a]
                for a in axes_meta.axis_order
                if a not in {"state", "subgroup"}
            ],
        )

        def _flatten(state_tensor: np.ndarray) -> np.ndarray:
            arr = np.asarray(state_tensor, dtype=np.float64)
            return arr.reshape(-1)

        def _unflatten(state_flat: np.ndarray) -> np.ndarray:
            arr = np.asarray(state_flat, dtype=np.float64)
            return arr.reshape(shape_dims)

        return shape_dims, _flatten, _unflatten


__all__ = ["OpSystemSystem", "__version__"]
