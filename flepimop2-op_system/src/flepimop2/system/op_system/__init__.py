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
import sys
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    cast,
)

import numpy as np
from flepimop2.parameter.abc import ModelStateSpecification, ParameterRequest
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

__version__ = "0.2.0"

if TYPE_CHECKING:
    from collections.abc import Callable

    from flepimop2.axis import AxisCollection
    from flepimop2.typing import Float64NDArray

    from op_system._operators import OperatorDescriptor


class _AxesMeta(NamedTuple):
    axis_order: tuple[str, ...]
    axis_sizes: dict[str, int]
    axis_coords: dict[str, np.ndarray]


class OpSystemSystem(SystemABC, module="flepimop2.system.op_system"):  # noqa: D101
    state_change: StateChangeEnum = StateChangeEnum.FLOW

    spec: dict[str, object] = Field(
        default=..., description="Inline op_system RHS specification (already loaded)"
    )

    model_config = ConfigDict(extra="allow")

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        """Compile `op_system` specification and prepare stepper and shape helpers.

        The compiled ``eval_fn`` is **namespace-polymorphic**: it infers
        its array namespace from the input ``state`` at call time via
        ``state.__array_namespace__()``. Callers therefore pick the
        backend (NumPy, JAX, ...) by passing arrays of that namespace;
        the connector performs no backend coercion.
        """
        del context

        spec_obj = self.spec
        compiled = compile_spec(spec_obj)
        n_state = len(compiled.state_names)

        axes_meta = self._extract_axes_meta(compiled)
        mixing_kernels = self._build_mixing_kernels(compiled, axes_meta.axis_coords)
        shape_dims, flatten_fn, unflatten_fn = self._make_shape_helpers(
            n_state, axes_meta
        )

        operators = self._extract_operators(compiled)
        operator_axis = self._extract_operator_axis(compiled)
        # Preserve the user-declared coordinate labels (typically strings
        # like ``unvaccinated`` / ``age65to100``) alongside the numeric
        # ``axis_coords`` arrays.  Engine plugins use this to translate
        # shaped-IC coord assignments back to integer indices.
        axis_labels: dict[str, tuple[str, ...]] = {}
        for ax in compiled.meta.get("axes", []) or []:
            name = ax.get("name")
            coords = ax.get("coords")
            if isinstance(name, str) and isinstance(coords, (list, tuple)):
                axis_labels[name] = tuple(str(c) for c in coords)
        self.options = {
            **dict(self.options or {}),
            "axis_order": axes_meta.axis_order,
            "axis_sizes": axes_meta.axis_sizes,
            "axis_coords": axes_meta.axis_coords,
            "axis_labels": axis_labels,
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
            shape = getattr(state, "shape", None)
            if shape != (n_state,):
                msg = (
                    "state must be a 1D array matching the spec state length; "
                    f"expected ({n_state},), got {shape}."
                )
                raise ValueError(msg)
            params = dict(mixing_kernels)
            params.update(kwargs)
            return compiled.eval_fn(time, state, **params)

        self._stepper = _stepper
        self._compiled_rhs = compiled  # handy for debugging/adapters

    @override
    def _bind_impl(
        self, params: dict[IdentifierString, Any] | None = None
    ) -> SystemProtocol:
        """Return a stepper with any static parameters partially applied.

        The returned callable additionally exposes ``option(name, default=...)``
        so that engine plugins (notably the diffrax wrapper engine) can read
        compiled metadata — ``state_names``, ``initial_state`` map, axis
        order/coords/sizes, declared operators, etc. — from the bound
        stepper.  This bridges the WrapperEngine contract (which only forwards
        the bound callable, not the System object) with op_system's need to
        publish per-spec layout information to its consumers.
        """
        bound = functools.partial(self._stepper, **(params or {}))
        bound.option = self.option  # type: ignore[attr-defined]
        return cast("SystemProtocol", bound)

    @override
    def requested_parameters(  # noqa: C901, PLR0912
        self,
        axes: AxisCollection,
    ) -> dict[IdentifierString, ParameterRequest]:
        """Declare the parameters consumed by the compiled RHS.

        Returns:
            Mapping of parameter name to a `ParameterRequest`. Names that
            appear as shaped-parameter references (``theta[imm]``) in the
            spec are requested as shaped (one ndarray); time-varying
            parameters (e.g. ``beta[time, age]``) are requested with their
            full axes including the configured time axis.

        Raises:
            ValueError: If an `initial_state` entry has an unexpected type,
                or a shaped IC entry collides with an existing request that
                declares a different axis tuple for the same parameter name.

        Notes:
            Initial-state seed names (referenced from `meta['initial_state']`
            but not necessarily from any rate expression) are also requested
            so engine plugins can assemble the state vector from `params`.
            Mixing-kernel names are computed internally and excluded.
        """
        requests: dict[IdentifierString, ParameterRequest] = {}
        mixing_kernel_names = set((self.options or {}).get("mixing_kernels", {}).keys())
        # Names provided by the compiled-RHS runtime environment, not by config
        # (numpy/jax namespace, simulation time, and built-in summation helpers).
        builtin_names = {"np", "t", "sum_state", "sum_prefix"}
        # Normalize-time synthesized constants (e.g. one-hot masks for pinned
        # transition selectors) are injected into ``params`` by ``compile_rhs``
        # and must not be requested from configuration.
        synth_const_names = set(
            (self._compiled_rhs.meta.get("op_system_synth_constants") or {}).keys()
        )

        # Non-time-varying shaped parameters first.  ``shaped_params`` in
        # meta carries reduced (non-time) axes for tv params, but tv names
        # are emitted separately below with their full axes; skip them here.
        shaped_meta = self._compiled_rhs.meta.get("shaped_params") or ()
        time_varying_meta = self._compiled_rhs.meta.get("time_varying_params") or ()
        time_varying_names = {name for name, _ in time_varying_meta}
        for shaped_name, shaped_axes in shaped_meta:
            if shaped_name in mixing_kernel_names or shaped_name in builtin_names:
                continue
            if shaped_name in synth_const_names:
                continue
            if shaped_name in time_varying_names:
                continue
            requests[shaped_name] = ParameterRequest(
                name=shaped_name, axes=tuple(shaped_axes)
            )

        # Time-varying parameters: a single ParameterRequest per name with
        # the full axes tuple (including the configured time axis).  At
        # call time the compiled wrapper interpolates along the time axis
        # using the time axis's declared ``coords`` as the grid, so the
        # supplied array must be shaped exactly as declared in the spec.
        for tv_name, tv_axes in time_varying_meta:
            if tv_name in mixing_kernel_names or tv_name in builtin_names:
                continue
            requests[tv_name] = ParameterRequest(name=tv_name, axes=tuple(tv_axes))

        for name in self._compiled_rhs.param_names:
            if (
                name in mixing_kernel_names
                or name in builtin_names
                or name in synth_const_names
                or name in requests
            ):
                continue
            requests[name] = ParameterRequest(name=name)
        init_map = (self.options or {}).get("initial_state") or {}
        for entry in init_map.values():
            if isinstance(entry, str):
                # Scalar IC entry: a single shared parameter.
                if entry in requests:
                    continue
                requests[entry] = ParameterRequest(name=entry)
            elif isinstance(entry, Mapping) and "shaped" in entry:
                # Shaped IC entry: one ParameterRequest per (name, axes)
                # tuple, requested with the declared shape so the engine
                # can index into it per state cell using `entry["coords"]`.
                shaped_name = entry["shaped"]
                shaped_axes = tuple(entry.get("axes", ()))
                existing = requests.get(shaped_name)
                if existing is not None:
                    if tuple(existing.axes) != shaped_axes:
                        msg = (
                            f"initial_state shaped entry for "
                            f"{shaped_name!r} declares axes "
                            f"{shaped_axes!r} but the same name was "
                            f"already requested with axes "
                            f"{tuple(existing.axes)!r}"
                        )
                        raise ValueError(msg)
                    continue
                requests[shaped_name] = ParameterRequest(
                    name=shaped_name, axes=shaped_axes
                )
            else:
                msg = (
                    f"initial_state entry has unexpected type "
                    f"{type(entry).__name__!r}; expected a parameter name "
                    "string or a shaped-entry mapping"
                )
                raise ValueError(msg)
        # Operator velocity/rate fields are consumed by engine plugins (not by
        # the compiled rhs eval_fn), but still need to be in the params dict.
        for op in self._compiled_rhs.operators:
            self._collect_operator_param_requests(op, requests)
        return requests

    @staticmethod
    def _collect_operator_param_requests(
        op: OperatorDescriptor,
        requests: dict[IdentifierString, ParameterRequest],
    ) -> None:
        """Add velocity/rate/kernel-param names referenced by `op` to `requests`."""
        for value in (op.velocity, op.rate):
            if (
                value is not None
                and value.isidentifier()
                and value != "t"
                and value not in requests
            ):
                requests[value] = ParameterRequest(name=value)
        if op.kernel is None:
            return
        kernel_params = op.kernel.get("params")
        if not isinstance(kernel_params, Mapping):
            return
        for value in kernel_params.values():
            if (
                isinstance(value, str)
                and value.isidentifier()
                and value not in requests
            ):
                requests[value] = ParameterRequest(name=value)

    @override
    def model_state(
        self,
        axes: AxisCollection,
    ) -> ModelStateSpecification | None:
        """Declare an empty model-state specification.

        Returns:
            An empty `ModelStateSpecification`.

        Notes:
            op_system's compartment vector is not assembled by flepimop2 from
            `ModelStateSpecification.parameter_names` because seed parameters
            are reused across many state cells (e.g. one ``zero_init`` for
            every empty compartment), which `ModelStateSpecification` forbids
            (`parameter_names must be unique`).  Engine plugins instead
            consume ``system.options['initial_state']`` (a
            ``state_name -> param_name`` map) together with the resolved
            ``params`` dict to build the state vector themselves.  Returning
            an empty specification here just keeps the simulator's resolve
            path satisfied without claiming an assembly contract op_system
            does not own.
        """
        return ModelStateSpecification(parameter_names=())

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
    def _extract_operators(
        compiled: CompiledRhs,
    ) -> tuple[OperatorDescriptor, ...]:
        """Return the typed operator descriptors from the compiled spec.

        Returns:
            Tuple of :class:`OperatorDescriptor` instances (empty if none declared).
        """
        return compiled.operators

    @staticmethod
    def _extract_operator_axis(compiled: CompiledRhs) -> str | None:
        """Return the single shared axis name if all operators act on the same axis.

        Returns:
            Shared axis name, or ``None`` if there are zero or multiple axes.
        """
        if not compiled.operators:
            return None
        axes = {op.axis for op in compiled.operators}
        if len(axes) == 1:
            return next(iter(axes))
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
