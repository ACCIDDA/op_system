"""Optional native state-derivative actions on compiled RHS and observables."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from numbers import Real
from typing import TYPE_CHECKING, Literal, Protocol, cast

from op_system._array import array_namespace
from op_system.identity import StructuralModelIdentity

if TYPE_CHECKING:
    from collections.abc import Callable

    from op_system.compile import CompiledRhs

__all__ = ["NativeStateActions", "state_actions"]


class _StateFunction(Protocol):
    def __call__(self, t: object, y: object, **params: object) -> object: ...


class _Transforms(Protocol):
    def jvp(
        self,
        function: Callable[[object], object],
        primals: tuple[object],
        tangents: tuple[object],
    ) -> tuple[object, object]: ...

    def vjp(
        self, function: Callable[[object], object], primal: object
    ) -> tuple[object, Callable[[object], tuple[object]]]: ...


@dataclass(frozen=True, slots=True)
class NativeStateActions:
    """Caller-owned JAX/Torch state JVP and transpose/VJP adapters.

    The transform acts directly on a compiled callable, without requesting a
    numerical Jacobian matrix. Forward/reverse AD storage and compiler memory
    remain backend/workload dependent. No state, bound parameter, reverse tape
    or solve cache survives a call. Parameters and time are fixed for the state
    action; separately qualified outer native differentiation can retain them.

    Attributes:
        model: Existing compiler source identity in natural state coordinates.
        backend: Explicit optional native AD backend.
        observable: Named observable, or None for the RHS.
        state_names: Ordered flat current-state coordinates.
        parameter_names: Required runtime inputs for this target callable.
        output_shape: RHS vector or declared observable shape.
    """

    model: StructuralModelIdentity
    backend: Literal["jax", "torch"]
    observable: str | None
    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    output_shape: tuple[int, ...]
    _function: _StateFunction = field(repr=False, compare=False)
    _transforms: _Transforms = field(repr=False, compare=False)

    def _native(self, value: object, state: object | None, name: str) -> None:
        xp = array_namespace(value)
        namespace = getattr(xp, "__name__", "")
        expected = (
            {"jax.numpy"}
            if self.backend == "jax"
            else {"torch", "array_api_compat.torch"}
        )
        dtype = getattr(value, "dtype", None)
        # NumPy/JAX dtype instances compare equal to scalar classes but need
        # not share their hashes; set membership would reject valid arrays.
        if namespace not in expected or dtype not in (xp.float32, xp.float64):  # ruff: ignore[literal-membership] - equal dtype hashes can differ
            msg = f"{name} requires native {self.backend} float32/64 arrays"
            raise ValueError(msg)
        if state is not None and (
            dtype != getattr(state, "dtype", None)
            or (
                self.backend == "torch"
                and getattr(value, "device", None) != getattr(state, "device", None)
            )
        ):
            msg = f"{name} precision or Torch device differs from state"
            raise ValueError(msg)

    def _binding(
        self, t: object, y: object, parameters: dict[str, object]
    ) -> Callable[[object], object]:
        self._native(y, None, "state")
        if getattr(y, "shape", None) != (len(self.state_names),):
            msg = "State shape differs from the declared flat layout"
            raise ValueError(msg)
        if set(parameters) != set(self.parameter_names):
            msg = "Action parameter names must match the target input layout"
            raise ValueError(msg)
        for name, value in parameters.items():
            self._native(value, y, name)
        if isinstance(t, Real) and not isinstance(t, bool):
            pass
        else:
            # JAX traces plain real times, including integer zero, as weak
            # scalars. Preserve native promotion with either float precision.
            weak_time = (
                self.backend == "jax"
                and getattr(t, "weak_type", False)
                and getattr(getattr(t, "dtype", None), "kind", None) in {"i", "u", "f"}
            )
            if not weak_time:
                self._native(t, y, "time")
            if getattr(t, "shape", None) != ():
                msg = "Action time must be scalar"
                raise ValueError(msg)

        def function(state: object) -> object:
            output = self._function(t, state, **parameters)
            if getattr(output, "shape", None) != self.output_shape:
                msg = "Compiled action output differs from its declared shape"
                raise ValueError(msg)
            return output

        return function

    def jvp(
        self, t: object, y: object, tangent: object, **parameters: object
    ) -> object:
        """Apply the target's state Jacobian to a flat native tangent.

        Returns:
            Native output with the target's declared output shape.

        Raises:
            ValueError: Input names, layout, precision or backend differ.
        """
        function = self._binding(t, y, parameters)
        self._native(tangent, y, "tangent")
        if getattr(tangent, "shape", None) != (len(self.state_names),):
            msg = "Tangent shape differs from the flat state layout"
            raise ValueError(msg)
        return self._transforms.jvp(function, (y,), (tangent,))[1]

    def vjp(
        self, t: object, y: object, cotangent: object, **parameters: object
    ) -> object:
        """Apply the target's transpose state Jacobian to a native cotangent.

        Returns:
            Native flat state cotangent. The reverse pullback is call-local.

        Raises:
            ValueError: Input names, layout, precision or backend differ.
        """
        function = self._binding(t, y, parameters)
        self._native(cotangent, y, "cotangent")
        if getattr(cotangent, "shape", None) != self.output_shape:
            msg = "Cotangent shape differs from the target output layout"
            raise ValueError(msg)
        _, pullback = self._transforms.vjp(function, y)
        return pullback(cotangent)[0]


def state_actions(
    compiled: CompiledRhs,
    *,
    backend: Literal["jax", "torch"],
    observable: str | None = None,
) -> NativeStateActions:
    """Bind optional native state actions without constructing a dense Jacobian.

    Runtime inputs are homogeneous native float32/64 arrays. Plain real scalar
    time is also accepted. Parameter shapes retain the compiler's validation.
    Array observations use their original declared shape, not a flattened guess.
    The bounded dense symbolic API remains available separately for NumPy.

    Args:
        compiled: Unmodified compiler-produced source in natural coordinates.
        backend: JAX or Torch; imported only when explicitly requested.
        observable: One compiled observable name, or None for the RHS.

    Returns:
        Static source/layout adapter with call-local native forward/reverse AD.

    Raises:
        ValueError: Backend, source or requested observable is invalid.
        NotImplementedError: Operators/history/gauge-fixed execution is declared.
        ImportError: The selected optional native backend is unavailable.
    """
    if backend not in {"jax", "torch"}:
        msg = "Native state actions require explicit jax or torch backend"
        raise ValueError(msg)
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    if rhs is None:
        msg = "Native state actions require retained compiler source"
        raise ValueError(msg)
    if tuple(rhs.state_names) != compiled.state_names:
        msg = "Native action source and compiled state layouts differ"
        raise ValueError(msg)
    if (
        compiled.operators
        or compiled.history_requirements
        or compiled.gauge_fix is not None
    ):
        msg = "Native actions currently exclude operators/history/gauge-fixed execution"
        raise NotImplementedError(msg)
    model = StructuralModelIdentity.from_compiled(compiled)
    function: _StateFunction = compiled.eval_fn
    shape: tuple[int, ...] = (len(compiled.state_names),)
    names = tuple(
        dict.fromkeys((
            *compiled.param_names,
            *(name for name, _ in rhs.shaped_params),
            *(name for name, _ in rhs.time_varying_params),
        ))
    )
    if observable is not None:
        if observable not in compiled.observables:
            msg = "Unknown compiled observable target"
            raise ValueError(msg)
        target = compiled.observables[observable]
        function = target.eval_fn
        shape = target.output_shape
        names = tuple(name for name, _ in target.parameter_shapes)
    try:
        module = import_module(backend)
    except ImportError as error:
        msg = f"Native actions require the optional op_system[{backend}] dependency"
        raise ImportError(msg) from error
    transforms = cast("_Transforms", module if backend == "jax" else module.func)
    return NativeStateActions(
        model,
        backend,
        observable,
        compiled.state_names,
        names,
        shape,
        function,
        transforms,
    )
