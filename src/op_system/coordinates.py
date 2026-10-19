"""Explicit mixed identity/log solver coordinates with natural-state outputs.

The original CompiledRhs remains in physical coordinates. Construct a view once,
encode initial states, and use its evaluator for deterministic integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar, cast

from ._array import array_namespace
from ._coordinate_rhs import _quotient_source
from ._errors import UnsupportedFeatureError
from ._gauge_fix import _bind_anchors

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ._typing import Array
    from .compile import CompiledRhs, EvalFn, Float64Array, PytreeEvalFn, StateDict
    from .observables import (
        CompiledObservable,
        ObservableEvalFn,
        ObservablePytreeEvalFn,
    )

_StateT = TypeVar("_StateT")


class CoordinateResult(NamedTuple, Generic[_StateT]):
    """Transformed state and a namespace-native scalar domain flag.

    Attributes:
        state: Array or PyTree after transformation; no floor or clamp is applied.
        valid: Finite state with strictly positive selected physical components.
            For decoding this also flags exp underflow to zero and overflow.
    """

    state: _StateT
    valid: Array


def _array_and_mask(
    state: Array, mask_values: tuple[bool, ...], axis: int
) -> tuple[Array, Array]:
    """Validate static layout/dtype and form a broadcast state mask.

    Returns:
        Native array and mask.

    Raises:
        ValueError: If the component axis is absent or has the wrong length.
        TypeError: If coordinates are not real floating arrays.
    """
    xp = array_namespace(state)
    n_states = len(mask_values)
    if (
        not state.shape
        or axis < -len(state.shape)
        or axis >= len(state.shape)
        or state.shape[axis] != n_states
    ):
        msg = "Coordinate component axis does not match the compiled state layout"
        raise ValueError(msg)
    if not xp.isdtype(state.dtype, "real floating"):
        msg = "State coordinates require real floating arrays in the solver dtype"
        raise TypeError(msg)
    shape = [1] * len(state.shape)
    shape[axis] = n_states
    mask = xp.asarray(mask_values)
    return cast("Array", xp.asarray(state)), cast(
        "Array", xp.reshape(mask, tuple(shape))
    )


def _native_flat(
    state: Array, mask_values: tuple[bool, ...], axis: int, *, encode: bool
) -> CoordinateResult[Array]:
    """Transform selected components without extracting host values.

    Returns:
        Native transformed state and finite/positive-domain flag.
    """
    xp = array_namespace(state)
    values, mask = _array_and_mask(state, mask_values, axis)
    finite = xp.all(xp.isfinite(values))
    if encode:
        valid = xp.logical_and(
            finite, xp.all(xp.logical_or(xp.logical_not(mask), xp.greater(values, 0)))
        )
        safe = xp.where(mask, values, xp.ones_like(values))
        result = xp.where(mask, xp.log(safe), values)
    else:
        safe = xp.where(mask, values, xp.zeros_like(values))
        result = xp.where(mask, xp.exp(safe), values)
        valid = xp.logical_and(finite, xp.all(xp.isfinite(result)))
        valid = xp.logical_and(
            valid, xp.all(xp.logical_or(xp.logical_not(mask), xp.greater(result, 0)))
        )
    return CoordinateResult(cast("Array", result), cast("Array", valid))


def _require_valid(valid: Array, operation: str) -> None:
    """Check an eager flag and point traced callers to native methods.

    Raises:
        TypeError: If a transformed call attempts a host-side check.
        ValueError: If the coordinate domain or representability check fails.
    """
    try:
        accepted = bool(valid)
    except TypeError as exc:
        msg = f"Use {operation}_native inside JIT/vmap and check its native valid flag"
        raise TypeError(msg) from exc
    if not accepted:
        msg = (
            "Log coordinates need finite states and "
            "strictly positive selected physical values"
        )
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StateCoordinateView:
    """Deterministic evaluators in explicitly selected solver coordinates.

    Natural model state, reactions, scientific declarations and observations
    remain available on ``natural``. This view does not export jump artifacts,
    balance laws or original derivative-affinity claims in solver coordinates.

    Attributes:
        natural: Source model, in natural coordinates (possibly gauge-fixed).
        log_indices: Selected expanded flat state components.
        log_bases: State bases using log coordinates, shared over their templates.
        eval_fn: Coordinate derivative, with the source flat evaluator contract.
        pytree_eval_fn: Optional coordinate derivative for shaped state buffers.
        block_pytree_eval_fn: Optional coordinate derivative for stripped blocks.
        observables: Natural-scale observables evaluated from encoded states.

    Examples:
        >>> import numpy as np
        >>> from op_system import compile_spec
        >>> model = compile_spec({
        ...     "kind": "expr",
        ...     "state": ["x"],
        ...     "state_metadata": {"x": {"scale": "log"}},
        ...     "equations": {"x": "-rate*x"},
        ... })
        >>> view = model.coordinate_view()
        >>> z0 = view.encode(np.array([2.0]))
        >>> np.allclose(view.eval_fn(0, z0, rate=0.5), [-0.5])
        True
        >>> np.allclose(view.decode(z0), [2.0])
        True
    """

    natural: CompiledRhs
    log_indices: tuple[int, ...]
    log_bases: tuple[str, ...]
    eval_fn: EvalFn
    pytree_eval_fn: PytreeEvalFn | None
    block_pytree_eval_fn: PytreeEvalFn | None
    observables: Mapping[str, CompiledObservable]
    _mask: tuple[bool, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache component selection once, independently of execution namespace."""
        selected = frozenset(self.log_indices)
        object.__setattr__(
            self,
            "_mask",
            tuple(index in selected for index in range(len(self.natural.state_names))),
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        """Source component order; selected values now store log states."""
        return self.natural.state_names

    @property
    def param_names(self) -> tuple[str, ...]:
        """Source equation parameter requirements, omitting gauge anchors."""
        return self.natural.param_names

    def encode_native(
        self, state: Array, *, state_axis: int = 0
    ) -> CoordinateResult[Array]:
        """Map physical arrays to selected log coordinates, with native flags.

        Args:
            state: Real floating natural state, with component axis in source order.
            state_axis: Component axis; use 1 for output trajectories.

        Returns:
            Encoded state and scalar flag without host extraction. Invalid
            selected inputs produce nonfinite logarithms; no floor is applied.
        """
        return _native_flat(state, self._mask, state_axis, encode=True)

    def encode(self, state: Array, *, state_axis: int = 0) -> Array:
        """Check a physical initial state eagerly before taking logarithms.

        Args:
            state: Finite floating natural state, positive in selected components.
            state_axis: Component axis, in source order.

        Returns:
            Initial state in solver coordinates. Use encode_native under JIT/AD.
        """
        xp = array_namespace(state)
        values, mask = _array_and_mask(state, self._mask, state_axis)
        valid = xp.logical_and(
            xp.all(xp.isfinite(values)),
            xp.all(xp.logical_or(xp.logical_not(mask), xp.greater(values, 0))),
        )
        _require_valid(valid, "encode")
        return self.encode_native(state, state_axis=state_axis).state

    def decode_native(
        self, state: Array, *, state_axis: int = 0
    ) -> CoordinateResult[Array]:
        """Decode natural values and flag finite-precision representation failures.

        Args:
            state: Floating solver coordinates.
            state_axis: Component axis; use 1 for output trajectories.

        Returns:
            Natural values and finite/positive-domain flag. exp can underflow or
            overflow; the flag is not an integration-accuracy assessment.
        """
        return _native_flat(state, self._mask, state_axis, encode=False)

    def decode(self, state: Array, *, state_axis: int = 0) -> Array:
        """Return native decoded values without an eager domain check.

        Args:
            state: Floating solver coordinates.
            state_axis: Component axis; use 1 for output trajectories.

        Returns:
            Natural-scale array, for transformed computations and observations.
            Use decode_native or decode_checked when checking representation.
        """
        return self.decode_native(state, state_axis=state_axis).state

    def decode_checked(self, state: Array, *, state_axis: int = 0) -> Array:
        """Decode and eagerly reject underflow/overflow or invalid values.

        Args:
            state: Floating solver coordinates.
            state_axis: Component axis in source order.

        Returns:
            A finite natural state, positive in selected components.
        """
        result = self.decode_native(state, state_axis=state_axis)
        _require_valid(result.valid, "decode")
        return result.state

    def _tree(
        self, state: Mapping[str, Array], *, encode: bool
    ) -> CoordinateResult[dict[str, Array]]:
        """Transform complete shaped buffers, including stripped block buffers.

        Returns:
            Native state dictionary and scalar domain flag.

        Raises:
            ValueError: If state base names do not match the source model.
            TypeError: If a state is not real floating.
        """
        shapes = self.natural.template_shapes
        if shapes is None or set(state) != set(shapes):
            msg = "Coordinate PyTree needs every declared state base exactly once"
            raise ValueError(msg)
        xp = array_namespace(next(iter(state.values())))
        valid = xp.ones((), dtype=xp.bool)
        result = {}
        for base, value in state.items():
            if not xp.isdtype(value.dtype, "real floating"):
                msg = "State coordinates require real floating buffers"
                raise TypeError(msg)
            current = xp.asarray(value)
            valid = xp.logical_and(valid, xp.all(xp.isfinite(current)))
            if base in self.log_bases:
                physical = current if encode else xp.exp(current)
                valid = xp.logical_and(
                    valid, xp.all(xp.logical_and(xp.isfinite(physical), physical > 0))
                )
                current = xp.log(current) if encode else physical
            result[base] = current
        return CoordinateResult(result, valid)

    def encode_pytree_native(
        self, state: Mapping[str, Array]
    ) -> CoordinateResult[dict[str, Array]]:
        """Encode complete buffers and return native finite/positive flags.

        Returns:
            Solver-coordinate buffers with a native domain flag.
        """
        return self._tree(state, encode=True)

    def decode_pytree_native(
        self, state: Mapping[str, Array]
    ) -> CoordinateResult[dict[str, Array]]:
        """Decode complete buffers and return native representation flags.

        Returns:
            Natural buffers with a native finite/positive flag.
        """
        return self._tree(state, encode=False)

    def bind(
        self, params: Mapping[str, object]
    ) -> Callable[[object, object], Float64Array]:
        """Bind source parameters to the coordinate derivative.

        Returns:
            A two-argument deterministic RHS, with solver-coordinate state.
        """
        values = dict(params)

        def evaluate(t: object, state: object) -> Float64Array:
            return self.eval_fn(t, state, **values)

        return evaluate

    def __reduce__(self) -> tuple[object, tuple[CompiledRhs]]:
        """Rebuild evaluators from the retained natural source when unpickling.

        Returns:
            Module factory and source model, without serialized closures.
        """
        return _build_state_coordinates, (self.natural,)


def _wrap_pytree(
    view: StateCoordinateView, callback: PytreeEvalFn | None
) -> PytreeEvalFn | None:
    """Decode incoming buffers for a quotient or observable evaluator.

    Returns:
        Same callback contract, or None when no source callback exists.
    """
    if callback is None:
        return None

    def evaluate(t: object, y: StateDict, **params: object) -> StateDict:
        decoded = view.decode_pytree_native(cast("Mapping[str, Array]", y)).state
        return callback(t, cast("StateDict", decoded), **params)

    return evaluate


def _wrap_observable(
    view: StateCoordinateView, callback: ObservableEvalFn
) -> ObservableEvalFn:
    """Evaluate a natural observable after decoding flat coordinates.

    Returns:
        Same observation contract, with encoded input state.
    """

    def evaluate(t: object, y: object, **params: object) -> object:
        return callback(t, view.decode(cast("Array", y)), **params)

    return evaluate


def _wrap_observable_tree(
    view: StateCoordinateView, callback: ObservablePytreeEvalFn
) -> ObservablePytreeEvalFn:
    """Evaluate a natural observable after decoding shaped buffers.

    Returns:
        Same observation contract, with encoded buffer inputs.
    """

    def evaluate(t: object, y: Mapping[str, object], **params: object) -> object:
        decoded = view.decode_pytree_native(cast("Mapping[str, Array]", y)).state
        return callback(t, decoded, **params)

    return evaluate


def _build_state_coordinates(natural: CompiledRhs) -> StateCoordinateView:
    """Compile physical quotients, then compose them with native decoding.

    Returns:
        Explicit coordinate view without replacing any natural-model interface.

    Raises:
        ValueError: If no normalized source was retained.
        UnsupportedFeatureError: If separate operator/history composition is needed.
    """
    from .compile import compile_rhs  # ruff: ignore[import-outside-top-level]

    source = natural._rhs  # ruff: ignore[private-member-access]
    if source is None:
        msg = "Coordinate views require a model made by compile_spec or compile_rhs"
        raise ValueError(msg)
    bases = tuple(
        name
        for name, declaration in natural.state_meta.items()
        if declaration.scale == "log"
    )
    indices = tuple(
        index
        for template in source.state_templates
        if template.base in bases
        for index in range(
            template.offset, template.offset + len(template.expanded_names)
        )
    )
    if indices and (natural.operators or natural.history_requirements):
        raise UnsupportedFeatureError(
            feature="log state coordinates",
            detail=(
                "Separate operator/history composition needs "
                "an explicit transformed contract"
            ),
        )
    quotient = (
        compile_rhs(_quotient_source(source, frozenset(indices)))
        if indices
        else natural
    )
    fixed = dict(natural.gauge_fix.fixed_values) if natural.gauge_fix else {}
    flat = (
        cast("EvalFn", _bind_anchors(quotient.eval_fn, fixed))
        if indices and fixed
        else quotient.eval_fn
    )
    tree = quotient.pytree_eval_fn
    block = quotient.block_pytree_eval_fn
    if indices and fixed:
        tree = cast("PytreeEvalFn", _bind_anchors(tree, fixed)) if tree else None
        block = cast("PytreeEvalFn", _bind_anchors(block, fixed)) if block else None
    # Evaluators close over this record after it is constructed. The placeholder
    # callbacks never escape; immutable replacement installs the composed views.
    view = StateCoordinateView(
        natural, indices, bases, flat, tree, block, natural.observables
    )

    def evaluate(t: object, y: object, **params: object) -> Float64Array:
        return flat(t, view.decode(cast("Array", y)), **params)

    observables = {
        name: observable._replace(
            eval_fn=_wrap_observable(view, observable.eval_fn),
            pytree_eval_fn=_wrap_observable_tree(view, observable.pytree_eval_fn),
        )
        for name, observable in natural.observables.items()
    }
    return StateCoordinateView(
        natural,
        indices,
        bases,
        evaluate,
        _wrap_pytree(view, tree),
        _wrap_pytree(view, block),
        MappingProxyType(observables),
    )


__all__ = ["CoordinateResult", "StateCoordinateView"]
