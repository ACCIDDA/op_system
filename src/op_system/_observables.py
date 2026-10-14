"""Named, fixed-mesh observation definitions and native evaluation artifacts."""

from __future__ import annotations

import ast
import keyword
import math
from collections.abc import Mapping
from numbers import Number
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, cast

from op_system._array import array_namespace
from op_system._errors import InvalidRhsSpecError, UnsupportedFeatureError
from op_system._ir import (
    Apply,
    AxisIndex,
    Expr,
    HistoryOp,
    Reduce,
    Subscript,
    Sym,
    ir_to_ast_expr,
    parse_expr_to_ir,
    walk,
)
from op_system._ir import (
    Literal as LiteralExpr,
)
from op_system._ir_lower import UnsupportedIRLoweringError, lower_to_vector_ast
from op_system._scientific_meta import AxisMeasure, StateDeclaration, _axis_declarations

if TYPE_CHECKING:
    from types import CodeType

    from op_system.compile import _Indexable
    from op_system.specs import NormalizedRhs


class ObservableReduction(NamedTuple):
    """Explicit reduction independent of legacy equation quadrature.

    Attributes:
        kind: Unweighted sum or integral with declared physical measures.
        axes: Named axes to remove, in the declared reduction order.
    """

    kind: Literal["sum", "integral"]
    axes: tuple[str, ...]


class ObservableDefinition(NamedTuple):
    """Backend-neutral definition of one instantaneous latent observable.

    Attributes:
        name: Unique output name within a compiled model.
        expression: Restricted pointwise state/parameter/time expression.
        axes: Output axes, after any explicit reduction.
        reduction: Optional unweighted sum or physical integral.
        units: Optional declared output units, without dimensional verification.
        description: Optional human-readable interpretation.
    """

    name: str
    expression: str
    axes: tuple[str, ...] = ()
    reduction: ObservableReduction | None = None
    units: str | None = None
    description: str | None = None

    @classmethod
    def from_mapping(cls, name: str, value: object) -> ObservableDefinition:
        """Validate the bounded observable schema.

        Args:
            name: Output identifier.
            value: Mapping with expr, optional axes/reduction/units/description.

        Returns:
            Detached immutable definition.

        Raises:
            InvalidRhsSpecError: If identifiers, fields or reductions are invalid.
        """
        if not _identifier(name):
            raise InvalidRhsSpecError(detail="observable names must be identifiers")
        if not isinstance(value, Mapping) or set(value) - {
            "expr",
            "axes",
            "reduction",
            "units",
            "description",
        }:
            raise InvalidRhsSpecError(detail=f"observable {name!r} has invalid fields")
        expression = _text(value, "expr", required=True)
        axes = _axes(value.get("axes", ()))
        reduction = None
        if "reduction" in value:
            raw = value["reduction"]
            if not isinstance(raw, Mapping) or set(raw) != {"kind", "axes"}:
                raise InvalidRhsSpecError(
                    detail="observable reduction needs kind and axes"
                )
            kind = raw["kind"]
            reduction_axes = _axes(raw["axes"])
            if (
                not isinstance(kind, str)
                or kind not in {"sum", "integral"}
                or not reduction_axes
            ):
                raise InvalidRhsSpecError(
                    detail="observable reduction must be a nonempty sum or integral"
                )
            reduction = ObservableReduction(
                cast('Literal["sum", "integral"]', kind), reduction_axes
            )
            if set(axes) & set(reduction_axes):
                raise InvalidRhsSpecError(
                    detail="observable output and reduction axes must be disjoint"
                )
        return cls(
            name,
            cast("str", expression),
            axes,
            reduction,
            _text(value, "units"),
            _text(value, "description"),
        )


class ObservableEvalFn(Protocol):
    """Evaluate an instantaneous observable from a native flat state array."""

    def __call__(self, t: object, y: object, **params: object) -> object: ...


class ObservablePytreeEvalFn(Protocol):
    """Evaluate an instantaneous observable from complete shaped state buffers."""

    def __call__(
        self, t: object, y: Mapping[str, object], **params: object
    ) -> object: ...


class CompiledObservable(NamedTuple):
    """Native evaluator and explicit fixed-layout scientific metadata.

    Attributes:
        definition: Backend-neutral expression and reduction description.
        output_shape: Shape in definition.axes order; empty for a scalar.
        coordinates: Coordinate labels for each output axis, in output order.
        state_dependencies: Referenced state base names, in sorted order.
        parameter_shapes: Required parameters and their complete input shapes.
            Scalars have empty shapes; these are separate from RHS requirements.
        time_dependent: Whether the expression explicitly references t.
        measure_requirements: Physical measures applied exactly once per axis.
        eval_fn: Evaluator for a complete flat state vector.
        pytree_eval_fn: Evaluator for complete shaped state buffers.
    """

    definition: ObservableDefinition
    output_shape: tuple[int, ...]
    coordinates: tuple[tuple[str, tuple[str, ...]], ...]
    state_dependencies: tuple[str, ...]
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    time_dependent: bool
    measure_requirements: tuple[tuple[str, AxisMeasure], ...]
    eval_fn: ObservableEvalFn
    pytree_eval_fn: ObservablePytreeEvalFn


class _ObservableIR(NamedTuple):
    definition: ObservableDefinition
    expression: Expr
    state_dependencies: tuple[str, ...]
    parameter_axes: tuple[tuple[str, tuple[str, ...]], ...]
    time_dependent: bool
    measures: tuple[tuple[str, AxisMeasure], ...]


def _identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.isidentifier()
        and not keyword.iskeyword(value)
        and not value.startswith("__")
    )


def _text(
    value: Mapping[str, object], name: str, *, required: bool = False
) -> str | None:
    text = value.get(name)
    if name not in value and not required:
        return None
    if not isinstance(text, str) or not text.strip():
        raise InvalidRhsSpecError(detail=f"observable {name} must be nonempty text")
    return text


def _axes(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not _identifier(axis) for axis in value
    ):
        raise InvalidRhsSpecError(
            detail="observable axes must be a list of identifiers"
        )
    if len(set(value)) != len(value):
        raise InvalidRhsSpecError(detail="observable axes must be unique")
    return tuple(value)


def _axis_coordinates(rhs: NormalizedRhs) -> dict[str, tuple[str, ...]]:
    return {
        axis["name"]: tuple(str(coord) for coord in axis["coords"])
        for axis in rhs.meta.get("axes", ())
    }


def _validate_layout(rhs: NormalizedRhs) -> None:
    bases = [template.base for template in rhs.state_templates]
    if set(bases) & {"np", "t"}:
        raise InvalidRhsSpecError(
            detail="observable state names must not shadow np or t"
        )
    if len(set(bases)) != len(bases) or any(
        not template.axes and template.expanded_names != (template.base,)
        for template in rhs.state_templates
    ):
        raise UnsupportedFeatureError(
            feature="observable partial or duplicate state templates"
        )


def _pin_coordinates(expression: Expr, rhs: NormalizedRhs) -> Expr:
    """Resolve positional literal coordinates against declared buffer axes.

    Returns:
        Compact IR with explicit axis labels for pinned coordinates.
    """
    known = {template.base: template.axes for template in rhs.state_templates}
    known.update(dict(rhs.shaped_params))
    if isinstance(expression, Apply):
        return Apply(
            expression.op, tuple(_pin_coordinates(arg, rhs) for arg in expression.args)
        )
    if not isinstance(expression, Subscript):
        return expression
    axes = known.get(expression.name)
    if axes is None or len(axes) != len(expression.indices):
        return expression
    return Subscript(
        expression.name,
        tuple(
            AxisIndex(axis, index.coord)
            if not index.axis and index.coord is not None
            else index
            for axis, index in zip(axes, expression.indices, strict=True)
        ),
    )


class _ReferenceContext(NamedTuple):
    states: Mapping[str, tuple[str, ...]]
    parameters: Mapping[str, tuple[str, ...]]
    coordinates: Mapping[str, tuple[str, ...]]
    forbidden: frozenset[str]
    time_varying: frozenset[str]


def _validate_operation(node: Expr) -> None:
    """Reject expression operations outside the initially qualified subset.

    Raises:
        UnsupportedFeatureError: If literals, reductions or operations are unsupported.
    """
    if isinstance(node, LiteralExpr) and isinstance(node.value, (str, bool)):
        raise UnsupportedFeatureError(feature="nonnumeric observable literals")
    if isinstance(node, Apply):
        arity = 1 if node.op in {"neg", "pos", "np.exp"} else 2
        if len(node.args) != arity:
            raise UnsupportedFeatureError(feature="observable operation arity")
    if isinstance(node, (Reduce, HistoryOp)) or (
        isinstance(node, Apply)
        and node.op not in {"+", "-", "*", "/", "pow", "neg", "pos", "np.exp"}
    ):
        raise UnsupportedFeatureError(
            feature="observable expression operation",
            detail=(
                "Use pointwise arithmetic/np.exp and the explicit reduction field; "
                "aliases, history and nested reductions are deferred."
            ),
        )


def _validate_reference_name(name: str, context: _ReferenceContext) -> None:
    """Validate a state, parameter or time reference.

    Raises:
        InvalidRhsSpecError: If a reference is a reserved or invalid symbol.
        UnsupportedFeatureError: If it needs an alias or time-buffer adapter.
    """
    if not _identifier(name) or name == "np" or name in context.coordinates:
        raise InvalidRhsSpecError(detail=f"invalid observable symbol {name!r}")
    if name in context.forbidden:
        raise UnsupportedFeatureError(
            feature="observable alias references", detail=name
        )
    if name in context.time_varying:
        raise UnsupportedFeatureError(
            feature="observable time-varying parameter buffers", detail=name
        )


def _reference_axes(
    node: Sym | Subscript, context: _ReferenceContext, targets: tuple[str, ...]
) -> tuple[str, ...]:
    """Resolve a reference without implicit spatial broadcasting of bare buffers.

    Returns:
        Complete parameter axes or declared state axes.

    Raises:
        InvalidRhsSpecError: If a subscript or bare buffer is inconsistent.
    """
    known = context.states.get(node.name, context.parameters.get(node.name))
    if isinstance(node, Sym):
        if known:
            raise InvalidRhsSpecError(
                detail=f"observable array {node.name!r} requires its named axes"
            )
        return ()
    axes = tuple(index.axis for index in node.indices)
    if any(axis not in context.coordinates for axis in axes) or len(set(axes)) != len(
        axes
    ):
        raise InvalidRhsSpecError(
            detail=f"observable subscript {node.name!r} requires unique named axes"
        )
    for index in node.indices:
        if index.coord is None and index.axis not in targets:
            raise InvalidRhsSpecError(
                detail=f"observable axis {index.axis!r} is neither output nor reduced"
            )
        if (
            index.coord is not None
            and index.coord not in context.coordinates[index.axis]
        ):
            raise InvalidRhsSpecError(
                detail=f"observable coordinate {index.coord!r} is not declared"
            )
    if known is not None and set(known) != set(axes):
        raise InvalidRhsSpecError(
            detail=f"observable axes for {node.name!r} disagree with its declaration"
        )
    return context.parameters.get(node.name, axes)


def _expression_dependencies(
    expression: Expr,
    *,
    rhs: NormalizedRhs,
    targets: tuple[str, ...],
    observable_names: set[str],
) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...], bool]:
    states = {template.base: template.axes for template in rhs.state_templates}
    aliases = {template.base for template in rhs.alias_templates} | set(rhs.aliases)
    parameters: dict[str, tuple[str, ...]] = dict.fromkeys(rhs.param_names, ())
    parameters.update(dict(rhs.shaped_params))
    forbidden = aliases | (observable_names - set(states) - set(parameters) - {"t"})
    context = _ReferenceContext(
        states,
        parameters,
        _axis_coordinates(rhs),
        frozenset(forbidden),
        frozenset(dict(rhs.time_varying_params)),
    )
    state_dependencies: set[str] = set()
    parameter_axes: dict[str, tuple[str, ...]] = {}
    time_dependent = False
    for node in walk(expression):
        _validate_operation(node)
        if not isinstance(node, (Sym, Subscript)):
            continue
        _validate_reference_name(node.name, context)
        if node.name == "t":
            if isinstance(node, Subscript):
                raise InvalidRhsSpecError(detail="observable t is scalar")
            time_dependent = True
            continue
        axes = _reference_axes(node, context, targets)
        if node.name in states:
            state_dependencies.add(node.name)
        elif parameter_axes.setdefault(node.name, axes) != axes:
            raise InvalidRhsSpecError(
                detail=f"observable parameter {node.name!r} has inconsistent axes"
            )
    return (
        tuple(sorted(state_dependencies)),
        tuple(sorted(parameter_axes.items())),
        time_dependent,
    )


def _physical_measures(
    definition: ObservableDefinition, states: tuple[str, ...], rhs: NormalizedRhs
) -> tuple[tuple[str, AxisMeasure], ...]:
    reduction = definition.reduction
    if reduction is None or reduction.kind != "integral":
        return ()
    declared_axes = _axis_declarations(rhs.meta.get("axes", ()))
    measures = []
    for axis in reduction.axes:
        measure = declared_axes[axis].measure
        if measure is None:
            raise InvalidRhsSpecError(
                detail=f"observable integral requires an explicit measure on {axis!r}"
            )
        measures.append((axis, measure))
    physical_axes = {axis for axis, measure in measures if measure.kind != "counting"}
    metadata: Mapping[str, StateDeclaration] = rhs.meta.get("state_declarations", {})
    required_axes: set[str] = set()
    for state in states:
        declaration = metadata.get(state)
        if declaration is None or declaration.representation not in {
            "density",
            "concentration",
        }:
            raise InvalidRhsSpecError(
                detail=(
                    f"observable integral requires intensive metadata for {state!r}; "
                    "use sum for extensive counts/mass/counters"
                )
            )
        if declaration.measure_axes is None:
            raise InvalidRhsSpecError(
                detail=f"observable integral measures disagree with {state!r}"
            )
        required_axes.update(set(declaration.measure_axes) & set(reduction.axes))
    if states and required_axes != physical_axes:
        raise InvalidRhsSpecError(
            detail="observable integral has inconsistent physical axes"
        )
    return tuple(measures)


def _normalize_observables(
    value: object, *, rhs: NormalizedRhs
) -> tuple[_ObservableIR, ...]:
    """Normalize explicitly requested observables without changing RHS symbols.

    Returns:
        Immutable compact-IR definitions in declaration order.

    Raises:
        InvalidRhsSpecError: If schema, axes, dependencies or measures are invalid.
    """
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise InvalidRhsSpecError(detail="observables must be a declaration mapping")
    if not value:
        return ()
    _validate_layout(rhs)
    coordinates = _axis_coordinates(rhs)
    result = []
    for name, raw in value.items():
        definition = ObservableDefinition.from_mapping(name, raw)
        targets = definition.axes + (
            definition.reduction.axes if definition.reduction else ()
        )
        if any(axis not in coordinates for axis in targets):
            raise InvalidRhsSpecError(
                detail=f"observable {name!r} references an unknown axis"
            )
        expression = _pin_coordinates(
            parse_expr_to_ir(definition.expression, lower_helpers=True), rhs
        )
        states, parameters, time = _expression_dependencies(
            expression, rhs=rhs, targets=targets, observable_names=set(value)
        )
        result.append(
            _ObservableIR(
                definition,
                expression,
                states,
                parameters,
                time,
                _physical_measures(definition, states, rhs),
            )
        )
    layouts: dict[str, tuple[str, ...]] = dict.fromkeys(rhs.param_names, ())
    layouts.update(dict(rhs.shaped_params))
    for artifact in result:
        for name, axes in artifact.parameter_axes:
            if layouts.setdefault(name, axes) != axes:
                raise InvalidRhsSpecError(
                    detail=f"observable parameter {name!r} has inconsistent axes"
                )
    buffers = {f"{template.base}_buf" for template in rhs.state_templates}
    buffers.update(f"{name}_buf" for name, axes in layouts.items() if axes)
    if buffers & (set(layouts) | {template.base for template in rhs.state_templates}):
        raise InvalidRhsSpecError(
            detail="observable names collide with buffer bindings"
        )
    return tuple(result)


def _native_value(value: object, reference: object) -> object:
    """Convert host constants or cast native arrays without detaching gradients.

    Returns:
        Value in the reference namespace and dtype.

    Raises:
        TypeError: If a numerical array belongs to a different namespace.
    """
    xp = array_namespace(reference)
    dtype = getattr(reference, "dtype", None)
    if isinstance(value, Number):
        return xp.asarray(value, dtype=dtype)
    try:
        namespace = array_namespace(value)
    except TypeError:
        return xp.asarray(value, dtype=dtype)
    if namespace is not xp:
        message = "observable array parameters must use the state array namespace"
        raise TypeError(message)
    return xp.astype(value, dtype, copy=False)


def _state_environment(
    t: object, y: Mapping[str, object], rhs: NormalizedRhs
) -> tuple[object, dict[str, object]]:
    """Validate complete shaped buffers and build the restricted evaluation environment.

    Returns:
        Reference array and namespace/state bindings.

    Raises:
        TypeError: If the state dtype is not real floating.
        ValueError: If buffers disagree with the declared native layout.
    """
    if set(y) != {template.base for template in rhs.state_templates}:
        message = "observable state buffers must match all compiled state bases"
        raise ValueError(message)
    reference = y[rhs.state_templates[0].base]
    xp = array_namespace(reference)
    dtype = getattr(reference, "dtype", None)
    if not xp.isdtype(dtype, "real floating"):
        message = "observable states must have a real floating dtype"
        raise TypeError(message)
    env: dict[str, object] = {"np": xp, "t": _native_value(t, reference)}
    for template in rhs.state_templates:
        value = y[template.base]
        if (
            array_namespace(value) is not xp
            or getattr(value, "dtype", None) != dtype
            or getattr(value, "shape", None) != template.shape
        ):
            message = f"observable state {template.base!r} has incompatible layout"
            raise ValueError(message)
        env[f"{template.base}_buf"] = value
        if not template.axes:
            env[template.base] = value
    return reference, env


def _parameter_environment(
    params: Mapping[str, object],
    shapes: Mapping[str, tuple[int, ...]],
    reference: object,
) -> dict[str, object]:
    """Bind only required observation parameters in the active namespace.

    Returns:
        Scalar parameter and shaped-buffer bindings.

    Raises:
        ValueError: If a required parameter is missing or has the wrong shape.
    """
    env = {}
    for name, shape in shapes.items():
        if name not in params:
            message = f"observable requires parameter {name!r}"
            raise ValueError(message)
        value = _native_value(params[name], reference)
        if getattr(value, "shape", None) != shape:
            message = f"observable parameter {name!r} must have shape {shape}"
            raise ValueError(message)
        env[name if not shape else f"{name}_buf"] = value
    return env


def _reduce_result(
    value: object,
    reference: object,
    artifact: _ObservableIR,
    coordinates: Mapping[str, tuple[str, ...]],
) -> object:
    """Apply measures and reductions with functional native operations.

    Returns:
        Native value with exactly the declared output shape.
    """
    xp = array_namespace(reference)
    dtype = getattr(reference, "dtype", None)
    targets = artifact.definition.axes + (
        artifact.definition.reduction.axes if artifact.definition.reduction else ()
    )
    shape = tuple(len(coordinates[axis]) for axis in targets)
    result = xp.broadcast_to(_native_value(value, reference), shape)
    for axis, measure in artifact.measures:
        index = targets.index(axis)
        weights_shape = (
            (1,) * index + (len(measure.weights),) + (1,) * (len(targets) - index - 1)
        )
        weights = xp.reshape(xp.asarray(measure.weights, dtype=dtype), weights_shape)
        result = xp.multiply(result, weights)
    if artifact.definition.reduction:
        result = xp.sum(
            result, axis=tuple(range(len(artifact.definition.axes), len(targets)))
        )
    return result


def _make_evaluators(
    code: CodeType, artifact: _ObservableIR, rhs: NormalizedRhs
) -> tuple[ObservableEvalFn, ObservablePytreeEvalFn]:
    coordinates = _axis_coordinates(rhs)
    parameter_shapes = {
        name: tuple(len(coordinates[axis]) for axis in axes)
        for name, axes in artifact.parameter_axes
    }

    def pytree_eval_fn(t: object, y: Mapping[str, object], **params: object) -> object:
        reference, env = _state_environment(t, y, rhs)
        env.update(_parameter_environment(params, parameter_shapes, reference))
        # This is compiler-generated code after the shared restricted-AST validator.
        result = eval(code, {"__builtins__": {}}, env)  # ruff: ignore[suspicious-eval-usage]
        return _reduce_result(result, reference, artifact, coordinates)

    def eval_fn(t: object, y: object, **params: object) -> object:
        xp = array_namespace(y)
        if getattr(y, "shape", None) != (len(rhs.state_names),):
            message = "observable flat state shape must match compiled state_names"
            raise ValueError(message)
        vector = cast("_Indexable", y)
        states = {
            template.base: xp.reshape(
                vector[template.offset : template.offset + math.prod(template.shape)],
                template.shape,
            )
            for template in rhs.state_templates
        }
        return pytree_eval_fn(t, states, **params)

    return eval_fn, pytree_eval_fn


def _lower_observable_expression(
    expression: Expr,
    targets: tuple[str, ...],
    buffers: Mapping[str, tuple[str, ...]],
    coordinates: Mapping[str, tuple[str, ...]],
) -> ast.expr:
    """Reuse operation ASTs and vector buffer lowering for the bounded subset.

    Returns:
        Native pointwise expression aligned with output/reduction axes.

    Raises:
        UnsupportedFeatureError: If an operation has no supported AST layout.
    """
    if not isinstance(expression, Apply):
        return lower_to_vector_ast(
            expression,
            target_axes=targets,
            buffer_axes=buffers,
            axis_names=frozenset(coordinates),
            axis_coords=coordinates,
        )
    arguments = [
        _lower_observable_expression(arg, targets, buffers, coordinates)
        for arg in expression.args
    ]
    body = ir_to_ast_expr(Apply(expression.op, (LiteralExpr(0),) * len(arguments)))
    if isinstance(body, ast.BinOp):
        body.left, body.right = arguments
    elif isinstance(body, ast.UnaryOp):
        body.operand = arguments[0]
    elif isinstance(body, ast.Call):
        body.args = arguments
    else:
        raise UnsupportedFeatureError(feature="observable operation AST layout")
    return body


def _compile_observables(rhs: NormalizedRhs) -> dict[str, CompiledObservable]:
    """Compile every requested definition or fail before returning an artifact.

    Returns:
        Native evaluators with concrete shapes and detached scientific metadata.

    Raises:
        UnsupportedFeatureError: If expression lowering is unsupported.
    """
    if not rhs.observables_ir:
        return {}
    # The expression validator lives with the RHS compiler. Importing here
    # avoids a module cycle while keeping one shared restricted-AST policy.
    from op_system.compile import _compile_expr  # ruff: ignore[import-outside-top-level]

    coordinates = _axis_coordinates(rhs)
    result = {}
    for artifact in rhs.observables_ir:
        definition = artifact.definition
        _compile_expr(definition.expression, artifact.expression)
        targets = definition.axes + (
            definition.reduction.axes if definition.reduction else ()
        )
        try:
            body = _lower_observable_expression(
                artifact.expression,
                targets,
                {
                    **{
                        template.base: template.axes for template in rhs.state_templates
                    },
                    **dict(artifact.parameter_axes),
                },
                coordinates,
            )
        except UnsupportedIRLoweringError as error:
            raise UnsupportedFeatureError(
                feature="observable expression lowering", detail=str(error)
            ) from error
        tree = ast.Expression(body=body)
        ast.fix_missing_locations(tree)
        code = compile(
            tree, filename=f"<op_system_observable:{definition.name}>", mode="eval"
        )
        eval_fn, pytree_eval_fn = _make_evaluators(code, artifact, rhs)
        result[definition.name] = CompiledObservable(
            definition,
            tuple(len(coordinates[axis]) for axis in definition.axes),
            tuple((axis, coordinates[axis]) for axis in definition.axes),
            artifact.state_dependencies,
            tuple(
                (name, tuple(len(coordinates[axis]) for axis in axes))
                for name, axes in artifact.parameter_axes
            ),
            artifact.time_dependent,
            artifact.measures,
            eval_fn,
            pytree_eval_fn,
        )
    return result
