"""Conservative input dependencies from retained compiler IR, without derivatives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from op_system._infer_reactants import (
    NotInferableError,
    StateReference,
    state_dependencies,
)
from op_system._ir import Subscript, Sym, walk
from op_system._ir_templates import inline_aliases
from op_system._observables import _axis_coordinates
from op_system.identity import StructuralModelIdentity

if TYPE_CHECKING:
    from collections.abc import Iterable

    from op_system._ir import Expr
    from op_system.compile import CompiledRhs
    from op_system.specs import NormalizedRhs

__all__ = [
    "BlockStructure",
    "DependencyRows",
    "DependencyStructure",
    "dependency_structure",
]


@dataclass(frozen=True, slots=True)
class DependencyRows:
    """Conservative input-column indices for each named output group.

    Each RHS output is one expanded scalar state derivative. Each observable
    output is one complete named scalar or array; its row unions dependencies
    across all cells. Parameter columns identify runtime input buffers, not
    expanded scalar parameter coordinates. A listed edge may have zero numerical
    derivative; an absent edge excludes a current-input read in this scope.
    """

    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    input_names: tuple[str, ...]
    rows: tuple[tuple[int, ...], ...]
    time_dependent: tuple[bool, ...]

    @property
    def edge_count(self) -> int:
        """Number of retained conservative edges."""
        return sum(map(len, self.rows))

    def to_dict(self) -> dict[str, object]:
        """Return detached primitive JSON fields.

        Returns:
            Output shapes, ordered inputs, row indices and time flags.
        """
        return {
            "output_names": list(self.output_names),
            "output_shapes": [list(shape) for shape in self.output_shapes],
            "input_names": list(self.input_names),
            "rows": [list(row) for row in self.rows],
            "time_dependent": list(self.time_dependent),
        }


@dataclass(frozen=True, slots=True)
class BlockStructure:
    """Immutable snapshot of one existing compiler-checked block-axis proof.

    Scalar parameters are implicitly broadcast. Shaped parameter positions and
    their time-axis convention retain the existing BlockAxisInfo meaning.
    No second separability analysis is performed by the export.
    """

    name: str
    size: int
    state_axis_positions: tuple[tuple[str, int], ...]
    parameter_axis_positions: tuple[tuple[str, int | None], ...]


@dataclass(frozen=True, slots=True)
class DependencyStructure:
    """Bounded compiler dependency report with source and coverage identity.

    Unsupported coverage has None patterns, distinct from supported empty rows.
    Observations/priors/noise and numerical operators belong to their consuming
    contracts. This report supplies no statistical-factorization or rank proof.
    """

    model: StructuralModelIdentity
    status: Literal["reported", "unsupported"]
    reasons: tuple[str, ...]
    state: DependencyRows | None
    parameters: DependencyRows | None
    observable_state: DependencyRows | None
    observable_parameters: DependencyRows | None
    parameter_axes: tuple[tuple[str, tuple[str, ...]], ...]
    block_axes: tuple[BlockStructure, ...]
    limits: tuple[int, int, int]

    def to_json(self) -> str:
        """Serialize a deterministic primitive report with the existing model ID.

        Returns:
            Schema-1 report text. Saved evidence still requires independently
            expected source identity and a caller-owned loading contract.
        """
        return json.dumps(
            {
                "format": "op-system-dependencies",
                "schema_version": 1,
                "model": json.loads(self.model.to_json()),
                "coordinates": "natural_compiled_state",
                "status": self.status,
                "reasons": self.reasons,
                "state": None if self.state is None else self.state.to_dict(),
                "parameters": None
                if self.parameters is None
                else self.parameters.to_dict(),
                "observable_state": None
                if self.observable_state is None
                else self.observable_state.to_dict(),
                "observable_parameters": None
                if self.observable_parameters is None
                else self.observable_parameters.to_dict(),
                "parameter_axes": self.parameter_axes,
                "block_axes": [
                    {
                        "name": block.name,
                        "size": block.size,
                        "state_axis_positions": block.state_axis_positions,
                        "parameter_axis_positions": block.parameter_axis_positions,
                    }
                    for block in self.block_axes
                ],
                "limits": dict(
                    zip(("states", "edges", "ir_nodes"), self.limits, strict=True)
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass
class _Collector:
    rhs: NormalizedRhs
    states: tuple[str, ...]
    parameters: tuple[str, ...]
    max_edges: int
    max_nodes: int
    edges: int = 0
    nodes: int = 0

    def __post_init__(self) -> None:
        self.state_index = {name: i for i, name in enumerate(self.states)}
        self.parameter_index = {name: i for i, name in enumerate(self.parameters)}
        self.templates = {
            template.base: template for template in self.rhs.state_templates
        }
        self.state_axes = {
            **dict.fromkeys(self.states, ()),
            **{name: template.axes for name, template in self.templates.items()},
        }
        self.coordinates = {
            name: list(coords) for name, coords in _axis_coordinates(self.rhs).items()
        }
        self.selections: dict[StateReference, tuple[int, ...]] = {}

    def _columns(self, reference: StateReference) -> tuple[int, ...]:
        if reference not in self.selections:
            columns: tuple[int, ...]
            if reference.base in self.state_index:
                columns = (self.state_index[reference.base],)
            else:
                template = self.templates[reference.base]
                columns = tuple(
                    self.state_index[name]
                    for name, coordinates in zip(
                        template.expanded_names, template.coord_assignments, strict=True
                    )
                    if all(
                        coordinates[axis] == coord for axis, coord in reference.pinned
                    )
                )
            self.selections[reference] = columns
        return self.selections[reference]

    def collect(
        self, expression: Expr
    ) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
        expression = inline_aliases(expression, self.rhs.aliases_ir)
        parameter_columns: set[int] = set()
        time = False
        varying = dict(self.rhs.time_varying_params)
        for node in walk(expression):
            self.nodes += 1
            if self.nodes > self.max_nodes:
                msg = "Dependency IR-node limit exceeded"
                raise ValueError(msg)
            if isinstance(node, Sym | Subscript):
                if node.name in self.parameter_index:
                    parameter_columns.add(self.parameter_index[node.name])
                time = time or node.name == "t" or node.name in varying
        state_columns = {
            column
            for reference in state_dependencies(
                expression, state_axes=self.state_axes, axis_lookup=self.coordinates
            )
            for column in self._columns(reference)
        }
        self.edges += len(state_columns) + len(parameter_columns)
        if self.edges > self.max_edges:
            msg = "Dependency edge limit exceeded"
            raise ValueError(msg)
        return tuple(sorted(state_columns)), tuple(sorted(parameter_columns)), time


def _patterns(
    collector: _Collector,
    expressions: Iterable[Expr | None],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[DependencyRows, DependencyRows]:
    """Collect state and buffer-parameter rows from one declared output family.

    Returns:
        Ordered conservative state/parameter dependency patterns.

    Raises:
        ValueError: An output expression or its declared layout is incomplete.
    """
    reads: list[tuple[tuple[int, ...], tuple[int, ...], bool]] = []
    for expression in expressions:
        if expression is None or len(reads) >= len(names):
            msg = "Dependency export requires complete output IR/layout"
            raise ValueError(msg)
        reads.append(collector.collect(expression))
    if len(reads) != len(names):
        msg = "Dependency export requires complete output IR/layout"
        raise ValueError(msg)
    times = tuple(read[2] for read in reads)
    return (
        DependencyRows(
            names, shapes, collector.states, tuple(read[0] for read in reads), times
        ),
        DependencyRows(
            names, shapes, collector.parameters, tuple(read[1] for read in reads), times
        ),
    )


def dependency_structure(
    compiled: CompiledRhs,
    *,
    max_states: int = 4096,
    max_edges: int = 262_144,
    max_nodes: int = 1_000_000,
) -> DependencyStructure:
    """Export input-read structure without creating numerical derivative matrices.

    Expanded state rows reuse retained typed IR and the existing state-selector
    analysis. Observable rows union all cells in each named output, including
    explicit reduction axes. Listed edges conservatively survive cancellation
    and zero runtime coefficients. Only existing checked block axes are copied.

    Args:
        compiled: Unmodified compiler-produced artifact with retained source IR.
        max_states: Positive expanded-state limit, checked before traversal.
        max_edges: Positive limit across all four output/input families.
        max_nodes: Positive traversal-node limit after alias/reduction expansion.

    Returns:
        Immutable rows and block snapshots, or explicit unsupported coverage
        with None rows for operators, history, gauge-fixed execution or state
        selections outside the existing selector-analysis contract.

    Raises:
        ValueError: Source/layout or limits are invalid, or a limit is exceeded.
    """
    limits = (max_states, max_edges, max_nodes)
    if any(type(value) is not int or value < 1 for value in limits):
        msg = "Dependency limits must be positive integers, not booleans"
        raise ValueError(msg)
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    if rhs is None or rhs.state_names != compiled.state_names:
        msg = "Dependency export requires matching retained compiler source/layout"
        raise ValueError(msg)
    if len(compiled.state_names) > max_states:
        msg = "Dependency expanded-state limit exceeded"
        raise ValueError(msg)
    model = StructuralModelIdentity.from_compiled(compiled)
    axes: dict[str, tuple[str, ...]] = dict.fromkeys(compiled.param_names, ())
    axes.update(dict(rhs.shaped_params))
    axes.update(dict(rhs.time_varying_params))
    for observable in rhs.observables_ir:
        axes.update(dict(observable.parameter_axes))
    blocks = tuple(
        BlockStructure(
            info.name,
            info.size,
            tuple(sorted(info.state_axis_pos.items())),
            tuple(sorted(info.param_axis_pos.items())),
        )
        for info in compiled.block_axes
    )
    reasons = tuple(
        name
        for name, present in (
            ("numerical_operators", bool(compiled.operators)),
            ("history", bool(compiled.history_requirements)),
            ("gauge_fixed_execution", compiled.gauge_fix is not None),
        )
        if present
    )
    if reasons:
        return DependencyStructure(
            model,
            "unsupported",
            reasons,
            None,
            None,
            None,
            None,
            tuple(axes.items()),
            blocks,
            limits,
        )
    collector = _Collector(rhs, compiled.state_names, tuple(axes), max_edges, max_nodes)
    if len(rhs.equations_ir) != len(compiled.state_names):
        msg = "Dependency export requires complete expanded equation IR"
        raise ValueError(msg)
    try:
        state, parameters = _patterns(
            collector,
            rhs.equations_ir,
            compiled.state_names,
            ((),) * len(compiled.state_names),
        )
        observable_state, observable_parameters = _patterns(
            collector,
            tuple(observable.expression for observable in rhs.observables_ir),
            tuple(observable.definition.name for observable in rhs.observables_ir),
            tuple(
                compiled.observables[observable.definition.name].output_shape
                for observable in rhs.observables_ir
            ),
        )
    except NotInferableError as error:
        return DependencyStructure(
            model,
            "unsupported",
            (f"state_selection:{error}",),
            None,
            None,
            None,
            None,
            tuple(axes.items()),
            blocks,
            limits,
        )
    return DependencyStructure(
        model,
        "reported",
        (),
        state,
        parameters,
        observable_state,
        observable_parameters,
        tuple(axes.items()),
        blocks,
        limits,
    )
