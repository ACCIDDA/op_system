"""op_system._validate — structured validation of RHS specifications.

``validate_spec`` runs normalization, compilation, and vectorization and
returns a ``ValidationReport`` instead of raising, so tools can report every
problem at once. When a state template fails to vectorize it also reports the
distinct expression shapes among that template's cells, which is what the
vectorizer needs to be identical: a template whose cells have different
shapes (for example, only some coordinates receive a pinned transition)
cannot be compiled on the vectorized path.

The cost summary counts the drivers of compile time: expanded states and
coordinate-pinned transitions (see #206).
"""

from __future__ import annotations

import ast
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from op_system._errors import UnsupportedFeatureError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from op_system.specs import NormalizedRhs

_STAGES = ("normalize", "compile", "vectorize")


@dataclass(frozen=True)
class ShapeGroup:
    """Cells of one state template that share an expression shape."""

    cells: int
    example_cell: str
    example_expression: str


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating one op_system specification.

    Attributes:
        stages: ``"passed"``, ``"failed"``, or ``"skipped"`` for normalize,
            compile, and vectorize.
        errors: Error messages from the failed stage.
        cost: Compile-cost drivers (expanded states, templates, transitions,
            coordinate-pinned transitions, operators).
        parameters: Parameters the spec consumes, mapped to their declared
            axes (empty tuple for scalars); operator parameters use
            ``kernel.param_axes`` when declared.
        shape_groups: For templates whose cells have more than one expression
            shape, the distinct shapes with a count and an example cell.
    """

    stages: dict[str, str]
    errors: list[str] = field(default_factory=list)
    cost: dict[str, int] = field(default_factory=dict)
    parameters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    shape_groups: dict[str, list[ShapeGroup]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every stage passed."""
        return all(status == "passed" for status in self.stages.values())


class _ShapeNormalizer(ast.NodeTransformer):
    """Replace cell-specific names and numeric literals with placeholders."""

    @override
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if "__" in node.id:
            return ast.copy_location(
                ast.Name(id=node.id.split("__")[0] + "[cell]"), node
            )
        return node

    @override
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)):
            return ast.copy_location(ast.Constant(value=0), node)
        return node


def _shape_signature(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return f"<unparsed> {expression}"
    return ast.dump(_ShapeNormalizer().visit(tree))


def _shape_groups(rhs: NormalizedRhs) -> dict[str, list[ShapeGroup]]:
    by_template: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for name, expression in zip(rhs.state_names, rhs.equations, strict=True):
        if "__" not in name:
            continue
        base = name.split("__")[0]
        signature = _shape_signature(str(expression))
        by_template.setdefault(base, {}).setdefault(signature, []).append((
            name,
            str(expression),
        ))
    groups: dict[str, list[ShapeGroup]] = {}
    for base, signatures in by_template.items():
        if len(signatures) < 2:
            continue
        groups[base] = sorted(
            (
                ShapeGroup(
                    cells=len(members),
                    example_cell=members[0][0],
                    example_expression=members[0][1],
                )
                for members in signatures.values()
            ),
            key=lambda group: -group.cells,
        )
    return groups


def _cost(spec: Mapping[str, Any], rhs: NormalizedRhs | None) -> dict[str, int]:
    transitions = spec.get("transitions") or []
    pinned = sum(
        1
        for t in transitions
        if isinstance(t, _MappingABC)
        and any("=" in str(t.get(side, "")) for side in ("from", "to"))
    )
    cost = {
        "transitions": len(transitions),
        "coordinate_pinned_transitions": pinned,
        "operators": len(spec.get("operators") or []),
    }
    if rhs is not None:
        cost["expanded_states"] = len(rhs.state_names)
        cost["state_templates"] = len({name.split("__")[0] for name in rhs.state_names})
    return cost


def _parameters(rhs: NormalizedRhs) -> dict[str, tuple[str, ...]]:
    parameters: dict[str, tuple[str, ...]] = dict.fromkeys(rhs.param_names, ())
    for name, axes in rhs.meta.get("shaped_params") or ():
        parameters[str(name)] = tuple(axes)
    for name, axes in rhs.meta.get("time_varying_params") or ():
        parameters[str(name)] = tuple(axes)
    for op in rhs.meta.get("operators") or ():
        if not isinstance(op, _MappingABC):
            continue
        for key in ("velocity", "rate"):
            value = op.get(key)
            if isinstance(value, str) and value.isidentifier():
                parameters.setdefault(value, ())
        kernel = op.get("kernel")
        if not isinstance(kernel, _MappingABC):
            continue
        declared = kernel.get("param_axes")
        declared = declared if isinstance(declared, _MappingABC) else {}
        params = kernel.get("params")
        for value in (params or {}).values() if isinstance(params, _MappingABC) else ():
            if isinstance(value, str) and value.isidentifier():
                parameters[value] = tuple(
                    declared.get(value, parameters.get(value, ()))
                )
    return parameters


def validate_spec(spec: Mapping[str, Any]) -> ValidationReport:
    """Validate an op_system specification without raising.

    Args:
        spec: A raw RHS specification mapping.

    Returns:
        A report with per-stage status, error messages, compile-cost drivers,
        consumed parameters, and expression-shape groups for templates whose
        cells differ.
    """
    from op_system import compile_rhs, normalize_rhs  # ruff: ignore[import-outside-top-level]

    stages = dict.fromkeys(_STAGES, "skipped")
    try:
        rhs = normalize_rhs(spec)
    except Exception as exc:  # ruff: ignore[blind-except] - reported, not raised
        stages["normalize"] = "failed"
        return ValidationReport(
            stages=stages, errors=[str(exc)], cost=_cost(spec, None)
        )
    stages["normalize"] = "passed"
    report_kwargs: dict[str, Any] = {
        "cost": _cost(spec, rhs),
        "parameters": _parameters(rhs),
        "shape_groups": _shape_groups(rhs),
    }
    try:
        compile_rhs(rhs)
    except UnsupportedFeatureError as exc:
        vectorize = "vectorized eval path" in str(exc)
        stages["compile"] = "passed" if vectorize else "failed"
        stages["vectorize"] = "failed" if vectorize else "skipped"
        return ValidationReport(stages=stages, errors=[str(exc)], **report_kwargs)
    except Exception as exc:  # ruff: ignore[blind-except]
        stages["compile"] = "failed"
        return ValidationReport(stages=stages, errors=[str(exc)], **report_kwargs)
    stages["compile"] = "passed"
    stages["vectorize"] = "passed"
    return ValidationReport(stages=stages, **report_kwargs)
