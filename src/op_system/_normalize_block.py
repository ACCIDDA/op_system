"""op_system._normalize_block.

Block-axis stripping: produce a :class:`~op_system._normalize.NormalizedRhs`
with one block axis removed so the vectorizer can compile a per-block-coord
eval function.

The strategy is to keep the *original* expanded cell names (e.g.
``S__age_y__loc_a``) in the stripped templates but with the block axis removed
from the ``axes`` and ``shape`` tuples, and to include only the cells whose
``coord_assignments[i][axis_name]`` equals the *first* coordinate of the block
axis.  This lets :func:`~op_system._ir_lower.lift_cell_ir_to_template` map the
original ``Sym`` leaves to their stripped templates without any renaming.

Public API
----------
- :func:`strip_block_axis` - remove a single block axis from a
  :class:`~op_system._normalize.NormalizedRhs`.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from op_system._errors import UnsupportedFeatureError
from op_system._ir import Apply, Literal, Reduce, Subscript, Sym

if TYPE_CHECKING:
    from collections.abc import Mapping

    from op_system._ir import Expr
    from op_system._normalize import NormalizedRhs
    from op_system._normalize_ir import StateTemplate


def _strip_axis_from_ir(expr: Expr, axis_name: str) -> Expr:  # noqa: PLR0911
    """Recursively remove *axis_name* from all Subscript index lists.

    Transitions-kind specs lift state references to template-form
    ``Subscript`` nodes during normalization.  After the block axis is
    stripped from the templates, the ``Subscript`` nodes still carry that
    axis in their ``indices`` tuple.  This function removes those
    ``AxisIndex`` entries so the downstream vectorizer sees a consistent
    picture: templates with ``(age,)`` axes and subscripts with only an
    ``age`` index.

    Args:
        expr: IR expression to rewrite.
        axis_name: Axis name to remove from all ``Subscript`` index lists.

    Returns:
        A new IR expression with the axis stripped; structurally equal to
        ``expr`` when no stripping occurs.
    """
    if isinstance(expr, (Literal, Sym)):
        return expr
    if isinstance(expr, Subscript):
        new_indices = tuple(idx for idx in expr.indices if idx.axis != axis_name)
        if new_indices == expr.indices:
            return expr
        return Subscript(name=expr.name, indices=new_indices)
    if isinstance(expr, Apply):
        new_args = tuple(_strip_axis_from_ir(a, axis_name) for a in expr.args)
        if new_args == expr.args:
            return expr
        return Apply(op=expr.op, args=new_args)
    if isinstance(expr, Reduce):
        new_body = _strip_axis_from_ir(expr.body, axis_name)
        if new_body is expr.body:
            return expr
        return Reduce(
            kind=expr.kind,
            bindings=expr.bindings,
            body=new_body,
            filters=expr.filters,
            kernel=expr.kernel,
        )
    return expr


def _strip_template(
    tpl: StateTemplate,
    axis_name: str,
    ref_coord: str,
) -> StateTemplate:
    """Return a copy of *tpl* with *axis_name* removed.

    Only the cells whose ``coord_assignments[i][axis_name] == ref_coord``
    are retained in the stripped template.  The original expanded cell
    names are preserved verbatim so ``lift_cell_ir_to_template`` can
    locate them by name.

    Args:
        tpl: Original :class:`~op_system._normalize_ir.StateTemplate`.
        axis_name: Name of the axis to strip.
        ref_coord: Coordinate value to keep (all cells with other values
            along *axis_name* are dropped).

    Returns:
        A new :class:`~op_system._normalize_ir.StateTemplate` with
        *axis_name* removed from ``axes``/``shape``/``coord_assignments``
        and with ``expanded_names`` narrowed to the reference slice.
    """
    # Lazy import avoids circular dependency at module level.
    from op_system._normalize_ir import StateTemplate  # noqa: PLC0415

    if axis_name not in tpl.axes:
        return tpl

    axis_idx = tpl.axes.index(axis_name)
    new_axes = tpl.axes[:axis_idx] + tpl.axes[axis_idx + 1 :]
    new_shape = tpl.shape[:axis_idx] + tpl.shape[axis_idx + 1 :]

    # Select cells matching ref_coord on axis_name.
    keep_indices = [
        i
        for i, ca in enumerate(tpl.coord_assignments)
        if ca.get(axis_name) == ref_coord
    ]

    new_names = tuple(tpl.expanded_names[i] for i in keep_indices)
    new_coord_assignments: tuple[Mapping[str, str], ...] = tuple(
        MappingProxyType({
            k: v for k, v in tpl.coord_assignments[i].items() if k != axis_name
        })
        for i in keep_indices
    )

    return StateTemplate(
        base=tpl.base,
        axes=new_axes,
        shape=new_shape,
        expanded_names=new_names,
        coord_assignments=new_coord_assignments,
        offset=0,  # will be patched in _assign_offsets
    )


def _assign_offsets(templates: tuple[StateTemplate, ...]) -> tuple[StateTemplate, ...]:
    """Return *templates* with corrected ``offset`` values.

    Args:
        templates: Tuple of :class:`~op_system._normalize_ir.StateTemplate`
            instances (``offset`` values may be stale).

    Returns:
        A new tuple with each template's ``offset`` field set to the
        cumulative count of expanded cells from previous templates.
    """
    out: list[StateTemplate] = []
    cursor = 0
    for tpl in templates:
        out.append(dataclasses.replace(tpl, offset=cursor))
        cursor += len(tpl.expanded_names)
    return tuple(out)


def strip_block_axis(rhs: NormalizedRhs, axis_name: str) -> NormalizedRhs:  # noqa: PLR0914
    """Return a copy of *rhs* with *axis_name* removed.

    The stripped :class:`~op_system._normalize.NormalizedRhs` covers only
    the first coordinate of *axis_name*, keeping the original per-cell
    names so the downstream vectorizer can compile a per-block-coord eval
    function via :func:`~op_system._ir_lower.lift_cell_ir_to_template`.

    Args:
        rhs: Source :class:`~op_system._normalize.NormalizedRhs`.  Must
            have ``factorize_axes`` in its ``meta`` and *axis_name* must
            be listed there.
        axis_name: Name of the axis to strip.  Must appear in
            ``rhs.meta["factorize_axes"]``.

    Returns:
        A new :class:`~op_system._normalize.NormalizedRhs` with *axis_name*
        removed from all templates, coordinate lists, shaped/TV param axes,
        and ``meta`` entries.

    Raises:
        UnsupportedFeatureError: If *axis_name* is not listed in
            ``rhs.meta["factorize_axes"]``.

    Examples:
        >>> from op_system.specs import normalize_rhs
        >>> from op_system._normalize_block import strip_block_axis
        >>> spec = {
        ...     "kind": "transitions",
        ...     "state": ["S[age,loc]", "I[age,loc]", "R[age,loc]"],
        ...     "transitions": [
        ...         {
        ...             "from": "S[age,loc]",
        ...             "to": "I[age,loc]",
        ...             "rate": "beta * I[age,loc]",
        ...         },
        ...         {"from": "I[age,loc]", "to": "R[age,loc]", "rate": "gamma"},
        ...     ],
        ...     "axes": [
        ...         {"name": "age", "type": "categorical", "coords": ["y", "o"]},
        ...         {"name": "loc", "type": "categorical", "coords": ["a", "b"]},
        ...     ],
        ...     "factorize_axes": ["loc"],
        ... }
        >>> full = normalize_rhs(spec)
        >>> stripped = strip_block_axis(full, "loc")
        >>> [t.axes for t in stripped.state_templates]
        [('age',), ('age',), ('age',)]
        >>> len(stripped.state_names)
        6
    """
    # ------------------------------------------------------------------
    # 1. Validate axis_name is in factorize_axes
    # ------------------------------------------------------------------
    factorize_axes: list[str] = list(rhs.meta.get("factorize_axes") or [])
    if axis_name not in factorize_axes:
        raise UnsupportedFeatureError(
            feature=f"strip_block_axis({axis_name!r})",
            detail=(
                f"Axis {axis_name!r} is not in rhs.meta['factorize_axes'] "
                f"({factorize_axes!r}); cannot strip it."
            ),
        )

    # ------------------------------------------------------------------
    # 2. Determine the reference coordinate (first coord of axis_name)
    # ------------------------------------------------------------------
    axes_meta: tuple[Mapping[str, Any], ...] = tuple(rhs.meta.get("axes") or ())
    ref_coord: str | None = None
    for ax in axes_meta:
        if ax.get("name") == axis_name:
            coords = ax.get("coords")
            if coords:
                ref_coord = str(coords[0])
            break
    if ref_coord is None:
        raise UnsupportedFeatureError(
            feature=f"strip_block_axis({axis_name!r})",
            detail=(
                f"Axis {axis_name!r} not found in rhs.meta['axes'] or has no coords."
            ),
        )

    # ------------------------------------------------------------------
    # 3. Strip state templates and rebuild offsets
    # ------------------------------------------------------------------
    stripped_state_templates = _assign_offsets(
        tuple(_strip_template(tpl, axis_name, ref_coord) for tpl in rhs.state_templates)
    )
    new_state_names: tuple[str, ...] = tuple(
        name for tpl in stripped_state_templates for name in tpl.expanded_names
    )

    # ------------------------------------------------------------------
    # 4. Compute which equation indices to keep (first-coord cells)
    # ------------------------------------------------------------------
    # The original state_names order is how equations are indexed.
    # Build a set of cell names that are in the stripped templates.
    keep_cells: frozenset[str] = frozenset(new_state_names)

    # Build index into original rhs.state_names for the kept cells.
    original_state_index: dict[str, int] = {
        name: i for i, name in enumerate(rhs.state_names)
    }
    eq_keep_indices = [original_state_index[n] for n in new_state_names]

    def _maybe_strip(ir: Expr | None) -> Expr | None:
        return _strip_axis_from_ir(ir, axis_name) if ir is not None else None

    new_equations = tuple(rhs.equations[i] for i in eq_keep_indices)
    new_equations_ir = tuple(_maybe_strip(rhs.equations_ir[i]) for i in eq_keep_indices)
    new_equations_ir_reduce = tuple(
        _maybe_strip(rhs.equations_ir_reduce[i]) for i in eq_keep_indices
    )

    # ------------------------------------------------------------------
    # 5. Strip alias templates and alias IR dicts
    # ------------------------------------------------------------------
    stripped_alias_templates = _assign_offsets(
        tuple(_strip_template(tpl, axis_name, ref_coord) for tpl in rhs.alias_templates)
    )

    # Collect the alias cell names that are in the first-coord slice.
    alias_keep_cells: frozenset[str] = frozenset(
        name for tpl in stripped_alias_templates for name in tpl.expanded_names
    )

    new_aliases_ir: dict[str, Any] = {
        k: _strip_axis_from_ir(v, axis_name)
        for k, v in rhs.aliases_ir.items()
        if k in alias_keep_cells
    }
    new_aliases_ir_reduce: dict[str, Any] = {
        k: _strip_axis_from_ir(v, axis_name)
        for k, v in rhs.aliases_ir_reduce.items()
        if k in alias_keep_cells
    }

    # ------------------------------------------------------------------
    # 6. Strip the string-form aliases dict
    #    The keys of rhs.aliases are per-cell expanded names (e.g.
    #    "I_total__loc_a"), same as aliases_ir.  Keep only first-coord.
    # ------------------------------------------------------------------
    new_aliases: dict[str, str] = {
        k: v for k, v in rhs.aliases.items() if k in alias_keep_cells
    }

    # ------------------------------------------------------------------
    # 7. Strip shaped_params and time_varying_params
    # ------------------------------------------------------------------
    new_shaped_params = tuple(
        (name, tuple(ax for ax in axes if ax != axis_name))
        for name, axes in rhs.shaped_params
    )
    new_time_varying_params = tuple(
        (name, tuple(ax for ax in axes if ax != axis_name))
        for name, axes in rhs.time_varying_params
    )

    # ------------------------------------------------------------------
    # 8. Strip meta: remove axis_name from factorize_axes and axes list
    # ------------------------------------------------------------------
    meta: dict[str, Any] = dict(rhs.meta)
    meta["factorize_axes"] = [a for a in factorize_axes if a != axis_name]
    new_axes_meta = [ax for ax in axes_meta if ax.get("name") != axis_name]
    meta["axes"] = new_axes_meta
    # shaped_params entry in meta mirrors rhs.shaped_params
    meta["shaped_params"] = new_shaped_params
    meta["time_varying_params"] = new_time_varying_params

    # ------------------------------------------------------------------
    # 9. Reassemble param_names / all_symbols from stripped state/alias sets
    # ------------------------------------------------------------------
    # We keep the existing param_names / all_symbols: removing the block
    # axis doesn't introduce or remove parameters — the stripped RHS still
    # references the same parameter names as the original.
    #
    # However we need to ensure state cell names that were dropped are
    # removed from all_symbols so the vectorizer doesn't try to map them.
    dropped_state_cells = frozenset(rhs.state_names) - keep_cells
    dropped_alias_cells = frozenset(rhs.aliases_ir.keys()) - alias_keep_cells
    new_all_symbols = rhs.all_symbols - dropped_state_cells - dropped_alias_cells

    return dataclasses.replace(
        rhs,
        state_names=new_state_names,
        equations=new_equations,
        aliases=new_aliases,
        all_symbols=new_all_symbols,
        meta=MappingProxyType(meta),
        state_templates=stripped_state_templates,
        shaped_params=new_shaped_params,
        time_varying_params=new_time_varying_params,
        aliases_ir=new_aliases_ir,
        equations_ir=new_equations_ir,
        aliases_ir_reduce=new_aliases_ir_reduce,
        equations_ir_reduce=new_equations_ir_reduce,
        alias_templates=stripped_alias_templates,
    )
