"""op_system._vectorize.

Vectorized compile path for templated RHS specifications.

Strategy: emit one shaped tensor expression per state template, operating on
buffers reshaped from contiguous slices of the flat state vector. For numpy
this slashes interpreter overhead; for JAX it slashes JIT compile time
massively because the emitted graph is O(#templates) instead of O(#cells).

Falls back to the scalar compile path if the spec doesn't fit the supported
subset (mixed scalar/templated states, alias names that can't be parsed,
expressions whose first cell isn't structurally identical to the last cell,
etc.).

Supported subset (v1):
- All states are pure-template wildcard entries (e.g. ``S[age, vax]``).
- Aliases are either scalar or pure-template wildcard entries whose expanded
  names parse cleanly as ``base__axis_coord__...``.
- Equation/alias expressions reference only: scalar params, ``t``, ``np``,
  templated states, templated aliases, and Python arithmetic operators.

Anything outside that subset triggers fallback.
"""

from __future__ import annotations

import ast
import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations as _comb
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from op_system._ir import Expr, ir_to_ast_expr
from op_system.compile import (
    _SAFE_BUILTINS,
    _check_numeric_dtype,
    _Indexable,
    _namespace_of,
    _parse_expr,
    _validate_ast,
    _validate_state_vector,
)

if TYPE_CHECKING:
    from types import CodeType

    from op_system.compile import EvalFn, Float64Array
    from op_system.specs import NormalizedRhs
else:
    Float64Array = Any


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BufferTemplate:
    """Records a templated buffer (state or alias) for the rewriter."""

    base: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    expanded_names: tuple[str, ...]
    coord_assignments: tuple[Mapping[str, str], ...]
    offset: int  # only meaningful for state templates; 0 otherwise


@dataclass(frozen=True, slots=True)
class _ScalarBinding:
    """Records a scalar (non-templated) symbol exposed to the env."""

    name: str  # symbol seen by expressions
    source: str  # state name / alias name / param name
    kind: str  # "state" | "alias" | "param"


@dataclass(frozen=True, slots=True)
class _EqGroup:
    """Plan for a single state template's equations.

    Each code yields an array of shape ``vec_shape`` (in ``vec_axes`` order).
    There are ``prod(unroll_shape)`` codes, one per Cartesian combination of
    ``unroll_axes`` coords. At eval time the per-code outputs are stacked into
    shape ``unroll_shape + vec_shape`` then transposed via ``assembly_perm``
    so the trailing flatten matches the template's natural cell order.
    """

    base: str
    codes: tuple[CodeType, ...]
    vec_axes: tuple[str, ...]
    vec_shape: tuple[int, ...]
    unroll_axes: tuple[str, ...]
    unroll_shape: tuple[int, ...]
    assembly_perm: tuple[int, ...]
    full_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _VectorPlan:
    """Compiled plan for vectorized evaluation of one RHS spec."""

    state_templates: tuple[_BufferTemplate, ...]
    alias_templates: tuple[_BufferTemplate, ...]
    param_templates: tuple[_BufferTemplate, ...]
    alias_codes: tuple[tuple[str, CodeType, tuple[int, ...]], ...]
    eq_groups: tuple[_EqGroup, ...]
    n_state: int
    # Shaped params declared by the normalizer (e.g. user-registered
    # hierarchical fields that don't appear as expanded scalar params) but
    # which equation/alias bodies subscript by axis name. Each entry is
    # ``(base, axes, shape)``; the eval_fn assembles ``<base>_buf`` from a
    # caller-supplied array passed under the bare base name.
    extra_param_buffers: tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...] = ()


# ---------------------------------------------------------------------------
# Name parsing / template inference
# ---------------------------------------------------------------------------


def _parse_expanded_name(
    name: str, axes: list[tuple[str, list[str]]]
) -> tuple[str, dict[str, str]]:
    """Parse an expanded name like ``base__age_y__vax_u`` -> (base, coords).

    Returns ``(name, {})`` if the name does not match the expected pattern.

    Returns:
        Tuple ``(base, coords)`` where ``coords`` maps axis name to coord
        value. Falls back to ``(name, {})`` if parsing fails.
    """
    parts = name.split("__")
    if len(parts) < 2:
        return name, {}
    base = parts[0]
    coords: dict[str, str] = {}
    for piece in parts[1:]:
        matched = False
        for ax_name, ax_coords in axes:
            prefix = ax_name + "_"
            if piece.startswith(prefix):
                coord_val = piece[len(prefix) :]
                if coord_val in ax_coords:
                    if ax_name in coords:
                        return name, {}
                    coords[ax_name] = coord_val
                    matched = True
                    break
        if not matched:
            return name, {}
    return base, coords


def _infer_templates_from_names(
    names: Iterable[str], axes: list[tuple[str, list[str]]]
) -> dict[str, _BufferTemplate] | None:
    """Group names by inferred template base.

    Returns ``None`` if any group is inconsistent (mismatched axes, missing
    cartesian-product members, etc.).

    A name with no parseable axis suffixes becomes a scalar template
    (axes=(), shape=()).

    Returns:
        A mapping from base name to ``_BufferTemplate`` on success, or
        ``None`` if the names cannot be grouped consistently.
    """
    by_base: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for name in names:
        base, coords = _parse_expanded_name(name, axes)
        by_base.setdefault(base, []).append((name, coords))

    axis_size = {ax: len(coord_list) for ax, coord_list in axes}
    axis_index = {
        ax: {c: i for i, c in enumerate(coord_list)} for ax, coord_list in axes
    }

    templates: dict[str, _BufferTemplate] = {}
    for base, members in by_base.items():
        if len(members) == 1 and not members[0][1]:
            templates[base] = _BufferTemplate(
                base=base,
                axes=(),
                shape=(),
                expanded_names=(members[0][0],),
                coord_assignments=({},),
                offset=0,
            )
            continue
        first_axes = tuple(members[0][1].keys())
        if not first_axes:
            return None
        for _, c in members:
            if tuple(c.keys()) != first_axes:
                return None
        shape = tuple(axis_size[a] for a in first_axes)
        if len(members) != math.prod(shape):
            return None
        members.sort(
            key=lambda item: tuple(axis_index[a][item[1][a]] for a in first_axes)
        )
        templates[base] = _BufferTemplate(
            base=base,
            axes=first_axes,
            shape=shape,
            expanded_names=tuple(n for n, _ in members),
            coord_assignments=tuple(c for _, c in members),
            offset=0,
        )
    return templates


def _infer_alias_templates(
    aliases: Mapping[str, str], axes: list[tuple[str, list[str]]]
) -> tuple[dict[str, _BufferTemplate], dict[str, str]] | None:
    """Group aliases by inferred template base.

    Returns:
        ``(templates_by_base, name_to_base)`` on success or ``None`` on
        failure.
    """
    templates = _infer_templates_from_names(aliases.keys(), axes)
    if templates is None:
        return None
    name_to_base = {n: t.base for t in templates.values() for n in t.expanded_names}
    return templates, name_to_base


# ---------------------------------------------------------------------------
# AST rewriter
# ---------------------------------------------------------------------------


def _build_access_ast(  # noqa: C901, PLR0912, PLR0913
    *,
    src_base: str,
    src_axes: tuple[str, ...],
    src_coords: Mapping[str, str],
    src_axis_index: Mapping[str, Mapping[str, int]],
    target_axes: tuple[str, ...],
    cell_coords: Mapping[str, str],
) -> ast.expr:
    """Build an AST node accessing a templated buffer for one cell of a target.

    Returns an expression with shape broadcastable to ``target_axes``' shape.
    Returns the bare buffer name when src and target axes match exactly.

    Returns:
        An ``ast.expr`` node accessing the source buffer in a way that
        broadcasts/transposes to the target template's cell layout.

    Raises:
        RuntimeError: If a source axis is not tied to and not fixed against
            the target axes (an internal inconsistency).
    """
    buf_name = ast.Name(id=f"{src_base}_buf", ctx=ast.Load())
    if not src_axes:
        return buf_name

    # Step 1: index the src buffer. Tied axes (axis in target_axes and same
    # coord as cell_coords[axis]) become slice(None); fixed axes become an
    # integer index.
    index_parts: list[ast.expr] = []
    tied_axis_order: list[str] = []  # in src axes order
    for ax in src_axes:
        coord_val = src_coords[ax]
        if ax in target_axes and cell_coords.get(ax) == coord_val:
            index_parts.append(ast.Slice(lower=None, upper=None, step=None))
            tied_axis_order.append(ax)
        else:
            idx = src_axis_index[ax][coord_val]
            index_parts.append(ast.Constant(value=idx))

    if len(index_parts) == 1:
        slice_node = index_parts[0]
    else:
        slice_node = ast.Tuple(elts=index_parts, ctx=ast.Load())
    accessed: ast.expr = ast.Subscript(value=buf_name, slice=slice_node, ctx=ast.Load())

    if not tied_axis_order:
        # Pure scalar after indexing.
        return accessed

    # Step 2: align tied axes to target_axes order, inserting size-1 dims for
    # target axes not in src.
    if tuple(tied_axis_order) == target_axes:
        return accessed

    # Build a tuple subscript with slice(None) for kept axes and None for
    # broadcast axes, plus a transpose if needed.
    if set(tied_axis_order) == set(target_axes):
        # Pure transpose to target order.
        perm = tuple(tied_axis_order.index(a) for a in target_axes)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="transpose",
                ctx=ast.Load(),
            ),
            args=[
                accessed,
                ast.Tuple(elts=[ast.Constant(value=p) for p in perm], ctx=ast.Load()),
            ],
            keywords=[],
        )

    # tied is a strict subset of target; transpose then add None axes.
    if not set(tied_axis_order).issubset(set(target_axes)):
        # src has axes target doesn't; should have been fully fixed above.
        msg = "internal: untied axis not in target"
        raise RuntimeError(msg)

    # Transpose tied axes into the order they appear within target_axes.
    target_kept_order = tuple(a for a in target_axes if a in tied_axis_order)
    if tuple(tied_axis_order) != target_kept_order:
        perm = tuple(tied_axis_order.index(a) for a in target_kept_order)
        accessed = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="transpose",
                ctx=ast.Load(),
            ),
            args=[
                accessed,
                ast.Tuple(elts=[ast.Constant(value=p) for p in perm], ctx=ast.Load()),
            ],
            keywords=[],
        )

    # Now insert size-1 axes via subscript with None placeholders.
    elts: list[ast.expr] = []
    for ax in target_axes:
        if ax in tied_axis_order:
            elts.append(ast.Slice(lower=None, upper=None, step=None))
        else:
            elts.append(ast.Constant(value=None))
    return ast.Subscript(
        value=accessed,
        slice=ast.Tuple(elts=elts, ctx=ast.Load()),
        ctx=ast.Load(),
    )


def _build_broadcast_access_ast(
    *,
    base: str,
    src_axes: tuple[str, ...],
    sub_axes: tuple[str, ...],
    target_axes: tuple[str, ...],
) -> ast.expr:
    """Build a broadcast-shaped access to ``<base>_buf`` for ``base[<sub_axes>]``.

    Used when a shaped (templated) parameter is referenced by axis-name
    subscript inside an alias / equation body, e.g. ``r0_loc[loc]``. The
    returned expression evaluates to an array that is the buffer's contents
    transposed and broadcast-padded so it aligns with ``target_axes``.

    ``sub_axes`` must be a permutation of ``src_axes`` and a subset of
    ``target_axes``; both invariants are checked by the caller.

    Returns:
        An ``ast.expr`` evaluating to a buffer view shaped to broadcast
        cleanly against the cell layout implied by ``target_axes``.
    """
    buf: ast.expr = ast.Name(id=f"{base}_buf", ctx=ast.Load())

    # Reorder src_axes -> sub_axes order, since the user may have written
    # ``base[ax2, ax1]`` even though the buffer is stored as ``(ax1, ax2)``.
    if tuple(sub_axes) != tuple(src_axes):
        perm = tuple(src_axes.index(a) for a in sub_axes)
        buf = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="transpose",
                ctx=ast.Load(),
            ),
            args=[
                buf,
                ast.Tuple(elts=[ast.Constant(value=p) for p in perm], ctx=ast.Load()),
            ],
            keywords=[],
        )

    # If sub_axes already covers target_axes in order, no reshape needed.
    if tuple(sub_axes) == tuple(target_axes):
        return buf

    # Insert size-1 dims for target axes not in sub_axes; reorder kept axes
    # to match their order within target_axes.
    target_kept = tuple(a for a in target_axes if a in sub_axes)
    if tuple(sub_axes) != target_kept:
        perm = tuple(sub_axes.index(a) for a in target_kept)
        buf = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="transpose",
                ctx=ast.Load(),
            ),
            args=[
                buf,
                ast.Tuple(elts=[ast.Constant(value=p) for p in perm], ctx=ast.Load()),
            ],
            keywords=[],
        )
    elts: list[ast.expr] = []
    for ax in target_axes:
        if ax in sub_axes:
            elts.append(ast.Slice(lower=None, upper=None, step=None))
        else:
            elts.append(ast.Constant(value=None))
    return ast.Subscript(
        value=buf,
        slice=ast.Tuple(elts=elts, ctx=ast.Load()),
        ctx=ast.Load(),
    )


class _NameRewriter(ast.NodeTransformer):
    """Rewrites Name nodes in a per-cell expression into shaped buffer access.

    Unrecognized names are left as-is (treated as scalar params / specials).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        target_axes: tuple[str, ...],
        cell_coords: Mapping[str, str],
        name_to_template: Mapping[str, _BufferTemplate],
        name_to_coords: Mapping[str, Mapping[str, str]],
        axis_index: Mapping[str, Mapping[str, int]],
        shaped_param_axes: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._target_axes = target_axes
        self._cell_coords = cell_coords
        self._name_to_template = name_to_template
        self._name_to_coords = name_to_coords
        self._axis_index = axis_index
        self._shaped_param_axes = shaped_param_axes or {}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        tpl = self._name_to_template.get(node.id)
        if tpl is None:
            return node
        if not tpl.axes:
            return ast.Name(id=f"{tpl.base}_buf", ctx=ast.Load())
        return _build_access_ast(
            src_base=tpl.base,
            src_axes=tpl.axes,
            src_coords=self._name_to_coords[node.id],
            src_axis_index=self._axis_index,
            target_axes=self._target_axes,
            cell_coords=self._cell_coords,
        )

    def visit_Subscript(  # noqa: C901, PLR0911, PLR0912, PLR0914
        self, node: ast.Subscript
    ) -> ast.AST:
        # Recognize ``<shaped_param_base>[<axis_name>(, <axis_name>)*]`` and
        # ``<shaped_param_base>[<int>(, <int>)*]`` (the latter being how the
        # normalizer expands ``base[axis]`` into per-cell literal indices)
        # and rewrite to a broadcast-aligned reference to ``<base>_buf``.
        # This lets shaped-param references inside transitions / aliases
        # vectorize along the indexed axis instead of forcing per-cell unroll
        # — without the rewrite each cell carries a different literal index
        # (``base[0]``, ``base[1]``, ...) whose AST differs across the
        # subscripted axis, defeating axis vectorization.
        if not isinstance(node.value, ast.Name):
            return self.generic_visit(node)
        base = node.value.id
        src_axes = self._shaped_param_axes.get(base)
        if src_axes is None or not src_axes:
            return self.generic_visit(node)
        slc = node.slice
        # Collect the per-axis subscript expressions in src_axes order.
        if isinstance(slc, ast.Tuple):
            sub_elts: list[ast.expr] = list(slc.elts)
        else:
            sub_elts = [slc]
        if len(sub_elts) != len(src_axes):
            return self.generic_visit(node)
        # Classify each subscript: axis-name (sub_axes mode) or integer
        # constant (per-cell-expanded mode). Mixing the two is unsupported
        # — fall back so the standard per-cell path handles it.
        all_names = all(isinstance(e, ast.Name) for e in sub_elts)
        all_consts = all(
            isinstance(e, ast.Constant) and isinstance(e.value, int) for e in sub_elts
        )
        if all_names:
            sub_axes = tuple(cast("ast.Name", e).id for e in sub_elts)
            # Subscripted axes must be a permutation of the param's axes and
            # all be present in target_axes (so we can broadcast cleanly).
            if set(sub_axes) != set(src_axes):
                return self.generic_visit(node)
            if any(ax not in self._target_axes for ax in sub_axes):
                return self.generic_visit(node)
            return _build_broadcast_access_ast(
                base=base,
                src_axes=src_axes,
                sub_axes=sub_axes,
                target_axes=self._target_axes,
            )
        if not all_consts:
            return self.generic_visit(node)
        # Integer-subscript form: each axis is fixed to a single coord index
        # for *this* cell. If the index matches the cell's own coord on that
        # axis (and the axis is a target axis), the access "ties" — i.e.
        # broadcasts along that axis — exactly mirroring the per-cell
        # expanded-name path used for templated state references.
        tied_axes: list[str] = []
        index_parts: list[ast.expr] = []
        for ax, elt in zip(src_axes, sub_elts, strict=True):
            idx = cast("ast.Constant", elt).value
            ax_index = self._axis_index.get(ax)
            cell_coord = self._cell_coords.get(ax)
            if (
                ax in self._target_axes
                and ax_index is not None
                and cell_coord is not None
                and ax_index.get(cell_coord) == idx
            ):
                index_parts.append(ast.Slice(lower=None, upper=None, step=None))
                tied_axes.append(ax)
            else:
                index_parts.append(ast.Constant(value=idx))
        buf_name = ast.Name(id=f"{base}_buf", ctx=ast.Load())
        slice_node: ast.expr
        if len(index_parts) == 1:
            slice_node = index_parts[0]
        else:
            slice_node = ast.Tuple(elts=index_parts, ctx=ast.Load())
        accessed: ast.expr = ast.Subscript(
            value=buf_name, slice=slice_node, ctx=ast.Load()
        )
        if not tied_axes:
            return accessed
        # Broadcast tied axes into target_axes layout (insert size-1 axes for
        # target axes not in tied_axes; transpose if order differs).
        if tuple(tied_axes) == self._target_axes:
            return accessed
        target_kept = tuple(a for a in self._target_axes if a in tied_axes)
        if tuple(tied_axes) != target_kept:
            perm = tuple(tied_axes.index(a) for a in target_kept)
            accessed = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="transpose",
                    ctx=ast.Load(),
                ),
                args=[
                    accessed,
                    ast.Tuple(
                        elts=[ast.Constant(value=p) for p in perm], ctx=ast.Load()
                    ),
                ],
                keywords=[],
            )
        elts: list[ast.expr] = []
        for ax in self._target_axes:
            if ax in tied_axes:
                elts.append(ast.Slice(lower=None, upper=None, step=None))
            else:
                elts.append(ast.Constant(value=None))
        return ast.Subscript(
            value=accessed,
            slice=ast.Tuple(elts=elts, ctx=ast.Load()),
            ctx=ast.Load(),
        )


def _rewrite_cell_to_vector(  # noqa: PLR0913
    *,
    expr: str,
    expr_ir: Expr | None = None,
    target_axes: tuple[str, ...],
    cell_coords: Mapping[str, str],
    name_to_template: Mapping[str, _BufferTemplate],
    name_to_coords: Mapping[str, Mapping[str, str]],
    axis_index: Mapping[str, Mapping[str, int]],
    shaped_param_axes: Mapping[str, tuple[str, ...]] | None = None,
) -> ast.Expression:
    """Rewrite a per-cell expression string into a shaped buffer expression.

    Returns:
        An ``ast.Expression`` whose body evaluates to an array shaped per
        ``target_axes`` (or a scalar when ``target_axes`` is empty).
    """
    tree = (
        ast.Expression(body=ir_to_ast_expr(expr_ir))
        if expr_ir is not None
        else _parse_expr(expr)
    )
    ast.fix_missing_locations(tree)
    _validate_ast(tree, expr=expr)
    rewriter = _NameRewriter(
        target_axes=target_axes,
        cell_coords=cell_coords,
        name_to_template=name_to_template,
        name_to_coords=name_to_coords,
        axis_index=axis_index,
        shaped_param_axes=shaped_param_axes,
    )
    new_tree = cast("ast.Expression", rewriter.visit(tree))
    ast.fix_missing_locations(new_tree)
    return new_tree


# ---------------------------------------------------------------------------
# Sum-pattern recognizer
# ---------------------------------------------------------------------------


def _flatten_addsub(node: ast.expr) -> list[tuple[int, ast.expr]]:
    """Flatten an Add/Sub tree into (sign, leaf-expr) terms (iterative).

    Returns:
        A list of ``(sign, leaf_expr)`` pairs in the original left-to-right
        order of the flattened tree.
    """
    terms: list[tuple[int, ast.expr]] = []
    stack: list[tuple[ast.expr, int]] = [(node, 1)]
    while stack:
        n, sign = stack.pop()
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            stack.extend(((n.right, sign), (n.left, sign)))
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub):
            stack.extend(((n.right, -sign), (n.left, sign)))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            stack.append((n.operand, -sign))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
            stack.append((n.operand, sign))
        else:
            terms.append((sign, n))
    return terms


def _classify_buf_subscript(  # noqa: PLR0911
    node: ast.expr,
) -> tuple[str, tuple[int, ...]] | None:
    """If ``node`` is ``<name>_buf[<int_tuple>]``, return (name, indices).

    Returns:
        ``(buffer_name, index_tuple)`` if the node matches the expected
        constant-indexed buffer subscript pattern, otherwise ``None``.
    """
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name):
        return None
    name = node.value.id
    if not name.endswith("_buf"):
        return None
    slc = node.slice
    if isinstance(slc, ast.Constant) and isinstance(slc.value, int):
        return name, (slc.value,)
    if isinstance(slc, ast.Tuple):
        idxs: list[int] = []
        for elt in slc.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                idxs.append(elt.value)
            else:
                return None
        return name, tuple(idxs)
    return None


def _make_sum_call(buf_name: str) -> ast.expr:
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="np", ctx=ast.Load()),
            attr="sum",
            ctx=ast.Load(),
        ),
        args=[ast.Name(id=buf_name, ctx=ast.Load())],
        keywords=[],
    )


def _extract_weighted_buf_subscript(
    node: ast.expr,
) -> tuple[float, str, tuple[int, ...]] | None:
    """If ``node`` reduces to ``(prod-of-consts) * <name>_buf[<int_tuple>]``.

    Walks an arbitrarily-nested ``Mult`` tree, accumulating numeric
    ``Constant`` factors (with ``UnaryOp(USub|UAdd, Constant)`` allowed) and
    matching exactly one buffer subscript leaf. Returns ``None`` if the leaf
    isn't of this shape or contains non-constant non-buffer factors.

    Returns:
        ``(weight, buffer_name, index_tuple)`` on match, else ``None``.
    """
    weight = 1.0
    buf_match: tuple[str, tuple[int, ...]] | None = None
    stack: list[ast.expr] = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            stack.extend((n.left, n.right))
            continue
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
            stack.append(n.operand)
            continue
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            inner = n.operand
            if isinstance(inner, ast.Constant) and isinstance(
                inner.value, (int, float)
            ):
                weight *= -float(inner.value)
                continue
            return None
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            weight *= float(n.value)
            continue
        cls = _classify_buf_subscript(n)
        if cls is None or buf_match is not None:
            return None
        buf_match = cls
    if buf_match is None:
        return None
    name, idx = buf_match
    return weight, name, idx


def _make_weighted_sum_call(
    buf_name: str,
    weights_in_index_order: list[float],
    shape: tuple[int, ...],
) -> ast.expr:
    """Build ``np.sum(np.asarray([...]).reshape(shape) * buf)``.

    ``weights_in_index_order`` is the flat list of per-cell weights in C-order
    matching ``np.ndindex(*shape)``. For 1-D buffers the ``reshape`` is
    omitted.

    The generated ``np.asarray`` call deliberately omits an explicit
    ``dtype=`` so that the weights adopt the calling backend's native float
    dtype: ``float64`` for plain numpy and the configured default for
    ``jax.numpy`` (which is ``float32`` unless ``jax_enable_x64`` is set).
    Forcing ``np.float64`` here previously triggered a noisy
    ``Explicitly requested dtype float64 ... will be truncated to dtype
    float32`` warning at every JIT trace under JAX float32 mode.

    Returns:
        The constructed ``ast.expr`` for the fused weighted reduction.
    """
    weights_list = ast.List(
        elts=[ast.Constant(value=float(w)) for w in weights_in_index_order],
        ctx=ast.Load(),
    )
    asarray: ast.expr = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="np", ctx=ast.Load()),
            attr="asarray",
            ctx=ast.Load(),
        ),
        args=[weights_list],
        keywords=[],
    )
    if len(shape) > 1:
        asarray = ast.Call(
            func=ast.Attribute(value=asarray, attr="reshape", ctx=ast.Load()),
            args=[
                ast.Tuple(
                    elts=[ast.Constant(value=int(s)) for s in shape],
                    ctx=ast.Load(),
                ),
            ],
            keywords=[],
        )
    product = ast.BinOp(
        left=asarray,
        op=ast.Mult(),
        right=ast.Name(id=buf_name, ctx=ast.Load()),
    )
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="np", ctx=ast.Load()),
            attr="sum",
            ctx=ast.Load(),
        ),
        args=[product],
        keywords=[],
    )


def _reassemble_addsub(terms: list[tuple[int, ast.expr]]) -> ast.expr:
    if not terms:
        return ast.Constant(value=0.0)
    sign, expr = terms[0]
    out: ast.expr = expr if sign > 0 else ast.UnaryOp(op=ast.USub(), operand=expr)
    for s, e in terms[1:]:
        op: ast.operator = ast.Add() if s > 0 else ast.Sub()
        out = ast.BinOp(left=out, op=op, right=e)
    return out


def _is_constant_expr(node: ast.expr) -> bool:
    """Return ``True`` if ``node`` evaluates to a numeric constant.

    Recognises numeric ``Constant`` literals, unary +/- on such literals, and
    ``Mult``/``Div`` of constant subexpressions. Used by the constant-over-Add
    distributor to decide whether ``c * (a + b)`` may be flattened to
    ``c*a + c*b``.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_constant_expr(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        return _is_constant_expr(node.left) and _is_constant_expr(node.right)
    return False


def _distribute_const_over_add(tree: ast.Expression) -> ast.Expression:
    """Rewrite ``c * (a + b)`` as ``c*a + c*b`` (and the symmetric form).

    A pre-pass for the buffer-sum collapser: the normalize-time expansion of
    nested ``apply_along`` calls produces structures like
    ``w_age_i * (S_buf[i,0] + S_buf[i,1] + ...)`` where the weight is gated
    behind a parenthesized inner sum, hiding the per-cell weighted leaves
    from the leaf-classifier. Distributing constant factors over Add chains
    flattens these into a single Add chain of ``w * S_buf[...]`` terms that
    the collapser can fold into one fused reduction.

    Iterative implementation: the top-level Add/Sub chain is flattened into a
    term list (via :func:`_flatten_addsub`, itself iterative) and each leaf
    is rewritten in isolation. This avoids the per-node Python frame that an
    ``ast.NodeTransformer.visit`` would consume on right- or left-nested
    Add chains thousands of nodes deep (e.g. aggregator aliases that sum
    every cell of a templated state). Within each leaf we use a recursive
    visitor; leaf depth is bounded by expression complexity (typically <=5),
    not by chain length, so it is safe.

    Distribution at a leaf may itself produce a new Add/Sub chain (when the
    rewritten leaf is ``c*a + c*b``). Such leaves are folded back into the
    outer term list with appropriate sign propagation, and the pass iterates
    to a fixed point.

    Only distributes when one side of the ``Mult`` is a numeric constant -
    never distributes runtime-variable factors (which would change semantics
    for backends with shaped operands).

    Returns:
        A possibly-rewritten ``ast.Expression`` semantically equal to
        ``tree`` with constant-multipliers pushed inside Add/Sub chains.
    """

    class _Distributor(ast.NodeTransformer):
        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
            # Recurse first so inner distributions surface upward.
            self.generic_visit(node)
            if not isinstance(node.op, ast.Mult):
                return node
            # Identify the (constant, addsub) sides; either ordering is fine.
            const_side: ast.expr | None = None
            chain_side: ast.expr | None = None
            if (
                _is_constant_expr(node.left)
                and isinstance(node.right, ast.BinOp)
                and isinstance(node.right.op, (ast.Add, ast.Sub))
            ):
                const_side = node.left
                chain_side = node.right
            elif (
                _is_constant_expr(node.right)
                and isinstance(node.left, ast.BinOp)
                and isinstance(node.left.op, (ast.Add, ast.Sub))
            ):
                const_side = node.right
                chain_side = node.left
            if const_side is None or chain_side is None:
                return node
            terms = _flatten_addsub(chain_side)
            new_terms: list[tuple[int, ast.expr]] = [
                (
                    sign,
                    ast.BinOp(left=const_side, op=ast.Mult(), right=leaf),
                )
                for sign, leaf in terms
            ]
            return _reassemble_addsub(new_terms)

    transformer = _Distributor()
    body = tree.body
    if _is_addsub_binop(body):
        body = _distribute_addsub_chain(body, transformer)
    else:
        # No top-level chain: the body itself is shallow enough to recurse.
        body = cast("ast.expr", transformer.visit(body))
    new_tree = ast.Expression(body=body)
    ast.fix_missing_locations(new_tree)
    return new_tree


def _is_addsub_binop(n: ast.AST) -> bool:
    """Return True iff ``n`` is an ``ast.BinOp`` with an Add/Sub operator."""
    return isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub))


def _distribute_addsub_chain(
    body: ast.expr, transformer: ast.NodeTransformer
) -> ast.expr:
    """Iteratively flatten an Add/Sub chain and rewrite each leaf in place.

    Loops to a fixed point so that distribution which produces new sub-chains
    (``c*(a+b) -> c*a + c*b``) is refolded into the outer chain. Each leaf is
    visited via ``transformer`` independently; leaf depth is bounded by leaf
    complexity (typically <=5), not by chain length, so the recursive visit
    is safe even when the outer chain has thousands of terms.

    Args:
        body: A top-level Add/Sub binop to distribute over.
        transformer: An ``ast.NodeTransformer`` (typically ``_Distributor``)
            that rewrites a single leaf expression.

    Returns:
        A possibly-rewritten ``ast.expr`` semantically equal to ``body``.
    """
    max_passes = 64  # safety bound; convergence is typically immediate
    for _ in range(max_passes):
        terms = _flatten_addsub(body)
        new_terms: list[tuple[int, ast.expr]] = []
        any_changed = False
        for sign, leaf in terms:
            rewritten = cast("ast.expr", transformer.visit(leaf))
            if _is_addsub_binop(rewritten):
                for ssign, sleaf in _flatten_addsub(rewritten):
                    new_terms.append((sign * ssign, sleaf))
                any_changed = True
            else:
                if rewritten is not leaf:
                    any_changed = True
                new_terms.append((sign, rewritten))
        body = _reassemble_addsub(new_terms)
        if not any_changed:
            break
    return body


def _collapse_full_buffer_sums(  # noqa: C901, PLR0915
    tree: ast.Expression, buf_shapes: Mapping[str, tuple[int, ...]]
) -> ast.Expression:
    """Replace sums-over-all-cells of a buffer with ``np.sum(buf)``.

    Operates iteratively to avoid blowing Python's recursion limit on the
    very long Add chains produced by aggregator aliases (e.g. an alias that
    sums every cell of a templated state). Within each Add/Sub chain found in
    the tree, leaves of the form ``<name>_buf[<int_tuple>]`` (or that pattern
    multiplied by a product of constant numeric factors) are grouped by
    ``(name, sign)``; if the set of indices for a group exhausts the full
    Cartesian product of the buffer's shape, those terms are collapsed into a
    single fused reduction. Bare-subscript groups become ``np.sum(buf)``;
    weighted groups become ``np.sum(weights * buf)`` where ``weights`` is a
    constant array shaped to ``buf``. This collapses the long weighted-sum
    expansion of ``apply_along(..., kernel=integrate)`` into a single fused
    multiply+reduce. Other terms in the chain are left untouched.

    Returns:
        A possibly-rewritten ``ast.Expression`` equivalent to ``tree`` but
        with full-buffer sums collapsed where detected.
    """

    def collapse_chain(node: ast.expr) -> ast.expr:  # noqa: C901, PLR0912, PLR0914, PLR0915
        terms = _flatten_addsub(node)
        # group key (buf_name, sign) -> list of (idx_tuple, weight); weight is
        # 1.0 when the leaf was a bare buffer subscript, otherwise the product
        # of constant factors multiplying the subscript.
        groups: dict[tuple[str, int], list[tuple[tuple[int, ...], float]]] = {}
        other: list[tuple[int, ast.expr]] = []
        order: list[tuple[str, int]] = []
        for sign, leaf in terms:
            name: str
            idx: tuple[int, ...]
            weight: float
            cls = _classify_buf_subscript(leaf)
            if cls is not None and cls[0] in buf_shapes:
                name, idx = cls
                weight = 1.0
            else:
                wcls = _extract_weighted_buf_subscript(leaf)
                if wcls is None or wcls[1] not in buf_shapes:
                    other.append((sign, leaf))
                    continue
                weight, name, idx = wcls
            key = (name, sign)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((idx, weight))
        collapsed: list[tuple[int, ast.expr]] = list(other)
        changed = False
        for key in order:
            name, sign = key
            items = groups[key]
            idxs = [t for t, _w in items]
            shape = buf_shapes[name]
            expected = math.prod(shape) if shape else 1
            full_cover = (
                len(idxs) == expected
                and len(set(idxs)) == expected
                and all(
                    len(t) == len(shape)
                    and all(0 <= ti < si for ti, si in zip(t, shape, strict=True))
                    for t in idxs
                )
            )
            if full_cover:
                # Three emission cases for a full-cover Add chain over ``buf``:
                #
                # 1. All weights exactly 1.0 (bare buffer subscripts) ->
                #    ``np.sum(buf)``.
                # 2. All weights equal to some constant ``c != 1.0`` (typical
                #    when ``_distribute_const_over_add`` pushed an outer
                #    constant factor like ``dt`` over a ``kernel=sum``
                #    apply_along chain) -> ``c * np.sum(buf)``. Without this
                #    branch we would emit ``np.sum(np.asarray([c,...]) * buf)``
                #    with an inlined uniform weight array of length
                #    ``prod(shape)``, which on large network/categorical
                #    models bloats the JAX trace and balloons XLA compile
                #    time (see op_system#103).
                # 3. Heterogeneous weights (true ``kernel=integrate`` with
                #    trapezoidal endpoints, or any non-uniform weight set)
                #    -> ``np.sum(weights_arr * buf)`` via the fused-reduce
                #    helper.
                #
                # Exact float equality is intentional: weights here come
                # either from explicit literal coefficients in the source
                # or from constant factors propagated by
                # ``_distribute_const_over_add``. Near-1.0 weights from
                # trapezoidal kernels (e.g. ``0.5``) take the weighted-sum
                # path in case 3.
                ws = [w for _t, w in items]
                if all(w == 1.0 for w in ws):  # noqa: RUF069
                    collapsed.append((sign, _make_sum_call(name)))
                elif all(w == ws[0] for w in ws):
                    collapsed.append((
                        sign,
                        ast.BinOp(
                            left=ast.Constant(value=float(ws[0])),
                            op=ast.Mult(),
                            right=_make_sum_call(name),
                        ),
                    ))
                else:
                    weight_map = dict(items)
                    flat_weights = (
                        [
                            weight_map[tuple(int(i) for i in t)]
                            for t in np.ndindex(*shape)
                        ]
                        if shape
                        else [weight_map[()]]
                    )
                    collapsed.append((
                        sign,
                        _make_weighted_sum_call(name, flat_weights, shape),
                    ))
                changed = True
            else:
                for t, w in items:
                    slice_node: ast.expr
                    if len(t) == 1:
                        slice_node = ast.Constant(value=t[0])
                    else:
                        slice_node = ast.Tuple(
                            elts=[ast.Constant(value=v) for v in t],
                            ctx=ast.Load(),
                        )
                    sub = ast.Subscript(
                        value=ast.Name(id=name, ctx=ast.Load()),
                        slice=slice_node,
                        ctx=ast.Load(),
                    )
                    leaf_expr: ast.expr
                    # Same intent as above: only an exact 1.0 weight may be
                    # dropped; any other value must be re-emitted as an
                    # explicit ``Constant * Subscript`` factor.
                    if w == 1.0:  # noqa: RUF069
                        leaf_expr = sub
                    else:
                        leaf_expr = ast.BinOp(
                            left=ast.Constant(value=float(w)),
                            op=ast.Mult(),
                            right=sub,
                        )
                    collapsed.append((sign, leaf_expr))
        if not changed:
            return node
        return _reassemble_addsub(collapsed)

    def is_addsub(n: ast.AST) -> bool:
        return isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub))

    # Collapse the root expression, then iteratively descend into the
    # collapsed body's non-Add/Sub children to find embedded chains. We
    # carefully avoid recursing into the original (uncollapsed) Add chain.
    body = tree.body
    if is_addsub(body):
        body = collapse_chain(body)
    new_tree = ast.Expression(body=body)

    stack: list[ast.AST] = [body]
    while stack:
        node = stack.pop()
        for fld, val in ast.iter_fields(node):
            if isinstance(val, list):
                for i, child in enumerate(val):
                    if not isinstance(child, ast.AST):
                        continue
                    if is_addsub(child):
                        new_child = collapse_chain(cast("ast.expr", child))
                        val[i] = new_child
                        stack.append(new_child)
                    else:
                        stack.append(child)
            elif isinstance(val, ast.AST):
                if is_addsub(val):
                    new_child = collapse_chain(cast("ast.expr", val))
                    setattr(node, fld, new_child)
                    stack.append(new_child)
                else:
                    stack.append(val)

    ast.fix_missing_locations(new_tree)
    return new_tree


def _vectorize_template_equations(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915
    *,
    template: _BufferTemplate,
    equations: tuple[str, ...],
    equations_ir: tuple[Expr | None, ...] | None = None,
    name_to_template: Mapping[str, _BufferTemplate],
    name_to_coords: Mapping[str, Mapping[str, str]],
    axis_index: Mapping[str, Mapping[str, int]],
    shaped_param_axes: Mapping[str, tuple[str, ...]] | None = None,
) -> (
    tuple[
        tuple[CodeType, ...],
        tuple[str, ...],  # vec_axes
        tuple[int, ...],  # vec_shape
        tuple[str, ...],  # unroll_axes
        tuple[int, ...],  # unroll_shape
        tuple[int, ...],  # assembly_perm
    ]
    | None
):
    """Build shaped code object(s) for a state template's equations.

    Tries vectorized axis subsets in increasing complexity:
    1. All axes vectorized (no unrolling).
    2. Each single axis unrolled, all others vectorized — handles boundary
       conditions on one axis (e.g. the X compartment's ``imm`` axis).
    3. Pairs of axes unrolled (rare).
    Within each candidate subset, the first and last cell of every unroll
    bin must produce structurally identical AST after rewriting.

    Returns:
        A tuple of ``(codes, vec_axes, vec_shape, unroll_axes,
        unroll_shape, assembly_perm)`` describing the chosen vectorization
        plan, or ``None`` if no candidate succeeds.
    """
    size = math.prod(template.shape) if template.shape else 1
    cell_exprs = equations[template.offset : template.offset + size]
    if len(cell_exprs) != size or size == 0:
        return None
    cell_irs = (
        equations_ir[template.offset : template.offset + size]
        if equations_ir is not None and len(equations_ir) == len(equations)
        else ()
    )
    n_axes = len(template.axes)

    # Enumerate candidate unroll-axis index subsets, smallest first.
    candidates: list[tuple[int, ...]] = []
    for k in range(n_axes + 1):
        candidates.extend(_comb(range(n_axes), k))

    # Strides for converting (axis-coord-index tuple) → flat cell index in
    # ``coord_assignments`` (template.axes order, last varies fastest).
    strides: list[int] = [0] * n_axes
    s = 1
    for i in range(n_axes - 1, -1, -1):
        strides[i] = s
        s *= template.shape[i]

    for unroll_idx in candidates:
        unroll_set = set(unroll_idx)
        vec_idx = tuple(i for i in range(n_axes) if i not in unroll_set)
        unroll_axes = tuple(template.axes[i] for i in unroll_idx)
        vec_axes = tuple(template.axes[i] for i in vec_idx)
        unroll_shape = tuple(template.shape[i] for i in unroll_idx)
        vec_shape = tuple(template.shape[i] for i in vec_idx)
        unroll_size = math.prod(unroll_shape) if unroll_shape else 1
        vec_size = math.prod(vec_shape) if vec_shape else 1

        # Iterate unroll bins in row-major over unroll_axes.
        codes: list[CodeType] = []
        ok = True
        for u in range(unroll_size):
            # Decode u into per-unroll-axis coord indices.
            u_coord_idx: list[int] = []
            rem = u
            for j in range(len(unroll_axes) - 1, -1, -1):
                u_coord_idx.insert(0, rem % unroll_shape[j])
                rem //= unroll_shape[j]
            # Compute the "base" flat index contribution from unroll axes.
            base_flat = 0
            for k_pos, ax_pos in enumerate(unroll_idx):
                base_flat += u_coord_idx[k_pos] * strides[ax_pos]

            # First and last cells of the bin = vec_axes coord indices all
            # zero / all (size-1).
            def _bin_flat(
                vec_coord_idx: list[int],
                base_flat: int = base_flat,
                vec_idx: tuple[int, ...] = vec_idx,
            ) -> int:
                f = base_flat
                for k_pos, ax_pos in enumerate(vec_idx):
                    f += vec_coord_idx[k_pos] * strides[ax_pos]
                return f

            first_idx = _bin_flat([0] * len(vec_idx))
            last_idx = _bin_flat([vec_shape[k] - 1 for k in range(len(vec_idx))])

            try:
                first_tree = _rewrite_cell_to_vector(
                    expr=cell_exprs[first_idx],
                    expr_ir=cell_irs[first_idx] if cell_irs else None,
                    target_axes=vec_axes,
                    cell_coords=template.coord_assignments[first_idx],
                    name_to_template=name_to_template,
                    name_to_coords=name_to_coords,
                    axis_index=axis_index,
                    shaped_param_axes=shaped_param_axes,
                )
            except (ValueError, RuntimeError):
                ok = False
                break
            if vec_size > 1:
                try:
                    last_tree = _rewrite_cell_to_vector(
                        expr=cell_exprs[last_idx],
                        expr_ir=cell_irs[last_idx] if cell_irs else None,
                        target_axes=vec_axes,
                        cell_coords=template.coord_assignments[last_idx],
                        name_to_template=name_to_template,
                        name_to_coords=name_to_coords,
                        axis_index=axis_index,
                        shaped_param_axes=shaped_param_axes,
                    )
                except (ValueError, RuntimeError):
                    ok = False
                    break
                if ast.dump(first_tree) != ast.dump(last_tree):
                    ok = False
                    break
            try:
                codes.append(
                    compile(first_tree, filename="<op_system_vec>", mode="eval")
                )
            except (ValueError, TypeError, SyntaxError):
                ok = False
                break
        if not ok:
            continue

        # Build assembly_perm: source axis order is unroll_axes + vec_axes,
        # target is template.axes. perm[i] = position-in-source of template.axes[i].
        source_order = list(unroll_axes) + list(vec_axes)
        source_pos = {a: i for i, a in enumerate(source_order)}
        assembly_perm = tuple(source_pos[a] for a in template.axes)
        return (
            tuple(codes),
            vec_axes,
            vec_shape,
            unroll_axes,
            unroll_shape,
            assembly_perm,
        )
    return None


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def build_vector_plan(rhs: NormalizedRhs) -> _VectorPlan | None:
    """Try to build a vectorized plan for ``rhs``.

    Returns:
        A ``_VectorPlan`` describing the vectorized layout, or ``None`` to
        signal that the spec is unsupported and the caller should fall back
        to the scalar engine.
    """
    # Several internal passes still rely on recursive AST walks (notably
    # ``_NameRewriter`` and CPython's ``ast.fix_missing_locations`` /
    # ``compile``). On models whose normalize-time expansion produces long
    # ``apply_along(kernel=sum)`` Add chains - e.g. aggregator aliases that
    # sum every cell of a templated state - those walks consume one Python
    # frame per chain node and blow the default 1000-frame recursion limit,
    # silently dropping the model to the scalar fallback. Bump the limit
    # for the duration of this call sized to a safe upper bound on the
    # longest alias-or-equation expression. This is a temporary safety net
    # while the recursive passes are migrated to iterative term-list form
    # (op_system#103); it does not eliminate the underlying C-stack ceiling
    # (~10-20k frames on typical builds), so it cannot rescue extreme
    # workloads (e.g. continuum models with >50k-cell aggregators) - those
    # require the structured-IR refactor tracked separately.
    longest_expr = 0
    for s in rhs.equations:
        if isinstance(s, str) and len(s) > longest_expr:
            longest_expr = len(s)
    for s in rhs.aliases.values():
        if isinstance(s, str) and len(s) > longest_expr:
            longest_expr = len(s)
    # Heuristic: each Add term in the expanded form is on the order of 8-16
    # characters ("S__loc_l0001 + "), and the recursive walk depth roughly
    # tracks term count plus a small constant for surrounding nodes.
    needed = max(2000, longest_expr // 4 + 1000)
    saved_limit = sys.getrecursionlimit()
    if needed > saved_limit:
        sys.setrecursionlimit(needed)
    try:
        return _build_vector_plan_inner(rhs)
    finally:
        sys.setrecursionlimit(saved_limit)


def _build_vector_plan_inner(  # noqa: C901, PLR0911, PLR0912, PLR0914, PLR0915
    rhs: NormalizedRhs,
) -> _VectorPlan | None:
    """Vectorized-plan construction body (see ``build_vector_plan``).

    Returns:
        A ``_VectorPlan`` describing the vectorized layout, or ``None`` to
        signal that the spec is unsupported and the caller should fall back
        to the scalar engine.
    """
    if not rhs.state_templates:
        return None
    # Require all states to be wildcard templates (have axes).
    if any(not tpl.shape for tpl in rhs.state_templates):
        return None
    # Require axes meta to be present.
    axes_meta = rhs.meta.get("axes") if isinstance(rhs.meta, Mapping) else None
    if not axes_meta:
        return None

    axes_pairs: list[tuple[str, list[str]]] = []
    for ax in axes_meta:
        if not isinstance(ax, Mapping):
            return None
        coords = ax.get("coords")
        if not coords:
            return None
        # Stringify coords to match the convention in
        # ``_templates.build_axis_lookup`` (and therefore the per-cell
        # ``coord_assignments`` carried by state templates).  For categorical
        # axes coords are already strings; for continuous axes declared with
        # explicit numeric coords (e.g. ``[0.0, 5.0, ...]``) the unconverted
        # raw values would key ``axis_index`` by ``float`` while lookups use
        # ``str(float)``, causing ``KeyError`` in ``_build_access_ast``.
        axes_pairs.append((ax["name"], [str(c) for c in coords]))
    axis_index = {ax: {c: i for i, c in enumerate(coords)} for ax, coords in axes_pairs}

    state_buffers: list[_BufferTemplate] = []
    name_to_template: dict[str, _BufferTemplate] = {}
    name_to_coords: dict[str, Mapping[str, str]] = {}
    for tpl in rhs.state_templates:
        buf = _BufferTemplate(
            base=tpl.base,
            axes=tpl.axes,
            shape=tpl.shape,
            expanded_names=tpl.expanded_names,
            coord_assignments=tpl.coord_assignments,
            offset=tpl.offset,
        )
        state_buffers.append(buf)
        for nm, coord_map in zip(
            tpl.expanded_names, tpl.coord_assignments, strict=True
        ):
            name_to_template[nm] = buf
            name_to_coords[nm] = coord_map

    alias_inference = _infer_alias_templates(rhs.aliases, axes_pairs)
    if alias_inference is None:
        return None
    alias_buffers_by_base, _ = alias_inference
    alias_buffers = list(alias_buffers_by_base.values())
    for buf in alias_buffers:
        for nm, coord_map in zip(
            buf.expanded_names, buf.coord_assignments, strict=True
        ):
            name_to_template[nm] = buf
            name_to_coords[nm] = coord_map

    # Param templates: same name-parsing approach as aliases. These are
    # gathered from caller-supplied scalar params at eval time into shaped
    # buffers exposed under ``<base>_buf``.
    param_templates_by_base = _infer_templates_from_names(rhs.param_names, axes_pairs)
    if param_templates_by_base is None:
        return None
    param_buffers = list(param_templates_by_base.values())
    for buf in param_buffers:
        # Skip scalars — they remain available under their original name.
        if not buf.axes:
            continue
        for nm, coord_map in zip(
            buf.expanded_names, buf.coord_assignments, strict=True
        ):
            name_to_template[nm] = buf
            name_to_coords[nm] = coord_map

    # Map of shaped-parameter base name → axes tuple, drawn from both the
    # normalizer-declared shaped params (e.g. user-registered hierarchical
    # fields like ``r0_loc``) and any param templates inferred from the
    # expanded ``param_names``. Used by ``_NameRewriter.visit_Subscript`` to
    # rewrite per-cell subscripts like ``r0_loc[<idx>]`` into broadcast
    # accesses against ``<base>_buf``, allowing the equation vectorizer to
    # keep the subscripted axis vectorized rather than unrolling it.
    shaped_param_axes: dict[str, tuple[str, ...]] = {}
    for name, ax_tuple in rhs.shaped_params:
        ax_t = tuple(ax_tuple)
        if ax_t:
            shaped_param_axes[name] = ax_t
    for buf in param_buffers:
        if buf.axes:
            shaped_param_axes.setdefault(buf.base, buf.axes)

    # Buffer-name → shape table, used by the sum-pattern collapser to
    # recognize sums-over-all-cells of a buffer and rewrite them as
    # ``np.sum(buf)``.
    buf_shapes: dict[str, tuple[int, ...]] = {}
    for buf in (*state_buffers, *alias_buffers, *param_buffers):
        if buf.axes:
            buf_shapes[f"{buf.base}_buf"] = buf.shape

    # Vectorize aliases (in declaration order — best-effort dependency order).
    alias_codes: list[tuple[str, CodeType, tuple[int, ...]]] = []
    for buf in alias_buffers:
        try:
            if buf.axes:
                tree = _rewrite_cell_to_vector(
                    expr=rhs.aliases[buf.expanded_names[0]],
                    expr_ir=rhs.aliases_ir.get(buf.expanded_names[0]),
                    target_axes=buf.axes,
                    cell_coords=buf.coord_assignments[0],
                    name_to_template=name_to_template,
                    name_to_coords=name_to_coords,
                    axis_index=axis_index,
                    shaped_param_axes=shaped_param_axes,
                )
                if len(buf.expanded_names) > 1:
                    last_tree = _rewrite_cell_to_vector(
                        expr=rhs.aliases[buf.expanded_names[-1]],
                        expr_ir=rhs.aliases_ir.get(buf.expanded_names[-1]),
                        target_axes=buf.axes,
                        cell_coords=buf.coord_assignments[-1],
                        name_to_template=name_to_template,
                        name_to_coords=name_to_coords,
                        axis_index=axis_index,
                        shaped_param_axes=shaped_param_axes,
                    )
                    if ast.dump(tree) != ast.dump(last_tree):
                        return None
            else:
                # Scalar alias: rewrite through the same engine so referenced
                # templated state cells become buffer-index accesses.
                tree = _rewrite_cell_to_vector(
                    expr=rhs.aliases[buf.expanded_names[0]],
                    expr_ir=rhs.aliases_ir.get(buf.expanded_names[0]),
                    target_axes=(),
                    cell_coords={},
                    name_to_template=name_to_template,
                    name_to_coords=name_to_coords,
                    axis_index=axis_index,
                    shaped_param_axes=shaped_param_axes,
                )
            tree = _distribute_const_over_add(tree)
            tree = _collapse_full_buffer_sums(tree, buf_shapes)
            code = compile(tree, filename="<op_system_vec>", mode="eval")
        except (ValueError, RuntimeError, TypeError, SyntaxError):
            return None
        alias_codes.append((buf.base, code, buf.shape))

    # Vectorize per-template equations.
    eq_groups: list[_EqGroup] = []
    for buf in state_buffers:
        result = _vectorize_template_equations(
            template=buf,
            equations=rhs.equations,
            equations_ir=rhs.equations_ir_raw,
            name_to_template=name_to_template,
            name_to_coords=name_to_coords,
            axis_index=axis_index,
            shaped_param_axes=shaped_param_axes,
        )
        if result is None:
            return None
        codes, vec_axes, vec_shape, unroll_axes, unroll_shape, assembly_perm = result
        eq_groups.append(
            _EqGroup(
                base=buf.base,
                codes=codes,
                vec_axes=vec_axes,
                vec_shape=vec_shape,
                unroll_axes=unroll_axes,
                unroll_shape=unroll_shape,
                assembly_perm=assembly_perm,
                full_shape=buf.shape,
            )
        )

    return _VectorPlan(
        state_templates=tuple(state_buffers),
        alias_templates=tuple(alias_buffers),
        param_templates=tuple(param_buffers),
        alias_codes=tuple(alias_codes),
        eq_groups=tuple(eq_groups),
        n_state=len(rhs.state_names),
        extra_param_buffers=tuple(
            (
                base,
                axes,
                tuple(len(axis_index[a]) for a in axes),
            )
            for base, axes in shaped_param_axes.items()
            if axes and base not in {buf.base for buf in param_buffers if buf.axes}
        ),
    )


# ---------------------------------------------------------------------------
# Runtime eval function
# ---------------------------------------------------------------------------


def make_vectorized_eval_fn(plan: _VectorPlan) -> EvalFn:  # noqa: C901, PLR0915
    """Return a namespace-polymorphic ``eval_fn(t, y, **params)`` driven by ``plan``.

    The compiled function infers its array namespace from the input ``y``
    via :meth:`y.__array_namespace__` at call time, so a single eval_fn
    handles NumPy and JAX (and any other Array-API backend) without
    branching. Calling it with JAX arrays (or tracers) yields a JAX-native
    computation.
    """
    state_templates = plan.state_templates
    alias_codes = plan.alias_codes
    eq_groups = plan.eq_groups
    n_state = plan.n_state

    # Pre-compute param buffer assembly recipes: (base, expanded_names, shape)
    # — at eval time we stack the corresponding param values.
    param_recipes: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = [
        (buf.base, buf.expanded_names, buf.shape)
        for buf in plan.param_templates
        if buf.axes
    ]
    # Extra shaped-param buffers (declared via ``rhs.shaped_params`` but with
    # no per-cell expanded names). Assembled from a caller-supplied array
    # passed under the bare base name.
    extra_param_buffers = plan.extra_param_buffers

    def eval_fn(t: object, y: object, **params: object) -> Float64Array:  # noqa: C901, PLR0912, PLR0914
        xp = _namespace_of(y)
        _check_numeric_dtype(xp, getattr(y, "dtype", None))
        y_arr = _validate_state_vector(y, n_state=n_state)
        y_idx = cast("_Indexable", y_arr)

        t_val: object = xp.asarray(t)
        env: dict[str, object] = {"np": xp, "t": t_val}
        env.update(params)

        # Assemble templated param buffers from caller-supplied scalars
        # (one value per expanded name) or, as a convenience, from a
        # directly-supplied array passed under the buffer's base name.
        for base, names, shape in param_recipes:
            if all(n in params for n in names):
                vals = [params[n] for n in names]
                env[f"{base}_buf"] = xp.reshape(xp.asarray(vals), shape)
            elif base in params:
                bare = params[base]
                env[f"{base}_buf"] = xp.reshape(xp.asarray(bare), shape)
            else:
                missing = next(n for n in names if n not in params)
                msg = (
                    f"missing param {missing!r} for templated buffer {base!r}"
                    f" (or supply {base!r} directly as a shape-{shape} array)"
                )
                raise ValueError(msg)

        # Assemble extra shaped-param buffers from caller-supplied arrays.
        for base, _axes, shape in extra_param_buffers:
            if base not in params:
                msg = f"missing shaped param {base!r}: supply as a shape-{shape} array"
                raise ValueError(msg)
            bare = params[base]
            env[f"{base}_buf"] = xp.reshape(xp.asarray(bare), shape)

        # Build state buffers via reshape on contiguous slices.
        for tpl in state_templates:
            size = math.prod(tpl.shape)
            slc = y_idx[tpl.offset : tpl.offset + size]
            env[f"{tpl.base}_buf"] = xp.reshape(slc, tpl.shape)

        # Eval alias buffers in declaration order.
        for base, code, shape in alias_codes:
            try:
                val = eval(  # noqa: S307
                    code, {"__builtins__": _SAFE_BUILTINS}, env
                )
            except (NameError, ValueError, TypeError, ArithmeticError) as exc:
                msg = f"alias {base!r} evaluation failed: {exc!r}"
                raise ValueError(msg) from exc
            if shape:
                val = xp.broadcast_to(val, shape)
            env[f"{base}_buf"] = val

        # Eval per-template equation buffers and concatenate.
        pieces: list[object] = []
        for grp in eq_groups:
            bin_results: list[object] = []
            for code in grp.codes:
                try:
                    val = eval(  # noqa: S307
                        code, {"__builtins__": _SAFE_BUILTINS}, env
                    )
                except (NameError, ValueError, TypeError, ArithmeticError) as exc:
                    msg = f"equation {grp.base!r} evaluation failed: {exc!r}"
                    raise ValueError(msg) from exc
                arr = xp.broadcast_to(xp.asarray(val), grp.vec_shape)
                bin_results.append(arr)

            if not grp.unroll_axes:
                # Fully vectorized — single result already in template order.
                whole = bin_results[0]
            else:
                stacked = xp.reshape(
                    xp.stack(bin_results, axis=0),
                    grp.unroll_shape + grp.vec_shape,
                )
                whole = xp.transpose(stacked, grp.assembly_perm)
            pieces.append(xp.reshape(whole, (-1,)))

        return cast("Float64Array", xp.concatenate(pieces))

    return eval_fn
