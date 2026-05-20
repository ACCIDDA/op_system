"""op_system._vectorize.

Vectorized compile path for templated RHS specifications.

Strategy: emit one shaped tensor expression per state template, operating on
buffers reshaped from contiguous slices of the flat state vector. For numpy
this slashes interpreter overhead; for JAX it slashes JIT compile time
massively because the emitted graph is O(#templates) instead of O(#cells).

Falls back to the scalar compile path if the spec doesn't fit the supported
subset (mixed scalar/templated states, IR lowering limitations, expressions
whose first cell isn't structurally identical to the last cell, etc.).

Supported subset (v1):
- All states are pure-template wildcard entries (e.g. ``S[age, vax]``).
- Aliases are either scalar or pure-template wildcard entries.
- Equation/alias expressions reference only: scalar params, ``t``, ``np``,
  templated states, templated aliases, shaped params, and Python arithmetic.

Anything outside that subset triggers fallback to the scalar engine.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations as _comb
from typing import TYPE_CHECKING, Any, cast

from op_system._ir import Apply, Literal, Sym, extract_common_subexpressions
from op_system._ir_lower import (
    UnsupportedIRLoweringError,
    lift_cell_ir_to_template,
    lower_to_vector_ast,
)
from op_system.compile import (
    _SAFE_BUILTINS,
    _check_numeric_dtype,
    _Indexable,
    _namespace_of,
    _validate_state_vector,
)

if TYPE_CHECKING:
    from types import CodeType

    from op_system._ir import Expr
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
    # Plan-level CSE temporaries: computed once after alias buffers are
    # assembled, before equation evaluation. Each entry is ``(name, code)``
    # where ``code`` evaluates to an array shaped to the common template axes.
    cse_codes: tuple[tuple[str, CodeType], ...] = ()


def _try_ir_fast_path(  # noqa: PLR0913
    expr_ir: Expr,
    *,
    target_axes: tuple[str, ...],
    name_to_template: Mapping[str, _BufferTemplate],
    axis_index: Mapping[str, Mapping[str, int]],
    reducible_axes: frozenset[str] = frozenset(),
    axis_weights: Mapping[str, tuple[float, ...]] | None = None,
    axis_coords: Mapping[str, tuple[str, ...]] | None = None,
    axis_types: Mapping[str, str] | None = None,
    shaped_param_axes: Mapping[str, tuple[str, ...]] | None = None,
) -> ast.Expression | None:
    """Attempt to lower per-cell IR straight to a vector AST.

    Lifts ``expr_ir`` (per-cell scalar names) to template-form IR via
    :func:`lift_cell_ir_to_template`, then runs the typed lowering in
    :func:`lower_to_vector_ast`. On any v1-scope violation (raises
    :class:`UnsupportedIRLoweringError`) returns ``None`` so the caller
    can try the raw-IR fast path.

    The fast path bypasses per-cell name expansion entirely: it produces
    the same shaped buffer expression for every cell of a template, which
    makes the downstream "first cell == last cell" structural check
    trivially pass and lets the vectorizer emit a single compiled code
    object for the whole template.

    Returns:
        A complete ``ast.Expression`` whose body broadcasts to
        ``target_axes``, or ``None`` if the IR falls outside the v1
        lowering subset.
    """
    cell_to_template: dict[str, tuple[str, tuple[str, ...]]] = {}
    buffer_axes: dict[str, tuple[str, ...]] = {}
    for cell_name, tpl in name_to_template.items():
        cell_to_template[cell_name] = (tpl.base, tpl.axes)
        if tpl.axes:
            buffer_axes[tpl.base] = tpl.axes
    if not buffer_axes:
        return None
    try:
        lifted = lift_cell_ir_to_template(expr_ir, cell_to_template=cell_to_template)
        body = lower_to_vector_ast(
            lifted,
            target_axes=target_axes,
            buffer_axes=buffer_axes,
            axis_names=frozenset(axis_index.keys()),
            reducible_axes=reducible_axes,
            axis_weights=axis_weights,
            axis_coords=axis_coords,
            axis_types=axis_types,
            shaped_param_axes=shaped_param_axes,
        )
    except UnsupportedIRLoweringError:
        return None
    tree = ast.Expression(body=body)
    ast.fix_missing_locations(tree)
    return tree


_ARITH_OPS: dict[str, type[ast.operator]] = {
    "+": ast.Add,
    "-": ast.Sub,
    "*": ast.Mult,
    "/": ast.Div,
}


def _try_collapse_to_full_sum(
    expr_ir: Expr,
    *,
    name_to_template: Mapping[str, _BufferTemplate],
) -> ast.expr | None:
    """If ``expr_ir`` is a flat sum of ALL cells of one buffer, emit ``np.sum(buf)``.

    Walks a tree of binary/N-ary ``Apply("+", ...)`` nodes whose leaves are
    all ``Sym`` nodes.  If every leaf maps to the same templated buffer and
    the set of referenced cells exhausts the buffer's full cell list, emits a
    compact ``np.sum(buf)`` call instead of constructing per-cell subscripts.

    Returns:
        An ``ast.expr`` for ``np.sum(<base>_buf)`` on a full-coverage match,
        otherwise ``None``.
    """
    # Collect all leaf Sym names from the flat + chain.
    syms: list[str] = []
    stack: list[Expr] = [expr_ir]
    while stack:
        node = stack.pop()
        if isinstance(node, Apply) and node.op == "+" and len(node.args) >= 2:
            stack.extend(node.args)
        elif isinstance(node, Sym):
            syms.append(node.name)
        else:
            return None  # non-sum, non-sym leaf → not a simple flat sum
    if not syms:
        return None
    first_tpl = name_to_template.get(syms[0])
    if first_tpl is None or not first_tpl.axes:
        return None
    expected = set(first_tpl.expanded_names)
    if set(syms) != expected:
        return None
    # All cells of one buffer are referenced exactly once → np.sum(buf).
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="np", ctx=ast.Load()),
            attr="sum",
            ctx=ast.Load(),
        ),
        args=[ast.Name(id=f"{first_tpl.base}_buf", ctx=ast.Load())],
        keywords=[],
    )


def _lower_multicell_sym_ir_to_ast(  # noqa: C901
    expr_ir: Expr,
    *,
    name_to_template: Mapping[str, _BufferTemplate],
    axis_index: Mapping[str, Mapping[str, int]],
) -> ast.expr | None:
    """Lower a scalar IR expression that references expanded cell names.

    For scalar aliases composed of Add/Sub arithmetic over expanded state
    cell names (e.g. ``I__age_y__loc_a + I__age_y__loc_b + ...``) where
    ``lift_cell_ir_to_template`` refuses due to multiple-cell ambiguity, we
    can directly substitute each ``Sym(nm)`` with ``buf[i, j]`` using the
    integer indices from the template's coord_assignments.

    When all referenced cells belong to the same buffer and cover it fully,
    emits ``np.sum(buf)`` directly (via :func:`_try_collapse_to_full_sum`)
    rather than constructing per-cell subscript AST nodes.  For other
    arithmetic ``Apply('+'/'-')`` trees of ``Sym`` and ``Literal`` nodes,
    falls back to per-cell substitution.

    Returns:
        An ``ast.expr`` or ``None`` if any subnode cannot be lowered.
    """
    # Fast path: plain sum of all cells of one buffer → np.sum(buf).
    collapsed = _try_collapse_to_full_sum(expr_ir, name_to_template=name_to_template)
    if collapsed is not None:
        return collapsed

    # Build a coord-to-index map: {cell_name: ast.Subscript(buf, (i, j))}
    cell_ast: dict[str, ast.expr] = {}
    for tpl in name_to_template.values():
        if not tpl.axes:
            continue
        for nm, ca in zip(tpl.expanded_names, tpl.coord_assignments, strict=True):
            indices: list[ast.expr] = []
            ok = True
            for ax in tpl.axes:
                coord = ca.get(ax)
                if coord is None or ax not in axis_index or coord not in axis_index[ax]:
                    ok = False
                    break
                indices.append(ast.Constant(value=axis_index[ax][coord]))
            if ok:
                buf_name = f"{tpl.base}_buf"
                if len(indices) == 1:
                    sl: ast.expr = indices[0]
                else:
                    sl = ast.Tuple(elts=indices, ctx=ast.Load())
                cell_ast[nm] = ast.Subscript(
                    value=ast.Name(id=buf_name, ctx=ast.Load()),
                    slice=sl,
                    ctx=ast.Load(),
                )

    def _lower(node: Expr) -> ast.expr | None:
        """Best-effort IR -> AST lowering for the cell-equation rewriter.

        Returns ``None`` (rather than raising) for any node shape outside
        the supported subset, which lets the caller fall back to the
        generic per-cell lowering.

        Returns:
            Lowered AST expression, or ``None`` if unsupported.
        """
        if isinstance(node, Sym):
            return cell_ast.get(node.name)
        if isinstance(node, Literal):
            return ast.Constant(value=node.value)
        if isinstance(node, Apply) and node.op in _ARITH_OPS and len(node.args) >= 2:
            result = _lower(node.args[0])
            if result is None:
                return None
            for arg in node.args[1:]:
                right = _lower(arg)
                if right is None:
                    return None
                result = ast.BinOp(left=result, op=_ARITH_OPS[node.op](), right=right)
            return result
        return None

    return _lower(expr_ir)


def _rewrite_cell_to_vector(  # noqa: PLR0913
    *,
    expr_ir: Expr | None = None,
    expr_ir_reduce: Expr | None = None,
    target_axes: tuple[str, ...],
    name_to_template: Mapping[str, _BufferTemplate],
    axis_index: Mapping[str, Mapping[str, int]],
    reducible_axes: frozenset[str] = frozenset(),
    axis_weights: Mapping[str, tuple[float, ...]] | None = None,
    axis_coords: Mapping[str, tuple[str, ...]] | None = None,
    axis_types: Mapping[str, str] | None = None,
    shaped_param_axes: Mapping[str, tuple[str, ...]] | None = None,
) -> ast.Expression:
    """Lower per-cell IR to a shaped buffer expression via the IR fast path.

    Returns:
        An ``ast.Expression`` whose body evaluates to an array shaped per
        ``target_axes`` (or a scalar when ``target_axes`` is empty).

    Raises:
        ValueError: If neither IR fast path succeeds.
    """
    # Prefer reduce-bearing IR (preserves helper structure as Reduce nodes)
    # so the typed lowering can emit ``np.sum`` directly. Fall back to the
    # post-expansion raw IR fast path on any v1-scope violation.
    if expr_ir_reduce is not None:
        fast = _try_ir_fast_path(
            expr_ir_reduce,
            target_axes=target_axes,
            name_to_template=name_to_template,
            axis_index=axis_index,
            reducible_axes=reducible_axes,
            axis_weights=axis_weights,
            axis_coords=axis_coords,
            axis_types=axis_types,
            shaped_param_axes=shaped_param_axes,
        )
        if fast is not None:
            return fast
    if expr_ir is not None:
        fast = _try_ir_fast_path(
            expr_ir,
            target_axes=target_axes,
            name_to_template=name_to_template,
            axis_index=axis_index,
            reducible_axes=reducible_axes,
            axis_weights=axis_weights,
            axis_coords=axis_coords,
            axis_types=axis_types,
            shaped_param_axes=shaped_param_axes,
        )
        if fast is not None:
            return fast
    # Final fallback for scalar aliases (target_axes == ()) whose expression
    # spans multiple cells of the same template (e.g. an explicit sum written
    # as ``I__age_y__loc_a + I__age_y__loc_b + ...``).  Directly substitute
    # each expanded cell name with ``buf[i, j]``; when all cells of a buffer
    # are present :func:`_lower_multicell_sym_ir_to_ast` collapses to
    # ``np.sum(buf)`` directly.
    if not target_axes:
        ir_to_try = expr_ir_reduce if expr_ir_reduce is not None else expr_ir
        if ir_to_try is not None:
            body = _lower_multicell_sym_ir_to_ast(
                ir_to_try,
                name_to_template=name_to_template,
                axis_index=axis_index,
            )
            if body is not None:
                tree = ast.Expression(body=body)
                ast.fix_missing_locations(tree)
                return tree
    msg = "no IR fast path succeeded; no fallback available"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Template-level CSE
# ---------------------------------------------------------------------------


def _try_cse_eq_plan(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0914
    *,
    state_buffers: list[_BufferTemplate],
    rhs: NormalizedRhs,
    name_to_template: Mapping[str, _BufferTemplate],
    axis_index: Mapping[str, Mapping[str, int]],
    reducible_axes: frozenset[str] = frozenset(),
    axis_weights: Mapping[str, tuple[float, ...]] | None = None,
    axis_coords: Mapping[str, tuple[str, ...]] | None = None,
    axis_types: Mapping[str, str] | None = None,
    shaped_param_axes: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[tuple[tuple[str, CodeType], ...], list[_EqGroup]] | None:
    """Build equation groups via template-level CSE across all state templates.

    Lifts the representative (first-cell) IR for each state template into
    template form, runs :func:`extract_common_subexpressions` across all
    lifted IRs to find shared sub-expressions (e.g. ``beta * S_buf *
    I_total_buf`` shared between the S and I equations of an SIR model), and
    compiles the resulting CSE bindings and rewritten equation codes.

    Requirements for CSE to apply:
    * At least two state templates (nothing to share in a single-template model).
    * All templates share the same ``axes`` tuple (so CSE temps have a single
      well-defined shape).
    * ``extract_common_subexpressions`` finds at least one common sub-expression
      with cost ≥ 2.

    Returns:
        ``(cse_codes, eq_groups)`` on success, or ``None`` if any guard fails
        or any lift/lower/compile step raises an error.  ``None`` causes the
        caller to fall back to the non-CSE :func:`_vectorize_template_equations`
        path.
    """
    if len(state_buffers) < 2:
        return None

    # All templates must share the same axes for CSE temps to have a
    # well-defined shape.  This is the common case in epidemic models where
    # all compartments are defined over the same (age x loc) grid.
    common_axes = state_buffers[0].axes
    if any(buf.axes != common_axes for buf in state_buffers[1:]):
        return None
    if not common_axes:
        return None

    # Build the cell→template and buffer_axes dicts needed by lower_to_vector_ast.
    cell_to_template: dict[str, tuple[str, tuple[str, ...]]] = {}
    buffer_axes: dict[str, tuple[str, ...]] = {}
    for cell_name, tpl in name_to_template.items():
        cell_to_template[cell_name] = (tpl.base, tpl.axes)
        if tpl.axes:
            buffer_axes[tpl.base] = tpl.axes

    axis_names = frozenset(axis_index.keys())

    # Lift the first-cell IR for each template to template form.
    # Prefer equations_ir_reduce (preserves Reduce helper structure) so that
    # lowering can emit np.sum() directly for apply_along terms.
    eq_ir_reduce = rhs.equations_ir_reduce
    eq_ir = rhs.equations_ir
    lifted_irs: list[Expr] = []
    for buf in state_buffers:
        first_idx = buf.offset
        cell_ir: Expr | None = None
        if eq_ir_reduce and len(eq_ir_reduce) > first_idx:
            cell_ir = eq_ir_reduce[first_idx]
        if cell_ir is None and eq_ir and len(eq_ir) > first_idx:
            cell_ir = eq_ir[first_idx]
        if cell_ir is None:
            return None
        try:
            lifted = lift_cell_ir_to_template(
                cell_ir, cell_to_template=cell_to_template
            )
        except UnsupportedIRLoweringError:
            return None
        lifted_irs.append(lifted)

    # Reserve names that must not be used as CSE temp names.
    reserved: set[str] = {buf.base for buf in state_buffers}
    reserved.update(cell_to_template.keys())

    bindings, rewritten_irs = extract_common_subexpressions(
        lifted_irs, prefix="_cse", min_cost=2, reserved_names=reserved
    )
    if not bindings:
        return None  # No shared sub-expressions found — CSE is a no-op here.

    # Compile CSE binding codes.  Each binding is computed once after alias
    # buffers are assembled; its result is added to the eval env under the
    # assigned name so rewritten equation codes can reference it by Sym.
    cse_codes_list: list[tuple[str, CodeType]] = []
    for name, binding_ir in bindings:
        try:
            body = lower_to_vector_ast(
                binding_ir,
                target_axes=common_axes,
                buffer_axes=buffer_axes,
                axis_names=axis_names,
                reducible_axes=reducible_axes,
                axis_weights=axis_weights,
                axis_coords=axis_coords,
                axis_types=axis_types,
                shaped_param_axes=shaped_param_axes,
            )
        except UnsupportedIRLoweringError:
            return None
        tree = ast.Expression(body=body)
        ast.fix_missing_locations(tree)
        try:
            code = compile(tree, filename="<op_system_cse>", mode="eval")
        except (ValueError, TypeError, SyntaxError):
            return None
        cse_codes_list.append((name, code))

    # Compile one rewritten equation code per state template.
    eq_groups: list[_EqGroup] = []
    for buf, rewritten_ir in zip(state_buffers, rewritten_irs, strict=True):
        try:
            body = lower_to_vector_ast(
                rewritten_ir,
                target_axes=buf.axes,
                buffer_axes=buffer_axes,
                axis_names=axis_names,
                reducible_axes=reducible_axes,
                axis_weights=axis_weights,
                axis_coords=axis_coords,
                axis_types=axis_types,
                shaped_param_axes=shaped_param_axes,
            )
        except UnsupportedIRLoweringError:
            return None
        tree = ast.Expression(body=body)
        ast.fix_missing_locations(tree)
        try:
            code = compile(tree, filename="<op_system_vec>", mode="eval")
        except (ValueError, TypeError, SyntaxError):
            return None
        # CSE path always produces a fully-vectorized code (no unrolled axes).
        assembly_perm = tuple(range(len(buf.axes)))
        eq_groups.append(
            _EqGroup(
                base=buf.base,
                codes=(code,),
                vec_axes=buf.axes,
                vec_shape=buf.shape,
                unroll_axes=(),
                unroll_shape=(),
                assembly_perm=assembly_perm,
                full_shape=buf.shape,
            )
        )

    return tuple(cse_codes_list), eq_groups


def _vectorize_template_equations(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915
    *,
    template: _BufferTemplate,
    equations: tuple[str, ...],
    equations_ir: tuple[Expr | None, ...] | None = None,
    equations_ir_reduce: tuple[Expr | None, ...] | None = None,
    name_to_template: Mapping[str, _BufferTemplate],
    axis_index: Mapping[str, Mapping[str, int]],
    reducible_axes: frozenset[str] = frozenset(),
    axis_weights: Mapping[str, tuple[float, ...]] | None = None,
    axis_coords: Mapping[str, tuple[str, ...]] | None = None,
    axis_types: Mapping[str, str] | None = None,
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
    cell_irs_reduce = (
        equations_ir_reduce[template.offset : template.offset + size]
        if equations_ir_reduce is not None
        and len(equations_ir_reduce) == len(equations)
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
                """Return the flat cell index for a vec-axis coord position."""
                f = base_flat
                for k_pos, ax_pos in enumerate(vec_idx):
                    f += vec_coord_idx[k_pos] * strides[ax_pos]
                return f

            first_idx = _bin_flat([0] * len(vec_idx))
            last_idx = _bin_flat([vec_shape[k] - 1 for k in range(len(vec_idx))])

            try:
                first_tree = _rewrite_cell_to_vector(
                    expr_ir=cell_irs[first_idx] if cell_irs else None,
                    expr_ir_reduce=(
                        cell_irs_reduce[first_idx] if cell_irs_reduce else None
                    ),
                    target_axes=vec_axes,
                    name_to_template=name_to_template,
                    axis_index=axis_index,
                    reducible_axes=reducible_axes,
                    axis_weights=axis_weights,
                    axis_coords=axis_coords,
                    axis_types=axis_types,
                    shaped_param_axes=shaped_param_axes,
                )
            except (ValueError, RuntimeError):
                ok = False
                break
            if vec_size > 1:
                try:
                    last_tree = _rewrite_cell_to_vector(
                        expr_ir=cell_irs[last_idx] if cell_irs else None,
                        expr_ir_reduce=(
                            cell_irs_reduce[last_idx] if cell_irs_reduce else None
                        ),
                        target_axes=vec_axes,
                        name_to_template=name_to_template,
                        axis_index=axis_index,
                        reducible_axes=reducible_axes,
                        axis_weights=axis_weights,
                        axis_coords=axis_coords,
                        axis_types=axis_types,
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
    return _build_vector_plan_inner(rhs)


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
    reducible_axes_set: set[str] = set()
    axis_weights: dict[str, tuple[float, ...]] = {}
    axis_types: dict[str, str] = {}
    for ax in axes_meta:
        if not isinstance(ax, Mapping):
            return None
        coords = ax.get("coords")
        if not coords:
            return None
        # Stringify coords to match the convention in
        # ``_templates.build_axis_lookup`` (and therefore the per-cell
        # ``coord_assignments`` carried by state templates). For categorical
        # axes coords are already strings; for continuous axes declared with
        # explicit numeric coords (e.g. ``[0.0, 5.0, ...]``) the unconverted
        # raw values would key ``axis_index`` by ``float`` while lookups use
        # ``str(float)``, causing a ``KeyError`` in downstream lowering.
        axes_pairs.append((ax["name"], [str(c) for c in coords]))
        ax_type = str(ax.get("type", "categorical")).strip().lower()
        axis_types[ax["name"]] = ax_type
        if ax_type in {"categorical", "ordinal"}:
            reducible_axes_set.add(ax["name"])
        deltas = ax.get("deltas")
        if deltas:
            axis_weights[ax["name"]] = tuple(float(d) for d in deltas)
    axis_index = {ax: {c: i for i, c in enumerate(coords)} for ax, coords in axes_pairs}
    axis_coords: dict[str, tuple[str, ...]] = {
        ax: tuple(coords) for ax, coords in axes_pairs
    }
    reducible_axes = frozenset(reducible_axes_set)

    state_buffers: list[_BufferTemplate] = []
    name_to_template: dict[str, _BufferTemplate] = {}
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
        for nm in tpl.expanded_names:
            name_to_template[nm] = buf

    # Guard: if aliases exist but alias_templates is not populated, fall back.
    if rhs.aliases and not rhs.alias_templates:
        return None
    alias_buffers: list[_BufferTemplate] = []
    for tpl in rhs.alias_templates:
        buf = _BufferTemplate(
            base=tpl.base,
            axes=tpl.axes,
            shape=tpl.shape,
            expanded_names=tpl.expanded_names,
            coord_assignments=tpl.coord_assignments,
            offset=0,
        )
        alias_buffers.append(buf)
        for nm in tpl.expanded_names:
            name_to_template[nm] = buf

    # Shaped-param buffer map: base name → axes tuple.
    # Used by _try_ir_fast_path (via lower_to_vector_ast).
    shaped_param_axes: dict[str, tuple[str, ...]] = {}
    for name, ax_tuple in rhs.shaped_params:
        ax_t = tuple(ax_tuple)
        if ax_t:
            shaped_param_axes[name] = ax_t

    # Vectorize aliases (in declaration order — best-effort dependency order).
    alias_codes: list[tuple[str, CodeType, tuple[int, ...]]] = []
    for buf in alias_buffers:
        try:
            if buf.axes:
                tree = _rewrite_cell_to_vector(
                    expr_ir=rhs.aliases_ir.get(buf.expanded_names[0]),
                    expr_ir_reduce=rhs.aliases_ir_reduce.get(buf.expanded_names[0]),
                    target_axes=buf.axes,
                    name_to_template=name_to_template,
                    axis_index=axis_index,
                    reducible_axes=reducible_axes,
                    axis_weights=axis_weights,
                    axis_coords=axis_coords,
                    axis_types=axis_types,
                    shaped_param_axes=shaped_param_axes,
                )
                if len(buf.expanded_names) > 1:
                    last_tree = _rewrite_cell_to_vector(
                        expr_ir=rhs.aliases_ir.get(buf.expanded_names[-1]),
                        expr_ir_reduce=rhs.aliases_ir_reduce.get(
                            buf.expanded_names[-1]
                        ),
                        target_axes=buf.axes,
                        name_to_template=name_to_template,
                        axis_index=axis_index,
                        reducible_axes=reducible_axes,
                        axis_weights=axis_weights,
                        axis_coords=axis_coords,
                        axis_types=axis_types,
                        shaped_param_axes=shaped_param_axes,
                    )
                    if ast.dump(tree) != ast.dump(last_tree):
                        return None
            else:
                # Scalar alias: rewrite so referenced templated state cells
                # become buffer-index accesses.
                tree = _rewrite_cell_to_vector(
                    expr_ir=rhs.aliases_ir.get(buf.expanded_names[0]),
                    expr_ir_reduce=rhs.aliases_ir_reduce.get(buf.expanded_names[0]),
                    target_axes=(),
                    name_to_template=name_to_template,
                    axis_index=axis_index,
                    reducible_axes=reducible_axes,
                    axis_weights=axis_weights,
                    axis_coords=axis_coords,
                    axis_types=axis_types,
                    shaped_param_axes=shaped_param_axes,
                )
            code = compile(tree, filename="<op_system_vec>", mode="eval")
        except (ValueError, RuntimeError, TypeError, SyntaxError):
            return None
        alias_codes.append((buf.base, code, buf.shape))

    # Try template-level IR CSE: finds shared sub-expressions across all
    # state templates and compiles them once as plan-level temporaries.
    cse_codes: tuple[tuple[str, CodeType], ...] = ()
    cse_result = _try_cse_eq_plan(
        state_buffers=state_buffers,
        rhs=rhs,
        name_to_template=name_to_template,
        axis_index=axis_index,
        reducible_axes=reducible_axes,
        axis_weights=axis_weights,
        axis_coords=axis_coords,
        axis_types=axis_types,
        shaped_param_axes=shaped_param_axes,
    )

    if cse_result is not None:
        cse_codes, eq_groups = cse_result
    else:
        # Fall back to per-template vectorization without cross-template CSE.
        eq_groups = []
        for buf in state_buffers:
            result = _vectorize_template_equations(
                template=buf,
                equations=rhs.equations,
                equations_ir=None,
                equations_ir_reduce=rhs.equations_ir_reduce,
                name_to_template=name_to_template,
                axis_index=axis_index,
                reducible_axes=reducible_axes,
                axis_weights=axis_weights,
                axis_coords=axis_coords,
                axis_types=axis_types,
                shaped_param_axes=shaped_param_axes,
            )
            if result is None:
                return None
            codes, vec_axes, vec_shape, unroll_axes, unroll_shape, assembly_perm = (
                result
            )
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
        param_templates=(),
        alias_codes=tuple(alias_codes),
        eq_groups=tuple(eq_groups),
        n_state=len(rhs.state_names),
        extra_param_buffers=tuple(
            (base, axes, tuple(len(axis_index[a]) for a in axes))
            for base, axes in shaped_param_axes.items()
        ),
        cse_codes=cse_codes,
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
    cse_codes = plan.cse_codes
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

    def eval_fn(t: object, y: object, **params: object) -> Float64Array:  # noqa: C901, PLR0912, PLR0914, PLR0915
        """Vectorized evaluation closure produced by :func:`build_eval_fn`.

        Computes the RHS for the entire state vector at once using the
        compiled per-bin AST callables, gathered into a single output
        array in ``y``'s array namespace.

        Returns:
            ``dydt`` array of shape ``(n_state,)`` in ``y``'s namespace.

        Raises:
            ValueError: If a required parameter is missing or has the
                wrong shape.
        """
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

        # Eval plan-level CSE temporaries (shared sub-expressions extracted
        # across state templates).  Computed after alias buffers so that
        # expressions referencing alias buffers (e.g. I_total_buf) resolve.
        for name, code in cse_codes:
            try:
                val = eval(  # noqa: S307
                    code, {"__builtins__": _SAFE_BUILTINS}, env
                )
            except (NameError, ValueError, TypeError, ArithmeticError) as exc:
                msg = f"CSE temp {name!r} evaluation failed: {exc!r}"
                raise ValueError(msg) from exc
            env[name] = val

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
