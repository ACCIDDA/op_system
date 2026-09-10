"""op_system._reactions.

Per-transition ("reaction") compiled artifacts for ``kind: transitions``
specs.

The deterministic RHS path (``_normalize._build_transition_equations_ir``)
deliberately discards per-transition identity: every transition's flow
contribution is folded into a per-state accumulator and summed into one
combined ``d(state)/dt`` expression, with heavy identity-based sharing and
deduplication (documented against issue #145) that makes reconstructing
per-transition structure from the summed equations impractical (a shared
``Expr`` object can legitimately serve many transitions/cells at once).

This module builds a SEPARATE, independent artifact instead of trying to
extend that hot path: for every *named* transition whose ``to``-side
wildcard axes are a subset of its ``from``-side wildcard axes (i.e. no
axis is "created" transitioning from -> to, and the rate expression
doesn't reference any axis outside that set), it captures:

- a template-form propensity expression, shaped like the ``from``-side
  template (one independent rate per source cell), and
- lightweight axis bookkeeping (which ``from``-axes get pinned to a fixed
  ``to``-coordinate, which get summed away) describing how a firing event
  at a given source cell maps onto the destination state.

This is deliberately narrower than the full transitions grammar (no
``to``-side axis not present on ``from``, no rate expression referencing
an axis absent from ``from``) -- see ``docs`` / the originating issue for
why: a stochastic/CTMC consumer needs "how many independent source cells
are firing, and where does each firing land", which these excluded shapes
don't have a well-defined answer for without design work beyond this
artifact's scope. Transitions outside this scope are simply omitted from
the artifact tuple, not an error -- mirrors how ``history_requirements``
is built opportunistically elsewhere in this package.

``from: null`` SOURCE-ONLY transitions (an exogenous hazard with no
compartment to deplete -- e.g. cross-district case importation) ARE in
scope: unlike the narrowing above, a source-only transition's firing-cell
count has a perfectly well-defined answer -- one independent Poisson
process per DESTINATION cell, with no source population to bound it
against. ``from_base`` is ``None`` and ``from_axes``/``full_axes`` are
taken from the ``to``-side template instead of a nonexistent ``from``-side
one; see ``ReactionArtifactIR``'s own field docs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from op_system._axes import _normalize_bracket_key
from op_system._helpers import _get_required_str
from op_system._ir import (
    Apply,
    AxisIndex,
    Expr,
    Reduce,
    Subscript,
    _map_children,
    iter_subscripts,
    parse_expr_to_ir,
    unparse_ir,
    walk,
)
from op_system._ir_expand import expand_reduce_pointwise
from op_system._ir_templates import expand_inline_templates
from op_system._templates import (
    PinnedToken,
    WildcardToken,
    parse_selector,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class ReactionArtifactIR:
    """Normalize-time record for one in-scope named transition.

    Attributes:
        name: The transition's own ``name:`` (required -- unnamed
            transitions are never included).
        from_base: State base name the reaction depletes, or ``None`` for a
            SOURCE-ONLY (``from: null``) transition -- an exogenous
            hazard (e.g. cross-district case importation) with no
            compartment to deplete, only a ``to_base`` to increment. A
            consumer must special-case ``from_base is None`` (skip every
            depletion/clamp step; only the ``to_base`` scatter applies) --
            see ``run_hybrid_ctmc`` in diphtheria_outbreakvacc.
        from_axes: Wildcard axes of the ``from`` template, in declaration
            order -- this is the shape of the compiled propensity. For a
            source-only transition (no ``from`` template to speak of),
            this is instead the ``to`` template's own wildcard axes: the
            propensity is one independent hazard per DESTINATION cell,
            with no source population to multiply by.
        full_axes: EVERY axis of ``from_base``'s true (unreduced) template,
            in declaration order, whether wildcard or pinned on this
            transition's ``from``-side selector -- i.e. ``from_axes`` plus
            any axis this transition pins on the from-side (e.g. a
            vaccination-dose-progression transition pinned at
            ``vax=unvaccinated`` on ``from``). Used to correctly reference
            ``from_base``'s true shape (see ``propensity_ir_full``) and,
            by a consumer, to build a complete scatter-target index into
            ``to_base`` (assumed to share this same axis order -- see
            ``run_hybrid_ctmc`` in diphtheria_outbreakvacc for the one
            current consumer's validation of that assumption). For a
            source-only transition this is the ``to`` template's own full
            axis order (there is no ``from`` template at all).
        to_base: State base name the reaction replenishes.
        to_axes: Wildcard axes of the ``to`` template, in declaration
            order. Always a subset of ``from_axes`` (enforced at build
            time) -- axes in ``from_axes`` but not here are summed away
            when a firing event is scattered onto the destination.
        pinned: ``(axis, coord)`` pairs for every axis pinned on the
            ``to``-side selector -- both axes that are wildcard on
            ``from`` and pinned on ``to`` (a "collapse to a fixed target"
            transition, e.g. recovery landing in a single ``vax=full``
            stratum regardless of the firing cell's own vax value) and
            axes that are ALSO pinned on ``from`` (a "point-to-point"
            transition between two specific coordinates on the same axis,
            e.g. ``vax=unvaccinated -> vax=partial`` dose progression --
            not a collapse, just a fixed single-cell shift, but the
            scatter target still needs this axis's coordinate fixed).
        from_pinned: ``(axis, coord)`` pairs for every axis pinned on the
            ``from``-side selector -- i.e. every axis in ``full_axes`` but
            not ``from_axes``. Distinct from ``pinned``: for a
            point-to-point transition, the same axis appears in both, but
            with DIFFERENT coordinates (e.g. ``vax=unvaccinated`` here vs.
            ``vax=partial`` in ``pinned``). A consumer needs this to know
            which single coordinate of ``from_base``'s true shape to
            deplete -- the compiled ``propensity_fn`` already reads from
            this coordinate internally (see ``propensity_ir_full``), but
            that isn't otherwise visible from the propensity array's own
            shape (``from_axes``), which carries no trace of it. Always
            ``()`` for a source-only transition (nothing to deplete, so
            nothing to pin on a from-side that doesn't exist).
        rate_ir_full: Template-form per-capita RATE IR (axes symbolic,
            Reduce nodes resolved) -- the bare rate expression as written
            in the transition's ``rate:`` field, kept for display/
            debugging. NOT what gets compiled as the propensity (see
            ``propensity_ir_full``).
        rate_string: Unparsed ``rate_ir_full``, for display/debugging.
        propensity_ir_full: Template-form PROPENSITY IR -- ``rate *
            from_state`` for a normal transition, matching the standard
            CTMC/tau-leaping per-cell hazard definition. For a
            source-only transition there is no ``from_state`` to
            multiply by, so this is just ``rate`` itself -- the hazard
            IS the rate, one independent Poisson process per destination
            cell (a genuine exogenous/immigration-style process). This is
            what compile-time lowering compiles.
        propensity_ir_reduce: Same, with Reduce nodes preserved (for the
            vector compile path, consistent with the rest of this
            package).
    """

    name: str
    from_base: str | None
    from_axes: tuple[str, ...]
    full_axes: tuple[str, ...]
    to_base: str
    to_axes: tuple[str, ...]
    pinned: tuple[tuple[str, str], ...]
    from_pinned: tuple[tuple[str, str], ...]
    rate_ir_full: Expr
    rate_string: str
    propensity_ir_full: Expr
    propensity_ir_reduce: Expr


def _in_order_wildcard_axes(tokens: list[Any]) -> list[str]:
    """Return wildcard axis names from parsed selector tokens, in order.

    Returns:
        Axis names for each :class:`WildcardToken` in ``tokens``, in the
        order they appear, without duplicates.
    """
    axes: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if isinstance(tok, WildcardToken) and tok.axis not in seen:
            axes.append(tok.axis)
            seen.add(tok.axis)
    return axes


@dataclass(frozen=True, slots=True)
class _AliasTemplate:
    """One alias declaration in template-symbolic (axes-still-free) form.

    Attributes:
        axes: The axes the alias's own selector declares, in selector
            order -- ``("age", "loc")`` for ``foi[age, loc]``. Order
            matters: a reference is matched to these positionally, so
            ``foi[age, loc:d1]`` binds ``loc`` and leaves ``age`` free.
        body: The alias's Reduce-preserving, axes-still-symbolic IR.
    """

    axes: tuple[str, ...]
    body: Expr


def _parse_alias_templates(
    aliases_raw: Mapping[str, Any],
    *,
    shaped_params: Mapping[str, tuple[str, ...]],
    axis_lookup: dict[str, list[str]],
) -> dict[str, _AliasTemplate]:
    """Parse each alias declaration into template-symbolic form.

    op_system's alias-inlining machinery (``inline_aliases`` /
    ``_build_aliases_ir_from_raw``) only ever operates on fully
    per-cell-expanded alias names -- there is no template-symbolic form of
    a templated alias body anywhere else in the package. This builds one
    locally, so a rate expression referencing an alias can be inlined at
    template scope, matching how this module already treats rate
    expressions in general.

    Bodies are NOT yet resolved against one another; see
    :func:`_resolve_alias_templates`.

    Returns:
        Mapping from alias base name to its :class:`_AliasTemplate`.
        Aliases with a non-string body or a non-wildcard LHS token (a
        pinned-axis alias declaration, e.g. ``foi[age, loc=d1]:``) are
        omitted -- out of scope here, not an error.
    """
    out: dict[str, _AliasTemplate] = {}
    for raw_key, expr_val in aliases_raw.items():
        if not isinstance(expr_val, str) or not expr_val.strip():
            continue
        base, tokens = parse_selector(_normalize_bracket_key(raw_key))
        token_list = list(tokens)
        if any(not isinstance(tok, WildcardToken) for tok in token_list):
            continue
        body = expand_inline_templates(
            parse_expr_to_ir(expr_val, lower_helpers=True),
            assignment={},
            shaped_params=shaped_params,
            axis_lookup=axis_lookup,
        )
        out[base] = _AliasTemplate(
            axes=tuple(_in_order_wildcard_axes(token_list)), body=body
        )
    return out


def _alias_index_substitution(
    declared_axes: tuple[str, ...],
    ref_indices: tuple[AxisIndex, ...],
    *,
    axis_names: frozenset[str],
) -> dict[str, AxisIndex] | None:
    """Map an alias's declared axes onto the indices a reference supplies.

    Three reference shapes occur in practice, distinguished exactly as
    :func:`op_system._ir.classify_axis_index` does:

    * FREE on its own axis (``gravity_pressure[loc]``) -- the body is used
      as written, no rewrite.
    * COORD (``infectious_weighted[loc:lp]``) -- the declared axis is
      bound to ``lp``. Covers both a reduction binding variable and a
      literal coordinate pin (``foi[age, loc:d1]``); they share this IR
      shape and want the same rewrite.
    * COORD_SYMBOL (``trv[v]`` inside ``apply_along(..., vax=v)``) -- the
      reference labels the position with the reduction's binding variable
      instead of the axis name, so the declared axis is bound to that
      variable.

    Returns:
        Mapping from declared axis name to the :class:`AxisIndex` every
        FREE occurrence of that axis in the body must become, or ``None``
        when the reference is out of scope (wrong arity, or renaming one
        registered axis to another -- the body's own buffer references
        need not carry the new axis at all, so that is not substitution
        this pass can safely perform).
    """
    if len(ref_indices) != len(declared_axes):
        return None
    subst: dict[str, AxisIndex] = {}
    for axis, idx in zip(declared_axes, ref_indices, strict=True):
        if idx.coord is not None:
            subst[axis] = AxisIndex(axis=axis, coord=idx.coord)
        elif idx.axis == axis:
            continue
        elif idx.axis in axis_names:
            return None
        else:
            subst[axis] = AxisIndex(axis=axis, coord=idx.axis)
    return subst


def _apply_axis_substitution(expr: Expr, subst: Mapping[str, AxisIndex]) -> Expr:
    """Bind an alias body's FREE axis indices per ``subst``.

    A ``Reduce`` binding does NOT shadow here: in
    ``apply_along(X[loc], loc=lp)`` the bare ``X[loc]`` still denotes the
    outer free axis (the bound form is ``X[loc:lp]``), which is precisely
    the co-occurrence ``_ir_lower._binding_collides_with_free_index``
    exists to detect. So every FREE occurrence is substituted regardless
    of enclosing bindings.

    Returns:
        A new IR expression with the substitution applied; ``expr``
        itself when nothing matches.
    """
    if isinstance(expr, Subscript):
        new_indices: list[AxisIndex] = []
        changed = False
        for idx in expr.indices:
            repl = subst.get(idx.axis) if idx.coord is None else None
            if repl is None:
                new_indices.append(idx)
                continue
            new_indices.append(repl)
            changed = True
        if changed:
            return Subscript(name=expr.name, indices=tuple(new_indices))
        return expr
    return _map_children(expr, lambda e: _apply_axis_substitution(e, subst))


def _rename_bound_vars(expr: Expr, renames: Mapping[str, str]) -> Expr:
    """Alpha-rename reduction binding variables, respecting their scopes.

    Only names actually bound by a ``Reduce`` inside ``expr`` are
    renamed, and only within that ``Reduce``'s own body -- a name that is
    merely referenced (bound by an enclosing scope outside ``expr``) is
    left alone.

    Returns:
        A new IR expression with the renames applied; ``expr`` itself
        when no enclosing binding matches.
    """
    if isinstance(expr, Reduce):
        active = {var: renames[var] for _, var in expr.bindings if var in renames}
        if not active:
            return _map_children(expr, lambda e: _rename_bound_vars(e, renames))
        bindings = tuple((axis, active.get(var, var)) for axis, var in expr.bindings)
        return Reduce(
            kind=expr.kind,
            bindings=bindings,
            body=_rename_bound_vars(_rename_var_references(expr.body, active), renames),
            filters=expr.filters,
            kernel=expr.kernel,
        )
    return _map_children(expr, lambda e: _rename_bound_vars(e, renames))


def _rename_var_references(expr: Expr, renames: Mapping[str, str]) -> Expr:
    """Rewrite subscript positions that name a renamed binding variable.

    A binding variable reaches a ``Subscript`` in two forms: as the coord
    of a bound position (``I[vax:v]``) and as a bare axis label where the
    spec uses the variable itself (``I[ap]``) -- both are rewritten, the
    same pair ``_ir_lower._lower_reduce`` accounts for.

    Returns:
        A new IR expression with matching positions renamed; ``expr``
        itself when nothing matches.
    """
    if isinstance(expr, Subscript):
        new_indices: list[AxisIndex] = []
        changed = False
        for idx in expr.indices:
            if idx.coord is not None and idx.coord in renames:
                new_indices.append(AxisIndex(axis=idx.axis, coord=renames[idx.coord]))
                changed = True
            elif idx.coord is None and idx.axis in renames:
                new_indices.append(AxisIndex(axis=renames[idx.axis]))
                changed = True
            else:
                new_indices.append(idx)
        if changed:
            return Subscript(name=expr.name, indices=tuple(new_indices))
        return expr
    return _map_children(expr, lambda e: _rename_var_references(e, renames))


def _bound_vars(expr: Expr) -> frozenset[str]:
    """Return every reduction binding variable bound anywhere under ``expr``.

    Returns:
        The set of binding variable names.
    """
    return frozenset(
        var
        for node in walk(expr)
        if isinstance(node, Reduce)
        for _, var in node.bindings
    )


def _inline_alias_body(
    template: _AliasTemplate,
    ref: Subscript,
    *,
    axis_names: frozenset[str],
) -> Expr | None:
    """Instantiate one alias reference against the alias's resolved body.

    Returns:
        The alias body rewritten for this reference site, or ``None``
        when the reference shape is out of scope.
    """
    subst = _alias_index_substitution(template.axes, ref.indices, axis_names=axis_names)
    if subst is None:
        return None
    body = template.body
    # Capture avoidance: the coord symbols we are about to substitute in
    # (e.g. the ``lp`` of ``infectious_weighted[loc:lp]``) belong to a
    # reduction enclosing the REFERENCE. If the body happens to bind the
    # same name itself, the substituted ``loc:lp`` would be captured by
    # that inner binding instead. Rename the body's colliding bindings.
    incoming = {idx.coord for idx in subst.values() if idx.coord is not None}
    clashing = incoming & _bound_vars(body)
    if clashing:
        body = _rename_bound_vars(
            body, {var: f"{var}__op_alias{n}" for n, var in enumerate(sorted(clashing))}
        )
    return _apply_axis_substitution(body, subst)


def _substitute_alias_refs(
    expr: Expr,
    templates: Mapping[str, _AliasTemplate],
    *,
    axis_names: frozenset[str],
) -> Expr:
    """Replace in-scope alias ``Subscript`` references with their bodies.

    ``templates`` bodies are expected to be already resolved (free of
    alias references themselves), so this performs a single pass.

    Returns:
        A new IR expression with in-scope alias references inlined;
        structurally equal to ``expr`` when no replacements occur.
    """
    if isinstance(expr, Subscript):
        template = templates.get(expr.name)
        if template is None:
            return expr
        inlined = _inline_alias_body(template, expr, axis_names=axis_names)
        return expr if inlined is None else inlined
    return _map_children(
        expr, lambda e: _substitute_alias_refs(e, templates, axis_names=axis_names)
    )


def _resolve_alias_templates(
    parsed: Mapping[str, _AliasTemplate],
    *,
    axis_names: frozenset[str],
) -> dict[str, _AliasTemplate]:
    """Inline alias-to-alias references so every body stands alone.

    Resolves the whole chain (issue #193), not just one level: an alias
    whose body references an alias that itself references a third is
    fully expanded. Bodies are resolved depth-first and memoized, so each
    is expanded once regardless of how many aliases reference it.

    Cycle detection mirrors ``_ir_templates._detect_alias_cycle`` but
    operates at template-symbolic scope, where a reference is a
    ``Subscript`` with symbolic axes rather than the fully-expanded
    per-cell ``Sym`` that function expects. An alias on a cycle -- and
    any alias that transitively references one -- is dropped from the
    result, so a rate referencing it is simply left unresolved (the same
    silent-omission behavior applied to every other out-of-scope shape
    here, not an error).

    Returns:
        Mapping from alias base name to a template whose body contains no
        remaining alias references.
    """
    resolved: dict[str, _AliasTemplate] = {}
    visiting: set[str] = set()
    failed: set[str] = set()

    def _resolve(name: str) -> _AliasTemplate | None:
        if name in resolved:
            return resolved[name]
        if name in failed or name in visiting:
            failed.add(name)  # cycle, or already known unresolvable
            return None
        template = parsed.get(name)
        if template is None:
            return None
        visiting.add(name)
        try:
            deps = {
                sub.name
                for sub in iter_subscripts(template.body)
                if sub.name in parsed and sub.name != name
            }
            usable: dict[str, _AliasTemplate] = {}
            for dep in deps:
                dep_template = _resolve(dep)
                if dep_template is None:
                    failed.add(name)
                    return None
                usable[dep] = dep_template
            body = (
                _substitute_alias_refs(template.body, usable, axis_names=axis_names)
                if usable
                else template.body
            )
        finally:
            visiting.discard(name)
        out = _AliasTemplate(axes=template.axes, body=body)
        resolved[name] = out
        return out

    for name in parsed:
        _resolve(name)
    return resolved


def _build_alias_bodies(
    aliases_raw: Mapping[str, Any],
    *,
    shaped_params: Mapping[str, tuple[str, ...]],
    axis_lookup: dict[str, list[str]],
) -> dict[str, _AliasTemplate]:
    """Build fully-resolved template-symbolic bodies for every alias.

    Returns:
        Mapping from alias base name to a template whose body contains no
        remaining alias references.
    """
    parsed = _parse_alias_templates(
        aliases_raw, shaped_params=shaped_params, axis_lookup=axis_lookup
    )
    return _resolve_alias_templates(parsed, axis_names=frozenset(axis_lookup))


def build_reaction_artifacts_ir(  # ruff: ignore[too-many-arguments, too-many-locals]
    transitions_raw: list[Mapping[str, Any]],
    *,
    axes: list[dict[str, Any]],
    axis_lookup: dict[str, list[str]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    time_axis_name: str | None = None,
    aliases_raw: Mapping[str, Any] | None = None,
) -> tuple[ReactionArtifactIR, ...]:
    """Build per-transition reaction artifacts for in-scope named transitions.

    Runs independently of (and has no effect on) the deterministic
    per-state equation construction in ``_build_transition_equations_ir``.

    Args:
        transitions_raw: The spec's raw (pre-expansion) ``transitions:``
            entries.
        axes: The spec's raw (pre-expansion) ``axes:`` entries.
        axis_lookup: Mapping from axis name to its declared coord list.
        shaped_params: Optional mapping from shaped-parameter base name to
            its registered axis tuple (see ``expand_inline_templates``).
        time_axis_name: Optional name of the spec's time axis, if any --
            excluded from the from-side wildcard-axis scope check since
            the engine handles it separately.
        aliases_raw: Optional raw ``aliases:`` mapping from the spec (LHS
            selector string, e.g. ``"foi[age, loc]"``, to RHS expression
            string). A rate referencing an alias has that alias's
            template-symbolic body inlined before the axis-scope check
            below, following the whole chain when that alias references
            further aliases -- see :func:`_build_alias_bodies`. An alias
            on a reference cycle, or one whose reference shape is out of
            scope, is left unresolved, same as when this argument is
            omitted.

    Returns:
        One :class:`ReactionArtifactIR` per named, in-scope transition, in
        declaration order. Transitions without a ``name:``, transitions
        whose ``to``-side introduces a wildcard axis absent from ``from``
        (not applicable to a source-only transition, which has no
        ``from``-side to compare against), and transitions whose rate
        (after alias inlining) references an axis outside the
        from-side wildcard set (the to-side wildcard set, for a
        source-only transition) are silently omitted (not an error -- see
        module docstring).
    """
    shaped = shaped_params or {}
    alias_bodies = _build_alias_bodies(
        aliases_raw or {}, shaped_params=shaped, axis_lookup=axis_lookup
    )
    axis_names = frozenset(axis_lookup)
    out: list[ReactionArtifactIR] = []

    for tr_map in transitions_raw:
        if not isinstance(tr_map, dict):
            continue
        name_s = tr_map.get("name")
        if not isinstance(name_s, str) or not name_s.strip():
            continue  # unnamed: not addressable, skip.

        frm_raw = tr_map.get("from")
        source_only = frm_raw is None

        to_s = _get_required_str(tr_map, idx=-1, key="to")
        rate_s = _get_required_str(tr_map, idx=-1, key="rate")
        to_base, to_tokens = parse_selector(to_s)
        to_wc_axes = _in_order_wildcard_axes(list(to_tokens))

        if source_only:
            # No from-side template at all -- the propensity's shape is
            # the destination's own wildcard axes (one independent hazard
            # per destination cell), and there's nothing to check the
            # to-side against.
            frm_base = None
            frm_tokens: list[Any] = []
            frm_wc_axes = to_wc_axes
            frm_wc_set = set(to_wc_axes)
        else:
            frm_s = _get_required_str(tr_map, idx=-1, key="from")
            frm_base, frm_tokens = parse_selector(frm_s)
            frm_wc_axes = _in_order_wildcard_axes(list(frm_tokens))
            frm_wc_set = set(frm_wc_axes)

            # to-side must not introduce a wildcard axis absent from from-side.
            if any(ax not in frm_wc_set for ax in to_wc_axes):
                continue

        ir_rate_raw = parse_expr_to_ir(rate_s, lower_helpers=True)
        if alias_bodies:
            ir_rate_raw = _substitute_alias_refs(
                ir_rate_raw, alias_bodies, axis_names=axis_names
            )
        # Rate must not reference an axis outside the from-side wildcard
        # set (other than the time axis, which is handled separately by
        # the engine, not baked into the propensity template).
        rate_axes = {
            ix.axis
            for sub in iter_subscripts(ir_rate_raw)
            for ix in sub.indices
            if ix.axis is not None
        }
        if any(
            ax not in frm_wc_set and ax != time_axis_name and ax in axis_lookup
            for ax in rate_axes
        ):
            continue

        rate_ir_reduce = expand_inline_templates(
            ir_rate_raw,
            assignment={},
            shaped_params=shaped,
            axis_lookup=axis_lookup,
        )
        rate_ir_full = expand_reduce_pointwise(
            rate_ir_reduce,
            axes=list(axes),
            shaped_params=shaped,
            lhs_assignment={},
            axis_coords=axis_lookup,
        )

        if source_only:
            # No from_state to multiply by -- the propensity IS the rate
            # itself (one independent Poisson hazard per destination
            # cell). full_axes is the to-side template's own full axis
            # order (there is no from-side template at all).
            full_axes = tuple(tok.axis for tok in to_tokens)
            propensity_ir_full = rate_ir_full
            propensity_ir_reduce = rate_ir_reduce
            from_pinned: tuple[tuple[str, str], ...] = ()
        else:
            # Propensity = rate * from_state (the actual per-cell hazard),
            # NOT the bare rate -- mirrors the deterministic path's
            # tpl_flow_full construction in
            # _normalize._build_transition_equations_ir.
            #
            # from_sub must reference from_base's TRUE full shape, not just
            # its wildcard axes: a from-side PinnedToken (e.g.
            # `S[age, vax=u, loc]`) still needs that coordinate baked into
            # the Subscript, or the reference silently points at the wrong
            # (or a shape-mismatched) slice. full_axes -- every token's
            # axis, wildcard or pinned, in the selector's own declared
            # order -- is exactly from_base's true axis order (a
            # well-formed selector mentions every axis of the state it
            # references), so build indices from ALL of frm_tokens, not
            # just the wildcard subset.
            full_axes = tuple(tok.axis for tok in frm_tokens)
            assert frm_base is not None  # ruff: ignore[assert]  # narrows for mypy: not source_only here
            from_sub = Subscript(
                name=frm_base,
                indices=tuple(
                    AxisIndex(
                        axis=tok.axis,
                        coord=(tok.coord if isinstance(tok, PinnedToken) else None),
                    )
                    for tok in frm_tokens
                ),
            )
            propensity_ir_full = Apply(op="*", args=(rate_ir_full, from_sub))
            propensity_ir_reduce = Apply(op="*", args=(rate_ir_reduce, from_sub))

            # Every from-side pinned axis (full_axes minus from_axes) needs
            # its own coordinate recorded separately from `pinned` -- for a
            # point-to-point transition the same axis is pinned on both
            # sides but to DIFFERENT coordinates (e.g. vax=unvaccinated
            # here vs. vax=partial in `pinned`), so this can't be derived
            # from `pinned`.
            from_pinned = tuple(
                (tok.axis, tok.coord)
                for tok in frm_tokens
                if isinstance(tok, PinnedToken)
            )

        # Every to-side pinned axis needs a fixed scatter-target coordinate,
        # not just ones that are wildcard on from (the "collapse" case) --
        # an axis pinned on BOTH sides (e.g. vax=unvaccinated -> vax=partial
        # dose progression) is a "point-to-point" shift on that axis, not a
        # collapse, but the target coordinate still needs recording.
        pinned = tuple(
            (tok.axis, tok.coord) for tok in to_tokens if isinstance(tok, PinnedToken)
        )

        out.append(
            ReactionArtifactIR(
                name=name_s,
                from_base=frm_base,
                from_axes=tuple(frm_wc_axes),
                full_axes=full_axes,
                to_base=to_base,
                to_axes=tuple(to_wc_axes),
                pinned=pinned,
                from_pinned=from_pinned,
                rate_ir_full=rate_ir_full,
                rate_string=unparse_ir(rate_ir_full),
                propensity_ir_full=propensity_ir_full,
                propensity_ir_reduce=propensity_ir_reduce,
            )
        )

    return tuple(out)


__all__ = ["ReactionArtifactIR", "build_reaction_artifacts_ir"]
