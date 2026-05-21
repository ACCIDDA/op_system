"""op_system._templates.

Centralized template selector parsing, rendering, and expansion utilities.

All spec sections that accept axis-templated names route through this module:

  - state template expansion
  - alias key expansion
  - transition endpoint expansion (``from``/``to``)
  - ``initial_state`` key expansion (supports pinned coords)
  - chain stage name generation
  - operator ``apply_to`` expansion
  - ``coord_shift`` ``apply_to`` expansion

Selector syntax
---------------
A selector is a string of the form::

    Name[token1, token2, ...]

where each token is either:

  - ``axis``         wildcard — expands over all coords of that axis
  - ``axis=coord``   pinned  — holds this axis fixed at ``coord``

Mixed selectors expand only the wildcard axes while embedding pinned coords
in the rendered concrete name.  A bare name without brackets is returned
unchanged (single-element expansion with empty assignment).

Examples::

    X[age, vax, loc, imm=X0]  →  X__age_a0__vax_u__loc_usa__imm_X0, ...
    S[age]                     →  S__age_a0, S__age_a1, ...
    S                          →  S   (pass-through)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from op_system._errors import InvalidRhsSpecError

# Matches "Name[...]" with optional surrounding whitespace.
_STATE_TEMPLATE_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\[(.+)\]\s*")
# Matches any "Name[...]" token inside an expression string.
_INLINE_TEMPLATE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[(.*?)\]")

# Per-``template_map`` cache of pre-parsed, filtered, regex-compiled
# template keys used by ``_apply_template_substitutions``. Built lazily
# on first use and keyed by ``id(template_map)`` so batch substitution
# over many cells against the same mapping pays the parse cost once
# rather than per cell (issue #145).
_APPLY_TPL_PREPARSE_CACHE: dict[
    int,
    tuple[
        object,
        tuple[tuple[str, str, tuple["WildcardToken", ...], "re.Pattern[str]"], ...],
    ],
] = {}
# Matches any "[...]" fragment to extract placeholder names from rate exprs.
_PLACEHOLDER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\[(.*?)\]")


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WildcardToken:
    """Axis token that expands over all coordinates of that axis."""

    axis: str


@dataclass(frozen=True, slots=True)
class PinnedToken:
    """Axis token pinned to a specific coordinate."""

    axis: str
    coord: str


SelectorToken = WildcardToken | PinnedToken


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------


def _sanitize_fragment(val: object) -> str:
    """Sanitize a coord value for safe embedding in a state name.

    Replaces characters that are not alphanumeric or underscores with ``_``.

    Returns:
        Identifier-safe string fragment.
    """
    s = val if type(val) is str else str(val)
    cached = _SANITIZE_CACHE.get(s)
    if cached is not None:
        return cached
    result = _SANITIZE_RE.sub("_", s)
    _SANITIZE_CACHE[s] = result
    return result


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]")
_SANITIZE_CACHE: dict[str, str] = {}


def build_axis_lookup(axes: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build a ``{axis_name: [coord, ...]}`` mapping from normalized axes.

    Args:
        axes: List of normalized axis dicts; each must contain a ``name``
            entry and may contain a ``coords`` list (continuous axes
            without explicit coords map to an empty list).

    Returns:
        Dict mapping each axis name to its coordinate list (as strings).

    Examples:
        >>> build_axis_lookup([
        ...     {"name": "age", "coords": ["y", "o"]},
        ...     {"name": "vax", "coords": ["u", "v"]},
        ... ])
        {'age': ['y', 'o'], 'vax': ['u', 'v']}
        >>> build_axis_lookup([{"name": "x", "domain": {"lb": 0, "ub": 1}}])
        {'x': []}
    """
    return {ax["name"]: [str(c) for c in ax.get("coords", [])] for ax in axes}


def parse_selector(s: str) -> tuple[str, list[SelectorToken]]:
    """Parse a selector string into a base name and a list of tokens.

    Supports:

    - Bare name: ``"S"``           → ``("S", [])``
    - Wildcard:  ``"S[age, vax]"`` → ``("S", [WildcardToken("age"),
      WildcardToken("vax")])``
    - Pinned:    ``"X[imm=X0]"``  → ``("X", [PinnedToken("imm", "X0")])``
    - Mixed:     ``"X[age, imm=X0]"`` → ``("X", [WildcardToken("age"),
      PinnedToken("imm", "X0")])``

    Args:
        s: Selector string. May include surrounding whitespace and
            whitespace inside the brackets; both are stripped.

    Returns:
        ``(base, tokens)`` where ``tokens`` is empty for bare names. Token
        order matches the order of declaration inside the brackets.

    Raises:
        InvalidRhsSpecError: If a pinned token has an empty axis or coord,
            or if any axis appears in more than one token.

    Examples:
        >>> parse_selector("S")
        ('S', [])
        >>> parse_selector("X[age, imm=X0]")
        ('X', [WildcardToken(axis='age'), PinnedToken(axis='imm', coord='X0')])
        >>> try:
        ...     parse_selector("X[age, age=a0]")
        ... except Exception as exc:
        ...     print(type(exc).__name__)
        InvalidRhsSpecError
    """
    m = _STATE_TEMPLATE_RE.fullmatch(s)
    if not m:
        return s.strip(), []

    base = m.group(1)
    raw_tokens = [t.strip() for t in m.group(2).split(",") if t.strip()]
    tokens: list[SelectorToken] = []

    for tok in raw_tokens:
        if "=" in tok:
            axis, coord = [p.strip() for p in tok.split("=", 1)]
            if not axis or not coord:
                raise InvalidRhsSpecError(
                    detail=f"invalid pinned selector token {tok!r} in {s!r}"
                )
            tokens.append(PinnedToken(axis=axis, coord=coord))
        else:
            tokens.append(WildcardToken(axis=tok))

    # Reject duplicate axis names across all tokens.
    seen_axes: set[str] = set()
    for tok in tokens:
        ax = tok.axis
        if ax in seen_axes:
            raise InvalidRhsSpecError(detail=f"duplicate axis {ax!r} in selector {s!r}")
        seen_axes.add(ax)

    return base, tokens


def render_selector(
    base: str,
    tokens: Sequence[SelectorToken],
    assignment: Mapping[str, str],
    *,
    axis_lookup: Mapping[str, list[str]] | None = None,
) -> str:
    """Render a concrete state name from a base, tokens, and an assignment.

    Wildcard tokens draw their value from ``assignment``; pinned tokens use
    their embedded coordinate.  When ``axis_lookup`` is provided, validates
    that referenced axes and pinned coords are declared.

    If a wildcard token's axis is absent from ``assignment``, the original
    bracketed form is returned unchanged (allows partial rendering).

    Returns:
        Concrete name string, e.g. ``"X__age_a0__vax_u__imm_X0"``.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    if not tokens:
        return base

    parts: list[str] = []
    for tok in tokens:
        if isinstance(tok, WildcardToken):
            if axis_lookup is not None and tok.axis not in axis_lookup:
                raise InvalidRhsSpecError(
                    detail=f"selector references unknown axis {tok.axis!r}"
                )
            if tok.axis not in assignment:
                # Partial assignment — caller did not cover this axis.
                return (
                    base
                    + "["
                    + ",".join(
                        (
                            f"{t.axis}={t.coord}"
                            if isinstance(t, PinnedToken)
                            else t.axis
                        )
                        for t in tokens
                    )
                    + "]"
                )
            parts.append(f"{tok.axis}_{_sanitize_fragment(assignment[tok.axis])}")
        else:  # PinnedToken
            if axis_lookup is not None:
                if tok.axis not in axis_lookup:
                    raise InvalidRhsSpecError(
                        detail=(
                            f"selector pinned axis {tok.axis!r} references unknown axis"
                        )
                    )
                if tok.coord not in axis_lookup[tok.axis]:
                    raise InvalidRhsSpecError(
                        detail=(
                            f"selector pinned coord {tok.coord!r} not in "
                            f"axis {tok.axis!r} coords"
                        )
                    )
            parts.append(f"{tok.axis}_{_sanitize_fragment(tok.coord)}")

    return base + "__" + "__".join(parts)


def expand_selector(
    s: str,
    *,
    axis_lookup: dict[str, list[str]],
    context: str = "",
) -> list[tuple[str, dict[str, str]]]:
    """Expand a selector string over all wildcard axis combinations.

    Validates all axis and coord references against ``axis_lookup``.

    Args:
        s: Selector string, e.g. ``"X[age, vax, imm=X0]"``.
        axis_lookup: Mapping of axis name → coordinate list.
        context: Optional context string for error messages.

    Returns:
        List of ``(concrete_name, assignment)`` pairs, one per wildcard
        combination.  For bare names with no tokens, returns a single
        ``[(name, {})]``.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    ctx = f" in {context}" if context else ""
    base, tokens = parse_selector(s)

    if not tokens:
        return [(base, {})]

    # Validate all tokens and collect wildcard axes.
    wildcards: list[WildcardToken] = []
    for tok in tokens:
        if tok.axis not in axis_lookup:
            raise InvalidRhsSpecError(
                detail=f"selector axis {tok.axis!r} not defined{ctx}"
            )
        if isinstance(tok, PinnedToken):
            if tok.coord not in axis_lookup[tok.axis]:
                raise InvalidRhsSpecError(
                    detail=(
                        f"selector pinned coord {tok.coord!r} not in "
                        f"axis {tok.axis!r} coords{ctx}"
                    )
                )
        else:
            wildcards.append(tok)

    if not wildcards:
        # All tokens are pinned — single concrete expansion.
        assignment: dict[str, str] = {
            tok.axis: tok.coord for tok in tokens if isinstance(tok, PinnedToken)
        }
        name = render_selector(base, tokens, assignment, axis_lookup=axis_lookup)
        return [(name, assignment)]

    coords_lists = [axis_lookup[wt.axis] for wt in wildcards]
    results: list[tuple[str, dict[str, str]]] = []
    for combo in product(*coords_lists):
        assignment = {wildcards[i].axis: combo[i] for i in range(len(wildcards))}
        # Add pinned coords to assignment for full record-keeping.
        for tok in tokens:
            if isinstance(tok, PinnedToken):
                assignment[tok.axis] = tok.coord
        name = render_selector(base, tokens, assignment, axis_lookup=axis_lookup)
        results.append((name, assignment))

    return results


# ---------------------------------------------------------------------------
# Chain stage name helper
# ---------------------------------------------------------------------------


def _build_chain_stage_names(cname: str, *, length: int) -> list[str]:
    """Build chain stage names, preserving template tokens when present.

    Returns:
        Ordered list of stage name strings for the chain.
    """
    base, tokens = parse_selector(cname)
    if tokens:
        suffix = (
            "["
            + ",".join(
                f"{t.axis}={t.coord}" if isinstance(t, PinnedToken) else t.axis
                for t in tokens
            )
            + "]"
        )
        return [f"{base}{i}{suffix}" for i in range(1, length + 1)]
    return [f"{cname}{i}" for i in range(1, length + 1)]


# ---------------------------------------------------------------------------
# Expression-level template substitution helpers
# ---------------------------------------------------------------------------


def _extract_placeholders_from_expr(
    expr: str,
    *,
    shaped_param_names: set[str] | None = None,
) -> set[str]:
    """
    Extract wildcard placeholder axis names from ``[...]`` tokens in an expression.

    Only bare axis names (not ``axis=coord`` pins) are treated as placeholders.
    When ``shaped_param_names`` is provided, ``[...]`` tokens whose base name
    is a registered shaped parameter are skipped — such references are
    rewritten to literal-index subscripts and must not trigger axis
    expansion at the call site.

    Returns:
        Set of placeholder names referenced inside bracket templates.
    """
    placeholders: set[str] = set()
    skip = shaped_param_names or set()
    for match in _INLINE_TEMPLATE_RE.finditer(expr):
        base = match.group(1)
        if base in skip:
            continue
        inner = match.group(2)
        for part in inner.split(","):
            part_s = part.strip()
            if part_s and "=" not in part_s:
                placeholders.add(part_s)
    return placeholders


def _render_template_name(
    base: str, placeholders: list[str], assignment: Mapping[str, str]
) -> str:
    """Render an expanded name from base + wildcard placeholders + assignment.

    Returns:
        Concrete state name, e.g. ``"S__pop_p1__age_0_5"``.
    """
    parts = [f"{ph}_{_sanitize_fragment(assignment[ph])}" for ph in placeholders]
    return base + "__" + "__".join(parts)


def _render_template_or_literal(name_s: str, assignment: Mapping[str, str]) -> str:
    """
    Render a template name if placeholders are covered by assignment.

    Returns the literal string unchanged if no placeholders are present or
    any wildcard axis is missing from the assignment.

    Returns:
        Rendered concrete name or the original literal string.
    """
    base, tokens = parse_selector(name_s)
    wildcards = [t for t in tokens if isinstance(t, WildcardToken)]
    if not wildcards or any(wt.axis not in assignment for wt in wildcards):
        return name_s
    return _render_template_name(base, [wt.axis for wt in wildcards], assignment)


def _apply_template_substitutions(
    expr_s: str,
    *,
    assignment: Mapping[str, str],
    template_map: Mapping[str, list[tuple[str, dict[str, str]]]],
    shaped_params: Mapping[str, tuple[str, ...]] | None = None,
    axis_lookup: Mapping[str, list[str]] | None = None,
) -> str:
    """Replace template references in ``expr_s`` using a concrete assignment.

    Handles both explicit template keys (e.g. ``S[age]`` in template_map)
    and inline placeholder syntax like ``theta[age,pop]`` when the
    placeholders are covered by ``assignment``.

    When ``shaped_params`` and ``axis_lookup`` are provided, references to
    a registered shaped parameter (e.g. ``theta[imm]`` for a parameter
    declared shaped over ``("imm",)``) are rewritten as literal-index
    subscripts (``theta[5]``) into the shaped buffer instead of being
    expanded to per-coord scalar names.

    Returns:
        Expression string with template tokens replaced.
    """
    expr_out = expr_s
    shaped = shaped_params or {}

    # Pre-parse template keys once per ``template_map`` identity: we
    # filter out keys with no wildcards and keys whose base collides
    # with a shaped param, and precompile the literal regex used by
    # ``re.sub`` below. This collapses O(cells * |template_map|)
    # ``parse_selector`` calls to O(|template_map|) per normalize call
    # (issue #145).
    cache_entry = _APPLY_TPL_PREPARSE_CACHE.get(id(template_map))
    if cache_entry is not None and cache_entry[0] is template_map:
        prepared = cache_entry[1]
    else:
        prepared_list: list[
            tuple[str, str, tuple[WildcardToken, ...], re.Pattern[str]]
        ] = []
        for template_key in template_map:
            base, tokens = parse_selector(template_key)
            if base in shaped:
                continue
            wildcards = tuple(t for t in tokens if isinstance(t, WildcardToken))
            if not wildcards:
                continue
            prepared_list.append((
                template_key,
                base,
                wildcards,
                re.compile(re.escape(template_key)),
            ))
        prepared = tuple(prepared_list)
        _APPLY_TPL_PREPARSE_CACHE[id(template_map)] = (template_map, prepared)

    for template_key, base, wildcards, pat in prepared:
        if any(wt.axis not in assignment for wt in wildcards):
            continue
        if template_key not in expr_out:
            continue
        rendered = _render_template_name(
            base, [wt.axis for wt in wildcards], assignment
        )
        expr_out = pat.sub(rendered, expr_out)

    # Inline placeholder syntax without an explicit template map entry.
    def _inline_replacer(match: re.Match[str]) -> str:
        inner_base = match.group(1)
        inner = match.group(2)
        phs = [p.strip() for p in inner.split(",") if p.strip() and "=" not in p]
        if not phs or any(ph not in assignment for ph in phs):
            return match.group(0)
        if inner_base in shaped:
            registered = shaped[inner_base]
            if tuple(phs) != registered:
                # Different ordering or axes — leave for downstream to flag.
                return match.group(0)
            if axis_lookup is None:
                return match.group(0)
            try:
                idxs = [axis_lookup[ax].index(assignment[ax]) for ax in registered]
            except (KeyError, ValueError):
                return match.group(0)
            return f"{inner_base}[{', '.join(str(i) for i in idxs)}]"
        return _render_template_name(inner_base, phs, assignment)

    return _INLINE_TEMPLATE_RE.sub(_inline_replacer, expr_out)


# ---------------------------------------------------------------------------
# apply_to expansion (operators + coord_shift)
# ---------------------------------------------------------------------------


def expand_apply_to(
    apply_to_raw: object,
    *,
    axis_lookup: dict[str, list[str]],
    state_set: set[str] | None = None,
    context: str = "",
) -> list[str]:
    """Expand and validate an ``apply_to`` list, supporting selector syntax.

    Each entry may be:
    - A bare state name: ``"X"``  (passed through, for coord_shift prefix logic)
    - A full selector:  ``"X[age, vax, loc]"``  (expanded to concrete names)

    Args:
        apply_to_raw: Raw ``apply_to`` value from the spec.
        axis_lookup: Axis name → coordinate list for expansion.
        state_set: When provided, each expanded name is validated against it.
        context: Context string for error messages.

    Returns:
        Flat list of concrete (or bare) state name strings.

    Raises:
        InvalidRhsSpecError: If validation fails.
    """
    ctx = f" in {context}" if context else ""
    if not isinstance(apply_to_raw, (list, tuple)) or not apply_to_raw:
        raise InvalidRhsSpecError(
            detail=f"apply_to must be a non-empty list of state names{ctx}"
        )

    result: list[str] = []
    for j, entry in enumerate(apply_to_raw):
        if not isinstance(entry, str) or not entry.strip():
            raise InvalidRhsSpecError(
                detail=f"apply_to[{j}] must be a non-empty string{ctx}"
            )
        entry_s = entry.strip()
        expanded = expand_selector(
            entry_s, axis_lookup=axis_lookup, context=f"apply_to[{j}]{ctx}"
        )
        for name, _ in expanded:
            if state_set is not None and name not in state_set:
                raise InvalidRhsSpecError(
                    detail=f"apply_to entry {name!r} not in state{ctx}"
                )
            result.append(name)

    return result
