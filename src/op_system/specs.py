"""op_system.specs.

RHS specification models and normalization utilities for op_system.

Design goals
------------
- Domain-agnostic core: no imports from flepimop2 or other adapters.
- YAML-friendly RHS specifications that compile into a normalized representation
  consumable by op_engine (and other numerical backends).
- Minimal v1 implementation that demonstrates the idea without blocking future
  multiphysics extensions (IMEX operators, PDE terms, sources, etc.).

Current supported RHS kinds
---------------------------
1) kind: "expr"
   - User provides explicit expressions for d(state)/dt per state variable.

2) kind: "transitions"
   - User provides diagram-style transitions and per-capita hazard expressions.
   - Each transition contributes a flow:
        flow = hazard_expr * from_state
     and updates derivatives:
        d(from)/dt -= flow
        d(to)/dt   += flow

Future-facing (not implemented, but reserved)
---------------------------------------------
- kind: "multiphysics" or additional top-level keys such as:
    - sources: explicit additive per-state terms (births/imports/forcing)
    - operators: implicit operator specs/factories for IMEX (diffusion, transport)
    - couplings: structured couplings across axes (space/age/traits) and fields
  The normalization outputs include placeholders to carry these blocks forward,
  so adapters/backends can extend without changing the fundamental contract.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# -----------------------------------------------------------------------------
# Error message constants
# -----------------------------------------------------------------------------

_INVALID_RHS_SPEC_PREFIX = "Invalid op_system RHS specification."
_INVALID_EXPRESSION_PREFIX = "Invalid op_system expression."
_UNSUPPORTED_FEATURE_PREFIX = "Unsupported op_system feature."


def _raise_invalid_rhs_spec(
    *, missing: list[str] | None = None, detail: str | None = None
) -> NoReturn:
    """Raise a standardized RHS specification error.

    Args:
        missing: Optional list of missing field names.
        detail: Optional additional detail string.

    Raises:
        ValueError: Always.
    """
    parts: list[str] = [_INVALID_RHS_SPEC_PREFIX]
    if missing:
        parts.append(f"Missing required field(s): {sorted(set(missing))}.")
    if detail:
        parts.append(f"Detail: {detail}")
    raise ValueError(" ".join(parts))


def _raise_invalid_expression(*, detail: str) -> NoReturn:
    """Raise a standardized expression error.

    Args:
        detail: Error detail.

    Raises:
        ValueError: Always.
    """
    msg = f"{_INVALID_EXPRESSION_PREFIX} Detail: {detail}"
    raise ValueError(msg)


def _raise_unsupported_feature(*, feature: str, detail: str | None = None) -> NoReturn:
    """Raise a standardized unsupported feature error.

    Args:
        feature: Feature identifier.
        detail: Optional additional detail.

    Raises:
        NotImplementedError: Always.
    """
    msg = f"{_UNSUPPORTED_FEATURE_PREFIX} Feature '{feature}' is not supported."
    if detail:
        msg = f"{msg} Detail: {detail}"
    raise NotImplementedError(msg)


# -----------------------------------------------------------------------------
# Normalized RHS representation (backend-facing)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalizedRhs:
    """Normalized RHS representation suitable for compilation/execution."""

    kind: str
    state_names: tuple[str, ...]
    equations: tuple[str, ...]
    aliases: Mapping[str, str]
    param_names: tuple[str, ...]
    all_symbols: frozenset[str]
    meta: Mapping[str, Any]


# -----------------------------------------------------------------------------
# AST helpers (minimal v1)
# -----------------------------------------------------------------------------


def _parse_expr(expr: str) -> ast.AST:
    """
    Parse a Python expression and return the AST node.

    Args:
        expr: Expression string.

    Returns:
        Parsed AST for the expression.
    """
    try:
        return ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover
        _raise_invalid_expression(detail=f"invalid expression syntax: {exc.msg}")


def _collect_names(tree: ast.AST) -> set[str]:
    """
    Collect all Name identifiers used in an expression AST.

    Args:
        tree: Parsed AST.

    Returns:
        Set of identifier names referenced in the expression.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(str(node.id))
    return names


def _ensure_str_list(x: object, *, name: str) -> list[str]:
    """
    Ensure x is a list of non-empty strings.

    Args:
        x: Input value.
        name: Field name for error messages.

    Returns:
        List of stripped, non-empty strings.
    """
    if not isinstance(x, (list, tuple)):
        _raise_invalid_rhs_spec(detail=f"{name} must be a list of strings")
    out: list[str] = []
    for i, v in enumerate(x):
        if not isinstance(v, str) or not v.strip():
            _raise_invalid_rhs_spec(detail=f"{name}[{i}] must be a non-empty string")
        out.append(v.strip())
    return out


def _ensure_str_dict(x: object, *, name: str) -> dict[str, str]:
    """
    Ensure x is a dict of non-empty strings.

    Args:
        x: Input value (mapping) or None.
        name: Field name for error messages.

    Returns:
        Dict of stripped string keys to stripped non-empty string values.
    """
    if x is None:
        return {}
    if not isinstance(x, dict):
        _raise_invalid_rhs_spec(detail=f"{name} must be a mapping of string->string")
    out: dict[str, str] = {}
    for k, v in x.items():
        if not isinstance(k, str) or not k.strip():
            _raise_invalid_rhs_spec(detail=f"{name} keys must be non-empty strings")
        if not isinstance(v, str) or not v.strip():
            _raise_invalid_rhs_spec(detail=f"{name}[{k!r}] must be a non-empty string")
        out[k.strip()] = v.strip()
    return out


def _sorted_unique(xs: Iterable[str]) -> tuple[str, ...]:
    """Return a sorted tuple of unique strings from the iterable."""
    return tuple(sorted(set(xs)))


# -----------------------------------------------------------------------------
# Public normalization entrypoint
# -----------------------------------------------------------------------------


def normalize_rhs(spec: Mapping[str, Any] | None) -> NormalizedRhs:
    """
    Normalize a RHS specification dict into a backend-facing representation.

    Args:
        spec: Raw RHS specification mapping.

    Returns:
        Backend-facing normalized RHS representation.
    """
    if spec is None:
        _raise_invalid_rhs_spec(detail="rhs specification is required")

    kind = str(spec.get("kind", "expr")).strip().lower()

    if kind == "expr":  # lowest-level escape hatch
        return normalize_expr_rhs(spec)

    if kind == "transitions":  # diagram-style hazards
        return normalize_transitions_rhs(spec)

    _raise_unsupported_feature(
        feature=f"rhs.kind={kind}",
        detail="Only 'expr' and 'transitions' are supported in v1.",
    )


# -----------------------------------------------------------------------------
# expr kind
# -----------------------------------------------------------------------------


def normalize_expr_rhs(spec: Mapping[str, Any]) -> NormalizedRhs:
    """
    Normalize an expression-based RHS specification.

    Args:
        spec: Raw RHS specification mapping.

    Returns:
        Backend-facing normalized RHS representation.
    """
    state = _ensure_str_list(spec.get("state"), name="state")
    if len(state) != len(set(state)):
        _raise_invalid_rhs_spec(detail="state contains duplicate names")

    equations_map = spec.get("equations")
    if not isinstance(equations_map, dict):
        _raise_invalid_rhs_spec(detail="equations must be a mapping of state->expr")

    aliases = _ensure_str_dict(spec.get("aliases"), name="aliases")

    meta: dict[str, Any] = {}
    for reserved_key in ("sources", "operators", "couplings", "constraints"):
        if reserved_key in spec:
            meta[reserved_key] = spec.get(reserved_key)

    missing = [s for s in state if s not in equations_map]
    if missing:
        _raise_invalid_rhs_spec(missing=missing, detail="Missing equation(s) for state")

    state_set = set(state)
    unknown = [k for k in equations_map if k not in state_set]
    if unknown:
        _raise_invalid_rhs_spec(detail=f"unknown equation key(s): {sorted(unknown)}")

    eqs: list[str] = []
    all_syms: set[str] = set()

    for expr in aliases.values():
        tree = _parse_expr(expr)
        all_syms |= _collect_names(tree)

    for s in state:
        expr = equations_map[s]
        if not isinstance(expr, str) or not expr.strip():
            _raise_invalid_rhs_spec(
                detail=f"equations[{s!r}] must be a non-empty string"
            )
        expr_s = expr.strip()
        tree = _parse_expr(expr_s)
        all_syms |= _collect_names(tree)
        eqs.append(expr_s)

    excluded = state_set | set(aliases.keys())
    params = _sorted_unique(sym for sym in all_syms if sym not in excluded)

    return NormalizedRhs(
        kind="expr",
        state_names=tuple(state),
        equations=tuple(eqs),
        aliases=aliases,
        param_names=params,
        all_symbols=frozenset(all_syms | set(aliases.keys())),
        meta=meta,
    )


# -----------------------------------------------------------------------------
# transitions kind (hazard semantics)
# -----------------------------------------------------------------------------


def _validate_transition_mapping(tr: object, *, idx: int) -> Mapping[str, Any]:
    """
    Validate and return a transition mapping.

    Args:
        tr: Transition object.
        idx: Transition index (for error messages).

    Returns:
        Transition mapping.
    """
    if not isinstance(tr, dict):
        _raise_invalid_rhs_spec(detail=f"transitions[{idx}] must be a mapping")
    return tr


def _get_required_str(tr: Mapping[str, Any], *, idx: int, key: str) -> str:
    """
    Fetch a required string field from a transition mapping.

    Args:
        tr: Transition mapping.
        idx: Transition index (for error messages).
        key: Field key.

    Returns:
        Stripped string value.
    """
    val = tr.get(key)
    if not isinstance(val, str) or not val.strip():
        _raise_invalid_rhs_spec(detail=f"transitions[{idx}].{key} must be a string")
    return val.strip()


def _apply_transition(
    *,
    idx: int,
    tr: Mapping[str, Any],
    state_set: set[str],
    all_syms: set[str],
    d_terms: dict[str, list[str]],
) -> None:
    """Apply a transition to the derivative-term accumulator."""
    frm_s = _get_required_str(tr, idx=idx, key="from")
    to_s = _get_required_str(tr, idx=idx, key="to")
    rate_s = _get_required_str(tr, idx=idx, key="rate")

    if frm_s not in state_set:
        _raise_invalid_rhs_spec(
            detail=f"transitions[{idx}].from={frm_s!r} not in state"
        )
    if to_s not in state_set:
        _raise_invalid_rhs_spec(detail=f"transitions[{idx}].to={to_s!r} not in state")

    tree = _parse_expr(rate_s)
    all_syms |= _collect_names(tree)

    flow = f"({rate_s})*({frm_s})"
    d_terms[frm_s].append(f"-({flow})")
    d_terms[to_s].append(f"+({flow})")


def normalize_transitions_rhs(spec: Mapping[str, Any]) -> NormalizedRhs:
    """
    Normalize a transition-based RHS specification (diagram/hazard semantics).

    Args:
        spec: Raw RHS specification mapping.

    Returns:
        Backend-facing normalized RHS representation.
    """
    state = _ensure_str_list(spec.get("state"), name="state")
    if len(state) != len(set(state)):
        _raise_invalid_rhs_spec(detail="state contains duplicate names")

    transitions = spec.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        _raise_invalid_rhs_spec(detail="transitions must be a non-empty list")

    aliases = _ensure_str_dict(spec.get("aliases"), name="aliases")

    meta: dict[str, Any] = {"transitions": transitions}
    for reserved_key in ("sources", "operators", "couplings", "constraints"):
        if reserved_key in spec:
            meta[reserved_key] = spec.get(reserved_key)

    state_set = set(state)
    d_terms: dict[str, list[str]] = {s: [] for s in state}
    all_syms: set[str] = set()

    for expr in aliases.values():
        tree = _parse_expr(expr)
        all_syms |= _collect_names(tree)

    for idx, tr_obj in enumerate(transitions):
        tr = _validate_transition_mapping(tr_obj, idx=idx)
        _apply_transition(
            idx=idx,
            tr=tr,
            state_set=state_set,
            all_syms=all_syms,
            d_terms=d_terms,
        )

    equations: list[str] = []
    for s in state:
        terms = d_terms[s]
        if not terms:
            equations.append("0.0")
            continue
        expr = " ".join(terms)
        if expr.startswith("+") and expr[1:2] == "(":
            expr = expr[1:]
        equations.append(expr)

    excluded = state_set | set(aliases.keys())
    params = _sorted_unique(sym for sym in all_syms if sym not in excluded)

    return NormalizedRhs(
        kind="transitions",
        state_names=tuple(state),
        equations=tuple(equations),
        aliases=aliases,
        param_names=params,
        all_symbols=frozenset(all_syms | set(aliases.keys())),
        meta=meta,
    )
