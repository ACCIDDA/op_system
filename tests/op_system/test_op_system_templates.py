"""Unit tests for op_system._templates (pytest).

Covers the centralized selector parser/renderer/expander introduced in #82:

- ``build_axis_lookup``: axis → coords dict construction
- ``parse_selector``: wildcard, pinned, mixed, bare, and invalid selectors
- ``render_selector``: concrete name rendering with validation
- ``expand_selector``: cartesian expansion with validation
- ``expand_apply_to``: apply_to list expansion and state validation
"""

from __future__ import annotations

import pytest

from op_system._templates import (
    PinnedToken,
    WildcardToken,
    build_axis_lookup,
    expand_apply_to,
    expand_selector,
    parse_selector,
    render_selector,
)

# Shared axis definitions used across multiple tests.
_AXES = [
    {"name": "age", "coords": ["a0", "a1"]},
    {"name": "vax", "coords": ["u", "v"]},
    {"name": "imm", "coords": ["X0", "X1"]},
]


# ---------------------------------------------------------------------------
# build_axis_lookup
# ---------------------------------------------------------------------------


def test_build_axis_lookup_basic() -> None:
    """build_axis_lookup returns correct axis→coords mapping."""
    lookup = build_axis_lookup(_AXES)
    assert lookup == {
        "age": ["a0", "a1"],
        "vax": ["u", "v"],
        "imm": ["X0", "X1"],
    }


def test_build_axis_lookup_continuous_axis_returns_empty_coords() -> None:
    """Continuous axes without explicit coords map to empty lists."""
    axes = [{"name": "x", "domain": {"lb": 0, "ub": 1}}]
    assert build_axis_lookup(axes) == {"x": []}


# ---------------------------------------------------------------------------
# parse_selector — valid inputs
# ---------------------------------------------------------------------------


def test_parse_selector_bare_name() -> None:
    """Bare name (no brackets) returns (name, [])."""
    base, tokens = parse_selector("S")
    assert base == "S"
    assert tokens == []


def test_parse_selector_bare_name_strips_whitespace() -> None:
    """Leading/trailing whitespace around a bare name is stripped."""
    base, tokens = parse_selector("  S  ")
    assert base == "S"
    assert tokens == []


def test_parse_selector_wildcard_single() -> None:
    """Single wildcard token produces one WildcardToken."""
    base, tokens = parse_selector("S[age]")
    assert base == "S"
    assert tokens == [WildcardToken("age")]


def test_parse_selector_wildcard_multi() -> None:
    """Multiple wildcard tokens are returned in declaration order."""
    base, tokens = parse_selector("S[age, vax]")
    assert base == "S"
    assert tokens == [WildcardToken("age"), WildcardToken("vax")]


def test_parse_selector_pinned_only() -> None:
    """Single pinned token produces one PinnedToken."""
    base, tokens = parse_selector("X[imm=X0]")
    assert base == "X"
    assert tokens == [PinnedToken("imm", "X0")]


def test_parse_selector_pinned_multi() -> None:
    """Multiple pinned tokens parse correctly."""
    base, tokens = parse_selector("X[age=a0, imm=X1]")
    assert base == "X"
    assert tokens == [PinnedToken("age", "a0"), PinnedToken("imm", "X1")]


def test_parse_selector_mixed() -> None:
    """Mixed wildcard and pinned tokens parse in declaration order."""
    base, tokens = parse_selector("X[age, vax, imm=X0]")
    assert base == "X"
    assert tokens == [
        WildcardToken("age"),
        WildcardToken("vax"),
        PinnedToken("imm", "X0"),
    ]


def test_parse_selector_whitespace_inside_brackets() -> None:
    """Whitespace around token names and around ``=`` is stripped."""
    base, tokens = parse_selector("X[ age , imm = X0 ]")
    assert base == "X"
    assert tokens == [WildcardToken("age"), PinnedToken("imm", "X0")]


# ---------------------------------------------------------------------------
# parse_selector — invalid inputs
# ---------------------------------------------------------------------------


def test_parse_selector_duplicate_wildcard_axis_raises() -> None:
    """Duplicate wildcard axis names in the same selector raise ValueError."""
    with pytest.raises(ValueError, match="duplicate axis"):
        parse_selector("X[age, age]")


def test_parse_selector_duplicate_pinned_axis_raises() -> None:
    """Duplicate pinned axis names (different coords) raise ValueError."""
    with pytest.raises(ValueError, match="duplicate axis"):
        parse_selector("X[imm=X0, imm=X1]")


def test_parse_selector_wildcard_and_pinned_same_axis_raises() -> None:
    """Combining wildcard and pinned token for the same axis raises ValueError."""
    with pytest.raises(ValueError, match="duplicate axis"):
        parse_selector("X[age, age=a0]")


def test_parse_selector_empty_pinned_coord_raises() -> None:
    """Pinned token with empty coord string raises ValueError."""
    with pytest.raises(ValueError, match="invalid pinned selector"):
        parse_selector("X[age=]")


def test_parse_selector_empty_pinned_axis_raises() -> None:
    """Pinned token with empty axis string raises ValueError."""
    with pytest.raises(ValueError, match="invalid pinned selector"):
        parse_selector("X[=X0]")


# ---------------------------------------------------------------------------
# render_selector — valid inputs
# ---------------------------------------------------------------------------


def test_render_selector_bare_name() -> None:
    """Bare base with no tokens renders to just the base."""
    assert render_selector("S", [], {}) == "S"


def test_render_selector_wildcard_single() -> None:
    """Single wildcard token renders to ``base__axis_coord``."""
    tokens = [WildcardToken("age")]
    assert render_selector("S", tokens, {"age": "a0"}) == "S__age_a0"


def test_render_selector_wildcard_multi() -> None:
    """Multiple wildcard tokens each contribute an ``axis_coord`` segment."""
    tokens = [WildcardToken("age"), WildcardToken("vax")]
    assert render_selector("S", tokens, {"age": "a1", "vax": "v"}) == "S__age_a1__vax_v"


def test_render_selector_pinned_only() -> None:
    """Pinned token renders with its fixed coord value embedded."""
    tokens = [PinnedToken("imm", "X0")]
    assert render_selector("X", tokens, {}) == "X__imm_X0"


def test_render_selector_mixed() -> None:
    """Mixed tokens: wildcard draws from assignment; pinned embeds its coord."""
    tokens: list[WildcardToken | PinnedToken] = [
        WildcardToken("age"),
        WildcardToken("vax"),
        PinnedToken("imm", "X0"),
    ]
    assert (
        render_selector("X", tokens, {"age": "a1", "vax": "v"})
        == "X__age_a1__vax_v__imm_X0"
    )


# ---------------------------------------------------------------------------
# render_selector — validation
# ---------------------------------------------------------------------------


def test_render_selector_validates_unknown_wildcard_axis() -> None:
    """Unknown wildcard axis with axis_lookup provided raises ValueError."""
    lookup = build_axis_lookup(_AXES)
    tokens = [WildcardToken("bad")]
    with pytest.raises(ValueError, match="unknown axis"):
        render_selector("S", tokens, {"bad": "x"}, axis_lookup=lookup)


def test_render_selector_validates_unknown_pinned_coord() -> None:
    """Unknown pinned coord with axis_lookup provided raises ValueError."""
    lookup = build_axis_lookup(_AXES)
    tokens = [PinnedToken("imm", "X99")]
    with pytest.raises(ValueError, match="pinned coord"):
        render_selector("X", tokens, {}, axis_lookup=lookup)


def test_render_selector_partial_assignment_returns_bracketed_form() -> None:
    """Wildcard axis absent from assignment returns bracketed (unexpanded) form."""
    tokens = [WildcardToken("age"), WildcardToken("vax")]
    # vax missing from assignment — partial render falls back to bracketed form.
    result = render_selector("S", tokens, {"age": "a0"})
    assert "[" in result


# ---------------------------------------------------------------------------
# expand_selector — valid inputs
# ---------------------------------------------------------------------------


def test_expand_selector_bare_name() -> None:
    """Bare name expands to a single ``(name, {})`` pair."""
    lookup = build_axis_lookup(_AXES)
    assert expand_selector("S", axis_lookup=lookup) == [("S", {})]


def test_expand_selector_wildcard_single() -> None:
    """Single wildcard expands over all coords of that axis in order."""
    lookup = build_axis_lookup(_AXES)
    results = expand_selector("S[age]", axis_lookup=lookup)
    assert [r[0] for r in results] == ["S__age_a0", "S__age_a1"]


def test_expand_selector_wildcard_multi_is_cartesian_product() -> None:
    """Multiple wildcards expand as the cartesian product of all coord lists."""
    lookup = build_axis_lookup(_AXES)
    results = expand_selector("S[age, vax]", axis_lookup=lookup)
    assert len(results) == 4  # 2 age x 2 vax
    names = {r[0] for r in results}
    assert names == {
        "S__age_a0__vax_u",
        "S__age_a0__vax_v",
        "S__age_a1__vax_u",
        "S__age_a1__vax_v",
    }


def test_expand_selector_pinned_only_single_expansion() -> None:
    """All-pinned selector yields exactly one expansion with the pinned name."""
    lookup = build_axis_lookup(_AXES)
    results = expand_selector("X[imm=X0]", axis_lookup=lookup)
    assert len(results) == 1
    name, assignment = results[0]
    assert name == "X__imm_X0"
    assert assignment["imm"] == "X0"


def test_expand_selector_mixed_expands_wildcards_only() -> None:
    """Mixed selector expands over wildcard axes; pinned axis is held fixed."""
    lookup = build_axis_lookup(_AXES)
    results = expand_selector("X[age, vax, imm=X0]", axis_lookup=lookup)
    assert len(results) == 4  # 2 age x 2 vax; imm pinned at X0
    names = {r[0] for r in results}
    assert all("imm_X0" in n for n in names)
    assert "X__age_a0__vax_u__imm_X0" in names
    assert "X__age_a1__vax_v__imm_X0" in names


def test_expand_selector_pinned_multi_all_axes_in_name() -> None:
    """Multi-pinned selector produces a single name with all pinned segments."""
    lookup = build_axis_lookup(_AXES)
    results = expand_selector("X[age=a0, imm=X1]", axis_lookup=lookup)
    assert len(results) == 1
    assert results[0][0] == "X__age_a0__imm_X1"


def test_expand_selector_mixed_assignment_includes_pinned_axes() -> None:
    """Assignment dicts from a mixed expansion include both wildcard and pinned keys."""
    lookup = build_axis_lookup(_AXES)
    results = expand_selector("X[age, imm=X1]", axis_lookup=lookup)
    for _, assignment in results:
        assert assignment.get("imm") == "X1"
        assert "age" in assignment


# ---------------------------------------------------------------------------
# expand_selector — invalid inputs
# ---------------------------------------------------------------------------


def test_expand_selector_unknown_wildcard_axis_raises() -> None:
    """Unknown wildcard axis raises ValueError."""
    lookup = build_axis_lookup(_AXES)
    with pytest.raises(ValueError, match="not defined"):
        expand_selector("S[bad]", axis_lookup=lookup)


def test_expand_selector_unknown_pinned_axis_raises() -> None:
    """Unknown pinned axis raises ValueError."""
    lookup = build_axis_lookup(_AXES)
    with pytest.raises(ValueError, match="not defined"):
        expand_selector("X[bad=v]", axis_lookup=lookup)


def test_expand_selector_unknown_pinned_coord_raises() -> None:
    """Unknown coord for a known pinned axis raises ValueError."""
    lookup = build_axis_lookup(_AXES)
    with pytest.raises(ValueError, match="pinned coord"):
        expand_selector("X[imm=X99]", axis_lookup=lookup)


# ---------------------------------------------------------------------------
# expand_apply_to
# ---------------------------------------------------------------------------


def test_expand_apply_to_bare_entries_pass_through() -> None:
    """Bare state names pass through unchanged when they are in state_set."""
    lookup = build_axis_lookup(_AXES)
    result = expand_apply_to(["S", "R"], axis_lookup=lookup, state_set={"S", "R"})
    assert result == ["S", "R"]


def test_expand_apply_to_selector_expands_to_concrete_names() -> None:
    """Selector entries expand to all matching concrete state names."""
    lookup = build_axis_lookup([{"name": "vax", "coords": ["u", "v"]}])
    state_set = {"S__vax_u", "S__vax_v"}
    result = expand_apply_to(["S[vax]"], axis_lookup=lookup, state_set=state_set)
    assert set(result) == {"S__vax_u", "S__vax_v"}


def test_expand_apply_to_expanded_name_not_in_state_set_raises() -> None:
    """Expanded name absent from state_set raises ValueError."""
    lookup = build_axis_lookup(_AXES)
    with pytest.raises(ValueError, match="not in state"):
        expand_apply_to(["X"], axis_lookup=lookup, state_set={"S", "R"})


def test_expand_apply_to_empty_list_raises() -> None:
    """Empty apply_to list raises ValueError."""
    lookup = build_axis_lookup(_AXES)
    with pytest.raises(ValueError, match="non-empty list"):
        expand_apply_to([], axis_lookup=lookup)


def test_expand_apply_to_non_string_entry_raises() -> None:
    """Non-string apply_to entry raises ValueError."""
    lookup = build_axis_lookup(_AXES)
    with pytest.raises(ValueError, match="non-empty string"):
        expand_apply_to([123], axis_lookup=lookup)


def test_expand_apply_to_without_state_set_skips_membership_check() -> None:
    """When state_set is None, expansion proceeds without membership validation."""
    lookup = build_axis_lookup([{"name": "vax", "coords": ["u", "v"]}])
    result = expand_apply_to(["S[vax]"], axis_lookup=lookup)
    assert set(result) == {"S__vax_u", "S__vax_v"}
