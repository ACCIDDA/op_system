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


@pytest.mark.parametrize(
    ("axes", "expected"),
    [
        pytest.param(
            _AXES,
            {"age": ["a0", "a1"], "vax": ["u", "v"], "imm": ["X0", "X1"]},
            id="categorical-multi-axis",
        ),
        pytest.param(
            [{"name": "x", "domain": {"lb": 0, "ub": 1}}],
            {"x": []},
            id="continuous-no-coords",
        ),
    ],
)
def test_build_axis_lookup(
    axes: list[dict[str, object]], expected: dict[str, list[str]]
) -> None:
    """build_axis_lookup returns the expected axis→coords mapping."""
    assert build_axis_lookup(axes) == expected


# ---------------------------------------------------------------------------
# parse_selector — valid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selector", "expected_base", "expected_tokens"),
    [
        pytest.param("S", "S", [], id="bare-name"),
        pytest.param("  S  ", "S", [], id="bare-name-stripped"),
        pytest.param("S[age]", "S", [WildcardToken("age")], id="wildcard-single"),
        pytest.param(
            "S[age, vax]",
            "S",
            [WildcardToken("age"), WildcardToken("vax")],
            id="wildcard-multi",
        ),
        pytest.param("X[imm=X0]", "X", [PinnedToken("imm", "X0")], id="pinned-single"),
        pytest.param(
            "X[age=a0, imm=X1]",
            "X",
            [PinnedToken("age", "a0"), PinnedToken("imm", "X1")],
            id="pinned-multi",
        ),
        pytest.param(
            "X[age, vax, imm=X0]",
            "X",
            [
                WildcardToken("age"),
                WildcardToken("vax"),
                PinnedToken("imm", "X0"),
            ],
            id="mixed-wildcard-pinned",
        ),
        pytest.param(
            "X[ age , imm = X0 ]",
            "X",
            [WildcardToken("age"), PinnedToken("imm", "X0")],
            id="whitespace-inside-brackets",
        ),
    ],
)
def test_parse_selector_valid(
    selector: str,
    expected_base: str,
    expected_tokens: list[WildcardToken | PinnedToken],
) -> None:
    """Valid selectors parse to the expected (base, tokens) pair."""
    base, tokens = parse_selector(selector)
    assert base == expected_base
    assert tokens == expected_tokens


# ---------------------------------------------------------------------------
# parse_selector — invalid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selector", "match"),
    [
        pytest.param("X[age, age]", "duplicate axis", id="duplicate-wildcard"),
        pytest.param("X[imm=X0, imm=X1]", "duplicate axis", id="duplicate-pinned"),
        pytest.param(
            "X[age, age=a0]", "duplicate axis", id="wildcard-and-pinned-same-axis"
        ),
        pytest.param("X[age=]", "invalid pinned selector", id="empty-pinned-coord"),
        pytest.param("X[=X0]", "invalid pinned selector", id="empty-pinned-axis"),
    ],
)
def test_parse_selector_invalid(selector: str, match: str) -> None:
    """Invalid selectors raise ValueError with the expected message fragment."""
    with pytest.raises(ValueError, match=match):
        parse_selector(selector)


# ---------------------------------------------------------------------------
# render_selector — valid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "tokens", "assignment", "expected"),
    [
        pytest.param("S", [], {}, "S", id="bare-name"),
        pytest.param(
            "S",
            [WildcardToken("age")],
            {"age": "a0"},
            "S__age_a0",
            id="wildcard-single",
        ),
        pytest.param(
            "S",
            [WildcardToken("age"), WildcardToken("vax")],
            {"age": "a1", "vax": "v"},
            "S__age_a1__vax_v",
            id="wildcard-multi",
        ),
        pytest.param(
            "X", [PinnedToken("imm", "X0")], {}, "X__imm_X0", id="pinned-only"
        ),
        pytest.param(
            "X",
            [
                WildcardToken("age"),
                WildcardToken("vax"),
                PinnedToken("imm", "X0"),
            ],
            {"age": "a1", "vax": "v"},
            "X__age_a1__vax_v__imm_X0",
            id="mixed",
        ),
    ],
)
def test_render_selector_valid(
    base: str,
    tokens: list[WildcardToken | PinnedToken],
    assignment: dict[str, str],
    expected: str,
) -> None:
    """render_selector emits the expected concrete name."""
    assert render_selector(base, tokens, assignment) == expected


# ---------------------------------------------------------------------------
# render_selector — validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "tokens", "assignment", "match"),
    [
        pytest.param(
            "S",
            [WildcardToken("bad")],
            {"bad": "x"},
            "unknown axis",
            id="unknown-wildcard-axis",
        ),
        pytest.param(
            "X",
            [PinnedToken("imm", "X99")],
            {},
            "pinned coord",
            id="unknown-pinned-coord",
        ),
    ],
)
def test_render_selector_validation_errors(
    base: str,
    tokens: list[WildcardToken | PinnedToken],
    assignment: dict[str, str],
    match: str,
) -> None:
    """render_selector validates against axis_lookup when provided."""
    lookup = build_axis_lookup(_AXES)
    with pytest.raises(ValueError, match=match):
        render_selector(base, tokens, assignment, axis_lookup=lookup)


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


@pytest.mark.parametrize(
    ("selector", "expected_names"),
    [
        pytest.param("S[age]", ["S__age_a0", "S__age_a1"], id="wildcard-single"),
        pytest.param(
            "S[age, vax]",
            [
                "S__age_a0__vax_u",
                "S__age_a0__vax_v",
                "S__age_a1__vax_u",
                "S__age_a1__vax_v",
            ],
            id="wildcard-multi-cartesian",
        ),
        pytest.param("X[imm=X0]", ["X__imm_X0"], id="pinned-only-single"),
        pytest.param(
            "X[age=a0, imm=X1]",
            ["X__age_a0__imm_X1"],
            id="pinned-multi-single",
        ),
    ],
)
def test_expand_selector_names(selector: str, expected_names: list[str]) -> None:
    """expand_selector yields the expected concrete names."""
    lookup = build_axis_lookup(_AXES)
    results = expand_selector(selector, axis_lookup=lookup)
    assert {r[0] for r in results} == set(expected_names)
    assert len(results) == len(expected_names)


def test_expand_selector_mixed_expands_wildcards_only() -> None:
    """Mixed selector expands over wildcard axes; pinned axis is held fixed."""
    lookup = build_axis_lookup(_AXES)
    results = expand_selector("X[age, vax, imm=X0]", axis_lookup=lookup)
    assert len(results) == 4  # 2 age x 2 vax; imm pinned at X0
    names = {r[0] for r in results}
    assert all("imm_X0" in n for n in names)
    assert "X__age_a0__vax_u__imm_X0" in names
    assert "X__age_a1__vax_v__imm_X0" in names


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


@pytest.mark.parametrize(
    ("selector", "match"),
    [
        pytest.param("S[bad]", "not defined", id="unknown-wildcard-axis"),
        pytest.param("X[bad=v]", "not defined", id="unknown-pinned-axis"),
        pytest.param("X[imm=X99]", "pinned coord", id="unknown-pinned-coord"),
    ],
)
def test_expand_selector_invalid(selector: str, match: str) -> None:
    """Invalid selectors raise ValueError with the expected message fragment."""
    lookup = build_axis_lookup(_AXES)
    with pytest.raises(ValueError, match=match):
        expand_selector(selector, axis_lookup=lookup)


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


def test_expand_apply_to_without_state_set_skips_membership_check() -> None:
    """When state_set is None, expansion proceeds without membership validation."""
    lookup = build_axis_lookup([{"name": "vax", "coords": ["u", "v"]}])
    result = expand_apply_to(["S[vax]"], axis_lookup=lookup)
    assert set(result) == {"S__vax_u", "S__vax_v"}


@pytest.mark.parametrize(
    ("apply_to", "state_set", "match"),
    [
        pytest.param(["X"], {"S", "R"}, "not in state", id="expanded-name-missing"),
        pytest.param([], None, "non-empty list", id="empty-list"),
        pytest.param([123], None, "non-empty string", id="non-string-entry"),
    ],
)
def test_expand_apply_to_invalid(
    apply_to: list[object],
    state_set: set[str] | None,
    match: str,
) -> None:
    """Invalid apply_to inputs raise ValueError with the expected message fragment."""
    lookup = build_axis_lookup(_AXES)
    with pytest.raises(ValueError, match=match):
        expand_apply_to(apply_to, axis_lookup=lookup, state_set=state_set)
