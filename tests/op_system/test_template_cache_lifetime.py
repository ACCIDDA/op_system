"""Substitution plans belong to a batch and preserve their shaped-input context."""

import gc
import weakref
from collections import UserDict

from op_system._normalize_initial_state import _expand_initial_state_templates
from op_system._templates import _apply_template_substitutions


class _TemplateMap(UserDict[str, list[tuple[str, dict[str, str]]]]):
    """Give the mapping a weak reference without instrumenting IR node layouts."""


def test_standalone_substitution_does_not_retain_template_mapping() -> None:
    """Dropping a caller-owned mapping releases all its expanded coordinate cells."""
    mapping = _TemplateMap({"X[loc]": [("X__loc_a", {"loc": "a"})]})
    reference = weakref.ref(mapping)
    assert (
        _apply_template_substitutions(
            "X[loc]", assignment={"loc": "a"}, template_map=mapping
        )
        == "X__loc_a"
    )
    del mapping
    gc.collect()
    assert reference() is None


def test_initial_state_batch_releases_its_expanded_mapping() -> None:
    """Batch reuse does not extend the model mapping's lifetime past normalization."""
    mapping = _TemplateMap({
        "X[loc]": [("X__loc_a", {"loc": "a"}), ("X__loc_b", {"loc": "b"})]
    })
    reference = weakref.ref(mapping)
    values = _expand_initial_state_templates(
        {"X[loc]": "gain * X[loc]"},
        axes=[{"name": "loc", "coords": ["a", "b"]}],
        template_map=mapping,
    )
    assert values == {"X__loc_a": "gain * X__loc_a", "X__loc_b": "gain * X__loc_b"}
    del mapping
    gc.collect()
    assert reference() is None


def test_substitution_reuse_respects_shaped_parameter_exclusions() -> None:
    """The same mapping can be used under distinct shaped-input declarations."""
    mapping = {"X[loc]": [("X__loc_a", {"loc": "a"})]}
    ordinary = _apply_template_substitutions(
        "X[loc]", assignment={"loc": "a"}, template_map=mapping
    )
    shaped = _apply_template_substitutions(
        "X[loc]",
        assignment={"loc": "a"},
        template_map=mapping,
        shaped_params={"X": ("loc",)},
        axis_lookup={"loc": ["a"]},
    )
    assert ordinary == "X__loc_a"
    assert shaped == "X[0]"
