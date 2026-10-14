"""Reject unbound state-template symbols before constructing any evaluator."""

from typing import Any

import numpy as np
import pytest

from op_system import compile_spec, validate_spec
from op_system._errors import InvalidRhsSpecError


def _spec(kind: str, expression: str, *, alias: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": kind,
        "axes": [{"name": "age", "coords": ["a0", "a1"]}],
        "state": ["S[age]", "I[age]", "Z"],
    }
    if alias:
        result["aliases"] = {"force[age]": expression}
        expression = "force[age]"
    if kind == "expr":
        result["equations"] = {
            "S[age]": f"-{expression} * S[age]",
            "I[age]": f"{expression} * S[age]",
            "Z": "0 * Z",
        }
    else:
        result["transitions"] = [
            {"name": "flow", "from": "S[age]", "to": "I[age]", "rate": expression}
        ]
    return result


@pytest.mark.parametrize("kind", ["expr", "transitions"])
@pytest.mark.parametrize("alias", [False, True])
def test_bare_template_state_is_rejected_with_user_context(
    kind: str, *, alias: bool
) -> None:
    """A free I cannot be silently treated as neither a state nor a parameter."""
    spec = _spec(kind, "beta * I", alias=alias)
    with pytest.raises(
        InvalidRhsSpecError,
        match=r"templated state 'I' without its axes; write I\[age\]",
    ) as failure:
        compile_spec(spec)
    assert (
        "aliases" if alias else "equations" if kind == "expr" else "transitions[0].rate"
    ) in str(failure.value)
    report = validate_spec(spec)
    assert not report.ok
    assert report.stages["normalize"] == "failed"
    assert "templated state 'I'" in " ".join(report.errors)


@pytest.mark.parametrize("kind", ["expr", "transitions"])
def test_indexed_templates_and_scalar_names_remain_valid(kind: str) -> None:
    """Explicit indices and ordinary scalar states retain their mathematical meaning."""
    compiled = compile_spec(_spec(kind, "beta * I[age] + Z", alias=True))
    state = {
        "S": np.array([9.0, 8.0]),
        "I": np.array([1.0, 2.0]),
        "Z": np.asarray(0.25),
    }
    assert compiled.pytree_eval_fn is not None
    result = compiled.pytree_eval_fn(0.0, state, beta=0.5)
    flow = (0.5 * state["I"] + state["Z"]) * state["S"]
    np.testing.assert_allclose(result["S"], -flow)
    np.testing.assert_allclose(result["I"], flow)
    assert compiled.param_names == ("beta",)


def test_alias_chain_cannot_hide_a_bare_template_state() -> None:
    """Validate the original offending alias even when another alias references it."""
    spec = _spec("transitions", "force[age]")
    spec["aliases"] = {"bad[age]": "beta * I", "force[age]": "bad[age]"}
    with pytest.raises(InvalidRhsSpecError, match=r"aliases\['bad\[age\]'\]"):
        compile_spec(spec)


def test_reduction_binding_does_not_look_like_a_bare_state() -> None:
    """Bound selectors can use the state base as their reduction-variable name."""
    compiled = compile_spec(_spec("transitions", "sum_over(I[age:I], age=I)"))
    assert compiled.param_names == ()
