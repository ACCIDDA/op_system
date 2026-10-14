# Numerical qualification of compiled equations

`validate_spec` checks structure, normalization, and compilation. Call
`qualify_rhs` separately to compare the numerical RHS against equations you
write independently. This is an opt-in, eager authoring tool: it evaluates
named fixtures and returns inspectable host diagnostics. It does not add work
to structural validation or to normal RHS evaluation.

## A first report

```python
import json
import numpy as np
from op_system import QualificationFixture, compile_spec, qualify_rhs

compiled = compile_spec({
    "kind": "expr",
    "axes": [{"name": "group", "coords": ["first", "second"]}],
    "factorize_axes": ["group"],
    "state": ["X[group]", "Y[group]"],
    "equations": {
        "X[group]": "-shared * local[group] * X[group]",
        "Y[group]": "shared * local[group] * X[group]",
    },
})


def reference(t, state, params):
    # A separate cell-wise calculation, not generated code or compiler IR.
    flux = np.empty_like(np.asarray(state["X"]))
    for group, x in enumerate(np.asarray(state["X"])):
        flux[group] = float(params["shared"]) * float(params["local"][group]) * x
    return {"X": -flux, "Y": flux}


fixtures = [
    QualificationFixture(
        "positive-closed",
        {"X": np.array([2.0, 3.0]), "Y": np.array([1.0, 0.5])},
        {"shared": 0.5, "local": np.array([0.2, 0.4])},
    )
]
report = qualify_rhs(
    compiled, fixtures, reference, evaluators=("flat", "structured", "block")
)
assert report.ok
print(json.dumps(report.to_dict(), indent=2, allow_nan=False))
```

Each fixture/layout record includes its name, time, array namespace, reference
tolerances, shapes and dtypes, finite-value checks, and absolute/relative
agreement errors. Failed numerical checks identify the state or callback.
Exceptions raised by a user reference are reported against that fixture;
other fixtures are still evaluated. JSON diagnostics use `null` for undefined
or non-finite error metrics rather than invalid JSON numbers.

Fixtures use structured mappings keyed by compiled state templates, even for
flat evaluation. Flat packing follows compiler metadata rather than fixture
dictionary order. Scalar-only models can request `evaluators=("flat",)` when
no structured evaluator is available.

## Conditional balance checks

Supply residual-to-zero callbacks through `InvariantCheck`. A conservation
claim requires its conditions: the closed model above conserves each group's
`X + Y`; external births, deaths, forcing, or sinks could invalidate that
claim. The condition is recorded with the check and its absolute tolerance.

```python
from op_system import InvariantCheck


def closed_balance(t, state, derivative, params):
    return derivative["X"] + derivative["Y"]


mass = InvariantCheck(
    "group total",
    closed_balance,
    "Closed X-to-Y transfer with no external sources",
    atol=1e-10,
)
balanced = qualify_rhs(compiled, fixtures, reference, invariants=(mass,))
assert balanced.ok
```

The utility checks the supplied residual; it cannot determine whether the
condition accurately describes your model. An incorrect equation or violated
balance produces a failed report. A coupled model that lacks a block evaluator
produces an explicit failed block record, with no fallback to independent
groups. Request only the evaluators applicable to that model.

## JAX state directions

Set a fixture's `direction` to a structured JAX state tangent to request an
additional JAX JVP check against central differences of the independent
reference. The state and tangent need matching keys, shapes, and compatible
floating dtypes. The reference is called on eager perturbed states and may
cross to NumPy. Qualification itself is not a function to put inside `jit`.

`QualificationConfig` controls value and derivative tolerances and the
finite-difference step. Defaults assume double precision. For float32, choose
larger tolerances and a larger step appropriate to the state/direction scales;
roundoff and cancellation can spoil a derivative comparison. A value-correct
reference with a wrong derivative is caught by the directional check.

The derivative helper is qualified for JAX state directions. A direction
requested on NumPy or Torch produces an explicit unqualified check; this
utility does not promise identical AD transforms across every backend. See the
[backend capability matrix](backend-qualification.md) for compiler contracts.

Agreement at selected fixtures is not a proof across all parameter regimes,
scientific validity, or solution accuracy of an integrator. Include varied
fixtures, boundary cases, meaningful balance conditions, and independent
domain-specific evidence for the conclusions you need.
