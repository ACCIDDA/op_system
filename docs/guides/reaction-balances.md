# Conditional balances from scalar reactions

`reaction_balances(compiled)` derives exact integer coordinate-sum identities
from named scalar unit-transfer reactions. It does not evaluate propensities or
change the compiled equations. The report distinguishes represented closed
transfers, explicit external sources, unsupported contributions and scientific
applicability conditions.

```python
import numpy as np
from op_system import compile_spec, reaction_balances

compiled = compile_spec({
    "kind": "transitions",
    "state": ["a", "b"],
    "transitions": [
        {"name": "transfer", "from": "a", "to": "b", "rate": "k"},
        {"name": "inflow", "to": "a", "rate": "q"},
    ],
})
report = reaction_balances(compiled)
law = report.laws[0]
assert report.fully_covered
assert law.coefficients == (1, 1)
assert not law.closed
assert law.sources[0].reaction_name == "inflow"
assert law.sources[0].coefficient == 1

derivative = compiled.eval_fn(0.0, np.asarray([2.0, 3.0]), k=0.2, q=0.5)
np.testing.assert_allclose(np.dot(law.coefficients, derivative), 0.5)
# The identity is d(a + b)/dt = q, not a closed conservation law.
```

Removing the `inflow` transition gives a closed coordinate total. Changing its
rate, its time dependence or the state dependence of the internal transfer
does not change the incidence identity. An isolated state receives its own
constant-coordinate law when it has no represented source or transfer.

| Supplied model | `fully_covered` | Issued law |
| --- | --- | --- |
| Complete scalar unit transfers | `True` | Closed component totals |
| Those transfers plus source-only reactions | `True` | Explicit open balances for affected components |
| Omitted, unsupported or inconsistent dynamics | `False` | No laws; inspect `coverage_gaps` |

## Exact derivation and recorded meaning

An internal column has `-1` at its consumed state and `+1` at its target.
The condition `c.T @ S_internal = 0` therefore requires equal coefficients
on states joined by a transfer. Indicators of the undirected connected
components form an exact integer basis. The implementation traverses this
graph without adding a symbolic/SVD dependency or selecting a numerical rank
tolerance. This conclusion depends on the unit-transfer contract; it does not
generalize to arbitrary molecular stoichiometry or all nonlinear first
integrals.

Every coefficient vector follows `report.state_names`. Component names follow
compiled first-state order and are stable within that report; they are not
persistent identities across revised models. `reaction_names` records every
covered channel. Each `ReactionBalanceSource` names a source-only channel,
its target and its integer propensity multiplier. Resolve its numerical flux
using the matching compiled reaction/parameters. The records contain no
callbacks or parameter values and can be inspected, hashed and pickled.

`origin="compiled_unit_transfers"` records the derivation method. It differs
from a user-authored `InvariantCheck` and from a numerical residual check.
The report is a conditional structural artifact, without authenticated
provenance or a claim about an arbitrary replacement runtime callback.

## Scientific metadata and coverage

`state_meta` and `axis_meta` are detached tuples of immutable declaration
records. Missing units or representation labels remain unknown. Each law and
report includes applicability assumptions: summed coordinates need compatible
units/representations, and execution must faithfully realize these channels
without additional state updates, fluxes, clipping, projections, resets or
impulses. Unit labels are compared for known component conflicts; no
dimensional analysis or unit conversion is performed. Disconnected components
can carry different unit labels.

A numeric coordinate sum is not automatically a physical mass or spatial
integral. Intensive quantities need their physical measures before making that
interpretation; no mesh weights are inferred here. Use
[state representations](state-representations.md) and
[numerical qualification](numerical-qualification.md) for declared measures and
independent runtime checks.

This first analysis requires scalar templates and complete, uniquely named
reaction artifacts matching normalized transition identities. Original
`ReactionGap` records retain their origin and selectors inside
`BalanceCoverageGap`. Templated/routed/shifted/spatial layouts, operators,
unknown or duplicate identities, missing artifacts, known conflicting
scientific labels, and declared `sources`/`couplings`/`constraints` blocks
(including empty reserved blocks) prevent all issued laws. Unsupported
contributions are never assigned zero flux. Reaction reactant/dependency
completeness is a separate contract: it constrains some stochastic methods,
while this analysis uses the represented net unit-transfer columns.

Weighted spatial balances, boundary/operator fluxes and more general
stoichiometric derivations remain extensions of
[issue #295](https://github.com/ACCIDDA/op_system/issues/295).
The analysis leaves normalized/compiled storage and numerical evaluation
unchanged, and adds no engine, framework or statistical dependency.
