# Renewal births into a pinned age bin

A source-only transition (`from: null`, or omitted `from`) adds its rate to
the destination without depleting a donor. Its rate is a total flux, rather
than a per-capita coefficient multiplied by the destination population.
It may reduce states and age-indexed parameters over an axis that is pinned
on the destination.

```python
import numpy as np

from op_system import compile_spec

spec = {
    "kind": "transitions",
    "axes": [{"name": "age", "coords": ["a0", "a1", "a2"]}],
    "state": ["N[age]", "M[age]"],
    "transitions": [
        {
            "name": "renewal",
            "from": None,
            "to": "N[age=a0]",
            "rate": "sum_over(B[age:a] * (N[age:a] + M[age:a]), age=a)",
            "reactants": [],
        }
    ],
}
compiled = compile_spec(spec)
state = {"N": np.array([10.0, 20.0, 30.0]), "M": np.array([1.0, 2.0, 3.0])}
fertility = np.array([0.1, 0.2, 0.3])
births = fertility @ (state["N"] + state["M"])
assert compiled.pytree_eval_fn is not None
drift = compiled.pytree_eval_fn(0.0, state, B=fertility)
np.testing.assert_allclose(drift["N"], [births, 0.0, 0.0])
np.testing.assert_array_equal(drift["M"], np.zeros(3))

(reaction,) = compiled.reactions
assert reaction.from_base is None
assert reaction.from_axes == reaction.to_axes == ()
assert reaction.full_axes == ("age",)
assert reaction.pinned == (("age", 0),)
np.testing.assert_allclose(
    np.asarray(reaction.propensity_fn(0.0, state, B=fertility)), births
)
```

Here `a` is a local reduction variable. The bracket form `age:a` binds the
state or parameter position to that variable; it does not create a free
reaction-channel axis. `sum_over(B[age] * N[age], age=age)` also binds `age`
locally and counts the resulting flux once. Unknown reduction axes raise
`InvalidRhsSpecError`.

Named transitions publish a reaction artifact. With every destination axis
pinned, the artifact has one scalar propensity: one Poisson stream into the
specified cell. The rate already contains the population dependence; a
consumer must apply only the destination increment for each firing. The
explicit `reactants: []` declares that no population is consumed by a birth.
It does not imply that the rate is independent of population. With
`reactants: auto` instead, a birth rate such as
`sum_over(B[age:a] * N[age:a], age=a)` publishes no reactants, its
`dependencies` (every `N` cell it reads), and `propensity_order: 1`, which is
what adaptive tau-leaping needs to bound how fast the hazard can change.

If a group axis remains free, for example `to: N[age=a0,group]` with rate
`sum_over(B[age:a] * N[age:a,group], age=a)`, the propensity has one entry per
group. The age reduction happens within each group; `from_axes` and `to_axes`
are both `("group",)`, while `full_axes` retains the destination's full axis
order. Bound variables inside aliases have the same scope. An unreduced
free axis outside the channel axes remains outside the reaction-artifact
contract. For example, multiplying this pinned birth flux by `B[age]`
without reducing that occurrence introduces an extra free age dimension.

The flat RHS, PyTree RHS, and reaction propensity use the same flux. The
flepimop2 provider publishes the artifact through `system.option("reactions")`
and the shaped RHS through `system.option("pytree_stepper_fn")`. Both support
NumPy and JAX arrays, including runtime state and parameter changes under
JAX tracing.

## A stationary living population

Births alone increase total population. Conservation of the living total
requires balancing departures. The following three-bin model has renewal
births, aging, and constant per-capita mortality `mu`. `D` is an absorbing
counter of departures, so its derivative is positive even when the living
population is stationary.

The bins represent `[0, h)`, `[h, 2h)`, and the open-ended tail `[2h, infinity)`.
For a constant aging hazard `g`, the stationary living proportions are
`(1-q, q*(1-q), q**2)`, where `q = g/(g+mu)`. Setting
`g = mu/expm1(mu*h)` makes these exactly the bin integrals of the normalized
continuum density `mu*exp(-mu*a)`. This is an exponential-fitted aging rate.
The usual upwind rate `g = 1/h` instead gives a discrete geometric profile
that approaches the continuum profile as `h` decreases.

```python
import math

import numpy as np

from op_system import compile_spec

spec = {
    "kind": "transitions",
    "axes": [{"name": "age", "coords": ["a0", "a1", "a2"]}],
    "state": ["N[age]", "D[age]"],
    "transitions": [
        {
            "name": "renewal",
            "from": None,
            "to": "N[age=a0]",
            "rate": "sum_over(B[age:a] * N[age:a], age=a)",
            "reactants": [],
        },
        {
            "name": "depart",
            "from": "N[age]",
            "to": "D[age]",
            "rate": "mu",
            "reactants": [{"state": "N[age]", "order": 1}],
        },
        *(
            {
                "name": f"age_{k}",
                "from": f"N[age=a{k}]",
                "to": f"N[age=a{k + 1}]",
                "rate": "aging",
                "reactants": [{"state": f"N[age=a{k}]", "order": 1}],
            }
            for k in range(2)
        ),
    ],
}
mu, h, total = 0.2, 1.0, 1000.0
q = math.exp(-mu * h)
population = total * np.array([1.0 - q, q * (1.0 - q), q**2])
params = {
    "mu": np.asarray(mu),
    "aging": np.asarray(mu / math.expm1(mu * h)),
    "B": np.full(3, mu),
}
compiled = compile_spec(spec)
assert compiled.pytree_eval_fn is not None
drift = compiled.pytree_eval_fn(0.0, {"N": population, "D": np.zeros(3)}, **params)
np.testing.assert_allclose(drift["N"], np.zeros(3), atol=1e-12)
np.testing.assert_allclose(drift["D"], mu * population)
```

The last living bin has no aging outflow because it represents the entire
tail, rather than a truncated finite age interval. Constant fertility
`B = mu` makes births equal total living departures at any state. Thus
`sum(dN/dt) = 0` for arbitrary living populations, while the stationary
profile also makes each individual living-bin derivative zero.

For a stochastic consumer, these are population-dependent birth and death
hazards with zero expected total living-population drift. Individual paths
still fluctuate and can become extinct; the departure counter is excluded
from the living total and age distribution. The ensemble stationarity check
belongs in the numerical engine's integration tests.
