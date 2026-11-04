# Recorded state maps

An `impulses` list records instantaneous state changes at fixed times. Between
these times, `eval_fn` continues to evaluate only the continuous right-hand side.
A solver must separately execute each published map. This is useful for synthetic
repeated-input models with different recorded events for individual coordinates.

```python
import numpy as np
from op_system import compile_spec

model = compile_spec({
    "kind": "expr",
    "axes": [{"name": "unit", "coords": ["a", "b"]}],
    "state": ["x[unit]"],
    "equations": {"x[unit]": "-rate[unit]*x[unit]"},
    "impulses": [
        {
            "time": 1.0,
            "updates": [
                {
                    "target": "x[unit=a]",
                    "operation": "add",
                    "expr": "amount[unit]",
                }
            ],
        }
    ],
})
state = np.array([2.0, 3.0])
event = model.impulses[0]
post = event.eval_fn(event.time, state, amount=np.array([5.0, 8.0]))
np.testing.assert_array_equal(post, [7.0, 3.0])
assert model.impulse_times == (1.0,)
assert event.parameter_axes == (("amount", ("unit",)),)
```

Every event needs a finite numeric `time` and a nonempty `updates` list. Times
must be strictly increasing; combine all updates at the same time into one event.
Selectors use the existing state syntax: `x[unit]` selects all coordinates and
`x[unit=a]` selects one. Include every declared axis, using wildcards for axes
that remain unrestricted. Unknown coordinates and overlapping writes (including
whole-axis/subset overlaps) fail during normalization.

Each update supplies `target`, `operation` and `expr`. `add` adds the expression
value to the pre-event state, `multiply` multiplies it, and `set` replaces it.
All expressions see the same pre-event state, regardless of declaration order.
Unselected coordinates remain unchanged. The input is never mutated.

The initial expression subset is pointwise arithmetic and `np.exp`, with scalar
or named-axis state and parameter references. Literal coordinate references
inside expressions use the expression syntax, such as `x['a']`; the `=` syntax
belongs to target selectors. Aliases, history operators, nested reductions and
time-indexed parameter buffers are explicitly unsupported in maps. Nonfinite
numeric literals, conflicting parameter axes and runtime shape errors are rejected.
Parameter values may remain differentiable native arrays. Map-only parameters
are listed in `event.parameter_axes` and `event.parameter_shapes` and do not
become required continuous-RHS parameters.

`CompiledImpulse.eval_fn(t, y, **params)` accepts a complete flat state;
`pytree_eval_fn` accepts all named state buffers. Both return the post-event state.
The convention is right-continuous: at an event time, saved observations should
use the post-event value. The solver owns step splitting, initial/final endpoint
policy, and restarting any cached integration history. For example, a solver may
treat `y0` as already post-event and apply maps in `(t0, tfinal]`; confirm its
published policy before use. Gradients are conditional on the recorded schedule;
there is no derivative contract for moving event times.

Schedules, selectors and map expressions are immutable compiler metadata and
participate in structural model identity. Changing any of them changes the
identity. A trusted `CompiledRhs` pickle rebuilds the maps. Models without events
retain their existing identity and behavior. Reaction-only consumers see a
`scheduled_state_maps` coverage gap, because their callbacks omit these changes.

The provider forwards `impulses` and `impulse_times` through its options and
requests map-only parameters. Its ordinary stepper raises for models with events.
An engine must validate schedule support before binding `option('impulse_rhs')`
and each map's `eval_fn`; this explicit handshake prevents old engines from
silently ignoring events. Structured/history callbacks remain unavailable on
this provider route until their solver event contracts are qualified.

This is partial support for issue #353. Table-derived schedules, automatic
mass-balance verification, state-triggered events and gradients with respect to
event times remain deferred. Gauge analysis/fixing and transformed state-coordinate views reject event models
until their map covariance contracts are implemented. Existing balance/export
tools describe the continuous equations; they do not certify or execute the full
event process.
Check conservation of any intended map with an independent invariant fixture.
