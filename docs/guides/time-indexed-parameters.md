# Time-indexed parameters

A parameter subscripted with the configured time axis is supplied as a table.
For example, `rate[time]` takes one value per declared time coordinate, while
`rate[group,time]` takes one table per group. The compiler removes the time
dimension at each evaluation and passes the resulting scalar or shaped value
to the equations and reaction propensities. The time axis can occupy any
position in the parameter array; its coordinate order defines the table order.

The top-level `time_interpolation` field chooses one policy for all
time-indexed parameters in the specification:

| Policy | Value between adjacent coordinates |
| --- | --- |
| `linear` | Linear interpolation; the default when the field is omitted |
| `previous` | Hold the value at `t_i` on `[t_i, t_{i+1})` |

Both policies clamp to the nearest endpoint outside the coordinate range.
`previous` is right-continuous: evaluating exactly at a coordinate returns the
new table value. For coordinates `[0, 1, 2]` and values `[0, 3, 1]`, evaluations
at `0.5`, `1`, and `1.5` return `[0, 3, 3]` with `previous` and `[1.5, 3, 2]`
with `linear`.

```python
import numpy as np

from op_system import compile_spec

spec = {
    "kind": "transitions",
    "time_interpolation": "previous",
    "axes": [
        {"name": "group", "coords": ["a"]},
        {"name": "time", "type": "continuous", "coords": [0.0, 1.0, 2.0]},
    ],
    "state": ["A[group]", "B[group]"],
    "transitions": [
        {
            "name": "transfer",
            "from": "A[group]",
            "to": "B[group]",
            "rate": "rate[time]",
            "reactants": [{"state": "A[group]", "order": 1}],
        }
    ],
}
compiled = compile_spec(spec)
table = np.asarray([0.0, 3.0, 1.0])
drift = compiled.eval_fn(1.5, np.asarray([10.0, 0.0]), rate=table)
np.testing.assert_array_equal(drift, [-30.0, 30.0])
assert compiled.meta["forcing_breakpoints"] == (1.0, 2.0)
```

Use `time_axis: day` with a declared `day` axis and subscripts such as
`rate[day]` to rename the time axis. Active time coordinates must be finite,
numeric, and strictly increasing. A continuous time axis may explicitly
declare a single coordinate; its table is constant for every evaluation time.
Other continuous axes retain their requirement for at least two coordinates.
Unknown policy values are rejected during normalization.

Flat, PyTree, block, and reaction evaluators use the same policy. They adopt
tables into the evolving state's namespace at call time, including JAX under
`jit` and `vmap`. Runtime table values are not baked into compiled functions.

## Metadata for numerical engines

Normalized and compiled metadata expose `time_axis`, `time_interpolation`,
`time_coordinates`, and `forcing_breakpoints`. Coordinates and breakpoints are
immutable tuples. The flepimop2 system provider exposes the same four values
through `system.option(...)`.

For an active `previous` table, forcing breakpoints are all coordinates after
the first. The first coordinate introduces no change because the first table
value also applies before it. A single-coordinate table has no breakpoints.
Linear interpolation and specifications without active time-indexed parameters
publish an empty breakpoint tuple.

An engine can pass the published forcing boundaries to its breakpoint-aware
stochastic solver independently of the observation grid. In `op_engine`,
pure stochastic direct SSA, fixed tau-leaping, and adaptive tau-leaping accept
`forcing_breakpoints`; provider consumers currently supply that configuration
explicitly. The interpolation setting changes table evaluation only. A rate
that also uses `t` directly can still vary between table coordinates. Exact
breakpoint SSA and adaptive exact fallback require all external time dependence
to be constant between declared changes while the state is unchanged.

Linear tables continue to produce smoothly varying rates. Listing their
coordinates as breakpoints does not make frozen-rate SSA exact; those rates
require a suitable time-dependent waiting-time method.
