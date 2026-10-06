# Aging chains with `coord_shift`

A `coord_shift` transition moves mass along one axis of a state template. It
has two forms.

The pairwise form moves one coordinate to another:

```yaml
- coord_shift: {age: "a0 -> a1"}
  rate: aging_rate[age]
  apply_to: [S, I]
```

An aging chain over `n` bins needs `n - 1` such entries. Each entry adds a
masked term to every cell's equation, so the compile cost grows with the
square of the number of bins. A named pairwise entry publishes one point-to-
point reaction per state, named `{name}_{state}`.

The axis-wide form shifts every coordinate in a single entry:

```yaml
- name: aging
  coord_shift: {axis: age, step: 1, rate: "aging_rate[age]", boundary: absorb}
  apply_to: [S, V, I, R]
```

- `step` is a nonzero integer smaller in magnitude than the axis length.
  Bin `k` moves to bin `k + step`; `step` defaults to `1`.
- `rate` is a per-capita rate. A free `age` index in the rate, including one
  inside an alias or a state reference, reads the **source** bin. `rate` may
  appear inside `coord_shift` or on the transition, but not in both places.
- `boundary` decides what happens to sources whose destination `k + step`
  falls off the axis. `absorb` removes that mass from the system (for
  example, culling at the oldest age). `stay` leaves it in place, giving an
  open-ended terminal bin. `boundary` has no default: omitting it, or
  writing `error`, rejects the spec and names the two alternatives.
- Every `apply_to` state needs one state template with the shifted axis as a
  wildcard and no pinned coordinates. Other axes, such as `vax` or `loc`,
  are carried through unchanged.

```python
import numpy as np

from op_system import compile_spec

ages = [f"a{k}" for k in range(4)]
spec = {
    "kind": "transitions",
    "axes": [{"name": "age", "type": "ordinal", "coords": ages}],
    "state": ["S[age]"],
    "transitions": [
        {
            "name": "aging",
            "coord_shift": {
                "axis": "age",
                "step": 1,
                "rate": "aging_rate[age]",
                "boundary": "absorb",
            },
            "apply_to": ["S"],
        }
    ],
}
compiled = compile_spec(spec)
rates = np.array([0.5, 0.4, 0.3, 0.2])
s = np.array([10.0, 20.0, 30.0, 40.0])
flow = rates * s
expected = -flow
expected[1:] += flow[:-1]
np.testing.assert_allclose(compiled.eval_fn(0.0, s, aging_rate=rates), expected)
```

With `boundary: stay` the last bin keeps its mass, so `expected[-1]` would be
`flow[-2]` and nothing would leave the system.

## Deterministic lowering

Each `apply_to` state lowers once, as matrix routing along the shifted axis
with a synthesized one-hot shift matrix. Cell values equal those of the
equivalent pairwise entries; for forward shifts they match bit for bit.
Compile time no longer grows with the number of coordinate pairs: a
240-bin, four-state chain compiles in tens of milliseconds, compared with
several seconds for 239 pairwise entries. Evaluation contracts the shift
matrix, which costs O(n²) per template, the same order as the masked
pairwise terms.

The shift matrix and the `stay` keep-mask are injected automatically under
reserved `__op_system_shift*` names. They are not model parameters.

## Reaction artifacts

A named entry publishes one templated reaction per `apply_to` state, named
`{name}_{state}` (`aging_S` above). The reaction does not have one pinned
artifact per coordinate pair. Its propensity is shaped like the full
template, with one channel per source cell:

```python
(reaction,) = compiled.reactions
assert reaction.name == "aging_S"
assert reaction.from_axes == reaction.full_axes == ("age",)
assert reaction.to_axes == ()
assert reaction.sum_axes == reaction.pinned == ()
assert reaction.offsets == (("age", 1),)
np.testing.assert_allclose(
    np.asarray(reaction.propensity_fn(0.0, {"S": s}, aging_rate=rates)), flow
)
```

The shifted axis appears in `offsets`, not in `to_axes` or `pinned`. A firing
at source index `k` removes one unit from `(S, k)` and adds one to
`(S, k + step)`. Other destination axes are copied from `to_axes` and fixed
by `pinned` as usual. When `k + step` is off the axis, the firing removes
the unit and deposits nothing (`absorb`). Under `stay`, those sources have
zero propensity, so they never fire.

A consumer that predates `offsets` fails its own metadata checks on these
reactions. The destination axes no longer cover `full_axes`, so the reaction
is rejected rather than silently treated as a no-op.

By default the reactants metadata is the consumed source alone. It is
complete when the rate reads no state, as for a constant aging rate, and
otherwise reports `reactants_complete=False`. One entry generates a reaction
per `apply_to` state, so it cannot list each one's consumed source. Instead,
declare the reactants beyond the shifted state as `catalysts`, or set
`catalysts: auto` to infer them from the rate; `catalysts: []` declares a
first-order shift. The generator adds the consumed source, and each reaction
reports `reactants_complete=True`, as adaptive tau-leaping requires.
Both `coord_shift` forms accept `catalysts` and reject `reactants`.
