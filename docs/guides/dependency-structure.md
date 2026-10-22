# Conservative dependency structure

`dependency_structure(compiled)` exports current-input reads from retained
compiler IR without constructing numerical derivative matrices. A consumer can
use the rows to choose its own sparse storage or numerical strategy. Rows are
conservative: a recorded read can have a zero derivative at a particular point
or cancel algebraically. Runtime parameter values never prune the structure.

```python
import json
from op_system import compile_spec, dependency_structure

compiled = compile_spec({
    "kind": "expr",
    "state": ["x[loc]"],
    "axes": [{"name": "loc", "type": "categorical", "coords": ["a", "b", "c"]}],
    "equations": {"x[loc]": "x[loc:a] - rate*x[loc]"},
    "observables": {
        "total": {"expr": "x[loc]", "reduction": {"kind": "sum", "axes": ["loc"]}}
    },
})
structure = dependency_structure(compiled)
assert structure.status == "reported"
assert structure.state is not None
assert structure.state.input_names == ("x__loc_a", "x__loc_b", "x__loc_c")
assert structure.state.rows == ((0,), (0, 1), (0, 2))
assert structure.observable_state is not None
assert structure.observable_state.rows == ((0, 1, 2),)
assert structure.block_axes == ()
record = json.loads(structure.to_json())
assert record["model"] == json.loads(structure.model.to_json())
```

Each RHS output row represents one expanded scalar state derivative in the
existing `state_names` order. The example includes a pinned nonlocal read; the
export reuses the compiler's existing selector analysis rather than treating only
bare symbols as dependencies. Expanded reductions and transitive aliases are included.

Each **observable** row represents the complete named output, with its declared
`output_shape`. For a vector observable, that row unions reads across all output
cells, rather than describing one scalar Jacobian row. Reduction axes are also
unioned. These conservative groups can overestimate individual-cell reads.

Parameter columns name complete runtime input buffers. Their declared axes are
recorded in `parameter_axes`; a shaped buffer contributes one named group rather
than flattened scalar columns. This layout is distinct from statistical unknown
parameters or a parameter-sensitivity matrix. Observable-only inputs are included.
Time flags retain explicit `t` reads and reads of time-varying input buffers.

## Coverage and checked blocks

The report copies existing compiler-checked `factorize_axes` metadata into
immutable block snapshots. It performs no second independence proof and infers
no new factorizable axes. A global reduction or a pinned cross-block selector
defeats the existing proof when that axis is declared factorizable. Without that
declaration the model can compile, and the dependency rows retain its coupling.

A global observable can couple otherwise independent state trajectories.
Statistical likelihoods and priors require their own dependency analysis; an RHS
block proof alone does not establish posterior or Hessian factorization.

Operators, history and gauge-fixed execution return `status="unsupported"` with
explicit reasons and **None patterns**. Supported constant equations return
empty input rows instead. Preserve the original compiler/source identity when
saving the report and compare independently expected identities in the consuming
artifact's loading contract. The JSON export supplies data, without a generic
report loader or source authentication.

The defaults bound expanded states to 4,096, total edges across four families to
262,144, and traversed expanded IR nodes to 1,000,000. Invalid/exceeded limits
raise instead of returning truncated rows. Storage follows the retained rows and
edges; lazy equations are traversed row by row so an exceeded budget does not
first materialize every expanded row. Genuinely dense coupling can require
quadratically many edges. Existing
source normalization, alias/reduction expansion and structural-identity costs
remain part of export construction. These limits are not a process RSS bound.

The existing [bounded symbolic derivatives](portable-ir.md) still provide dense
matrices and dense `jvp`/`vjp`. Optional [native state actions](native-actions.md)
have a separate CPU qualification and storage/cost contract.
Dependency rows establish potential input reads, not differentiability,
numerical rank, global identifiability or numerical solver adequacy.
