# Independent batches and normalization cost

Reaction callbacks enforce the declared `template_shapes`: a scalar state
expects shape `()`, and an undeclared leading batch dimension raises an error
with the reaction, state name, expected shape and received shape. Declare a
categorical replicate axis for batched reactions. Passing arrays to a scalar
specification does not implicitly add an independent replicate dimension.

A categorical axis can index independent parameter draws as well as model
groups. Normalization expands the declared state cells and resolves their
equations and parameter coordinates; it occurs before numerical evaluation or
backend JIT compilation. The compiler caches lookup maps across these cells,
so state membership, template selection, and coordinate lookup no longer
repeat a scan over the entire axis for each reference.

For large independent batches, another useful option is to compile one block
and map it over the batch. Declare `factorize_axes` only when no equation
reduces, shifts, or mixes across the proposed batch axis. The compiler checks
this condition and exposes `block_pytree_eval_fn` with that axis removed.
Slice group-local parameters along with the state, and share global parameters:

```python
import jax
import jax.numpy as jnp
from op_system import compile_spec

compiled = compile_spec({
    "kind": "expr",
    "axes": [{"name": "draw", "coords": ["first", "second"]}],
    "factorize_axes": ["draw"],
    "state": ["X[draw]"],
    "equations": {"X[draw]": "-shared * rate[draw] * X[draw]"},
})
block = compiled.block_pytree_eval_fn
assert block is not None


def evaluate_one(x, rate):
    return block(0.0, {"X": x}, shared=0.5, rate=rate)["X"]


values = jnp.array([1.0, 2.0, 3.0])
rates = jnp.array([0.2, 0.3, 0.4])
derivatives = jax.jit(jax.vmap(evaluate_one))(values, rates)
```

The numerical batch can have more draws than the small compilation fixture:
the block evaluator contains no `draw` dimension. JAX mapping requires array
parameters whose mapped dimensions match the state. CPU Torch can use the
same pattern through `torch.func.vmap`, with Torch tensors and array operations.
Qualify the operations in the particular model before assuming additional
compiler or GPU support.

Mapping blocks changes compilation and evaluation costs, and does not speed up
every workload. An equation such as `sum_over(X[draw:d], draw=d)` couples the
draws and requires the monolithic evaluator. Removing that coupling to obtain
a block evaluator would change the model.

Local rate aliases retain the same equations in full and block evaluation. For
example, `foi[loc] = beta * r0_loc[loc] * I[loc]` uses a shared `beta` and a
group-local `r0_loc`. Supply the selected local rate with the corresponding
single-group state; expression and transition specifications should produce
the same cell flows. Required CPU qualification checks this behavior in float32
and float64, including JAX compilation of the stripped evaluator.
