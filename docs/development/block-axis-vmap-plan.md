# Plan: `jax.vmap` over block-diagonal axes (`factorize_axes`)

**Status**: ✅ **complete** 2026-05-28. All four slices merged. Naming: keep `factorize_axes`.

## 1. What's already true today

| Layer | Status |
|---|---|
| `op_system.compile_rhs` | Produces both `eval_fn` (flat) and `pytree_eval_fn` (`StateDict → StateDict`), plus `template_shapes` and a passthrough `factorize_axes: tuple[str, ...]` field. |
| `op_system` vectorizer | Already emits one shaped tensor expression per state template; `factorize_axes` is **not** consumed by the vectorizer — it is advisory metadata. |
| `flepimop2-op_system` connector | Exposes `pytree_stepper_fn`, `template_shapes`, `factorize_axes`, `axis_order`, `axis_sizes`, `axis_labels`, `axis_coords` on `options`. |
| `diffrax_engine.runner` / `runner_vmap` | One ODE solve per draw; `runner_vmap` already vmaps **draws**. No partitioning across model axes. |
| Configs (`SMH_R19_op_system*`) | FOI is per-`loc` (no cross-loc reads). All state templates contain `loc`. No params index `loc` except `rho_h[loc]`, `rho_d[loc]` (used only in the observation likelihood, not the RHS). |

So the runtime contract `pytree_stepper_fn(t, {base: y}, **params) → {base: dy}` is already block-diagonal-friendly *if* the RHS is structurally separable across the partition axis. The pieces that don't yet exist are: **a structural separability check**, **declaration in the spec**, and **a vmap-over-partition-axis path in the engine.**

## 2. Concept and contract

Introduce a single, narrow concept: a **block axis** — a model axis `a` such that for every state template the RHS at coord `a=i` depends only on values at coord `a=i` of every other buffer (state or param) that carries axis `a`, and on values at every coord of buffers that do not carry axis `a`.

This is exactly what allows

```
dy/dt = f(t, y)   →    dy[..., i, ...]/dt = f_i(t, y[..., i, ...])
```

and therefore allows the engine to evaluate `f` with `jax.vmap(in_axes=axis_position)` over coord `i`.

Reuse the existing `factorize_axes` spec field as the **block axis declaration**. It already round-trips through normalize → compile → connector. We just give it real semantics in op_system and a real consumer in the engine.

## 3. Plan, by layer

### 3.1 `op_system` — promote `factorize_axes` from advisory to checked

Goal: at compile time, prove the declared axes are block-diagonal in the IR; emit per-template metadata the engine can trust without re-inspecting the IR.

1. **New IR pass** `analyze_block_axes(rhs) -> BlockAxisReport` in a new `_block_axes.py`:
   - For each axis `a` in `meta["factorize_axes"]`:
     - For each `equations_ir` / `aliases_ir` expression: walk subscripts.
     - **Reject** if any `apply_along(..., a=...)` / `sum_over(a)` / `Reduce` mentions `a` (cross-coord reduction).
     - **Reject** if any subscript pins `a` to a literal coord (e.g. `loc=AK`) without that same coord being the equation's cell coord on `a`.
     - **Reject** if any subscript renames `a` to a fresh bound name and uses it elsewhere at a different cell coord (`I[..., loc:l] * J[..., loc:m]` with `l != m`).
     - **Allow** any expression that touches `a` only as the matching free position of the cell template, or never mentions `a`.
   - Also walk the `operators` list so a kernel keyed on `a` is rejected statically.
   - Classify each **parameter** the RHS reads as `axis_kind ∈ {has_axis, lacks_axis}` for each block axis. (Already available via `time_varying_params` + `shaped_params` axes tuples.)
   - Output: `BlockAxisInfo(name, size, state_axis_pos: dict[base, int], param_axis_pos: dict[name, int|None])`.

2. **Wire into `compile_rhs`**:
   - Call `analyze_block_axes(rhs)` for axes in `meta["factorize_axes"]`.
   - On any failure: raise `UnsupportedFeatureError` with the specific equation/alias and offending subscript (fail loudly, like the existing vectorize bail).
   - On success: attach to `CompiledRhs` as a new field `block_axes: tuple[BlockAxisInfo, ...]`. Keep it pickle-stable: plain dataclass with builtin types.
   - Continue exposing `factorize_axes` as the raw tuple for backwards compat.

3. **Tests** in `tests/op_system/test_block_axes.py`: separable RHS passes; cross-loc RHS rejects with named reason; mixed state where one template lacks the axis rejects; operator on partition axis rejects.

4. **Docs** in `docs/api-reference/specs.md`: short section on block axes.

LOC estimate: ~400 src + ~250 tests.

### 3.2 `flepimop2-op_system` connector — publish the new info

`src/flepimop2/system/op_system/__init__.py`:

1. Forward `compiled.block_axes` onto `self.options["block_axes"]`.
2. No behaviour change in `_bind_impl`.
3. Tests in `flepimop2-op_system/tests/test_system.py`: round-trip for a config declaring `factorize_axes: [loc]`; empty otherwise.

LOC estimate: ~30 src + ~50 tests.

### 3.3 `diffrax_engine` — implement the partitioned solve

`COVID19_USA/model_input/plugins/diffrax_engine.py`:

1. **`_prepare_call` extension**: read `stepper.option("block_axes")`. Pick one block axis for the run via a new opt key `block_solve_axis` (default: first entry of `block_axes`; `None` disables). Stash on `prep` as `block_axis: BlockAxisInfo | None`.

2. **New solve closure** when `block_axis` is set and we are on the pytree path:
   - Build `_solve_one_block` over the sliced state dict + sliced params.
   - `_solve = jax.vmap(_solve_one_block, in_axes=(per_base_axis_pos, per_param_axis_pos))` — positions from `block_axis.state_axis_pos` / `block_axis.param_axis_pos` (None ⇒ broadcast). Use `out_axes` so vmap re-inserts at the original axis position.
   - The inner `_solve_one_block` calls `pytree_stepper_fn` with the sliced `state_dict` and sliced params; diffrax solves a much smaller ODE.

3. **`runner` semantics**: pytree return is still `{base: (T, *full_shape)}` — vmap reinserts the block axis automatically (use `out_axes` to land it at the original position; no reshape needed).

4. **`runner_vmap` semantics**: nest as `jax.vmap(draws, jax.vmap(block_axis, _solve_one_block))`. Static signatures already match across draws, so the existing "same `_solve`" guard still works.

5. **Operator-splitting** (`_make_advection_term_pytree`, `_make_jump_term_pytree`): build operator terms inside `_solve_one_block` (after slicing) so they see the per-block shapes.

6. **Fallback / safety**:
   - If `block_axes` is empty or `pytree_mode` is False: behave exactly as today (no regression).
   - If a param the engine routes as dynamic carries the block axis but `block_axis.param_axis_pos[name]` is `None`: raise clearly.

7. **Tests** in `COVID19_USA/tests/test_diffrax_engine.py`:
   - Numerical equivalence: monolithic solve vs. block-partitioned solve on a tiny 2-loc separable config — must match to solver tolerance.
   - Shape contract: `runner` output dict has correct full shapes.
   - Negative: monolithic-only config (no `factorize_axes`) still passes through unmodified.

LOC estimate: ~350 src + ~200 tests.

### 3.4 Configs — opt in

In each config that should solve per-location, add at the spec level:

```yaml
spec:
  kind: transitions
  factorize_axes: [loc]
  axes: ...
```

No other config changes needed — FOI is already loc-local in the continuum configs.

### 3.5 `flepimop2` core — no changes

`flepimop2`'s `EngineProtocol.runner / runner_vmap` signatures are unchanged. No core flepimop2 edits.

## 4. Order of work

1. op_system: `analyze_block_axes` + `CompiledRhs.block_axes` + tests.
2. Connector: forward `block_axes` to options + tests.
3. `diffrax_engine`: partitioned `_solve_one_block` + numerical-equivalence test.
4. Opt configs into `factorize_axes: [loc]` and run prior-predictive smoke.

## 5. Risks and open questions

- **Operator terms with non-local kernels on the block axis**: the separability analyzer rejects these — IR walk must include the `operators` list.
- **`runner_vmap` × block-vmap memory**: vmapping draws outside and block-axis inside means the inner solve has shapes `(D, n_loc, ...)`. Same total work as today but XLA may pick a different schedule; benchmark on GPU before committing nesting order.
- **Multiple block axes**: spec already accepts a list. v1: support exactly one runtime block axis (pick first; engine opt overrides). Multi-axis nesting is a follow-up.
