*issu# Issue #168 — Block-stripped `pytree_eval_fn` for `factorize_axes` vmap

**Repo**: ACCIDDA/op_system
**Branch convention**: `feat/168-<slug>`
**Filed**: 2026-05-28
**Depends on**: PRs from #162 / #158 / #156 already merged on main.

---

## Problem

`pytree_eval_fn` is rank-rigid. Its lowered body contains literal axis indices
(`np.transpose(buf, (2, 0, 1))`, `np.sum(buf, axis=k)`, alias-buffer reshapes).
When `diffrax_engine` wraps it in `jax.vmap(in_axes=<block_axis_pos>)`, vmap
traces it on per-block slices where the block axis is missing → `axis k out of
bounds for array of dimension k-1` from inside an alias body.

`analyze_block_axes` already proves the block axis is separable, so stripping
it is well-defined.

## Decision

Add a **second** compiled function `block_pytree_eval_fn` on `CompiledRhs` that
operates on block-stripped state. Engine swaps to it under vmap. Stripping
logic lives in op_system, not the engine.

---

## Slices (each ≤ ~500 LOC)

### Slice 1 — `feat/168-strip-block-axis-ir`
**Goal**: IR-level "strip a block axis" primitive + unit tests.

- New `src/op_system/_normalize_block.py`:
  - `strip_block_axis(rhs: ExprRhs | TransitionsRhs, axis_name: str) -> ExprRhs | TransitionsRhs`
  - Walks `equations_ir`, `aliases_ir`, `state_templates`, `shaped_params`, `time_varying_params`.
  - Deletes `axis_name` from every shape tuple; rebuilds `axis_order`.
  - Strips coord-pinned subscripts on `axis_name` (`[loc:l]` → bare scalar slice).
  - Drops `Reduce` nodes whose axis is `axis_name` (already proven absent by `analyze_block_axes`; assert).
  - Refuses if `axis_name` is not in the spec axes.
- Tests in `tests/op_system/test_block_axes.py`:
  - `test_strip_block_axis_removes_axis_from_template_shapes`
  - `test_strip_block_axis_removes_axis_from_alias_bodies`
  - `test_strip_block_axis_strips_tv_param_axes`  (`beta[time, loc]` → `beta[time]`)
  - `test_strip_block_axis_raises_on_unknown_axis`

**Files**: `_normalize_block.py` (new), `tests/op_system/test_block_axes.py`
**Dependencies**: none.
**Budget**: ~250 LOC.

---

### Slice 2 — `feat/168-block-compile-pass`
**Goal**: Wire stripping into `compile_rhs`; surface `block_pytree_eval_fn` + `block_template_shapes`.

- `src/op_system/_vectorize.py`:
  - After existing flat + pytree compile, if `compiled.block_axes` is non-empty AND `pytree_eval_fn is not None`:
    1. `stripped = strip_block_axis(rhs, block_axes[0].name)`
    2. Run `_lower_to_pytree_eval_fn(stripped, ...)` → `block_pytree_eval_fn`
    3. Compute `block_template_shapes` from stripped `state_templates`.
  - Attach both to `CompiledRhs` (new fields, default `None`).
- `src/op_system/compile.py`:
  - `@dataclass` add `block_pytree_eval_fn: PytreeEvalFn | None = None`
  - `@dataclass` add `block_template_shapes: dict[str, tuple[int, ...]] | None = None`
- `tests/op_system/test_block_axes.py`:
  - `test_compile_emits_block_pytree_eval_fn_when_block_axes_nonempty`
  - `test_block_pytree_eval_fn_matches_monolithic_at_each_block_coord`
    - Compile SIR-like spec with `factorize_axes: [loc]`.
    - Pick random `y_full` shape `(age, vax, loc)`.
    - For each `i`: `vmap(block_eval, in_axes=loc_pos)(y_full)[base][i]` ≈
      `pytree_eval_fn(y_full)[base][..., i, ...]` (within 1e-12).
  - `test_compile_no_block_axes_leaves_block_fields_none`

**Files**: `_vectorize.py`, `compile.py`, `tests/op_system/test_block_axes.py`
**Dependencies**: Slice 1 merged.
**Budget**: ~300 LOC.

---

### Slice 3 — `feat/168-engine-uses-block-fn` ✅ DONE
**Goal**: Switch `diffrax_engine` to use `block_pytree_eval_fn`; delete the engine-side stripping helper.

This slice lands in **ACCIDDA/COVID19_USA**, not op_system.

**What was done**:
- `model_input/plugins/diffrax_engine.py`:
  - Merged `_build_operator_terms_pytree_block` into `_build_operator_terms_pytree` (added `block_axis_name: str | None = None` parameter); deleted the separate 80-LOC block function.
  - Block path now reads `stepper.option("block_template_shapes")` (from Slice 4 shim) instead of computing shapes inline.
  - Replaced `else pytree_stepper_fn` fallback with a `RuntimeError` assertion — `block_pytree_stepper_fn` must be set when entering the block path.
  - Acceptance gate (`SMH_R19_op_system_hierarchical.yml -t prior_predictive`) passes: EXIT=0.
- `tests/test_diffrax_engine.py`:
  - `_MockStepper` now exposes `block_pytree_stepper_fn` and `block_template_shapes` from `compiled.*`.
  - All 4 existing tests pass with the real `block_pytree_eval_fn` path (no fallback).

**Files**: COVID19_USA `model_input/plugins/diffrax_engine.py`, `tests/test_diffrax_engine.py`
**Dependencies**: Slice 2 released (op_system version bump).
**Budget**: ~150 LOC of net deletes + ~80 LOC of new tests.

---

### Slice 4 — `feat/168-flepimop2-shim-block-fields`
**Goal**: Surface new fields through `flepimop2-op_system` connector.

- `flepimop2-op_system/src/flepimop2/system/op_system/__init__.py`:
  - Add `"block_pytree_eval_fn"` and `"block_template_shapes"` to `self.options` (None when unused).
- `flepimop2-op_system/tests/test_system.py`:
  - Extend round-trip test for `factorize_axes: [loc]` to assert both new options are populated.

**Files**: shim + tests.
**Dependencies**: Slice 2 merged.
**Budget**: ~50 LOC.
Can be combined with Slice 2 if the connector is bumped at the same time.

---

## Work Order

```
1 (strip-block-axis-ir, op_system)
  → 2 (block-compile-pass, op_system)
    → 4 (flepimop2 shim)
    → 3 (engine swap, COVID19_USA)        ← end-to-end gate
```

Slice 3 is the user-visible fix; everything before it is plumbing.

## Acceptance gate (end of Slice 3)

```bash
cd /home/jcmacdo/Documents/GitHub/ACCIDDA/COVID19_USA
PYTHONPATH=$PWD conda run -n dev --no-capture-output --cwd $PWD \
  flepimop2 process configs/SMH_R19_op_system_hierarchical.yml -t prior_predictive
```
must complete and produce a non-empty `.nc` + `.pdf`.

## Files to delete after all slices

- COVID19_USA `model_input/plugins/diffrax_engine.py`: `_build_operator_terms_pytree_block` (~80 LOC).
- The `_axis_dim` helper inside `_build_operator_terms_pytree` can be simplified to the per-base axis-position lookup already used elsewhere.

## Open questions

1. Multi-block-axis support — leave as out-of-scope for #168; add `factorize_axes: [a, b]` support in a follow-up if it lands as a real use case.
2. Should `block_pytree_eval_fn` accept a `block_coord: int` arg (for debug) or stay vmap-shaped? Decision: stay vmap-shaped (cheaper; matches how engine calls it).
3. Where do we put the assertion that `analyze_block_axes` approved the axis? Inside `strip_block_axis` — refuse silently-undefined stripping.
