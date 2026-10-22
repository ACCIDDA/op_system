# Compiler cache lifetimes

Identity keys are sound only while every keyed object stays alive. They are
not content hashes. A cache of a rewritten tree must not return another tree's
result after Python reuses an address.

The earlier alias cross-reference fix scopes `_InlineMemo.body_refs_for` to its
batch; there is no `_BODY_REFS_CACHE` process global. `_InlineMemo` retains
intermediate rewritten roots so its shared free-symbol memo cannot observe a
freed address. Expansion-result entries retain and verify their original node.
These protections remain in place.

| Cache | Owner and lifetime invariant |
| --- | --- |
| `_InlineMemo` free symbols/body references | One alias-inlining batch; retained intermediate roots and the current alias mapping stay alive until the memo is released. |
| `_ExpandResultMemo` | Expansion pass; each entry stores its original node and expanded result, and verifies identity before reuse. |
| `_free_axes_in` memo | Expansion pass; input DAGs remain owned by the caller. A shared memo must not walk freshly built, unretained trees. |
| `inline_aliases` result memos | Roots/terms remain owned by parsed inputs or `equations_ir_pre_inline` for the whole pass. Fresh unretained roots need a different ownership strategy. |
| Unparse memos | Equation/alias IR containers retain the input DAG for the complete rendering pass, including lazy equation wrappers that retain their source. |
| `apply_plus_dedup` | Stored `Apply` results retain all argument objects whose identities form their keys. |
| Template substitution table | An immutable prepared table belongs to the initial-state or transition-normalization batch. It contains selector keys/patterns, not expanded mapping values. It is never retained by a process-global mapping-identity cache. |

The previous `_APPLY_TPL_PREPARSE_CACHE` retained each template mapping and all
its expanded cell/assignment lists indefinitely. It also keyed preparation only
on mapping identity, although shaped-parameter exclusions affect the result.
Standalone substitutions now prepare from their current complete context.
Hot normalization callers prepare once per batch and reuse that table across
cells. The initial-state and transition-name batches release the table naturally.
No process-wide invalidation or weak-reference support on ordinary dicts is needed.

Focused regression checks use weak references to caller-owned mappings and check
that they are collectible after standalone and initial-state substitution. Another
check reuses one mapping under different shaped-input declarations and requires
`X[loc]` to become a state name in one context and an array index in the other.
All three fail on the previous implementation. Existing IR regressions exercise
intermediate-tree identity reuse and reject deliberately mismatched expansion
cache entries.

Content-keyed parse, selector/suffix and token caches are separate. This audit
does not prove bounded memory for an unbounded stream of distinct expressions,
Python allocator RSS, solver histories or training data. Measure those lifetimes
separately before adding global cache policies or claiming application memory gains.

## Repeated-compiler retention check

The [public receipt](../assets/cache-lifetimes/retention.json) contains the exact
probe, synthetic specification, source hashes and all checkpoints. Each fresh
process warms three compiles of an identical 64-state model with expression
initial conditions, then traces 30 additional compiles. Every compiled artifact
is dropped and garbage collection runs before sampling. At 30 compiles,
retained traced Python allocations above the warmed origin are 610271 bytes on
the previous implementation and 4896 bytes with batch-owned preparation. The
previous checkpoints grow roughly linearly; candidate checkpoints remain small.

This measures one specific retention path and a bounded repeated-input workload.
It does not imply a 99% reduction in process RSS or in a solver/training job.
Distinct-expression content caches and the Python allocator remain separate.
