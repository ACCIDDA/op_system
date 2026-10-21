# Structural model identity

`StructuralModelIdentity.from_compiled(compiled)` records the content of an
unmodified compiler-produced model. It adds no work to ordinary compilation,
binding, array evaluation or JIT unless explicitly called.

```python
from op_system import StructuralModelIdentity, compile_spec, structural_model_json

model = compile_spec({
    "kind": "expr",
    "state": ["x"],
    "equations": {"x": "-rate*x"},
})
identity = StructuralModelIdentity.from_compiled(model)
reference_json = identity.to_json()
assert StructuralModelIdentity.from_json(reference_json) == identity
manifest_json = structural_model_json(model)
```

The digest is `sha256:` followed by the lowercase SHA-256 of the canonical
manifest's UTF-8 bytes. The small reference contains exactly `schema_version`
and `digest`. Its loader rejects missing/unknown versions, booleans or floats
in place of the integer version, extra fields and malformed digests. A loaded
reference is a claim: guarded downstream loaders must compare it with the
expected identity supplied independently by the caller.

## Included structure

The manifest includes typed normalized expression/reduction/alias IR, ordered
state and parameter names, template shapes and axes, coordinate labels in their
original order, shaped/time-varying parameter axis order, normalized metadata,
declared constants/initial-state definitions, observables in declared order,
reaction-channel IR and gaps, operators/boundaries, factorization and declared
state-coordinate/gauge options. Normalized scientific annotations are included
conservatively. Gauge anchors are compilation choices; fitted runtime values
passed later to `bind` or evaluators are absent.

State, parameter, coordinate and observable order must be preserved because they
define binding and output layouts. Mapping key insertion order alone does not
define a layout and is canonicalized. Independent compilations and compiler
pickle round-trips reproduce the identity. This is conservative content equality:
algebraically equivalent programs can have different identities. It does not
change `CompiledRhs` equality or Python `hash` behavior.

## Schema 1 serialization

The top-level JSON object has `format: "op-system-structural-model"`, integer
`schema_version: 1`, a `nodes` table and a `root` table index. Each node is
`[type_tag, labels, child_indices]`. Tags distinguish `none`, `bool`, `int`,
`float`, `str`, `sequence`, `mapping` and compiler-owned typed `record` values.
Integers use decimal strings; floats use Python's exact hexadecimal string
representation, preserving signed zero. Strings use ASCII JSON escaping.
Sequence order is retained; tuples and lists share the sequence representation.
Mapping labels are sorted string keys. Record labels contain the type name and
its declared field names in order. Their children are those field values.

Nodes are visited in deterministic depth-first, left-to-right postorder and
interned by their complete shallow JSON representation. Equal subtrees share a
table entry even when allocated independently. Traversal is iterative; it does
not recursively expand shared IR or materialize deferred per-cell scalar IR.
Object addresses only memoize traversal and never enter the serialized content.
JSON keys are sorted, separators are `,`/`:`, and no nonstandard JSON numbers
are emitted. Changes to this encoding, selected structure, record fields or
normalization contract require deliberate compatibility review and a schema
change when the existing serialization contract changes. A golden digest
guards the current format.

## Boundaries

The retained compiler source is the authority. Do not modify its nested metadata
or replace evaluator closures and continue to identify it as the same compiler
artifact. Manual evaluator-only containers and legacy artifacts without typed
IR are rejected. Live arrays, arbitrary objects, non-string mapping keys and
cyclic metadata are rejected rather than silently copied or ignored. Runtime
parameter arrays belong in separate fitted/design/initial-state provenance.

This manifest is content bookkeeping. It neither authenticates a publisher nor
qualifies numerical/statistical behavior. Keep prior/design/statistical
observation identities, fitted values, numerical solver settings, derivative
contracts and dependency/source versions separately. The identity manifest does
not load executable models or implement the external Julia/IR export.
