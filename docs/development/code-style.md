# Code style

This page captures conventions adopted across `op_system`. They are not strict
rules — exceptions exist where the trade-offs differ — but new code should
follow them by default and call out any deviation in the PR description.

## Records: `NamedTuple` vs `@dataclass`

`op_system` parsers and normalizers produce many small "value records": parsed
selectors, normalized axes, validated constraint rules, compiled kernel
descriptors, etc. For these we have a default and a fallback.

**Default to `typing.NamedTuple`** for small, value-like records that are
produced once by a parser and read many times.

Reasons:

- Immutable — downstream code cannot silently mutate the record.
- Hashable — usable as `dict` keys and in `set`s without extra work.
- Tuple-compatible — legacy unpacking (`base, tokens = parse_selector(s)`)
  keeps working and `NamedTuple` records compare equal to plain tuples of the
  same values.
- Zero boilerplate — fields and types in one declaration, no `__init__`
  required.

Use `@dataclass` (preferably `@dataclass(frozen=True, slots=True)`) when you
need any of:

- Mutability.
- `field(default_factory=...)` for mutable defaults.
- `__post_init__` validation that cannot live in a parser/classmethod.
- Inheritance.
- Keyword-only construction (`@dataclass(kw_only=True)`).
- Computed fields beyond what a `from_*` classmethod would express.

## Colocate parsers with the type they produce

Whenever a record has a non-trivial parsing/validation path, expose it as a
classmethod on the record type — typically `from_string`, `from_mapping`, or
`from_yaml_node`. The pattern is modeled on `ParsedShorthand.from_string` in
`flepimop2._utils._module`. This keeps the constructor of record next to the
spec for what a valid record looks like.

## Internal vs public names

Modules whose name starts with an underscore (`_axes`, `_constraints`,
`_helpers`, `_normalize`, `_symbols`, `_templates`) are internal to
`op_system`. Types and helpers defined in those modules should also be
underscore-prefixed unless they are deliberately re-exported from a public
module (e.g. `op_system.specs`, `op_system.compile`, the package `__init__`).

This applies to record classes too: an internal record produced by a private
parser is `_ConstraintRule`, not `ConstraintRule`. Promoting an internal type
to public should be a deliberate decision tied to a stability commitment.

## Error handling

Raise the specific exception type at the point of failure rather than
delegating to a `_raise_*` helper. The shared exception types live in
`op_system._errors`:

- `InvalidRhsSpecError(ValueError)` — structural problems with a normalized
  RHS spec (missing fields, wrong types, unknown references).
- `InvalidExpressionError(ValueError)` — expression strings that fail
  parsing/AST validation.
- `UnsupportedFeatureError(NotImplementedError)` — features declared in a
  spec that are not yet implemented.

All three subclass a built-in (`ValueError` / `NotImplementedError`), so
existing `except ValueError` / `except NotImplementedError` sites in
downstream code keep working while new code can catch the more specific
subclasses when needed.

Document the specific exception type in the function's `Raises:` block.
