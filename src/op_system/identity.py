"""Versioned structural content identity for compiler artifacts.

Downstream forward and artifact provenance consume this manifest. It records
compiler structure, not fitted values, solver settings, runtime versions or proof of
numerical equivalence. It never changes Python object equality or hashing.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from op_system.compile import CompiledRhs

__all__ = ["StructuralModelIdentity", "structural_model_json"]


def _json(value: object) -> str:
    """Encode portable JSON deterministically.

    Returns:
        Compact ASCII JSON without nonstandard numeric literals.
    """
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )


@dataclass(frozen=True, slots=True)
class StructuralModelIdentity:
    """Validate a portable reference to a structural manifest.

    Attributes:
        schema_version: Canonical manifest and reference format, currently 1.
        digest: SHA-256 of the exact canonical manifest's UTF-8 bytes.

    Examples:
        >>> from op_system import compile_spec
        >>> model = compile_spec({
        ...     "kind": "expr",
        ...     "state": ["x"],
        ...     "equations": {"x": "-k*x"},
        ... })
        >>> identity = StructuralModelIdentity.from_compiled(model)
        >>> StructuralModelIdentity.from_json(identity.to_json()) == identity
        True
        >>> identity.digest.startswith("sha256:")
        True
    """

    schema_version: int
    digest: str

    def __post_init__(self) -> None:
        """Refuse unknown formats and malformed content references.

        Raises:
            ValueError: Version or digest is missing, malformed or unsupported.
        """
        if type(self.schema_version) is not int or self.schema_version != 1:
            msg = "Unsupported structural model identity schema version (expected 1)"
            raise ValueError(msg)
        if (
            not isinstance(self.digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest) is None
        ):
            msg = "Structural model identity requires a lowercase SHA-256 digest"
            raise ValueError(msg)

    @classmethod
    def from_compiled(cls, compiled: CompiledRhs) -> StructuralModelIdentity:
        """Identify compiler structure without binding runtime parameters.

        Args:
            compiled: An unmodified artifact produced by the compiler.

        Returns:
            Reference to :func:`structural_model_json`'s canonical bytes.
        """
        payload = structural_model_json(compiled).encode("utf-8")
        return cls(1, f"sha256:{hashlib.sha256(payload).hexdigest()}")

    def to_json(self) -> str:
        """Serialize the small reference, excluding the model manifest.

        Returns:
            Versioned reference JSON; this does not serialize a runnable model.
        """
        return _json({"schema_version": self.schema_version, "digest": self.digest})

    @classmethod
    def from_json(cls, payload: str) -> StructuralModelIdentity:
        """Read a reference without treating it as an independently expected identity.

        Args:
            payload: Exact two-field reference JSON.

        Returns:
            Validated reference. Downstream guarded loaders must additionally
            compare an independently supplied expected identity.

        Raises:
            ValueError: The reference is malformed or uses an unknown schema.
        """
        value = json.loads(payload)
        if not isinstance(value, dict) or set(value) != {"schema_version", "digest"}:
            msg = "Structural model identity requires schema_version and digest only"
            raise ValueError(msg)
        return cls(value["schema_version"], value["digest"])


class _ManifestEncoder:
    """Intern a typed DAG without expanding shared trees or Python recursion."""

    def __init__(self) -> None:
        self.nodes: list[object] = []
        self._content: dict[str, int] = {}
        self._objects: dict[int, int] = {}

    def _intern(self, node: object) -> int:
        """Intern a shallow node by content.

        Returns:
            Deterministic postorder index, independent of object sharing.
        """
        key = _json(node)
        if key not in self._content:
            self._content[key] = len(self.nodes)
            self.nodes.append(node)
        return self._content[key]

    def encode(self, root: object) -> int:
        """Traverse source values iteratively in semantic order.

        Returns:
            Root node index in the canonical node table.

        Raises:
            TypeError: Source metadata contains a cycle.
        """
        stack: list[tuple[object, bool]] = [(root, False)]
        active: set[int] = set()
        while stack:
            value, visited = stack.pop()
            if id(value) in self._objects:
                continue
            tag, labels, children = _parts(value)
            if children and not visited:
                if id(value) in active:
                    msg = "Structural metadata must not contain cycles"
                    raise TypeError(msg)
                active.add(id(value))
                stack.append((value, True))
                stack.extend((child, False) for child in reversed(children))
                continue
            references = [self._objects[id(child)] for child in children]
            self._objects[id(value)] = self._intern([tag, labels, references])
            active.discard(id(value))
        return self._objects[id(root)]


def _parts(value: object) -> tuple[str, object, tuple[object, ...]]:  # ruff: ignore[complex-structure, too-many-return-statements]
    """Describe one value without coercing arrays or serializing executable code.

    Returns:
        Type tag, scalar/field labels and ordered children.

    Raises:
        TypeError: Unsupported metadata would otherwise be silently omitted.
    """
    if value is None:
        return "none", None, ()
    if isinstance(value, bool):
        return "bool", value, ()
    if isinstance(value, int):
        return "int", str(value), ()
    if isinstance(value, float):
        return "float", value.hex(), ()
    if isinstance(value, str):
        return "str", str(value), ()
    if is_dataclass(value) and not isinstance(value, type):
        names = tuple(field.name for field in fields(value))
        return _record_parts(value, names)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return _record_parts(value, cast("tuple[str, ...]", value._fields))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            msg = "Structural metadata mapping keys must be strings"
            raise TypeError(msg)
        keys = tuple(sorted(value))
        return "mapping", keys, tuple(value[key] for key in keys)
    if isinstance(value, (tuple, list)):
        return "sequence", None, tuple(value)
    msg = f"Unsupported structural metadata type: {type(value).__qualname__}"
    raise TypeError(msg)


def _record_parts(
    value: object, names: tuple[str, ...]
) -> tuple[str, object, tuple[object, ...]]:
    """Describe a compiler-owned typed record.

    Returns:
        Stable type/field tags and ordered field values.

    Raises:
        TypeError: A user object is not a compiler metadata record.
    """
    kind = type(value)
    if not kind.__module__.startswith("op_system."):
        msg = "Structural manifests accept only compiler-owned typed records"
        raise TypeError(msg)
    return (
        "record",
        [kind.__name__, names],
        tuple(getattr(value, name) for name in names),
    )


def structural_model_json(compiled: CompiledRhs) -> str:
    """Serialize typed normalized IR and ordered compiler layouts as a canonical DAG.

    Args:
        compiled: An unmodified compiler-produced artifact with retained source IR.

    Returns:
        Canonical schema-1 UTF-8/ASCII JSON text. Mapping keys are sorted;
        state, parameter, axis, coordinate, operation and observable sequences
        retain their semantic order. Scalars have explicit type tags and floats
        use exact hexadecimal strings. Shared typed trees are content-interned.
        The manifest includes normalized declarations/constants/operators and
        gauge/coordinate options, but never bound runtime values or closures.
        This is an identity manifest, not the external executable IR export.

    Raises:
        ValueError: The artifact has no typed normalized source.
    """
    rhs = compiled._rhs  # ruff: ignore[private-member-access]
    if (
        rhs is None
        or not rhs.equations_ir_reduce
        or any(expr is None for expr in rhs.equations_ir_reduce)
    ):
        msg = "Structural identity requires compiler-produced typed normalized IR"
        raise ValueError(msg)
    source = {
        "kind": type(rhs).__name__,
        "state_names": compiled.state_names,
        "param_names": compiled.param_names,
        "state_templates": rhs.state_templates,
        "shaped_params": rhs.shaped_params,
        "time_varying_params": rhs.time_varying_params,
        "alias_templates": rhs.alias_templates,
        "aliases_ir": rhs.aliases_ir_reduce,
        "equations_ir": rhs.equations_ir_reduce,
        "metadata": rhs.meta,
        "reactions_ir": getattr(rhs, "reactions_ir", ()),
        "reaction_gaps": getattr(rhs, "reaction_gaps_ir", ()),
        "observables": tuple(compiled.observables),
        "template_shapes": compiled.template_shapes,
        "block_template_shapes": compiled.block_template_shapes,
    }
    encoder = _ManifestEncoder()
    root = encoder.encode(source)
    return _json({
        "format": "op-system-structural-model",
        "schema_version": 1,
        "nodes": encoder.nodes,
        "root": root,
    })
