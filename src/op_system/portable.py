"""Versioned typed IR interchange bound to the compiler's structural identity.

The envelope contains data, never pickled objects or executable source. The
canonical typed DAG retains axes, ordered layouts, aliases, observables, constants,
known time-varying inputs and declaration metadata without eager cell expansion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from op_system.identity import StructuralModelIdentity, structural_model_json

if TYPE_CHECKING:
    from op_system.compile import CompiledRhs

__all__ = ["IRRecord", "PortableIR", "export_ir"]

_FORMAT = "op-system-external-ir"
_MAX_PAYLOAD_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class IRRecord:
    """Readonly compiler record as a portable type name and ordered fields.

    Decoding does not import or instantiate the named Python class. Consumers
    explicitly select supported record/operation types before evaluation.
    """

    type_name: str
    fields: tuple[tuple[str, object], ...]

    def field(self, name: str) -> object:
        """Return one declared field.

        Returns:
            The readonly decoded field value.

        Raises:
            KeyError: The record has no such field.
        """
        for key, value in self.fields:
            if key == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class PortableIR:
    """Immutable canonical IR data with independently checkable source identity.

    ``manifest`` is the canonical structural DAG. Its hash is the exact existing
    StructuralModelIdentity, not a hash of fitted values or emitted code. Use
    :meth:`from_json` with an independently expected model identity on import.
    A digest detects changes against that expectation; it is not authentication.
    """

    manifest: str
    identity: StructuralModelIdentity

    def __post_init__(self) -> None:
        """Validate canonical format, bounds and a non-executable typed DAG.

        Raises:
            ValueError: Format, bounds or identity disagree.
        """
        if (
            not isinstance(self.manifest, str)
            or len(self.manifest.encode()) > _MAX_PAYLOAD_BYTES
        ):
            msg = "Portable IR manifest must be bounded UTF-8 JSON text"
            raise ValueError(msg)
        source = _load_json(self.manifest)
        canonical = _json(source)
        expected = f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
        if canonical != self.manifest or expected != self.identity.digest:
            msg = "Portable IR canonical bytes differ from structural model identity"
            raise ValueError(msg)
        _decode(source)

    def decode(self) -> Mapping[str, object]:
        """Return fresh readonly data without executing or instantiating records."""
        return _decode(_load_json(self.manifest))

    def to_json(self) -> str:
        """Return canonical schema-1 external interchange JSON."""
        return _json({
            "format": _FORMAT,
            "schema_version": 1,
            "source": _load_json(self.identity.to_json()),
            "manifest": _load_json(self.manifest),
        })

    @classmethod
    def from_json(
        cls, payload: str, *, expected: StructuralModelIdentity
    ) -> PortableIR:
        """Read bounded data only when its source matches an independent identity.

        Unsupported versions, extra envelope fields and changed IR fail before
        the consumer gets decoded data. No arbitrary Python/Julia source runs.

        Returns:
            Validated canonical interchange with the expected source identity.

        Raises:
            ValueError: Payload, version or source identity differs.
        """
        if not isinstance(payload, str) or len(payload.encode()) > _MAX_PAYLOAD_BYTES:
            msg = "Portable IR payload exceeds its UTF-8 byte limit"
            raise ValueError(msg)
        data = _load_json(payload)
        if (
            not isinstance(data, dict)
            or set(data) != {"format", "schema_version", "source", "manifest"}
            or data["format"] != _FORMAT
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
        ):
            msg = "Unsupported portable IR envelope/schema"
            raise ValueError(msg)
        source = StructuralModelIdentity.from_json(_json(data["source"]))
        if source != expected:
            msg = "Portable IR source differs from independently expected identity"
            raise ValueError(msg)
        return cls(_json(data["manifest"]), source)


def export_ir(compiled: CompiledRhs) -> PortableIR:
    """Export retained compiler IR without materializing lazy scalar equations.

    Source metadata and ordering are those of :func:`structural_model_json`.
    This interchange does not establish that every external tool can evaluate
    every IR operation or that different tools use the same rank definition.

    Returns:
        Canonical schema-1 IR data bound to the source model identity.
    """
    manifest = structural_model_json(compiled)
    return PortableIR(
        manifest,
        StructuralModelIdentity(
            1, f"sha256:{hashlib.sha256(manifest.encode()).hexdigest()}"
        ),
    )


def _json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )


def _decode(source: object) -> Mapping[str, object]:
    if (
        not isinstance(source, dict)
        or set(source) != {"format", "schema_version", "nodes", "root"}
        or source["format"] != "op-system-structural-model"
        or type(source["schema_version"]) is not int
        or source["schema_version"] != 1
    ):
        msg = "Unsupported typed structural DAG schema"
        raise ValueError(msg)
    nodes = source["nodes"]
    root = source["root"]
    if (
        not isinstance(nodes, list)
        or type(root) is not int
        or not 0 <= root < len(nodes)
    ):
        msg = "Typed DAG requires a valid node table and root"
        raise ValueError(msg)
    decoded: list[object] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, list) or len(node) != 3:
            msg = "Malformed typed DAG node"
            raise ValueError(msg)
        tag, labels, references = node
        if not isinstance(references, list) or any(
            type(reference) is not int or not 0 <= reference < index
            for reference in references
        ):
            msg = "Typed DAG references must point to earlier nodes"
            raise ValueError(msg)
        children = tuple(decoded[reference] for reference in references)
        decoded.append(_value(tag, labels, children))
    result = decoded[root]
    if not isinstance(result, Mapping) or any(
        not isinstance(name, str) for name in result
    ):
        msg = "Typed model DAG must decode to named fields"
        raise ValueError(msg)
    return result


def _value(  # ruff: ignore[complex-structure, too-many-return-statements]
    tag: object, labels: object, children: tuple[object, ...]
) -> object:
    if tag == "sequence" and labels is None:
        return children
    if tag == "mapping":
        names = _names(labels, len(children))
        return MappingProxyType(dict(zip(names, children, strict=True)))
    if tag == "record":
        if (
            not isinstance(labels, list)
            or len(labels) != 2
            or not isinstance(labels[0], str)
        ):
            msg = "Typed record requires a type and field names"
            raise ValueError(msg)
        names = _names(labels[1], len(children))
        return IRRecord(labels[0], tuple(zip(names, children, strict=True)))
    if children:
        msg = "Typed scalar nodes cannot have children"
        raise ValueError(msg)
    if tag == "none" and labels is None:
        return None
    if tag == "bool" and type(labels) is bool:
        return labels
    if tag == "str" and isinstance(labels, str):
        return labels
    if tag == "int" and isinstance(labels, str):
        return int(labels)
    if tag == "float" and isinstance(labels, str):
        return float.fromhex(labels)
    msg = f"Unsupported typed DAG scalar tag: {tag!r}"
    raise ValueError(msg)


def _names(labels: object, count: int) -> tuple[str, ...]:
    if (
        not isinstance(labels, list)
        or len(labels) != count
        or any(not isinstance(name, str) for name in labels)
        or len(set(labels)) != count
    ):
        msg = "Typed field names must be unique and match child count"
        raise ValueError(msg)
    return tuple(labels)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for name, value in pairs:
        if name in result:
            msg = "Portable IR JSON must not contain duplicate object keys"
            raise ValueError(msg)
        result[name] = value
    return result


def _load_json(text: str) -> object:
    return json.loads(text, object_pairs_hook=_unique_object)
