"""Validated parameter declarations independent of numerical execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

from op_system._errors import InvalidRhsSpecError

if TYPE_CHECKING:
    from collections.abc import Collection

_SupportKind = Literal["real", "positive", "unit_interval", "bounded"]


class ParameterSupport(NamedTuple):
    """Declared domain; bounds are inclusive and positivity is strict.

    Attributes:
        kind: Domain identifier; unit_interval includes zero and one.
        lower: Finite lower bound for bounded support, otherwise None.
        upper: Finite upper bound for bounded support, otherwise None.
    """

    kind: _SupportKind
    lower: float | None = None
    upper: float | None = None

    @classmethod
    def from_mapping(cls, value: object) -> ParameterSupport:
        """Parse a domain name or bounded-domain mapping.

        Args:
            value: Domain string or mapping with kind, lower and upper.

        Returns:
            Immutable support declaration.

        Raises:
            InvalidRhsSpecError: If the domain or bounds are invalid.
        """
        if isinstance(value, str) and value in {"real", "positive", "unit_interval"}:
            return cls(cast("_SupportKind", value))
        if not isinstance(value, Mapping) or set(value) != {"kind", "lower", "upper"}:
            raise InvalidRhsSpecError(
                detail=(
                    "parameter support must be real, positive, unit_interval, "
                    "or a bounded mapping with kind, lower and upper"
                )
            )
        lower = _finite_bound(value["lower"])
        upper = _finite_bound(value["upper"])
        if value["kind"] != "bounded" or lower >= upper:
            raise InvalidRhsSpecError(
                detail="bounded parameter support requires finite numeric lower < upper"
            )
        return cls("bounded", float(lower), float(upper))


class ParameterDeclaration(NamedTuple):
    """Optional scientific metadata for an inferred runtime parameter.

    Attributes:
        support: Declared domain, or None when it was not specified.
        units: Unit annotation, without dimensional-analysis verification.
        description: Optional human-readable parameter description.
        fixed: Whether inference should treat the parameter as fixed. This
            does not bind a value or remove a runtime parameter.
    """

    support: ParameterSupport | None = None
    units: str | None = None
    description: str | None = None
    fixed: bool = False

    @classmethod
    def from_mapping(cls, value: object) -> ParameterDeclaration:
        """Validate and copy one parameter declaration.

        Args:
            value: Mapping with optional support, units, description and fixed.

        Returns:
            Immutable parameter declaration.

        Raises:
            InvalidRhsSpecError: If an annotation is malformed or unknown.
        """
        if not isinstance(value, Mapping) or any(
            key not in {"support", "units", "description", "fixed"} for key in value
        ):
            raise InvalidRhsSpecError(
                detail=(
                    "parameter declaration must contain only support, units, "
                    "description and fixed"
                )
            )
        fixed = value.get("fixed", False)
        if not isinstance(fixed, bool):
            raise InvalidRhsSpecError(detail="parameter fixed must be a boolean")
        support = (
            ParameterSupport.from_mapping(value["support"])
            if "support" in value
            else None
        )
        return cls(
            support,
            _optional_text(value, "units"),
            _optional_text(value, "description"),
            fixed,
        )


def _optional_text(value: Mapping[str, object], name: str) -> str | None:
    """Read a nonempty optional annotation.

    Returns:
        Supplied text, or None when no annotation was supplied.

    Raises:
        InvalidRhsSpecError: If a supplied annotation is not nonempty text.
    """
    if name not in value:
        return None
    text = value[name]
    if not isinstance(text, str) or not text.strip():
        raise InvalidRhsSpecError(detail=f"parameter {name} must be nonempty text")
    return text


def _normalize_parameter_declarations(
    value: object, *, known_names: Collection[str]
) -> dict[str, ParameterDeclaration]:
    """Validate declarations against scalar, shaped and time-varying names.

    Returns:
        Copied declarations keyed by inferred runtime name.

    Raises:
        InvalidRhsSpecError: If names or declaration values are invalid.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvalidRhsSpecError(detail="parameters must be a declaration mapping")
    declarations = {}
    for name, declaration in value.items():
        if not isinstance(name, str) or name not in known_names:
            raise InvalidRhsSpecError(
                detail=(
                    f"parameter declaration {name!r} is not an inferred "
                    "runtime parameter"
                )
            )
        declarations[name] = ParameterDeclaration.from_mapping(declaration)
    return declarations


def _finite_bound(value: object) -> float:
    """Parse a finite representable bound.

    Returns:
        Bound as a float.

    Raises:
        InvalidRhsSpecError: If the bound is not a finite real number.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidRhsSpecError(detail="parameter bounds must be finite numbers")
    try:
        bound = float(value)
    except OverflowError as exc:
        raise InvalidRhsSpecError(
            detail="parameter bounds must be finite numbers"
        ) from exc
    if not math.isfinite(bound):
        raise InvalidRhsSpecError(detail="parameter bounds must be finite numbers")
    return bound


def _parameter_declaration_metadata(
    value: object, *, known_names: Collection[str]
) -> dict[str, object]:
    """Build optional normalized metadata without changing undeclared specs.

    Returns:
        Parameter metadata when supplied, otherwise an empty mapping.

    """
    if value is None:
        return {}
    return {
        "parameter_declarations": _normalize_parameter_declarations(
            value, known_names=known_names
        )
    }
