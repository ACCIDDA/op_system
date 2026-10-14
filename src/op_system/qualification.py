"""Opt-in, eager numerical qualification of compiled RHS functions.

This module checks user-supplied fixtures against independent equations and
optional balance and JAX directional-derivative checks. It does not replace
structural validation or establish scientific validity or integration accuracy.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np

from op_system._array import array_namespace

if TYPE_CHECKING:
    from op_system import Array, CompiledRhs
    from op_system.compile import StateDict

Evaluator = Literal["flat", "structured", "block"]
ReferenceRhs = Callable[
    [float, Mapping[str, "Array"], Mapping[str, object]], Mapping[str, "Array"]
]
BalanceCheck = Callable[
    [float, Mapping[str, "Array"], Mapping[str, "Array"], Mapping[str, object]], "Array"
]


class QualificationFixture(NamedTuple):
    """Named structured state/parameter fixture and optional state direction."""

    name: str
    state: Mapping[str, Array]
    params: Mapping[str, object]
    time: float = 0.0
    direction: Mapping[str, Array] | None = None


class InvariantCheck(NamedTuple):
    """A residual-to-zero balance check with explicit applicability conditions."""

    name: str
    callback: BalanceCheck
    condition: str
    atol: float = 1e-8


class NumericalCheck(NamedTuple):
    """A named numerical check and any informative failure messages."""

    name: str
    passed: bool
    max_abs_error: float | None
    errors: tuple[str, ...] = ()
    condition: str | None = None
    rtol: float = 0.0
    atol: float = 0.0
    step: float | None = None


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    """Reference and derivative tolerances for eager qualification.

    Defaults assume double precision. Choose wider tolerances and a larger
    finite-difference step for float32 fixtures.
    """

    rtol: float = 1e-7
    atol: float = 1e-9
    derivative_step: float = 1e-5
    derivative_rtol: float = 1e-4
    derivative_atol: float = 1e-6

    def __post_init__(self) -> None:
        """Validate controls.

        Raises:
            ValueError: If a tolerance or finite-difference step is invalid.
        """
        for name in (
            "rtol",
            "atol",
            "derivative_rtol",
            "derivative_atol",
            "derivative_step",
        ):
            value = getattr(self, name)
            numeric = isinstance(value, Real) and not isinstance(value, bool)
            if (
                not numeric
                or not math.isfinite(value)
                or value < 0
                or (name == "derivative_step" and value == 0)
            ):
                bound = "positive" if name == "derivative_step" else "non-negative"
                msg = f"{name} must be finite and {bound}"
                raise ValueError(msg)


class ArrayComparison(NamedTuple):
    """Reference parity, shape/dtype preservation, and finite-value diagnostics."""

    state: str
    shape: tuple[int, ...]
    reference_shape: tuple[int, ...]
    input_dtype: str
    output_dtype: str
    reference_dtype: str
    finite: bool
    passed: bool
    max_abs_error: float | None
    max_relative_error: float | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FixtureQualification:
    """Qualification of one named fixture through one evaluator.

    All diagnostics are host values: this report is an eager authoring tool,
    even when the evaluated RHS uses JAX or Torch arrays.
    """

    fixture: str
    evaluator: Evaluator
    backend: str
    time: float
    rtol: float
    atol: float
    arrays: tuple[ArrayComparison, ...] = ()
    invariants: tuple[NumericalCheck, ...] = ()
    derivative: NumericalCheck | None = None
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every requested check passed."""
        return (
            not self.errors
            and bool(self.arrays)
            and all(item.passed for item in self.arrays)
            and all(item.passed for item in self.invariants)
            and (self.derivative is None or self.derivative.passed)
        )


@dataclass(frozen=True, slots=True)
class QualificationReport:
    """Inspectable results for all requested fixture/evaluator combinations."""

    fixtures: tuple[FixtureQualification, ...]

    @property
    def ok(self) -> bool:
        """Whether all fixtures and requested evaluators qualified."""
        return bool(self.fixtures) and all(item.ok for item in self.fixtures)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible fields, including per-fixture pass/fail status.

        Returns:
            A mapping suitable for ``json.dumps(..., allow_nan=False)``.
        """
        return {
            "ok": self.ok,
            "fixtures": [
                {
                    **asdict(item),
                    "ok": item.ok,
                    "arrays": [value._asdict() for value in item.arrays],
                    "invariants": [value._asdict() for value in item.invariants],
                    "derivative": item.derivative._asdict()
                    if item.derivative
                    else None,
                }
                for item in self.fixtures
            ],
        }


