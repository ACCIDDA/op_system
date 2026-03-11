"""State-string parsing and serialization for op_system."""

from __future__ import annotations

import re
from typing import Any, Final, TypedDict

from pydantic import BaseModel, model_serializer, model_validator

from op_system._identifer_string import IdentifierString  # noqa: TC001

_STATE_STRING_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9]*)(?:\[(?P<dims>.*)\])?$"
)


class StateStringDict(TypedDict):
    """
    Structured mapping for `StateString` model fields.

    Light typing aid for the `_parse_string` method. Attributes mirror the fields of
    `StateString`, but are not validated or stripped.

    Attributes:
        name: State name.
        dims: State dimensions.
    """

    name: str
    dims: tuple[str, ...]


class StateString(BaseModel, frozen=True, str_strip_whitespace=False):
    """
    Structured representation of a state string.

    A state string is either a bare state name like `"S"` or a state name
    followed immediately by bracketed dimensions like `"R[age,vax]"`.

    Examples:
        >>> StateString.model_validate("S")
        StateString(name='S', dims=())
        >>> recovery = StateString.model_validate("R[age,vax]")
        >>> recovery
        StateString(name='R', dims=('age', 'vax'))
        >>> print(recovery)
        R[age,vax]
        >>> recovery.model_dump()
        'R[age,vax]'
        >>> StateString.model_validate("Foobar[ age , vax ]")
        StateString(name='Foobar', dims=('age', 'vax'))
    """

    name: IdentifierString
    dims: tuple[IdentifierString, ...]

    @model_validator(mode="before")
    @classmethod
    def _parse_state_string(cls, value: Any) -> Any:  # noqa: ANN401
        """
        Parse compact state-string input before model validation.

        Args:
            value: Raw value passed to the model.

        Returns:
            Mapping-like data suitable for normal field validation.

        Raises:
            TypeError: If `value` is of an unsupported type.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls._parse_string(value)
        if isinstance(value, dict):
            return value
        msg = "StateString must be validated from a string or mapping."
        raise TypeError(msg)

    @classmethod
    def _parse_string(cls, value: str) -> StateStringDict:
        """
        Parse a compact state string into model fields.

        Args:
            value: Compact state string.

        Returns:
            Parsed model field mapping.

        Raises:
            ValueError: If the compact form is invalid.

        Examples:
            >>> StateString._parse_string("S")
            {'name': 'S', 'dims': ()}
            >>> StateString._parse_string("R[age,vax]")
            {'name': 'R', 'dims': ('age', 'vax')}
            >>> StateString._parse_string("Foobar[ age , vax ]")
            {'name': 'Foobar', 'dims': (' age ', ' vax ')}
            >>> StateString._parse_string("Fizz[dim]")
            {'name': 'Fizz', 'dims': ('dim',)}
            >>> StateString._parse_string("lambda[dim1, dim2, dim3]")
            {'name': 'lambda', 'dims': ('dim1', ' dim2', ' dim3')}
            >>> StateString._parse_string("Invalid[dim")
            Traceback (most recent call last):
                ...
            ValueError: Invalid state string. Expected 'Name' or 'Name[dim1,dim2]' with no whitespace before '['.
            >>> StateString._parse_string("123Invalid[dim]")
            Traceback (most recent call last):
                ...
            ValueError: Invalid state string. Expected 'Name' or 'Name[dim1,dim2]' with no whitespace before '['.
        """  # noqa: E501
        stripped_value = value.strip()
        match = _STATE_STRING_RE.fullmatch(stripped_value)
        if match is None:
            msg = (
                "Invalid state string. Expected 'Name' or 'Name[dim1,dim2]' "
                "with no whitespace before '['."
            )
            raise ValueError(msg)
        dims_group = match.group("dims")
        return {
            "name": match.group("name"),
            "dims": tuple(dim for dim in (dims_group).split(","))
            if (dims_group := match.group("dims"))
            else (),
        }

    def __str__(self) -> str:
        """
        Return the compact string form.

        Returns:
            Compact state string.

        Examples:
            >>> str(StateString(name="S", dims=()))
            'S'
            >>> str(StateString(name="R", dims=("age",)))
            'R[age]'
            >>> str(StateString(name="R", dims=("age", "vax")))
            'R[age,vax]'
            >>> str(StateString(name="lambda", dims=("age", "vax", "state")))
            'lambda[age,vax,state]'
        """
        if not self.dims:
            return self.name
        dims = ",".join(self.dims)
        return f"{self.name}[{dims}]"

    @model_serializer(mode="plain")
    def _serialize(self) -> str:
        """
        Serialize the model to its compact string form.

        Returns:
            Compact state string.
        """
        return str(self)
