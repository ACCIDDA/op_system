"""Validated identifier string type for op_system."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator


def _validate_identifier_string(value: str) -> str:
    """
    Validate an identifier-like string.

    Identifier strings must be non-empty, contain only alphanumeric characters, and
    start with a letter. Leading and trailing whitespace is stripped before validation.

    Args:
        value: Candidate identifier string.

    Returns:
        The validated identifier string.

    Raises:
        TypeError: If `value` is not a string.
        ValueError: If `value` is empty.
        ValueError: If `value` contains non-alphanumeric characters or does not start
            with a letter.

    Examples:
        >>> _validate_identifier_string("S")
        'S'
        >>> _validate_identifier_string("Foobar")
        'Foobar'
        >>> _validate_identifier_string(123)
        Traceback (most recent call last):
            ...
        TypeError: IdentifierString must be a string.
        >>> _validate_identifier_string("")
        Traceback (most recent call last):
            ...
        ValueError: IdentifierString must not be empty.
        >>> _validate_identifier_string("   ")
        Traceback (most recent call last):
            ...
        ValueError: IdentifierString must not be empty.
        >>> _validate_identifier_string("123abc")
        Traceback (most recent call last):
            ...
        ValueError: IdentifierString must contain only alphanumerical characters and start with a letter.
        >>> _validate_identifier_string("abc-123")
        Traceback (most recent call last):
            ...
        ValueError: IdentifierString must contain only alphanumerical characters and start with a letter.
    """  # noqa: E501
    if not isinstance(value, str):
        msg = "IdentifierString must be a string."
        raise TypeError(msg)
    value = value.strip()
    if not value:
        msg = "IdentifierString must not be empty."
        raise ValueError(msg)
    if not value[0].isalpha() or not value.isalnum():
        msg = (
            "IdentifierString must contain only alphanumerical "
            "characters and start with a letter."
        )
        raise ValueError(msg)
    return value


IdentifierString = Annotated[str, AfterValidator(_validate_identifier_string)]
"""
Custom `pydantic` type for validated identifier strings used in `op_system`.

Identifier strings are used for state names, dimension names, and other keys in the
system. They must be non-empty, contain only alphanumeric characters, and start with a
letter. Leading and trailing whitespace is stripped before validation.

Examples:
    >>> from pydantic import BaseModel
    >>> from op_system import IdentifierString
    >>> class ExampleModel(BaseModel):
    ...     identifier: IdentifierString
    ...
    >>> ExampleModel(identifier="S")
    ExampleModel(identifier='S')
    >>> ExampleModel(identifier="  Foobar  ")
    ExampleModel(identifier='Foobar')
    >>> ExampleModel(identifier="123abc")
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for ExampleModel
    identifier
    Value error, IdentifierString must contain only alphanumerical characters and start with a letter. [...]
        For further information visit ...
    >>> ExampleModel(identifier="")
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for ExampleModel
    identifier
    Value error, IdentifierString must not be empty. [...]
        For further information visit ...
"""  # noqa: E501
