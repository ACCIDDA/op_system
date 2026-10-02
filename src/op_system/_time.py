"""Validation and metadata for time-indexed parameter interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from op_system._errors import InvalidRhsSpecError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _normalize_time_metadata(
    spec: Mapping[str, object],
    *,
    time_axis_name: str,
    axes_meta: Sequence[Mapping[str, object]],
    time_varying_params: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    """Validate the interpolation policy and snapshot active forcing times.

    Returns:
        Policy, immutable time coordinates, and hold-mode forcing boundaries.

    Raises:
        InvalidRhsSpecError: If the policy or active time coordinates are invalid.
    """
    policy = spec.get("time_interpolation", "linear")
    if not isinstance(policy, str) or policy not in {"linear", "previous"}:
        raise InvalidRhsSpecError(
            detail="time_interpolation must be 'linear' or 'previous'"
        )
    coordinates: tuple[float, ...] = ()
    if time_varying_params:
        raw_coords = next(
            (axis["coords"] for axis in axes_meta if axis["name"] == time_axis_name),
            (),
        )
        try:
            times = np.asarray(raw_coords, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise InvalidRhsSpecError(
                detail="time-axis coordinates must be finite real times"
            ) from error
        if (
            times.ndim != 1
            or times.size == 0
            or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0)
        ):
            raise InvalidRhsSpecError(
                detail="time-axis coordinates must be finite and strictly increasing"
            )
        coordinates = tuple(float(time) for time in times)
    return {
        "time_interpolation": policy,
        "time_coordinates": coordinates,
        "forcing_breakpoints": coordinates[1:] if policy == "previous" else (),
    }
