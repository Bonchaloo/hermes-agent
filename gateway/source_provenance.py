"""Process-local capabilities for authentic inbound ``SessionSource`` objects.

The registry deliberately keys by object identity instead of equality.  A source
copy therefore cannot inherit transport trust.  The immutable fingerprint also
invalidates the exact object if any source field changes after registration.
Nothing in this module is serialized.
"""

from __future__ import annotations

import dataclasses
import weakref
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return (type(value).__qualname__, _freeze(enum_value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def source_auth_fingerprint(source: Any) -> tuple[Any, ...]:
    """Return an immutable snapshot of every declared source field."""
    if not dataclasses.is_dataclass(source):
        raise TypeError("source must be a dataclass instance")
    return tuple(
        (field.name, _freeze(getattr(source, field.name)))
        for field in dataclasses.fields(source)
    )


class SourceProvenanceRegistry:
    """Weak, exact-identity source registry with optional connection epoch."""

    def __init__(self) -> None:
        self._records: dict[
            int, tuple[weakref.ReferenceType[Any], tuple[Any, ...], str | None]
        ] = {}

    def register(self, source: Any, *, epoch: str | None = None) -> None:
        source_id = id(source)

        def discard(reference: weakref.ReferenceType[Any]) -> None:
            current = self._records.get(source_id)
            if current is not None and current[0] is reference:
                self._records.pop(source_id, None)

        reference = weakref.ref(source, discard)
        self._records[source_id] = (
            reference,
            source_auth_fingerprint(source),
            epoch,
        )

    def verifies(self, source: Any, *, epoch: str | None = None) -> bool:
        record = self._records.get(id(source))
        if record is None:
            return False
        reference, fingerprint, recorded_epoch = record
        if reference() is not source or recorded_epoch != epoch:
            return False
        try:
            return fingerprint == source_auth_fingerprint(source)
        except (AttributeError, TypeError, ValueError):
            return False

    def clear(self) -> None:
        self._records.clear()
