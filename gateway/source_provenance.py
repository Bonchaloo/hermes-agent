"""Process-local capabilities for authentic inbound ``SessionSource`` objects.

The registry deliberately keys by object identity instead of equality.  A source
copy therefore cannot inherit transport trust.  The immutable fingerprint also
invalidates the exact object if any source field changes after registration.
Nothing in this module is serialized.
"""

from __future__ import annotations

import dataclasses
import weakref
from enum import Enum
from typing import Any


def _freeze(value: Any) -> Any:
    concrete_type = type(value)
    type_tag = (concrete_type,)
    if concrete_type is dict:
        return (
            *type_tag,
            frozenset((_freeze(key), _freeze(item)) for key, item in value.items()),
        )
    if concrete_type is list:
        return (*type_tag, tuple(_freeze(item) for item in value))
    if concrete_type is tuple:
        return (*type_tag, tuple(_freeze(item) for item in value))
    if concrete_type in (set, frozenset):
        return (*type_tag, frozenset(_freeze(item) for item in value))
    if isinstance(value, Enum):
        return (*type_tag, _freeze(value.value))
    if concrete_type is float:
        return (*type_tag, value.hex())
    if value is None or concrete_type in (bool, int, str, bytes):
        return (*type_tag, value)
    # SessionSource fields are expected to use only the primitives and
    # containers above.  Never invoke arbitrary equality, ``repr``, or a
    # synthetic ``.value`` attribute on an unexpected object (for example,
    # MagicMock dynamically creates one).  Reject it so registration fails
    # closed instead of granting trust to an incomplete snapshot.
    raise TypeError(f"unsupported source fingerprint type: {concrete_type.__qualname__}")


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

        try:
            fingerprint = source_auth_fingerprint(source)
        except Exception:
            # Registration is a trust boundary.  A malformed field must leave
            # no usable capability, including one recorded by an earlier call.
            self._records.pop(source_id, None)
            return

        def discard(reference: weakref.ReferenceType[Any]) -> None:
            current = self._records.get(source_id)
            if current is not None and current[0] is reference:
                self._records.pop(source_id, None)

        reference = weakref.ref(source, discard)
        self._records[source_id] = (
            reference,
            fingerprint,
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
        except Exception:
            return False

    def clear(self) -> None:
        self._records.clear()
