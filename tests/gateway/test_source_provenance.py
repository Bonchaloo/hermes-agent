"""Regression tests for process-local source provenance fingerprints."""

from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock

from gateway.source_provenance import (
    SourceProvenanceRegistry,
    source_auth_fingerprint,
)


@dataclass
class SourceFixture:
    value: object


class SourceKind(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class OtherSourceKind(Enum):
    PRIMARY = "primary"


class DynamicValue:
    def __getattr__(self, name: str):
        if name == "value":
            return self
        raise AttributeError(name)


class ExplodingRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr must not be called")


def test_non_enum_dynamic_values_fail_closed_without_recursion():
    for dynamic_value in (DynamicValue(), MagicMock(name="dynamic-value")):
        assert not isinstance(dynamic_value, Enum)

        source = SourceFixture(value=dynamic_value)
        registry = SourceProvenanceRegistry()

        registry.register(source)

        assert registry.verifies(source) is False


def test_malformed_post_registration_value_fails_closed_without_repr():
    source = SourceFixture("owner")
    registry = SourceProvenanceRegistry()
    registry.register(source)

    source.value = ExplodingRepr()

    assert registry.verifies(source) is False


def test_signed_float_change_invalidates_registration():
    source = SourceFixture(0.0)
    registry = SourceProvenanceRegistry()
    registry.register(source)

    source.value = -0.0

    assert registry.verifies(source) is False


def test_enum_fingerprint_preserves_enum_type_and_value():
    assert source_auth_fingerprint(
        SourceFixture(SourceKind.PRIMARY)
    ) != source_auth_fingerprint(
        SourceFixture(OtherSourceKind.PRIMARY)
    )

    source = SourceFixture(SourceKind.PRIMARY)
    registry = SourceProvenanceRegistry()
    registry.register(source)

    source.value = SourceKind.SECONDARY

    assert registry.verifies(source) is False


def test_enum_fingerprint_preserves_concrete_class_identity():
    first = Enum(
        "Platform",
        {"DISCORD": "discord"},
        module="gateway.config",
        qualname="Platform",
    )
    second = Enum(
        "Platform",
        {"DISCORD": "discord"},
        module="gateway.config",
        qualname="Platform",
    )
    source = SourceFixture(first.DISCORD)
    registry = SourceProvenanceRegistry()
    registry.register(source)

    source.value = second.DISCORD

    assert registry.verifies(source) is False


def test_supported_builtin_subclass_fails_closed_without_dynamic_equality():
    equality_called = False

    class HostileString(str):
        def __eq__(self, other):
            nonlocal equality_called
            equality_called = True
            raise RuntimeError("dynamic equality must not run")

        __hash__ = str.__hash__

    source = SourceFixture(HostileString("owner"))
    registry = SourceProvenanceRegistry()

    registry.register(source)

    assert registry.verifies(source) is False
    assert equality_called is False


def test_scalar_type_change_invalidates_registration():
    source = SourceFixture(True)
    registry = SourceProvenanceRegistry()
    registry.register(source)

    source.value = 1

    assert registry.verifies(source) is False


def test_container_type_change_invalidates_registration():
    source = SourceFixture(["owner"])
    registry = SourceProvenanceRegistry()
    registry.register(source)

    source.value = ("owner",)

    assert registry.verifies(source) is False
