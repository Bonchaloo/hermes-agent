"""Tests for relay upstream-enforced authorization at the gateway layer.

Background: the relay adapter fronts the Team Gateway connector over a
per-instance-authenticated WebSocket. The connector performs owner-only
author-binding resolution BEFORE delivering an inbound event — a message only
reaches this gateway because the connector resolved it to THIS instance's bound
user (``user_instance_binding``, keyed on the connector-observed author id,
never a gateway claim). So a relay inbound is already authorized by a trusted,
authenticated upstream.

Before this fix, ``_is_user_authorized`` had no notion of upstream
authorization: ``Platform.RELAY`` matched no ``*_ALLOWED_USERS`` allowlist and
isn't in the HA/WEBHOOK always-authorized set, so every relay user hit the
default-deny ("No user allowlists configured. All unauthorized users will be
denied.") and the agent never saw the message. This was the live staging bug:
the message routed correctly through the connector to the instance, then the
instance's authz layer dropped it as ``Unauthorized user``.

The fix adds a generic ``BasePlatformAdapter.authorization_is_upstream``
capability (default ``False``) that the relay adapter overrides to ``True``,
plus a dedicated trusted branch in ``_is_user_authorized``. It is delegation to
a trusted upstream, NOT a fail-open: it fires only for an adapter that
explicitly declares the flag; every direct network-exposed adapter leaves it
``False`` and the env-allowlist default-deny is unchanged.
"""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "DISCORD_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_runner(
    *,
    platform: Platform,
    authorization_is_upstream: bool,
    transport=None,
):
    """Build a bare GatewayRunner with one adapter for *platform*.

    ``authorization_is_upstream`` controls whether that adapter declares the
    upstream-authz capability.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    if transport is not None and not hasattr(
        transport, "authenticated_connection_epoch"
    ):
        from gateway.source_provenance import SourceProvenanceRegistry

        transport.authenticated_connection_epoch = "authenticated-test-epoch"
        transport._source_provenance = SourceProvenanceRegistry()
        transport.gateway_id = "test-gateway"
        transport.upgrade_secret = "test-secret"
    adapter = SimpleNamespace(
        send=AsyncMock(),
        authorization_is_upstream=authorization_is_upstream,
        enforces_own_access_policy=False,
        _transport=transport,
    )
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    return runner, adapter


def _relay_source(**kw) -> SessionSource:
    base = dict(
        platform=Platform.RELAY,
        user_id="428014785045725184",
        chat_id="1400724139874058314",
        user_name="definitely_not_cthulhu",
        chat_type="group",
    )
    base.update(kw)
    return SessionSource(**base)


# ---------------------------------------------------------------------------
# Capability contract
# ---------------------------------------------------------------------------


def test_base_adapter_defaults_to_not_upstream_authorized():
    """The base property is False — direct adapters keep env default-deny."""
    from gateway.platforms.base import BasePlatformAdapter

    assert BasePlatformAdapter.authorization_is_upstream.fget(object()) is False


# ---------------------------------------------------------------------------
# Authorization behavior
# ---------------------------------------------------------------------------


def test_non_upstream_adapter_still_default_denies(monkeypatch):
    """A direct adapter that does NOT declare the flag still default-denies.

    Guards against the fix becoming a blanket fail-open: an adapter with
    authorization_is_upstream=False and no env allowlist must remain denied.
    """
    _clear_auth_env(monkeypatch)
    runner, _ = _make_runner(platform=Platform.DISCORD, authorization_is_upstream=False)
    src = SessionSource(
        platform=Platform.DISCORD,
        user_id="123",
        chat_id="456",
        user_name="someone",
        chat_type="dm",
    )
    assert runner._is_user_authorized(src) is False


def test_native_source_with_nonexistent_profile_fails_closed(monkeypatch):
    """A native adapter cannot route an unknown named profile into base scope."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    runner, _ = _make_runner(
        platform=Platform.TELEGRAM,
        authorization_is_upstream=False,
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-a",
        chat_type="group",
        user_id="owner-a",
        profile="does-not-exist",
    )

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: False)
    assert runner._is_user_authorized(source) is False


# ---------------------------------------------------------------------------
# The underlying-platform regression: a relay *message* inbound carries the
# UNDERLYING platform (source.platform == Platform.DISCORD), not Platform.RELAY,
# because the connector's wire payload sets platform="discord" and
# ws_transport._event_from_wire maps it straight onto SessionSource. The relay
# adapter is registered ONLY under Platform.RELAY, so keying upstream-authz off
# source.platform misses and the user hits default-deny ("Unauthorized user
# <id> (<name>) on discord"). Authorization must be bound to the exact relay
# transport that received the event, not a caller-constructible marker.
# ---------------------------------------------------------------------------


def test_relay_message_with_underlying_discord_platform_authorized(monkeypatch):
    """A Discord source received by the registered relay transport is trusted."""
    from gateway.relay.ws_transport import _event_from_wire

    _clear_auth_env(monkeypatch)
    transport = SimpleNamespace()
    runner, _ = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
        transport=transport,
    )
    event = _event_from_wire(
        {
            "text": "hello",
            "source": {
                "platform": "discord",
                "user_id": "267171776755269633",
                "chat_id": "1400724139874058314",
                "user_name": "rewbs",
                "chat_type": "dm",
            },
        },
        transport=transport,
    )
    assert runner._is_user_authorized(event.source) is True


def test_caller_constructed_relay_marker_does_not_authorize(monkeypatch):
    _clear_auth_env(monkeypatch)
    runner, _ = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
    )
    forged = SessionSource(
        platform=Platform.DISCORD,
        chat_id="forged-chat",
        delivered_via_upstream_relay=True,
    )
    marker = getattr(forged, "_mark_authenticated_relay_delivery", None)
    if callable(marker):
        marker()

    assert runner._is_user_authorized(forged) is False


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_copied_native_discord_source_loses_transport_authorization(
    monkeypatch,
    copier,
):
    """Transport trust belongs to one exact source object and immutable fields."""
    from gateway.config import GatewayConfig, PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter

    _clear_auth_env(monkeypatch)
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"allow_all_users": True, "allowed_channels": "*"},
        )
    )
    runner, _ = _make_runner(
        platform=Platform.DISCORD,
        authorization_is_upstream=False,
    )
    runner.config = GatewayConfig()
    runner.adapters = {Platform.DISCORD: adapter}
    original = adapter.build_source(
        chat_id="channel-a",
        chat_type="group",
        user_id="owner-a",
        guild_id="guild-a",
    )
    assert runner._is_user_authorized(original) is True

    forged = copier(original)
    forged.chat_id = "channel-b"
    forged.user_id = "attacker-b"
    forged.profile = None

    assert runner._is_user_authorized(forged) is False


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_copied_relay_source_loses_authenticated_connection_authorization(
    monkeypatch,
    copier,
):
    """A source copy cannot retain a live relay connection capability."""
    from gateway.relay.ws_transport import _event_from_wire

    _clear_auth_env(monkeypatch)
    transport = SimpleNamespace()
    runner, _ = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
        transport=transport,
    )
    original = _event_from_wire(
        {
            "text": "hello",
            "source": {
                "platform": "discord",
                "chat_id": "channel-a",
                "chat_type": "group",
                "user_id": "owner-a",
                "scope_id": "guild-a",
            },
        },
        transport=transport,
    ).source
    assert runner._is_user_authorized(original) is True

    forged = copier(original)
    forged.chat_id = "channel-b"
    forged.user_id = "attacker-b"

    assert runner._is_user_authorized(forged) is False


def test_direct_mutation_of_registered_relay_source_revokes_authorization(monkeypatch):
    from gateway.relay.ws_transport import _event_from_wire

    _clear_auth_env(monkeypatch)
    transport = SimpleNamespace()
    runner, _ = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
        transport=transport,
    )
    source = _event_from_wire(
        {
            "text": "hello",
            "source": {
                "platform": "discord",
                "chat_id": "channel-a",
                "chat_type": "group",
                "user_id": "owner-a",
                "scope_id": "guild-a",
            },
        },
        transport=transport,
    ).source
    assert runner._is_user_authorized(source) is True

    source.chat_id = "channel-mutated"
    source.user_id = "attacker-mutated"

    assert runner._is_user_authorized(source) is False


def test_callable_upstream_marker_cannot_forge_relay_authorization(monkeypatch):
    from gateway.relay.ws_transport import _event_from_wire

    _clear_auth_env(monkeypatch)
    transport = SimpleNamespace()
    runner, adapter = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
        transport=transport,
    )
    event = _event_from_wire(
        {
            "text": "hello",
            "source": {
                "platform": "discord",
                "chat_id": "channel-a",
                "chat_type": "group",
                "user_id": "owner-a",
                "scope_id": "guild-a",
            },
        },
        transport=transport,
    )
    adapter.authorization_is_upstream = lambda: True

    assert runner._is_user_authorized(event.source) is False


def test_gateway_exposes_no_generic_source_provenance_stamper():
    from gateway.run import GatewayRunner

    assert not hasattr(GatewayRunner, "_source_with_transport_provenance")


def test_relay_without_upgrade_credentials_never_authorizes_upstream(monkeypatch):
    """A descriptor on an unauthenticated development socket is not auth trust."""
    from gateway.relay.ws_transport import _event_from_wire

    _clear_auth_env(monkeypatch)
    transport = SimpleNamespace(
        gateway_id=None,
        upgrade_secret=None,
        authenticated_connection_epoch=None,
    )
    runner, _ = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
        transport=transport,
    )
    source = _event_from_wire(
        {
            "text": "hello",
            "source": {
                "platform": "discord",
                "chat_id": "channel-a",
                "chat_type": "group",
                "user_id": "owner-a",
                "scope_id": "guild-a",
            },
        },
        transport=transport,
    ).source

    assert runner._is_user_authorized(source) is False


def test_relay_discord_ignored_channel_denies_before_upstream_grant(monkeypatch):
    from gateway.relay.ws_transport import _event_from_wire

    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "blocked-channel")
    transport = SimpleNamespace()
    runner, _ = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
        transport=transport,
    )
    event = _event_from_wire(
        {
            "text": "hello",
            "source": {
                "platform": "discord",
                "chat_id": "blocked-channel",
                "chat_type": "group",
                "user_id": "267171776755269633",
                "scope_id": "guild-1",
            },
        },
        transport=transport,
    )

    assert runner._is_user_authorized(event.source) is False


def test_relay_discord_omitted_chat_type_fails_closed(monkeypatch):
    from gateway.relay.ws_transport import _event_from_wire

    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "#blocked-room")
    transport = SimpleNamespace()
    runner, _ = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
        transport=transport,
    )
    with pytest.raises(ValueError):
        _event_from_wire(
            {
                "text": "hello",
                "source": {
                    "platform": "discord",
                    "chat_id": "123456789",
                    "chat_name": "Guild / #blocked-room",
                    "user_id": "267171776755269633",
                    "scope_id": "guild-1",
                },
            },
            transport=transport,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", 123),
        ("profile", "definitely-missing-profile"),
        ("chat_topic", {"unexpected": "mapping"}),
    ],
)
def test_relay_discord_malformed_wire_metadata_fails_closed(
    monkeypatch,
    field,
    value,
):
    from gateway.relay.ws_transport import _event_from_wire

    _clear_auth_env(monkeypatch)
    transport = SimpleNamespace()
    runner, _ = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
        transport=transport,
    )
    wire_source = {
        "platform": "discord",
        "chat_id": "channel-1",
        "chat_type": "group",
        "user_id": "267171776755269633",
        "scope_id": "guild-1",
        field: value,
    }
    if value == "definitely-missing-profile":
        event = _event_from_wire(
            {"text": "hello", "source": wire_source},
            transport=transport,
        )
        assert runner._is_user_authorized(event.source) is False
    else:
        with pytest.raises(ValueError):
            _event_from_wire(
                {"text": "hello", "source": wire_source},
                transport=transport,
            )


@pytest.mark.parametrize(
    "wire_source",
    [
        [],
        {
            "platform": {"unexpected": "mapping"},
            "chat_id": "channel-1",
            "chat_type": "group",
        },
    ],
)
def test_relay_non_object_or_malformed_platform_fails_closed(
    monkeypatch,
    wire_source,
):
    from gateway.relay.ws_transport import _event_from_wire

    _clear_auth_env(monkeypatch)
    transport = SimpleNamespace()
    runner, _ = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
        transport=transport,
    )
    with pytest.raises(ValueError):
        _event_from_wire(
            {"text": "hello", "source": wire_source},
            transport=transport,
        )


def test_relay_discord_channel_policy_uses_source_profile_scope(
    monkeypatch,
    tmp_path,
):
    from agent.secret_scope import set_multiplex_active
    from gateway.relay.ws_transport import _event_from_wire

    _clear_auth_env(monkeypatch)
    secondary_home = tmp_path / "profiles" / "coder"
    secondary_home.mkdir(parents=True)
    (secondary_home / ".env").write_text(
        "DISCORD_ALLOWED_CHANNELS=secondary-channel\n",
        encoding="utf-8",
    )
    transport = SimpleNamespace()
    runner, _ = _make_runner(
        platform=Platform.RELAY,
        authorization_is_upstream=True,
        transport=transport,
    )
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = lambda _source: secondary_home
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "coder")

    def event_for(chat_id):
        return _event_from_wire(
            {
                "text": "hello",
                "source": {
                    "platform": "discord",
                    "chat_id": chat_id,
                    "chat_type": "group",
                    "user_id": "267171776755269633",
                    "scope_id": "guild-1",
                    "profile": "coder",
                },
            },
            transport=transport,
        )

    set_multiplex_active(True)
    try:
        assert runner._is_user_authorized(event_for("secondary-channel").source) is True
        assert runner._is_user_authorized(event_for("default-only-channel").source) is False
    finally:
        set_multiplex_active(False)


def test_upstream_discord_adapter_cannot_bypass_channel_policy(monkeypatch):
    from gateway.config import GatewayConfig, PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter

    class UpstreamDiscordAdapter(DiscordAdapter):
        @property
        def authorization_is_upstream(self):
            return True

    _clear_auth_env(monkeypatch)
    adapter = UpstreamDiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "allowed_channels": "different-channel",
                "ignored_channels": "blocked-channel",
            },
        )
    )
    runner, _ = _make_runner(
        platform=Platform.DISCORD,
        authorization_is_upstream=True,
    )
    runner.config = GatewayConfig()
    runner.adapters = {Platform.DISCORD: adapter}
    source = adapter.build_source(
        chat_id="blocked-channel",
        chat_type="group",
        user_id="267171776755269633",
        guild_id="guild-1",
    )

    assert runner._is_user_authorized(source) is False


def test_event_from_wire_stamps_routed_profile():
    """A connector-routed profile on the wire source lands on SessionSource.

    In multiplex mode the connector resolves the target HERMES profile for a
    Team-Gateway message and stamps ``profile`` on the wire source. The relay
    transport must carry it through so build_session_key namespaces the session
    and the agent turn resolves that profile's config/credentials.
    """
    from gateway.relay.ws_transport import _event_from_wire

    event = _event_from_wire(
        {
            "text": "hello!",
            "source": {
                "platform": "discord",
                "chat_id": "123",
                "chat_type": "dm",
                "user_id": "267171776755269633",
                "user_name": "rewbs",
                "profile": "reviewer",
            },
        }
    )
    assert event.source.profile == "reviewer"
