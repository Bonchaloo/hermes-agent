"""Multiplex authorization must never borrow another profile's env gates."""

from unittest.mock import MagicMock

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter, _GATE_ENV_KEYS


_AUTH_GATES = (
    ("DISCORD_ALLOW_ALL_USERS", "true"),
    ("DISCORD_ALLOWED_USERS", "attacker"),
    ("GATEWAY_ALLOW_ALL_USERS", "true"),
    ("GATEWAY_ALLOWED_USERS", "attacker"),
)


@pytest.fixture(autouse=True)
def _reset_multiplex_scope(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists",
        lambda name: name in {"default", "secondary"},
    )
    secret_scope.set_multiplex_active(False)
    token = secret_scope.set_secret_scope(None)
    yield
    secret_scope.reset_secret_scope(token)
    secret_scope.set_multiplex_active(False)


def _runner(
    *,
    default_extra=None,
    secondary_extra=None,
    default_adapter=None,
    secondary_adapter=None,
):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    if default_adapter is None:
        default_adapter = _discord_adapter()
        default_adapter.config.extra.update(default_extra or {})
    if secondary_adapter is None:
        secondary_adapter = _discord_adapter()
        secondary_adapter.config.extra.update(secondary_extra or {})
    runner.adapters = {Platform.DISCORD: default_adapter}
    runner._profile_adapters = {
        "secondary": {Platform.DISCORD: secondary_adapter}
    }
    runner._active_profile_name = lambda: "default"
    runner._profile_name_for_source = lambda _source: None
    pairing_store = MagicMock()
    pairing_store.is_approved.return_value = False
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_stores = {"secondary": pairing_store}
    default_adapter.gateway_runner = runner
    secondary_adapter.gateway_runner = runner
    return runner


def _discord_adapter(
    *,
    allow_from: str = "",
    allowed_channels: str = "",
    gate_env: dict[str, str] | None = None,
) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    extra = {}
    if allow_from:
        extra["allow_from"] = allow_from
    if allowed_channels:
        extra["allowed_channels"] = allowed_channels
    adapter.config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra=extra,
    )
    adapter._gate_env_snapshot = {key: "" for key in _GATE_ENV_KEYS}
    adapter._gate_env_snapshot.update(gate_env or {})
    adapter._allowed_user_ids = adapter._get_allowed_users()
    adapter._allowed_role_ids = set()
    adapter._is_pairing_approved_user = lambda _user_id: False
    return adapter


def _discord_runner(*, allow_from: str):
    adapter = _discord_adapter(allow_from=allow_from)
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    return runner, adapter


def _discord_source(
    adapter: DiscordAdapter,
    *,
    user_id: str,
    chat_type: str,
    chat_id: str | None = None,
    chat_name: str | None = None,
    parent_chat_id: str | None = None,
    role_authorized: bool = False,
    is_bot: bool = False,
):
    return adapter.build_source(
        chat_id=chat_id or f"{chat_type}-1",
        chat_name=chat_name,
        user_id=user_id,
        user_name=user_id,
        chat_type=chat_type,
        thread_id="thread-1" if chat_type == "thread" else None,
        parent_chat_id=parent_chat_id,
        role_authorized=role_authorized,
        is_bot=is_bot,
    )


def _source(runner, profile="secondary") -> SessionSource:
    adapter = (
        runner.adapters[Platform.DISCORD]
        if profile == "default"
        else runner._profile_adapters[profile][Platform.DISCORD]
    )
    return adapter.build_source(
        user_id="attacker",
        chat_id="dm-1",
        user_name="attacker",
        chat_type="dm",
        _profile_override=profile,
    )


@pytest.mark.parametrize(("env_name", "env_value"), _AUTH_GATES)
def test_multiplex_scoped_miss_does_not_borrow_process_auth_gate(
    monkeypatch, env_name, env_value
):
    for name, _ in _AUTH_GATES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, env_value)

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"UNRELATED": "secondary-profile"})
    try:
        runner = _runner()
        assert runner._is_user_authorized(_source(runner)) is False
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize(("env_name", "env_value"), _AUTH_GATES)
def test_multiplex_unscoped_auth_does_not_borrow_process_auth_gate(
    monkeypatch, env_name, env_value
):
    """Missing caller scope must fail closed while multiplexing is active."""
    for name, _ in _AUTH_GATES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, env_value)

    secret_scope.set_multiplex_active(True)

    runner = _runner()
    assert runner._is_user_authorized(_source(runner)) is False


@pytest.mark.parametrize(("env_name", "env_value"), _AUTH_GATES)
def test_multiplex_scoped_auth_gate_still_authorizes(
    monkeypatch, env_name, env_value
):
    for name, _ in _AUTH_GATES:
        monkeypatch.delenv(name, raising=False)

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({env_name: env_value})
    try:
        runner = _runner()
        assert runner._is_user_authorized(_source(runner)) is True
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("profile", ("default", "secondary"))
def test_multiplex_profile_adapter_allow_all_authorizes_its_own_profile(
    monkeypatch, profile
):
    """Profile-local Discord config must remain an authorization authority."""
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        extras = {"allow_all_users": "true"}
        runner = _runner(
            default_extra=extras if profile == "default" else None,
            secondary_extra=extras if profile == "secondary" else None,
        )

        assert runner._is_user_authorized(_source(runner, profile)) is True
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize(
    ("source_profile", "configured_profile"),
    (("default", "secondary"), ("secondary", "default")),
)
def test_multiplex_profile_adapter_allow_all_is_not_borrowed_across_profiles(
    monkeypatch, source_profile, configured_profile
):
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        extras = {"allow_all_users": "true"}
        runner = _runner(
            default_extra=extras if configured_profile == "default" else None,
            secondary_extra=extras if configured_profile == "secondary" else None,
        )

        assert runner._is_user_authorized(_source(runner, source_profile)) is False
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("chat_type", ("dm", "group", "thread"))
def test_discord_yaml_allow_from_survives_intake_and_final_authorizer(
    monkeypatch, chat_type
):
    monkeypatch.delenv("DISCORD_ALLOWED_USERS", raising=False)
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        runner, adapter = _discord_runner(allow_from="trusted-user")
        source = _discord_source(
            adapter,
            user_id="trusted-user",
            chat_type=chat_type,
        )

        assert adapter._is_allowed_user(
            "trusted-user",
            is_dm=chat_type == "dm",
        ) is True
        assert runner._is_user_authorized(source) is True
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("chat_type", ("dm", "group", "thread"))
def test_discord_yaml_allow_from_does_not_borrow_process_global_grant(
    monkeypatch, chat_type
):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "other-user")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        runner, adapter = _discord_runner(allow_from="trusted-user")
        source = _discord_source(
            adapter,
            user_id="other-user",
            chat_type=chat_type,
        )

        assert adapter._is_allowed_user(
            "other-user",
            is_dm=chat_type == "dm",
        ) is False
        assert runner._is_user_authorized(source) is False
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("config_source", ("yaml", "env"))
@pytest.mark.parametrize(
    ("chat_type", "chat_id", "parent_chat_id"),
    (("group", "allowed-channel", None), ("thread", "thread-1", "allowed-channel")),
)
def test_discord_channel_only_grant_survives_intake_and_final_authorizer(
    config_source,
    chat_type,
    chat_id,
    parent_chat_id,
):
    adapter_kwargs = (
        {"allowed_channels": "allowed-channel"}
        if config_source == "yaml"
        else {"gate_env": {"DISCORD_ALLOWED_CHANNELS": "allowed-channel"}}
    )
    configured_adapter = _discord_adapter(**adapter_kwargs)
    runner = _runner(secondary_adapter=configured_adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    configured_adapter.gateway_runner = runner
    source = _discord_source(
        configured_adapter,
        user_id="any-guild-user",
        chat_type=chat_type,
        chat_id=chat_id,
        parent_chat_id=parent_chat_id,
    )
    channel_ids = {chat_id, parent_chat_id} - {None}

    assert configured_adapter._is_allowed_user(
        "any-guild-user",
        is_dm=False,
        channel_ids=channel_ids,
    ) is True
    assert runner._is_user_authorized(source) is True


@pytest.mark.parametrize("config_source", ("yaml", "env"))
def test_discord_channel_only_grant_denies_wrong_channel(config_source):
    adapter_kwargs = (
        {"allowed_channels": "allowed-channel"}
        if config_source == "yaml"
        else {"gate_env": {"DISCORD_ALLOWED_CHANNELS": "allowed-channel"}}
    )
    adapter = _discord_adapter(**adapter_kwargs)
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    source = _discord_source(
        adapter,
        user_id="any-guild-user",
        chat_type="group",
        chat_id="wrong-channel",
    )

    assert adapter._is_allowed_user(
        "any-guild-user",
        is_dm=False,
        channel_ids={"wrong-channel"},
    ) is False
    assert runner._is_user_authorized(source) is False


@pytest.mark.parametrize("config_source", ("yaml", "env"))
@pytest.mark.parametrize("allow_from", ("<@123>", "<@!123>", "user:123"))
def test_discord_normalized_dm_allow_from_survives_final_authorizer(
    config_source,
    allow_from,
):
    adapter_kwargs = (
        {"allow_from": allow_from}
        if config_source == "yaml"
        else {"gate_env": {"DISCORD_ALLOWED_USERS": allow_from}}
    )
    adapter = _discord_adapter(**adapter_kwargs)
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    source = _discord_source(adapter, user_id="123", chat_type="dm")

    assert adapter._is_allowed_user("123", is_dm=True) is True
    assert runner._is_user_authorized(source) is True


def test_discord_normalized_dm_allow_from_denies_wrong_user():
    adapter = _discord_adapter(allow_from="<@123>")
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    source = _discord_source(adapter, user_id="456", chat_type="dm")

    assert adapter._is_allowed_user("456", is_dm=True) is False
    assert runner._is_user_authorized(source) is False


def test_discord_grants_do_not_leak_across_registered_profile_transports():
    default_adapter = _discord_adapter(allow_from="default-user")
    secondary_adapter = _discord_adapter(allow_from="secondary-user")
    runner = _runner(
        default_adapter=default_adapter,
        secondary_adapter=secondary_adapter,
    )
    default_adapter.gateway_runner = runner
    secondary_adapter.gateway_runner = runner

    runner._profile_name_for_source = lambda _source: "secondary"
    secondary_source = _discord_source(
        secondary_adapter,
        user_id="default-user",
        chat_type="dm",
    )
    runner._profile_name_for_source = lambda _source: "default"
    default_source = _discord_source(
        default_adapter,
        user_id="secondary-user",
        chat_type="dm",
    )

    assert runner._is_user_authorized(secondary_source) is False
    assert runner._is_user_authorized(default_source) is False


def test_discord_channel_grants_do_not_leak_across_profile_transports():
    default_adapter = _discord_adapter(allowed_channels="default-channel")
    secondary_adapter = _discord_adapter(allowed_channels="secondary-channel")
    runner = _runner(
        default_adapter=default_adapter,
        secondary_adapter=secondary_adapter,
    )
    default_adapter.gateway_runner = runner
    secondary_adapter.gateway_runner = runner

    runner._profile_name_for_source = lambda _source: "secondary"
    secondary_source = _discord_source(
        secondary_adapter,
        user_id="guild-user",
        chat_type="group",
        chat_id="default-channel",
    )
    runner._profile_name_for_source = lambda _source: "default"
    default_source = _discord_source(
        default_adapter,
        user_id="guild-user",
        chat_type="group",
        chat_id="secondary-channel",
    )

    assert runner._is_user_authorized(secondary_source) is False
    assert runner._is_user_authorized(default_source) is False


def test_discord_grant_fails_closed_when_source_transport_is_unregistered():
    adapter = _discord_adapter(allow_from="<@123>")
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    source = _discord_source(adapter, user_id="123", chat_type="dm")
    runner._profile_adapters.clear()

    assert runner._is_user_authorized(source) is False


def test_discord_verified_role_grant_survives_final_authorizer():
    adapter = _discord_adapter()
    adapter._allowed_role_ids = {42}
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    source = _discord_source(
        adapter,
        user_id="role-user",
        chat_type="group",
        role_authorized=True,
    )

    assert runner._is_user_authorized(source) is True


def test_discord_channel_does_not_bypass_configured_role_requirement():
    adapter = _discord_adapter(allowed_channels="allowed-channel")
    adapter._allowed_role_ids = {42}
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    source = _discord_source(
        adapter,
        user_id="no-role-user",
        chat_type="group",
        chat_id="allowed-channel",
    )

    assert runner._is_user_authorized(source) is False


def test_discord_channel_grant_requires_nonempty_channel_provenance():
    adapter = _discord_adapter(allowed_channels="allowed-channel")
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    source = adapter.build_source(
        chat_id="",
        user_id="guild-user",
        chat_type="channel",
    )

    assert runner._is_user_authorized(source) is False


def test_discord_channel_check_exception_fails_closed():
    adapter = _discord_adapter(allowed_channels="allowed-channel")
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner

    def raise_probe_error(_channel_ids):
        raise RuntimeError("probe failed")

    adapter._discord_channel_ids_allowed = raise_probe_error
    source = _discord_source(
        adapter,
        user_id="guild-user",
        chat_type="group",
        chat_id="allowed-channel",
    )

    assert runner._is_user_authorized(source) is False


def test_discord_grant_fails_closed_without_retained_transport_provenance():
    adapter = _discord_adapter(allowed_channels="allowed-channel")
    runner = _runner(secondary_adapter=adapter)
    forged = SessionSource(
        platform=Platform.DISCORD,
        user_id="guild-user",
        chat_id="allowed-channel",
        chat_type="group",
        profile="secondary",
    )

    assert runner._is_user_authorized(forged) is False


def test_discord_plain_allow_from_fails_closed_without_transport_provenance():
    adapter = _discord_adapter(allow_from="trusted-user")
    runner = _runner(secondary_adapter=adapter)
    forged = SessionSource(
        platform=Platform.DISCORD,
        user_id="trusted-user",
        chat_id="dm-1",
        chat_type="dm",
        profile="secondary",
    )

    assert runner._is_user_authorized(forged) is False


def test_discord_upstream_policy_requires_registered_transport_provenance():
    class UpstreamDiscordAdapter(DiscordAdapter):
        @property
        def authorization_is_upstream(self):
            return True

    adapter = UpstreamDiscordAdapter(
        PlatformConfig(enabled=True, token="token-secondary")
    )
    adapter._gate_env_snapshot = {}
    runner = _runner(secondary_adapter=adapter)
    forged = SessionSource(
        platform=Platform.DISCORD,
        user_id="attacker",
        chat_id="dm-1",
        chat_type="dm",
        profile="secondary",
    )
    live = _discord_source(adapter, user_id="transport-user", chat_type="dm")

    assert runner._is_user_authorized(forged) is False
    assert runner._is_user_authorized(live) is True


@pytest.mark.parametrize("chat_type", [None, "", "bogus", "dm"])
def test_discord_malformed_chat_type_cannot_bypass_ignored_channel(chat_type):
    adapter = _discord_adapter(
        allow_from=["allowed-user"],
        gate_env={"DISCORD_IGNORED_CHANNELS": "blocked-channel"},
    )
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    source = adapter.build_source(
        chat_id="blocked-channel",
        chat_type=chat_type,
        user_id="allowed-user",
        guild_id="guild-1",
    )

    assert runner._is_user_authorized(source) is False


def test_discord_role_boolean_fails_closed_without_transport_provenance():
    adapter = _discord_adapter()
    adapter._allowed_role_ids = {42}
    runner = _runner(secondary_adapter=adapter)
    forged = SessionSource(
        platform=Platform.DISCORD,
        user_id="role-user",
        chat_id="group-1",
        chat_type="group",
        profile="secondary",
        role_authorized=True,
    )

    assert runner._is_user_authorized(forged) is False


@pytest.mark.parametrize(
    "grant",
    (
        "user",
        "role",
        "pairing",
        "channel",
        "discord_allow_all",
        "global_allow_all",
        "bot",
    ),
)
def test_discord_ignored_channel_denies_before_every_grant(monkeypatch, grant):
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("DISCORD_ALLOW_BOTS", raising=False)
    adapter = _discord_adapter(
        allow_from="guild-user" if grant == "user" else "",
        allowed_channels="blocked-channel" if grant == "channel" else "",
        gate_env={"DISCORD_IGNORED_CHANNELS": "blocked-channel"},
    )
    if grant == "role":
        adapter._allowed_role_ids = {42}
    elif grant == "discord_allow_all":
        adapter.config.extra["allow_all_users"] = "true"
    elif grant == "global_allow_all":
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    elif grant == "bot":
        monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")

    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    if grant == "pairing":
        runner.pairing_stores["secondary"].is_approved.return_value = True
    source = _discord_source(
        adapter,
        user_id="guild-user",
        chat_type="group",
        chat_id="blocked-channel",
        role_authorized=grant == "role",
        is_bot=grant == "bot",
    )

    assert runner._is_user_authorized(source) is False


@pytest.mark.parametrize(
    "grant",
    (
        "user",
        "role",
        "pairing",
        "discord_allow_all",
        "global_allow_all",
        "bot",
    ),
)
def test_discord_allowed_channels_restricts_every_grant(monkeypatch, grant):
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("DISCORD_ALLOW_BOTS", raising=False)
    adapter = _discord_adapter(
        allow_from="guild-user" if grant == "user" else "",
        allowed_channels="allowed-channel",
    )
    if grant == "role":
        adapter._allowed_role_ids = {42}
    elif grant == "discord_allow_all":
        adapter.config.extra["allow_all_users"] = "true"
    elif grant == "global_allow_all":
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    elif grant == "bot":
        monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")

    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    adapter.gateway_runner = runner
    if grant == "pairing":
        runner.pairing_stores["secondary"].is_approved.return_value = True
    source = _discord_source(
        adapter,
        user_id="guild-user",
        chat_type="group",
        chat_id="other-channel",
        role_authorized=grant == "role",
        is_bot=grant == "bot",
    )

    assert runner._is_user_authorized(source) is False


def test_discord_name_form_ignored_channel_denies_user_grant():
    adapter = _discord_adapter(
        allow_from="guild-user",
        gate_env={"DISCORD_IGNORED_CHANNELS": "#blocked-room"},
    )
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    source = _discord_source(
        adapter,
        user_id="guild-user",
        chat_type="group",
        chat_id="123456789",
        chat_name="Guild / #blocked-room",
    )

    assert runner._is_user_authorized(source) is False


def test_discord_name_form_allowed_channel_grants_guild_source():
    adapter = _discord_adapter(allowed_channels="#allowed-room")
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    source = _discord_source(
        adapter,
        user_id="guild-user",
        chat_type="group",
        chat_id="123456789",
        chat_name="Guild / #allowed-room",
    )

    assert runner._is_user_authorized(source) is True


def test_discord_ignored_channel_check_exception_fails_closed_before_user_grant():
    adapter = _discord_adapter(allow_from="guild-user")

    def raise_probe_error(_channel_ids):
        raise RuntimeError("probe failed")

    adapter._discord_channel_ids_ignored = raise_probe_error
    runner = _runner(secondary_adapter=adapter)
    runner._profile_name_for_source = lambda _source: "secondary"
    source = _discord_source(
        adapter,
        user_id="guild-user",
        chat_type="group",
        chat_id="channel-1",
    )

    assert runner._is_user_authorized(source) is False


def test_trusted_restored_discord_source_rebuilds_registered_provenance():
    adapter = _discord_adapter(allow_from="trusted-user")
    runner = _runner(secondary_adapter=adapter)
    restored = SessionSource(
        platform=Platform.DISCORD,
        user_id="trusted-user",
        chat_id="dm-1",
        chat_type="dm",
        profile="secondary",
    )

    rebuilt = runner._restored_source_for_authorization(restored, adapter)

    assert rebuilt is not None
    assert rebuilt.profile == "secondary"
    assert runner._registered_transport_adapter(rebuilt) is adapter
    assert runner._is_user_authorized(rebuilt) is True


def test_trusted_restore_does_not_retain_transient_discord_role_grant():
    adapter = _discord_adapter()
    adapter._allowed_role_ids = {42}
    runner = _runner(secondary_adapter=adapter)
    restored = SessionSource(
        platform=Platform.DISCORD,
        user_id="former-role-user",
        chat_id="group-1",
        chat_type="group",
        profile="secondary",
        role_authorized=True,
    )

    rebuilt = runner._restored_source_for_authorization(restored, adapter)

    assert rebuilt is not None
    assert rebuilt.role_authorized is False
    assert runner._is_user_authorized(rebuilt) is False


def test_restored_discord_source_with_omitted_chat_type_fails_closed():
    with pytest.raises(ValueError):
        SessionSource.from_dict(
            {
                "platform": "discord",
                "user_id": "guild-user",
                "chat_id": "123456789",
                "chat_name": "Guild / #blocked-room",
                "profile": "secondary",
            }
        )
