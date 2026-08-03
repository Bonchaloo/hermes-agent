"""Multiplex authorization must never borrow another profile's env gates."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import secret_scope
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter, _GATE_ENV_KEYS


_AUTH_GATES = (
    ("DISCORD_ALLOW_ALL_USERS", "true"),
    ("DISCORD_ALLOWED_USERS", "attacker"),
    ("GATEWAY_ALLOW_ALL_USERS", "true"),
    ("GATEWAY_ALLOWED_USERS", "attacker"),
)


@pytest.fixture(autouse=True)
def _reset_multiplex_scope():
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
    if default_adapter is None:
        default_adapter = SimpleNamespace(
            authorization_is_upstream=False,
            enforces_own_access_policy=False,
            config=SimpleNamespace(extra=default_extra or {}),
        )
    if secondary_adapter is None:
        secondary_adapter = SimpleNamespace(
            authorization_is_upstream=False,
            enforces_own_access_policy=False,
            config=SimpleNamespace(extra=secondary_extra or {}),
        )
    runner.adapters = {Platform.DISCORD: default_adapter}
    runner._profile_adapters = {
        "secondary": {Platform.DISCORD: secondary_adapter}
    }
    runner._active_profile_name = lambda: "default"
    pairing_store = MagicMock()
    pairing_store.is_approved.return_value = False
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_stores = {"secondary": pairing_store}
    return runner


def _discord_adapter(*, allow_from: str) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"allow_from": allow_from},
    )
    adapter._gate_env_snapshot = {key: "" for key in _GATE_ENV_KEYS}
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


def _discord_source(adapter: DiscordAdapter, *, user_id: str, chat_type: str):
    return adapter.build_source(
        chat_id=f"{chat_type}-1",
        user_id=user_id,
        user_name=user_id,
        chat_type=chat_type,
        thread_id="thread-1" if chat_type == "thread" else None,
    )


def _source(profile="secondary") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="attacker",
        chat_id="dm-1",
        user_name="attacker",
        chat_type="dm",
        profile=profile,
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
        assert _runner()._is_user_authorized(_source()) is False
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

    assert _runner()._is_user_authorized(_source()) is False


@pytest.mark.parametrize(("env_name", "env_value"), _AUTH_GATES)
def test_multiplex_scoped_auth_gate_still_authorizes(
    monkeypatch, env_name, env_value
):
    for name, _ in _AUTH_GATES:
        monkeypatch.delenv(name, raising=False)

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({env_name: env_value})
    try:
        assert _runner()._is_user_authorized(_source()) is True
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

        assert runner._is_user_authorized(_source(profile)) is True
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

        assert runner._is_user_authorized(_source(source_profile)) is False
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
