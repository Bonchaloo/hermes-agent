"""User-authorization methods for ``GatewayRunner``.

Extracted from ``gateway/run.py`` as part of the god-file decomposition campaign
(``~/.hermes/plans/god-file-decomposition.md``, Phase 3 mechanical mixin lifts).
This mixin holds the inbound-message authorization cluster: whether a user/chat
is allowed to talk to the agent, the per-adapter DM policy, and the
unauthorized-DM behavior.

Behavior-neutral: every method is lifted verbatim from ``GatewayRunner``.
``self.*`` calls resolve unchanged via the MRO. Neutral dependencies import at
module top; the module-level ``logger`` is imported lazily inside the one method
that uses it (``from gateway.run import logger`` resolves at call time, when
``gateway.run`` is fully loaded) so this module never imports ``gateway.run`` at
import time -> no import cycle. The lazy import preserves the exact logger name
(``"gateway.run"``) so log records are unchanged.
"""

from __future__ import annotations

import os
from typing import Optional

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.whatsapp_identity import (
    expand_whatsapp_aliases as _expand_whatsapp_auth_aliases,
    normalize_whatsapp_identifier as _normalize_whatsapp_identifier,
)


def _platform_gate_env(name: str, default: str = "") -> str:
    """Read an authorization gate env var with per-profile isolation.

    The profile secret scope is authoritative under multiplex. A missing scope
    or a key absent from the installed scope returns ``default`` instead of
    falling through to ``os.environ``. Under multiplex the process env may hold
    ANOTHER profile's first-writer-bridged value (the YAML→env bridges in the
    Discord/Telegram adapters' ``_apply_yaml_config`` are first-writer-wins),
    so falling through would leak profile A's allowlist into profile B
    (issue #72348). Single-profile deployments — multiplex off — behave exactly
    like the legacy ``os.getenv`` read.
    """
    if not name:
        return default
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        if is_multiplex_active():
            scope = current_secret_scope()
            if scope is None:
                return default
            val = scope.get(name)
            if val is None:
                return default
            return str(val).strip()
    except Exception:
        return default
    return (os.getenv(name) or default).strip()


def _coerce_allow_set(raw) -> set[str]:
    """Parse allowlist values from config or env var into a set of strings.

    Handles both list inputs (YAML sequences) and comma-separated string
    inputs (env vars or scalar YAML values).  A scalar string is split on
    commas so ``allow_from: "123,456"`` yields ``{"123", "456"}``, not
    ``{"1", "2", "3", ",", ...}``.
    """
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}


class GatewayAuthorizationMixin:
    """User/chat authorization methods for ``GatewayRunner``."""

    def _authorization_adapter(
        self,
        platform: Optional[Platform],
        profile: Optional[str] = None,
    ):
        """Resolve the live adapter whose intake policy should gate authorization.

        In multiplex mode, secondary-profile adapters live in
        ``_profile_adapters[profile]`` while the default/active profile uses
        ``self.adapters``. ``SessionSource.profile`` selects which map to consult.
        When a stamped profile has its own adapter registry entry, the default
        profile's same-platform adapter must not be consulted as a fallback.
        """
        if not platform:
            return None
        if profile is not None and not isinstance(profile, str):
            return None
        profile_name = (profile or "").strip() or None
        if profile_name and profile_name != "default":
            active_profile = None
            active_profile_fn = getattr(self, "_active_profile_name", None)
            if callable(active_profile_fn):
                try:
                    active_profile = active_profile_fn()
                except Exception:
                    active_profile = None
            if profile_name == active_profile:
                adapters = getattr(self, "adapters", None) or {}
                return adapters.get(platform)
            profile_adapters = getattr(self, "_profile_adapters", None) or {}
            if profile_name in profile_adapters:
                return profile_adapters[profile_name].get(platform)
            # Fail closed: a stamped secondary profile with no registry entry
            # (e.g. its adapter failed to connect) must NOT fall back to the
            # default profile's adapter — that sends replies out the wrong bot.
            return None
        adapters = getattr(self, "adapters", None) or {}
        return adapters.get(platform)

    def _adapter_for_source(self, source: Optional[SessionSource]):
        """Resolve the live adapter for an inbound ``SessionSource``."""
        if source is None:
            return None
        transport_adapter = self._registered_transport_adapter(source)
        if transport_adapter is not None:
            return transport_adapter
        # Relay ingress deliberately keeps the underlying platform on the
        # source so session keys and display policy remain Slack/Discord/etc.
        # Delivery still has to use the one live RelayAdapter that owns the
        # authenticated connector socket. Looking up the underlying platform
        # here silently disables streaming, typing, and tool progress when a
        # managed gateway does not also run that platform's native adapter.
        if getattr(source, "delivered_via_upstream_relay", False) is True:
            # One process-level RelayAdapter owns the connector socket for all
            # multiplexed profiles. Secondary profiles intentionally do not
            # register their own relay adapters, so profile-aware lookup would
            # fail and suppress streamed delivery for those profiles.
            adapters = getattr(self, "adapters", None) or {}
            return adapters.get(Platform.RELAY)
        # ``getattr`` guards test fixtures that build a bare source via
        # SimpleNamespace and omit ``profile`` (see AGENTS.md pitfall #17).
        return self._authorization_adapter(
            getattr(source, "platform", None),
            getattr(source, "profile", None),
        )

    def _candidate_adapter_for_restored_source(
        self, source: Optional[SessionSource]
    ):
        """Select a live adapter candidate before restored provenance exists.

        ``transport_route`` is only a discriminator: the reconstruction gate
        still verifies exact registration, advertised logical platform,
        authenticated epoch, profile existence, and current authorization.
        """
        if source is None:
            return None
        if getattr(source, "transport_route", None) == "relay":
            return (getattr(self, "adapters", None) or {}).get(Platform.RELAY)
        return self._adapter_for_source(source)

    def _registered_transport_adapter(self, source: SessionSource):
        """Return the registered adapter that owns this exact immutable source."""
        platform = getattr(source, "platform", None)
        candidates = []
        default_adapter = (getattr(self, "adapters", None) or {}).get(platform)
        if default_adapter is not None:
            candidates.append(default_adapter)
        for profile_adapters in (
            getattr(self, "_profile_adapters", None) or {}
        ).values():
            adapter = profile_adapters.get(platform)
            if adapter is not None:
                candidates.append(adapter)
        for adapter in candidates:
            registry = getattr(adapter, "_source_provenance", None)
            if registry is not None and registry.verifies(source):
                return adapter
        return None

    def _registered_relay_transport_adapter(self, source: SessionSource):
        """Return the live authenticated relay epoch that received *source*."""
        candidates = []
        default_adapter = (getattr(self, "adapters", None) or {}).get(Platform.RELAY)
        if default_adapter is not None:
            candidates.append(default_adapter)
        for profile_adapters in (
            getattr(self, "_profile_adapters", None) or {}
        ).values():
            adapter = profile_adapters.get(Platform.RELAY)
            if adapter is not None:
                candidates.append(adapter)
        for adapter in candidates:
            transport = getattr(adapter, "_transport", None)
            epoch = getattr(transport, "authenticated_connection_epoch", None)
            registry = getattr(transport, "_source_provenance", None)
            if (
                epoch
                and registry is not None
                and registry.verifies(source, epoch=epoch)
            ):
                return adapter
        return None

    def _discord_transport_authorizes_source(
        self,
        source: SessionSource,
        user_id: str,
    ) -> bool:
        """Re-check Discord intake evidence on the selected live transport.

        Discord normalizes configured user entries (for example ``<@123>``
        and ``user:123``) and can intentionally admit guild traffic using only
        ``allowed_channels``. The generic env/config checks below cannot safely
        reconstruct either decision: their values are raw, and process-global
        env may belong to another multiplex profile. Consult only the adapter
        selected by the source's retained transport provenance. Profile/platform
        lookup is intentionally insufficient: restored or hand-built sources
        must not borrow a live adapter's intake-only grants. A missing or
        unregistered transport adapter provides no authorization.
        """
        adapter = self._registered_transport_adapter(source)
        if adapter is None:
            return False

        allowed_users = getattr(adapter, "_allowed_user_ids", None) or set()
        allowed_roles = getattr(adapter, "_allowed_role_ids", None) or set()
        if allowed_users:
            return "*" in allowed_users or user_id in allowed_users
        if allowed_roles:
            # Role grants are stamped on SessionSource only after the Discord
            # adapter verifies membership in the originating guild. That gate
            # is checked separately before this helper is called.
            return False

        if source.chat_type not in {"group", "forum", "channel", "thread"}:
            return False
        channel_check = getattr(adapter, "_discord_channel_ids_allowed", None)
        if not callable(channel_check):
            return False
        channel_keys = self._discord_source_channel_keys(source, adapter)
        if not channel_keys:
            return False
        try:
            return bool(channel_check(channel_keys))
        except Exception:
            return False

    @staticmethod
    def _discord_source_channel_keys(source: SessionSource, adapter=None) -> set[str]:
        """Return Discord channel-policy keys available from a source.

        Native intake accepts channel snowflakes, bare names, ``#name``, and
        thread-parent forms. Persisted sources retain the channel display name;
        a live adapter may additionally resolve the current channel object from
        cache so a thread's parent name participates in the same policy.
        """
        keys = {
            str(value).strip()
            for value in (source.chat_id, source.thread_id, source.parent_chat_id)
            if value is not None and str(value).strip()
        }
        chat_name = source.chat_name
        if isinstance(chat_name, str) and chat_name.strip():
            display_name = chat_name.strip()
            keys.add(display_name)
            leaf_name = display_name.rsplit(" / ", 1)[-1].strip()
            normalized_name = leaf_name.removeprefix("#").strip()
            if normalized_name:
                keys.add(normalized_name)
                keys.add(f"#{normalized_name}")

        client = getattr(adapter, "_client", None)
        get_channel = getattr(client, "get_channel", None)
        key_builder = getattr(adapter, "_discord_channel_keys_from_channel", None)
        if callable(get_channel) and callable(key_builder):
            channel_id = source.thread_id or source.chat_id
            try:
                channel = get_channel(int(channel_id))
            except (TypeError, ValueError):
                channel = None
            if channel is not None:
                keys.update(key_builder(channel, source.parent_chat_id))
        return keys

    def _discord_transport_denies_source(
        self,
        source: SessionSource,
        adapter,
        *,
        allow_unconfigured_without_adapter: bool = False,
    ) -> bool:
        """Apply Discord's channel restrictions before every grant.

        Native Discord ingress always has retained adapter provenance and a
        channel identity for guild traffic. Missing provenance, malformed guild
        context, unavailable policy helpers, and policy errors all fail closed.
        Structurally valid DMs intentionally do not participate in Discord
        channel policy. Unknown chat types and DM sources carrying guild/thread
        structure are malformed and fail closed. Authenticated relay ingress has
        no native Discord transport; in that case scoped environment policy is
        still enforced, while an unconfigured policy does not erase upstream
        owner authentication.
        """
        if adapter is None and not allow_unconfigured_without_adapter:
            return True
        chat_type = source.chat_type
        server_chat_types = {"group", "forum", "channel", "thread"}
        if not isinstance(chat_type, str) or chat_type not in server_chat_types | {"dm"}:
            return True
        if chat_type == "dm":
            return bool(
                source.scope_id
                or source.guild_id
                or source.parent_chat_id
                or source.thread_id
            )
        if chat_type not in server_chat_types:
            return False
        channel_keys = self._discord_source_channel_keys(source, adapter)
        if not channel_keys:
            return True
        try:
            if adapter is not None:
                ignored_check = getattr(adapter, "_discord_channel_ids_ignored", None)
                allowed_getter = getattr(adapter, "_get_allowed_channels", None)
                if not callable(ignored_check) or not callable(allowed_getter):
                    return True
                if bool(ignored_check(channel_keys)):
                    return True
                allowed = allowed_getter()
            else:
                ignored = _coerce_allow_set(
                    _platform_gate_env("DISCORD_IGNORED_CHANNELS")
                )
                if "*" in ignored or bool(channel_keys & ignored):
                    return True
                allowed = _coerce_allow_set(
                    _platform_gate_env("DISCORD_ALLOWED_CHANNELS")
                )
            return bool(
                allowed
                and "*" not in allowed
                and not (channel_keys & allowed)
            )
        except Exception:
            return True

    def _restored_native_source_for_authorization(
        self, source: SessionSource, adapter
    ) -> Optional[SessionSource]:
        """Rebuild one persisted native source solely for startup recovery."""
        if getattr(source, "_persisted_metadata_valid", True) is not True:
            return None
        if (
            adapter is None
            or source.platform == Platform.RELAY
            or getattr(adapter, "platform", None) != source.platform
        ):
            return None
        valid_chat_types = {"dm", "group", "channel", "thread", "forum"}
        if not isinstance(source.chat_id, str) or not source.chat_id.strip():
            return None
        if not isinstance(source.chat_type, str) or source.chat_type not in valid_chat_types:
            return None
        for value in (
            source.chat_name,
            source.chat_topic,
            source.user_id,
            source.user_name,
            source.thread_id,
            source.parent_chat_id,
            source.message_id,
            source.scope_id,
        ):
            if value is not None and not isinstance(value, str):
                return None

        profile = source.profile
        if profile is not None:
            if not isinstance(profile, str) or not profile.strip():
                return None
            if not bool(
                getattr(getattr(self, "config", None), "multiplex_profiles", False)
            ):
                return None
            try:
                from hermes_cli.profiles import profile_exists

                if not profile_exists(profile):
                    return None
            except Exception:
                return None
        build_source = getattr(adapter, "build_source", None)
        if not callable(build_source):
            return None
        try:
            rebuilt = build_source(
                chat_id=source.chat_id,
                chat_name=source.chat_name,
                chat_type=source.chat_type,
                user_id=source.user_id,
                user_name=source.user_name,
                thread_id=source.thread_id,
                chat_topic=source.chat_topic,
                user_id_alt=source.user_id_alt,
                chat_id_alt=source.chat_id_alt,
                is_bot=source.is_bot,
                scope_id=source.scope_id,
                guild_id=source.guild_id,
                parent_chat_id=source.parent_chat_id,
                message_id=source.message_id,
                # Role membership is adapter-local evidence for one live event.
                # Never promote a persisted boolean into fresh transport trust;
                # role-only sessions must be reauthorized by a new live event.
                role_authorized=False,
                auto_thread_created=source.auto_thread_created,
                auto_thread_initial_name=source.auto_thread_initial_name,
                _profile_override=source.profile,
            )
        except Exception:
            from gateway.run import logger

            logger.warning(
                "Skipping malformed restored %s source during authorization: ***",
                source.platform.value,
            )
            return None
        if not isinstance(rebuilt, SessionSource):
            return None
        return rebuilt

    def _restored_source_for_authorization(
        self, source: SessionSource, adapter
    ) -> Optional[SessionSource]:
        """Rebuild one persisted source against a current process adapter.

        Relay persistence carries only the non-authoritative ``transport_route``
        discriminator.  Startup may promote that descriptor into a live relay
        source only after binding it to the one current process RelayAdapter,
        validating its exact underlying Discord platform/profile, and registering
        the rebuilt object in the current authenticated connection epoch.
        """
        if getattr(source, "transport_route", None) != "relay":
            return self._restored_native_source_for_authorization(source, adapter)

        if getattr(source, "_persisted_metadata_valid", True) is not True:
            return None
        if source.platform != Platform.DISCORD:
            return None
        if adapter is not (getattr(self, "adapters", None) or {}).get(Platform.RELAY):
            return None
        if getattr(adapter, "platform", None) != Platform.RELAY:
            return None
        fronts_platform = getattr(adapter, "fronts_platform", None)
        if not callable(fronts_platform) or not fronts_platform(source.platform):
            return None

        valid_chat_types = {"dm", "group", "channel", "thread", "forum"}
        if not isinstance(source.chat_id, str) or not source.chat_id.strip():
            return None
        if not isinstance(source.chat_type, str) or source.chat_type not in valid_chat_types:
            return None
        for value in (
            source.chat_name,
            source.chat_topic,
            source.user_id,
            source.user_name,
            source.thread_id,
            source.parent_chat_id,
            source.message_id,
            source.scope_id,
        ):
            if value is not None and not isinstance(value, str):
                return None

        profile = source.profile
        if profile is not None:
            if not isinstance(profile, str) or not profile.strip():
                return None
            if not bool(getattr(getattr(self, "config", None), "multiplex_profiles", False)):
                return None
            try:
                from hermes_cli.profiles import profile_exists

                if not profile_exists(profile):
                    return None
            except Exception:
                return None

        transport = getattr(adapter, "_transport", None)
        epoch = getattr(transport, "authenticated_connection_epoch", None)
        registry = getattr(transport, "_source_provenance", None)
        if not epoch or registry is None:
            return None

        rebuilt = SessionSource(
            platform=source.platform,
            chat_id=source.chat_id,
            chat_name=source.chat_name,
            chat_type=source.chat_type,
            user_id=source.user_id,
            user_name=source.user_name,
            thread_id=source.thread_id,
            chat_topic=source.chat_topic,
            user_id_alt=source.user_id_alt,
            chat_id_alt=source.chat_id_alt,
            is_bot=source.is_bot,
            scope_id=source.scope_id,
            parent_chat_id=source.parent_chat_id,
            message_id=source.message_id,
            role_authorized=False,
            profile=profile,
            transport_route="relay",
            auto_thread_created=source.auto_thread_created,
            auto_thread_initial_name=source.auto_thread_initial_name,
            prospective_thread_id=source.prospective_thread_id,
            delivered_via_upstream_relay=True,
        )
        registry.register(rebuilt, epoch=epoch)

        # Rehydrate only delivery discriminators.  Authorization below is always
        # recomputed from the current profile-scoped Discord policy; no persisted
        # role grant or prior-process provenance survives this boundary.
        capture_scope = getattr(adapter, "_capture_scope", None)
        if callable(capture_scope):
            capture_scope(type("RestoredRelayEvent", (), {"source": rebuilt})())
        return rebuilt

    def _adapter_profile_for_source(self, source: SessionSource) -> Optional[str]:
        """Resolve the transport-owning profile for adapter policy lookups."""
        adapter = self._registered_transport_adapter(source)
        platform = getattr(source, "platform", None)
        if adapter is not None:
            if adapter is (getattr(self, "adapters", None) or {}).get(platform):
                return None
            for profile, profile_adapters in (
                getattr(self, "_profile_adapters", None) or {}
            ).items():
                if adapter is profile_adapters.get(platform):
                    return profile
        return getattr(source, "profile", None)

    def _adapter_authorization_is_upstream(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> bool:
        """Whether the adapter for *platform* delegates authz to a trusted upstream.

        Mirrors ``BasePlatformAdapter.authorization_is_upstream``. The relay
        adapter sets this True: the Team Gateway connector authenticates the
        gateway's WS and resolves owner-only author bindings before delivering,
        so an inbound relay event is already authorized as this instance's bound
        user. Unlike ``_adapter_enforces_own_access_policy`` (a LOCAL config
        policy the gateway mirrors only when it's an allowlist), this is an
        UPSTREAM decision the gateway honors directly. Defaults to ``False`` when
        the adapter is unknown or doesn't expose the flag.
        """
        if not platform:
            return False
        adapter = self._authorization_adapter(platform, profile)
        if adapter is None:
            return False
        return bool(getattr(adapter, "authorization_is_upstream", False))

    def _adapter_enforces_own_access_policy(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> bool:
        """Whether the adapter for *platform* gates access at intake itself.

        Mirrors ``BasePlatformAdapter.enforces_own_access_policy``. Adapters
        such as WeCom, Weixin, Yuanbao, QQBot, and WhatsApp evaluate their
        documented ``dm_policy`` / ``group_policy`` / ``allow_from`` config before a
        message is dispatched to the gateway. The flag alone is NOT "already
        authorized": these adapters default to ``open``, which forwards every
        sender, so ``_is_user_authorized`` only trusts the adapter when its
        effective policy for the chat type is an actual ``allowlist`` restriction
        (see that method). Defaults to ``False`` when the adapter is unknown or
        doesn't expose the flag.
        """
        if not platform:
            return False
        # Some test helpers build a bare GatewayRunner via object.__new__ and
        # never set ``adapters``; treat a missing/empty map as "no adapter"
        # rather than raising (see pitfalls.md #17).
        adapter = self._authorization_adapter(platform, profile)
        if adapter is None:
            return False
        return bool(getattr(adapter, "enforces_own_access_policy", False))

    def _adapter_dm_policy(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> str:
        """Best-effort read of an own-policy adapter's effective DM policy.

        Returns the lowercased ``dm_policy`` (``"open"`` / ``"allowlist"`` /
        ``"disabled"`` / ``"pairing"``) for *platform*, or ``""`` when unknown.
        Prefers the live adapter's resolved ``_dm_policy`` — which already folds
        in both ``config.extra`` and the ``<PLATFORM>_DM_POLICY`` env var (the
        env var is not always bridged back into ``config.extra``) — and falls
        back to ``config.extra`` for bare runners built without a live adapter.

        Used by ``_is_user_authorized`` to decide whether an own-policy adapter
        actually restricted DM senders to a configured allowlist (trustworthy)
        or merely forwarded everyone under ``dm_policy: open`` / for a pairing
        handshake (not authorization). "Reached the gateway" only carries an
        authorization signal in the ``allowlist`` case.
        """
        if not platform:
            return ""
        adapter = self._authorization_adapter(platform, profile)
        policy = getattr(adapter, "_dm_policy", None) if adapter is not None else None
        if policy is None:
            config = getattr(self, "config", None)
            platform_cfg = (
                config.platforms.get(platform)
                if config is not None and hasattr(config, "platforms")
                else None
            )
            extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
            if isinstance(extra, dict):
                policy = extra.get("dm_policy")
        return str(policy or "").strip().lower()

    def _adapter_group_policy(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> str:
        """Best-effort read of an own-policy adapter's effective group policy.

        Mirror of ``_adapter_dm_policy`` for group / forum / channel traffic:
        returns the lowercased ``group_policy`` (``"open"`` / ``"allowlist"`` /
        ``"disabled"``) for *platform*, or ``""`` when unknown. Prefers the live
        adapter's resolved ``_group_policy`` and falls back to ``config.extra``
        for bare runners built without a live adapter.

        Used by ``_is_user_authorized`` to decide whether an own-policy adapter
        restricted group senders to a configured allowlist (trustworthy) or
        forwarded the whole channel under ``group_policy: open`` (not
        authorization).
        """
        if not platform:
            return ""
        adapter = self._authorization_adapter(platform, profile)
        policy = getattr(adapter, "_group_policy", None) if adapter is not None else None
        if policy is None:
            config = getattr(self, "config", None)
            platform_cfg = (
                config.platforms.get(platform)
                if config is not None and hasattr(config, "platforms")
                else None
            )
            extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
            if isinstance(extra, dict):
                policy = extra.get("group_policy")
        return str(policy or "").strip().lower()

    def _adapter_group_has_sender_allowlist(
        self,
        platform: Optional[Platform],
        chat_id: Optional[str],
        *,
        profile: Optional[str] = None,
    ) -> bool:
        """Whether a per-group sender allowlist gated this group message.

        WeCom supports ``groups.<group_id>.allow_from`` on top of the top-level
        ``group_policy``. A group may be open at the chat level while still
        restricting which senders inside that group can invoke Hermes. If such a
        message reached the gateway, the adapter already checked that sender
        allowlist, so it is a trustworthy intake decision rather than the
        fail-open ``group_policy: open`` case.
        """
        if not platform or not chat_id:
            return False
        adapter = self._authorization_adapter(platform, profile)
        groups = getattr(adapter, "_groups", None) if adapter is not None else None
        if groups is None:
            config = getattr(self, "config", None)
            platform_cfg = (
                config.platforms.get(platform)
                if config is not None and hasattr(config, "platforms")
                else None
            )
            extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
            if isinstance(extra, dict):
                groups = extra.get("groups")
        if not isinstance(groups, dict):
            return False

        chat_id_str = str(chat_id)
        group_cfg = groups.get(chat_id_str)
        if not isinstance(group_cfg, dict):
            lowered = chat_id_str.lower()
            for key, value in groups.items():
                if isinstance(key, str) and key.lower() == lowered and isinstance(value, dict):
                    group_cfg = value
                    break
        if not isinstance(group_cfg, dict):
            group_cfg = groups.get("*")
        if not isinstance(group_cfg, dict):
            return False

        sender_allow = group_cfg.get("allow_from") or group_cfg.get("allowFrom")
        if isinstance(sender_allow, str):
            return bool(sender_allow.strip())
        if isinstance(sender_allow, (list, tuple, set)):
            return any(str(item).strip() for item in sender_allow)
        return False

    def _pairing_store_for(self, source: "SessionSource"):
        """Pick the per-profile PairingStore for a source, falling back to global.

        In a multiplexing gateway, each profile owns its own pairing whitelist
        so isolation is preserved. When the source has no profile (single-
        profile gateway, or a path that hasn't stamped profile yet) or the
        profile isn't registered, fall back to ``self.pairing_store`` (the
        global default) so existing behavior is preserved.
        """
        per_profile = getattr(self, "pairing_stores", None) or {}
        profile = getattr(source, "profile", None)
        if profile and profile in per_profile:
            return per_profile[profile]
        return getattr(self, "pairing_store", None)

    def _is_user_authorized(self, source: SessionSource) -> bool:
        """Authorize under the source profile's exact runtime secret scope."""
        profile = getattr(source, "profile", None)
        if profile is not None and (
            not isinstance(profile, str) or not profile.strip()
        ):
            return False
        try:
            from agent.secret_scope import is_multiplex_active

            multiplex_active = is_multiplex_active()
        except Exception:
            multiplex_active = False

        relay_delivery = (
            getattr(source, "delivered_via_upstream_relay", False) is True
        )
        if profile is not None:
            try:
                from hermes_cli.profiles import profile_exists

                if not profile_exists(profile):
                    return False
            except Exception:
                return False

        if multiplex_active and relay_delivery:
            try:
                from gateway.run import _profile_runtime_scope

                profile_home = self._resolve_profile_home_for_source(source)
                with _profile_runtime_scope(profile_home):
                    return self._is_user_authorized_in_scope(source)
            except Exception:
                from gateway.run import logger

                logger.warning(
                    "Authorization denied: could not install profile scope",
                    exc_info=True,
                )
                return False
        return self._is_user_authorized_in_scope(source)

    def _is_user_authorized_in_scope(self, source: SessionSource) -> bool:
        """
        Check if a user is authorized to use the bot.
        
        Checks in order:
        1. Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        2. Environment variable allowlists (TELEGRAM_ALLOWED_USERS, etc.)
        3. DM pairing approved list
        4. Global allow-all (GATEWAY_ALLOW_ALL_USERS=true)
        5. Default: deny
        """
        from gateway.run import logger
        # Home Assistant events are system-generated (state changes), not
        # user-initiated messages.  The HASS_TOKEN already authenticates the
        # connection, so HA events are always authorized.
        # Webhook events are authenticated via HMAC signature validation in
        # the adapter itself — no user allowlist applies.
        if source.platform in {Platform.HOMEASSISTANT, Platform.WEBHOOK}:
            return True

        adapter_profile = self._adapter_profile_for_source(source)

        # Relay (and any adapter whose authorization is enforced by a trusted
        # authenticated upstream): the Team Gateway connector authenticates this
        # gateway's WS with a per-instance secret and resolves owner-only author
        # bindings BEFORE delivering, so an inbound relay event was already
        # authorized as this instance's bound user (the author id is the one the
        # connector observed, never gateway-asserted). There is no local
        # RELAY_ALLOWED_USERS env allowlist to consult, and default-denying for
        # its absence is the bug this branch fixes. This is delegation to a
        # trusted upstream, NOT a fail-open: it fires only for an event that was
        # actually delivered over the authenticated relay WS (the transport
        # stamps ``delivered_via_upstream_relay``), or whose platform's adapter
        # explicitly declares ``authorization_is_upstream=True``; every direct
        # network-exposed adapter leaves the flag False and its events unmarked,
        # so the env-allowlist default-deny below still applies unchanged.
        #
        # The delivery marker is the PRIMARY signal: a relay *message* inbound
        # carries the UNDERLYING platform (``source.platform`` == discord/…),
        # NOT ``Platform.RELAY``, because that's what session-keying and egress
        # need — so keying authz off ``source.platform`` would miss (the relay
        # adapter is registered under ``Platform.RELAY``) and default-deny the
        # user ("Unauthorized user <id> on discord"). The adapter-flag check is
        # retained for events whose ``source.platform`` IS ``Platform.RELAY``
        # (e.g. the interaction-passthrough path).
        # The public marker supports routing, but authorization additionally
        # requires the exact live relay transport attached at authenticated
        # wire reconstruction and owned by a registered RelayAdapter.
        # Constructor-set, persisted, or copied markers therefore cannot
        # manufacture upstream trust.
        if source.delivered_via_upstream_relay is True:
            relay_adapter = self._registered_relay_transport_adapter(source)
            if (
                relay_adapter is None
                or getattr(relay_adapter, "authorization_is_upstream", False)
                is not True
            ):
                return False
            if source.platform == Platform.DISCORD:
                discord_policy_adapter = self._authorization_adapter(
                    Platform.DISCORD,
                    profile=adapter_profile,
                )
                if self._discord_transport_denies_source(
                    source,
                    discord_policy_adapter,
                    allow_unconfigured_without_adapter=True,
                ):
                    return False
            return True

        # Relay sources are accepted only through the authenticated transport
        # branch above. A caller-built Platform.RELAY source has no provenance.
        if source.platform == Platform.RELAY:
            return False

        if self._adapter_authorization_is_upstream(
            source.platform,
            profile=adapter_profile,
        ):
            if source.platform != Platform.DISCORD:
                return True
            upstream_transport = self._registered_transport_adapter(source)
            if upstream_transport is not None and bool(
                getattr(upstream_transport, "authorization_is_upstream", False)
            ):
                if self._discord_transport_denies_source(
                    source,
                    upstream_transport,
                ):
                    return False
                return True
            return False

        # Every native Discord grant depends on evidence owned by the receiving
        # adapter. Require the exact retained, still-registered transport before
        # evaluating any grant, then apply ignored-channel policy as a universal
        # deny before bot, allow-all, role, pairing, user, channel, or global
        # allowlist paths can return True. Relay-delivered Discord events were
        # handled by the authenticated-upstream branch above.
        discord_adapter = None
        if source.platform == Platform.DISCORD:
            discord_adapter = self._registered_transport_adapter(source)
            if self._discord_transport_denies_source(source, discord_adapter):
                return False

        user_id = source.user_id

        # Telegram (and similar) authorize entire group/forum/channel chats
        # by chat ID via TELEGRAM_GROUP_ALLOWED_CHATS / QQ_GROUP_ALLOWED_USERS.
        # That allowlist is chat-scoped, so it must work even when
        # source.user_id is None — Telegram emits anonymous-admin posts,
        # sender_chat traffic, and channel broadcasts with no `from_user`,
        # and an operator who explicitly listed the chat expects those to
        # be honored. Run this check before the no-user-id guard below so
        # documented behavior matches reality
        # (website/docs/reference/environment-variables.md,
        # website/docs/user-guide/messaging/telegram.md).
        if source.chat_type in {"group", "forum", "channel"} and source.chat_id:
            chat_allowlist_env = {
                Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_CHATS",
                Platform.QQBOT: "QQ_GROUP_ALLOWED_USERS",
            }.get(source.platform, "")
            if chat_allowlist_env:
                raw_chat_allowlist = _platform_gate_env(chat_allowlist_env)
                if raw_chat_allowlist:
                    allowed_group_ids = {
                        cid.strip()
                        for cid in raw_chat_allowlist.split(",")
                        if cid.strip()
                    }
                    if "*" in allowed_group_ids or source.chat_id in allowed_group_ids:
                        return True

            # Fallback: also check adapter-level config (config.yaml)
            # for platforms.<platform>.extra.group_allowed_chats.
            # The Telegram observe-unmentioned mode strips user_id from
            # triggered group messages (_apply_telegram_group_observe_attribution),
            # so the env-var-only check above misses config.yaml-configured
            # allowlists.  Read the live adapter's config.extra as a fallback.
            try:
                adapter = self._adapter_for_source(source)
                if adapter is not None:
                    extra = getattr(getattr(adapter, "config", None), "extra", None) or {}
                    adapter_group_allowed = extra.get("group_allowed_chats")
                    if adapter_group_allowed:
                        allowed = _coerce_allow_set(adapter_group_allowed)
                        if "*" in allowed or source.chat_id in allowed:
                            return True
            except Exception:
                pass

        # Bots admitted by {PLATFORM}_ALLOW_BOTS bypass the human allowlist (#4466).
        # Checked before the no-user-id guard below: some platforms deliver
        # bot/automation traffic with no user_id at all -- e.g. Slack Workflow
        # Builder posts arrive as subtype=bot_message with user=None -- so
        # deferring past the guard would reject them outright (the same reason
        # the chat-scoped allowlist above runs early).
        platform_allow_bots_map = {
            Platform.DISCORD: "DISCORD_ALLOW_BOTS",
            Platform.FEISHU: "FEISHU_ALLOW_BOTS",
            Platform.TELEGRAM: "TELEGRAM_ALLOW_BOTS",
            Platform.SLACK: "SLACK_ALLOW_BOTS",
        }
        if getattr(source, "is_bot", False):
            allow_bots_var = platform_allow_bots_map.get(source.platform)
            if allow_bots_var and _platform_gate_env(allow_bots_var, "none").lower().strip() in {"mentions", "all"}:
                return True

        if not user_id:
            return False

        platform_env_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
            Platform.DISCORD: "DISCORD_ALLOWED_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
            Platform.WHATSAPP_CLOUD: "WHATSAPP_CLOUD_ALLOWED_USERS",
            Platform.SLACK: "SLACK_ALLOWED_USERS",
            Platform.SIGNAL: "SIGNAL_ALLOWED_USERS",
            Platform.EMAIL: "EMAIL_ALLOWED_USERS",
            Platform.SMS: "SMS_ALLOWED_USERS",
            Platform.MATTERMOST: "MATTERMOST_ALLOWED_USERS",
            Platform.MATRIX: "MATRIX_ALLOWED_USERS",
            Platform.DINGTALK: "DINGTALK_ALLOWED_USERS",
            Platform.FEISHU: "FEISHU_ALLOWED_USERS",
            Platform.WECOM: "WECOM_ALLOWED_USERS",
            Platform.WECOM_CALLBACK: "WECOM_CALLBACK_ALLOWED_USERS",
            Platform.WEIXIN: "WEIXIN_ALLOWED_USERS",
            Platform.BLUEBUBBLES: "BLUEBUBBLES_ALLOWED_USERS",
            Platform.QQBOT: "QQ_ALLOWED_USERS",
            Platform.YUANBAO: "YUANBAO_ALLOWED_USERS",
        }
        platform_group_user_env_map = {
            Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_USERS",
        }
        platform_group_chat_env_map = {
            Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_CHATS",
            Platform.QQBOT: "QQ_GROUP_ALLOWED_USERS",
        }
        platform_allow_all_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOW_ALL_USERS",
            Platform.DISCORD: "DISCORD_ALLOW_ALL_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOW_ALL_USERS",
            Platform.WHATSAPP_CLOUD: "WHATSAPP_CLOUD_ALLOW_ALL_USERS",
            Platform.SLACK: "SLACK_ALLOW_ALL_USERS",
            Platform.SIGNAL: "SIGNAL_ALLOW_ALL_USERS",
            Platform.EMAIL: "EMAIL_ALLOW_ALL_USERS",
            Platform.SMS: "SMS_ALLOW_ALL_USERS",
            Platform.MATTERMOST: "MATTERMOST_ALLOW_ALL_USERS",
            Platform.MATRIX: "MATRIX_ALLOW_ALL_USERS",
            Platform.DINGTALK: "DINGTALK_ALLOW_ALL_USERS",
            Platform.FEISHU: "FEISHU_ALLOW_ALL_USERS",
            Platform.WECOM: "WECOM_ALLOW_ALL_USERS",
            Platform.WECOM_CALLBACK: "WECOM_CALLBACK_ALLOW_ALL_USERS",
            Platform.WEIXIN: "WEIXIN_ALLOW_ALL_USERS",
            Platform.BLUEBUBBLES: "BLUEBUBBLES_ALLOW_ALL_USERS",
            Platform.QQBOT: "QQ_ALLOW_ALL_USERS",
            Platform.YUANBAO: "YUANBAO_ALLOW_ALL_USERS",
        }

        # Plugin platforms: check the registry for auth env var names
        if source.platform not in platform_env_map:
            try:
                from gateway.platform_registry import platform_registry
                entry = platform_registry.get(source.platform.value)
                if entry:
                    if entry.allowed_users_env:
                        platform_env_map[source.platform] = entry.allowed_users_env
                    if entry.allow_all_env:
                        platform_allow_all_map[source.platform] = entry.allow_all_env
            except Exception:
                pass

        # Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        platform_allow_all_var = platform_allow_all_map.get(source.platform, "")
        if platform_allow_all_var and _platform_gate_env(platform_allow_all_var).lower() in {"true", "1", "yes"}:
            return True

        # Discord's profile loader seeds config.yaml authorization gates into
        # the live adapter's PlatformConfig.extra. Under multiplexing it
        # intentionally does not bridge those values into process-global env,
        # so consult only the profile-bound adapter selected by this source.
        # ``discord_adapter`` was bound to retained transport provenance above;
        # profile/platform lookup is not authorization evidence.
        if source.platform == Platform.DISCORD:
            adapter = discord_adapter
            extra = getattr(getattr(adapter, "config", None), "extra", None) or {}
            if str(extra.get("allow_all_users", "")).strip().lower() in {
                "true",
                "1",
                "yes",
            }:
                return True

        # Adapter-verified role auth: Discord stamps this only after confirming
        # the role at intake. Bind that in-process stamp to the exact registered
        # transport that created the source so a hand-built SessionSource cannot
        # forge the boolean. Other platforms retain their historical bool gate.
        # Compare with ``is True`` so MagicMock test sources do not auto-truthy.
        if getattr(source, "role_authorized", False) is True:
            if source.platform != Platform.DISCORD:
                return True
            role_adapter = self._registered_transport_adapter(source)
            if role_adapter is not None and bool(
                getattr(role_adapter, "_allowed_role_ids", None)
            ):
                return True

        # Check pairing store. A pairing entry is a first-class authorization
        # grant, created only by a trusted operator approving a pairing code
        # (hermes gateway pairing approve / the authenticated dashboard) — an
        # inbound sender can never reach approve_code, so this is not an
        # attacker-controlled path. Honored as a UNION with the allowlist: a
        # paired user is authorized regardless of the allowlist, and when an
        # allowlist IS configured, operator approval also writes the user into
        # that allowlist (see PairingStore._approve_user), keeping a single
        # operator-visible source of truth. (#23778: the original bypass was the
        # inbound message/approval-button gate, not this gate; that gate is
        # fixed separately.)
        # In multiplex gateways, route to the per-profile PairingStore so each
        # profile's whitelist is isolated; falls back to the global store when
        # the source has no profile or the profile isn't registered.
        platform_name = source.platform.value if source.platform else ""
        pairing_store = self._pairing_store_for(source)
        if pairing_store is not None and pairing_store.is_approved(platform_name, user_id):
            return True

        # Discord's adapter owns two pieces of authorization evidence that the
        # generic gate cannot derive safely: normalized allow_from principals
        # and channel-only guild grants. Re-check both against the exact live,
        # registered transport selected for this source.
        if source.platform == Platform.DISCORD and self._discord_transport_authorizes_source(
            source,
            user_id,
        ):
            return True

        # Check platform-specific and global allowlists
        platform_allowlist = _platform_gate_env(platform_env_map.get(source.platform, ""))
        group_user_allowlist = ""
        group_chat_allowlist = ""
        if source.chat_type in {"group", "forum"}:
            group_user_allowlist = _platform_gate_env(platform_group_user_env_map.get(source.platform, ""))
            group_chat_allowlist = _platform_gate_env(platform_group_chat_env_map.get(source.platform, ""))
        global_allowlist = _platform_gate_env("GATEWAY_ALLOWED_USERS")

        if not platform_allowlist and not group_user_allowlist and not group_chat_allowlist and not global_allowlist:
            # No env allowlist configured. Adapters that own their own
            # config-driven access policy (dm_policy / group_policy /
            # allow_from / group_allow_from) gate access at intake, so for those
            # platforms we can honor the adapter's decision instead of the
            # env-only default-deny below -- but ONLY when that decision was an
            # actual allowlist restriction.
            #
            # The adapters default dm_policy / group_policy to "open", which
            # forwards EVERY sender. Reading "reached the gateway" as
            # authorization in that case would admit the whole external network
            # with no operator-configured allowlist -- the fail-open SECURITY.md
            # §2.6 forbids ("an allowlist is required for every enabled
            # network-exposed adapter ... code paths that fail open when no
            # allowlist is configured are code bugs"). "disabled" never
            # forwards, and "pairing" forwards unpaired DMs only so the gateway
            # can run its pairing handshake (the pairing-store check above
            # already denied this sender). So trust the adapter only when its
            # effective policy for THIS chat type is "allowlist"; for "open" /
            # "pairing" / anything else, fall through to default-deny, where
            # GATEWAY_ALLOW_ALL_USERS, the per-platform {PLATFORM}_ALLOW_ALL_USERS
            # flag (checked above), and the pairing flow remain the explicit
            # opt-ins to broader access. (#34515 follow-up: trusting "open" was a
            # fail-open.)
            if self._adapter_enforces_own_access_policy(
                source.platform,
                profile=adapter_profile,
            ):
                if source.chat_type in {"group", "forum", "channel"}:
                    effective_policy = self._adapter_group_policy(
                        source.platform,
                        profile=adapter_profile,
                    )
                    if self._adapter_group_has_sender_allowlist(
                        source.platform,
                        source.chat_id,
                        profile=adapter_profile,
                    ):
                        return True
                else:
                    effective_policy = self._adapter_dm_policy(
                        source.platform,
                        profile=adapter_profile,
                    )
                if effective_policy == "allowlist":
                    # Trust allowlist intake only when the live adapter still
                    # allowlists this sender. Pairing revoke can clear
                    # WHATSAPP_ALLOWED_USERS while a construction-time
                    # ``_allow_from`` snapshot would otherwise keep authorizing
                    # until restart; re-check when the adapter exposes a DM
                    # allowlist helper. Adapters without that helper keep the
                    # historical "reached the gateway under allowlist policy"
                    # rubber-stamp (#34515).
                    if source.chat_type not in {"group", "forum", "channel"}:
                        adapter = self._authorization_adapter(
                            source.platform,
                            profile=adapter_profile,
                        )
                        dm_check = (
                            getattr(adapter, "_is_dm_allowed", None)
                            if adapter is not None
                            else None
                        )
                        if callable(dm_check):
                            return bool(dm_check(user_id))
                    return True
            # Some adapters (e.g. Telegram) gate access via config.extra.allow_from /
            # group_allow_from at intake but do not override enforces_own_access_policy.
            # Check their allowlist here so config.yaml-configured allow_from works
            # without requiring a separate {PLATFORM}_ALLOWED_USERS env var.
            adapter = (
                discord_adapter
                if source.platform == Platform.DISCORD
                else self._adapter_for_source(source)
            )
            if adapter is not None:
                extra = getattr(getattr(adapter, "config", None), "extra", None) or {}
                if source.chat_type in {"group", "forum", "channel"}:
                    adapter_allow = extra.get("group_allow_from")
                    if not adapter_allow and source.platform == Platform.DISCORD:
                        # Discord has one user allowlist for both DMs and guild
                        # traffic. Preserve the registered transport adapter's
                        # resolved intake list instead of requiring the generic
                        # group_allow_from key that Discord does not expose.
                        resolved_allow = getattr(adapter, "_allowed_user_ids", None)
                        adapter_allow = list(resolved_allow) if resolved_allow else None
                        if not adapter_allow:
                            adapter_allow = extra.get("allow_from")
                else:
                    adapter_allow = extra.get("allow_from")
                if adapter_allow:
                    allowed = _coerce_allow_set(adapter_allow)
                    if user_id in allowed or "*" in allowed:
                        return True
            # No allowlists configured -- check global allow-all flag
            return _platform_gate_env("GATEWAY_ALLOW_ALL_USERS").lower() in {"true", "1", "yes"}

        # Telegram can optionally authorize group traffic by chat ID.
        # Keep this separate from TELEGRAM_GROUP_ALLOWED_USERS, which gates
        # the sender user ID for group/forum messages.
        if group_chat_allowlist and source.chat_type in {"group", "forum"} and source.chat_id:
            allowed_group_ids = {
                chat_id.strip() for chat_id in group_chat_allowlist.split(",") if chat_id.strip()
            }
            if "*" in allowed_group_ids or source.chat_id in allowed_group_ids:
                return True

        # Backward-compat shim for #15027: prior to PR #17686,
        # TELEGRAM_GROUP_ALLOWED_USERS was (mis)used as a chat-ID allowlist.
        # Values starting with "-" are Telegram chat IDs, not user IDs, so if
        # users still have those in TELEGRAM_GROUP_ALLOWED_USERS we honor them
        # as chat IDs and warn once. The correct var is now
        # TELEGRAM_GROUP_ALLOWED_CHATS.
        if (
            source.platform == Platform.TELEGRAM
            and group_user_allowlist
            and source.chat_type in {"group", "forum"}
            and source.chat_id
        ):
            legacy_chat_ids = {
                v.strip()
                for v in group_user_allowlist.split(",")
                if v.strip().startswith("-")
            }
            if legacy_chat_ids:
                if not getattr(self, "_warned_telegram_group_users_legacy", False):
                    logger.warning(
                        "TELEGRAM_GROUP_ALLOWED_USERS contains chat-ID-shaped values "
                        "(%s). Treating them as chat IDs for backward compatibility. "
                        "Move chat IDs to TELEGRAM_GROUP_ALLOWED_CHATS — the _USERS var "
                        "is now for sender user IDs.",
                        ",".join(sorted(legacy_chat_ids)),
                    )
                    self._warned_telegram_group_users_legacy = True
                if source.chat_id in legacy_chat_ids:
                    return True

        # Check if user is in any allowlist. In group/forum chats,
        # TELEGRAM_GROUP_ALLOWED_USERS is the scoped allowlist and should not
        # imply DM access; TELEGRAM_ALLOWED_USERS remains the platform-wide
        # allowlist and still works everywhere for backward compatibility.
        allowed_ids = set()
        if platform_allowlist:
            allowed_ids.update(uid.strip() for uid in platform_allowlist.split(",") if uid.strip())
        if group_user_allowlist:
            allowed_ids.update(uid.strip() for uid in group_user_allowlist.split(",") if uid.strip())
        if global_allowlist:
            allowed_ids.update(uid.strip() for uid in global_allowlist.split(",") if uid.strip())

        # "*" in any allowlist means allow everyone (consistent with
        # SIGNAL_GROUP_ALLOWED_USERS precedent)
        if "*" in allowed_ids:
            return True

        check_ids = {user_id}
        if "@" in user_id:
            check_ids.add(user_id.split("@")[0])

        # WhatsApp (Baileys + Cloud): resolve phone↔LID / JID aliases so
        # device-suffix and bare-phone allowlist entries match the same principal.
        if source.platform in {Platform.WHATSAPP, Platform.WHATSAPP_CLOUD}:
            normalized_allowed_ids = set()
            for allowed_id in allowed_ids:
                normalized_allowed_ids.update(_expand_whatsapp_auth_aliases(allowed_id))
            if normalized_allowed_ids:
                allowed_ids = normalized_allowed_ids

            check_ids.update(_expand_whatsapp_auth_aliases(user_id))
            normalized_user_id = _normalize_whatsapp_identifier(user_id)
            if normalized_user_id:
                check_ids.add(normalized_user_id)

        # SimpleX: SIMPLEX_ALLOWED_USERS accepts either the numeric contactId
        # or the contact's display name. The adapter sets user_id=contactId for
        # stability across renames, but the SimpleX UI never surfaces the
        # numeric id — operators only see display names, so that's what they
        # naturally put in the env var. Match both so the allowlist works
        # regardless of which form was chosen.
        # Plugin platform: compare by value since Platform.SIMPLEX is not a
        # hardcoded enum member (it's a dynamic plugin platform).
        if (
            source.platform is not None
            and source.platform.value == "simplex"
            and source.user_name
        ):
            check_ids.add(source.user_name)

        return bool(check_ids & allowed_ids)

    def _get_unauthorized_dm_behavior(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> str:
        """Return how unauthorized DMs should be handled for a platform.

        Resolution order:
        1. Explicit per-platform ``unauthorized_dm_behavior`` in config — always wins.
        2. Email defaults to ``"ignore"`` unless explicitly opted into
           pairing. Inboxes may contain arbitrary unread human messages, so
           replying with pairing codes is not a safe platform default.
        3. Explicit global ``unauthorized_dm_behavior`` in config — wins for
           chat-shaped platforms when no per-platform override is set.
        4. When an adapter-level DM policy opts into pairing or silent drop, honor it.
        5. When an allowlist (``PLATFORM_ALLOWED_USERS``,
           ``PLATFORM_GROUP_ALLOWED_USERS`` / ``PLATFORM_GROUP_ALLOWED_CHATS``,
           or ``GATEWAY_ALLOWED_USERS``) is configured, default to ``"ignore"`` —
           the allowlist signals that the owner has deliberately restricted
           access; spamming unknown contacts with pairing codes is both noisy
           and a potential info-leak. (#9337)
        6. No allowlist and no explicit config → ``"pair"`` (open-gateway default).
        """
        config = getattr(self, "config", None)

        # Check for an explicit per-platform override first.
        if config and hasattr(config, "get_unauthorized_dm_behavior") and platform:
            platform_cfg = config.platforms.get(platform) if hasattr(config, "platforms") else None
            if platform_cfg and "unauthorized_dm_behavior" in getattr(platform_cfg, "extra", {}):
                # Operator explicitly configured behavior for this platform — respect it.
                return config.get_unauthorized_dm_behavior(platform)

        # Email is inbox-shaped, not chat-shaped: an agent mailbox may contain
        # unrelated unread human email. Require an explicit per-platform
        # ``unauthorized_dm_behavior: pair`` opt-in before replying to unknown
        # senders with pairing codes. Keep this before the global fallback to
        # match GatewayConfig.get_unauthorized_dm_behavior().
        if platform == Platform.EMAIL:
            return "ignore"

        # Check for an explicit global config override.
        if config and hasattr(config, "unauthorized_dm_behavior"):
            if config.unauthorized_dm_behavior != "pair":  # non-default → explicit override
                return config.unauthorized_dm_behavior

        # Config-driven dm_policy (WeCom / Weixin / Yuanbao / QQBot). An
        # allowlist or disabled DM policy means the operator restricted access,
        # so unauthorized DMs should be dropped silently rather than answered
        # with a pairing code. An explicit pairing policy opts back into codes.
        # Prefer the profile-scoped live adapter's resolved policy in multiplex
        # mode; fall back to the default profile's config.extra.
        if platform:
            dm_policy = self._adapter_dm_policy(platform, profile=profile)
            if not dm_policy and config and hasattr(config, "platforms"):
                platform_cfg = config.platforms.get(platform)
                extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
                if isinstance(extra, dict):
                    dm_policy = str(extra.get("dm_policy") or "").strip().lower()
            if dm_policy == "pairing":
                return "pair"
            if dm_policy in {"allowlist", "disabled"}:
                return "ignore"

        # No explicit override.  Fall back to allowlist-aware default:
        # if any allowlist is configured for this platform, silently drop
        # unauthorized messages instead of sending pairing codes.
        if platform:
            platform_env_map = {
                Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
                Platform.DISCORD:  "DISCORD_ALLOWED_USERS",
                Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
                Platform.WHATSAPP_CLOUD: "WHATSAPP_CLOUD_ALLOWED_USERS",
                Platform.SLACK:    "SLACK_ALLOWED_USERS",
                Platform.SIGNAL:   "SIGNAL_ALLOWED_USERS",
                Platform.EMAIL:    "EMAIL_ALLOWED_USERS",
                Platform.SMS:      "SMS_ALLOWED_USERS",
                Platform.MATTERMOST: "MATTERMOST_ALLOWED_USERS",
                Platform.MATRIX:   "MATRIX_ALLOWED_USERS",
                Platform.DINGTALK: "DINGTALK_ALLOWED_USERS",
                Platform.FEISHU:   "FEISHU_ALLOWED_USERS",
                Platform.WECOM:    "WECOM_ALLOWED_USERS",
                Platform.WECOM_CALLBACK: "WECOM_CALLBACK_ALLOWED_USERS",
                Platform.WEIXIN:   "WEIXIN_ALLOWED_USERS",
                Platform.BLUEBUBBLES: "BLUEBUBBLES_ALLOWED_USERS",
                Platform.QQBOT:    "QQ_ALLOWED_USERS",
            }
            platform_group_env_map = {
                Platform.TELEGRAM: (
                    "TELEGRAM_GROUP_ALLOWED_USERS",
                    "TELEGRAM_GROUP_ALLOWED_CHATS",
                ),
                Platform.QQBOT: ("QQ_GROUP_ALLOWED_USERS",),
            }
            if _platform_gate_env(platform_env_map.get(platform, "")).strip():
                return "ignore"
            for env_key in platform_group_env_map.get(platform, ()):
                if _platform_gate_env(env_key).strip():
                    return "ignore"

        if _platform_gate_env("GATEWAY_ALLOWED_USERS").strip():
            return "ignore"

        return "pair"
