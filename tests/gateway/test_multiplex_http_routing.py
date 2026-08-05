"""Phase 1: HTTP-inbound /p/<profile>/ routing for the webhook adapter."""
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, build_session_key


class TestSessionSourceProfileField:
    def test_non_discord_chat_type_and_profile_roundtrip(self):
        s = SessionSource(
            platform=Platform.WEBHOOK if hasattr(Platform, "WEBHOOK") else Platform.TELEGRAM,
            chat_id="c1",
            chat_type="webhook",
            profile="coder",
        )
        restored = SessionSource.from_dict(s.to_dict())
        assert restored.platform == s.platform
        assert restored.chat_type == "webhook"
        assert restored.profile == "coder"

    @pytest.mark.parametrize("chat_type", [None, 123, "", "   \t"])
    def test_supplied_malformed_non_discord_chat_type_fails_closed(self, chat_type):
        with pytest.raises(ValueError):
            SessionSource.from_dict(
                {
                    "platform": "webhook",
                    "chat_id": "c1",
                    "chat_type": chat_type,
                }
            )


class TestWebhookProfileResolution:
    """_resolve_request_profile validates the /p/<profile>/ prefix."""

    def _adapter(self, multiplex: bool, served=("default", "coder")):
        from gateway.platforms.webhook import WebhookAdapter, _PROFILE_REJECTED

        class _FakeReq:
            def __init__(self, profile):
                self.match_info = {"profile": profile} if profile is not None else {}

        cfg = GatewayConfig(multiplex_profiles=multiplex)

        class _Runner:
            config = cfg

        # Construct minimally; we only call _resolve_request_profile.
        adapter = WebhookAdapter.__new__(WebhookAdapter)
        adapter.gateway_runner = _Runner()
        return adapter, _FakeReq, _PROFILE_REJECTED, served

    def test_no_prefix_returns_none(self):
        adapter, Req, _REJ, _ = self._adapter(multiplex=True)
        assert adapter._resolve_request_profile(Req(None)) is None


