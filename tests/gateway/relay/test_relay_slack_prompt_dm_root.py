"""Slack relay: interactive prompts follow the turn's thread stamp.

The threading MODE (flat DM vs thread-per-message) is decided in exactly ONE
place: run.py's ``_resolve_progress_thread_id``, which reads
``platforms.slack.extra.reply_in_thread`` and encodes the verdict into the
outbound ``metadata`` stamp:

  * flat mode  -> the synthetic self-anchor is suppressed in run.py, so prompt
    metadata arrives with NO ``thread_id`` and the card posts at the DM root;
  * thread-per-message (default) -> ``metadata.thread_id`` is stamped for the
    whole turn; on the FIRST turn it legitimately equals the triggering
    message's ts (the synthetic root IS the thread).

The prompt lane must TRUST that stamp, like ``_resolve_reply_to_for_send``
does. Re-deriving the mode here (the old unconditional
``thread_id == message_id`` strip) exiled the approval card and its
resolved-state swap to the DM root while progress bubbles honoured the thread
(the 2026-07-27 mixed-placement report).

These are behaviour-contract tests: they assert how the outbound ``prompt``
frame relates to the inherited thread metadata (the invariant the connector
depends on), not a snapshot. They drive the REAL ``RelayAdapter`` +
``StubConnector`` end to end.
"""

from __future__ import annotations

import copy
import json
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.ws_transport import WebSocketRelayTransport, _event_from_wire
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.source_provenance import SourceProvenanceRegistry

from tests.gateway.relay.stub_connector import StubConnector

FULL_OPS = ("send", "edit", "typing", "get_chat_info", "send_media", "prompt", "react")


def _slack_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="slack",
        label="Slack",
        max_message_length=4000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="mrkdwn",
        len_unit="chars",
        supported_ops=FULL_OPS,
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


def _wire(
    chat_id: str,
    chat_type: str,
    *,
    user_id="U1",
    scope_id=None,
    platform=Platform.SLACK,
):
    """A RelayAdapter fronting Slack, with inbound scope + chat_type captured."""
    stub = StubConnector(_slack_desc())
    adapter = RelayAdapter(PlatformConfig(), _slack_desc(), transport=stub)
    src = SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        scope_id=scope_id,
    )
    adapter._capture_scope(
        MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)
    )
    return adapter, stub


def _last_prompt(stub) -> dict:
    prompts = [f for f in stub.sent if f["op"] == "prompt"]
    assert prompts, "expected a prompt op on the wire"
    return prompts[-1]


# ---------------------------------------------------------------------------
# Flat mode: run.py stamps NO thread_id -> the card posts at the DM root.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exec_approval_flat_mode_posts_at_dm_root():
    """Flat-DM turn (reply_in_thread=false): run.py suppressed the synthetic
    anchor upstream, so prompt metadata has no thread_id and none appears on
    the wire — the card posts at the DM root."""
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    md = {"message_id": "1700000000.000100", "scope_id": "T1"}
    result = await adapter.send_exec_approval(
        "D1", "rm -rf /tmp/x", "sess:1", description="deletes files", metadata=md
    )
    assert result.success is True
    frame = _last_prompt(stub)
    meta = frame["metadata"] or {}
    assert "thread_id" not in meta
    assert "thread_ts" not in meta
    # reply_to on the outbound action stays unset — a root-level post.
    assert frame["reply_to"] is None
    # Tenant scope is preserved untouched (egress routing must not break).
    assert meta.get("scope_id") == "T1"


# ---------------------------------------------------------------------------
# Thread-per-message mode, end-to-end placement contract: run.py stamps the
# turn's thread (first turn: the triggering message's own ts) and the adapter
# forwards prompt metadata UNTOUCHED — no re-derivation, no strip. Mixed
# placement (progress threaded, card at root) was the 2026-07-27 regression.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exec_approval_forwards_run_py_thread_stamp_untouched():
    """The adapter must forward run.py's thread stamp verbatim: the approval
    card posts INTO the stamped thread. Any adapter-side re-derivation or
    strip exiled the card to the home channel (2026-07-27 report)."""
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    md = {
        "thread_id": "1700000000.000100",
        "message_id": "1700000000.000100",
        "scope_id": "T1",
    }
    result = await adapter.send_exec_approval(
        "D1", "rm -rf /tmp/x", "sess:1", description="deletes files", metadata=md
    )
    assert result.success is True
    frame = _last_prompt(stub)
    meta = frame["metadata"] or {}
    assert meta.get("thread_id") == "1700000000.000100", (
        "first-turn self-anchor is the thread root; the prompt must honour it"
    )
    assert meta.get("scope_id") == "T1"


@pytest.mark.asyncio
async def test_clarify_forwards_run_py_thread_stamp_untouched():
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    md = {
        "thread_id": "1700000000.000200",
        "message_id": "1700000000.000200",
        "scope_id": "T1",
    }
    result = await adapter.send_clarify(
        "D1", "Which env?", ["prod", "staging"], "cl-1", "sess:1", metadata=md
    )
    assert result.success is True
    frame = _last_prompt(stub)
    meta = frame["metadata"] or {}
    assert meta.get("thread_id") == "1700000000.000200"
    assert meta.get("scope_id") == "T1"


@pytest.mark.asyncio
async def test_slash_confirm_forwards_run_py_thread_stamp_untouched():
    """The forward-untouched rule covers every prompt surface (single
    _send_prompt choke point)."""
    adapter, stub = _wire("D1", "dm")
    md = {"thread_id": "1700000000.000300", "message_id": "1700000000.000300"}
    await adapter.send_slash_confirm(
        "D1", "Reload MCP", "invalidates cache", "s", "cf-1", metadata=md
    )
    frame = _last_prompt(stub)
    assert (frame["metadata"] or {}).get("thread_id") == "1700000000.000300"


# ---------------------------------------------------------------------------
# Regression guards: a REAL thread and non-DM / non-Slack chats are untouched
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exec_approval_in_real_thread_keeps_thread_id():
    """A DM prompt raised inside a REAL thread (thread_id distinct from the
    triggering message ts) stays in that thread."""
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    md = {
        "thread_id": "1699000000.999000",
        "message_id": "1700000000.000100",
        "scope_id": "T1",
    }
    await adapter.send_exec_approval("D1", "cmd", "s", metadata=md)
    frame = _last_prompt(stub)
    assert frame["metadata"]["thread_id"] == "1699000000.999000"


@pytest.mark.asyncio
async def test_channel_approval_keeps_thread_id():
    """A Slack CHANNEL prompt keeps its thread_id (autoThread / real thread)."""
    adapter, stub = _wire("C1", "channel", scope_id="T1")
    md = {
        "thread_id": "1700000000.000400",
        "message_id": "1700000000.000400",
        "scope_id": "T1",
    }
    await adapter.send_exec_approval("C1", "cmd", "s", metadata=md)
    frame = _last_prompt(stub)
    assert frame["metadata"]["thread_id"] == "1700000000.000400"


@pytest.mark.asyncio
async def test_non_slack_dm_approval_keeps_thread_id():
    """A non-Slack relay DM keeps thread_id (its connector owns its own
    threading semantics)."""
    adapter, stub = _wire("dc1", "dm", platform=Platform.DISCORD)
    md = {"thread_id": "9000", "message_id": "9000"}
    await adapter.send_exec_approval("dc1", "cmd", "s", metadata=md)
    frame = _last_prompt(stub)
    assert frame["metadata"]["thread_id"] == "9000"


# ---------------------------------------------------------------------------
# Rich status: the relay advertises Slack's text status line and carries
# the live per-tool phrase on the typing frame (native set_status_text parity).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_slack_relay_advertises_status_text():
    adapter, _stub = _wire("D1", "dm")
    assert adapter.supports_status_text is True


@pytest.mark.asyncio
async def test_non_slack_relay_does_not_advertise_status_text():
    stub = StubConnector(_slack_desc(platform="discord"))
    adapter = RelayAdapter(
        PlatformConfig(), _slack_desc(platform="discord"), transport=stub
    )
    assert adapter.supports_status_text is False


@pytest.mark.asyncio
async def test_typing_carries_live_status_phrase():
    """set_status_text() -> the next typing frame carries the phrase as
    content; clearing it (None) reverts to a content-less heartbeat frame
    (never an empty string, which is Slack's explicit clear)."""
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    adapter.set_status_text("D1", "is running pytest…")
    await adapter.send_typing("D1", metadata={"scope_id": "T1"})
    typing = [f for f in stub.sent if f["op"] == "typing"]
    assert typing and typing[-1].get("content") == "is running pytest…"

    adapter.set_status_text("D1", None)
    await adapter.send_typing("D1", metadata={"scope_id": "T1"})
    typing = [f for f in stub.sent if f["op"] == "typing"]
    assert "content" not in typing[-1], (
        "cleared phrase must omit content (empty string means CLEAR on Slack)"
    )


# ---------------------------------------------------------------------------
# Status thread anchor: typing frames synthesize the per-message thread
# root in thread-per-message mode (the status line is thread-only on Slack).
# ---------------------------------------------------------------------------
def _wire_with_ts(chat_id, chat_type, message_id, **kw):
    adapter, stub = _wire(chat_id, chat_type, **kw)
    src = SessionSource(
        platform=Platform.SLACK, chat_id=chat_id, chat_type=chat_type,
        user_id="U1", scope_id=kw.get("scope_id"),
    )
    ev = MessageEvent(
        text="hi", source=src, message_type=MessageType.TEXT, message_id=message_id
    )
    adapter._capture_scope(ev)
    return adapter, stub


@pytest.mark.asyncio
async def test_typing_synthesizes_thread_anchor_in_thread_mode():
    """Top-level DM turn, thread-per-message mode: the typing frame gains the
    triggering ts as thread_id so the connector's setStatus targets the
    per-message thread instead of no-oping threadless."""
    adapter, stub = _wire_with_ts("D1", "dm", "1700.0042")
    await adapter.send_typing("D1", metadata=None)
    typing = [f for f in stub.sent if f["op"] == "typing"]
    assert typing and typing[-1]["metadata"].get("thread_id") == "1700.0042"


@pytest.mark.asyncio
async def test_typing_flat_mode_status_anchors_to_trigger_ts_by_default():
    """Flat-DM liveliness: the STATUS still anchors to the triggering ts
    (renders in the footer space, no message artifact) while replies stay
    flat — the send lane strips its anchors, so placement cannot inherit this."""
    adapter, stub = _wire_with_ts("D1", "dm", "1700.0042")
    adapter.config.extra = {"reply_in_thread": False}
    await adapter.send_typing("D1", metadata=None)
    typing = [f for f in stub.sent if f["op"] == "typing"]
    assert typing and typing[-1]["metadata"].get("thread_id") == "1700.0042"


@pytest.mark.asyncio
async def test_typing_anchors_unconditionally_in_both_modes():
    """Liveliness is not a preference: the status anchors whenever an inbound
    ts exists, regardless of reply_in_thread. Placement safety comes from the
    send-side anchor strip, not from suppressing the status."""
    for extra in ({}, {"slack": {"reply_in_thread": False}}):
        adapter, stub = _wire_with_ts("D1", "dm", "1700.0042")
        adapter.config.extra = extra
        await adapter.send_typing("D1", metadata=None)
        typing = [f for f in stub.sent if f["op"] == "typing"]
        assert typing and typing[-1]["metadata"].get("thread_id") == "1700.0042"


@pytest.mark.asyncio
async def test_flat_mode_sends_stay_flat_with_status_anchor_active():
    """The liveliness anchor must NOT leak into reply placement: sends in
    flat mode still strip the synthetic anchor (send-lane contract)."""
    adapter, stub = _wire_with_ts("D1", "dm", "1700.0042")
    adapter.config.extra = {"reply_in_thread": False}
    await adapter.send_typing("D1", metadata=None)
    await adapter.send("D1", "the answer", reply_to="1700.0042")
    frame = [f for f in stub.sent if f["op"] == "send"][-1]
    assert frame["reply_to"] is None
    assert "thread_id" not in (frame["metadata"] or {})


@pytest.mark.asyncio
async def test_typing_honours_real_thread_anchor():
    """Metadata that already names a thread wins over the synthetic cache."""
    adapter, stub = _wire_with_ts("D1", "dm", "1700.0042")
    await adapter.send_typing("D1", metadata={"thread_id": "1699.9000"})
    typing = [f for f in stub.sent if f["op"] == "typing"]
    assert typing[-1]["metadata"]["thread_id"] == "1699.9000"


@pytest.mark.asyncio
async def test_stop_typing_clear_targets_same_synthesized_thread():
    """The clear frame targets the same synthesized thread as the heartbeat
    (else the status line sticks)."""
    adapter, stub = _wire_with_ts("D1", "dm", "1700.0042")
    await adapter.send_typing("D1", metadata=None)
    await adapter.stop_typing("D1", metadata=None)
    clears = [
        f for f in stub.sent if f["op"] == "typing" and f.get("content") == ""
    ]
    assert clears and clears[-1]["metadata"].get("thread_id") == "1700.0042"


# ---------------------------------------------------------------------------
# Session keying: a top-level Slack DM message gets its own ts stamped as
# source.thread_id (native inbound parity) so each message keys a FRESH
# session in thread-per-message mode; flat mode and real threads untouched.
# ---------------------------------------------------------------------------
def _inbound_event(chat_id="D1", message_id="1700.0100", thread_id=None):
    src = SessionSource(
        platform=Platform.SLACK, chat_id=chat_id, chat_type="dm",
        user_id="U1", scope_id="T1", thread_id=thread_id,
    )
    return MessageEvent(
        text="hi", source=src, message_type=MessageType.TEXT,
        message_id=message_id,
    )


def _authenticated_stack(epoch="authenticated-current"):
    transport = WebSocketRelayTransport(
        "wss://relay.invalid/ws",
        "slack",
        "B1",
        gateway_id="gateway-1",
        upgrade_secret="secret-1",
    )
    transport._authenticated_connection_epoch = epoch
    adapter = RelayAdapter(
        PlatformConfig(enabled=True),
        _slack_desc(),
        transport=transport,
    )
    transport.set_inbound_handler(adapter._on_inbound)
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.RELAY: adapter}
    runner._profile_adapters = {}
    runner.pairing_store = MagicMock()
    return transport, adapter, runner


def _wire_top_level_dm(
    transport=None,
    *,
    message_id="1700.0100",
    thread_id=None,
):
    source = {
        "platform": "slack",
        "chat_id": "D1",
        "chat_type": "dm",
        "user_id": "U1",
        "scope_id": "T1",
    }
    if thread_id is not None:
        source["thread_id"] = thread_id
    return _event_from_wire(
        {
            "text": "hello",
            "message_type": "text",
            "message_id": message_id,
            "source": source,
        },
        transport=transport,
    )


async def _authorize_through_relay_ingress(adapter, runner, event):
    verdicts = []

    async def authorize(normalized_event):
        verdicts.append(runner._is_user_authorized(normalized_event.source))

    adapter.handle_message = authorize
    await adapter._on_inbound(event)
    assert len(verdicts) == 1
    return verdicts[0]


def test_top_level_dm_gets_session_thread_stamp():
    adapter, _ = _wire("D1", "dm")
    ev = _inbound_event(message_id="1700.0100")
    adapter._stamp_slack_session_thread(ev)
    assert ev.source.thread_id == "1700.0100"


@pytest.mark.asyncio
async def test_authenticated_top_level_dm_survives_ingress_normalization_and_authorization():
    """Production order stays authorized: transport register -> Slack stamp -> runner."""
    transport = WebSocketRelayTransport(
        "wss://relay.invalid/ws",
        "slack",
        "B1",
        gateway_id="gateway-1",
        upgrade_secret="secret-1",
    )
    adapter = RelayAdapter(
        PlatformConfig(enabled=True),
        _slack_desc(),
        transport=transport,
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.RELAY: adapter}
    runner._profile_adapters = {}
    runner.pairing_store = MagicMock()

    observed = {}

    async def authorize_after_normalization(event):
        epoch = transport.authenticated_connection_epoch
        observed["thread_id"] = event.source.thread_id
        observed["verified"] = transport._source_provenance.verifies(
            event.source,
            epoch=epoch,
        )
        observed["authorized"] = runner._is_user_authorized(event.source)

    adapter.handle_message = authorize_after_normalization
    transport.set_inbound_handler(adapter._on_inbound)
    await transport._handle_frame(
        json.dumps(
            {
                "type": "descriptor",
                "descriptor": json.loads(_slack_desc().to_json()),
            }
        )
    )
    assert transport.authenticated_connection_epoch

    await transport._handle_frame(
        json.dumps(
            {
                "type": "inbound",
                "event": {
                    "text": "hello",
                    "message_type": "text",
                    "message_id": "1700.0100",
                    "source": {
                        "platform": "slack",
                        "chat_id": "D1",
                        "chat_type": "dm",
                        "user_id": "U1",
                        "scope_id": "T1",
                    },
                },
            }
        )
    )

    assert observed["thread_id"] == "1700.0100"
    assert observed["verified"] is True
    assert observed["authorized"] is True


@pytest.mark.asyncio
async def test_unregistered_wire_source_is_not_resealed_and_denies():
    transport, adapter, runner = _authenticated_stack()
    event = _wire_top_level_dm(None)

    assert await _authorize_through_relay_ingress(adapter, runner, event) is False
    assert event.source.thread_id == "1700.0100"
    assert not transport._source_provenance.verifies(
        event.source,
        epoch=transport.authenticated_connection_epoch,
    )


@pytest.mark.asyncio
async def test_forged_relay_marker_is_not_resealed_and_denies():
    transport, adapter, runner = _authenticated_stack()
    event = _inbound_event()
    event.source.delivered_via_upstream_relay = True
    event.source.transport_route = "relay"

    assert await _authorize_through_relay_ingress(adapter, runner, event) is False
    assert event.source.thread_id == "1700.0100"
    assert not transport._source_provenance.verifies(
        event.source,
        epoch=transport.authenticated_connection_epoch,
    )


@pytest.mark.asyncio
async def test_copied_registered_source_is_not_resealed_and_denies():
    transport, adapter, runner = _authenticated_stack()
    registered = _wire_top_level_dm(transport)
    event = copy.copy(registered)
    event.source = copy.copy(registered.source)

    assert await _authorize_through_relay_ingress(adapter, runner, event) is False
    assert event.source.thread_id == "1700.0100"
    assert not transport._source_provenance.verifies(
        event.source,
        epoch=transport.authenticated_connection_epoch,
    )


@pytest.mark.asyncio
async def test_stale_epoch_source_is_not_resealed_and_denies():
    transport, adapter, runner = _authenticated_stack("authenticated-old")
    event = _wire_top_level_dm(transport)
    transport._authenticated_connection_epoch = "authenticated-current"

    assert await _authorize_through_relay_ingress(adapter, runner, event) is False
    assert event.source.thread_id == "1700.0100"
    assert not transport._source_provenance.verifies(
        event.source,
        epoch=transport.authenticated_connection_epoch,
    )


@pytest.mark.asyncio
async def test_absent_epoch_source_is_not_resealed_and_denies():
    transport, adapter, runner = _authenticated_stack()
    event = _wire_top_level_dm(transport)
    transport._authenticated_connection_epoch = None

    assert await _authorize_through_relay_ingress(adapter, runner, event) is False
    assert event.source.thread_id == "1700.0100"
    assert not transport._source_provenance.verifies(event.source, epoch=None)


@pytest.mark.asyncio
async def test_already_mutated_source_is_not_resealed_and_denies():
    transport, adapter, runner = _authenticated_stack()
    event = _wire_top_level_dm(transport)
    event.source.user_id = "forged-user"

    assert await _authorize_through_relay_ingress(adapter, runner, event) is False
    assert event.source.thread_id == "1700.0100"
    assert not transport._source_provenance.verifies(
        event.source,
        epoch=transport.authenticated_connection_epoch,
    )


@pytest.mark.asyncio
async def test_source_registered_by_different_transport_is_not_resealed_and_denies():
    owner_transport, _owner_adapter, _owner_runner = _authenticated_stack()
    event = _wire_top_level_dm(owner_transport)
    foreign_transport, foreign_adapter, foreign_runner = _authenticated_stack()

    assert (
        await _authorize_through_relay_ingress(
            foreign_adapter,
            foreign_runner,
            event,
        )
        is False
    )
    assert event.source.thread_id == "1700.0100"
    assert not foreign_transport._source_provenance.verifies(
        event.source,
        epoch=foreign_transport.authenticated_connection_epoch,
    )


@pytest.mark.asyncio
async def test_source_normalized_by_wrong_adapter_on_same_transport_denies():
    transport, owner_adapter, owner_runner = _authenticated_stack()
    event = _wire_top_level_dm(transport)
    wrong_adapter = RelayAdapter(
        PlatformConfig(enabled=True),
        _slack_desc(),
        transport=transport,
    )

    assert transport._inbound == owner_adapter._on_inbound
    assert (
        await _authorize_through_relay_ingress(
            wrong_adapter,
            owner_runner,
            event,
        )
        is False
    )
    assert event.source.thread_id == "1700.0100"
    assert not transport._source_provenance.verifies(
        event.source,
        epoch=transport.authenticated_connection_epoch,
    )


@pytest.mark.asyncio
async def test_epoch_change_during_normalization_prevents_reseal_and_denies():
    transport, adapter, runner = _authenticated_stack()
    event = _wire_top_level_dm(transport)
    stamp = adapter._stamp_slack_session_thread

    def stamp_after_epoch_change(event):
        transport._authenticated_connection_epoch = "authenticated-replaced"
        stamp(event)

    adapter._stamp_slack_session_thread = stamp_after_epoch_change

    assert await _authorize_through_relay_ingress(adapter, runner, event) is False
    assert event.source.thread_id == "1700.0100"
    assert not transport._source_provenance.verifies(
        event.source,
        epoch=transport.authenticated_connection_epoch,
    )


@pytest.mark.asyncio
async def test_registry_change_during_normalization_prevents_reseal_and_denies():
    transport, adapter, runner = _authenticated_stack()
    event = _wire_top_level_dm(transport)
    stamp = adapter._stamp_slack_session_thread
    replacement_registry = SourceProvenanceRegistry()

    def stamp_after_registry_change(event):
        transport._source_provenance = replacement_registry
        stamp(event)

    adapter._stamp_slack_session_thread = stamp_after_registry_change

    assert await _authorize_through_relay_ingress(adapter, runner, event) is False
    assert event.source.thread_id == "1700.0100"
    assert not replacement_registry.verifies(
        event.source,
        epoch=transport.authenticated_connection_epoch,
    )


@pytest.mark.asyncio
async def test_callback_replacement_during_normalization_prevents_reseal_and_denies(
    monkeypatch,
):
    transport, adapter, runner = _authenticated_stack()
    event = _wire_top_level_dm(transport)
    registry = transport._source_provenance
    epoch = transport.authenticated_connection_epoch

    assert getattr(transport._inbound, "__self__", None) is adapter
    assert getattr(transport._inbound, "__func__", None) is adapter._on_inbound.__func__
    assert registry.verifies(event.source, epoch=epoch)
    assert runner._is_user_authorized(event.source) is True

    register = MagicMock(wraps=registry.register)
    monkeypatch.setattr(registry, "register", register)
    stamp = adapter._stamp_slack_session_thread

    async def replacement_inbound(_event):
        return None

    def stamp_then_replace_callback(event):
        stamp(event)
        monkeypatch.setattr(transport, "_inbound", replacement_inbound)

    monkeypatch.setattr(
        adapter,
        "_stamp_slack_session_thread",
        stamp_then_replace_callback,
    )

    assert await _authorize_through_relay_ingress(adapter, runner, event) is False
    assert event.source.thread_id == "1700.0100"
    register.assert_not_called()
    assert not registry.verifies(event.source, epoch=epoch)
    assert runner._is_user_authorized(event.source) is False


@pytest.mark.asyncio
async def test_transport_replacement_during_normalization_prevents_reseal_and_denies(
    monkeypatch,
):
    transport, adapter, runner = _authenticated_stack()
    event = _wire_top_level_dm(transport)
    foreign_transport, _foreign_adapter, _foreign_runner = _authenticated_stack(
        "authenticated-foreign"
    )
    registry = transport._source_provenance
    foreign_registry = foreign_transport._source_provenance
    epoch = transport.authenticated_connection_epoch
    foreign_epoch = foreign_transport.authenticated_connection_epoch

    assert adapter._transport is transport
    assert getattr(transport._inbound, "__self__", None) is adapter
    assert getattr(transport._inbound, "__func__", None) is adapter._on_inbound.__func__
    assert registry.verifies(event.source, epoch=epoch)
    assert runner._is_user_authorized(event.source) is True

    register = MagicMock(wraps=registry.register)
    foreign_register = MagicMock(wraps=foreign_registry.register)
    monkeypatch.setattr(registry, "register", register)
    monkeypatch.setattr(foreign_registry, "register", foreign_register)
    stamp = adapter._stamp_slack_session_thread

    def stamp_then_replace_transport(event):
        stamp(event)
        monkeypatch.setattr(adapter, "_transport", foreign_transport)

    monkeypatch.setattr(
        adapter,
        "_stamp_slack_session_thread",
        stamp_then_replace_transport,
    )

    assert await _authorize_through_relay_ingress(adapter, runner, event) is False
    assert event.source.thread_id == "1700.0100"
    register.assert_not_called()
    foreign_register.assert_not_called()
    assert not registry.verifies(event.source, epoch=epoch)
    assert not foreign_registry.verifies(event.source, epoch=foreign_epoch)
    assert runner._is_user_authorized(event.source) is False


@pytest.mark.asyncio
async def test_registered_real_thread_stays_verified_without_reseal(monkeypatch):
    transport, adapter, runner = _authenticated_stack()
    event = _wire_top_level_dm(
        transport,
        message_id="1700.0300",
        thread_id="1700.0100",
    )
    registry = transport._source_provenance
    epoch = transport.authenticated_connection_epoch

    assert getattr(transport._inbound, "__self__", None) is adapter
    assert getattr(transport._inbound, "__func__", None) is adapter._on_inbound.__func__
    assert registry.verifies(event.source, epoch=epoch)
    assert runner._is_user_authorized(event.source) is True

    register = MagicMock(wraps=registry.register)
    monkeypatch.setattr(registry, "register", register)

    assert await _authorize_through_relay_ingress(adapter, runner, event) is True
    assert event.source.thread_id == "1700.0100"
    register.assert_not_called()
    assert registry.verifies(event.source, epoch=epoch)
    assert runner._is_user_authorized(event.source) is True


@pytest.mark.asyncio
async def test_later_mutation_revokes_resealed_source_again():
    transport, adapter, runner = _authenticated_stack()
    event = _wire_top_level_dm(transport)

    assert await _authorize_through_relay_ingress(adapter, runner, event) is True
    assert transport._source_provenance.verifies(
        event.source,
        epoch=transport.authenticated_connection_epoch,
    )

    event.source.user_id = "later-forged-user"

    assert runner._is_user_authorized(event.source) is False
    assert not transport._source_provenance.verifies(
        event.source,
        epoch=transport.authenticated_connection_epoch,
    )


def test_two_top_level_messages_key_distinct_sessions():
    from gateway.session import build_session_key
    adapter, _ = _wire("D1", "dm")
    e1 = _inbound_event(message_id="1700.0100")
    e2 = _inbound_event(message_id="1700.0200")
    adapter._stamp_slack_session_thread(e1)
    adapter._stamp_slack_session_thread(e2)
    k1 = build_session_key(e1.source)
    k2 = build_session_key(e2.source)
    assert k1 != k2, "each top-level message must be its own session"


def test_real_thread_reply_keeps_its_thread_session():
    adapter, _ = _wire("D1", "dm")
    ev = _inbound_event(message_id="1700.0300", thread_id="1700.0100")
    adapter._stamp_slack_session_thread(ev)
    assert ev.source.thread_id == "1700.0100", (
        "an in-thread reply must keep resolving to its thread's session"
    )


def test_flat_mode_keeps_shared_dm_session():
    adapter, _ = _wire("D1", "dm")
    adapter.config.extra = {"reply_in_thread": False}
    ev = _inbound_event(message_id="1700.0400")
    adapter._stamp_slack_session_thread(ev)
    assert ev.source.thread_id is None, (
        "flat mode: shared rolling DM session (steer/queue) is intended UX"
    )


def test_nested_relay_slack_config_subset_wins():
    """Enterprise knob shape: platforms.relay.extra.slack.reply_in_thread."""
    adapter, _ = _wire("D1", "dm")
    adapter.config.extra = {"slack": {"reply_in_thread": False}}
    assert adapter._effective_reply_in_thread() is False
    adapter.config.extra = {"slack": {"reply_in_thread": True}}
    assert adapter._effective_reply_in_thread() is True
    # Legacy flat key still honoured when no nested object exists.
    adapter.config.extra = {"reply_in_thread": False}
    assert adapter._effective_reply_in_thread() is False
    # Default: thread-per-message.
    adapter.config.extra = {}
    assert adapter._effective_reply_in_thread() is True


# ---------------------------------------------------------------------------
# Cross-module boundary pin (review 2026-07-28): the adapter deliberately has
# NO prompt-side strip — flat-mode placement depends entirely on run.py's
# _resolve_progress_thread_id suppressing the synthetic self-anchor upstream.
# If that suppression regresses, prompt cards silently thread again. These
# tests pin the boundary in BOTH modes so the coupling is load-bearing.
# ---------------------------------------------------------------------------
def test_run_py_suppresses_self_anchor_in_flat_mode():
    from gateway.run import _resolve_progress_thread_id

    # Flat mode + synthetic self-anchor (thread_id == own message id) => None:
    # prompt/progress metadata arrives at the adapter with NO thread anchor.
    assert (
        _resolve_progress_thread_id(
            "slack", "1700.001", "1700.001", reply_in_thread=False
        )
        is None
    )
    # Flat mode + REAL thread (ids differ) => the real thread survives.
    assert (
        _resolve_progress_thread_id(
            "slack", "1699.000", "1700.001", reply_in_thread=False
        )
        == "1699.000"
    )


def test_run_py_keeps_self_anchor_in_thread_mode():
    from gateway.run import _resolve_progress_thread_id

    # Thread-per-message mode: the first-turn self-anchor IS the thread root
    # and must flow through to the adapter unchanged.
    assert (
        _resolve_progress_thread_id(
            "slack", "1700.001", "1700.001", reply_in_thread=True
        )
        == "1700.001"
    )
    # No source thread at all: Slack synthesizes the root from the message id.
    assert (
        _resolve_progress_thread_id("slack", None, "1700.001", reply_in_thread=True)
        == "1700.001"
    )


# ---------------------------------------------------------------------------
# Native parity escape hatch: platforms.relay.extra.slack.
# dm_top_level_threads_as_sessions=false keeps threaded replies but ONE
# rolling DM session (mirrors native SlackAdapter._dm_top_level_threads_as_sessions).
# Without the knob, reply_in_thread alone couples placement AND session
# keying — a posture native operators can express and relay ones could not.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_session_stamp_opt_out_keeps_rolling_dm_session():
    adapter, stub = _wire("D1", "dm")
    adapter.config.extra = {
        "slack": {
            "reply_in_thread": True,
            "dm_top_level_threads_as_sessions": False,
        }
    }
    event = _inbound_event("D1", message_id="1700.0001", thread_id=None)
    adapter._stamp_slack_session_thread(event)
    assert getattr(event.source, "thread_id", None) is None, (
        "opt-out: top-level DM must NOT be stamped — one rolling session"
    )


@pytest.mark.asyncio
async def test_session_stamp_default_remains_per_message():
    adapter, stub = _wire("D1", "dm")
    adapter.config.extra = {"slack": {"reply_in_thread": True}}
    event = _inbound_event("D1", message_id="1700.0002", thread_id=None)
    adapter._stamp_slack_session_thread(event)
    assert getattr(event.source, "thread_id", None) == "1700.0002", (
        "default (native parity): per-message sessions stay on"
    )
