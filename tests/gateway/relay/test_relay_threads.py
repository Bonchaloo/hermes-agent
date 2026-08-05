"""Relay Phase 4 tests — thread lifecycle ops, reply_to enrichment parse,
auto-thread markers, and the hello command manifest.

Covers:
  - create_handoff_thread routes through `thread_create` (op-gated; None
    fallback contract preserved for the handoff watcher);
  - rename_thread routes through `thread_rename` with the
    only_if_current_name guard on the wire (op-gated; False on decline);
  - the relay semantic-rename lane parity: a relay source carrying the
    connector-stamped auto-thread markers satisfies the same field contract
    the native _is_discord_auto_thread_lane reads;
  - _event_from_wire maps reply_to {text,author,is_own} onto the native
    MessageEvent reply-context fields and the auto-thread markers onto
    SessionSource;
  - the ws transport sends command_manifest on the DISCORD hello only;
  - the manifest builder satisfies Discord CHAT_INPUT naming rules.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.command_manifest import build_relay_command_manifest
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.ws_transport import _event_from_wire
from gateway.stream_consumer import GatewayStreamConsumer

from tests.gateway.relay.stub_connector import StubConnector

FULL_OPS = (
    "send",
    "edit",
    "typing",
    "get_chat_info",
    "thread_create",
    "thread_rename",
    "send_media",
    "prompt",
)


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Discord",
        max_message_length=2000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
        supported_ops=FULL_OPS,
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


def _adapter(**desc_kw) -> tuple[RelayAdapter, StubConnector]:
    stub = StubConnector(make_desc(**desc_kw))
    adapter = RelayAdapter(PlatformConfig(), make_desc(**desc_kw), transport=stub)
    return adapter, stub


# ── thread_create (handoff) ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_handoff_thread_routes_thread_create():
    adapter, stub = _adapter()
    stub.next_send_result = {"success": True}  # unused; thread op has own arm

    async def send_outbound(action, *, platform=None):
        stub.sent.append(action)
        stub.sent_platforms.append(platform)
        return {"success": True, "thread_id": "th77"}

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    thread_id = await adapter.create_handoff_thread("chan1", "fix the build")
    assert thread_id == "th77"
    action = stub.sent[-1]
    assert action["op"] == "thread_create"
    assert action["chat_id"] == "chan1"
    assert action["thread_name"] == "fix the build"


# ── rename_thread (semantic rename) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_rename_thread_carries_the_no_clobber_guard():
    adapter, stub = _adapter()
    ok = await adapter.rename_thread(
        "th1", "Fix the build", only_if_current_name="Hermes"
    )
    assert ok is True
    action = stub.sent[-1]
    assert action["op"] == "thread_rename"
    assert action["message_id"] == "th1"
    assert action["thread_name"] == "Fix the build"
    assert action["only_if_current_name"] == "Hermes"
    # chat_id defaults to the thread id (Discord ignores it; Telegram callers
    # pass parent_chat_id explicitly).
    assert action["chat_id"] == "th1"


@pytest.mark.asyncio
async def test_rename_thread_parent_chat_and_gating():
    adapter, stub = _adapter()
    await adapter.rename_thread("42", "topic", parent_chat_id="-100999")
    assert stub.sent[-1]["chat_id"] == "-100999"
    assert "only_if_current_name" not in stub.sent[-1]

    gated, gated_stub = _adapter(supported_ops=("send",))
    assert await gated.rename_thread("42", "x") is False
    assert gated_stub.sent == []


@pytest.mark.asyncio
async def test_rename_thread_prefers_connector_owned_guard():
    """The relay lane sends only_if_connector_created (connector resolves the
    no-clobber guard from its own created-name memory) instead of the fragile
    cross-repo only_if_current_name string."""
    adapter, stub = _adapter()
    ok = await adapter.rename_thread(
        "th9", "Real Session Title", prefer_connector_created=True
    )
    assert ok is True
    action = stub.sent[-1]
    assert action["op"] == "thread_rename"
    assert action["only_if_connector_created"] is True
    # The fragile string guard is NOT sent when the connector owns the check.
    assert "only_if_current_name" not in action


@pytest.mark.asyncio
async def test_rename_thread_connector_guard_takes_precedence_over_string():
    """prefer_connector_created wins even if a legacy string is also passed."""
    adapter, stub = _adapter()
    await adapter.rename_thread(
        "th9",
        "Title",
        prefer_connector_created=True,
        only_if_current_name="ignored initial words",
    )
    action = stub.sent[-1]
    assert action["only_if_connector_created"] is True
    assert "only_if_current_name" not in action


@pytest.mark.asyncio
async def test_rename_thread_resolves_scope_from_parent_chat_not_thread():
    """The connector's egress guard resolves the owning tenant from the
    outbound metadata's scope_id / user_id, and the adapter's discriminator
    caches are keyed by the PARENT channel chat_id (learned at inbound), never
    the thread id. A rename that passes parent_chat_id must carry that
    discriminator; a rename keyed only on the thread id must not — reproducing
    the live decline ("target not routed to an onboarded tenant") and its fix.
    """
    adapter, stub = _adapter()
    # Simulate the inbound-learned scope for the PARENT channel only.
    adapter._scope_by_chat["chan-parent"] = "guild-123"

    # Fix: pass the parent chat id -> scope_id resolves.
    await adapter.rename_thread(
        "th-9",
        "Real Title",
        prefer_connector_created=True,
        parent_chat_id="chan-parent",
    )
    fixed = stub.sent[-1]
    assert fixed["metadata"].get("scope_id") == "guild-123"

    # Regression shape: keyed on the thread id alone (no parent) -> no scope_id,
    # which is exactly what made the connector decline the op.
    await adapter.rename_thread(
        "th-9",
        "Real Title",
        prefer_connector_created=True,
    )
    unscoped = stub.sent[-1]
    assert "scope_id" not in unscoped["metadata"]


# ── the relay semantic-rename lane (marker parity) ───────────────────────


# ── reply_to wire parse ──────────────────────────────────────────────────


def test_event_from_wire_reply_to_absent_and_partial():
    plain = _event_from_wire(
        {
            "text": "hi",
            "message_type": "text",
            "source": {"platform": "telegram", "chat_id": "5", "chat_type": "dm"},
        }
    )
    assert plain.reply_to_text is None
    assert plain.reply_to_is_own_message is False
    partial = _event_from_wire(
        {
            "text": "re",
            "message_type": "text",
            "source": {"platform": "whatsapp", "chat_id": "1", "chat_type": "dm"},
            "reply_to_message_id": "wamid.x",
            "reply_to": {"author": "Alice"},  # text leg missed the cache
        }
    )
    assert partial.reply_to_author_name == "Alice"
    assert partial.reply_to_text is None


# ── hello command manifest ───────────────────────────────────────────────




# ── auto-thread routing feedback (send-result thread_id) ─────────────────


@pytest.mark.asyncio
async def test_send_captures_auto_thread_feedback():
    """A send result carrying thread_id + auto_thread_name (the connector's
    auto-thread egress policy routed the reply into a thread it created)
    populates auto_thread_info_for_chat for the semantic-rename lane."""
    adapter, stub = _adapter()

    async def send_outbound(action, *, platform=None):
        stub.sent.append(action)
        return {
            "success": True,
            "message_id": "m1",
            "thread_id": "th-auto-1",
            "auto_thread_name": "What is a duck",
        }

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    result = await adapter.send("chan1", "quack", reply_to="msg-1")
    assert result.success
    assert adapter.auto_thread_info_for_chat("chan1", "msg-1") == (
        "th-auto-1",
        "What is a duck",
    )
    # Plain results (no auto-thread) leave no feedback for other chats.
    assert adapter.auto_thread_info_for_chat("chan-other", "msg-1") is None


@pytest.mark.asyncio
async def test_auto_thread_feedback_is_correlated_and_consumed_once():
    adapter, stub = _adapter()

    async def send_outbound(action, *, platform=None):
        return {
            "success": True,
            "message_id": "m1",
            "thread_id": "th-current",
            "auto_thread_name": "Initial title",
        }

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    await adapter.send("chan-parent", "reply", reply_to="msg-current")

    assert adapter.auto_thread_info_for_chat("chan-parent", "msg-old") is None
    assert adapter.auto_thread_info_for_chat("chan-parent", "msg-current") == (
        "th-current",
        "Initial title",
    )
    assert adapter.auto_thread_info_for_chat("chan-parent", "msg-current") is None


@pytest.mark.asyncio
async def test_send_without_thread_feedback_leaves_no_info():
    adapter, stub = _adapter()

    async def send_outbound(action, *, platform=None):
        return {"success": True, "message_id": "m2"}

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    await adapter.send("chan2", "hello", reply_to="msg-2")
    assert adapter.auto_thread_info_for_chat("chan2", "msg-2") is None


def test_stream_consumer_scopes_relay_feedback_generation_per_response():
    adapter, _stub = _adapter()
    first = GatewayStreamConsumer(adapter, "chan2")
    second = GatewayStreamConsumer(adapter, "chan2")

    first_generation = first._metadata_for_send()["_response_generation"]
    assert first._metadata_for_send()["_response_generation"] == first_generation
    assert second._metadata_for_send()["_response_generation"] != first_generation


@pytest.mark.asyncio
async def test_canonical_generation_drives_feedback_and_never_reaches_wire():
    adapter, stub = _adapter()
    stub.next_send_result = {
        "success": True,
        "message_id": "m-generation",
        "thread_id": "thread-generation",
        "auto_thread_name": "Initial generation title",
    }

    await adapter.send(
        "chan2",
        "hello",
        reply_to="trigger",
        metadata={
            "_response_generation": "generation-1",
            "_relay_logical_platform": "discord",
            "visible": "yes",
        },
    )

    assert stub.sent[-1]["metadata"] == {"visible": "yes"}
    assert stub.sent_platforms[-1] == "discord"
    assert adapter.auto_thread_info_for_chat(
        "chan2", "trigger", response_generation="generation-1"
    ) == ("thread-generation", "Initial generation title")
    assert adapter.auto_thread_info_for_chat(
        "chan2", "trigger", response_generation="generation-1"
    ) is None


@pytest.mark.asyncio
async def test_same_anchor_generations_are_isolated_and_feedback_cache_is_bounded():
    adapter, stub = _adapter()

    async def send_outbound(action, *, platform=None):
        generation = action["content"]
        return {
            "success": True,
            "message_id": f"message-{generation}",
            "thread_id": f"thread-{generation}",
            "auto_thread_name": f"title-{generation}",
        }

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    for index in range(300):
        generation = f"generation-{index}"
        await adapter.send(
            "same-chat",
            generation,
            reply_to="same-anchor",
            metadata={"_response_generation": generation},
        )

    assert adapter.auto_thread_info_for_chat(
        "same-chat", "same-anchor", response_generation="generation-0"
    ) is None
    assert adapter.auto_thread_info_for_chat(
        "same-chat", "same-anchor", response_generation="generation-299"
    ) == ("thread-generation-299", "title-generation-299")
    assert adapter.auto_thread_info_for_chat(
        "same-chat", "same-anchor", response_generation="generation-299"
    ) is None


@pytest.mark.asyncio
async def test_concurrent_same_anchor_feedback_is_generation_scoped_consume_once_and_bounded():
    adapter, stub = _adapter()

    async def send_one(index: int):
        generation = f"concurrent-{index}"

        async def send_outbound(action, *, platform=None):
            stub.sent.append(action)
            return {
                "success": True,
                "message_id": f"message-{index}",
                "thread_id": f"thread-{index}",
                "auto_thread_name": f"title-{index}",
            }

        original = stub.send_outbound
        stub.send_outbound = send_outbound  # type: ignore[method-assign]
        try:
            await adapter.send(
                "same-chat",
                f"response-{index}",
                reply_to="same-anchor",
                metadata={"_response_generation": generation},
            )
        finally:
            stub.send_outbound = original  # type: ignore[method-assign]
        return generation

    # The production cache stays bounded even when many responses share one anchor.
    generations = []
    for index in range(300):
        generations.append(await send_one(index))
    assert len(adapter._auto_thread_by_chat) <= 256

    generation = generations[-1]
    consumed = await asyncio.gather(
        *[
            asyncio.to_thread(
                adapter.auto_thread_info_for_chat,
                "same-chat",
                "same-anchor",
                response_generation=generation,
            )
            for _ in range(16)
        ]
    )
    assert consumed.count(("thread-299", "title-299")) == 1
    assert consumed.count(None) == 15


@pytest.mark.asyncio
async def test_every_relay_metadata_egress_strips_private_keys_behaviorally():
    adapter, stub = _adapter()
    private_metadata = {
        "_response_generation": "canonical",
        "_hermes_stream_generation": "legacy",
        "_future_private": "must-not-leak",
        "visible": "yes",
    }

    await adapter.send("chat", "send", metadata=private_metadata)
    await adapter.send_for_platform(
        Platform.DISCORD, "chat", "explicit", metadata=private_metadata
    )
    await adapter.edit_message("chat", "message", "edit", metadata=private_metadata)
    await adapter.send_typing("chat", metadata=private_metadata)
    await adapter._send_media(
        "chat",
        media_kind="image",
        source="https://example.test/image.png",
        source_is_path=False,
        metadata=private_metadata,
    )
    await adapter._send_prompt(
        "chat",
        prompt_kind="clarify",
        text="pick",
        prompt_id="prompt-id",
        options=[],
        metadata=private_metadata,
    )
    await adapter.send_follow_up(
        "session", "discord.interaction_token", "follow up", private_metadata
    )

    serialized = [*stub.sent, *stub.follow_ups]
    assert {action["op"] for action in serialized} >= {
        "send",
        "edit",
        "typing",
        "send_media",
        "prompt",
        "follow_up",
    }
    for action in serialized:
        assert action["metadata"].get("visible") == "yes"
        assert not any(key.startswith("_") for key in action["metadata"])


@pytest.mark.asyncio
async def test_split_send_preserves_first_auto_thread_feedback_until_consumed():
    adapter, stub = _adapter()
    results = iter([
        {
            "success": True,
            "message_id": "chunk-1",
            "thread_id": "th-split",
            "auto_thread_name": "Initial split title",
        },
        {"success": True, "message_id": "chunk-2"},
    ])

    async def send_outbound(action, *, platform=None):
        stub.sent.append(action)
        return next(results)

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    generation = {"_hermes_stream_generation": 7}
    await adapter.send("chan2", "first", reply_to="msg-2", metadata=generation)
    await adapter.send("chan2", "second", reply_to="msg-2", metadata=generation)

    assert all(
        "_hermes_stream_generation" not in action["metadata"]
        for action in stub.sent
    )
    assert adapter.auto_thread_info_for_chat(
        "chan2", "msg-2", response_generation="7"
    ) == (
        "th-split",
        "Initial split title",
    )
    assert adapter.auto_thread_info_for_chat(
        "chan2", "msg-2", response_generation="7"
    ) is None


@pytest.mark.asyncio
async def test_no_feedback_generation_does_not_steal_other_generation():
    adapter, stub = _adapter()
    results = iter([
        {
            "success": True,
            "message_id": "chunk-1",
            "thread_id": "th-stale",
            "auto_thread_name": "Stale title",
        },
        {"success": True, "message_id": "chunk-2"},
    ])

    async def send_outbound(action, *, platform=None):
        return next(results)

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    await adapter.send(
        "chan2",
        "first response",
        reply_to="msg-2",
        metadata={"_hermes_stream_generation": 7},
    )
    await adapter.send(
        "chan2",
        "later response",
        reply_to="msg-2",
        metadata={"_hermes_stream_generation": 8},
    )

    assert adapter.auto_thread_info_for_chat(
        "chan2", "msg-2", response_generation="8"
    ) is None
    assert adapter.auto_thread_info_for_chat(
        "chan2", "msg-2", response_generation="7"
    ) == ("th-stale", "Stale title")


@pytest.mark.asyncio
async def test_positive_feedback_is_isolated_by_response_generation():
    adapter, stub = _adapter()
    results = iter(
        [
            {
                "success": True,
                "thread_id": "thread-A",
                "auto_thread_name": "A",
            },
            {
                "success": True,
                "thread_id": "thread-B",
                "auto_thread_name": "B",
            },
        ]
    )

    async def send_outbound(_action, *, platform=None):
        return next(results)

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    await adapter.send(
        "chan2",
        "response A",
        reply_to="msg-2",
        metadata={"_hermes_stream_generation": "gen-A"},
    )
    await adapter.send(
        "chan2",
        "response B",
        reply_to="msg-2",
        metadata={"_hermes_stream_generation": "gen-B"},
    )

    assert adapter.auto_thread_info_for_chat(
        "chan2", "msg-2", response_generation="gen-A"
    ) == ("thread-A", "A")
    assert adapter.auto_thread_info_for_chat(
        "chan2", "msg-2", response_generation="gen-B"
    ) == ("thread-B", "B")


@pytest.mark.asyncio
async def test_ungenerated_retry_clears_stale_auto_thread_feedback():
    adapter, stub = _adapter()
    results = iter([
        {
            "success": True,
            "message_id": "chunk-1",
            "thread_id": "th-stale",
            "auto_thread_name": "Stale title",
        },
        {"success": True, "message_id": "chunk-2"},
    ])

    async def send_outbound(action, *, platform=None):
        return next(results)

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    await adapter.send("chan2", "first response", reply_to="msg-2")
    await adapter.send("chan2", "retry", reply_to="msg-2")

    assert adapter.auto_thread_info_for_chat("chan2", "msg-2") is None


@pytest.mark.asyncio
async def test_auto_thread_feedback_is_bounded():
    adapter, stub = _adapter()

    async def send_outbound(action, *, platform=None):
        return {
            "success": True,
            "message_id": "m",
            "thread_id": f"th-{action['chat_id']}",
            "auto_thread_name": "n",
        }

    stub.send_outbound = send_outbound  # type: ignore[method-assign]
    for i in range(300):
        await adapter.send(f"c{i}", "x", reply_to=f"m{i}")
    assert len(adapter._auto_thread_by_chat) <= 256
    # Newest entries survive the bound.
    assert adapter.auto_thread_info_for_chat("c299", "m299") == ("th-c299", "n")


# ── title-turn rename: registration shape-gate + fire-time cache poll ────


def _mk_runner_stub():
    """Minimal object carrying the three GatewayRunner methods under test."""
    import asyncio as _asyncio
    from gateway.run import GatewayRunner

    class _Stub:
        _is_relay_discord_channel_lane = GatewayRunner._is_relay_discord_channel_lane
        _relay_auto_thread_info = GatewayRunner._relay_auto_thread_info
        _is_discord_auto_thread_lane = GatewayRunner._is_discord_auto_thread_lane
        _sanitize_discord_thread_title = GatewayRunner._sanitize_discord_thread_title
        _rename_discord_auto_thread_for_session_title = (
            GatewayRunner._rename_discord_auto_thread_for_session_title
        )

        def __init__(self, adapter):
            self.adapters = {Platform.RELAY: adapter}

        def _adapter_for_source(self, source):
            return self.adapters.get(Platform.RELAY)

    return _Stub


def _relay_channel_source():
    from types import SimpleNamespace

    return SimpleNamespace(
        platform=Platform.DISCORD,
        chat_id="chan-parent",
        chat_type="group",
        thread_id=None,
        message_id="msg-current",
        delivered_via_upstream_relay=True,
        auto_thread_created=False,
        auto_thread_initial_name=None,
    )


def test_relay_channel_lane_shape_gate():
    from types import SimpleNamespace
    from gateway.config import Platform as P

    stub = _mk_runner_stub()(adapter=None)
    src = _relay_channel_source()
    assert stub._is_relay_discord_channel_lane(src) is True
    # thread events, DMs, and native (non-relay) events do not match
    assert (
        stub._is_relay_discord_channel_lane(
            SimpleNamespace(**{**src.__dict__, "thread_id": "t1"})
        )
        is False
    )
    assert (
        stub._is_relay_discord_channel_lane(
            SimpleNamespace(**{**src.__dict__, "chat_type": "dm"})
        )
        is False
    )
    assert (
        stub._is_relay_discord_channel_lane(
            SimpleNamespace(**{**src.__dict__, "delivered_via_upstream_relay": False})
        )
        is False
    )


@pytest.mark.asyncio
async def test_relay_auto_thread_info_prefers_prospective_thread_id():
    """When the connector stamps prospective_thread_id, the rename lane uses it
    directly (deterministic, per-thread) and does NOT consult the per-chat
    send-result cache — the empty initial-name marker defers no-clobber to the
    connector's own created-name guard."""
    from types import SimpleNamespace

    adapter, _ = _adapter()
    # Poison the per-chat cache with a DIFFERENT (stale sibling) thread to prove
    # the prospective id wins and the cache is not read.
    adapter._auto_thread_by_chat["chan-parent"] = ("th-STALE", "old words")
    runner = _mk_runner_stub()(adapter)
    src = SimpleNamespace(
        **{**_relay_channel_source().__dict__, "prospective_thread_id": "th-B"}
    )
    assert runner._relay_auto_thread_info(src) == ("th-B", "")


@pytest.mark.asyncio
async def test_sibling_threads_in_one_channel_each_rename_to_own_thread():
    """Two auto-threads spawned from the SAME parent channel must each rename
    to their OWN thread id. Before the prospective_thread_id fix the per-chat
    cache held one slot, so only the first thread renamed (staging repro
    2026-08-02: thread A renamed, sibling thread B stuck at raw text)."""
    from types import SimpleNamespace

    adapter, _ = _adapter()
    renames: list = []

    async def rename_thread(
        thread_id,
        name,
        *,
        only_if_current_name=None,
        prefer_connector_created=False,
        parent_chat_id=None,
    ):
        renames.append((thread_id, name, prefer_connector_created, parent_chat_id))
        return True

    adapter.rename_thread = rename_thread  # type: ignore[method-assign]
    runner = _mk_runner_stub()(adapter)
    base = _relay_channel_source().__dict__

    # A and B share the parent channel but carry distinct prospective thread ids.
    src_a = SimpleNamespace(**{**base, "prospective_thread_id": "th-A"})
    src_b = SimpleNamespace(**{**base, "prospective_thread_id": "th-B"})
    await runner._rename_discord_auto_thread_for_session_title(
        src_a, "sessA", "Sea Shanty Draft"
    )
    await runner._rename_discord_auto_thread_for_session_title(
        src_b, "sessB", "Exotic Short Story"
    )
    # Each renamed ITS OWN thread, via the connector-owned guard, passing the
    # parent channel id for tenant discriminator resolution.
    assert renames == [
        ("th-A", "Sea Shanty Draft", True, "chan-parent"),
        ("th-B", "Exotic Short Story", True, "chan-parent"),
    ]


@pytest.mark.asyncio
async def test_title_rename_polls_feedback_that_arrives_late():
    """The auto-title races delivery: feedback lands AFTER the rename lane
    starts. The lane must poll the adapter cache and still rename."""
    import asyncio

    adapter, stub_conn = _adapter()
    renames: list = []

    async def rename_thread(
        thread_id,
        name,
        *,
        only_if_current_name=None,
        prefer_connector_created=False,
        parent_chat_id=None,
    ):
        renames.append((thread_id, name, prefer_connector_created, parent_chat_id))
        return True

    adapter.rename_thread = rename_thread  # type: ignore[method-assign]
    runner = _mk_runner_stub()(adapter)
    src = _relay_channel_source()

    async def land_feedback_late():
        await asyncio.sleep(0.7)  # past the first poll tick
        adapter._auto_thread_by_chat[("chan-parent", "msg-current", "")] = (
            "th-9",
            "Initial words",
        )

    task = asyncio.create_task(land_feedback_late())
    await runner._rename_discord_auto_thread_for_session_title(
        src, "sess1", "Debugging the flux capacitor"
    )
    await task
    # Relay lane uses the connector-owned guard (prefer_connector_created=True),
    # not the fragile cross-repo initial-name string. It MUST pass the PARENT
    # channel chat_id so the connector's egress guard can resolve the tenant
    # (the discriminator caches are keyed by the parent channel, not the thread;
    # omitting it made the connector decline "target not routed to an onboarded
    # tenant" — the live failure on staging 2026-08-01).
    assert renames == [
        ("th-9", "Debugging the flux capacitor", True, "chan-parent")
    ]


@pytest.mark.asyncio
async def test_title_rename_ignores_stale_feedback_from_prior_message():
    """A title for a new parent-channel turn must never consume feedback from
    an earlier triggering message while current-send feedback is still racing."""
    import asyncio

    adapter, _ = _adapter()
    renames: list = []

    async def rename_thread(
        thread_id,
        name,
        *,
        only_if_current_name=None,
        prefer_connector_created=False,
        parent_chat_id=None,
    ):
        renames.append((thread_id, name, prefer_connector_created, parent_chat_id))
        return True

    adapter.rename_thread = rename_thread  # type: ignore[method-assign]
    adapter._auto_thread_by_chat[("chan-parent", "msg-old", "")] = (
        "th-stale",
        "Old title",
    )
    runner = _mk_runner_stub()(adapter)
    src = _relay_channel_source()
    src.message_id = "msg-current"

    async def land_current_feedback():
        await asyncio.sleep(0.7)
        adapter._auto_thread_by_chat[("chan-parent", "msg-current", "")] = (
            "th-current",
            "Current initial title",
        )

    task = asyncio.create_task(land_current_feedback())
    await runner._rename_discord_auto_thread_for_session_title(
        src, "sess-current", "Current semantic title"
    )
    await task

    assert renames == [
        ("th-current", "Current semantic title", True, "chan-parent")
    ]


@pytest.mark.asyncio
async def test_native_title_rename_keeps_native_adapter_signature():
    """The native adapter must not receive relay-only keyword arguments."""
    from types import SimpleNamespace

    native_calls: list[tuple[str, str, str | None]] = []

    class NativeAdapter:
        async def rename_thread(
            self,
            thread_id,
            name,
            *,
            only_if_current_name=None,
        ):
            native_calls.append((thread_id, name, only_if_current_name))
            return True

    runner = _mk_runner_stub()(NativeAdapter())
    source = SimpleNamespace(
        platform=Platform.DISCORD,
        chat_id="native-thread",
        chat_type="thread",
        thread_id="native-thread",
        delivered_via_upstream_relay=False,
        auto_thread_created=True,
        auto_thread_initial_name="raw prompt",
    )

    await runner._rename_discord_auto_thread_for_session_title(
        source,
        "sess-native",
        "Semantic title",
    )

    assert native_calls == [
        ("native-thread", "Semantic title", "raw prompt")
    ]


@pytest.mark.asyncio
async def test_title_rename_true_miss_noops(monkeypatch):
    """No feedback ever arrives (connector didn't auto-thread): no rename."""
    import gateway.run as run_mod

    adapter, _ = _adapter()
    renames: list = []

    async def rename_thread(thread_id, name, **kw):
        renames.append(thread_id)
        return True

    adapter.rename_thread = rename_thread  # type: ignore[method-assign]
    runner = _mk_runner_stub()(adapter)
    src = _relay_channel_source()
    # Shrink the poll loop for test speed: 20 ticks of 0.5s -> patch sleep.
    orig_sleep = run_mod.asyncio.sleep

    async def fast_sleep(_s):
        await orig_sleep(0)

    monkeypatch.setattr(run_mod.asyncio, "sleep", fast_sleep)
    await runner._rename_discord_auto_thread_for_session_title(
        src, "sess1", "A title"
    )
    assert renames == []
