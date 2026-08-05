"""WebSocketRelayTransport against a real in-process WebSocket server.

Exercises the production transport over an actual ``websockets`` server (no
mock socket): handshake (hello -> descriptor), inbound frame -> handler,
outbound request/response correlation, and follow_up routing. Proves the wire
framing (newline-delimited JSON) and the request/response future plumbing work
end to end on a live socket.

Skipped cleanly if the optional ``websockets`` dependency is absent.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio

from gateway.relay.ws_transport import WebSocketRelayTransport, WEBSOCKETS_AVAILABLE

pytestmark = pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")

if WEBSOCKETS_AVAILABLE:
    import websockets


DESCRIPTOR = {
    "contract_version": 1,
    "platform": "discord",
    "label": "Discord",
    "max_message_length": 2000,
    "supports_draft_streaming": False,
    "supports_edit": True,
    "supports_threads": True,
    "markdown_dialect": "discord",
    "len_unit": "chars",
}


class _StubConnectorServer:
    """Minimal connector: answers hello with a descriptor, echoes outbound."""

    def __init__(self):
        self.received: list[dict] = []
        self._server = None
        self.url = ""
        # Push channel: tests set this to a frame dict to deliver inbound.
        self._to_push: list[dict] = []

    async def start(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        sock = next(iter(self._server.sockets))
        port = sock.getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws):
        async for raw in ws:
            for line in str(raw).split("\n"):
                if not line.strip():
                    continue
                frame = json.loads(line)
                self.received.append(frame)
                await self._on_frame(ws, frame)

    async def _on_frame(self, ws, frame):
        ftype = frame.get("type")
        if ftype == "hello":
            await ws.send(json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n")
            # Deliver any queued inbound frames right after handshake.
            for f in self._to_push:
                await ws.send(json.dumps(f) + "\n")
        elif ftype == "outbound":
            action = frame.get("action", {})
            # Echo a successful result correlated by requestId.
            result = {"success": True, "message_id": f"srv-{action.get('op')}"}
            await ws.send(
                json.dumps({"type": "outbound_result", "requestId": frame["requestId"], "result": result})
                + "\n"
            )


@pytest_asyncio.fixture
async def server():
    srv = _StubConnectorServer()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.mark.asyncio
async def test_handshake_negotiates_descriptor(server):
    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    await t.connect()
    try:
        desc = await t.handshake()
        assert desc.platform == "discord"
        assert desc.max_message_length == 2000
        # The hello carried the platform + botId.
        hello = next(f for f in server.received if f["type"] == "hello")
        assert hello["platform"] == "discord"
        assert hello["botId"] == "appShared"
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_inbound_frame_reaches_handler(server):
    server._to_push = [
        {
            "type": "inbound",
            "event": {
                "text": "hello from connector",
                "message_type": "text",
                "source": {"platform": "discord", "chat_id": "chan1", "chat_type": "group", "scope_id": "guildA"},
            },
            "bufferId": "buf-1",
        }
    ]
    received = []
    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    t.set_inbound_handler(lambda ev: received.append(ev) or asyncio.sleep(0))
    await t.connect()
    try:
        await t.handshake()
        # Give the reader a tick to deliver the pushed inbound frame.
        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].text == "hello from connector"
        assert received[0].source.scope_id == "guildA"
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_authenticated_epoch_changes_and_stale_sources_are_revoked(server):
    from unittest.mock import MagicMock

    from gateway.config import Platform, PlatformConfig
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.run import GatewayRunner

    server._to_push = [
        {
            "type": "inbound",
            "event": {
                "text": "epoch event",
                "message_type": "text",
                "source": {
                    "platform": "discord",
                    "chat_id": "chan-epoch",
                    "chat_type": "group",
                    "user_id": "owner-epoch",
                    "scope_id": "guild-epoch",
                },
            },
        }
    ]
    received = []
    transport = WebSocketRelayTransport(
        server.url,
        "discord",
        "appShared",
        gateway_id="gateway-1",
        upgrade_secret="secret-1",
    )
    transport.set_inbound_handler(
        lambda event: received.append(event) or asyncio.sleep(0)
    )
    adapter = RelayAdapter(
        PlatformConfig(enabled=True),
        descriptor=CapabilityDescriptor.from_json(json.dumps(DESCRIPTOR)),
        transport=transport,
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.RELAY: adapter}
    runner._profile_adapters = {}
    runner.pairing_store = MagicMock()

    await transport.connect()
    try:
        await transport.handshake()
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.01)
        first_source = received[-1].source
        first_epoch = transport.authenticated_connection_epoch
        assert first_epoch
        assert runner._is_user_authorized(first_source) is True

        await transport.disconnect()
        assert transport.authenticated_connection_epoch is None
        assert runner._is_user_authorized(first_source) is False

        received.clear()
        await transport.connect()
        await transport.handshake()
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.01)
        second_source = received[-1].source
        assert transport.authenticated_connection_epoch
        assert transport.authenticated_connection_epoch != first_epoch
        assert runner._is_user_authorized(first_source) is False
        assert runner._is_user_authorized(second_source) is True
    finally:
        await transport.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_frame",
    [
        [],
        {"type": "inbound", "event": []},
        {"type": "inbound", "event": {"source": []}},
        {
            "type": "inbound",
            "event": {
                "text": "bad reply",
                "message_type": "text",
                "source": {
                    "platform": "discord",
                    "chat_id": "bad",
                    "chat_type": "group",
                },
                "reply_to": [],
            },
        },
        {
            "type": "inbound",
            "event": {
                "text": "bad platform",
                "message_type": "text",
                "source": {
                    "platform": [],
                    "chat_id": "bad",
                    "chat_type": "group",
                },
            },
        },
        {
            "type": "inbound",
            "event": {
                "text": "bad type",
                "message_type": [],
                "source": {
                    "platform": "discord",
                    "chat_id": "bad",
                    "chat_type": "group",
                },
            },
        },
        {
            "type": "inbound",
            "event": {
                "text": "bad metadata",
                "message_type": "text",
                "metadata": [],
                "source": {
                    "platform": "discord",
                    "chat_id": "bad",
                    "chat_type": "group",
                },
            },
        },
    ],
    ids=(
        "frame",
        "event",
        "source",
        "reply_to",
        "platform",
        "message_type",
        "metadata",
    ),
)
async def test_malformed_inbound_is_dropped_without_ending_current_reader(
    server,
    bad_frame,
):
    """Every malformed event is isolated; the next frame uses the same socket."""
    valid = {
        "type": "inbound",
        "event": {
            "text": "valid after malformed",
            "message_type": "text",
            "source": {
                "platform": "discord",
                "chat_id": "good",
                "chat_type": "group",
            },
        },
    }
    server._to_push = [bad_frame, valid]
    received = []
    transport = WebSocketRelayTransport(server.url, "discord", "appShared")
    transport.set_inbound_handler(
        lambda event: received.append(event) or asyncio.sleep(0)
    )

    await transport.connect()
    try:
        await transport.handshake()
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.01)
        assert [event.text for event in received] == ["valid after malformed"]
        assert transport._reader is not None and not transport._reader.done()
    finally:
        await transport.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_source",
    [
        {"platform": "webhook", "chat_id": "bad"},
        {"platform": "webhook", "chat_id": "bad", "chat_type": None},
        {"platform": "webhook", "chat_id": "bad", "chat_type": 7},
        {"platform": "webhook", "chat_id": "bad", "chat_type": ""},
        {"platform": "webhook", "chat_id": "bad", "chat_type": "   \t"},
    ],
    ids=("missing", "none", "non-string", "empty", "whitespace"),
)
async def test_non_discord_chat_type_boundary_isolates_bad_event_then_dispatches_valid(
    server,
    bad_source,
):
    """Every platform requires an explicit nonempty string, not Discord's enum."""
    server._to_push = [
        {
            "type": "inbound",
            "event": {
                "text": "malformed webhook",
                "message_type": "text",
                "source": bad_source,
            },
        },
        {
            "type": "inbound",
            "event": {
                "text": "valid webhook",
                "message_type": "text",
                "source": {
                    "platform": "webhook",
                    "chat_id": "good",
                    "chat_type": "webhook",
                },
            },
        },
    ]
    received = []
    transport = WebSocketRelayTransport(server.url, "discord", "appShared")
    transport.set_inbound_handler(
        lambda event: received.append(event) or asyncio.sleep(0)
    )

    await transport.connect()
    try:
        await transport.handshake()
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.01)
        assert [event.text for event in received] == ["valid webhook"]
        assert received[0].source.chat_type == "webhook"
        assert transport._reader is not None and not transport._reader.done()
    finally:
        await transport.disconnect()


@pytest.mark.asyncio
async def test_discord_chat_type_outside_closed_set_isolated_from_next_event(server):
    server._to_push = [
        {
            "type": "inbound",
            "event": {
                "text": "bad Discord type",
                "message_type": "text",
                "source": {
                    "platform": "discord",
                    "chat_id": "bad",
                    "chat_type": "webhook",
                },
            },
        },
        {
            "type": "inbound",
            "event": {
                "text": "valid Discord type",
                "message_type": "text",
                "source": {
                    "platform": "discord",
                    "chat_id": "good",
                    "chat_type": "channel",
                },
            },
        },
    ]
    received = []
    transport = WebSocketRelayTransport(server.url, "discord", "appShared")
    transport.set_inbound_handler(
        lambda event: received.append(event) or asyncio.sleep(0)
    )

    await transport.connect()
    try:
        await transport.handshake()
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.01)
        assert [event.text for event in received] == ["valid Discord type"]
        assert transport._reader is not None and not transport._reader.done()
    finally:
        await transport.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("guild_id", ["guild-name", "123456789012345678"])
async def test_relay_deprecated_guild_alias_falls_back_to_scope(server, guild_id):
    server._to_push = [
        {
            "type": "inbound",
            "event": {
                "text": "legacy guild alias",
                "message_type": "text",
                "source": {
                    "platform": "discord",
                    "chat_id": "channel-1",
                    "chat_type": "channel",
                    "scope_id": None,
                    "guild_id": guild_id,
                },
            },
        }
    ]
    received = []
    transport = WebSocketRelayTransport(server.url, "discord", "appShared")
    transport.set_inbound_handler(
        lambda event: received.append(event) or asyncio.sleep(0)
    )

    await transport.connect()
    try:
        await transport.handshake()
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.01)
        assert len(received) == 1
        assert received[0].source.scope_id == guild_id
        assert received[0].source.guild_id == guild_id
    finally:
        await transport.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_aliases",
    [
        {"scope_id": "guild-name", "guild_id": []},
        {"scope_id": "guild-name", "guild_id": ""},
        {"scope_id": "guild-name", "guild_id": "   \t"},
        {"scope_id": "guild-name", "guild_id": "other-guild"},
        {"scope_id": [], "guild_id": "guild-name"},
        {"scope_id": "", "guild_id": "guild-name"},
        {"scope_id": "   \t", "guild_id": "guild-name"},
    ],
    ids=(
        "malformed-guild",
        "empty-guild",
        "whitespace-guild",
        "conflicting-guild",
        "malformed-scope",
        "empty-scope",
        "whitespace-scope",
    ),
)
async def test_relay_alias_boundary_isolates_bad_event_then_dispatches_valid(
    server,
    bad_aliases,
):
    server._to_push = [
        {
            "type": "inbound",
            "event": {
                "text": "bad aliases",
                "message_type": "text",
                "source": {
                    "platform": "discord",
                    "chat_id": "bad",
                    "chat_type": "channel",
                    **bad_aliases,
                },
            },
        },
        {
            "type": "inbound",
            "event": {
                "text": "valid aliases",
                "message_type": "text",
                "source": {
                    "platform": "discord",
                    "chat_id": "good",
                    "chat_type": "channel",
                    "scope_id": "123456789012345678",
                    "guild_id": "123456789012345678",
                },
            },
        },
    ]
    received = []
    transport = WebSocketRelayTransport(server.url, "discord", "appShared")
    transport.set_inbound_handler(
        lambda event: received.append(event) or asyncio.sleep(0)
    )

    await transport.connect()
    try:
        await transport.handshake()
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.01)
        assert [event.text for event in received] == ["valid aliases"]
        assert received[0].source.scope_id == "123456789012345678"
        assert received[0].source.guild_id == "123456789012345678"
        assert transport._reader is not None and not transport._reader.done()
    finally:
        await transport.disconnect()


# ── Phase 7 Unit 7d-B: terminal 4401 (opt-out revocation) ────────────────────


class _Revoking4401Server:
    """Connector stub that, on hello, optionally sends a descriptor and then
    closes the socket with application code 4401 (unauthorized) — the shape of a
    connector that has revoked this gateway's per-gateway secret (opt-out)."""

    def __init__(self, *, send_descriptor_first: bool):
        self._server = None
        self.url = ""
        self._send_descriptor_first = send_descriptor_first

    async def start(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        port = next(iter(self._server.sockets)).getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws):
        async for raw in ws:
            for line in str(raw).split("\n"):
                if not line.strip():
                    continue
                frame = json.loads(line)
                if frame.get("type") == "hello":
                    if self._send_descriptor_first:
                        await ws.send(
                            json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n"
                        )
                        # Let the descriptor flush + be processed before the close.
                        await asyncio.sleep(0.05)
                    # Close with 4401 (the connector's "unauthorized" close).
                    await ws.close(code=4401, reason="unauthorized")
                    return


@pytest.mark.asyncio
async def test_4401_after_handshake_is_terminal_no_reconnect():
    """A 4401 close AFTER a successful handshake = a revoked credential (opt-out):
    the transport latches auth_revoked and does NOT spin the reconnect supervisor."""
    srv = _Revoking4401Server(send_descriptor_first=True)
    await srv.start()
    try:
        t = WebSocketRelayTransport(
            srv.url, "discord", "appShared",
            gateway_id="gw-x", upgrade_secret="secret-x",
            reconnect=True, reconnect_backoff_s=0.05,
        )
        await t.connect()
        await t.handshake()  # records _handshake_succeeded
        # Wait for the server's 4401 close to propagate through the read loop.
        for _ in range(100):
            if t.auth_revoked:
                break
            await asyncio.sleep(0.02)
        assert t.auth_revoked is True
        # Terminal: no reconnect supervisor was spawned.
        assert t._supervisor is None
        # Give a reconnect (if it were going to happen) time to NOT happen.
        await asyncio.sleep(0.2)
        assert t._supervisor is None
    finally:
        await t.disconnect()
        await srv.stop()


