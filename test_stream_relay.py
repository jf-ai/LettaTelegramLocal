"""Tests for the background stream relay — SSE parsing and Telegram-shaped output.

Driven with asyncio.run() inside each test (the project's convention, see test_bridge.py)
so no pytest-asyncio configuration is needed.
"""
import asyncio

import httpx
import pytest

import stream_relay
from stream_relay import StreamRelay

AGENT_ID = "agent-1"


def _relay(emitted: list[str], suppress=None) -> StreamRelay:
    async def send(text: str) -> None:
        emitted.append(text)

    return StreamRelay("http://server", AGENT_ID, send=send, suppress=suppress)


def _serve(monkeypatch, body: str, status: int = 200) -> dict:
    """Point the relay's httpx client at a canned SSE body."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(status, content=body.encode(), headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=None)
    monkeypatch.setattr(stream_relay.httpx, "AsyncClient", lambda **kwargs: client)
    return captured


class TestEventRelay:
    def test_relays_bookends_text_and_tool_calls(self, monkeypatch):
        """Token deltas are coalesced into one message per text part."""
        captured = _serve(monkeypatch, (
            'event: RunStartedEvent\ndata: {"prompt": "run the tests"}\n\n'
            'event: PartStartEvent\ndata: {"part": {"part_kind": "text", "content": "Hello"}}\n\n'
            'event: PartDeltaEvent\ndata: {"delta": {"content_delta": " world"}}\n\n'
            'event: PartEndEvent\ndata: {}\n\n'
            'event: FunctionToolCallEvent\ndata: {"part": {"tool_name": "Bash"}}\n\n'
            'event: RunCompletedEvent\ndata: {"status": "success"}\n\n'
        ))
        emitted: list[str] = []

        asyncio.run(_relay(emitted)._consume_once())

        assert emitted == [
            "▶ run started: run the tests",
            "Hello world",
            "→ Bash",
            "■ run finished (success)",
        ]
        assert captured["url"] == f"http://server/agents/{AGENT_ID}/stream"

    def test_thinking_deltas_are_dropped(self, monkeypatch):
        """Reasoning is noise on a phone; only text parts are relayed."""
        _serve(monkeypatch, (
            'event: PartStartEvent\ndata: {"part": {"part_kind": "thinking"}}\n\n'
            'event: PartDeltaEvent\ndata: {"delta": {"content_delta": "pondering"}}\n\n'
            'event: PartEndEvent\ndata: {}\n\n'
            'event: PartStartEvent\ndata: {"part": {"part_kind": "text", "content": "Done."}}\n\n'
            'event: PartEndEvent\ndata: {}\n\n'
        ))
        emitted: list[str] = []

        asyncio.run(_relay(emitted)._consume_once())

        assert emitted == ["Done."]

    def test_error_event_is_surfaced(self, monkeypatch):
        _serve(monkeypatch, 'event: Error\ndata: {"message": "boom"}\n\n')
        emitted: list[str] = []

        asyncio.run(_relay(emitted)._consume_once())

        assert emitted == ["⚠ boom"]

    def test_suppressed_run_is_not_relayed(self, monkeypatch):
        """Runs this bridge started itself are already delivered via the tag path."""
        _serve(monkeypatch, (
            'event: RunStartedEvent\ndata: {"prompt": "hi"}\n\n'
            'event: PartStartEvent\ndata: {"part": {"part_kind": "text", "content": "hello"}}\n\n'
            'event: PartEndEvent\ndata: {}\n\n'
            'event: RunCompletedEvent\ndata: {"status": "success"}\n\n'
        ))
        emitted: list[str] = []

        asyncio.run(_relay(emitted, suppress=lambda: True)._consume_once())

        assert emitted == []

    def test_telegram_tag_markup_is_stripped(self, monkeypatch):
        """TUI output is written for the phone; the relay must not show the XML."""
        _serve(monkeypatch, (
            'event: PartStartEvent\ndata: {"part": {"part_kind": "text", "content": "<telegram>Hello</telegram>"}}\n\n'
            'event: PartEndEvent\ndata: {}\n\n'
        ))
        emitted: list[str] = []

        asyncio.run(_relay(emitted)._consume_once())

        assert emitted == ["Hello"]

    def test_unknown_events_are_ignored(self, monkeypatch):
        _serve(monkeypatch, (
            'event: AgentRunResultEvent\ndata: {}\n\n'
            'event: SomethingNew\ndata: {"x": 1}\n\n'
        ))
        emitted: list[str] = []

        asyncio.run(_relay(emitted)._consume_once())

        assert emitted == []

    def test_non_200_stream_raises_so_the_loop_can_reconnect(self, monkeypatch):
        _serve(monkeypatch, "nope", status=503)

        with pytest.raises(RuntimeError, match="stream HTTP 503"):
            asyncio.run(_relay([])._consume_once())


class TestLifecycle:
    def test_stop_ends_the_task_and_says_so(self):
        emitted: list[str] = []
        relay = _relay(emitted)

        async def never_connects() -> None:
            await asyncio.Event().wait()

        relay._consume_once = never_connects  # type: ignore[method-assign]

        async def scenario() -> None:
            relay.start()
            await asyncio.sleep(0.01)
            assert relay.is_running
            await relay.stop("stream off (test)")

        asyncio.run(scenario())

        assert not relay.is_running
        assert emitted == ["[stream off (test)]"]

    def test_status_reports_endless_versus_timed(self):
        async def scenario() -> tuple[str, str]:
            relay = _relay([])
            relay._deadline = None
            relay._task = asyncio.create_task(asyncio.Event().wait())
            endless = relay.status()
            relay._deadline = asyncio.get_running_loop().time() + 300
            timed = relay.status()
            relay._task.cancel()
            return endless, timed

        endless, timed = asyncio.run(scenario())

        assert endless == "on (endless)"
        assert timed == "on (~5 min left)"
