"""Relay an agent's background event stream into a Telegram chat.

The bridge normally sees only the runs it starts itself: it POSTs a message and consumes
that run's SSE response. Work started elsewhere — a TUI session, another client, a
scheduled ping — is invisible to it. Agent Home broadcasts every run for an agent through
``GET /agents/{id}/stream`` regardless of who started it, so subscribing to that stream is
what lets a phone watch over a long tool-using run happening in a terminal.

Relay shape: run bookends, assistant text, and tool calls. Raw token deltas are coalesced
and flushed on a timer — Telegram rate-limits to roughly one message per second per chat,
so forwarding deltas would be both unusable and self-defeating.

Kept independent of python-telegram-bot: the caller passes a ``send`` coroutine, which is
what makes the relay testable without a bot.
"""

import asyncio
import contextlib
import json
import logging
import re
from collections.abc import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_SECONDS = 2.0
RECONNECT_DELAY_SECONDS = 2.0
PROMPT_PREVIEW_CHARS = 80
TELEGRAM_TAG_RE = re.compile(r"</?telegram>", re.IGNORECASE)


class StreamRelay:
    """Forwards one agent's broadcast stream to a chat until stopped.

    send:      async callable(text) delivering one logical message (the caller handles
               platform limits and its own formatting).
    suppress:  callable returning True while the bridge is itself running a turn; the
               relay then drops output, because that run is already being delivered
               through the normal reply path and would otherwise arrive twice.
    """

    def __init__(
        self,
        server_url: str,
        agent_id: str,
        send: Callable[[str], Awaitable[None]],
        suppress: Callable[[], bool] | None = None,
    ) -> None:
        self._server_url = server_url
        self._agent_id = agent_id
        self._send = send
        self._suppress = suppress or (lambda: False)
        self._task: asyncio.Task | None = None
        self._deadline: float | None = None
        self._buffer: list[str] = []
        self._in_thinking = False

    # --- lifecycle ---------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> str:
        if not self.is_running:
            return "off"
        if self._deadline is None:
            return "on (endless)"
        remaining = max(0.0, self._deadline - asyncio.get_running_loop().time())
        return f"on (~{remaining / 60:.0f} min left)"

    def start(self, max_minutes: float | None = None) -> None:
        """Begin relaying. max_minutes=None streams until stop()."""
        loop = asyncio.get_running_loop()
        self._deadline = None if max_minutes is None else loop.time() + max_minutes * 60
        self._task = asyncio.create_task(self._run(), name="stream-relay")

    async def stop(self, reason: str = "stream stopped") -> None:
        if self._task is None:
            return
        task, self._task = self._task, None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._buffer.clear()
        await self._emit(f"[{reason}]")

    # --- connection loop ---------------------------------------------------

    async def _run(self) -> None:
        """One connection at a time, reconnecting for as long as the relay is wanted.

        A dropped stream is normal (server restart, network blip) and must not end an
        endless relay, so the loop says so in chat and reconnects.
        """
        try:
            while True:
                if self._deadline is not None and asyncio.get_running_loop().time() >= self._deadline:
                    await self._emit("[stream ended: time limit reached]")
                    return
                try:
                    await self._consume_once()
                    await self._emit("[stream closed by server; reconnecting]")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("stream relay disconnected: %s", e)
                    await self._emit(f"[stream disconnected: {type(e).__name__}; reconnecting]")
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)
        finally:
            self._buffer.clear()

    async def _consume_once(self) -> None:
        """Read one SSE connection to completion (or until the stream ends)."""
        url = f"{self._server_url}/agents/{self._agent_id}/stream"
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def read_lines() -> None:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("GET", url, headers={"Accept": "text/event-stream"}) as response:
                        if response.status_code != 200:
                            raise RuntimeError(f"stream HTTP {response.status_code}")
                        async for line in response.aiter_lines():
                            await queue.put(line)
            finally:
                await queue.put(None)

        reader = asyncio.create_task(read_lines())
        event_type: str | None = None
        data_lines: list[str] = []
        try:
            while True:
                # The timeout is what drives flushing: a quiet stream still needs the
                # pending text pushed out, and cancelling queue.get() is safe (unlike
                # cancelling an async generator's __anext__).
                try:
                    item = await asyncio.wait_for(queue.get(), FLUSH_INTERVAL_SECONDS)
                except TimeoutError:
                    await self._flush()
                    continue
                if item is None:
                    await reader  # surface HTTP/connection errors raised in the reader
                    return
                if item.startswith("event:"):
                    event_type, data_lines = item[6:].strip(), []
                elif item.startswith("data:"):
                    data_lines.append(item[5:].strip())
                elif item == "" and event_type:
                    await self._handle_event(event_type, self._parse(data_lines))
                    event_type, data_lines = None, []
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader

    @staticmethod
    def _parse(data_lines: list[str]) -> dict:
        if not data_lines:
            return {}
        try:
            parsed = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # --- event handling ----------------------------------------------------

    async def _handle_event(self, event_type: str, data: dict) -> None:
        if event_type == "RunStartedEvent":
            await self._flush()
            prompt = str(data.get("prompt", "")).strip()
            if prompt:
                preview = prompt.replace("\n", " ")[:PROMPT_PREVIEW_CHARS]
                await self._emit(f"▶ run started: {preview}")
            else:
                await self._emit("▶ run started")

        elif event_type == "PartStartEvent":
            part = data.get("part", {})
            kind = part.get("part_kind")
            self._in_thinking = kind == "thinking"
            if kind == "text" and part.get("content"):
                self._buffer.append(part["content"])

        elif event_type == "PartDeltaEvent":
            if not self._in_thinking:
                delta = data.get("delta", {}).get("content_delta")
                if delta:
                    self._buffer.append(delta)

        elif event_type == "PartEndEvent":
            await self._flush()

        elif "ToolCall" in event_type:
            await self._flush()
            part = data.get("part", {})
            tool_name = part.get("tool_name") or "tool"
            await self._emit(f"→ {tool_name}")

        elif event_type == "RunCompletedEvent":
            await self._flush()
            await self._emit(f"■ run finished ({data.get('status', 'unknown')})")

        elif event_type in ("Error", "RunErrorEvent"):
            await self._flush()
            await self._emit(f"⚠ {data.get('message', 'run error')}")

        # Everything else (AgentRunResultEvent, tool results, thinking deltas) is noise
        # on a phone and is deliberately dropped.

    async def _flush(self) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        if text:
            await self._emit(text)

    async def _emit(self, text: str) -> None:
        # Agent output written for the phone carries <telegram> tags; here the relay *is*
        # the transport, so the markup would be shown literally. Keep the words, drop the tags.
        text = TELEGRAM_TAG_RE.sub("", text).strip()
        if not text or self._suppress():
            return
        try:
            await self._send(text)
        except Exception as e:  # a failed send must not kill the relay loop
            logger.warning("stream relay send failed: %s", e)
