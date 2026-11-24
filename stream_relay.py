"""Relay an agent's background event stream into a Telegram chat.

The bridge normally sees only the runs it starts itself: it POSTs a message and consumes
that run's SSE response. Work started elsewhere is invisible unless the bridge subscribes
to ``GET /agents/{id}/stream``, where Agent Home publishes every run for an agent
regardless of who started it (subject to the other client not suppressing its broadcast).

What the relay does with those events is not its own decision: it feeds them to the same
OutputPolicy the foreground turn uses, so verbosity means the same thing on both paths.
The relay owns only the connection — subscription, reconnection, and saying so when the
stream drops.

Kept independent of python-telegram-bot: the caller passes a ``send`` coroutine, which is
what makes the relay testable without a bot.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

import httpx

from agent_home_client import SSEEventParser
from output_policy import OutputPolicy

logger = logging.getLogger(__name__)

RECONNECT_DELAY_SECONDS = 2.0


class StreamRelay:
    """Forwards one agent's broadcast stream to a chat until stopped.

    send:     async callable(text) delivering one logical message (the caller handles
              platform limits).
    mode:     callable returning the current OutputMode, consulted per event, so a
              mid-run /verbose toggle takes effect from the next event.
    suppress: callable returning True while the bridge is itself running a turn; the
              relay then stays quiet, since that run is already delivered by the tag path.
    """

    # TODO: suppression is client-side because the server's Suppress-Broadcast header
    # hides a run from *every* subscriber. The wanted shape is per-subscriber: a client
    # declares an identity when subscribing and the server does not echo a run back to its
    # originator. Needs server support; parked until then.

    def __init__(
        self,
        server_url: str,
        agent_id: str,
        send: Callable[[str], Awaitable[None]],
        mode: Callable[[], str],
        suppress: Callable[[], bool] | None = None,
    ) -> None:
        self._server_url = server_url
        self._agent_id = agent_id
        self._send = send
        self._suppress = suppress or (lambda: False)
        self._policy = OutputPolicy(mode)
        self._task: asyncio.Task | None = None
        self._deadline: float | None = None

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
        await self._emit(f"[{reason}]")

    # --- connection loop ---------------------------------------------------

    async def _run(self) -> None:
        """One connection at a time, reconnecting while the relay is wanted.

        A dropped stream is normal (server restart, network blip) and must not end an
        endless relay, so the loop says so in chat and reconnects.
        """
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
        parser = SSEEventParser()
        try:
            while True:
                item = await queue.get()
                if item is None:
                    await reader  # surface HTTP/connection errors raised in the reader
                    return
                parsed = parser.feed(item)
                if parsed is not None:
                    for text in self._policy.feed(*parsed):
                        await self._emit(text)
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader

    async def _emit(self, text: str) -> None:
        # Operational markers (disconnects, stop confirmations) go through here too: they
        # are the relay's own voice, not agent output, so the policy must not filter them.
        if not text.strip() or self._suppress():
            return
        try:
            await self._send(text)
        except Exception as e:  # a failed send must not kill the relay loop
            logger.warning("stream relay send failed: %s", e)
