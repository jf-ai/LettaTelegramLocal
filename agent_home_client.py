"""Shared Agent Home client logic — single source of truth for the SSE protocol.

Used by both the Telegram bridge (async) and group chat (sync). Provides:
- send_message_async / send_message_sync: POST /agents/{id}/messages and
  consume the SSE stream, accumulating the text response (thinking excluded)
- resolve_agent_id_by_name: look up an agent ID from the registry by name

Both drivers share _ResponseAccumulator, which owns the event-format logic
(PartStartEvent/PartDeltaEvent parsing, thinking-part detection). Drivers
only differ in transport (sync vs async httpx).

Error convention: hard failures raise AgentStreamError; timeouts return
whatever text accumulated plus a notice (partial response beats none).
"""

import json
from collections.abc import Awaitable, Callable

import httpx

DEFAULT_TIMEOUT_SECONDS = 480  # agent runs can be long (agentic tool use)


class AgentStreamError(RuntimeError):
    """Hard failure during an agent run (HTTP error, stream Error event, connection error)."""


class SSEEventParser:
    """Incremental SSE framing: feed lines, get (event_type, data) as each block completes.

    Shared by every consumer of an agent event stream (foreground turn, background relay)
    so framing is defined once. Data is decoded as JSON when possible and passed through
    as a dict; a malformed block yields {} rather than raising, matching the server's
    own tolerance for partial data.
    """

    def __init__(self) -> None:
        self._event_type: str | None = None
        self._data_lines: list[str] = []

    def feed(self, line: str) -> tuple[str, dict] | None:
        if line.startswith("event:"):
            self._event_type = line[6:].strip()
            self._data_lines = []
        elif line.startswith("data:"):
            self._data_lines.append(line[5:].strip())
        elif line == "" and self._event_type:
            event_type, data = self._event_type, self._parse(self._data_lines)
            self._event_type, self._data_lines = None, []
            return event_type, data
        return None

    @staticmethod
    def _parse(data_lines: list[str]) -> dict:
        if not data_lines:
            return {}
        try:
            parsed = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


class _ResponseAccumulator:
    """Feed SSE lines from an agent run stream; accumulate the text response.

    Thinking parts are excluded — only text parts contribute. Sets .error
    when an Error event arrives so drivers can surface the failure.
    """

    def __init__(self):
        self.text_parts: list[str] = []
        self._in_thinking = False
        self._parser = SSEEventParser()
        self.error: str | None = None

    def feed(self, line: str) -> tuple[str, dict] | None:
        """Process one line; return the event once its block completes."""
        parsed = self._parser.feed(line)
        if parsed is not None:
            self._process_event(*parsed)
        return parsed

    def _process_event(self, event_type: str, data: dict) -> None:

        if event_type == "PartStartEvent":
            part = data.get("part", {})
            part_kind = part.get("part_kind")
            if part_kind == "thinking":
                self._in_thinking = True
            elif part_kind == "text":
                self._in_thinking = False
                if content := part.get("content", ""):
                    self.text_parts.append(content)
        elif event_type == "PartDeltaEvent":
            if not self._in_thinking:
                if content := data.get("delta", {}).get("content_delta", ""):
                    self.text_parts.append(content)
        elif event_type == "Error":
            self.error = data.get("message", "Unknown error")

    def text(self) -> str:
        return "".join(self.text_parts).strip()


def _timeout_text(acc: _ResponseAccumulator, timeout_seconds: float) -> str:
    """Format partial response on timeout — accumulated text plus a notice."""
    suffix = f"\n[timed out after {timeout_seconds}s]"
    text = acc.text()
    return (text + suffix) if text else suffix.strip()


async def send_message_async(server_url: str, agent_id: str, message: str,
                             timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                             on_event: Callable[[str, dict], Awaitable[None]] | None = None) -> str:
    """Send a message and return the agent's text response (thinking excluded).

    Raises AgentStreamError on hard failures. On timeout, returns partial
    text plus a notice.

    on_event: optional async callback invoked with (event_type, data) as each SSE block
    completes — the raw pydantic-ai events, before any accumulation. Callers that render
    (the telegram bridge, the background relay) consume these through an OutputPolicy;
    callers that only want the text ignore it.
    """
    url = f"{server_url}/agents/{agent_id}/messages"
    acc = _ResponseAccumulator()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            async with client.stream("POST", url, json={"message": message}) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode(errors="replace")[:200]
                    raise AgentStreamError(f"HTTP {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    parsed = acc.feed(line)
                    if acc.error is not None:
                        raise AgentStreamError(acc.error)
                    if parsed is not None and on_event is not None:
                        await on_event(*parsed)
    except httpx.TimeoutException:
        return _timeout_text(acc, timeout_seconds)
    except httpx.RequestError as e:
        raise AgentStreamError(f"Connection error: {e}") from e
    return acc.text()


def send_message_sync(server_url: str, agent_id: str, message: str,
                      timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Sync twin of send_message_async — same semantics, blocking transport."""
    url = f"{server_url}/agents/{agent_id}/messages"
    acc = _ResponseAccumulator()
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            with client.stream("POST", url, json={"message": message}) as response:
                if response.status_code != 200:
                    detail = response.read().decode(errors="replace")[:200]
                    raise AgentStreamError(f"HTTP {response.status_code}: {detail}")
                for line in response.iter_lines():
                    acc.feed(line)
                    if acc.error is not None:
                        raise AgentStreamError(acc.error)
    except httpx.TimeoutException:
        return _timeout_text(acc, timeout_seconds)
    except httpx.RequestError as e:
        raise AgentStreamError(f"Connection error: {e}") from e
    return acc.text()


async def cancel_agent_run_async(server_url: str, agent_id: str) -> bool:
    """Ask the server to cancel the agent's in-flight run. True if a cancel was accepted.

    Cancels whichever run is active for that agent — including one started by another
    client (the TUI), not just this bridge's. 409 means nothing was running. Note the
    server's cancellation is cooperative: it is serviced between streamed events, so a
    request that is hung before producing output cannot be interrupted promptly.
    """
    url = f"{server_url}/agents/{agent_id}/cancel"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url)
    except httpx.HTTPError as e:
        raise AgentStreamError(f"Cancel request failed: {e}") from e
    if resp.status_code == 202:
        return True
    if resp.status_code == 409:
        return False
    raise AgentStreamError(f"Cancel rejected: HTTP {resp.status_code}: {resp.text[:200]}")


def resolve_agent_id_by_name_sync(server_url: str, agent_name: str) -> str:
    """Look up an agent ID by name from the Agent Home registry."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{server_url}/agents/by-name/{agent_name}")
            if resp.status_code == 404:
                raise AgentStreamError(f"Agent {agent_name!r} not found on {server_url}")
            resp.raise_for_status()
            return resp.json()["id"]
    except httpx.HTTPError as e:
        raise AgentStreamError(f"Registry lookup failed: {e}") from e


async def resolve_agent_id_by_name_async(server_url: str, agent_name: str) -> str:
    """Async twin of resolve_agent_id_by_name_sync."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{server_url}/agents/by-name/{agent_name}")
            if resp.status_code == 404:
                raise AgentStreamError(f"Agent {agent_name!r} not found on {server_url}")
            resp.raise_for_status()
            return resp.json()["id"]
    except httpx.HTTPError as e:
        raise AgentStreamError(f"Registry lookup failed: {e}") from e