"""Deciding what a run's event stream should look like on the phone.

Two modes, orthogonal to which stream is being watched:

  TAGS     only content the agent wrapped in <telegram>...</telegram>, delivered as each
           tag closes (early acks are the point, so this stays progressive)
  VERBOSE  all assistant text, thinking excluded and tag markup stripped, emitted one
           message per event boundary, plus tool activity

Both modes share one cursor over the run's accumulated text: each mode emits from the
cursor forward and advances it past what it emitted. That is what makes a mid-run mode
change safe — nothing already sent is replayed, and a tag that closes after a switch to
verbose is sent once as plain text rather than twice.

Telegram is a message interface, not a terminal: nothing here streams deltas to the chat.
"""

import re
from collections.abc import Callable

TELEGRAM_TAG_PATTERN = re.compile(r"<telegram>(.*?)</telegram>", re.DOTALL | re.IGNORECASE)
TELEGRAM_TAG_RE = re.compile(r"</?telegram>", re.IGNORECASE)
PROMPT_PREVIEW_CHARS = 80
TOOL_RESULT_PREVIEW_CHARS = 200


class OutputMode:
    TAGS = "tags"
    VERBOSE = "verbose"


def strip_tags(text: str) -> str:
    """Remove telegram tag markup, keeping the content."""
    return TELEGRAM_TAG_RE.sub("", text).strip()


class OutputPolicy:
    """Turns run events into chat messages according to the current mode.

    mode is a callable so the answer can change mid-run (the operator toggles it from
    chat); it is consulted at each emission point, never captured.
    """

    def __init__(self, mode: Callable[[], str], report_failures: bool = True) -> None:
        self._mode = mode
        # False for the foreground turn, which replies "Error: …" itself when the run
        # raises — otherwise one failure arrives three times (error, bookend, reply).
        self._report_failures = report_failures
        self._buffer = ""
        self._cursor = 0
        self._in_thinking = False
        self.sent_count = 0   # messages emitted for the current run

    def reset(self) -> None:
        """Start of a run. The bridge's own stream has no RunStarted bookend, so the
        caller resets explicitly."""
        self._buffer = ""
        self._cursor = 0
        self._in_thinking = False

    def feed(self, event_type: str, data: dict) -> list[str]:
        """Consume one event; return the messages to send right now (possibly none)."""
        messages = self._feed(event_type, data)
        self.sent_count += len(messages)
        return messages

    def _feed(self, event_type: str, data: dict) -> list[str]:
        if event_type == "RunStartedEvent":
            self.reset()
            if self._mode() == OutputMode.VERBOSE:
                prompt = str(data.get("prompt", "")).strip().replace("\n", " ")
                preview = prompt[:PROMPT_PREVIEW_CHARS]
                return [f"▶ run started: {preview}" if prompt else "▶ run started"]
            return []

        if event_type == "PartStartEvent":
            part = data.get("part", {})
            kind = part.get("part_kind")
            self._in_thinking = kind == "thinking"
            if kind == "text" and part.get("content"):
                self._buffer += part["content"]
                return self._new_tag_messages()
            return []

        if event_type == "PartDeltaEvent":
            if self._in_thinking:
                return []
            delta = data.get("delta", {}).get("content_delta")
            if not delta:
                return []
            self._buffer += delta
            return self._new_tag_messages()

        if event_type == "PartEndEvent":
            return self._end_of_part()

        if "ToolCall" in event_type:
            if self._mode() != OutputMode.VERBOSE:
                return []
            tool_name = data.get("part", {}).get("tool_name") or "tool"
            return [f"→ {tool_name}"]

        if "ToolResult" in event_type:
            if self._mode() != OutputMode.VERBOSE:
                return []
            return self._tool_result_message(data)

        if event_type == "AgentRunResultEvent":
            # End of a foreground run. The hub route yields RunCompletedEvent instead, so
            # this is what "the turn is over" looks like on the bridge's own stream.
            return ["■ run finished"] if self._mode() == OutputMode.VERBOSE else []

        if event_type == "RunCompletedEvent":
            status = data.get("status", "unknown")
            if status == "success":
                return [f"■ run finished ({status})"] if self._mode() == OutputMode.VERBOSE else []
            # Failures: narrated here only when nobody else will. The foreground turn
            # raises and replies "Error: …", so it opts out in both modes.
            return [f"■ run finished ({status})"] if self._report_failures else []

        if event_type in ("Error", "RunErrorEvent"):
            if not self._report_failures:
                return []
            message = str(data.get("message", "run error")).strip()
            return [f"⚠ {message}"]

        return []

    # --- internals ---------------------------------------------------------

    def _new_tag_messages(self) -> list[str]:
        """Tags only: deliver each newly closed tag, keeping acks early."""
        if self._mode() != OutputMode.TAGS:
            return []
        pending = self._buffer[self._cursor:]
        messages = []
        for match in TELEGRAM_TAG_PATTERN.finditer(pending):
            content = match.group(1).strip()
            if content:
                messages.append(content)
            self._cursor += match.end()
        return messages

    def _end_of_part(self) -> list[str]:
        if self._mode() == OutputMode.VERBOSE:
            text = strip_tags(self._buffer[self._cursor:])
            self._cursor = len(self._buffer)
            return [text] if text else []
        # Tags mode: untagged text is deliberately dropped, but the cursor must move past
        # it or a later switch to verbose would replay history the operator never wanted.
        self._cursor = len(self._buffer)
        return []

    def _tool_result_message(self, data: dict) -> list[str]:
        part = data.get("part", {})
        content = part.get("content")
        if content is None:
            return []
        text = content if isinstance(content, str) else repr(content)
        text = " ".join(text.split())
        if not text:
            return []
        if len(text) > TOOL_RESULT_PREVIEW_CHARS:
            text = text[:TOOL_RESULT_PREVIEW_CHARS].rstrip() + "…"
        return [f"  {text}"]
