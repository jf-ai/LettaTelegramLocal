"""Tests for the output policy: what a run looks like on the phone, in each mode.

The interesting cases are mode switches mid-run, since both modes share one cursor.
"""
import pytest

from output_policy import OutputMode, OutputPolicy, strip_tags


class _Mode:
    """Mutable mode holder — the policy reads it at every emission point."""

    def __init__(self, mode: str) -> None:
        self.value = mode

    def __call__(self) -> str:
        return self.value


def _feed(policy: OutputPolicy, *events: tuple[str, dict]) -> list[str]:
    messages: list[str] = []
    for event_type, data in events:
        messages.extend(policy.feed(event_type, data))
    return messages


def _text_delta(text: str) -> tuple[str, dict]:
    return ("PartDeltaEvent", {"delta": {"content_delta": text}})


def _text_part(text: str) -> tuple[str, dict]:
    return ("PartStartEvent", {"part": {"part_kind": "text", "content": text}})


END_PART = ("PartEndEvent", {})


class TestTagsMode:
    def test_only_tagged_content_is_sent(self):
        policy = OutputPolicy(_Mode(OutputMode.TAGS))

        messages = _feed(policy, _text_delta("thinking out loud <telegram>hello</telegram> more out loud"), END_PART)

        assert messages == ["hello"]

    def test_a_tag_is_delivered_as_soon_as_it_closes(self):
        """Early acks depend on this: not waiting for the part to end."""
        policy = OutputPolicy(_Mode(OutputMode.TAGS))

        messages = _feed(policy, _text_delta("<telegram>first</telegram> and then"))

        assert messages == ["first"]

    def test_multiple_tags_in_one_delta_all_arrive(self):
        policy = OutputPolicy(_Mode(OutputMode.TAGS))

        messages = _feed(policy, _text_delta("<telegram>a</telegram> filler <telegram>b</telegram>"))

        assert messages == ["a", "b"]

    def test_an_unclosed_tag_is_not_sent(self):
        policy = OutputPolicy(_Mode(OutputMode.TAGS))

        messages = _feed(policy, _text_delta("<telegram>half a thou"))

        assert messages == []

    def test_a_tag_closed_in_a_later_delta_is_sent_once(self):
        policy = OutputPolicy(_Mode(OutputMode.TAGS))

        messages = _feed(policy, _text_delta("<telegram>sp"), _text_delta("lit"), _text_delta("</telegram>"))

        assert messages == ["split"]

    def test_thinking_deltas_are_ignored(self):
        policy = OutputPolicy(_Mode(OutputMode.TAGS))

        messages = _feed(
            policy,
            ("PartStartEvent", {"part": {"part_kind": "thinking"}}),
            _text_delta("<telegram>not really</telegram>"),
            END_PART,
        )

        assert messages == []

    def test_tool_activity_is_silent(self):
        policy = OutputPolicy(_Mode(OutputMode.TAGS))

        messages = _feed(
            policy,
            ("FunctionToolCallEvent", {"part": {"tool_name": "Bash"}}),
            ("FunctionToolResultEvent", {"part": {"content": "output"}}),
            ("RunCompletedEvent", {"status": "success"}),
        )

        assert messages == []

    def test_a_failed_run_is_reported_even_in_tags_mode(self):
        """Silence about a failure would look identical to a quiet successful run. Used by
        the background relay, which has no other way to report it."""
        policy = OutputPolicy(_Mode(OutputMode.TAGS))

        messages = _feed(policy, ("RunCompletedEvent", {"status": "error"}))

        assert messages == ["■ run finished (error)"]

    def test_the_foreground_turn_reports_its_own_failures(self):
        """handle_message replies "Error: …"; the policy must not say it a second time."""
        policy = OutputPolicy(_Mode(OutputMode.TAGS), report_failures=False)

        messages = _feed(
            policy,
            ("Error", {"message": "boom"}),
            ("RunCompletedEvent", {"status": "error"}),
        )

        assert messages == []


class TestVerboseMode:
    def test_all_untagged_text_is_sent_at_the_part_boundary(self):
        policy = OutputPolicy(_Mode(OutputMode.VERBOSE))

        assert _feed(policy, _text_delta("some "), _text_delta("private prose")) == []
        assert _feed(policy, END_PART) == ["some private prose"]

    def test_tag_markup_is_stripped(self):
        policy = OutputPolicy(_Mode(OutputMode.VERBOSE))

        _feed(policy, _text_delta("<telegram>hi</telegram> and more"))

        assert _feed(policy, END_PART) == ["hi and more"]

    def test_tool_call_and_result_are_reported(self):
        policy = OutputPolicy(_Mode(OutputMode.VERBOSE))

        messages = _feed(
            policy,
            ("FunctionToolCallEvent", {"part": {"tool_name": "Bash"}}),
            ("FunctionToolResultEvent", {"part": {"content": "a\nmulti-line output"}}),
        )

        assert messages == ["→ Bash", "  a multi-line output"]

    def test_a_huge_tool_result_is_truncated(self):
        """Tool returns can be 100 KB; the chat limit is 4096 characters."""
        policy = OutputPolicy(_Mode(OutputMode.VERBOSE))

        messages = _feed(policy, ("FunctionToolResultEvent", {"part": {"content": "x" * 5000}}))

        assert len(messages) == 1
        assert messages[0].endswith("…")
        assert len(messages[0]) == 203  # two-space indent + 200 chars + ellipsis

    def test_bookends_bracket_the_run(self):
        policy = OutputPolicy(_Mode(OutputMode.VERBOSE))

        messages = _feed(
            policy,
            ("RunStartedEvent", {"prompt": "do the thing"}),
            _text_delta("working"),
            END_PART,
            ("RunCompletedEvent", {"status": "success"}),
        )

        assert messages == ["▶ run started: do the thing", "working", "■ run finished (success)"]

    def test_the_end_of_a_foreground_run_is_marked(self):
        """The bridge's own stream has no RunCompletedEvent; AgentRunResultEvent is the
        only signal that the turn is over."""
        policy = OutputPolicy(_Mode(OutputMode.VERBOSE))

        messages = _feed(policy, _text_delta("done"), END_PART, ("AgentRunResultEvent", {}))

        assert messages == ["done", "■ run finished"]

    def test_the_end_of_a_foreground_run_is_silent_in_tags_mode(self):
        policy = OutputPolicy(_Mode(OutputMode.TAGS))

        assert _feed(policy, ("AgentRunResultEvent", {})) == []

    def test_reset_clears_text_between_runs(self):
        policy = OutputPolicy(_Mode(OutputMode.VERBOSE))
        _feed(policy, _text_delta("first run"))
        policy.reset()

        messages = _feed(policy, _text_delta("second run"), END_PART)

        assert messages == ["second run"]


class TestModeSwitchMidRun:
    def test_switching_to_verbose_does_not_replay_what_tags_mode_dropped_before_it(self):
        mode = _Mode(OutputMode.TAGS)
        policy = OutputPolicy(mode)
        _feed(policy, _text_delta("dropped prose "), END_PART)

        mode.value = OutputMode.VERBOSE
        messages = _feed(policy, _text_delta("visible prose"), END_PART)

        assert messages == ["visible prose"]

    def test_a_tag_already_sent_as_verbose_text_is_not_sent_again_when_it_closes(self):
        """The failure this guards: content emitted raw in verbose, then re-emitted when
        the tag closes under tags mode."""
        mode = _Mode(OutputMode.VERBOSE)
        policy = OutputPolicy(mode)
        assert _feed(policy, _text_delta("<telegram>body of the message"), END_PART) == ["body of the message"]

        mode.value = OutputMode.TAGS
        messages = _feed(policy, _text_delta(" continues</telegram>"))

        assert messages == []

    def test_content_never_emitted_before_the_switch_is_sent_once_after_it(self):
        """If verbose never reached a boundary, nothing was delivered, so the closing tag
        under tags mode is the first chance to send it — once, not zero times."""
        mode = _Mode(OutputMode.VERBOSE)
        policy = OutputPolicy(mode)
        _feed(policy, _text_delta("<telegram>half a message"))

        mode.value = OutputMode.TAGS
        messages = _feed(policy, _text_delta(" finished</telegram>"))

        assert messages == ["half a message finished"]

    def test_switching_to_tags_lets_the_next_tag_through(self):
        mode = _Mode(OutputMode.VERBOSE)
        policy = OutputPolicy(mode)
        _feed(policy, _text_delta("noise"), END_PART)

        mode.value = OutputMode.TAGS
        messages = _feed(policy, _text_delta("<telegram>now only tags</telegram>"))

        assert messages == ["now only tags"]


def test_strip_tags_keeps_content():
    assert strip_tags("<telegram>keep me</telegram>") == "keep me"
