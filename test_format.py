"""Tests for what the agent is told about delivery.

The header and guide are the agent's only evidence about whether its prose is private, so
they must say exactly what OutputPolicy does in the current mode — no more, no less. An
agent that believes untagged text is private in verbose mode will write things it should
not, and that is the failure these tests exist to prevent.
"""
import asyncio
import os
from types import SimpleNamespace

# Config is read at import time; use a name with no token so no real credential loads.
os.environ.setdefault("AGENT_NAME", "probe_agent")
os.environ.setdefault("TELEGRAM_BOT_TOKEN_PROBE_AGENT", "123:test")
os.environ.setdefault("AUTHORIZED_USER", "4242")
os.environ.setdefault("SKIP_FIRST_PING", "True")
os.environ.setdefault("TELEGRAM_PASSWORD", "correct-horse")

import pytest  # noqa: E402

import LettaTelegramLocal as bridge  # noqa: E402  (must follow the env setup)

MESSAGE = "hello there"


def _guide(verbose: bool) -> str:
    return bridge.format_incoming_message(MESSAGE, include_guide=True, verbose=verbose)


def _header(verbose: bool) -> str:
    return bridge.format_incoming_message(MESSAGE, include_guide=False, verbose=verbose)


class TestGuide:
    def test_first_message_carries_the_guide_and_the_message(self):
        guide = _guide(verbose=False)

        assert guide.startswith(bridge.TELEGRAM_GUIDE_PREFIX)
        assert guide.endswith(f"[via Telegram]\n{MESSAGE}")

    @pytest.mark.parametrize("verbose", [False, True])
    def test_guide_lists_the_agent_commands(self, verbose):
        assert bridge.TELEGRAM_COMMANDS_GUIDE in _guide(verbose)

    def test_tags_mode_promises_privacy_for_untagged_text(self):
        guide = _guide(verbose=False)

        assert "Untagged text stays PRIVATE" in guide
        assert "VERBOSE" not in guide

    def test_verbose_mode_says_everything_is_delivered_and_nothing_is_private(self):
        guide = _guide(verbose=True)

        assert "VERBOSE MODE IS ON" in guide
        assert "Untagged text is NOT private" in guide
        assert "delivered to James's phone" in guide

    def test_verbose_mode_says_tags_are_unnecessary(self):
        """Otherwise the agent keeps wrapping output in tags it believes are load-bearing."""
        guide = _guide(verbose=True)

        assert "<telegram> tags are unnecessary now" in guide


class TestHeader:
    def test_tags_mode_header_states_the_tag_rule(self):
        assert _header(verbose=False) == (
            "[via Telegram: wrap sends in <telegram>tags</telegram> — untagged = private, "
            f"no tags = nothing sent]\n{MESSAGE}"
        )

    def test_verbose_mode_header_says_nothing_is_private(self):
        header = _header(verbose=True)

        assert "nothing here is private" in header
        assert "tags optional and stripped" in header
        assert "untagged = private" not in header

    def test_the_header_never_claims_privacy_in_verbose_mode(self):
        """The specific lie this change removes: the tags-mode reminder appearing while
        verbose delivery is active."""
        assert "private" in _header(verbose=True)  # as part of "nothing here is private"
        assert "untagged = private" not in _header(verbose=True)

    def test_identical_wording_comes_from_the_policy_modes(self):
        """The map is keyed by OutputMode, so a new mode cannot silently reuse a wording."""
        assert set(bridge.TELEGRAM_HEADERS) == {bridge.OutputMode.TAGS, bridge.OutputMode.VERBOSE}
        assert set(bridge.DELIVERY_RULES) == {bridge.OutputMode.TAGS, bridge.OutputMode.VERBOSE}


class TestWiring:
    """The formatting function is only half the contract: handle_message must pass the
    live mode in. A unit-tested formatter wired to a constant would still lie."""

    @pytest.mark.parametrize("verbose", [False, True])
    def test_the_prompt_carries_the_mode_in_force(self, monkeypatch, verbose):
        captured = {}

        async def fake_send(prompt, state, bot=None, chat_id=None):
            captured["prompt"] = prompt
            return {"success": True, "result": "", "updates_sent": 0}

        monkeypatch.setattr(bridge, "send_message", fake_send)

        state = bridge.BridgeState(agent_id="agent-1")
        state.verbose_mode = bridge.OutputMode.VERBOSE if verbose else bridge.OutputMode.TAGS
        state.authenticated_users.add(4242)
        replies = []

        async def reply_text(text):
            replies.append(text)

        message = SimpleNamespace(text="hi", chat_id=4242, reply_text=reply_text)
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=4242),
            message=message,
            effective_chat=SimpleNamespace(type="private", id=4242),
        )
        context = SimpleNamespace(
            bot=None, job_queue=None, job=None, application=SimpleNamespace(job_queue=None)
        )

        asyncio.run(bridge.handle_message(update, context, state))

        # First message of the session, so the guide travels rather than the header; the
        # guide's delivery section is the part that must match the mode.
        mode = bridge.OutputMode.VERBOSE if verbose else bridge.OutputMode.TAGS
        assert bridge.DELIVERY_RULES[mode] in captured["prompt"]

    @pytest.mark.parametrize("verbose", [False, True])
    def test_a_later_prompt_carries_the_mode_header(self, monkeypatch, verbose):
        captured = {}

        async def fake_send(prompt, state, bot=None, chat_id=None):
            captured["prompt"] = prompt
            return {"success": True, "result": "", "updates_sent": 0}

        monkeypatch.setattr(bridge, "send_message", fake_send)

        state = bridge.BridgeState(agent_id="agent-1")
        state.verbose_mode = bridge.OutputMode.VERBOSE if verbose else bridge.OutputMode.TAGS
        state.authenticated_users.add(4242)
        state.briefed_users.add(4242)   # already seen the guide this session

        async def reply_text(text):
            pass

        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=4242),
            message=SimpleNamespace(text="hi", chat_id=4242, reply_text=reply_text),
            effective_chat=SimpleNamespace(type="private", id=4242),
        )
        context = SimpleNamespace(
            bot=None, job_queue=None, job=None, application=SimpleNamespace(job_queue=None)
        )

        asyncio.run(bridge.handle_message(update, context, state))

        mode = bridge.OutputMode.VERBOSE if verbose else bridge.OutputMode.TAGS
        assert captured["prompt"].startswith(bridge.TELEGRAM_HEADERS[mode])
