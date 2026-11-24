"""Every handler except /start must be auth-gated.

The Telegram bridge is the most exposed surface in Agent Home: anyone who finds the bot
can send updates to it. Per-handler guards made the gate a thing to remember in each
handler — and it was forgotten in none only because it was copy-pasted in each. This file
asserts the structural property instead, so a new handler registered without the gate
fails the suite rather than quietly relying on a reviewer.
"""
import asyncio
import os
from types import SimpleNamespace

# Config is read at import time; point at a name with no token in .env so no real
# credential is ever loaded, even though nothing here talks to Telegram.
os.environ.setdefault("AGENT_NAME", "probe_agent")
os.environ.setdefault("TELEGRAM_BOT_TOKEN_PROBE_AGENT", "123:test")
os.environ.setdefault("AUTHORIZED_USER", "4242")
os.environ.setdefault("SKIP_FIRST_PING", "True")
os.environ.setdefault("TELEGRAM_PASSWORD", "correct-horse")

import pytest  # noqa: E402
from telegram.ext import CommandHandler  # noqa: E402

import LettaTelegramLocal as bridge  # noqa: E402  (must follow the env setup)


class _FakeMessage:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


def _update(user_id: int) -> SimpleNamespace:
    return SimpleNamespace(effective_user=SimpleNamespace(id=user_id), message=_FakeMessage())


def test_only_start_is_registered_ungated():
    """One ungated handler, and it is /start itself — by identity, not by name.

    Stated as a property rather than an expected table of handlers: a new handler joins
    the app without touching this test (it must be gated, or this fails), and there is no
    slot in which to record "this one is allowed to be ungated". The exemption is the
    start *function*, so a differently-named callback called "start" does not qualify.
    """
    app = bridge.build_application()
    handlers = [handler for group in app.handlers.values() for handler in group]

    ungated = [handler.callback for handler in handlers if not getattr(handler.callback, "__auth_gated__", False)]

    assert ungated == [bridge.start], f"ungated handlers: {[cb.__qualname__ for cb in ungated]}"


class TestGateBehaviour:
    def test_known_but_unauthenticated_user_is_told_how_and_handler_does_not_run(self):
        calls: list[str] = []

        async def handler(update, context):
            calls.append("ran")

        update = _update(bridge.ALLOWED_USER_IDS[0])

        asyncio.run(bridge._auth_required(handler)(update, None))

        assert calls == []
        assert update.message.replies == ["Please connect first with /start"]

    def test_unknown_user_is_dropped_silently(self):
        """No reply, no information: an unknown sender learns nothing about the bot."""
        calls: list[str] = []

        async def handler(update, context):
            calls.append("ran")

        update = _update(user_id=999)

        asyncio.run(bridge._auth_required(handler)(update, None))

        assert calls == []
        assert update.message.replies == []

    def test_authenticated_user_reaches_the_handler(self):
        calls: list[str] = []

        async def handler(update, context):
            calls.append("ran")

        user_id = bridge.ALLOWED_USER_IDS[0]
        bridge.authenticated_users.add(user_id)
        try:
            update = _update(user_id)
            asyncio.run(bridge._auth_required(handler)(update, None))
        finally:
            bridge.authenticated_users.discard(user_id)

        assert calls == ["ran"]
        assert update.message.replies == []


class _FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


class _FakeApplication:
    def __init__(self) -> None:
        self.stopped = 0

    def stop_running(self) -> None:
        self.stopped += 1


class _DeleteableMessage(_FakeMessage):
    chat_id = None

    async def delete(self) -> None:
        pass


def _auth_update(user_id: int, chat_id: int, text_args: list[str] | None) -> SimpleNamespace:
    message = _DeleteableMessage()
    message.chat_id = chat_id
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        message=message,
        args=text_args,
        bot=_FakeBot(),
        application=_FakeApplication(),
    )


class TestFailedAttemptTripwire:
    """Repeated wrong passwords take the bridge down. Only the authorized account can
    reach the failure branch at all, so this is a tripwire for a compromised session
    rather than a DoS lever a stranger could pull."""

    @pytest.fixture(autouse=True)
    def _reset_counter(self):
        bridge.failed_auth_attempts = 0
        yield
        bridge.failed_auth_attempts = 0

    def _fail(self, attempts: int, chat_id: int = None) -> SimpleNamespace:
        update = None
        for _ in range(attempts):
            update = self._send_wrong_password(chat_id)
        return update

    def _send_wrong_password(self, chat_id: int = None) -> SimpleNamespace:
        update = _auth_update(bridge.ALLOWED_USER_IDS[0], chat_id or bridge.ALLOWED_USER_IDS[0], ["wrong"])
        asyncio.run(bridge.start(update, update))
        return update

    def test_four_failures_leave_the_bridge_running(self):
        update = self._fail(4)

        assert update.application.stopped == 0

    def test_fifth_failure_stops_the_bridge(self):
        update = self._fail(5)

        assert update.application.stopped == 1

    def test_shutdown_sends_nothing_to_telegram(self):
        """Every address available belongs to the account that just failed, so a notice
        would be a response to whoever holds it. Silence; the host log carries the reason."""
        update = self._fail(5)

        assert update.bot.sent == []

    def test_fifth_attempt_is_indistinguishable_from_an_ordinary_failure(self):
        """The vague reply must not change on the trip, or the attacker learns the
        threshold was reached."""
        self._fail(4)
        update = self._send_wrong_password()

        assert update.message.replies == ["Hmm, I don't understand. Please try again?"]

    def test_successful_auth_resets_the_count(self):
        self._fail(4)

        ok = _auth_update(bridge.ALLOWED_USER_IDS[0], bridge.ALLOWED_USER_IDS[0], ["correct-horse"])
        asyncio.run(bridge.start(ok, ok))

        assert bridge.failed_auth_attempts == 0
        self._send_wrong_password()

        assert bridge.failed_auth_attempts == 1


def test_main_starts_polling_with_a_clean_backlog(monkeypatch):
    """A restarted bridge must not replay updates sent while it was down: stale messages
    are answered too late, and a backlog of failed /start attempts would trip the kill
    switch before the operator can type."""
    captured: dict = {}
    monkeypatch.setattr(bridge.asyncio, "run", lambda coro: (coro.close(), "agent-1")[1])
    monkeypatch.setattr(bridge.Application, "run_polling", lambda self, **kwargs: captured.update(kwargs))

    bridge.main()

    assert captured == {"drop_pending_updates": True}


class TestCommandRegistry:
    """COMMANDS is the single declaration; these tests stop it drifting from reality."""

    def test_every_registry_entry_is_registered_as_a_command_handler(self):
        app = bridge.build_application()
        registered = {
            command
            for group in app.handlers.values()
            for handler in group
            if isinstance(handler, CommandHandler)
            for command in handler.commands
        }

        assert registered == {entry.name for entry in bridge.COMMANDS}

    def test_every_command_is_described_for_help(self):
        assert [entry.name for entry in bridge.COMMANDS if not entry.description.strip()] == []

    def test_the_start_command_is_declared_apart_from_the_gated_ones(self):
        """Structure over flags: the ungated command cannot be inside the gated list."""
        assert bridge.START_CMD.name == "start"
        assert bridge.START_CMD not in bridge.AUTH_GATED_CMDS
        assert bridge.COMMANDS == (bridge.START_CMD,) + bridge.AUTH_GATED_CMDS

    def test_help_lists_every_command_with_its_description(self):
        update = _update(bridge.ALLOWED_USER_IDS[0])

        asyncio.run(bridge.help_cmd(update, None))

        text = update.message.replies[0]
        assert ["/" + entry.name for entry in bridge.COMMANDS] == [
            line.split(" — ")[0] for line in text.splitlines()[1:]
        ]
