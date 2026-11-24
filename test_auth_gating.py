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
