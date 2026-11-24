"""A fake Telegram Bot API — for exercising the bridge without Telegram or real tokens.

Point the bridge at it with TELEGRAM_API_BASE_URL=http://127.0.0.1:8900/bot. It implements
just enough of the Bot API for python-telegram-bot's long-polling loop, records everything
the bot sends, and lets a test inject updates as if the authorized user had typed them.

    uv run uvicorn fake_bot_api:app --port 8900
    curl -X POST 'localhost:8900/inject?text=/stream_on'
    curl -s localhost:8900/sent | python -m json.tool

Not part of the bridge: no auth, no persistence, in-memory only.
"""

import asyncio
import json
import time
from urllib.parse import parse_qs
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

_pending: asyncio.Queue[str] = asyncio.Queue()
_sent: list[dict] = []
_next_update_id = 1
_next_message_id = 100
CHAT_ID = 4242
USER_ID = 4242


async def _parse_body(request: Request) -> dict:
    """Bot API params arrive form-encoded in python-telegram-bot 22.x; JSON is accepted too."""
    raw = await request.body()
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        return json.loads(raw) if raw else {}
    if ctype.startswith("application/x-www-form-urlencoded"):
        return {k: v[0] for k, v in parse_qs(raw.decode()).items()}
    return {}


def _message(text: str) -> dict[str, Any]:
    global _next_message_id
    _next_message_id += 1
    message = {
        "message_id": _next_message_id,
        "date": int(time.time()),
        "text": text,
        "chat": {"id": CHAT_ID, "type": "private"},
        "from": {"id": USER_ID, "is_bot": False, "first_name": "Test"},
    }
    # Commands are only recognized because Telegram marks them as entities; without this
    # python-telegram-bot treats "/start x" as ordinary text and no CommandHandler fires.
    if text.startswith("/"):
        command = text.split()[0]
        message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(command)}]
    return message


def _update(text: str) -> dict[str, Any]:
    global _next_update_id
    update = {"update_id": _next_update_id, "message": _message(text)}
    _next_update_id += 1
    return update


@app.post("/inject")
async def inject(text: str) -> dict:
    """Queue an inbound message, as if the user typed it into the chat."""
    await _pending.put(text)
    return {"queued": text}


@app.get("/sent")
async def sent() -> list[dict]:
    """Everything the bot has sent so far (text only)."""
    return _sent


@app.post("/clear")
async def clear() -> dict:
    _sent.clear()
    return {"cleared": True}


@app.post("/bot{token}/{method}")
async def bot_method(token: str, method: str, request: Request):
    global _sent
    body = await _parse_body(request)

    if method == "getMe":
        return {
            "ok": True,
            "result": {
                "id": 1,
                "is_bot": True,
                "first_name": "Fake Bot",
                "username": "fake_bot",
                "can_join_groups": False,
                "can_read_all_group_messages": False,
                "supports_inline_queries": False,
            },
        }

    if method == "getUpdates":
        timeout = min(float(body.get("timeout", 0) or 0), 5.0)
        updates = []
        try:
            text = await asyncio.wait_for(_pending.get(), timeout=timeout or 0.01)
            updates.append(_update(text))
            # Drain anything else queued so a burst arrives in one poll.
            while not _pending.empty():
                updates.append(_update(_pending.get_nowait()))
        except TimeoutError:
            pass
        return {"ok": True, "result": updates}

    if method == "sendMessage":
        chat_id = int(body.get("chat_id", CHAT_ID))
        text = body.get("text", "")
        _sent.append({"chat_id": chat_id, "text": text})
        return {"ok": True, "result": _message(text) | {"chat": {"id": chat_id, "type": "private"}}}

    if method == "deleteMessage":
        return {"ok": True, "result": True}

    if method in ("deleteWebhook", "setMyCommands", "getWebhookInfo"):
        return {"ok": True, "result": {"url": ""} if method == "getWebhookInfo" else True}

    return JSONResponse({"ok": False, "error_code": 404, "description": f"Method {method} not found"}, status_code=404)
