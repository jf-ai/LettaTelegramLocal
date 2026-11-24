"""Run the bridge against fake_bot_api.py instead of Telegram, for manual end-to-end tests.

The bridge itself has no switch for this. A base-URL setting in production config is a
way to point a real bot at a server that can forge any update (the token travels in the
URL path, so it would leak there too). The redirect lives here instead: this script wraps
Application.builder before the bridge builds its app.

    uv run --with fastapi --with uvicorn uvicorn fake_bot_api:app --port 8900
    AGENT_NAME=Opus SERVER_URL=http://localhost:8008 uv run python run_against_fake_api.py
    curl -X POST 'localhost:8900/inject?text=/start%20fake-password'
"""

import os

FAKE_API_BASE_URL = "http://127.0.0.1:8900/bot"
FAKE_TOKEN = "123:fake"

# Set before the bridge is imported, because it reads config at import time; these win over
# the real .env because load_dotenv never overrides a variable that is already set.
os.environ.update(
    AUTHORIZED_USER="4242",           # fake_bot_api.USER_ID
    TELEGRAM_PASSWORD="fake-password",
    TELEGRAM_BOT_TOKEN=FAKE_TOKEN,    # lets the import succeed where no real token exists
)
os.environ.setdefault("SKIP_FIRST_PING", "True")

from telegram.ext import Application  # noqa: E402

import LettaTelegramLocal as bridge  # noqa: E402  (must follow the env setup)

# A registry entry or TELEGRAM_BOT_TOKEN_<NAME> still wins over the generic variable above,
# so a real token may have been resolved; it must never reach the fake server.
bridge.TELEGRAM_BOT_TOKEN = FAKE_TOKEN

_real_builder = Application.builder
Application.builder = staticmethod(lambda: _real_builder().base_url(FAKE_API_BASE_URL))

if __name__ == "__main__":
    bridge.main()
