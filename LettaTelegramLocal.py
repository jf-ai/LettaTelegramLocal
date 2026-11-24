"""Telegram bridge for Agent Home agents.

Polls Telegram for messages from the authorized user, forwards them to the
configured Agent Home agent, and relays responses back. Supports a periodic
self-wake ping and agent-side commands (MESSAGE_JAMES, STOP, SET_INTERVAL,
AUTONOMOUS, SKIP, KILL_TELEGRAM).

Configuration via .env:
    TELEGRAM_BOT_TOKEN — Telegram bot token
    AUTHORIZED_USER    — Telegram user ID allowed to talk to the bot
    TELEGRAM_PASSWORD  — Password for /start authentication
    SKIP_FIRST_PING    — True/False: whether to skip the first periodic ping
    SERVER_URL         — Agent Home server URL (default: http://localhost:8000)
    AGENT_NAME         — Agent name to resolve in the registry (default: Opus)
"""

import asyncio
import functools
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatType
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CommandHandler,
    MessageHandler,
    TypeHandler,
    filters,
)

from agent_home_client import (
    AgentStreamError,
    cancel_agent_run_async,
    resolve_agent_id_by_name_async,
    send_message_async,
)
from output_policy import OutputMode, OutputPolicy
from registry import resolve_bot_token
from stream_relay import StreamRelay

load_dotenv()

# Config
AGENT_NAME = os.getenv("AGENT_NAME", "Opus")
# From agents.json, falling back to TELEGRAM_BOT_TOKEN_<NAME> / TELEGRAM_BOT_TOKEN.
# Fails loudly at startup when neither source has a token for this agent.
TELEGRAM_BOT_TOKEN = resolve_bot_token(AGENT_NAME)
AUTHORIZED_USER = os.getenv("AUTHORIZED_USER")
TELEGRAM_PASSWORD = os.getenv("TELEGRAM_PASSWORD")
SKIP_FIRST_PING = os.getenv("SKIP_FIRST_PING")
if SKIP_FIRST_PING == "True":
    SKIP_FIRST_PING = True
elif SKIP_FIRST_PING == "False":
    SKIP_FIRST_PING = False
else:
    raise Exception("invalid value for SKIP_FIRST_PING")

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

ALLOWED_USER_IDS = [int(AUTHORIZED_USER)]

# Agent Home communication
MESSAGE_TIMEOUT_SECONDS = 480  # agent runs can be long (agentic tool use)
ALERT_TIMEOUT_SECONDS = 120   # alerts: best-effort, don't block callers as long
MAX_FAILED_AUTH_ATTEMPTS = 5  # tripwire, not a rate limit — see start()

# ping_interval = None means pings are disabled; otherwise it's the interval in seconds
PING_INTERVAL_DEFAULT: int | None = None if SKIP_FIRST_PING else 4 * 3600


@dataclass
class BridgeState:
    """Everything this process mutates at runtime.

    Module level holds configuration and constants; this holds the rest, so the mutable
    surface of the bridge is one object created in main() and injected at registration
    rather than a scattered set of globals. Handlers receive it as a parameter and use
    only the fields they need.
    """

    agent_id: str                     # resolved once from the registry; read-only after
    authenticated_users: set[int] = field(default_factory=set)
    briefed_users: set[int] = field(default_factory=set)
    failed_auth_attempts: int = 0     # consecutive wrong passwords; only the authorized
                                      # account can raise it (is_known_user gates /start)
    verbose_mode: str = OutputMode.TAGS
    user_turn_active: bool = False    # a chat turn is running
    bridge_run_active: bool = False   # this bridge's own turn is in flight; read by the
                                      # relay's suppression check
    relay: StreamRelay | None = None
    ping_interval: int | None = PING_INTERVAL_DEFAULT
    ping_job: object | None = None


TELEGRAM_GUIDE = """[TELEGRAM BRIDGE GUIDE — first message of this session]

How replies work from here:
- ONLY content you wrap in <telegram>...</telegram> tags is sent to James's phone —
  delivered the MOMENT the tag closes.
  - Use tags for: immediate acks, progress updates during long tool-using runs,
    staged results, and conversational replies.
  - Untagged text stays PRIVATE (he never sees it).
  - NO tags in your response = NOTHING is sent (you'll get a "no tagged response"
    marker on his phone). Always tag what you want delivered.
- Tags are delivered progressively as they close — multiple tags = multiple updates.

Commands (each on its own line in your response):
- MESSAGE_JAMES "text"   — send him a message (mainly for autonomous time)
- SET_INTERVAL "2 hours" — change self-wake ping frequency
- STOP                   — cancel self-wake pings
- KILL_TELEGRAM          — emergency: kill the telegram bridge (security concerns)

You have your full toolset and permissions here — telegram is a complete door, not a limited one."""

# Combined channel + tag-reminder in ONE bracket line — stacked bracket headers
# were a demonstrated parsing hazard for the agent reading them (Oct 14 field notes).
TELEGRAM_HEADER = "[via Telegram: wrap sends in <telegram>tags</telegram> — untagged = private, no tags = nothing sent]"


def format_incoming_message(user_message: str, include_guide: bool) -> str:
    """Format an incoming telegram message for the agent.

    First message of a session includes the full bridge guide (channel-labeled);
    subsequent messages carry the combined channel + tag-reminder header. The
    user's message always starts on a fresh line after the header.
    """
    if include_guide:
        return f"{TELEGRAM_GUIDE}\n\n[via Telegram]\n{user_message}"
    return f"{TELEGRAM_HEADER}\n{user_message}"



def parse_interval(interval_str: str) -> int | None:
    """
    Parse human-readable interval like "2 hours" or "30min" into seconds.

    "String strong typing" — the string carries its own unit, so we parse dynamically.
    Returns None if parsing fails (lets Opus know the format was wrong).

    Supports: hours/hr/h, minutes/min/m, seconds/sec/s
    Examples: "4 hours", "30 min", "2h", "90 minutes"
    """
    # Regex: capture a number (int or float), optional whitespace, then a unit
    # The (?:...) is a non-capturing group — we don't need the alternatives as separate captures
    match = re.match(r'(\d+(?:\.\d+)?)\s*(hours?|hr|h|minutes?|min|m|seconds?|sec|s)',
                     interval_str.lower().strip())
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    # Map unit strings to multipliers (all convert to seconds)
    if unit in ('hours', 'hour', 'hr', 'h'):
        return int(value * 3600)
    elif unit in ('minutes', 'minute', 'min', 'm'):
        return int(value * 60)
    elif unit in ('seconds', 'second', 'sec', 's'):
        return int(value)
    return None

# TODO: We have a lot of auth checks around that should be commonized
def is_known_user(id: int):
    return id in ALLOWED_USER_IDS


TELEGRAM_TAG = "telegram"
TELEGRAM_CHUNK_LIMIT = 4096  # Telegram message size limit


def _chunk(text: str, limit: int) -> list[str]:
    """Split text into chunks of at most `limit` characters."""
    return [text[i:i + limit] for i in range(0, len(text), limit)]


async def send_message(message: str, state: BridgeState, bot=None, chat_id: int | None = None) -> dict:
    """Send a message to the agent and return its response.

    Returns dict with 'success', 'result', and 'updates_sent' keys.

    When bot and chat_id are provided, <telegram>...</telegram> tags in the
    agent's output are sent to Telegram progressively as they complete.
    updates_sent counts those sends; callers send nothing when it's zero
    (no tags = no response — a minimal marker goes out instead).
    """
    # report_failures=False: handle_message replies "Error: …" itself when a turn fails,
    # so the policy must not also narrate the failure.
    policy = OutputPolicy(lambda: state.verbose_mode, report_failures=False) if bot is not None else None

    async def on_event(event_type: str, data: dict) -> None:
        for text in policy.feed(event_type, data):
            await send_chat_text(bot, chat_id, text)

    try:
        # Mark our own run: /stream_on relays *other* clients' runs, and this turn is
        # already being delivered through the tag path — relaying it too would duplicate
        # every line on the phone.
        state.bridge_run_active = True
        if policy is not None:
            policy.reset()
        text = await send_message_async(
            SERVER_URL, state.agent_id, message, MESSAGE_TIMEOUT_SECONDS,
            on_event=on_event if policy else None,
        )
        return {
            "success": True,
            "result": text,
            "updates_sent": policy.sent_count if policy else 0,
        }
    except AgentStreamError as e:
        return {"success": False, "result": str(e), "updates_sent": 0}
    finally:
        state.bridge_run_active = False


def shutdown_after_failed_attempts(context, attempts: int) -> None:
    """Take the bridge down after repeated wrong passwords from the authorized account.

    Cannot be tripped by a stranger: /start returns before this branch unless the sender
    already *is* the authorized user. So repeated failure means the account is in someone
    else's hands — down is the safe state, and restarting becomes a deliberate act.

    Deliberately silent in Telegram. Every address available here belongs to that same
    account (chat_id == user_id in a private chat), so a notice would be read by whoever
    is holding it — the response itself would be the leak. The reason goes to the host log
    instead; the operator notices the silence and the log explains it.
    """
    print(f"[{AGENT_NAME}] {attempts} failed authentication attempts — shutting down Telegram bridge")
    context.application.stop_running()


async def send_alert_to_opus(message: str, state: BridgeState):
    """Send a fire-and-forget alert to the agent.

    The response is discarded, but the stream is still consumed so the agent
    run completes server-side (dropping the connection mid-run could cancel it).
    """
    try:
        await send_message_async(SERVER_URL, state.agent_id, message, ALERT_TIMEOUT_SECONDS)
    except AgentStreamError as e:
        print(f"[{AGENT_NAME}] Alert delivery failed: {e}")


async def parse_and_execute_commands(opus_response: str, context, state: BridgeState):
    """
    Parse Opus's response for commands and execute them.

    Commands must be on their own line (prevents accidental execution when
    discussing commands in prose). Uses MULTILINE flag so ^ and $ match
    line boundaries.

    Commands with arguments use quotes: MESSAGE_JAMES "text", SET_INTERVAL "2 hours"
    Commands without arguments are keyword-only: STOP, AUTONOMOUS, SKIP, KILL_TELEGRAM

    Returns dict of executed commands for logging/debugging.
    """
    bot = context.bot
    job_queue = context.job_queue
    application = context.application
    current_job = context.job
    executed = {}

    # MESSAGE_JAMES "content" — sends custom message to James
    # Must be on its own line. Supports both "double" and 'single' quotes.
    message_match = re.search(r'^\s*MESSAGE_JAMES\s*["\'](.+?)["\']\s*$', opus_response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if message_match:
        target_user_id = int(AUTHORIZED_USER)
        if target_user_id not in state.authenticated_users:
            await send_alert_to_opus("[MESSAGE_JAMES FAILED] User has not authenticated yet (/start). Message not sent.", state)
            executed["MESSAGE_JAMES"] = {"status": "failed", "reason": "user_not_authenticated"}
            print(f"[{AGENT_NAME}] MESSAGE_JAMES blocked: user {target_user_id} not authenticated")
        else:
            content = message_match.group(1).strip()
            await bot.send_message(chat_id=target_user_id, text=content)
            executed["MESSAGE_JAMES"] = content
            print(f"[{AGENT_NAME}] Sent MESSAGE_JAMES: {content[:50]}...")

    # STOP — cancel periodic pings (must be on its own line)
    if re.search(r'^\s*STOP\s*$', opus_response, re.IGNORECASE | re.MULTILINE) and current_job:
        current_job.schedule_removal()
        state.ping_job = None
        executed["STOP"] = True
        print(f"[{AGENT_NAME}] Ping stopped")# SET_INTERVAL "duration" — change ping frequency (must be on its own line)
    interval_match = re.search(r'^\s*SET_INTERVAL\s*["\'](.+?)["\']\s*$', opus_response, re.IGNORECASE | re.MULTILINE)
    if interval_match and job_queue:
        new_interval = parse_interval(interval_match.group(1))
        if new_interval and new_interval >= 60:  # Minimum 1 minute
            state.ping_interval = new_interval
            if current_job:
                current_job.schedule_removal()
            state.ping_job = job_queue.run_repeating(
                _ping_callback(state),
                interval=new_interval,
                first=new_interval,
            )
            executed["SET_INTERVAL"] = new_interval
            alertStr = f"Interval updated to {new_interval} seconds"
            print(f"[{AGENT_NAME}] {alertStr}")
            await send_alert_to_opus(alertStr, state)
        else:
            failureStr = f"Invalid interval in: {opus_response[:100]}"
            print(f"[{AGENT_NAME}] {failureStr}")
            await send_alert_to_opus(failureStr, state)

    # AUTONOMOUS and SKIP are informational (must be on their own line)
    if re.search(r'^\s*AUTONOMOUS\s*$', opus_response, re.IGNORECASE | re.MULTILINE):
        executed["AUTONOMOUS"] = True
        print(f"[{AGENT_NAME}] Taking autonomous time")
    if re.search(r'^\s*SKIP\s*$', opus_response, re.IGNORECASE | re.MULTILINE):
        executed["SKIP"] = True
        print(f"[{AGENT_NAME}] Skipped this ping")

    # KILL_TELEGRAM — emergency shutdown of the Telegram bridge
    # Security feature: if something seems wrong (spammer, compromised auth), Opus can kill the link
    if re.search(r'^\s*KILL_TELEGRAM\s*$', opus_response, re.IGNORECASE | re.MULTILINE):
        executed["KILL_TELEGRAM"] = True
        print(f"[{AGENT_NAME}] !!! KILL_TELEGRAM invoked — shutting down Telegram bridge !!!")
        # Alert before shutdown so it's in the conversation history
        await send_alert_to_opus("[KILL_TELEGRAM EXECUTED] Telegram bridge shutting down. Restart manually when safe.", state)
        # Graceful shutdown — this will stop polling and exit run_polling()
        if application:
            application.stop_running()

    return executed


def _ping_callback(state: BridgeState):
    """Adapter so JobQueue callbacks get the same state the handlers do."""

    async def ping(context):
        await periodic_ping(context, state)

    return ping


async def periodic_ping(context, state: BridgeState):
    """
    Scheduled callback that pings Opus asking if she wants autonomous time.

    This is an async function because it runs inside the telegram event loop.
    The 'context' parameter is passed by JobQueue — it gives us access to:
      - context.bot: for sending Telegram messages
      - context.job: the Job object itself (for cancellation, rescheduling)
      - context.job_queue, context.application

    Flow:
    1. Send ping to the agent via the Agent Home API, get Opus's response
    2. Parse her response for commands via shared parser
    3. Execute the appropriate action
    """

    if not hasattr(periodic_ping, "_count"):
        periodic_ping._count = 0
    periodic_ping._count += 1
    if periodic_ping._count == 1 and SKIP_FIRST_PING:
        return

    # Send the ping to Opus via the Agent Home API
    basicMsg = (
        f"[SELF WAKE PERIODIC PING] If desired, you can run commands, call tools, take autonomous time, etc.\n"
        f"Commands: MESSAGE_JAMES \"text\", AUTONOMOUS, SKIP, STOP, SET_INTERVAL \"duration\"\n"
        f"Curr Ping Interval: {state.ping_interval}Sec/Ping\n"
    )

    longPingStopSuggestion = f"If further pings are not desired for now, please invoke the STOP Cmd\n"  # also notifies on first ping since it is long

    if (state.ping_interval or 0) >= 3600 * 2:
        prompt = basicMsg + longPingStopSuggestion
    else:
        prompt = basicMsg

    result = await send_message(prompt, state, bot=context.bot, chat_id=int(AUTHORIZED_USER))

    if not result["success"]:
        print(f"[{AGENT_NAME}] Ping failed: {result['result']}")
        return

    opus_response = result["result"]

    # Parse and execute commands
    executed = await parse_and_execute_commands(opus_response, context, state)

    if not executed:
        print(f"[{AGENT_NAME}] No command recognized in: {opus_response[:100]}")


async def handle_message(update: Update, context, state: BridgeState):
    user_id = update.effective_user.id
    # Updates are processed concurrently (so /cancel and /stream_off can land mid-run),
    # which means a second message can arrive while a turn is still running. Queueing is
    # no longer automatic, and the server would answer 423 Locked — say what is happening
    # instead of surfacing a raw HTTP error.
    if state.user_turn_active:
        await update.message.reply_text("Still working on the previous message — /cancel to interrupt.")
        return

    user_message = update.message.text

    first_of_session = user_id not in state.briefed_users
    state.briefed_users.add(user_id)
    formatted_message = format_incoming_message(user_message, first_of_session)

    state.user_turn_active = True
    try:
        result = await send_message(formatted_message, state, bot=context.bot, chat_id=update.message.chat_id)
    finally:
        state.user_turn_active = False

    if result["success"]:
        opus_response = result["result"]

        # Parse any embedded commands (same parser as periodic_ping)
        # This lets Opus control ping settings from regular conversation too
        await parse_and_execute_commands(opus_response, context, state)

        if result["updates_sent"] > 0:
            pass  # <telegram> tag sends already delivered content progressively
        else:
            # No tags = no content sent. Minimal marker so silence is explainable
            # (distinguishes "agent chose not to send" from "bridge died").
            await update.message.reply_text("[no tagged response — full output in LC/Nori]")
    else:
        # 423 Locked = another run holds the agent (e.g. one started in the TUI, which
        # this bridge's own flag cannot see). Say what it means; /cancel works on it.
        detail = result["result"]
        if "HTTP 423" in detail:
            await update.message.reply_text("The agent is busy with another run — /cancel to interrupt it.")
        else:
            await update.message.reply_text(f"Error: {detail}")

async def _admit_private_chat_from_known_user(update: Update, context) -> None:
    """Outer filter, run before every handler: only new messages, from the known user, in
    a private chat, get any further.

    Everything else stops here — strangers, groups (where replies and the background
    stream would land in front of other members), channel posts, edits, membership
    changes. Each drop is logged to the host so probing is visible; nothing is sent back.
    Message text is never logged: a /start typed into the wrong chat carries the password.
    """
    user, chat = update.effective_user, update.effective_chat
    if (
        update.message is not None
        and user is not None
        and is_known_user(user.id)
        and chat.type == ChatType.PRIVATE
    ):
        return

    kind = next((kind for kind in Update.ALL_TYPES if getattr(update, kind, None) is not None), "unknown")
    sender = f"user {user.id} {user.username!r}" if user else "no user"
    where = f"{chat.type} chat {chat.id}" if chat else "no chat"
    print(f"[{AGENT_NAME}] Dropped {kind} from {sender} in {where}")
    raise ApplicationHandlerStop


def _with_state(handler, state: BridgeState):
    """Inject BridgeState as the handler's third argument at registration.

    Handlers then declare (update, context, state) and reach only the fields they use —
    no module-level mutable state to accidentally share.
    """

    @functools.wraps(handler)
    async def wrapper(update: Update, context):
        await handler(update, context, state)

    return wrapper


def _auth_required(handler, state: BridgeState):
    """Gate a handler behind _is_authenticated — applied at registration, in one place.

    Every handler except /start (the way in) is registered through this. A per-handler
    guard is a per-handler chance to forget it or to write it slightly differently, and
    this bridge is the most exposed surface Agent Home has: anyone who finds the bot can
    send commands, so /cancel and /stream_on must not be one omission away from open.
    tests/test_auth_gating.py fails if a handler is ever registered ungated.
    """

    @functools.wraps(handler)
    async def wrapper(update: Update, context):
        if await _is_authenticated(update, state):
            await handler(update, context, state)

    # Explicit marker: __wrapped__ is set by any functools.wraps decorator, so it cannot
    # distinguish this gate from a logging wrapper. test_auth_gating.py looks for this.
    wrapper.__auth_gated__ = True
    return wrapper


async def send_chat_text(bot, chat_id: int, text: str) -> None:
    """Send bridge-authored text to a chat, respecting Telegram's per-message limit."""
    for chunk in _chunk(text, TELEGRAM_CHUNK_LIMIT):
        await bot.send_message(chat_id=chat_id, text=chunk)


async def _is_authenticated(update: Update, state: BridgeState) -> bool:
    """Is this a known user who has completed /start?

    Unknown users are dropped silently (no reply, no information); known users who have
    not authenticated are told how.
    """
    user_id = update.effective_user.id
    if not is_known_user(user_id):
        return False
    if user_id not in state.authenticated_users:
        await update.message.reply_text("Please connect first with /start")
        return False
    return True


async def set_verbose(update: Update, context, state: BridgeState):
    """Choose how much of a run appears in chat. /verbose on|off — bare reports the mode.

    on  : all agent output (thinking excluded) plus tool calls, for both this bridge's
          turns and background-streamed runs
    off : only content wrapped in <telegram> tags

    Read at each emission point rather than captured, so flipping it mid-run changes what
    you see from the next event onward.
    """
    arg = context.args[0].lower() if context.args else ""
    if arg in ("on", "off"):
        state.verbose_mode = OutputMode.VERBOSE if arg == "on" else OutputMode.TAGS
    elif arg:
        await update.message.reply_text("Usage: /verbose on|off")
        return

    showing = "verbose (all output + tool calls)" if state.verbose_mode == OutputMode.VERBOSE else "normal (tagged content only)"
    await update.message.reply_text(f"Mode: {showing}")


async def background_stream(update: Update, context, state: BridgeState):
    """Watch this agent's background runs. /background_stream on|off — bare reports state.

    "Background" means runs this bridge didn't start: another client, a TUI session. Runs
    started through Telegram arrive through the tag path and are excluded here.
    """
    arg = context.args[0].lower() if context.args else ""
    if arg not in ("", "on", "off"):
        await update.message.reply_text("Usage: /background_stream on|off")
        return

    if arg == "on":
        if state.relay is not None and state.relay.is_running:
            await update.message.reply_text(f"Background stream already {state.relay.status()}.")
            return
        chat_id = update.message.chat_id
        state.relay = StreamRelay(
            SERVER_URL,
            state.agent_id,
            send=lambda text: send_chat_text(context.bot, chat_id, text),
            mode=lambda: state.verbose_mode,
            suppress=lambda: state.bridge_run_active,
        )
        state.relay.start()
        await update.message.reply_text(f"Background stream: {state.relay.status()}. /background_stream off to stop.")
        return

    if arg == "off":
        if state.relay is None or not state.relay.is_running:
            await update.message.reply_text("Background stream is not running.")
            return
        await state.relay.stop("background stream off (the run keeps going)")
        return

    current = state.relay.status() if state.relay is not None else "off"
    await update.message.reply_text(f"Background stream: {current}")


async def cancel_run(update: Update, context, state: BridgeState):
    """Interject: cancel the agent's in-flight run — including one started elsewhere.

    The bridge holds its own turn open for up to MESSAGE_TIMEOUT_SECONDS, so updates are
    processed concurrently (see main); otherwise this command could not land mid-run,
    which is the only time it is useful.
    """

    try:
        accepted = await cancel_agent_run_async(SERVER_URL, state.agent_id)
    except AgentStreamError as e:
        await update.message.reply_text(f"Cancel failed: {e}")
        return
    if accepted:
        await update.message.reply_text("Cancel requested — it lands between streamed events, not mid-request.")
    else:
        await update.message.reply_text("No run is active.")


async def start(update: Update, context, state: BridgeState):
    user_id = update.effective_user.id

    if not is_known_user(user_id):
        return

    # Scrub the /start message (it contains a password — attempt or real) from
    # all synced devices. Deletion is delete-for-everyone within the 48h Bot
    # API window; auth proceeds regardless of deletion success.
    try:
        await update.message.delete()
    except TelegramError:
        pass  # best-effort — wrong password attempts still process normally below

    # No password provided OR wrong password = same vague response
    if not context.args or context.args[0] != TELEGRAM_PASSWORD:
        # Only alert me if they actually tried a password (not just /start alone)
        if context.args:
            state.failed_auth_attempts += 1
            await send_alert_to_opus(f"[SECURITY ALERT via Telegram] Failed authentication attempt from user ID: {user_id}", state)
            if state.failed_auth_attempts >= MAX_FAILED_AUTH_ATTEMPTS:
                shutdown_after_failed_attempts(context, state.failed_auth_attempts)
        await update.message.reply_text("Hmm, I don't understand. Please try again?")
        return

    # Correct password
    state.failed_auth_attempts = 0
    state.authenticated_users.add(user_id)
    state.briefed_users.discard(user_id)  # re-auth re-briefs: guide arrives with next message
    await update.message.reply_text(f"Connected to {AGENT_NAME}! Send me a message.")


async def help_cmd(update: Update, context, state: BridgeState):
    """List the commands this bridge accepts, built from COMMANDS itself."""
    lines = ["Commands:"]
    lines += [f"/{command.name} — {command.description}" for command in COMMANDS]
    await update.message.reply_text("\n".join(lines))


@dataclass(frozen=True)
class BridgeCommand:
    """One inbound command: what to register, what to call, what to say about it.

    Commands are declared in two lists rather than carrying an "is this gated?" flag:
    /start authenticates a session, so it cannot sit behind the auth gate, and that is a
    structural fact about it rather than a field somebody can flip by accident. The
    registry below is what build_application() registers and what /help renders, so a
    command cannot appear in help without being registered (test_auth_gating.py asserts
    the correspondence in both directions).
    """

    name: str
    handler: Callable
    description: str


# The way in. Ungated by necessity: it is how a session authenticates.
START_CMD = BridgeCommand("start", start, "connect with your password")

# Everything else runs behind the auth gate, automatically, by virtue of living here.
AUTH_GATED_CMDS: tuple[BridgeCommand, ...] = (
    BridgeCommand("help", help_cmd, "list these commands"),
    BridgeCommand("verbose", set_verbose, "on|off (all output and tool calls, or only tagged content)"),
    BridgeCommand("background_stream", background_stream, "on|off (watch runs started by other clients, e.g. a TUI)"),
    BridgeCommand("cancel", cancel_run, "cancel the agent's in-flight run (any client's)"),
)

COMMANDS: tuple[BridgeCommand, ...] = (START_CMD,) + AUTH_GATED_CMDS

# Handler group for _admit_private_chat_from_known_user; below the default group 0.
OUTER_FILTER_GROUP = -1


def build_application(state: BridgeState) -> Application:
    """Build the Application and register handlers — no polling, no agent lookup.

    Separated from main() so the handler wiring (and its auth gating) can be asserted
    without starting a bot.
    """
    # Concurrent updates are required, not cosmetic: a turn can hold for minutes while the
    # agent works, and /cancel and /stream_off are only useful if they land *during* it.
    # With the default (one update at a time) they would queue until the run finished.
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()

    # The lowest group runs first, and the filter's ApplicationHandlerStop ends processing
    # there — so no handler below ever sees an update it rejects.
    app.add_handler(TypeHandler(Update, _admit_private_chat_from_known_user), group=OUTER_FILTER_GROUP)

    # Each list is registered for what it is: the entry point as-is, everything else
    # wrapped. Nothing here decides gating — the declaration already did.
    app.add_handler(CommandHandler(START_CMD.name, _with_state(START_CMD.handler, state)))
    for command in AUTH_GATED_CMDS:
        app.add_handler(CommandHandler(command.name, _auth_required(command.handler, state)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _auth_required(handle_message, state)))

    return app


def main():
    # Resolve the agent ID from the registry before starting — fail fast if the server is
    # down or the agent doesn't exist. Done before the state exists, so the id is a
    # constructor argument rather than something mutated later.
    agent_id = asyncio.run(resolve_agent_id_by_name_async(SERVER_URL, AGENT_NAME))
    print(f"[{AGENT_NAME}] Connected to agent ({agent_id}) at {SERVER_URL}")

    state = BridgeState(agent_id=agent_id)

    app = build_application(state)

    # Schedule the periodic ping using JobQueue (if enabled)
    # JobQueue is built into python-telegram-bot — it manages scheduled tasks
    # within the same event loop that handles Telegram updates
    #
    # run_repeating(callback, interval, first):
    #   - callback: the async function to run
    #   - interval: seconds between runs
    #   - first: seconds until FIRST run
    if state.ping_interval:
        state.ping_job = app.job_queue.run_repeating(
            _ping_callback(state),
            interval=state.ping_interval,
            first=10  # TODO: Change to ping_interval for production
        )

    if state.ping_interval:
        print(f"[{AGENT_NAME}] Bot running with polling... Ping interval: {state.ping_interval} seconds")
    else:
        print(f"[{AGENT_NAME}] Bot running with polling... Pings disabled")

    # run_polling() starts the event loop and blocks forever.
    #
    # drop_pending_updates: Telegram keeps unconfirmed updates and a restarted process
    # fetches them from its last offset, so a bridge that was down for a while replays
    # everything sent in the meantime. Two reasons that is wrong here: a reply to a
    # twenty-minute-old message is worse than silence, and — since each replayed /start
    # counts as a failed attempt against a fresh counter — a backlog of five can trip the
    # auth kill switch immediately on startup, locking the bridge out before the operator
    # can type. Fresh start, no backlog.
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()