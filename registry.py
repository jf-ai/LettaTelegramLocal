"""Agent registry — which bot token belongs to which agent.

Preferred source is ``agents.json`` next to this file::

    {"agents": [
        {"name": "Opus",          "bot_token": "123:abc"},
        {"name": "model_test_agent", "bot_token": "456:def"}
    ]}

The name is the agent's record name on the Agent Home server (the bridge resolves it via
``/agents/by-name/{name}``, which is case-sensitive). Before this file existed the token
was inferred from the environment as ``TELEGRAM_BOT_TOKEN_<UPPERCASED_NAME>``; that still
works as a fallback so an existing .env keeps running, and it is also used when
``agents.json`` exists but doesn't list the requested name (a half-migrated config
warns rather than silently failing over).

Tokens are never printed: the CLI lists names only.
"""

import json
import os
from dataclasses import dataclass

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents.json")


class RegistryError(RuntimeError):
    """No usable token for an agent, or an unreadable/malformed registry."""


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    bot_token: str


def load_registry(path: str = REGISTRY_PATH) -> list[RegistryEntry] | None:
    """Return the registry entries, or None when there is no agents.json.

    A malformed file raises instead of being treated as absent: falling back to env vars
    would hide a typo in a file the operator just wrote.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RegistryError(f"Cannot read agent registry {path}: {e}") from e

    raw_entries = data.get("agents") if isinstance(data, dict) else None
    if not isinstance(raw_entries, list):
        raise RegistryError(f"Agent registry {path} must be an object with an 'agents' list")

    entries: list[RegistryEntry] = []
    seen: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not raw["name"]:
            raise RegistryError(f"Agent registry {path}: every entry needs a non-empty 'name'")
        if raw["name"] in seen:
            raise RegistryError(f"Agent registry {path}: duplicate name {raw['name']!r}")
        seen.add(raw["name"])
        token = raw.get("bot_token", "")
        if not isinstance(token, str):
            raise RegistryError(f"Agent registry {path}: bot_token for {raw['name']!r} must be a string")
        entries.append(RegistryEntry(name=raw["name"], bot_token=token))
    return entries


def resolve_bot_token(name: str, path: str = REGISTRY_PATH) -> str:
    """Return the bot token for ``name``: registry entry first, then environment.

    Raises RegistryError when neither source has a usable token, naming where it looked.
    """
    entries = load_registry(path)
    if entries is not None:
        by_name = {entry.name: entry for entry in entries}
        entry = by_name.get(name)
        if entry is not None:
            if not entry.bot_token.strip():
                raise RegistryError(
                    f"agents.json lists {name!r} but its bot_token is empty — "
                    f"add the token from BotFather to {path}"
                )
            return entry.bot_token
        print(
            f"[registry] {name!r} is not in {os.path.basename(path)}; "
            f"falling back to TELEGRAM_BOT_TOKEN_{name.upper()}"
        )

    env_specific = os.getenv(f"TELEGRAM_BOT_TOKEN_{name.upper()}")
    env_generic = os.getenv("TELEGRAM_BOT_TOKEN")
    token = env_specific or env_generic
    if not token:
        known = ", ".join(entry.name for entry in (entries or [])) or "none"
        raise RegistryError(
            f"No bot token for agent {name!r}. Looked in {path} (names: {known}), "
            f"TELEGRAM_BOT_TOKEN_{name.upper()} and TELEGRAM_BOT_TOKEN."
        )
    return token


def agent_names(path: str = REGISTRY_PATH) -> list[str]:
    """Names from the registry, in file order. Empty when there is no agents.json."""
    entries = load_registry(path)
    return [entry.name for entry in entries] if entries else []


if __name__ == "__main__":
    # CLI for the launcher: one name per line, nothing when there is no registry file.
    try:
        for agent_name in agent_names():
            print(agent_name)
    except RegistryError as e:
        raise SystemExit(str(e))
