"""Tests for the agent registry (name → bot token, agents.json preferred)."""
import json

import pytest

from registry import RegistryError, agent_names, load_registry, resolve_bot_token


def _write(tmp_path, payload) -> str:
    path = tmp_path / "agents.json"
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload, encoding="utf-8")
    return str(path)


class TestLoadRegistry:
    def test_missing_file_is_not_an_error(self, tmp_path):
        """No agents.json = legacy setup; the caller falls back to env vars."""
        assert load_registry(str(tmp_path / "absent.json")) is None

    def test_entries_preserve_order_and_tokens(self, tmp_path):
        path = _write(tmp_path, {"agents": [
            {"name": "Opus", "bot_token": "1:a"},
            {"name": "model_test_agent", "bot_token": "2:b"},
        ]})

        entries = load_registry(path)

        assert [(e.name, e.bot_token) for e in entries] == [("Opus", "1:a"), ("model_test_agent", "2:b")]

    @pytest.mark.parametrize(
        "payload, message",
        [
            ({"agents": [{"name": "A"}, {"name": "A"}]}, "duplicate name"),
            ({"agents": [{"bot_token": "x"}]}, "non-empty 'name'"),
            ({"nope": []}, "'agents' list"),
            ("{not json", "Cannot read agent registry"),
        ],
    )
    def test_malformed_registry_fails_loudly(self, tmp_path, payload, message):
        """A broken file must not silently fall back to env vars — a typo would hide."""
        path = _write(tmp_path, payload)

        with pytest.raises(RegistryError, match=message):
            load_registry(path)


class TestResolveBotToken:
    def test_registry_entry_wins_over_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN_OPUS", "env:token")
        path = _write(tmp_path, {"agents": [{"name": "Opus", "bot_token": "file:token"}]})

        assert resolve_bot_token("Opus", path) == "file:token"

    def test_name_missing_from_file_falls_back_to_env(self, tmp_path, monkeypatch, capsys):
        """Half-migrated config: file exists but doesn't list this agent yet."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN_SONNET", "env:token")
        path = _write(tmp_path, {"agents": [{"name": "Opus", "bot_token": "file:token"}]})

        assert resolve_bot_token("Sonnet", path) == "env:token"
        assert "not in agents.json" in capsys.readouterr().out

    def test_generic_env_token_is_last_resort(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "generic:token")

        assert resolve_bot_token("AnyAgent", str(tmp_path / "absent.json")) == "generic:token"

    def test_empty_token_in_file_is_an_error(self, tmp_path):
        """An entry with a blank token means the operator started wiring it up and stopped."""
        path = _write(tmp_path, {"agents": [{"name": "model_test_agent", "bot_token": ""}]})

        with pytest.raises(RegistryError, match="bot_token is empty"):
            resolve_bot_token("model_test_agent", path)

    def test_nothing_configured_names_where_it_looked(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN_GHOST", raising=False)
        path = _write(tmp_path, {"agents": [{"name": "Opus", "bot_token": "file:token"}]})

        with pytest.raises(RegistryError) as exc:
            resolve_bot_token("Ghost", path)

        message = str(exc.value)
        assert "Ghost" in message and "Opus" in message and "TELEGRAM_BOT_TOKEN_GHOST" in message


class TestAgentNames:
    def test_lists_names_for_the_launcher(self, tmp_path):
        path = _write(tmp_path, {"agents": [{"name": "Opus", "bot_token": "1:a"}, {"name": "Haiku", "bot_token": "2:b"}]})

        assert agent_names(path) == ["Opus", "Haiku"]

    def test_absent_file_yields_empty_list(self, tmp_path):
        assert agent_names(str(tmp_path / "absent.json")) == []
