#!/usr/bin/env python3
"""Send a plain-text Telegram message using the project's existing bot
credentials. Used by ops scripts (e.g. backup_dropbox.sh) to alert on failure.

Token resolution: $TELEGRAM_BOT_TOKEN, else notifications.telegram.bot_token in
config/config.yaml. chat_id comes from config/config.yaml. Standard library
only, so it runs under systemd without activating the project venv.

Usage:
    notify_telegram.py "message text"   # send a message
    notify_telegram.py --check          # verify creds resolve; send nothing

Exit 0 on success; non-zero (with a token-scrubbed reason on stderr) otherwise.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def resolve_credentials() -> tuple[str, str]:
    """Return (bot_token, chat_id); either may be empty if unresolved."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = ""
    try:
        import yaml  # PyYAML — present in system python3 and the project venv

        with open(CONFIG_PATH) as fh:
            data = yaml.safe_load(fh) or {}
        telegram = ((data.get("notifications") or {}).get("telegram")) or {}
        if not token:
            token = str(telegram.get("bot_token") or "").strip()
        chat_id = str(telegram.get("chat_id") or "").strip()
    except FileNotFoundError:
        pass
    except Exception as exc:  # malformed YAML etc. — never fatal for env-token path
        sys.stderr.write(f"notify_telegram: could not read config: {exc.__class__.__name__}\n")
    return token, chat_id


def send(token: str, chat_id: str, text: str) -> bool:
    """POST a sendMessage. Returns True on HTTP 200. Never leaks the token."""
    def scrub(s: str) -> str:
        return s.replace(token, "<bot_token>") if token else s

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(url, data=payload, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True
            sys.stderr.write(f"notify_telegram: HTTP {resp.status}\n")
            return False
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode()[:200]
        except Exception:
            pass
        sys.stderr.write(scrub(f"notify_telegram: HTTP {exc.code}: {body}\n"))
        return False
    except Exception as exc:
        sys.stderr.write(scrub(f"notify_telegram: {exc.__class__.__name__}: {exc}\n"))
        return False


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: notify_telegram.py <message> | --check\n")
        return 2

    token, chat_id = resolve_credentials()
    missing = [name for name, val in (("bot_token", token), ("chat_id", chat_id)) if not val]
    if missing:
        sys.stderr.write(f"notify_telegram: missing {', '.join(missing)}\n")
        return 1

    if argv[1] in ("--check", "--dry-run"):
        tail = token[-4:] if len(token) >= 4 else "?"
        print(f"OK: resolved bot_token (…{tail}) and chat_id {chat_id}; no message sent.")
        return 0

    return 0 if send(token, chat_id, argv[1]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
