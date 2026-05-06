#!/usr/bin/env python3
"""Print the chat IDs from recent messages to your Telegram bot.

Usage:
    1. Send any message to your bot from Telegram.
    2. From the repo root: scripts/get_telegram_chat_id.py
       (Make sure TELEGRAM_BOT_TOKEN is set, e.g. via .env.)

Each row shows: chat_id  |  type  |  display name  |  message text.
The chat_id is what you put in config.yaml under notifications.telegram.
"""

from __future__ import annotations

import os
import sys

import httpx


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        sys.stderr.write(
            "TELEGRAM_BOT_TOKEN env var is not set. "
            "Put it in .env or export it before running.\n"
        )
        return 2

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        resp = httpx.get(url, timeout=10.0)
    except httpx.HTTPError as e:
        sys.stderr.write(f"HTTP error: {e}\n")
        return 1

    if resp.status_code != 200:
        sys.stderr.write(f"Telegram returned HTTP {resp.status_code}: {resp.text[:200]}\n")
        return 1

    payload = resp.json()
    if not payload.get("ok"):
        sys.stderr.write(f"Telegram error: {payload.get('description')}\n")
        return 1

    updates = payload.get("result", [])
    if not updates:
        print(
            "No updates yet. Send any message to your bot from Telegram, "
            "then re-run this script.",
            file=sys.stderr,
        )
        return 0

    seen: set[tuple[int, str]] = set()
    print(f"{'chat_id':>14}  {'type':<10}  {'name':<32}  text")
    print("-" * 80)
    for upd in updates:
        msg = upd.get("message") or upd.get("edited_message") or {}
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        chat_type = chat.get("type", "")
        if chat_id is None:
            continue
        key = (chat_id, chat_type)
        if key in seen:
            continue
        seen.add(key)
        name = (
            chat.get("title")
            or " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")]))
            or chat.get("username")
            or "?"
        )
        text = (msg.get("text") or "").replace("\n", " ")[:40]
        print(f"{chat_id:>14}  {chat_type:<10}  {name:<32}  {text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
