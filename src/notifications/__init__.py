"""Notification subsystem (Telegram for now).

Public entry point: `build_dispatcher()`. Returns a NotificationDispatcher
ready to be called from the pipeline, or None if disabled / misconfigured.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from config.settings import settings
from src.notifications.dispatcher import NotificationDispatcher
from src.notifications.telegram import TelegramNotifier
from src.notifications.watchlist import GateConfig, WatchlistGate

logger = logging.getLogger(__name__)

_STATE_PATH = settings.project_root / "run" / "notifications.json"


def build_dispatcher() -> NotificationDispatcher | None:
    """Construct a dispatcher from settings + environment, or None if off."""
    cfg = settings.telegram
    if not cfg.enabled:
        return None
    if not cfg.chat_id:
        logger.warning("Telegram notifications enabled but no chat_id set")
        return None
    if not cfg.watchlist:
        logger.warning("Telegram notifications enabled but watchlist is empty")
        return None
    if not cfg.site_base_url:
        logger.warning("Telegram notifications enabled but site_base_url is empty")
        return None

    # Prefer config.yaml (notifications.telegram.bot_token); fall back to
    # the TELEGRAM_BOT_TOKEN env var. config.yaml is gitignored.
    token = (cfg.bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
    if not token:
        logger.warning(
            "Telegram notifications enabled but no bot token configured "
            "(set notifications.telegram.bot_token in config.yaml or the "
            "TELEGRAM_BOT_TOKEN env var); notifications will not be sent"
        )
        return None

    gate = WatchlistGate(
        GateConfig(
            enabled=cfg.enabled,
            chat_id=cfg.chat_id,
            site_base_url=cfg.site_base_url,
            photo_kind=cfg.photo_kind,
            cooldown_seconds=cfg.cooldown_seconds,
            daily_cap=cfg.daily_cap,
            quiet_hours_start=cfg.quiet_hours_start,
            quiet_hours_end=cfg.quiet_hours_end,
            watchlist=dict(cfg.watchlist),
            timezone=settings.timezone,
        ),
        state_path=_STATE_PATH,
    )
    notifier = TelegramNotifier(bot_token=token)
    dispatcher = NotificationDispatcher(notifier=notifier, gate=gate)
    logger.info(
        "Telegram notifications: enabled, watchlist=%d species, "
        "cooldown=%ds, cap=%d/day, quiet=%s-%s",
        len(cfg.watchlist),
        cfg.cooldown_seconds,
        cfg.daily_cap,
        cfg.quiet_hours_start or "off",
        cfg.quiet_hours_end or "off",
    )
    return dispatcher


def photo_url_for(detection_id: int, kind: str) -> str:
    """Build the URL Telegram will fetch for the photo body."""
    base = settings.telegram.site_base_url
    # kind ∈ {annotated, crop, frame} — matches the FastAPI image routes.
    return f"{base}/api/detections/{detection_id}/{kind}"


def deep_link_for(detection_id: int) -> str:
    """Build the URL the user will be taken to when tapping the message."""
    base = settings.telegram.site_base_url
    return f"{base}/dashboard/?d={detection_id}"
