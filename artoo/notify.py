"""Push messages to channels from anywhere (crons, errors, periodic updates).

Telegram-only for now. When more channels go live, this becomes the dispatch
layer that picks the right one.
"""
from __future__ import annotations

import logging

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

from . import config

log = logging.getLogger("artoo.notify")

_bot: Bot | None = None


def _telegram_bot() -> Bot:
    global _bot
    if _bot is None:
        _bot = Bot(token=config.require("TELEGRAM_BOT_TOKEN"))
    return _bot


async def to_telegram(text: str, chat_id: str | int | None = None) -> None:
    """Send `text` to a Telegram chat. Defaults to TELEGRAM_HOME_CHANNEL."""
    target = chat_id or config.optional("TELEGRAM_HOME_CHANNEL")
    if not target:
        log.warning("notify: no chat_id and no TELEGRAM_HOME_CHANNEL set; dropping message")
        return
    try:
        await _telegram_bot().send_message(chat_id=int(target), text=text)
    except Exception as e:  # noqa: BLE001
        log.error("notify: telegram send failed: %s", e)


async def to_telegram_with_keyboards(
    text: str,
    keyboards: list[list[list[tuple[str, str]]]],
    chat_id: str | int | None = None,
) -> None:
    """Post a digest message followed by one inline-keyboard message per
    item. Each keyboard spec is a list-of-rows; each row a list of
    (label, callback_data) tuples. Telegram caps callback_data at 64
    bytes — keep callbacks short (`prefix:action:id` is plenty).

    Sending the body and each keyboard as separate messages keeps the
    digest scannable and lets each pair's buttons live next to its row
    in the chat. Errors per-message are logged but don't abort the rest
    so a single failure doesn't lose the whole digest.
    """
    target = chat_id or config.optional("TELEGRAM_HOME_CHANNEL")
    if not target:
        log.warning("notify: no chat_id and no TELEGRAM_HOME_CHANNEL set; dropping message")
        return
    bot = _telegram_bot()
    try:
        await bot.send_message(chat_id=int(target), text=text, parse_mode="Markdown")
    except Exception as e:  # noqa: BLE001
        log.error("notify: digest body send failed: %s", e)
        return
    for idx, rows in enumerate(keyboards):
        markup = InlineKeyboardMarkup([
            [InlineKeyboardButton(label, callback_data=cb) for (label, cb) in row]
            for row in rows
        ])
        try:
            await bot.send_message(
                chat_id=int(target),
                text=f"Pair {idx + 1}:",
                reply_markup=markup,
            )
        except Exception as e:  # noqa: BLE001
            log.error("notify: keyboard %d send failed: %s", idx, e)
