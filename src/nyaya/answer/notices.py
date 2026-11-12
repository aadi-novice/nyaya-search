"""The four standing notices, shown on every response (design, "Notices")."""

from __future__ import annotations

from nyaya.config import Settings


def notices(settings: Settings) -> list[str]:
    """Not legal advice; overruling not tracked; hosted model; IPC/CrPC/Evidence Act replaced."""
    return list(settings.notices)
