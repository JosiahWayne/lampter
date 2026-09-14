"""Duration and timestamp formatting shared by the TUI and the CLI."""

from __future__ import annotations

from datetime import datetime


def humanize_seconds(seconds: float | None) -> str:
    """Compact, fixed-width-ish duration: ``45s``, ``12m30s``, ``15h42m``, ``2d17h``.

    Returns ``"-"`` for ``None``. Seconds are only shown below the minute mark and
    below the hour mark, which keeps the busiest columns narrow.
    """
    if seconds is None:
        return "-"
    try:
        total = int(seconds)
    except (TypeError, ValueError):
        return "-"
    # A job that overran its limit must not render as a negative duration.
    total = max(total, 0)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, secs = divmod(total, 60)
        return f"{minutes}m{secs:02d}s"
    if total < 86400:
        hours, rem = divmod(total, 3600)
        minutes = rem // 60
        return f"{hours}h{minutes:02d}m"
    days, rem = divmod(total, 86400)
    hours = rem // 3600
    return f"{days}d{hours:02d}h"


def format_epoch(seconds: int | None, fmt: str = "%m-%d %H:%M") -> str:
    """Render an epoch timestamp in the local timezone, or ``"-"``."""
    if not seconds:
        return "-"
    try:
        return datetime.fromtimestamp(int(seconds)).strftime(fmt)
    except (OverflowError, OSError, ValueError):
        return "-"


def format_time_limit(minutes: int | None, infinite: bool = False) -> str:
    """Render a Slurm time limit in the controller's own ``D-HH:MM:SS`` notation."""
    if infinite or minutes is None:
        return "UNLIMITED"
    total = int(minutes) * 60
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def format_tres_memory(raw: str | None) -> str:
    """Tidy a TRES memory value: Slurm writes ``128G``, ``1500M`` and ``64000M``."""
    if not raw:
        return "-"
    return raw


def progress_bar(fraction: float, width: int = 10) -> str:
    """A tiny text progress bar used to show how much of the time limit is spent."""
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)
