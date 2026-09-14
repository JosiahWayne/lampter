"""Duration and time-limit formatting."""

from __future__ import annotations

import pytest

from lampter.duration import (
    format_epoch,
    format_time_limit,
    humanize_seconds,
    progress_bar,
)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "-"),
        (0, "0s"),
        (45, "45s"),
        (59, "59s"),
        (60, "1m00s"),
        (754, "12m34s"),
        (3600, "1h00m"),
        (56545, "15h42m"),
        (86400, "1d00h"),
        (235422, "2d17h"),
    ],
)
def test_humanize_seconds(seconds, expected):
    assert humanize_seconds(seconds) == expected


def test_humanize_seconds_clamps_negatives():
    """A job that overran its limit shows 0s rather than a negative duration."""
    assert humanize_seconds(-10) == "0s"


@pytest.mark.parametrize(
    ("minutes", "infinite", "expected"),
    [
        (None, False, "UNLIMITED"),
        (4320, True, "UNLIMITED"),
        (30, False, "00:30:00"),
        (240, False, "04:00:00"),
        (4320, False, "3-00:00:00"),
    ],
)
def test_format_time_limit(minutes, infinite, expected):
    assert format_time_limit(minutes, infinite) == expected


def test_format_epoch_handles_missing_values():
    assert format_epoch(None) == "-"
    assert format_epoch(0) == "-"


def test_format_epoch_renders_a_timestamp():
    # Rendered in local time, so assert the shape rather than the value.
    rendered = format_epoch(1789057843, "%Y-%m-%d")
    assert len(rendered) == 10 and rendered[4] == "-"


def test_progress_bar_endpoints():
    assert progress_bar(0.0, 5) == "░░░░░"
    assert progress_bar(1.0, 5) == "█████"
    assert progress_bar(0.5, 4) == "██░░"


def test_progress_bar_clamps_out_of_range():
    assert progress_bar(-1.0, 5) == "░░░░░"
    assert progress_bar(2.0, 5) == "█████"
