"""Being a good citizen of a shared cluster.

Two failure modes this covers, neither of which the tool used to guard against:

* a configuration that asks a shared login node for data far too often. The probe cannot
  notice -- it does as it is told -- and the person setting `refresh_interval = 0.5` gets
  no feedback from anywhere, because the cost lands on somebody else's machine.
* a monitor left running behind a dropped connection, retrying on schedule forever. Each
  attempt is a fresh SSH handshake that fails, and an unattended terminal produces
  thousands of them.

The arithmetic is deliberately in one module so the thresholds cannot drift apart between
`doctor`, the config loader and the dashboard.
"""

from __future__ import annotations

import itertools

import pytest

from lampter.config import Config, load_config
from lampter.load import (
    BUSY_COMMANDS_PER_HOUR,
    COMMANDS_PER_REFRESH,
    DEFAULT_COMMANDS_PER_HOUR,
    MAX_BACKOFF_SEC,
    MIN_INTERVAL,
    PAUSE_AFTER_FAILURES,
    commands_per_hour,
    describe_load,
    intervals_of,
    load_warnings,
    retry_delay,
)

# ------------------------------------------------------------------ the cost model


def test_the_defaults_cost_what_the_documentation_claims():
    """The README and the example config both quote this number at length.

    If the model changes, those documents are wrong, so the figure is pinned rather than
    recomputed from whatever the code happens to say.
    """
    rate = commands_per_hour(intervals_of(Config()))
    assert rate == pytest.approx(DEFAULT_COMMANDS_PER_HOUR, rel=0.05)
    assert rate == pytest.approx(156.0, abs=2)


def test_the_accounts_section_is_accounted_for_as_the_expensive_one():
    """It is five commands, not one: sacctmgr, sshare, sacctmgr show qos, squeue, sacct."""
    assert COMMANDS_PER_REFRESH["accounts"] == 5
    every_second = dict.fromkeys(COMMANDS_PER_REFRESH, 1.0)
    rate = commands_per_hour(every_second)
    # Every section runs 3600 times an hour, weighted by its command count.
    assert rate == pytest.approx(sum(COMMANDS_PER_REFRESH.values()) * 3600)


def test_a_disabled_collector_costs_nothing():
    intervals = intervals_of(Config())
    intervals["accounts"] = 0.0
    assert commands_per_hour(intervals) < commands_per_hour(intervals_of(Config()))


def test_halving_an_interval_doubles_its_cost():
    """Only that section's share changes, and it changes by exactly its own share."""
    base = commands_per_hour(intervals_of(Config()))
    intervals = intervals_of(Config())
    jobs_share = commands_per_hour({**dict.fromkeys(COMMANDS_PER_REFRESH, 1e9), "jobs": 60.0})
    intervals["jobs"] = 30.0

    doubled = commands_per_hour(intervals)

    assert jobs_share == pytest.approx(60.0)  # one squeue a minute
    assert doubled - base == pytest.approx(jobs_share)


def test_describe_load_fits_on_a_line():
    text = describe_load(intervals_of(Config()))
    assert "commands/min" in text
    assert "/hour" in text
    assert len(text) < 60


# ------------------------------------------------------------------ the warnings


def test_the_defaults_produce_no_warnings():
    """A shipped default that complains about itself would be noise."""
    assert load_warnings(intervals_of(Config())) == []


def test_a_busy_cadence_is_reported_with_its_own_number():
    """The message has to say how bad, or the reader cannot judge what to change."""
    intervals = intervals_of(Config())
    intervals["jobs"] = 1.0

    rate = commands_per_hour(intervals)
    warnings = load_warnings(intervals)

    assert rate > BUSY_COMMANDS_PER_HOUR
    assert any("commands an hour" in warning for warning in warnings)
    # The number in the message is the reader's own, not a generic figure.
    assert any(f"{rate:.0f}" in warning for warning in warnings)


def test_an_impossible_interval_is_reported_separately():
    """Below the floor the poll rate is the network's, not the interval's."""
    intervals = intervals_of(Config())
    intervals["jobs"] = 0.5

    warnings = load_warnings(intervals)

    assert any("below the" in warning and "floor" in warning for warning in warnings)


def test_the_threshold_leaves_room_for_a_deliberate_choice():
    """Four times the defaults is nobody's business; a missing zero is not."""
    intervals = intervals_of(Config())
    intervals["jobs"] = 15.0  # a deliberately brisk 4/minute
    assert load_warnings(intervals) == []
    assert commands_per_hour(intervals) < BUSY_COMMANDS_PER_HOUR


def test_the_config_loader_actually_says_something(tmp_path):
    """End to end: the warning has to reach `config.warnings`, which `doctor` prints."""
    config_file = tmp_path / "lampter.toml"
    config_file.write_text("refresh_interval = 0.5\n")

    config = load_config(path=config_file, env={})

    assert any("floor" in warning for warning in config.warnings)
    assert any("commands an hour" in warning for warning in config.warnings)


def test_the_floor_is_the_floor():
    """Exactly at the floor there is nothing to complain about on those grounds.

    A rate warning may still fire -- two seconds between `sinfo` calls is genuinely a
    lot -- but "below the floor" no longer applies, and the two complaints mean
    different things.
    """
    at_floor = intervals_of(Config(partition_interval=MIN_INTERVAL))
    assert not any("floor" in warning for warning in load_warnings(at_floor))

    just_under = intervals_of(Config(partition_interval=MIN_INTERVAL - 0.1))
    assert any("floor" in warning for warning in load_warnings(just_under))


# ------------------------------------------------------------------ the retry policy


def test_the_first_failure_waits_the_normal_interval():
    assert retry_delay(60.0, 1) == 60.0


def test_repeated_failures_back_off_and_then_stop_growing():
    """A dropped VPN is not fixed by asking more often."""
    delays = [retry_delay(60.0, n) for n in range(1, 12)]
    assert delays[0] == 60.0
    assert delays[1] == 120.0
    assert delays[2] == 240.0
    assert delays[3] == MAX_BACKOFF_SEC  # 480 would exceed the ceiling
    assert delays[-1] == MAX_BACKOFF_SEC
    # Monotonic, and never below the configured interval.
    assert all(b >= a for a, b in itertools.pairwise(delays))


def test_backoff_is_capped_for_a_very_short_interval():
    assert retry_delay(2.0, 20) == MAX_BACKOFF_SEC


def test_the_pause_threshold_is_after_the_backoff_has_had_a_chance():
    """Ten attempts at the backed-off pace is minutes of patience, not seconds."""
    intervals = intervals_of(Config())
    total = sum(retry_delay(intervals["jobs"], n) for n in range(1, PAUSE_AFTER_FAILURES))
    assert total > 60
    assert PAUSE_AFTER_FAILURES >= 5
