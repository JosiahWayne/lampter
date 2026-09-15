"""How much this costs the cluster, and how not to be a nuisance.

The whole design is about restraint: one SSH round trip per refresh, and collectors
staggered onto their own clocks so the expensive ones run rarely. Two things break that
promise, and neither used to be guarded:

* **A configuration that asks too often.** Nothing stopped ``refresh_interval = 0.5``,
  which asks a shared login node for ``squeue`` twice a second -- and, because the SSH
  connection is what actually carries it, that is also twice a second of ``sshd`` work on
  a machine everybody shares. The probe cannot notice this; it just does as it is told.
* **A broken connection left running.** A monitor sitting behind a dropped VPN or a
  laptop that suspended keeps trying, once per interval, forever. Each attempt is a fresh
  SSH handshake that fails, and an unattended terminal can produce thousands of them
  overnight -- the sort of pattern that gets an address throttled or blocked.

So the cost is modelled explicitly in one place, the thresholds live here rather than
being scattered through the UI, and the retry policy is a function that can be tested
without a terminal.
"""

from __future__ import annotations

from collections.abc import Mapping

#: SLURM commands one refresh of each section issues.
#:
#: These are the numbers the README and the example config quote, derived from what the
#: probe's collectors actually run: ``jobs`` is one ``squeue``, ``partitions`` one
#: ``sinfo``, ``history`` one ``sacct``, ``usage`` one ``sstat`` (skipped entirely when
#: nothing is running), and ``accounts`` is the expensive one -- ``sacctmgr`` for the
#: associations, ``sshare``, a filtered ``sacctmgr show qos``, a cluster-wide ``squeue``
#: and a windowed ``sacct``.
COMMANDS_PER_REFRESH: dict[str, int] = {
    "jobs": 1,
    "partitions": 1,
    "history": 1,
    "usage": 1,
    "accounts": 5,
}

#: Section name -> the config field holding its interval.
SECTION_INTERVALS: dict[str, str] = {
    "jobs": "refresh_interval",
    "partitions": "partition_interval",
    "history": "history_interval",
    "usage": "usage_interval",
    "accounts": "accounts_interval",
}

#: What the shipped defaults cost, for comparison in messages and in `doctor`.
DEFAULT_COMMANDS_PER_HOUR = 156.0

#: Above this, the tool says so. The defaults are ~156/hour, so this is roughly four
#: times the shipped configuration: high enough that a deliberately brisker setup is
#: nobody's business, low enough to catch a misplaced decimal point.
BUSY_COMMANDS_PER_HOUR = 600.0

#: No interval may be shorter than this, whatever the config says. Anything faster
#: cannot complete a round trip on a WAN link anyway, so it is always a mistake -- and an
#: expensive one, because the failed attempts still reach the cluster.
MIN_INTERVAL = 2.0

#: Ceiling for the interactive `+`/`-` keys, matching what the config accepts.
MAX_INTERVAL = 600.0

#: Consecutive failures after which auto-refresh stops instead of retrying forever.
#: Ten attempts at a backed-off pace is a few minutes of patience; beyond that the
#: retries are no longer plausibly going to succeed, and continuing is just load.
PAUSE_AFTER_FAILURES = 10

#: Ceiling on the retry delay once failures pile up.
MAX_BACKOFF_SEC = 300.0


def intervals_of(config: object) -> dict[str, float]:
    """The section intervals of a config-like object, keyed by section.

    Duck-typed on purpose: this module is imported *by* the config loader, so it must
    not import it back.
    """
    return {
        section: float(getattr(config, attr, 0.0) or 0.0)
        for section, attr in SECTION_INTERVALS.items()
    }


def commands_per_hour(intervals: Mapping[str, float]) -> float:
    """Estimated SLURM commands an hour at these intervals.

    A section whose interval is missing or non-positive is treated as never polled,
    which is what the probe does with a disabled collector.
    """
    total = 0.0
    for section, cost in COMMANDS_PER_REFRESH.items():
        interval = intervals.get(section) or 0.0
        if interval > 0:
            total += cost * 3600.0 / interval
    return total


def describe_load(intervals: Mapping[str, float]) -> str:
    """A one-line summary of the polling cost, for `doctor` and for warnings."""
    per_hour = commands_per_hour(intervals)
    return f"~{per_hour / 60:.1f} SLURM commands/min (~{per_hour:.0f}/hour)"


def load_warnings(intervals: Mapping[str, float]) -> list[str]:
    """Complaints about a polling configuration, in plain language.

    Returned rather than printed so the caller decides how loudly to say it: `doctor`
    prints them, the dashboard toasts them, and neither has to know the arithmetic.
    """
    warnings: list[str] = []
    for section, _attr in SECTION_INTERVALS.items():
        interval = intervals.get(section, 0.0)
        if 0 < interval < MIN_INTERVAL:
            warnings.append(
                f"the {section} interval is {interval:g}s, below the {MIN_INTERVAL:g}s "
                "floor; a round trip takes about that long, so this does not poll more "
                "often, it polls as fast as the network allows"
            )

    per_hour = commands_per_hour(intervals)
    if per_hour > BUSY_COMMANDS_PER_HOUR:
        warnings.append(
            f"this would issue about {per_hour:.0f} SLURM commands an hour against a "
            f"shared controller (the defaults are about {DEFAULT_COMMANDS_PER_HOUR:.0f}); "
            "raise the intervals unless you meant it"
        )
    return warnings


def retry_delay(interval: float, failures: int) -> float:
    """How long to wait before the next attempt after consecutive failures.

    Doubles with each failure up to a ceiling. A monitor whose connection has gone away
    should slow down rather than keep knocking: the retries are usually a suspended
    laptop or a dropped VPN, and neither is fixed by asking more often.
    """
    if failures <= 1:
        return interval
    return min(interval * 2 ** min(failures - 1, 8), MAX_BACKOFF_SEC)
