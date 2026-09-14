"""Rendering of the capacity, history and alert tables.

One test here guards a bug that was found the hard way: giving every column a
``min_width`` made the jobs table ~168 characters wide, so on an 80-column terminal
rich folded cells into blank filler lines and pushed ``WAIT`` off screen. Every table
must therefore fit the console it is drawn into.
"""

from __future__ import annotations

import io

import pytest
from fixture_data import snapshot as fixture_snapshot
from rich.console import Console

from lampter.models import Alert, Job, Partition, Snapshot
from lampter.render import (
    PARTITION_COLUMNS,
    PARTITION_COMPACT_COLUMNS,
    WIDE_LAYOUT,
    WIDE_LAYOUT_MIN_WIDTH,
    account_column_width,
    alert_cell_map,
    build_alert_table,
    build_history_table,
    build_partition_table,
    build_table,
    columns_for_view,
    gpus_cell,
    health_cell,
    job_cell_map,
    partition_cell_map,
)


def job(**overrides) -> Job:
    record = {"job_id": 1, "name": "train", "partition": "h100", "state": "RUNNING"}
    record.update(overrides)
    return Job.from_wire(record)


def render(table, width: int = 160) -> str:
    buffer = io.StringIO()
    console = Console(width=width, file=buffer, force_terminal=False)
    console.print(table)
    return buffer.getvalue()


def widest_line(text: str) -> int:
    return max((len(line) for line in text.splitlines()), default=0)


# ------------------------------------------------------------------ partitions


def test_partition_table_lists_gpu_partitions_first():
    out = render(build_partition_table(fixture_snapshot(), width=160), width=160)
    assert "PARTITION" in out
    assert "GPUS free" in out
    assert "h200" in out
    # GPU partitions sort ahead of CPU-only ones.
    assert out.index("h200") < out.index("cl") if "cl" in out else True


def test_partition_table_marks_my_held_gpus():
    snapshot = fixture_snapshot()
    held = snapshot.held_gpus_by_partition()
    if not held:
        pytest.skip("fixture has no running GPU jobs")
    out = render(build_partition_table(snapshot, width=200), width=200)
    assert "MINE" in out


def test_partition_table_only_mine():
    snapshot = fixture_snapshot()
    mine = snapshot.partitions_for_my_jobs()
    out = render(build_partition_table(snapshot, width=160, only_mine=True), width=160)
    for partition in mine:
        assert partition.name in out


def test_partition_table_uses_compact_columns_when_narrow():
    snapshot = fixture_snapshot()
    wide = render(build_partition_table(snapshot, width=160), width=160)
    narrow = render(build_partition_table(snapshot, width=80), width=80)
    assert "NODES idle/tot" in wide
    assert "NODES idle/tot" not in narrow


@pytest.mark.parametrize("width", [80, 124, 125, 160, 214, 215, 240])
def test_every_table_fits_the_console(width):
    """Nothing may be wider than the terminal it is rendered into.

    Widths either side of every layout threshold are covered, because a table whose
    minimum widths exceed the terminal gets clipped -- which is how the WAIT column
    was lost the first time.
    """
    snapshot = fixture_snapshot()
    for table in (
        build_table(snapshot, 0.0, width=width),
        build_partition_table(snapshot, width=width),
        build_history_table(snapshot.finished(20), 0.0, width=width),
    ):
        assert widest_line(render(table, width=width)) <= width


def test_partition_column_sets_agree_on_keys():
    """Both partition layouts must be renderable from the same cell map."""
    cells = partition_cell_map(Partition.from_wire({"name": "p", "gpus_total": 8}))
    for columns in (PARTITION_COLUMNS, PARTITION_COMPACT_COLUMNS):
        for column in columns:
            assert column.key in cells, column.key


# ------------------------------------------------------------------ cells


def test_gpus_cell_colours_by_availability():
    full = Partition.from_wire({"name": "p", "gpus_total": 8, "gpus_used": 8})
    assert "bold red" in str(gpus_cell(full).style)

    tight = Partition.from_wire({"name": "p", "gpus_total": 100, "gpus_used": 98})
    assert "yellow" in str(gpus_cell(tight).style)

    roomy = Partition.from_wire({"name": "p", "gpus_total": 100, "gpus_used": 10})
    assert "bold green" in str(gpus_cell(roomy).style)

    cpu_only = Partition.from_wire({"name": "p", "gpus_total": 0})
    assert gpus_cell(cpu_only).plain == "-"


def test_health_cell_reports_broken_nodes_and_cpu_partitions():
    broken = Partition.from_wire(
        {"name": "p", "gpus_total": 8, "nodes_total": 10, "nodes_other": 2}
    )
    assert "2 bad" in health_cell(broken).plain
    assert "bold red" in str(health_cell(broken).style)

    cpu_only = Partition.from_wire({"name": "p", "gpus_total": 0, "nodes_total": 5})
    assert health_cell(cpu_only).plain == "cpu"

    full = Partition.from_wire({"name": "p", "gpus_total": 8, "gpus_used": 8})
    assert health_cell(full).plain == "full"

    free = Partition.from_wire({"name": "p", "gpus_total": 8, "gpus_used": 1})
    assert health_cell(free).plain == "free"


# ------------------------------------------------------------------ history


def test_history_table_shows_states_exits_and_waits():
    snapshot = fixture_snapshot()
    finished = snapshot.finished(10)
    assert finished
    out = render(build_history_table(finished, snapshot.generated_at, width=160), width=160)
    assert "STATE" in out
    assert "EXIT" in out
    assert "WAIT" in out
    assert finished[0].display_id in out


def test_history_table_handles_an_empty_list():
    out = render(build_history_table([], 0.0, width=160), width=160)
    assert "JOBID" in out  # header only, no crash


# ------------------------------------------------------------------ alerts


def test_alert_table_shows_severity_and_message():
    alerts = [
        Alert(job_key="1", at=100.0, kind="failed:TIMEOUT", severity="error", message="boom"),
        Alert(job_key="2", at=50.0, kind="pending_long", severity="warning", message="starving"),
    ]
    out = render(build_alert_table(alerts, 200.0), width=120)
    assert "ERROR" in out
    assert "WARNING" in out
    assert "boom" in out
    assert "starving" in out


def test_alert_cell_map_renders_relative_time():
    alert = Alert(job_key="1", at=100.0, kind="k", severity="info", message="m")
    cells = alert_cell_map(alert, 160.0)
    assert cells["when"].plain == "1m00s ago"
    assert cells["level"].plain == "INFO"


# ------------------------------------------------------------------ account column


def test_account_column_width_is_data_driven():
    """Sized from the accounts on screen, because names are site-specific."""
    from lampter.render import ACCOUNT_MAX_WIDTH, ACCOUNT_MIN_WIDTH

    none_yet = Snapshot.from_payload(
        {"generated_at": 1, "sections": ["jobs"], "jobs": []}, fetched_at=1.0
    )
    assert account_column_width(none_yet) == ACCOUNT_MIN_WIDTH

    long_name = "a" * (ACCOUNT_MAX_WIDTH + 20)
    huge = Snapshot.from_payload(
        {
            "generated_at": 1,
            "sections": ["jobs"],
            "jobs": [{"job_id": 1, "account": long_name}],
        },
        fetched_at=1.0,
    )
    assert account_column_width(huge) == ACCOUNT_MAX_WIDTH


def test_account_names_are_never_truncated():
    """Cutting an account name can make two different accounts look identical.

    ``acct_alpha`` and ``acct_beta_advanced`` differ only
    in the middle, so neither end can be sacrificed. A hard-coded width of 27 silently
    clipped the 28-character names, which is what this guards against.
    """
    snapshot = fixture_snapshot()
    accounts = sorted({job.account for job in snapshot.jobs if job.account})
    assert accounts, "the fixture should carry accounts"

    for width in (125, 150, 180, 214, 215, 240):
        out = render(build_table(snapshot, 0.0, width=width), width=width)
        for account in accounts:
            assert account in out, f"{account} truncated at width {width}"


def test_account_column_is_shown_whenever_there_is_room():
    """Three tiers: wide, compact-with-account, and compact without it.

    ACCOUNT is dropped rather than truncated below
    :data:`ACCOUNT_COMPACT_MIN_WIDTH`, because a truncated account name can be
    ambiguous while a missing one is merely inconvenient.
    """
    from lampter.render import ACCOUNT_COMPACT_MIN_WIDTH

    assert "ACCOUNT" in [c.header for c in columns_for_view("jobs", 200)]
    assert "ACCOUNT" in [c.header for c in columns_for_view("jobs", 140)]
    assert "ACCOUNT" in [
        c.header for c in columns_for_view("jobs", ACCOUNT_COMPACT_MIN_WIDTH)
    ]


def test_the_wide_layout_is_comfortable_not_merely_fitting():
    """The threshold is a quality bar, not the point where nothing is clipped.

    At 200 columns the flexible columns were squeezed to ~11 characters, which cut job
    names to `long_job_...`. A table that technically fits but mangles the names is
    worse than the compact layout, so the threshold sits above that.
    """
    snapshot = fixture_snapshot()
    width = WIDE_LAYOUT_MIN_WIDTH
    assert [c.header for c in columns_for_view("jobs", width)] == [
        c.header for c in WIDE_LAYOUT
    ]

    out = render(build_table(snapshot, 0.0, width=width), width=width)
    # Every job name and account name survives whole, and so does the memory figure.
    for name in {job.name for job in snapshot.jobs if job.name}:
        assert name in out, f"name {name!r} truncated at the wide threshold"
    for account in {job.account for job in snapshot.jobs if job.account}:
        assert account in out, f"account {account!r} truncated at the wide threshold"
    for usage in snapshot.usage.values():
        if usage.max_rss_gb:
            assert f"{usage.max_rss_gb:.1f}G" in out


def test_account_column_is_omitted_rather_than_truncated_when_narrow():
    """At 80 columns there is not room for both ACCOUNT and WAIT; WAIT wins."""
    from lampter.render import ACCOUNT_COMPACT_MIN_WIDTH

    just_below = ACCOUNT_COMPACT_MIN_WIDTH - 1
    headers = [c.header for c in columns_for_view("jobs", just_below)]
    assert "ACCOUNT" not in headers
    assert "WAIT" in headers

    out = render(build_table(fixture_snapshot(), 0.0, width=just_below), width=just_below)
    assert "WAIT" in out
    assert "ACCOUNT" not in out


def test_account_cell_falls_back_to_a_dash():
    assert job_cell_map(job(), 0.0)["account"].plain == "-"
    assert job_cell_map(job(account="acct"), 0.0)["account"].plain == "acct"
