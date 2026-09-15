"""Shared presentation layer.

The CLI's one-shot table and the interactive TUI must agree on *what* a job row
contains, otherwise the two drift the moment a column changes. So the column
definitions and the per-job cell rendering live here, and both front ends consume
them.

Layouts
-------
A twelve-column table of Slurm metadata does not fit a normal terminal, and asking
``rich`` to squeeze it collapses the flexible columns to one character each (every
column also costs 2 for padding plus a border). So there are two explicit layouts,
selected by terminal width via :func:`layout_for_width`:

* :data:`WIDE_LAYOUT` -- everything, for a genuinely wide window.
* :data:`COMPACT_LAYOUT` -- the columns that answer "what is it doing, and how long
  has it waited". ``WAIT`` is deliberately kept, since it is the number this tool
  exists to show.

A layout *is* a tuple of :class:`Column`, so the widths can differ per layout and
the cell order always follows the columns.
"""

from __future__ import annotations

from typing import NamedTuple

from rich.table import Table
from rich.text import Text

from .duration import format_epoch, format_time_limit, humanize_seconds, progress_bar
from .gpu_policy import (
    VERDICT_CANCELLED,
    VERDICT_WARNING,
    GpuPolicy,
    policy_for_prefixes,
)
from .models import (
    AccountUsage,
    Alert,
    HistoryJob,
    Job,
    Partition,
    QosPressure,
    Snapshot,
    Usage,
    sort_jobs,
)

#: Colour per Slurm state, so a glance at STATE separates "working", "waiting" and
#: "went wrong" without reading the words.
STATE_STYLES: dict[str, str] = {
    "RUNNING": "bold green",
    "COMPLETING": "green",
    "CONFIGURING": "green",
    "RESIZING": "green",
    "STAGE_OUT": "green",
    "PENDING": "yellow",
    "SUSPENDED": "magenta",
    "STOPPED": "magenta",
    "COMPLETED": "dim green",
    "CANCELLED": "red",
    "PREEMPTED": "red",
    "FAILED": "bold red",
    "TIMEOUT": "bold red",
    "NODE_FAIL": "bold red",
    "OUT_OF_MEMORY": "bold red",
    "BOOT_FAIL": "bold red",
    "DEADLINE": "bold red",
}

#: Wait-phase colouring: an eligible-but-starved job is a different problem from
#: one blocked by its own dependency chain.
PHASE_STYLES: dict[str, str] = {
    "competing": "yellow",
    "blocked": "dim",
    "scheduled": "cyan",
    "-": "dim",
}

WAIT_STYLES: dict[str, str] = {
    "competing": "bold yellow",
    "blocked": "dim",
    "scheduled": "cyan",
}

#: Flexible columns will not be squeezed below this, so a cramped terminal clips
#: the table rather than rendering unreadable one-character columns.
MIN_FLEX_WIDTH = 8

#: Bounds for the ACCOUNT column, which is sized from the data rather than fixed.
#:
#: Account names are site-specific and can be long. They must NOT be truncated to fit:
#: `acct_alpha` and `acct_beta_advanced` differ only in the
#: middle, so cutting either end can make two different accounts render identically --
#: which is worse than showing no account at all.
ACCOUNT_MIN_WIDTH = 12
ACCOUNT_MAX_WIDTH = 44


def account_column_width(snapshot: Snapshot) -> int:
    """How wide ACCOUNT has to be for the accounts actually on screen."""
    longest = max((len(job.account) for job in snapshot.jobs if job.account), default=0)
    return max(ACCOUNT_MIN_WIDTH, min(longest, ACCOUNT_MAX_WIDTH))


#: Below this width the wide layout is not worth having. Measured rather than
#: guessed: at 200 columns the four flexible columns were squeezed to ~11 characters,
#: which truncated job names to `long_job_...` and memory to `128.0G...`. The
#: threshold is where every one of them gets at least ~13 characters, which
#: tests/test_render.py asserts.
WIDE_LAYOUT_MIN_WIDTH = 215


class Column(NamedTuple):
    """One table column: a stable ``key``, a header and sizing hints."""

    key: str
    header: str
    width: int
    #: Flexible columns absorb leftover width; fixed ones keep ``width``.
    flex: bool = False


WIDE_LAYOUT: tuple[Column, ...] = (
    Column("id", "JOBID", 18),
    Column("name", "NAME", 20, flex=True),
    Column("state", "STATE", 10),
    # Fixed, not flexible: an account name is what tells two otherwise identical
    # jobs apart, and any truncation is unsafe here -- `acct_alpha`
    # and `acct_beta_advanced` differ in the middle, so cutting either end
    # makes them the same string.
    Column("account", "ACCOUNT", 27),
    Column("phase", "PHASE", 9),
    Column("partition", "PARTITION", 22, flex=True),
    Column("gpu", "GPU UTIL", 9),
    Column("mem", "MEM", 8),
    Column("run", "RUN", 14),
    Column("wait", "WAIT", 10),
    Column("limit", "LIMIT", 9),
    Column("left", "LEFT", 9),
    Column("nodes", "NODES", 10, flex=True),
    Column("reason", "REASON", 20, flex=True),
)

#: The narrow-terminal view. Fixed widths are deliberately tight: at 80 columns the
#: six fixed columns plus padding and borders land just under the limit.
COMPACT_LAYOUT: tuple[Column, ...] = (
    Column("id", "JOBID", 18),
    Column("name", "NAME", 14, flex=True),
    Column("state", "STATE", 9),
    Column("gpu", "GPU UTIL", 9),
    Column("wait", "WAIT", 8),
    Column("reason", "REASON", 16, flex=True),
)

#: The narrow layout plus ACCOUNT. Kept as a separate tuple so
#: :func:`columns_for_view` can return one of three stable objects and callers can
#: still compare layouts by identity.
COMPACT_LAYOUT_WITH_ACCOUNT: tuple[Column, ...] = (
    Column("id", "JOBID", 18),
    Column("name", "NAME", 14, flex=True),
    Column("state", "STATE", 9),
    # The nominal width; build_table overrides it with the width the accounts on
    # screen actually need.
    Column("account", "ACCOUNT", 27),
    # PARTITION is as useful as ACCOUNT and cheap, so the middle tier carries it.
    Column("partition", "PARTITION", 13),
    Column("gpu", "GPU UTIL", 9),
    Column("wait", "WAIT", 8),
    Column("reason", "REASON", 16, flex=True),
)

#: Below this width the compact layout cannot also carry an account name.
#:
#: The arithmetic is unforgiving: an 18-character job id plus a 28-character account
#: is 46 of the ~58 usable characters at 80 columns, leaving nothing for NAME and
#: REASON and squeezing WAIT out entirely. Dropping the column is the honest choice --
#: truncating it would make `acct_alpha` and
#: `acct_beta_advanced` render identically.
# 125 rather than 105 because the middle tier now carries PARTITION as well.
ACCOUNT_COMPACT_MIN_WIDTH = 125


def layout_for_width(width: int | None) -> tuple[Column, ...]:
    """Choose a layout for a terminal of ``width`` columns."""
    if width is None:
        return WIDE_LAYOUT
    return WIDE_LAYOUT if width >= WIDE_LAYOUT_MIN_WIDTH else COMPACT_LAYOUT


# ------------------------------------------------------------------ cells


def state_cell(job: Job) -> Text:
    return Text(job.state, style=STATE_STYLES.get(job.state, "white"))


def wait_cell(job: Job, now: float) -> Text:
    """The headline number: how long this job has waited, or waited already.

    Identical semantics for pending and running jobs (:meth:`Job.queue_wait_sec`),
    so the column is directly comparable all the way down the table.
    """
    text = humanize_seconds(job.queue_wait_sec(now))
    style = WAIT_STYLES.get(job.wait_phase(now))
    return Text(text, style=style) if style else Text(text)


def run_cell(job: Job, now: float, with_bar: bool = True) -> Text:
    """Elapsed run time, optionally with a bar showing limit consumption."""
    elapsed = job.run_time_sec(now)
    if elapsed is None:
        return Text("-", style="dim")
    label = humanize_seconds(elapsed)
    if with_bar:
        fraction = job.used_fraction(now)
        if fraction is not None:
            label = f"{label} {progress_bar(fraction, 5)}"
    return Text(label, style="green")


def left_cell(job: Job, now: float) -> Text:
    """Time left before the wall-clock limit kills the job."""
    if not job.is_running:
        return Text("-", style="dim")
    remaining = job.time_left_sec(now)
    if remaining is None:
        return Text("-", style="dim")
    if remaining <= 0:
        return Text("overdue", style="bold red")
    label = humanize_seconds(remaining)
    fraction = job.used_fraction(now)
    # Warn as the limit approaches: this is how jobs die unexpectedly.
    if fraction is not None and fraction >= 0.9:
        return Text(f"{label} !", style="bold red")
    if fraction is not None and fraction >= 0.75:
        return Text(label, style="yellow")
    return Text(label)


def gpu_utilisation(job: Job, usage: Usage | None) -> tuple[float | None, GpuPolicy]:
    """Per-GPU utilisation and the idle-GPU thresholds it should be judged against.

    Returns ``(None, policy)`` whenever there is nothing trustworthy to show, which
    covers three different situations the caller must not conflate:

    * the job asked for no GPU at all;
    * it has one but no measurement was collected;
    * the measurement could not be interpreted (see :meth:`Usage.per_gpu_util`).

    A job that has not started has no nodes, so its policy is the catch-all default
    rather than anything node-specific.
    """
    policy = policy_for_prefixes(job.node_prefixes)
    if usage is None or usage.per_gpu_util(job.gpu_count) is None:
        return None, policy
    return usage.per_gpu_util(job.gpu_count), policy


def gpu_cell(job: Job, usage: Usage | None) -> Text:
    """GPU count and mean utilisation, coloured against the site's idle-GPU policy.

    This is the column that says "you are about to be emailed about this job": Torch
    cancels a job whose GPUs average below a per-node-family threshold. It is an
    indicator rather than a verdict, because the site judges *each GPU separately*
    while Slurm pools the job's GPUs -- see the README.
    """
    count = job.gpu_count
    if not count:
        return Text("-", style="dim")

    per_gpu, policy = gpu_utilisation(job, usage)
    if per_gpu is None:
        if usage is not None and usage.gpu_util is not None:
            # A measurement exists but dividing it by the GPU count gave something
            # impossible, so the pooled-vs-per-GPU assumption does not hold here.
            return Text(f"{count} ?", style="bold yellow")
        return Text(str(count))

    verdict = policy.verdict(per_gpu)
    style = {
        VERDICT_CANCELLED: "bold red",
        VERDICT_WARNING: "yellow",
    }.get(verdict, "green")

    cell = Text(f"{count} ")
    # `~` marks a figure pooled across several GPUs: honest that it is a mean, and the
    # reader needs to know a single idle GPU is hidden inside it.
    cell.append("~" if count > 1 else "")
    cell.append(f"{per_gpu:.0f}%", style=style)
    return cell


def format_memory_kb(kb: int | None) -> str:
    """Compact memory from a KB figure: ``4.3G``, ``640M``, ``512K``."""
    if kb is None:
        return "-"
    if kb >= 1024 * 1024:
        return f"{kb / (1024 * 1024):.1f}G"
    if kb >= 1024:
        return f"{kb / 1024:.0f}M"
    return f"{kb}K"


def memory_cell(job: Job, usage: Usage | None) -> Text:
    """Peak memory of a running job, coloured against the memory it asked for.

    The raw RSS number is only actionable next to the request: 40 GB is alarming for
    a job that asked for 48 GB and unremarkable for one that asked for 400 GB. Slurm
    kills a job that exceeds its request, so approaching it is worth a warning.
    """
    if usage is None or usage.max_rss_kb is None:
        return Text("-", style="dim")

    label = format_memory_kb(usage.max_rss_kb)
    limit_mb = job.memory_limit_mb
    if not limit_mb:
        return Text(label, style="cyan")

    used_mb = usage.max_rss_kb / 1024
    fraction = used_mb / limit_mb
    if fraction >= 0.9:
        return Text(f"{label} !", style="bold red")
    if fraction >= 0.75:
        return Text(label, style="yellow")
    return Text(label, style="cyan")


def job_cell_map(
    job: Job,
    now: float,
    *,
    wide: bool = True,
    usage: Usage | None = None,
) -> dict[str, Text]:
    """Render one job as cells keyed by :class:`Column` key.

    ``wide`` selects the wide-layout extras: the limit bar, the glossed reason, and the
    live memory figure. ``usage`` is always passed in, whatever the layout, because GPU
    utilisation must not disappear in the narrow view -- withholding ``usage`` here once
    made the whole column silently blank for anyone below the wide threshold.
    """
    phase = job.wait_phase(now)
    nodes = job.nodelist or (f"{job.node_count}n" if job.node_count else "-")
    reason = (job.reason_text if wide else job.reason) or "-"
    return {
        "id": Text(job.display_id),
        "name": Text(job.name),
        "state": state_cell(job),
        "account": Text(job.account or "-", style="dim" if not job.account else ""),
        "phase": Text(phase, style=PHASE_STYLES.get(phase, "")),
        "partition": Text(job.partition),
        "gpu": gpu_cell(job, usage),
        "mem": memory_cell(job, usage) if wide else Text("-", style="dim"),
        "run": run_cell(job, now, with_bar=wide),
        "wait": wait_cell(job, now),
        "limit": Text(format_time_limit(job.time_limit_min, job.time_limit_infinite)),
        "left": left_cell(job, now),
        "nodes": Text(nodes, style="cyan"),
        "reason": Text(reason, style="dim" if job.is_running else ""),
    }


def job_cells(
    job: Job,
    now: float,
    layout: tuple[Column, ...] = WIDE_LAYOUT,
    usage: Usage | None = None,
) -> list[Text]:
    """Render one job as table cells, in ``layout`` order.

    The wide layout has room for the limit-consumption bar, the plain-English
    reason and the live memory figure; the narrow one does not.
    """
    wide = layout is WIDE_LAYOUT
    # `usage` goes in either way: MEM is a wide-only luxury, but GPU utilisation is not.
    cells = job_cell_map(job, now, wide=wide, usage=usage)
    return [cells[column.key] for column in layout if column.key in cells]


# ------------------------------------------------------------------ assembly


def visible_jobs(
    snapshot: Snapshot,
    now: float,
    sort_mode: str = "wait",
    reverse: bool = False,
    max_rows: int = 0,
) -> list[Job]:
    """Jobs in display order, optionally truncated."""
    jobs = sort_jobs(list(snapshot.jobs), sort_mode, now, reverse)
    if max_rows and max_rows > 0:
        return jobs[:max_rows]
    return jobs


def build_table(
    snapshot: Snapshot,
    now: float,
    *,
    sort_mode: str = "wait",
    reverse: bool = False,
    max_rows: int = 0,
    title: str | None = None,
    layout: tuple[Column, ...] | None = None,
    width: int | None = None,
) -> Table:
    """A static ``rich`` table, used by the one-shot ``status`` command.

    Goes through :func:`columns_for_view` rather than :func:`layout_for_width` so the
    CLI and the dashboard always choose the same column set -- they diverged once, and
    the CLI silently rendered a layout without ACCOUNT.
    """
    resolved = layout if layout is not None else columns_for_view("jobs", width)
    rows = [
        job_cells(job, now, resolved, snapshot.usage_for(job))
        for job in visible_jobs(snapshot, now, sort_mode, reverse, max_rows)
    ]
    return _table_for(
        resolved,
        rows,
        title=title,
        widths={"account": account_column_width(snapshot)},
    )


def _table_for(
    columns: tuple[Column, ...],
    rows: list[list[Text]],
    *,
    title: str | None = None,
    widths: dict[str, int] | None = None,
) -> Table:
    """Build a table for any of the column sets.

    Every view goes through here so the flexible-column handling stays in one place:
    without ``min_width`` a cramped terminal squeezes the flexible columns down to a
    single character each, which is how the WAIT column got lost the first time.
    """
    table = Table(
        title=title,
        header_style="bold",
        title_style="bold",
        title_justify="left",
        expand=True,
        pad_edge=False,
    )
    for column in columns:
        if column.flex:
            table.add_column(
                column.header,
                ratio=column.width,
                min_width=MIN_FLEX_WIDTH,
                no_wrap=True,
                overflow="ellipsis",
            )
        else:
            # A width override lets a column be sized from the data; only fixed
            # columns take one, since a flexible column's number is a ratio.
            table.add_column(
                column.header,
                width=(widths or {}).get(column.key, column.width),
                no_wrap=True,
                overflow="ellipsis",
            )
    for row in rows:
        table.add_row(*row)
    return table


# ------------------------------------------------------------------ partitions

PARTITION_COLUMNS: tuple[Column, ...] = (
    Column("name", "PARTITION", 22, flex=True),
    Column("gpus", "GPUS free", 16),
    Column("held", "MINE", 5),
    Column("nodes", "NODES idle/tot", 15),
    Column("cpus", "CPUS idle/tot", 15),
    Column("mem", "MEM free", 10),
    Column("gputype", "TYPE", 7),
    Column("health", "STATE", 9),
)

PARTITION_COMPACT_COLUMNS: tuple[Column, ...] = (
    Column("name", "PARTITION", 20, flex=True),
    Column("gpus", "GPUS free", 16),
    Column("held", "MINE", 5),
    Column("health", "STATE", 9),
)

PARTITION_WIDE_MIN_WIDTH = 130


def gpus_cell(partition: Partition) -> Text:
    """Free/total GPUs, coloured by how much room is actually left."""
    if not partition.has_gpus:
        return Text("-", style="dim")
    label = f"{partition.gpus_free}/{partition.gpus_total}"
    if partition.gpus_free == 0:
        return Text(label, style="bold red")
    # Treat the last few percent as tight: those will be gone by the time a job
    # gets through the queue anyway.
    if partition.gpus_free <= max(1, partition.gpus_total // 20):
        return Text(label, style="yellow")
    return Text(label, style="bold green")


def health_cell(partition: Partition) -> Text:
    if not partition.is_healthy:
        return Text(f"{partition.nodes_other} bad", style="bold red")
    if not partition.has_gpus:
        return Text("cpu", style="dim")
    if partition.gpus_free == 0:
        return Text("full", style="yellow")
    return Text("free", style="green")


def partition_cell_map(partition: Partition, held_gpus: int = 0) -> dict[str, Text]:
    """Cells for one capacity row.

    ``held_gpus`` is how many GPUs my own running jobs occupy here. It sits next to
    the free count because that is what makes a ``QOSMaxGRESPerUser`` wait legible:
    the constraint is my own quota, not cluster congestion.
    """
    memory = f"{partition.mem_free_gb}G" if partition.mem_total_mb else "-"
    return {
        "name": Text(partition.name),
        "gpus": gpus_cell(partition),
        "held": Text(
            str(held_gpus) if held_gpus else "-",
            style="bold cyan" if held_gpus else "dim",
        ),
        "nodes": Text(f"{partition.nodes_idle}/{partition.nodes_total}"),
        "cpus": Text(f"{partition.cpus_free}/{partition.cpus_total}"),
        "mem": Text(memory),
        "gputype": Text(partition.gpu_type_label),
        "health": health_cell(partition),
    }


def build_partition_table(
    snapshot: Snapshot,
    *,
    width: int | None = None,
    only_mine: bool = False,
    title: str | None = None,
) -> Table:
    """Capacity per partition, busiest GPU partitions first."""
    columns = (
        PARTITION_COLUMNS
        if width is None or width >= PARTITION_WIDE_MIN_WIDTH
        else PARTITION_COMPACT_COLUMNS
    )
    held = snapshot.held_gpus_by_partition()
    partitions = ordered_partitions(
        snapshot.partitions_for_my_jobs() if only_mine else list(snapshot.partitions)
    )
    rows = [
        [partition_cell_map(p, held.get(p.name, 0))[c.key] for c in columns]
        for p in partitions
    ]
    return _table_for(columns, rows, title=title)


# ------------------------------------------------------------------ history

HISTORY_COLUMNS: tuple[Column, ...] = (
    Column("id", "JOBID", 18),
    Column("name", "NAME", 18, flex=True),
    Column("state", "STATE", 11),
    Column("exit", "EXIT", 6),
    Column("elapsed", "RUN", 9),
    Column("wait", "WAIT", 9),
    Column("ended", "ENDED", 13),
    Column("partition", "PARTITION", 18, flex=True),
)

HISTORY_COMPACT_COLUMNS: tuple[Column, ...] = (
    Column("id", "JOBID", 18),
    Column("state", "STATE", 11),
    Column("exit", "EXIT", 6),
    Column("wait", "WAIT", 9),
    Column("partition", "PARTITION", 16, flex=True),
)

HISTORY_WIDE_MIN_WIDTH = 130


def history_cell_map(record: HistoryJob, now: float) -> dict[str, Text]:
    return {
        "id": Text(record.display_id),
        "name": Text(record.name),
        "state": Text(record.state, style=STATE_STYLES.get(record.state, "white")),
        "exit": Text(
            record.exit_label,
            style="bold red" if record.is_failed else "dim",
        ),
        "elapsed": Text(humanize_seconds(record.elapsed_sec)),
        "wait": Text(humanize_seconds(record.queue_wait_sec(now))),
        "ended": Text(format_epoch(record.end_time, "%m-%d %H:%M")),
        "partition": Text(record.partition),
    }


def build_history_table(
    records: list[HistoryJob] | tuple[HistoryJob, ...],
    now: float,
    *,
    width: int | None = None,
    title: str | None = None,
) -> Table:
    columns = (
        HISTORY_COLUMNS
        if width is None or width >= HISTORY_WIDE_MIN_WIDTH
        else HISTORY_COMPACT_COLUMNS
    )
    rows = [
        [history_cell_map(record, now)[c.key] for c in columns] for record in records
    ]
    return _table_for(columns, rows, title=title)


# ------------------------------------------------------------------ alerts

SEVERITY_STYLES: dict[str, str] = {
    "error": "bold red",
    "warning": "yellow",
    "info": "dim green",
}

ALERT_COLUMNS: tuple[Column, ...] = (
    Column("when", "WHEN", 10),
    Column("level", "LEVEL", 8),
    Column("message", "ALERT", 70, flex=True),
)


def alert_cell_map(alert: Alert, now: float) -> dict[str, Text]:
    return {
        "when": Text(humanize_seconds(max(0.0, now - alert.at)) + " ago"),
        "level": Text(
            alert.severity.upper(),
            style=SEVERITY_STYLES.get(alert.severity, ""),
        ),
        "message": Text(alert.message),
    }


def build_alert_table(
    alerts: list[Alert] | tuple[Alert, ...],
    now: float,
    *,
    title: str | None = None,
) -> Table:
    rows = [[alert_cell_map(a, now)[c.key] for c in ALERT_COLUMNS] for a in alerts]
    return _table_for(ALERT_COLUMNS, rows, title=title)


def truncate_text(text: Text, width: int) -> Text:
    """Clip a ``Text`` to ``width`` cells, marking the cut with an ellipsis.

    Textual can ellipsize a widget itself, but doing it here keeps the dashboard and the
    one-shot commands consistent and makes the behaviour testable without a terminal --
    and it preserves the styles of whatever survives.
    """
    if width <= 0:
        return Text()
    if text.cell_len <= width:
        return text
    clipped = text.copy()
    clipped.truncate(width, overflow="ellipsis")
    return clipped


def fit_line(pieces: list[tuple[int, Text]], width: int | None) -> Text:
    """Join priority-tagged pieces, dropping the least important until they fit.

    A status bar that wraps pushes the table down and makes the whole screen jump; one
    that clips mid-word ("longest wait 16h…") is barely better. So the bar is assembled
    from pieces, each tagged with how readily it can go, and the least important are
    dropped -- at a boundary a reader can understand -- before anything is truncated.

    Pieces are given in display order and each carries its own leading separator, so
    dropping one from the middle can never leave a dangling `│`. The first piece is never
    dropped: it is where the important thing is.
    """
    kept = list(pieces)
    while width is not None and len(kept) > 1 and _cells(kept) > width:
        least_important = min(range(len(kept)), key=lambda index: kept[index][0])
        del kept[least_important]
    line = Text()
    for _priority, piece in kept:
        line.append_text(piece)
    return truncate_text(line, width) if width is not None else line


def _cells(pieces: list[tuple[int, Text]]) -> int:
    return sum(piece.cell_len for _priority, piece in pieces)


#: How readily each part of the summary survives a narrow terminal: higher is kept
#: longer, and the least important are dropped first. The host and the job counts are
#: what the bar exists to say; the rest is detail a narrow window cannot afford, and
#: saying less is better than moving the table under the reader's cursor.
#:
#: A piece that only makes sense after another must sit *below* it, since dropping
#: always starts at the bottom: `sort` can vanish without taking `view` with it, but
#: never the other way round.
SUMMARY_PRIORITIES: dict[str, int] = {
    "target": 100,
    "counts": 95,
    "longest_wait": 85,
    "updated": 75,
    "alerts": 70,
    "view": 68,
    "auto": 55,
    "sort": 50,
    "ages": 40,
    "slurm": 30,
}


def summary_pieces(
    snapshot: Snapshot,
    now: float,
    *,
    target: str,
    interval: float,
    auto: bool,
    next_refresh_in: float | None = None,
    show_section_ages: bool = False,
) -> list[tuple[int, Text]]:
    """The summary bar's segments, each tagged with how readily it can be dropped.

    Returned rather than joined so a caller with extra segments of its own -- the
    dashboard adds which view is on screen -- can put them through the same budget
    instead of gluing them on afterwards and watching them get cut in half.

    ``show_section_ages`` adds how stale the slower collectors are, which matters in the
    dashboard -- where a jobs-only refresh is normal -- and is noise in a one-shot
    command, where everything was just fetched together.
    """
    counts = snapshot.counts()
    pieces: list[tuple[int, Text]] = []

    text = Text()
    text.append(target, style="bold cyan")
    pieces.append((SUMMARY_PRIORITIES["target"], text))

    if snapshot.slurm_version:
        pieces.append(
            (
                SUMMARY_PRIORITIES["slurm"],
                Text.assemble((" · slurm ", "dim"), (snapshot.slurm_version, "dim")),
            )
        )

    text = Text("  │  ")
    text.append(f"{counts['running']} running", style="bold green")
    text.append(" · ")
    text.append(f"{counts['pending']} pending", style="bold yellow")
    if counts["other"]:
        text.append(" · ")
        text.append(f"{counts['other']} other", style="dim")
    pieces.append((SUMMARY_PRIORITIES["counts"], text))

    longest = snapshot.longest_wait(now)
    if longest is not None:
        pieces.append(
            (
                SUMMARY_PRIORITIES["longest_wait"],
                Text.assemble(
                    ("  │  longest wait ", ""), (humanize_seconds(longest), "bold yellow")
                ),
            )
        )

    text = Text("  │  updated ")
    text.append(f"{humanize_seconds(snapshot.age_sec(now))} ago")
    if snapshot.probe_elapsed_sec is not None:
        text.append(f" (probe {snapshot.probe_elapsed_sec:.2f}s)", style="dim")
    pieces.append((SUMMARY_PRIORITIES["updated"], text))

    if auto:
        when = "-" if next_refresh_in is None else humanize_seconds(max(0, next_refresh_in))
        pieces.append(
            (
                SUMMARY_PRIORITIES["auto"],
                Text(f"  │  auto {humanize_seconds(interval)} · next {when}", style="dim"),
            )
        )
    else:
        pieces.append(
            (SUMMARY_PRIORITIES["auto"], Text("  │  auto-refresh off", style="dim"))
        )

    if show_section_ages:
        ages = Text()
        partitions_age = snapshot.partitions_age_sec(now)
        if partitions_age is not None:
            ages.append(f"  │  capacity {humanize_seconds(partitions_age)}", style="dim")
        history_age = snapshot.history_age_sec(now)
        if history_age is not None:
            ages.append(f" · history {humanize_seconds(history_age)}", style="dim")
        usage_age = snapshot.usage_age_sec(now)
        if usage_age is not None:
            ages.append(f" · usage {humanize_seconds(usage_age)}", style="dim")
        accounts_age = snapshot.accounts_age_sec(now)
        if accounts_age is not None:
            ages.append(f" · accts {humanize_seconds(accounts_age)}", style="dim")
        if ages.cell_len:
            pieces.append((SUMMARY_PRIORITIES["ages"], ages))

    return pieces


def summary_line(
    snapshot: Snapshot,
    now: float,
    *,
    target: str,
    interval: float,
    auto: bool,
    next_refresh_in: float | None = None,
    show_section_ages: bool = False,
    width: int | None = None,
) -> Text:
    """The one-line summary shown above the table in both front ends.

    ``width``, when given, is a budget in cells: the least important parts are dropped
    until the line fits rather than being allowed to wrap.
    """
    return fit_line(
        summary_pieces(
            snapshot,
            now,
            target=target,
            interval=interval,
            auto=auto,
            next_refresh_in=next_refresh_in,
            show_section_ages=show_section_ages,
        ),
        width,
    )


# ------------------------------------------------------------------ views

#: The three things the dashboard can show. Keys 1/2/3 (and Tab) switch between
#: them, and the CLI's subcommands map onto the same three cell maps, so a column
#: can never mean one thing in the table and something else in the dashboard.
VIEWS: tuple[str, ...] = ("jobs", "partitions", "accounts", "qos", "history")

#: Short labels for the per-view digit keys in the footer.
VIEW_KEY_LABELS: dict[str, str] = {
    "jobs": "Jobs",
    "partitions": "Capacity",
    "accounts": "Accounts",
    "qos": "QOS",
    "history": "History",
}

VIEW_TITLES: dict[str, str] = {
    "jobs": "my jobs",
    "partitions": "partition capacity",
    "accounts": "account usage",
    "qos": "QOS limits",
    "history": "recent outcomes",
}

#: How many history rows the dashboard keeps, most recent first.
HISTORY_VIEW_LIMIT = 200


def columns_for_view(view: str, width: int | None) -> tuple[Column, ...]:
    """Column set for one view at one terminal width."""
    if view == "accounts":
        if width is None or width >= ACCOUNT_WIDE_MIN_WIDTH:
            return ACCOUNT_COLUMNS
        return ACCOUNT_COMPACT_COLUMNS
    if view == "qos":
        if width is None or width >= QOS_WIDE_MIN_WIDTH:
            return QOS_COLUMNS
        return QOS_COMPACT_COLUMNS
    if view == "partitions":
        if width is None or width >= PARTITION_WIDE_MIN_WIDTH:
            return PARTITION_COLUMNS
        return PARTITION_COMPACT_COLUMNS
    if view == "history":
        if width is None or width >= HISTORY_WIDE_MIN_WIDTH:
            return HISTORY_COLUMNS
        return HISTORY_COMPACT_COLUMNS
    # Jobs: wide, then compact with ACCOUNT, then compact without it when there is
    # simply not enough room for a full account name.
    if width is None or width >= WIDE_LAYOUT_MIN_WIDTH:
        return WIDE_LAYOUT
    if width >= ACCOUNT_COMPACT_MIN_WIDTH:
        return COMPACT_LAYOUT_WITH_ACCOUNT
    return COMPACT_LAYOUT


def ordered_partitions(partitions) -> list[Partition]:
    """GPU-bearing partitions first, then the largest, then alphabetical.

    That is the order someone hunting for somewhere to submit would scan in.
    """
    return sorted(partitions, key=lambda p: (not p.has_gpus, -p.gpus_total, p.name))


def view_rows(
    snapshot: Snapshot,
    now: float,
    view: str,
    columns: tuple[Column, ...],
    *,
    sort_mode: str = "wait",
    reverse: bool = False,
    max_rows: int = 0,
    only_mine: bool = False,
) -> tuple[list[list[Text]], list[str]]:
    """Cells and stable row keys for one view, in display order.

    Returning the keys alongside the cells is what lets the dashboard keep the
    cursor on the same row across refreshes: rows move as jobs start and finish.
    """
    if view == "partitions":
        held = snapshot.held_gpus_by_partition()
        chosen = (
            snapshot.partitions_for_my_jobs() if only_mine else list(snapshot.partitions)
        )
        ordered = ordered_partitions(chosen)
        rows = [
            [partition_cell_map(p, held.get(p.name, 0))[c.key] for c in columns]
            for p in ordered
        ]
        return rows, [p.name for p in ordered]

    if view == "accounts":
        rows = [
            [account_cell_map(a)[c.key] for c in columns] for a in snapshot.accounts
        ]
        return rows, [a.account for a in snapshot.accounts]

    if view == "qos":
        rows = [
            [qos_cell_map(q)[c.key] for c in columns] for q in snapshot.qos_pressure
        ]
        return rows, [q.name for q in snapshot.qos_pressure]

    if view == "history":
        records = snapshot.finished(HISTORY_VIEW_LIMIT)
        rows = [
            [history_cell_map(record, now)[c.key] for c in columns] for record in records
        ]
        return rows, [record.display_id for record in records]

    jobs = visible_jobs(snapshot, now, sort_mode, reverse, max_rows)
    return (
        [job_cells(job, now, columns, snapshot.usage_for(job)) for job in jobs],
        [job.display_id for job in jobs],
    )


# ------------------------------------------------------------------ accounts

ACCOUNT_COLUMNS: tuple[Column, ...] = (
    Column("account", "ACCOUNT", 30, flex=True),
    Column("qos", "QOS", 9),
    Column("gpus", "GPUS now", 8),
    Column("cpus", "CPUS now", 8),
    Column("nodes", "NODES", 6),
    Column("users", "USERS", 6),
    Column("gpu_hours", "GPU-h", 9),
    Column("cpu_hours", "CPU-h", 9),
    Column("jobs", "JOBS", 7),
    Column("fairshare", "FAIR", 6),
    Column("effectv", "EFF", 6),
    Column("note", "NOTE", 30, flex=True),
)

ACCOUNT_COMPACT_COLUMNS: tuple[Column, ...] = (
    # Fixed rather than flexible: at 80 columns a flexible ACCOUNT column gets
    # squeezed to an ellipsis, and the account name is the one thing you must read.
    Column("account", "ACCOUNT", 28),
    Column("gpus", "GPUS", 5),
    Column("gpu_hours", "GPU-h", 9),
    Column("fairshare", "FAIR", 6),
    Column("note", "NOTE", 26, flex=True),
)

ACCOUNT_WIDE_MIN_WIDTH = 150


def _number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def account_cell_map(account: AccountUsage, window_days: int | None = None) -> dict[str, Text]:
    """Cells for one account row.

    ``GPU-h`` / ``CPU-h`` are the reporting numbers -- what a project actually spent
    over the window -- while ``FAIR`` and ``EFF`` explain the *queue*: Slurm lowers an
    association's priority as its effective usage outgrows its share.
    """
    days = window_days or account.window_days
    gpus = Text(str(account.running_gpus), style="bold green" if account.running_gpus else "dim")
    note = account.priority_note
    note_style = ""
    if "over its share" in note:
        note_style = "yellow"
    elif "under its share" in note:
        note_style = "green"
    return {
        "account": Text(account.account),
        "qos": Text(account.qos_label),
        "gpus": gpus,
        "cpus": Text(str(account.running_cpus)),
        "nodes": Text(str(account.running_nodes)),
        "users": Text(
            str(account.running_users),
            # More than one user means the numbers are not only mine.
            style="yellow" if account.is_shared else "dim",
        ),
        "gpu_hours": Text(f"{account.window_gpu_hours:,.1f}"),
        "cpu_hours": Text(f"{account.window_cpu_hours:,.1f}"),
        "jobs": Text(f"{account.window_jobs:,}"),
        "fairshare": Text(_number(account.fairshare)),
        "effectv": Text(_number(account.effectv_usage)),
        "note": Text(note or f"last {days}d", style=note_style or "dim"),
    }


def build_account_table(
    snapshot: Snapshot,
    *,
    width: int | None = None,
    title: str | None = None,
) -> Table:
    columns = (
        ACCOUNT_COLUMNS
        if width is None or width >= ACCOUNT_WIDE_MIN_WIDTH
        else ACCOUNT_COMPACT_COLUMNS
    )
    rows = [
        [account_cell_map(a, snapshot.account_days)[c.key] for c in columns]
        for a in snapshot.accounts
    ]
    return _table_for(columns, rows, title=title)


# ------------------------------------------------------------------ qos

QOS_COLUMNS: tuple[Column, ...] = (
    Column("name", "QOS", 14, flex=True),
    Column("wall", "MAXWALL", 10),
    Column("my_gpus", "MY GPU", 7),
    Column("user_gpus", "PER-USER", 9),
    Column("user_free", "FREE", 5),
    Column("all_gpus", "ALL GPU", 8),
    Column("group_gpus", "GROUP", 6),
    Column("group_free", "FREE", 5),
    Column("users", "USERS", 6),
    Column("verdict", "VERDICT", 26, flex=True),
)

QOS_COMPACT_COLUMNS: tuple[Column, ...] = (
    Column("name", "QOS", 13, flex=True),
    Column("my_gpus", "MINE", 5),
    Column("user_gpus", "LIMIT", 6),
    Column("user_free", "FREE", 5),
    Column("verdict", "VERDICT", 26, flex=True),
)

QOS_WIDE_MIN_WIDTH = 140


def qos_cell_map(qos: QosPressure) -> dict[str, Text]:
    """Cells for one QOS row, with the two caps kept visibly distinct."""
    if qos.blocks_new_gpu_job:
        verdict = Text("would refuse another GPU", style="bold red")
    elif qos.my_gpu_headroom is not None and qos.my_gpu_headroom <= 1:
        verdict = Text("one GPU left for you", style="yellow")
    elif qos.is_limited:
        verdict = Text("room to submit", style="green")
    else:
        verdict = Text("no GPU cap configured", style="dim")

    def free(limit: int | None, headroom: int | None) -> Text:
        if limit is None or headroom is None:
            return Text("-", style="dim")
        if headroom == 0:
            return Text("0", style="bold red")
        return Text(str(headroom), style="yellow" if headroom <= 1 else "")

    return {
        "name": Text(qos.name),
        "wall": Text(qos.max_wall or "-"),
        "my_gpus": Text(str(qos.my_gpus), style="cyan" if qos.my_gpus else "dim"),
        "user_gpus": Text("-" if qos.user_gpu_limit is None else str(qos.user_gpu_limit)),
        "user_free": free(qos.user_gpu_limit, qos.my_gpu_headroom),
        "all_gpus": Text(str(qos.running_gpus)),
        "group_gpus": Text(
            "-" if qos.group_gpu_limit is None else str(qos.group_gpu_limit)
        ),
        "group_free": free(qos.group_gpu_limit, qos.group_gpu_headroom),
        "users": Text(str(qos.running_users), style="dim"),
        "verdict": verdict,
    }


def build_qos_table(
    snapshot: Snapshot,
    *,
    width: int | None = None,
    title: str | None = None,
) -> Table:
    columns = (
        QOS_COLUMNS if width is None or width >= QOS_WIDE_MIN_WIDTH else QOS_COMPACT_COLUMNS
    )
    rows = [[qos_cell_map(q)[c.key] for c in columns] for q in snapshot.qos_pressure]
    return _table_for(columns, rows, title=title)
