"""Command-line entry point: ``tui`` (default), ``status`` and ``doctor``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

from . import __version__
from .config import Config, load_config
from .duration import format_epoch, format_time_limit
from .models import (
    SORT_MODES,
    AccountUsage,
    Alert,
    HistoryJob,
    Job,
    Partition,
    QosPressure,
    Snapshot,
    Usage,
)
from .render import (
    WIDE_LAYOUT,
    build_account_table,
    build_alert_table,
    build_history_table,
    build_partition_table,
    build_qos_table,
    build_table,
    gpu_utilisation,
    summary_line,
)
from .store import HistoryStore, StoreError
from .transport import SSHTransport, TransportError

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

SORT_CHOICES = [mode for mode, _ in SORT_MODES]


# --------------------------------------------------------------------- helpers


def opt(args: argparse.Namespace, name: str, default=None):
    """Read an optional parsed value.

    Needed because some flags are suppressed by default (see :func:`add_common_flags`)
    and because the no-subcommand case leaves subcommand flags entirely unset.
    """
    return getattr(args, name, default)


def make_transport(config: Config) -> SSHTransport:
    return SSHTransport(
        config.host,
        user=config.user,
        ssh_binary=config.ssh_binary,
        connect_timeout=config.connect_timeout,
        command_timeout=config.command_timeout,
        batch_mode=config.batch_mode,
        history_hours=config.history_hours,
        account_days=config.account_days,
    )


def make_store(config: Config) -> HistoryStore | None:
    """Open the local history database, or ``None`` when history is switched off.

    A broken or unreadable database must not take the whole monitor down with it, so
    the failure is reported and the caller carries on with live data and no alerts.
    """
    if not config.history_enabled:
        return None
    try:
        return HistoryStore(
            config.history_path,
            pending_alert_sec=config.pending_alert_sec,
            limit_soon_sec=config.limit_soon_sec,
        )
    except StoreError as exc:
        print(f"warning: {exc}", file=sys.stderr)
        return None


def fetch_snapshot(config: Config) -> Snapshot:
    """Run one probe and build a snapshot. Raises :class:`TransportError`."""
    result = make_transport(config).fetch()
    return Snapshot.from_payload(result.payload, time.time(), result.elapsed_sec)


def gpu_to_dict(job: Job, usage: Usage | None) -> dict:
    """The GPU utilisation figures and the idle-GPU policy they were judged against.

    The raw value is included alongside the derived per-GPU one so a caller can see for
    themselves whether the division makes sense -- that assumption is the weakest link
    in the column (see the README).
    """
    per_gpu, policy = gpu_utilisation(job, usage)
    return {
        "gpu_count": job.gpu_count,
        "gpu_util_raw": usage.gpu_util if usage else None,
        "gpu_util_per_gpu": round(per_gpu, 2) if per_gpu is not None else None,
        "gpu_util_verdict": policy.verdict(per_gpu),
        "gpu_util_policy": policy.pattern,
        "gpu_util_cancel_pct": policy.cancel_pct,
        "gpu_util_warn_pct": policy.warn_pct,
        "gpu_memory_mb": usage.gpu_memory_mb if usage else None,
    }


def job_to_dict(job: Job, now: float, usage: Usage | None = None) -> dict:
    """Flat, script-friendly representation used by ``status --json``.

    ``usage`` carries the live resource figures from ``sstat`` when they were
    collected, alongside the memory the job asked for -- the pair is what makes
    the number mean something.
    """
    return {
        "display_id": job.display_id,
        "job_id": job.job_id,
        "array_job_id": job.array_job_id,
        "array_task_id": job.array_task_id,
        "array_task_string": job.array_task_string,
        "name": job.name,
        "state": job.state,
        "phase": job.wait_phase(now),
        "partition": job.partition,
        "qos": job.qos,
        "account": job.account,
        "priority": job.priority,
        "gpus": job.gpu_count,
        "gpu_label": job.gpu_label,
        "cpus": job.cpus,
        "memory": job.memory_label,
        "memory_limit_mb": job.memory_limit_mb,
        "memory_used_kb": usage.max_rss_kb if usage else None,
        "memory_used_gb": (round(usage.max_rss_gb, 2) if usage and usage.max_rss_gb else None),
        "cpu_seconds": usage.cpu_seconds if usage else None,
        "usage_steps": usage.steps if usage else 0,
        **gpu_to_dict(job, usage),
        "node_count": job.node_count,
        "nodelist": job.nodelist,
        "time_limit_min": job.time_limit_min,
        "time_limit": format_time_limit(job.time_limit_min, job.time_limit_infinite),
        "submit_time": job.submit_time,
        "submit_time_local": format_epoch(job.submit_time, "%Y-%m-%dT%H:%M:%S"),
        "start_time": job.start_time,
        "start_time_local": format_epoch(job.start_time, "%Y-%m-%dT%H:%M:%S"),
        # The two derived numbers this tool exists to report.
        "queue_wait_sec": job.queue_wait_sec(now),
        "run_time_sec": job.run_time_sec(now),
        "time_left_sec": job.time_left_sec(now),
        "reason": job.reason,
        "reason_text": job.reason_text,
        "dependency": job.dependency,
        "stdout": job.stdout,
        "stderr": job.stderr,
        "workdir": job.workdir,
    }


def partition_to_dict(partition: Partition) -> dict:
    return {
        "name": partition.name,
        "gpus_total": partition.gpus_total,
        "gpus_used": partition.gpus_used,
        "gpus_free": partition.gpus_free,
        "gpu_type": partition.gpu_type,
        "gpu_types": list(partition.gpu_types),
        "nodes_total": partition.nodes_total,
        "nodes_idle": partition.nodes_idle,
        "nodes_mixed": partition.nodes_mixed,
        "nodes_allocated": partition.nodes_allocated,
        "nodes_other": partition.nodes_other,
        "cpus_total": partition.cpus_total,
        "cpus_free": partition.cpus_free,
        "mem_total_mb": partition.mem_total_mb,
        "mem_free_gb": partition.mem_free_gb,
        "max_time_min": partition.max_time_min,
        "max_time_infinite": partition.max_time_infinite,
    }


def history_to_dict(record: HistoryJob, now: float) -> dict:
    return {
        "display_id": record.display_id,
        "job_id": record.job_id,
        "array_job_id": record.array_job_id,
        "array_task_id": record.array_task_id,
        "name": record.name,
        "state": record.state,
        "partition": record.partition,
        "qos": record.qos,
        "account": record.account,
        "exit_status": record.exit_status,
        "exit_code": record.exit_code,
        "exit_signal": record.exit_signal,
        "exit_label": record.exit_label,
        "is_failed": record.is_failed,
        "elapsed_sec": record.elapsed_sec,
        "submit_time": record.submit_time,
        "start_time": record.start_time,
        "end_time": record.end_time,
        "queue_wait_sec": record.queue_wait_sec(now),
        "gpus": record.gpus,
        "outcome": record.outcome,
    }


def account_to_dict(account: AccountUsage) -> dict:
    return {
        "account": account.account,
        "qos": list(account.qos),
        "running_jobs": account.running_jobs,
        "running_gpus": account.running_gpus,
        "running_cpus": account.running_cpus,
        "running_nodes": account.running_nodes,
        "running_users": account.running_users,
        "is_shared": account.is_shared,
        "window_days": account.window_days,
        "window_gpu_hours": account.window_gpu_hours,
        "window_cpu_hours": account.window_cpu_hours,
        "window_jobs": account.window_jobs,
        "norm_shares": account.norm_shares,
        "effectv_usage": account.effectv_usage,
        "fairshare": account.fairshare,
        "level_fs": account.level_fs,
        "budget_configured": account.budget_configured,
        "priority_note": account.priority_note,
    }


def qos_to_dict(qos: QosPressure) -> dict:
    return {
        "name": qos.name,
        "max_wall": qos.max_wall,
        "group_gpu_limit": qos.group_gpu_limit,
        "group_cpu_limit": qos.group_cpu_limit,
        "group_memory_limit_mb": qos.group_memory_limit_mb,
        "user_gpu_limit": qos.user_gpu_limit,
        "user_cpu_limit": qos.user_cpu_limit,
        "user_memory_limit_mb": qos.user_memory_limit_mb,
        "budget_configured": qos.budget_configured,
        "running_gpus": qos.running_gpus,
        "running_cpus": qos.running_cpus,
        "running_jobs": qos.running_jobs,
        "running_users": qos.running_users,
        "my_gpus": qos.my_gpus,
        "my_cpus": qos.my_cpus,
        "my_jobs": qos.my_jobs,
        "my_gpu_headroom": qos.my_gpu_headroom,
        "group_gpu_headroom": qos.group_gpu_headroom,
        "blocks_new_gpu_job": qos.blocks_new_gpu_job,
    }


def alert_to_dict(alert: Alert) -> dict:
    return {
        "job_key": alert.job_key,
        "at": alert.at,
        "kind": alert.kind,
        "severity": alert.severity,
        "message": alert.message,
    }


def snapshot_to_dict(
    snapshot: Snapshot,
    now: float,
    alerts: tuple[Alert, ...] | list[Alert] = (),
) -> dict:
    from .models import sort_jobs

    return {
        "hostname": snapshot.hostname,
        "user": snapshot.user,
        "slurm_version": snapshot.slurm_version,
        "generated_at": snapshot.generated_at,
        "fetched_at": snapshot.fetched_at,
        "probe_elapsed_sec": snapshot.probe_elapsed_sec,
        "sections": list(snapshot.sections),
        "counts": snapshot.counts(),
        "longest_wait_sec": snapshot.longest_wait(now),
        "errors": list(snapshot.errors),
        "timings_ms": snapshot.timings_ms,
        "jobs": [
            job_to_dict(job, now, snapshot.usage_for(job))
            for job in sort_jobs(list(snapshot.jobs), "wait", now)
        ],
        "partitions": [partition_to_dict(p) for p in snapshot.partitions],
        "history": [history_to_dict(h, now) for h in snapshot.history],
        "usage": [
            {
                "job_id": entry.job_id,
                "max_rss_kb": entry.max_rss_kb,
                "max_rss_gb": round(entry.max_rss_gb, 2) if entry.max_rss_gb else None,
                "cpu_seconds": entry.cpu_seconds,
                "tasks": entry.tasks,
                "steps": entry.steps,
            }
            for entry in sorted(snapshot.usage.values(), key=lambda u: u.job_id or 0)
        ],
        "accounts": [account_to_dict(a) for a in snapshot.accounts],
        "qos": [qos_to_dict(q) for q in snapshot.qos_pressure],
        "account_days": snapshot.account_days,
        "alerts": [alert_to_dict(a) for a in alerts],
    }


# --------------------------------------------------------------------- commands


def record_snapshot(config: Config, snapshot: Snapshot, now: float) -> tuple[Alert, ...]:
    """Feed one snapshot to the local database and return whatever it newly raised.

    Called by every command, not just the dashboard, so that history and alerts
    accumulate from ordinary one-shot checks too. Never fatal: a database problem is
    reported and the caller continues.
    """
    store = make_store(config)
    if store is None:
        return ()
    try:
        return tuple(store.record(snapshot, now))
    except StoreError as exc:
        print(f"warning: {exc}", file=sys.stderr)
        return ()
    finally:
        store.close()


def cmd_status(args: argparse.Namespace, config: Config) -> int:
    """One snapshot, printed once. The scriptable, non-interactive view."""
    try:
        snapshot = fetch_snapshot(config)
    except TransportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    now = time.time()
    alerts = record_snapshot(config, snapshot, now)

    if opt(args, "json", False):
        json.dump(
            snapshot_to_dict(snapshot, now, alerts),
            sys.stdout,
            indent=2,
            default=str,
        )
        sys.stdout.write("\n")
        return EXIT_OK

    from rich.console import Console

    console = Console()
    console.print(
        summary_line(
            snapshot,
            now,
            target=make_transport(config).describe(),
            interval=config.refresh_interval,
            auto=False,
        ),
        # Keep the summary on one line even in a narrow window; wrapping it pushes
        # the table down and looks like a rendering bug.
        no_wrap=True,
        overflow="ellipsis",
    )
    console.print(
        build_table(
            snapshot,
            now,
            sort_mode=opt(args, "sort", "wait"),
            reverse=opt(args, "reverse", False),
            max_rows=opt(args, "limit", 0),
            # `--wide` forces every column even if the terminal is narrow.
            layout=WIDE_LAYOUT if opt(args, "wide", False) else None,
            width=console.width,
        )
    )
    if snapshot.errors:
        for error in snapshot.errors:
            console.print(f"[yellow]probe: {error}[/yellow]")
    if config.warnings:
        for warning in config.warnings:
            console.print(f"[yellow]config: {warning}[/yellow]")
    if alerts:
        console.print()
        console.print(build_alert_table(alerts, now, title="new alerts"))
    limit = opt(args, "limit", 0)
    if limit and len(snapshot.jobs) > limit:
        console.print(f"[dim]{len(snapshot.jobs) - limit} more not shown (--limit)[/dim]")
    return EXIT_OK


def cmd_partitions(args: argparse.Namespace, config: Config) -> int:
    """Show where there is room to run something right now."""
    try:
        snapshot = fetch_snapshot(config)
    except TransportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    now = time.time()
    record_snapshot(config, snapshot, now)

    only_mine = opt(args, "mine", False)
    shown = snapshot.partitions_for_my_jobs() if only_mine else list(snapshot.partitions)

    if opt(args, "json", False):
        json.dump(
            {
                "generated_at": snapshot.generated_at,
                "sections": list(snapshot.sections),
                "partitions": [partition_to_dict(p) for p in shown],
                "held_gpus": snapshot.held_gpus_by_partition(),
                "my_partitions": list(snapshot.my_partitions),
            },
            sys.stdout,
            indent=2,
            default=str,
        )
        sys.stdout.write("\n")
        return EXIT_OK

    from rich.console import Console

    console = Console()
    if not snapshot.partitions:
        console.print("[yellow]no partition data: sinfo returned nothing[/yellow]")
        return EXIT_FAILURE
    console.print(build_partition_table(snapshot, width=console.width, only_mine=only_mine))
    held = snapshot.held_gpus_by_partition()
    if held:
        summary = ", ".join(f"{name}: {count}" for name, count in sorted(held.items()))
        console.print(f"[dim]GPUs held by your running jobs — {summary}[/dim]")
    if not only_mine:
        console.print(
            f"[dim]{len(shown)} partitions. Torch's partitions overlap "
            "(project partitions share hardware, and `all` pools everything), so do "
            "not add these up. Use --mine for just the ones your jobs are in.[/dim]"
        )
    for error in snapshot.errors:
        console.print(f"[yellow]probe: {error}[/yellow]")
    return EXIT_OK


def cmd_accounts(args: argparse.Namespace, config: Config) -> int:
    """Which accounts I can use, what they consume, and what caps are in the way."""
    try:
        snapshot = fetch_snapshot(config)
    except TransportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    now = time.time()
    record_snapshot(config, snapshot, now)
    show_qos = bool(opt(args, "qos", False))

    if opt(args, "json", False):
        json.dump(
            {
                "generated_at": snapshot.generated_at,
                "account_days": snapshot.account_days,
                "sections": list(snapshot.sections),
                "accounts": [account_to_dict(a) for a in snapshot.accounts],
                "qos": [qos_to_dict(q) for q in snapshot.qos_pressure],
            },
            sys.stdout,
            indent=2,
            default=str,
        )
        sys.stdout.write("\n")
        return EXIT_OK

    from rich.console import Console

    console = Console()
    if not snapshot.accounts:
        console.print(
            "[yellow]no account data: sacctmgr returned no associations for you "
            "(it is often restricted to admins on other sites)[/yellow]"
        )
        for error in snapshot.errors:
            console.print(f"[yellow]probe: {error}[/yellow]")
        return EXIT_FAILURE

    if show_qos:
        console.print(build_qos_table(snapshot, width=console.width))
        console.print(
            "[dim]PER-USER caps you personally; GROUP caps every user in that QOS. "
            "QOSMaxGRESPerUser comes from the former, QOSGrpGRES from the latter.[/dim]"
        )
    else:
        console.print(build_account_table(snapshot, width=console.width))
        console.print(
            "[dim]Usage is account-wide, so USERS > 1 means the figures are not only "
            "yours. GPU-h/CPU-h are the reporting numbers; FAIR/EFF explain queue "
            "priority. Use --qos for the caps that refuse a job.[/dim]"
        )
        blockers = snapshot.blocking_qos()
        if blockers:
            console.print()
            names = ", ".join(f"{q.name} ({q.my_gpus}/{q.user_gpu_limit} GPU)" for q in blockers)
            console.print(f"[bold red]at your per-user GPU cap in: {names}[/bold red]")
            console.print(
                "[dim]a further GPU job there would be refused with "
                "QOSMaxGRESPerUser[/dim]"
            )
    for error in snapshot.errors:
        console.print(f"[yellow]probe: {error}[/yellow]")
    return EXIT_OK


def cmd_history(args: argparse.Namespace, config: Config) -> int:
    """Show what happened recently, from sacct plus everything recorded locally."""
    try:
        snapshot = fetch_snapshot(config)
    except TransportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    now = time.time()
    alerts = record_snapshot(config, snapshot, now)
    limit = opt(args, "limit", 50)

    if opt(args, "json", False):
        json.dump(
            {
                "generated_at": snapshot.generated_at,
                "history_hours": snapshot.history_hours,
                "finished": [history_to_dict(h, now) for h in snapshot.finished(limit)],
                "failures": [
                    history_to_dict(h, now) for h in snapshot.recent_failures(limit)
                ],
                "alerts": [alert_to_dict(a) for a in alerts],
                "recorded": _recorded_history(config, limit),
            },
            sys.stdout,
            indent=2,
            default=str,
        )
        sys.stdout.write("\n")
        return EXIT_OK

    from rich.console import Console

    console = Console()
    window = snapshot.history_hours or config.history_hours
    finished = snapshot.finished(limit)
    if not finished:
        console.print(f"[yellow]no finished jobs in the last {window}h[/yellow]")
    else:
        console.print(
            build_history_table(
                finished,
                now,
                width=console.width,
                title=f"finished in the last {window}h (newest first)",
            )
        )
    failures = snapshot.recent_failures(limit)
    if failures:
        console.print()
        console.print(
            build_history_table(
                failures, now, width=console.width, title="failures worth a look"
            )
        )
    stats = _queue_wait_rows(config)
    if stats:
        console.print()
        console.print("[bold]observed queue wait[/bold]")
        console.print(
            "[dim]from jobs that actually started, recorded locally over the last "
            "7 days[/dim]"
        )
        for stat in stats:
            console.print(
                f"  {stat['partition']:<24} n={stat['samples']:<5} "
                f"avg {stat['average_sec'] / 60:.0f}m  "
                f"median {stat['median_sec'] / 60:.0f}m  "
                f"max {stat['max_sec'] / 60:.0f}m"
            )
    if alerts:
        console.print()
        console.print(build_alert_table(alerts, now, title="new alerts"))
    for error in snapshot.errors:
        console.print(f"[yellow]probe: {error}[/yellow]")
    return EXIT_OK


def _queue_wait_rows(config: Config, since_days: int = 7) -> list[dict]:
    """Queue-wait statistics from the local database, if there is one."""
    store = make_store(config)
    if store is None:
        return []
    try:
        since = time.time() - since_days * 86400
        return [
            {
                "partition": stat.partition,
                "samples": stat.samples,
                "average_sec": stat.average_sec,
                "median_sec": stat.median_sec,
                "max_sec": stat.max_sec,
            }
            for stat in store.queue_wait_stats(since)
        ]
    except StoreError:
        return []
    finally:
        store.close()


def _recorded_history(config: Config, limit: int) -> list[dict]:
    """Jobs the local database has seen end, going back further than sacct's window."""
    store = make_store(config)
    if store is None:
        return []
    try:
        return store.finished_jobs(limit)
    except StoreError:
        return []
    finally:
        store.close()


# --------------------------------------------------------------------- logs


class JobMatch(NamedTuple):
    """The outcome of resolving a job reference from the command line."""

    job: Job | None = None
    problem: str = ""
    #: Set when the match was a judgement call worth mentioning, e.g. one of several
    #: running array tasks.
    note: str = ""


def resolve_job(snapshot: Snapshot, key: str) -> JobMatch:
    """Find a job by display id, numeric id, array master id or name fragment.

    Being permissive matters because the identifiers are long and awkward to retype.
    ``17325640_42`` is exact, but ``17325640`` should find the array, and ``train``
    should find the job by name when that is unambiguous.
    """
    wanted = key.strip()
    exact = snapshot.by_id(wanted)
    if exact is not None:
        return JobMatch(exact)

    lowered = wanted.lower()
    if not lowered:
        return JobMatch(problem="no job specified")

    candidates: list[Job] = []
    if lowered.isdigit():
        number = int(lowered)
        candidates = [
            job
            for job in snapshot.jobs
            if job.job_id == number or job.array_job_id == number
        ]
    if not candidates:
        candidates = [job for job in snapshot.jobs if lowered in job.name.lower()]

    if not candidates:
        return JobMatch(
            problem=f"no job matching {wanted!r} in your queue (try `lampter status`)"
        )
    if len(candidates) == 1:
        return JobMatch(candidates[0])

    running = [job for job in candidates if job.is_running]
    if len(running) == 1:
        return JobMatch(running[0])

    if running and len({job.name for job in running}) == 1:
        # Several tasks of the same array are running. Their logs are near-identical
        # and the freshest is the interesting one, so pick it rather than making
        # somebody disambiguate four long ids.
        freshest = max(running, key=lambda job: job.start_time or 0)
        return JobMatch(
            freshest,
            note=(
                f"{wanted!r} matches {len(candidates)} jobs; showing "
                f"{freshest.display_id}, the most recently started"
            ),
        )

    shown = ", ".join(job.display_id for job in candidates[:8])
    extra = "" if len(candidates) <= 8 else f" (+{len(candidates) - 8} more)"
    return JobMatch(
        problem=(
            f"{wanted!r} matches several jobs: {shown}{extra}. "
            "Pass a full id such as 17325640_42."
        )
    )


def cmd_logs(args: argparse.Namespace, config: Config) -> int:
    """Show, and optionally follow, a job's output file.

    This is the one command that is not a single round trip: following a log keeps a
    channel open. It only runs when asked, so it adds nothing to the background cost
    of the monitor.
    """
    explicit_path = opt(args, "path")
    job: Job | None = None

    if explicit_path:
        path = explicit_path
    else:
        try:
            snapshot = fetch_snapshot(config)
        except TransportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_FAILURE
        record_snapshot(config, snapshot, time.time())

        match = resolve_job(snapshot, opt(args, "job", ""))
        if match.job is None:
            print(f"error: {match.problem}", file=sys.stderr)
            return EXIT_FAILURE
        if match.note:
            print(f"note: {match.note}", file=sys.stderr)
        job = match.job
        path = job.error_log_path if opt(args, "stderr", False) else job.log_path
        if not path:
            print(
                f"error: {job.display_id} has no log file recorded "
                "(its sbatch script had no --output/--error)",
                file=sys.stderr,
            )
            return EXIT_FAILURE

    lines = opt(args, "lines", 200)
    follow = bool(opt(args, "follow", False))
    if follow and job is not None and not job.is_running:
        # Following a file nobody is writing to just hangs, which looks like a bug.
        print(
            f"note: {job.display_id} is {job.state}, so there is nothing more to "
            "follow; showing the existing log",
            file=sys.stderr,
        )
        follow = False

    if job is not None:
        print(
            f"# {job.display_id} {job.name} on {job.partition} ({job.state})",
            file=sys.stderr,
        )
    print(f"# tail -n {lines}{' -f' if follow else ''} {path}", file=sys.stderr)

    transport = make_transport(config)
    try:
        for line in transport.stream_file(path, lines=lines, follow=follow):
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    except TransportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        # Ctrl-C is how you stop a follow; that is not an error.
        return EXIT_OK
    return EXIT_OK


def cmd_alerts(args: argparse.Namespace, config: Config) -> int:
    """Show alerts raised over the whole local history, newest first."""
    limit = opt(args, "limit", 50)
    store = make_store(config)
    if store is None:
        print("error: history is disabled, so no alerts have been recorded", file=sys.stderr)
        return EXIT_FAILURE
    try:
        alerts = store.recent_alerts(limit)
        stats = store.stats()
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    now = time.time()
    if opt(args, "json", False):
        json.dump(
            {
                "alerts": [alert_to_dict(a) for a in alerts],
                "store": stats,
                "path": str(store.path),
            },
            sys.stdout,
            indent=2,
            default=str,
        )
        sys.stdout.write("\n")
        store.close()
        return EXIT_OK

    from rich.console import Console

    console = Console()
    if not alerts:
        console.print("[green]no alerts recorded[/green]")
    else:
        console.print(build_alert_table(alerts, now, title="alerts (newest first)"))
    console.print(
        f"[dim]database {store.path} — {stats.get('jobs', 0)} jobs, "
        f"{stats.get('transitions', 0)} transitions, {stats.get('alerts', 0)} alerts[/dim]"
    )
    store.close()
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    """Explain the connection end to end, so failures are diagnosable."""
    from rich.console import Console

    console = Console()
    console.print("[bold]lampter doctor[/bold]")

    console.print(f"  client version     {__version__}")
    console.print(f"  config file        {config.source or '(none found, using defaults)'}")
    console.print(f"  ssh host           {config.host}")
    console.print(f"  ssh user           {config.user or '(from ~/.ssh/config)'}")
    console.print(f"  ssh binary         {config.ssh_binary}")
    console.print(f"  batch mode         {config.batch_mode}")
    console.print(f"  connect timeout    {config.connect_timeout}s")
    console.print(f"  command timeout    {config.command_timeout}s")
    console.print(f"  refresh interval   {config.refresh_interval}s (jobs)")
    console.print(f"  capacity interval  {config.partition_interval}s (sinfo)")
    console.print(f"  history interval   {config.history_interval}s (sacct)")
    console.print(f"  usage interval     {config.usage_interval}s (sstat)")
    console.print(f"  accounts interval  {config.accounts_interval}s (sacctmgr/sshare)")
    console.print(f"  accounting window  {config.account_days}d")
    console.print(f"  history window     {config.history_hours}h")
    if not config.history_enabled:
        console.print("  local history      disabled (no alerts, no history view)")
    else:
        store = None
        try:
            store = HistoryStore(
                config.history_path,
                pending_alert_sec=config.pending_alert_sec,
                limit_soon_sec=config.limit_soon_sec,
            )
            stats = store.stats()
            console.print(
                f"  local history      {store.path} "
                f"({stats.get('jobs', 0)} jobs, {stats.get('transitions', 0)} "
                f"transitions, {stats.get('alerts', 0)} alerts)"
            )
        except StoreError as exc:
            console.print(f"  [red]local history      unusable: {exc}[/red]")
        finally:
            if store is not None:
                store.close()
    for warning in config.warnings:
        console.print(f"  [yellow]config warning   {warning}[/yellow]")

    transport = make_transport(config)
    console.print("\n[bold]running probe…[/bold]")
    try:
        result = transport.fetch()
    except TransportError as exc:
        console.print(f"  [bold red]FAILED[/bold red] {exc}")
        console.print(
            "\n[dim]hints: is the host name in ~/.ssh/config? can you run "
            f"`ssh {config.host} hostname` by hand? is the VPN connected?[/dim]"
        )
        return EXIT_FAILURE

    payload = result.payload
    console.print(f"  [green]ok[/green] round trip {result.elapsed_sec:.2f}s")
    console.print(f"  remote host        {payload.get('hostname')}")
    console.print(f"  remote user        {payload.get('user')}")
    console.print(f"  slurm version      {payload.get('slurm_version') or 'unknown'}")
    console.print(f"  probe schema       {payload.get('schema')}")
    console.print(f"  probe timings      {payload.get('timings_ms')}")
    console.print(f"  jobs returned      {len(payload.get('jobs') or [])}")

    errors = payload.get("errors") or []
    if errors:
        console.print("\n[bold yellow]probe reported problems:[/bold yellow]")
        for error in errors:
            console.print(f"  - {error}")
    else:
        console.print("\n[green]no probe errors[/green]")

    if not payload.get("slurm_version"):
        console.print(
            "[yellow]note: could not read the Slurm version; `sinfo` may not be on "
            "the login node's PATH.[/yellow]"
        )
    return EXIT_OK


def cmd_tui(args: argparse.Namespace, config: Config) -> int:
    """Launch the interactive dashboard."""
    try:
        from .ui import SlurmMonitorApp
    except ImportError as exc:
        print(
            f"error: the TUI needs the 'textual' package ({exc}).\n"
            "       install it with: pip install 'lampter[tui]'\n"
            "       or use the non-interactive view: lampter status",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    app = SlurmMonitorApp(
        make_transport(config),
        interval=config.refresh_interval,
        auto=not opt(args, "no_auto", False),
        sort_mode=opt(args, "sort", "wait"),
        reverse=opt(args, "reverse", False),
        max_rows=opt(args, "limit", 0),
        partition_interval=config.partition_interval,
        history_interval=config.history_interval,
        usage_interval=config.usage_interval,
        accounts_interval=config.accounts_interval,
        store=make_store(config),
    )
    try:
        app.run()
    finally:
        if app.store is not None:
            app.store.close()
    return EXIT_OK


# --------------------------------------------------------------------- parser


def add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Add the flags accepted both before and after the subcommand.

    Defaults are ``argparse.SUPPRESS`` deliberately. argparse parses a subcommand
    into its own namespace and then copies that namespace over the parent's, so an
    ordinary default here would silently erase options given *before* the
    subcommand -- ``lampter --host foo status`` would lose the host. With
    ``SUPPRESS`` an absent flag sets nothing and so cannot clobber anything.
    """
    parser.add_argument(
        "--host",
        default=argparse.SUPPRESS,
        help="SSH host or ~/.ssh/config alias (default: torch)",
    )
    parser.add_argument(
        "--user",
        default=argparse.SUPPRESS,
        help="remote username (default: the SSH login user)",
    )
    parser.add_argument(
        "--ssh", dest="ssh_binary", default=argparse.SUPPRESS, help="ssh binary to invoke"
    )
    parser.add_argument(
        "--config", type=Path, default=argparse.SUPPRESS, help="path to a TOML config file"
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=argparse.SUPPRESS,
        help="SSH connect timeout, seconds",
    )
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=argparse.SUPPRESS,
        help="whole-probe timeout, seconds",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=argparse.SUPPRESS,
        help="job-list refresh interval, seconds (default: 60; adjustable with +/-)",
    )
    parser.add_argument(
        "--no-batch-mode",
        action="store_true",
        default=argparse.SUPPRESS,
        help="allow ssh to prompt for a password instead of failing fast",
    )
    parser.add_argument(
        "--partition-interval",
        type=float,
        default=argparse.SUPPRESS,
        help="seconds between capacity (sinfo) refreshes (default: 200)",
    )
    parser.add_argument(
        "--history-interval",
        type=float,
        default=argparse.SUPPRESS,
        help="seconds between history (sacct) refreshes (default: 200)",
    )
    parser.add_argument(
        "--accounts-interval",
        type=float,
        default=argparse.SUPPRESS,
        help="seconds between account/QOS refreshes (default: 600)",
    )
    parser.add_argument(
        "--account-days",
        type=int,
        default=argparse.SUPPRESS,
        help="days of per-account consumption to total (default: 7)",
    )
    parser.add_argument(
        "--usage-interval",
        type=float,
        default=argparse.SUPPRESS,
        help="seconds between live resource-use (sstat) refreshes (default: 120)",
    )
    parser.add_argument(
        "--history-hours",
        type=int,
        default=argparse.SUPPRESS,
        help="how far back sacct is queried (default: 12)",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        default=argparse.SUPPRESS,
        help="do not use the local history database (no alerts, no history view)",
    )


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common_flags(common)

    parser = argparse.ArgumentParser(
        prog="lampter",
        description=(
            "A terminal dashboard for your SLURM jobs on a remote cluster, pulling data "
            "over SSH from your own machine. Works with any Slurm 23.11+ controller; "
            "developed against NYU's Torch cluster, which is the default host."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
        epilog=(
            "examples:\n"
            "  lampter                     # interactive dashboard\n"
            "  lampter status              # print one snapshot and exit\n"
            "  lampter status --json       # machine-readable\n"
            "  lampter partitions --mine   # where can I run right now?\n"
            "  lampter accounts --qos      # caps that would refuse a job\n"
            "  lampter history             # recent outcomes and queue waits\n"
            "  lampter logs <job> -f       # follow a job's output\n"
            "  lampter alerts              # everything that has gone wrong\n"
            "  lampter doctor              # diagnose the SSH connection\n"
            "\n"
            "connection flags may appear before or after the subcommand, e.g.\n"
            "  lampter --host torch status  ==  lampter status --host torch\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    def add_view(name: str, help_text: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text, parents=[common])
        child.add_argument(
            "--sort", choices=SORT_CHOICES, default="wait", help="sort column (default: wait)"
        )
        child.add_argument("--reverse", action="store_true", help="reverse the sort order")
        child.add_argument("--limit", type=int, default=0, help="show at most N jobs (0 = all)")
        return child

    tui = add_view("tui", "interactive dashboard (default)")
    tui.add_argument("--no-auto", action="store_true", help="start with auto-refresh disabled")

    status = add_view("status", "print a single snapshot and exit")
    status.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    status.add_argument(
        "--wide",
        action="store_true",
        help="show every column even if the terminal is too narrow",
    )

    partitions = sub.add_parser(
        "partitions",
        help="show free capacity per partition",
        parents=[common],
    )
    partitions.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    partitions.add_argument(
        "--mine",
        action="store_true",
        help="only the partitions your own jobs occupy",
    )

    history = sub.add_parser(
        "history",
        help="recent job outcomes, failures and observed queue waits",
        parents=[common],
    )
    history.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    history.add_argument(
        "--limit", type=int, default=50, help="show at most N records (default: 50)"
    )

    accounts = sub.add_parser(
        "accounts",
        help="accounts you can use, their consumption, and QOS caps",
        parents=[common],
    )
    accounts.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    accounts.add_argument(
        "--qos",
        action="store_true",
        help="show the QOS caps instead of per-account usage",
    )

    logs = sub.add_parser(
        "logs",
        help="show (and optionally follow) a job's output file",
        parents=[common],
    )
    logs.add_argument(
        "job",
        nargs="?",
        default="",
        help="job id, array task id, or part of the job name",
    )
    logs.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="keep streaming as the file grows (Ctrl-C to stop)",
    )
    logs.add_argument(
        "-n",
        "--lines",
        type=int,
        default=200,
        help="how many lines of history to show first (default: 200)",
    )
    logs.add_argument(
        "--stderr",
        action="store_true",
        help="read the stderr file instead of stdout",
    )
    logs.add_argument(
        "--path",
        default=None,
        help="read this remote file directly, bypassing the job lookup",
    )

    alerts = sub.add_parser(
        "alerts",
        help="alerts recorded by the local history database",
        parents=[common],
    )
    alerts.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    alerts.add_argument(
        "--limit", type=int, default=50, help="show at most N alerts (default: 50)"
    )

    sub.add_parser(
        "doctor",
        help="diagnose the connection and remote Slurm setup",
        parents=[common],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(
        path=opt(args, "config"),
        overrides={
            "host": opt(args, "host"),
            "user": opt(args, "user"),
            "ssh_binary": opt(args, "ssh_binary"),
            "connect_timeout": opt(args, "connect_timeout"),
            "command_timeout": opt(args, "command_timeout"),
            "refresh_interval": opt(args, "interval"),
            "batch_mode": False if opt(args, "no_batch_mode") else None,
            "partition_interval": opt(args, "partition_interval"),
            "history_interval": opt(args, "history_interval"),
            "usage_interval": opt(args, "usage_interval"),
            "accounts_interval": opt(args, "accounts_interval"),
            "account_days": opt(args, "account_days"),
            "history_hours": opt(args, "history_hours"),
            "history_enabled": False if opt(args, "no_history") else None,
        },
    )

    command = opt(args, "command") or "tui"
    try:
        if command == "status":
            return cmd_status(args, config)
        if command == "partitions":
            return cmd_partitions(args, config)
        if command == "accounts":
            return cmd_accounts(args, config)
        if command == "history":
            return cmd_history(args, config)
        if command == "alerts":
            return cmd_alerts(args, config)
        if command == "logs":
            return cmd_logs(args, config)
        if command == "doctor":
            return cmd_doctor(args, config)
        return cmd_tui(args, config)
    except KeyboardInterrupt:
        return EXIT_OK
