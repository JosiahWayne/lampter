"""Local SQLite history and alerting.

Neither Slurm command can answer "what changed since I last looked?":

* ``squeue`` forgets a job the instant it leaves the queue, so a job that fails
  between two refreshes simply vanishes.
* ``sacct`` can describe a job that has already ended, but only because we ask it
  again -- it still cannot tell us what happened *while we were watching*.

An alert is precisely that difference, so every refresh is recorded here: one row
per job ever seen, plus an append-only log of state transitions. Alerts are then
derived from those transitions, from the job's own scheduling outcome, and from how
long a pending job has been starving.

Two rules keep this from becoming noise:

* Alerts are deduplicated by ``(job_key, kind)`` in the database, so restarting the
  monitor or refreshing twice never repeats a warning.
* Only jobs we actually watched in ``squeue`` raise alerts (``watched = 1``).
  Backfilled ``sacct`` history is stored for the history view and for queue-wait
  statistics, but a job that finished before the monitor started is not news.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .models import Alert, HistoryJob, Job, Snapshot

#: Default alert threshold: how long a job may sit pending before it is worth
#: mentioning. Six hours is well past normal backfill on Torch.
DEFAULT_PENDING_ALERT_SEC = 6 * 3600

#: Warn when a running job is this close to its wall-clock limit, because that is
#: how jobs die unexpectedly.
DEFAULT_LIMIT_SOON_SEC = 15 * 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_key        TEXT PRIMARY KEY,
    job_id         INTEGER,
    array_job_id   INTEGER,
    array_task_id  INTEGER,
    name           TEXT NOT NULL DEFAULT '',
    partition      TEXT NOT NULL DEFAULT '',
    qos            TEXT NOT NULL DEFAULT '',
    account        TEXT NOT NULL DEFAULT '',
    gpus           INTEGER,
    submit_time    INTEGER,
    eligible_time  INTEGER,
    first_seen     REAL NOT NULL,
    last_seen      REAL NOT NULL,
    start_time     INTEGER,
    end_time       INTEGER,
    last_state     TEXT NOT NULL,
    final_state    TEXT,
    final_exit     TEXT,
    max_queue_wait INTEGER,
    watched        INTEGER NOT NULL DEFAULT 0,
    alerted_pending INTEGER NOT NULL DEFAULT 0,
    alerted_limit   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transitions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_key    TEXT NOT NULL,
    at         REAL NOT NULL,
    from_state TEXT,
    to_state   TEXT NOT NULL,
    reason     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS transitions_at ON transitions(at DESC);
CREATE INDEX IF NOT EXISTS transitions_job ON transitions(job_key, at);

CREATE TABLE IF NOT EXISTS alerts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_key  TEXT NOT NULL,
    at       REAL NOT NULL,
    kind     TEXT NOT NULL,
    severity TEXT NOT NULL,
    message  TEXT NOT NULL,
    UNIQUE(job_key, kind)
);
CREATE INDEX IF NOT EXISTS alerts_at ON alerts(at DESC);
"""


class StoreError(RuntimeError):
    """Raised when the history database cannot be read or written."""


@dataclass(frozen=True)
class QueueWaitStat:
    """Observed queue wait for one partition, over jobs that have started."""

    partition: str
    samples: int
    average_sec: float
    median_sec: float
    max_sec: int


def default_history_path() -> Path:
    """Where the database lives, following the XDG-ish convention used elsewhere."""
    return Path.home() / ".local" / "share" / "lampter" / "history.db"


class HistoryStore:
    """A small SQLite database of everything the monitor has observed."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        pending_alert_sec: int = DEFAULT_PENDING_ALERT_SEC,
        limit_soon_sec: int = DEFAULT_LIMIT_SOON_SEC,
    ) -> None:
        self.path = Path(path) if path is not None else default_history_path()
        self.pending_alert_sec = pending_alert_sec
        self.limit_soon_sec = limit_soon_sec
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(self.path), timeout=10)
            self._connection.row_factory = sqlite3.Row
            # WAL keeps a `status` one-shot from blocking the running TUI, and the
            # busy timeout absorbs the brief overlap when both write.
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=10000")
            self._connection.executescript(SCHEMA)
            self._connection.commit()
        except (OSError, sqlite3.Error) as exc:
            raise StoreError(f"cannot open history database {self.path}: {exc}") from exc

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        try:
            self._connection.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> HistoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---------------------------------------------------------------- recording

    def record(self, snapshot: Snapshot, now: float | None = None) -> list[Alert]:
        """Ingest one snapshot; return the alerts it newly raised.

        Safe to call with a snapshot from any source, and safe to call repeatedly:
        transitions are only logged when a state actually changes, and alerts are
        deduplicated by the database.
        """
        moment = time.time() if now is None else now
        try:
            with self._connection:  # one transaction per refresh
                raised = self._record_live_jobs(snapshot, moment)
                raised += self._record_history(snapshot, moment)
                raised += self._record_disappearances(snapshot, moment)
                raised += self._record_pending_alerts(snapshot, moment)
                raised += self._record_limit_alerts(snapshot, moment)
            return raised
        except sqlite3.Error as exc:
            raise StoreError(f"cannot record snapshot: {exc}") from exc

    def _record_live_jobs(self, snapshot: Snapshot, now: float) -> list[Alert]:
        """Upsert the jobs currently in the queue and log their state changes.

        State changes are recorded here but do not themselves raise alerts: a job
        moving PENDING -> RUNNING is expected, and the interesting transitions
        (failure, completion) are recognised from sacct in :meth:`_record_history`,
        which is the only source that knows *why* a job left the queue.
        """
        for job in snapshot.jobs:
            key = job.display_id
            existing = self._get_job(key)
            wait = job.queue_wait_sec(now)

            if existing is None:
                self._insert_job(job, now, watched=True)
                self._log_transition(key, now, None, job.state, job.reason)
                continue

            if existing["last_state"] != job.state:
                self._log_transition(key, now, existing["last_state"], job.state, job.reason)
            self._update_job(job, now, wait)
        return []

    def _insert_job(self, job: Job, now: float, *, watched: bool) -> None:
        self._connection.execute(
            """
            INSERT INTO jobs (
                job_key, job_id, array_job_id, array_task_id, name, partition, qos,
                account, gpus, submit_time, eligible_time, first_seen, last_seen,
                start_time, end_time, last_state, final_state, final_exit,
                max_queue_wait, watched
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?,?)
            """,
            (
                job.display_id,
                job.job_id,
                job.array_job_id,
                job.array_task_id,
                job.name,
                job.partition,
                job.qos,
                job.account,
                job.gpu_count,
                job.submit_time,
                job.eligible_time,
                now,
                now,
                job.start_time,
                job.end_time,
                job.state,
                job.queue_wait_sec(now),
                1 if watched else 0,
            ),
        )

    def _update_job(self, job: Job, now: float, wait: int | None) -> None:
        self._connection.execute(
            """
            UPDATE jobs SET
                last_seen = ?, last_state = ?, start_time = ?, end_time = ?,
                submit_time = COALESCE(submit_time, ?),
                eligible_time = COALESCE(eligible_time, ?),
                gpus = COALESCE(gpus, ?),
                max_queue_wait = MAX(COALESCE(max_queue_wait, 0), COALESCE(?, 0)),
                watched = 1
            WHERE job_key = ?
            """,
            (
                now,
                job.state,
                job.start_time,
                job.end_time,
                job.submit_time,
                job.eligible_time,
                job.gpu_count,
                wait,
                job.display_id,
            ),
        )

    def _log_transition(
        self,
        job_key: str,
        now: float,
        from_state: str | None,
        to_state: str,
        reason: str,
    ) -> None:
        self._connection.execute(
            "INSERT INTO transitions (job_key, at, from_state, to_state, reason)"
            " VALUES (?,?,?,?,?)",
            (job_key, now, from_state, to_state, reason or ""),
        )

    def _record_history(self, snapshot: Snapshot, now: float) -> list[Alert]:
        """Fold sacct's outcomes in, and alert on failures we watched happen.

        squeue can never show a job as FAILED: by the time it would, the job has
        already left the queue. sacct is the only way to learn how a job we were
        watching actually ended.
        """
        alerts: list[Alert] = []
        for record in snapshot.history:
            key = record.display_id
            existing = self._get_job(key)

            if existing is None:
                # Before the monitor was running: keep it for the history view and
                # for queue-wait statistics, but it is not news.
                self._insert_history_job(record, now)
                continue

            if existing["final_state"] is not None:
                continue  # already resolved
            if not record.is_terminal:
                # Still live; keep the record's own view as a fallback for the case
                # where squeue and sacct disagree about the state.
                continue

            self._connection.execute(
                """
                UPDATE jobs SET final_state = ?, final_exit = ?, end_time = ?,
                    start_time = COALESCE(start_time, ?),
                    submit_time = COALESCE(submit_time, ?),
                    last_seen = ?
                WHERE job_key = ?
                """,
                (
                    record.state,
                    record.exit_label,
                    record.end_time,
                    record.start_time,
                    record.submit_time,
                    now,
                    key,
                ),
            )
            self._log_transition(key, now, existing["last_state"], record.state, record.reason)

            if record.is_failed:
                alerts.append(
                    self._raise_alert(
                        key,
                        now,
                        kind=f"failed:{record.state}",
                        severity="error",
                        message=(
                            f"{record.name or key} ({key}) {record.outcome} on "
                            f"{record.partition} (exit {record.exit_label})"
                        ),
                    )
                )
            elif record.state == "COMPLETED" and existing["array_task_id"] is None:
                # One line per finished job, but not per array task -- a 52-task
                # array would otherwise bury everything else.
                alerts.append(
                    self._raise_alert(
                        key,
                        now,
                        kind="completed",
                        severity="info",
                        message=(
                            f"{record.name or key} ({key}) finished in "
                            f"{record.elapsed_sec or 0}s on {record.partition}"
                        ),
                    )
                )
        return [a for a in alerts if a is not None]

    def _insert_history_job(self, record: HistoryJob, now: float) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO jobs (
                job_key, job_id, array_job_id, array_task_id, name, partition, qos,
                account, gpus, submit_time, eligible_time, first_seen, last_seen,
                start_time, end_time, last_state, final_state, final_exit,
                max_queue_wait, watched
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
            """,
            (
                record.display_id,
                record.job_id,
                record.array_job_id,
                record.array_task_id,
                record.name,
                record.partition,
                record.qos,
                record.account,
                record.gpus,
                record.submit_time,
                record.eligible_time,
                now,
                now,
                record.start_time,
                record.end_time,
                record.state,
                record.state if record.is_terminal else None,
                record.exit_label if record.is_terminal else None,
                record.queue_wait_sec(now),
            ),
        )

    def _record_disappearances(self, snapshot: Snapshot, now: float) -> list[Alert]:
        """Mark jobs that left the queue, so the history view can show them gone.

        The outcome itself comes from sacct in :meth:`_record_history`; here we only
        make sure a job that vanished without a sacct record yet is not still
        displayed as running forever.
        """
        live = {job.display_id for job in snapshot.jobs}
        rows = self._connection.execute(
            "SELECT job_key, last_state, name FROM jobs"
            " WHERE watched = 1 AND final_state IS NULL"
        ).fetchall()
        for row in rows:
            key = row["job_key"]
            if key in live:
                continue
            # Leave the state alone; sacct will resolve it. Just stop refreshing it
            # as if it were live by recording that we last saw it here.
            self._connection.execute(
                "UPDATE jobs SET last_seen = ? WHERE job_key = ?", (now, key)
            )
        return []

    def _record_pending_alerts(self, snapshot: Snapshot, now: float) -> list[Alert]:
        """Warn about jobs that have been queued past the threshold."""
        alerts: list[Alert] = []
        for job in snapshot.jobs:
            if not job.is_pending:
                continue
            wait = job.queue_wait_sec(now)
            if wait is None or wait < self.pending_alert_sec:
                continue
            key = job.display_id
            row = self._get_job(key)
            if row is not None and row["alerted_pending"]:
                continue
            alert = self._raise_alert(
                key,
                now,
                kind="pending_long",
                severity="warning",
                message=(
                    f"{job.name or key} ({key}) has been queued for "
                    f"{wait // 3600}h{(wait % 3600) // 60:02d}m on {job.partition}"
                    + (f" ({job.reason_text})" if job.reason_text else "")
                ),
            )
            self._connection.execute(
                "UPDATE jobs SET alerted_pending = 1 WHERE job_key = ?", (key,)
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _record_limit_alerts(self, snapshot: Snapshot, now: float) -> list[Alert]:
        """Warn once when a running job is close to its wall-clock limit."""
        alerts: list[Alert] = []
        for job in snapshot.jobs:
            if not job.is_running:
                continue
            left = job.time_left_sec(now)
            if left is None or left > self.limit_soon_sec:
                continue
            if left < 0:
                continue  # Slack between the controller and us; not actionable.
            key = job.display_id
            row = self._get_job(key)
            if row is not None and row["alerted_limit"]:
                continue
            alert = self._raise_alert(
                key,
                now,
                kind="limit_soon",
                severity="warning",
                message=(
                    f"{job.name or key} ({key}) has {left // 60}m left of its "
                    f"time limit on {job.partition}"
                ),
            )
            self._connection.execute(
                "UPDATE jobs SET alerted_limit = 1 WHERE job_key = ?", (key,)
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    # ---------------------------------------------------------------- alerting

    def _raise_alert(
        self,
        job_key: str,
        now: float,
        *,
        kind: str,
        severity: str,
        message: str,
    ) -> Alert | None:
        """Insert an alert unless this job already raised this kind.

        Returns ``None`` when suppressed, so callers can concatenate the results
        without filtering.
        """
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO alerts (job_key, at, kind, severity, message)"
            " VALUES (?,?,?,?,?)",
            (job_key, now, kind, severity, message),
        )
        if cursor.rowcount == 0:
            return None
        return Alert(job_key=job_key, at=now, kind=kind, severity=severity, message=message)

    # ---------------------------------------------------------------- queries

    def _get_job(self, job_key: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM jobs WHERE job_key = ?", (job_key,)
        ).fetchone()

    def recent_alerts(self, limit: int = 50) -> list[Alert]:
        rows = self._connection.execute(
            "SELECT job_key, at, kind, severity, message FROM alerts"
            " ORDER BY at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            Alert(
                job_key=row["job_key"],
                at=row["at"],
                kind=row["kind"],
                severity=row["severity"],
                message=row["message"],
            )
            for row in rows
        ]

    def unacknowledged_alert_count(self, since: float) -> int:
        """Alerts raised after ``since``, used for the "new alerts" badge."""
        row = self._connection.execute(
            "SELECT COUNT(*) AS n FROM alerts WHERE at > ?", (since,)
        ).fetchone()
        return int(row["n"]) if row else 0

    def recent_transitions(self, limit: int = 100) -> list[dict]:
        rows = self._connection.execute(
            """
            SELECT t.job_key, t.at, t.from_state, t.to_state, t.reason, j.name,
                   j.partition
            FROM transitions t LEFT JOIN jobs j ON j.job_key = t.job_key
            ORDER BY t.at DESC, t.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def queue_wait_stats(self, since: float | None = None, limit: int = 0) -> list[QueueWaitStat]:
        """Observed queue wait per partition over jobs that actually started.

        Derived from ``start_time - submit_time``, which is the same definition the
        dashboard shows, so the statistics and the live column agree.
        """
        sql = (
            "SELECT partition, (start_time - submit_time) AS wait FROM jobs"
            " WHERE start_time IS NOT NULL AND submit_time IS NOT NULL"
            "   AND start_time >= submit_time"
        )
        params: tuple = ()
        if since is not None:
            sql += " AND last_seen >= ?"
            params = (since,)
        rows = self._connection.execute(sql, params).fetchall()

        grouped: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            grouped[row["partition"] or "?"].append(int(row["wait"]))

        stats: list[QueueWaitStat] = []
        for partition, waits in grouped.items():
            waits.sort()
            count = len(waits)
            # Median rather than mean: queue waits are long-tailed, and one 40-hour
            # outlier should not make a partition look uniformly slow.
            if count % 2:
                median = waits[count // 2]
            else:
                median = (waits[count // 2 - 1] + waits[count // 2]) // 2
            stats.append(
                QueueWaitStat(
                    partition=partition,
                    samples=count,
                    average_sec=sum(waits) / count,
                    median_sec=float(median),
                    max_sec=waits[-1],
                )
            )
        stats.sort(key=lambda s: s.max_sec, reverse=True)
        return stats[:limit] if limit and limit > 0 else stats

    def finished_jobs(self, limit: int = 50) -> list[dict]:
        """Locally recorded jobs that have ended, most recent first."""
        rows = self._connection.execute(
            """
            SELECT job_key, name, partition, last_state, final_state, final_exit,
                   submit_time, start_time, end_time, max_queue_wait, watched
            FROM jobs WHERE final_state IS NOT NULL
            ORDER BY COALESCE(end_time, last_seen) DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict:
        """Row counts, for `doctor` to show that history is actually accumulating."""
        result = {}
        for table in ("jobs", "transitions", "alerts"):
            row = self._connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            result[table] = int(row["n"]) if row else 0
        return result
