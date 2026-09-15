"""The interactive terminal dashboard.

Design notes that matter for how this behaves in real use:

* **One refresh, one SSH round trip.** All fetching happens in
  :meth:`SlurmMonitorApp._fetch_worker`, a Textual *thread* worker, so a slow
  connection never freezes the UI -- you can keep scrolling and sorting the last
  snapshot while a refresh is in flight.
* **Stale beats blank.** A failed refresh keeps the previous snapshot on screen and
  reports the error in the notice line. Losing your whole dashboard because the VPN
  blipped for two seconds would be worse than showing slightly old numbers.
* **A 1-second tick drives everything.** Auto-refresh, the countdown and the
  "updated Ns ago" text all derive from one timer, which keeps them consistent and
  makes interval changes take effect immediately.
"""

from __future__ import annotations

import time
from typing import ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from ..duration import humanize_seconds
from ..models import SORT_MODES, Alert, Snapshot
from ..render import (
    VIEW_KEY_LABELS,
    VIEW_TITLES,
    VIEWS,
    WIDE_LAYOUT,
    Column,
    columns_for_view,
    summary_line,
    view_rows,
)
from ..store import HistoryStore, StoreError
from ..transport import Transport, TransportError
from .logview import LogScreen

#: Guards against a refresh interval so short that it becomes a self-inflicted
#: denial of service against the login node.
MIN_INTERVAL = 2.0
MAX_INTERVAL = 600.0


class SlurmMonitorApp(App[None]):
    """A live view of the monitored user's SLURM jobs."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #summary {
        height: auto;
        padding: 0 1;
    }
    #notice {
        height: auto;
        padding: 0 1;
    }
    #jobs {
        height: 1fr;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("r", "refresh_now", "Refresh", priority=True),
        Binding("a", "toggle_auto", "Auto"),
        Binding("plus,equals_sign", "interval_up", "Slower"),
        Binding("minus", "interval_down", "Faster"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("v", "toggle_reverse", "Reverse"),
        # One digit per view, generated from VIEWS. Hand-written digits are how the
        # accounts and qos views ended up unreachable while "3" still meant "history".
        *(
            Binding(
                str(index + 1),
                f"show_view('{view}')",
                VIEW_KEY_LABELS.get(view, view.title()),
            )
            for index, view in enumerate(VIEWS)
        ),
        Binding("tab", "cycle_view", "Next view", priority=True),
        Binding("l", "toggle_only_mine", "Mine only"),
        Binding("o", "open_log", "Log"),
        Binding("enter", "open_log", "Log", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        transport: Transport,
        *,
        interval: float = 15.0,
        auto: bool = True,
        sort_mode: str = "wait",
        reverse: bool = False,
        max_rows: int = 0,
        partition_interval: float = 60.0,
        history_interval: float = 120.0,
        usage_interval: float = 60.0,
        accounts_interval: float = 600.0,
        store: HistoryStore | None = None,
    ) -> None:
        super().__init__()
        self.transport = transport
        self.refresh_interval = interval
        #: Capacity and history are polled on their own, slower clocks. A shared
        #: cluster controller should not be asked for `sinfo` every 15 seconds for
        #: data that is nearly always identical.
        self.partition_interval = partition_interval
        self.history_interval = history_interval
        self.usage_interval = usage_interval
        self.accounts_interval = accounts_interval
        # NOTE: this must NOT be called `auto_refresh`. `DOMNode.auto_refresh` is a
        # Textual property whose setter creates a repaint timer via `set_interval`,
        # so assigning a bool to it would call `set_interval(True)` from __init__ --
        # which raises "no running event loop" outside a test harness, and leaves a
        # stray timer that errors during shutdown.
        self.auto_refresh_enabled = auto
        self.sort_mode = sort_mode
        self.sort_reverse = reverse
        self.max_rows = max_rows
        #: Optional local history database. Alerts are only raised when one exists.
        self.store = store
        #: Newest alerts raised by the store, newest first.
        self.alerts: list[Alert] = []

        self.snapshot: Snapshot | None = None
        #: Which of the three tables is on screen: jobs, partitions or history.
        self.view = "jobs"
        #: Restrict the capacity view to the partitions my own jobs occupy.
        self.only_mine = False
        #: Transport-level failure (ssh died). Distinct from the probe's own
        #: partial-failure list, which is reported separately and is not fatal.
        self.last_error: str | None = None
        #: Non-fatal history-store problem, reported on the notice line.
        self.store_error: str | None = None

        self._fetching = False
        self._next_refresh_at = time.monotonic() + interval
        #: When each slow collector was last fetched, or ``None`` for "never". The
        #: distinction matters: ``0.0`` looks like a natural "long ago" but these are
        #: compared against :func:`time.monotonic`, which counts seconds since **boot**.
        #: On a machine that booted less than ``accounts_interval`` ago -- a container, a
        #: CI runner, a laptop switched on five minutes ago -- nothing was ever "due", so
        #: the first refresh asked only for the job list and the capacity, history and
        #: accounts views opened empty. This was found by CI, whose runners boot seconds
        #: before the test runs.
        self._last_partitions_at: float | None = None
        self._last_history_at: float | None = None
        self._last_usage_at: float | None = None
        self._last_accounts_at: float | None = None
        #: Display id of the row under the cursor, so refreshes don't move it.
        self._selected_id: str | None = None
        self._row_ids: list[str] = []
        #: Active column layout; re-chosen whenever the terminal is resized.
        self._layout: tuple[Column, ...] = WIDE_LAYOUT

    # ---------------------------------------------------------------- polling plan

    def sections_due(self, now: float | None = None) -> tuple[str, ...]:
        """Which collectors this refresh should ask for.

        Always the job list, plus capacity and history only once their own intervals
        have elapsed. This is the whole point of the section mechanism: a 15-second
        refresh stays at one cheap ``squeue`` most of the time, and the expensive
        collectors run on the schedule they actually need.

        A collector that has never run is always due, whatever the clock says.
        """
        moment = time.monotonic() if now is None else now

        def due(last: float | None, interval: float) -> bool:
            return last is None or moment - last >= interval

        sections = ["jobs"]
        if due(self._last_partitions_at, self.partition_interval):
            sections.append("partitions")
        if due(self._last_history_at, self.history_interval):
            sections.append("history")
        if due(self._last_usage_at, self.usage_interval):
            sections.append("usage")
        if due(self._last_accounts_at, self.accounts_interval):
            sections.append("accounts")
        return tuple(sections)

    # ---------------------------------------------------------------- layout

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("connecting…", id="summary")
        yield Static("", id="notice")
        yield DataTable(id="jobs", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "lampter"
        self.sub_title = self.transport.describe()
        self._setup_columns()
        self.query_one(DataTable).focus()
        self.set_interval(1.0, self._tick)
        self._trigger_fetch()

    def on_resize(self, event) -> None:
        """Re-choose the column layout when the window changes width.

        ``event.size`` is used rather than ``self.size`` because the event carries
        the new geometry, whereas the widget's own size may not be updated until
        after layout has run.
        """
        if columns_for_view(self.view, event.size.width) is not self._layout:
            self._setup_columns(event.size.width)
            self._render_view()
            self._update_summary()

    def _sync_layout(self) -> bool:
        """Fall back to polling the width; returns ``True`` if the layout changed.

        Resize events normally arrive on their own, but a resize that does not
        produce one (terminal multiplexers, some remote terminals) would otherwise
        leave the table in the wrong shape indefinitely. The once-a-second tick is
        a cheap safety net.
        """
        if columns_for_view(self.view, self.size.width) is not self._layout:
            self._setup_columns()
            return True
        return False

    def _setup_columns(self, width: int | None = None) -> None:
        """(Re)build the table's columns for the current view and layout."""
        table = self.query_one(DataTable)
        resolved = width if width is not None else self.size.width
        self._layout = columns_for_view(self.view, resolved)
        table.clear(columns=True)
        for column in self._layout:
            table.add_column(column.header, key=column.key)
        # Column changes invalidate every row, so the cursor memory goes too.
        self._row_ids = []

    # ---------------------------------------------------------------- refresh cycle

    def _tick(self) -> None:
        """One-second heartbeat: refresh when due, then repaint the summary."""
        if self._sync_layout():
            self._render_view()
        if (
            self.auto_refresh_enabled
            and not self._fetching
            and time.monotonic() >= self._next_refresh_at
        ):
            self._trigger_fetch()
        self._update_summary()

    def _trigger_fetch(self) -> None:
        if self._fetching:
            return
        self._fetching = True
        now = time.monotonic()
        self._next_refresh_at = now + self.refresh_interval
        # Decide the collector set here, on the UI thread, so the wall-clock
        # schedule is not perturbed by how long the SSH call takes.
        sections = self.sections_due(now)
        self._update_summary()
        self._fetch_worker(sections)

    @work(thread=True, exclusive=True)
    def _fetch_worker(self, sections: tuple[str, ...]) -> None:
        """Runs off the UI thread; the SSH call can block for seconds."""
        try:
            result = self.transport.fetch(sections)
        except TransportError as exc:
            self.call_from_thread(self._on_fetch_error, str(exc))
            return
        except Exception as exc:  # defensive: never kill the app from a worker
            self.call_from_thread(self._on_fetch_error, f"{type(exc).__name__}: {exc}")
            return
        self.call_from_thread(self._on_fetch_ok, result, sections)

    def _on_fetch_ok(self, result, sections: tuple[str, ...]) -> None:
        self._fetching = False
        moment = time.time()
        self.last_error = None

        for section in sections:
            if section == "partitions":
                self._last_partitions_at = time.monotonic()
            elif section == "history":
                self._last_history_at = time.monotonic()
            elif section == "usage":
                self._last_usage_at = time.monotonic()
            elif section == "accounts":
                self._last_accounts_at = time.monotonic()

        fresh = Snapshot.from_payload(result.payload, moment, result.elapsed_sec)
        # Carry forward whatever this refresh did not ask for, so a jobs-only poll
        # never blanks the capacity or history views.
        self.snapshot = fresh.merged_with(self.snapshot)

        self._record_in_store(fresh)
        self._render_view()
        self._update_summary()

    def _record_in_store(self, snapshot: Snapshot) -> None:
        """Feed the local database and surface whatever it newly has to say."""
        if self.store is None:
            return
        try:
            raised = self.store.record(snapshot)
        except StoreError as exc:
            self.store_error = str(exc)
            return
        self.store_error = None
        if not raised:
            return
        self.alerts = (raised + self.alerts)[:50]
        for alert in raised:
            severity = "error" if alert.severity == "error" else "warning"
            self.notify(alert.message, severity=severity, timeout=10)

    def _on_fetch_error(self, message: str) -> None:
        self._fetching = False
        self.last_error = message
        # Deliberately keep whatever snapshot we already had on screen.
        self._update_summary()

    # ---------------------------------------------------------------- painting

    def _render_view(self) -> None:
        """Rebuild the table for the current view, preserving the cursor.

        The cursor is remembered by row *identity* (job id, partition name) rather
        than row index, because rows move as jobs are submitted, start and finish --
        and when switching views, the identities change entirely.
        """
        if self.snapshot is None:
            return
        table = self.query_one(DataTable)

        if self._row_ids:
            row = table.cursor_row
            if 0 <= row < len(self._row_ids):
                self._selected_id = self._row_ids[row]

        now = time.time()
        rows, keys = view_rows(
            self.snapshot,
            now,
            self.view,
            self._layout,
            sort_mode=self.sort_mode,
            reverse=self.sort_reverse,
            max_rows=self.max_rows,
            only_mine=self.only_mine,
        )

        table.clear()
        row_keys: list[str] = []
        seen: dict[str, int] = {}
        for key, cells in zip(keys, rows, strict=True):
            # Textual requires unique row keys. The same job id can legitimately
            # appear twice in the history view (a requeued job), so duplicates get
            # a suffix rather than raising.
            unique = key
            if unique in seen:
                seen[key] += 1
                unique = f"{key}#{seen[key]}"
            else:
                seen[key] = 0
            table.add_row(*cells, key=unique)
            row_keys.append(unique)
        self._row_ids = row_keys

        if self._selected_id is not None and self._selected_id in self._row_ids:
            table.move_cursor(row=self._row_ids.index(self._selected_id))
        elif self._row_ids:
            table.move_cursor(row=0)

    def _update_summary(self) -> None:
        summary = self.query_one("#summary", Static)
        notice = self.query_one("#notice", Static)

        if self.snapshot is None:
            line = Text(f"{self.transport.describe()} — waiting for first snapshot…", style="dim")
            summary.update(line)
        else:
            remaining = (
                self._next_refresh_at - time.monotonic() if self.auto_refresh_enabled else None
            )
            line = summary_line(
                self.snapshot,
                time.time(),
                target=self.transport.describe(),
                interval=self.refresh_interval,
                auto=self.auto_refresh_enabled,
                next_refresh_in=remaining,
            )
            arrow = "↓" if self.sort_reverse else "↑"
            line.append(f"  │  {self.view}", style="bold")
            if self.view == "jobs":
                line.append(f" sort {self.sort_mode} {arrow}", style="dim")
            elif self.view == "partitions" and self.only_mine:
                line.append(" mine only", style="dim")
            # Say how old the slower collectors are, rather than implying that a
            # jobs-only refresh also refreshed capacity.
            now = time.time()
            partitions_age = self.snapshot.partitions_age_sec(now)
            history_age = self.snapshot.history_age_sec(now)
            if partitions_age is not None:
                line.append(
                    f"  │  capacity {humanize_seconds(partitions_age)}", style="dim"
                )
            if history_age is not None:
                line.append(f" · history {humanize_seconds(history_age)}", style="dim")
            usage_age = self.snapshot.usage_age_sec(now)
            if usage_age is not None:
                line.append(f" · usage {humanize_seconds(usage_age)}", style="dim")
            accounts_age = self.snapshot.accounts_age_sec(now)
            if accounts_age is not None:
                line.append(f" · accts {humanize_seconds(accounts_age)}", style="dim")
            if self.alerts:
                line.append(f"  │  {len(self.alerts)} alert", style="bold yellow")
                if len(self.alerts) > 1:
                    line.append("s", style="bold yellow")
            summary.update(line)

        parts: list[Text] = []
        if self.last_error:
            parts.append(Text(f"⚠ {self.last_error}", style="bold red"))
        elif self._fetching:
            parts.append(Text("refreshing…", style="dim"))
        if self.store_error:
            parts.append(Text(f"history db: {self.store_error}", style="yellow"))
        if self.snapshot is not None and self.snapshot.errors:
            parts.append(Text("probe: " + "; ".join(self.snapshot.errors), style="yellow"))
        if self.alerts:
            latest = self.alerts[0]
            style = "bold red" if latest.severity == "error" else "yellow"
            parts.append(Text(f"● {latest.message}", style=style))
        notice.update(Text("\n").join(parts) if parts else Text(""))

    # ---------------------------------------------------------------- actions

    def action_refresh_now(self) -> None:
        if self._fetching:
            self.notify("already refreshing", severity="warning", timeout=2)
            return
        self._trigger_fetch()

    def action_toggle_auto(self) -> None:
        self.auto_refresh_enabled = not self.auto_refresh_enabled
        if self.auto_refresh_enabled:
            self._next_refresh_at = time.monotonic() + self.refresh_interval
        state = "on" if self.auto_refresh_enabled else "off"
        self.notify(f"auto-refresh {state}", timeout=2)
        self._update_summary()

    def action_interval_up(self) -> None:
        self._set_interval(self.refresh_interval * 1.5)

    def action_interval_down(self) -> None:
        self._set_interval(self.refresh_interval / 1.5)

    def _set_interval(self, value: float) -> None:
        self.refresh_interval = max(MIN_INTERVAL, min(MAX_INTERVAL, value))
        self._next_refresh_at = time.monotonic() + self.refresh_interval
        self.notify(
            f"refresh every {humanize_seconds(self.refresh_interval)} "
            f"({self.refresh_interval:.0f}s)",
            timeout=2,
        )
        self._update_summary()

    def action_cycle_sort(self) -> None:
        modes = [mode for mode, _ in SORT_MODES]
        index = modes.index(self.sort_mode) if self.sort_mode in modes else 0
        self.sort_mode = modes[(index + 1) % len(modes)]
        self._render_view()
        self._update_summary()
        self.notify(f"sort by {self.sort_mode}", timeout=2)

    def action_toggle_reverse(self) -> None:
        self.sort_reverse = not self.sort_reverse
        self._render_view()
        self._update_summary()
        self.notify("reversed" if self.sort_reverse else "forward", timeout=2)

    # ---------------------------------------------------------------- views

    def action_show_view(self, view: str) -> None:
        """Switch views. The digit keys are bound to this with a view name."""
        self._set_view(view)

    def action_cycle_view(self) -> None:
        index = VIEWS.index(self.view) if self.view in VIEWS else 0
        self._set_view(VIEWS[(index + 1) % len(VIEWS)])

    def action_toggle_only_mine(self) -> None:
        """Restrict the capacity view to the partitions my own jobs occupy."""
        self.only_mine = not self.only_mine
        if self.view != "partitions":
            self._set_view("partitions")
        else:
            self._render_view()
            self._update_summary()
        self.notify(
            "showing only my partitions" if self.only_mine else "showing all partitions",
            timeout=2,
        )

    def _set_view(self, view: str) -> None:
        if view == self.view:
            return
        self.view = view
        # Row identities differ between views, so the remembered cursor is meaningless.
        self._selected_id = None
        self._setup_columns()
        self._render_view()
        self._update_summary()
        self.notify(VIEW_TITLES.get(view, view), timeout=2)

    # ---------------------------------------------------------------- logs

    def action_open_log(self) -> None:
        """Follow the selected job's output file.

        Only the jobs view has log paths: ``squeue`` reports them, ``sacct`` does not.
        """
        if self.snapshot is None or not self._row_ids:
            return
        row = self.query_one(DataTable).cursor_row
        if not (0 <= row < len(self._row_ids)):
            return

        # Row keys can carry a "#n" suffix when a display id repeats.
        key = self._row_ids[row].split("#", 1)[0]
        job = self.snapshot.by_id(key)
        if job is None:
            self.notify(
                "log files are only known for jobs currently in the queue",
                severity="warning",
                timeout=3,
            )
            return
        path = job.log_path
        if not path:
            self.notify(
                f"{key} has no log file recorded (no --output in its sbatch script)",
                severity="warning",
                timeout=4,
            )
            return

        self.push_screen(
            LogScreen(
                self.transport,
                path,
                header=f"{job.display_id} {job.name}",
                # Following a finished job would hang on a static file.
                follow=job.is_running,
                lines=300,
            )
        )
