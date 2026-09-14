"""Interactive dashboard tests.

The TUI is the main deliverable, so it gets driven for real through Textual's
``run_test`` harness rather than left to manual inspection: the app is mounted at a
known size, a stub transport stands in for SSH, and keys are actually pressed.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable, Static

from lampter.render import COMPACT_LAYOUT, WIDE_LAYOUT, WIDE_LAYOUT_MIN_WIDTH
from lampter.store import HistoryStore, StoreError
from lampter.transport import ProbeResult, TransportError
from lampter.ui import SlurmMonitorApp

FIXTURE = Path(__file__).parent / "fixtures" / "squeue_payload.json"


def expected_row_count() -> int:
    """How many jobs the current fixture holds, whatever it happens to be."""
    return len(json.loads(FIXTURE.read_text())["jobs"])


def expected_counts() -> dict:
    """State groups the fixture holds, derived rather than hard-coded."""
    from lampter.models import Snapshot

    raw = json.loads(FIXTURE.read_text())
    return Snapshot.from_payload(raw, fetched_at=0.0).counts()


class StubTransport:
    """Stands in for :class:`SSHTransport` without touching the network."""

    def __init__(self, payload: dict | None = None, error: str | None = None) -> None:
        self.payload = payload if payload is not None else make_payload()
        self.error = error
        self.calls = 0
        #: Section sets requested, so tests can assert on the polling cadence.
        self.section_requests: list[tuple[str, ...]] = []

    def describe(self) -> str:
        return "stub-host"

    def fetch(self, sections=None) -> ProbeResult:
        self.calls += 1
        self.section_requests.append(tuple(sections) if sections else ())
        if self.error:
            raise TransportError(self.error)
        return ProbeResult(payload=self.payload, elapsed_sec=0.02)


def make_payload() -> dict:
    payload = json.loads(FIXTURE.read_text())
    # Re-stamp it so "updated Ns ago" reflects a fresh refresh.
    payload["generated_at"] = int(time.time())
    return payload


def run(coro):
    return asyncio.run(coro)


def test_app_renders_every_job():
    async def scenario():
        transport = StubTransport()
        app = SlurmMonitorApp(transport, interval=60)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()
            for _ in range(20):
                if app.query_one(DataTable).row_count:
                    break
                await pilot.pause()
            table = app.query_one(DataTable)
            assert table.row_count == expected_row_count()
            assert app.snapshot is not None
            assert app.snapshot.counts() == expected_counts()
            assert transport.calls == 1
            # The queue wait of a running job is exact: start minus submit.
            job = app.snapshot.by_id("17325640_43")
            assert job is not None
            assert job.queue_wait_sec(time.time()) == job.start_time - job.submit_time

    run(scenario())


def wide_size() -> tuple[int, int]:
    """A terminal big enough for the wide layout, derived from its own threshold."""
    return (WIDE_LAYOUT_MIN_WIDTH + 20, 40)


def test_wide_terminal_uses_the_wide_layout():
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=60)
        async with app.run_test(size=wide_size()):
            assert app._layout is WIDE_LAYOUT
            assert len(app.query_one(DataTable).columns) == len(WIDE_LAYOUT)

    run(scenario())


def test_narrow_terminal_switches_to_the_compact_layout():
    """A cramped window must keep WAIT on screen rather than collapse columns."""

    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=60)
        async with app.run_test(size=(80, 24)):
            assert app._layout is COMPACT_LAYOUT
            headers = [column.header for column in app._layout]
            assert "WAIT" in headers
            # The table itself was rebuilt to match the chosen layout.
            rendered = [str(column.label) for column in app.query_one(DataTable).columns.values()]
            assert rendered == headers

    run(scenario())


def test_manual_refresh_key_triggers_another_fetch():
    async def scenario():
        transport = StubTransport()
        app = SlurmMonitorApp(transport, interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()
            for _ in range(20):
                if transport.calls:
                    break
                await pilot.pause()
            assert transport.calls == 1

            await pilot.press("r")
            for _ in range(20):
                if transport.calls >= 2:
                    break
                await pilot.pause()
            assert transport.calls == 2

    run(scenario())


def test_auto_refresh_can_be_toggled():
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()
            assert app.auto_refresh_enabled is True
            await pilot.press("a")
            await pilot.pause()
            assert app.auto_refresh_enabled is False
            await pilot.press("a")
            await pilot.pause()
            assert app.auto_refresh_enabled is True

    run(scenario())


def test_construction_leaves_textual_alone():
    """Regression guard for a launch-crashing name collision.

    ``DOMNode.auto_refresh`` is a Textual property whose setter calls
    ``set_interval``. Assigning our own boolean flag to that name made ``__init__``
    touch the event loop, which raised "no running event loop" when the real CLI
    built the app in a plain sync function -- while passing under the test
    harness's loop. So: constructing an app must never need a loop, and Textual's
    own property must stay untouched.
    """
    app = SlurmMonitorApp(StubTransport(), interval=30)
    assert app.auto_refresh_enabled is True
    # Textual's property still reports "no repaint timer", i.e. we did not hijack it.
    assert app.auto_refresh is None

    disabled = SlurmMonitorApp(StubTransport(), interval=30, auto=False)
    assert disabled.auto_refresh_enabled is False
    assert disabled.auto_refresh is None


def test_app_runs_outside_the_test_harness():
    """Construct and run the app the way the CLI does, then exit.

    ``run_test`` supplies a running event loop, which hid the bug above; a real
    ``App.run()`` does not. This exercises that path end to end.
    """
    app = SlurmMonitorApp(StubTransport(), interval=600)

    def stop_soon() -> None:
        time.sleep(1.5)
        app.exit()

    threading.Thread(target=stop_soon, daemon=True).start()
    app.run(headless=True)  # must not raise


def test_interval_keys_adjust_and_clamp():
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=10.0)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()
            await pilot.press("plus")
            await pilot.pause()
            assert app.refresh_interval == pytest.approx(15.0)
            await pilot.press("minus")
            await pilot.pause()
            assert app.refresh_interval == pytest.approx(10.0)

            # 10 -> 6.67 -> 4.44 -> 2.96 -> 1.98, which clamps to the 2s floor.
            for _ in range(5):
                await pilot.press("minus")
                await pilot.pause()
            assert app.refresh_interval == 2.0

    run(scenario())


def test_sort_key_cycles_through_modes():
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()
            assert app.sort_mode == "wait"
            await pilot.press("s")
            await pilot.pause()
            assert app.sort_mode == "state"
            await pilot.press("v")
            await pilot.pause()
            assert app.sort_reverse is True

    run(scenario())


def test_cursor_stays_on_the_same_job_across_refresh():
    """Refreshes must not yank the cursor to the top while you are reading."""

    async def scenario():
        transport = StubTransport()
        app = SlurmMonitorApp(transport, interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()
            for _ in range(20):
                if app.query_one(DataTable).row_count:
                    break
                await pilot.pause()

            table = app.query_one(DataTable)
            table.move_cursor(row=3)
            await pilot.pause()
            selected = app._row_ids[3]

            await pilot.press("r")
            for _ in range(20):
                if transport.calls >= 2:
                    break
                await pilot.pause()

            assert app._row_ids[app.query_one(DataTable).cursor_row] == selected

    run(scenario())


def test_failed_refresh_keeps_the_last_snapshot_and_reports_the_error():
    """Stale data beats a blank screen: the old rows stay, the error is shown."""

    async def scenario():
        transport = StubTransport(error="ssh to torch failed: connection refused")
        app = SlurmMonitorApp(transport, interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()
            for _ in range(20):
                if app.last_error:
                    break
                await pilot.pause()

            assert app.last_error is not None
            assert "connection refused" in app.last_error
            assert app.snapshot is None
            notice = app.query_one("#notice", Static)
            assert "connection refused" in str(notice.render())

    run(scenario())


def test_snapshot_error_line_is_shown_without_treating_it_as_fatal():
    async def scenario():
        payload = make_payload()
        payload["errors"] = ["slurm warning: something odd"]
        app = SlurmMonitorApp(StubTransport(payload), interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()
            for _ in range(20):
                if app.snapshot is not None:
                    break
                await pilot.pause()
            assert app.last_error is None
            assert app.snapshot is not None
            assert app.snapshot.errors == ("slurm warning: something odd",)
            # Rows are still rendered despite the probe complaint.
            assert app.query_one(DataTable).row_count == expected_row_count()

    run(scenario())


def test_snapshot_survives_a_resize_between_layouts():
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            for _ in range(20):
                if app.query_one(DataTable).row_count:
                    break
                await pilot.pause()
            assert app._layout is COMPACT_LAYOUT
            assert app.query_one(DataTable).row_count == expected_row_count()

            await pilot.resize_terminal(*wide_size())
            await pilot.pause()
            assert app._layout is WIDE_LAYOUT
            # Rows are rebuilt for the new column set, not lost.
            assert app.query_one(DataTable).row_count == expected_row_count()

    run(scenario())


# ------------------------------------------------------------------ polling cadence


def test_sections_due_staggers_the_slow_collectors():
    """The job list polls often; capacity and history poll on their own clocks.

    This is the load-limiting behaviour: without it a 15-second refresh would ask a
    shared controller for sinfo 240 times an hour.
    """
    app = SlurmMonitorApp(
        StubTransport(),
        interval=15,
        partition_interval=60,
        history_interval=120,
        usage_interval=45,
        accounts_interval=200,
    )
    app._last_partitions_at = 1000.0
    app._last_history_at = 1000.0
    app._last_usage_at = 1000.0
    app._last_accounts_at = 1000.0

    assert app.sections_due(1000.0) == ("jobs",)
    # Usage comes due first, at 45s, before capacity at 60s.
    assert app.sections_due(1044.0) == ("jobs",)
    assert app.sections_due(1045.0) == ("jobs", "usage")
    assert app.sections_due(1060.0) == ("jobs", "partitions", "usage")
    # Accounts is the slowest clock, so it is still not due at 1120s.
    assert app.sections_due(1120.0) == ("jobs", "partitions", "history", "usage")
    assert app.sections_due(1200.0) == (
        "jobs",
        "partitions",
        "history",
        "usage",
        "accounts",
    )


def test_first_refresh_asks_for_everything():
    async def scenario():
        transport = StubTransport()
        app = SlurmMonitorApp(transport, interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            for _ in range(30):
                if transport.section_requests:
                    break
                await pilot.pause()
            from lampter.remote_probe import DEFAULT_SECTIONS

            assert transport.section_requests[0] == tuple(DEFAULT_SECTIONS)

    run(scenario())


def test_jobs_only_refresh_keeps_the_previous_capacity_view():
    """A poll that skips sinfo must not blank the partition data."""

    async def scenario():
        transport = StubTransport()
        app = SlurmMonitorApp(transport, interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            for _ in range(30):
                if app.snapshot is not None and app.snapshot.partitions:
                    break
                await pilot.pause()
            assert app.snapshot is not None
            partitions = app.snapshot.partitions
            assert partitions

            # Force a jobs-only poll and let it land.
            app.partition_interval = 10_000
            app.history_interval = 10_000
            before = transport.calls
            await pilot.press("r")
            for _ in range(30):
                if transport.calls > before and not app._fetching:
                    break
                await pilot.pause()

            assert app.snapshot is not None
            assert app.snapshot.partitions == partitions
            # ...and the staleness is reported rather than hidden.
            assert app.snapshot.partitions_age_sec(time.time()) is not None

    run(scenario())


# ------------------------------------------------------------------ alerts


def test_store_alerts_are_surfaced(tmp_path):
    async def scenario():
        payload = make_payload()
        moment = int(time.time())
        for record in payload["jobs"]:
            if record["state"] == "PENDING":
                record["submit_time"] = moment - 30 * 3600

        store = HistoryStore(tmp_path / "history.db", pending_alert_sec=3600)
        try:
            app = SlurmMonitorApp(StubTransport(payload), interval=600, store=store)
            async with app.run_test(size=(200, 40)) as pilot:
                for _ in range(40):
                    if app.alerts:
                        break
                    await pilot.pause()
                assert app.alerts
                assert app.alerts[0].kind == "pending_long"
                assert app.store_error is None
                notice = app.query_one("#notice", Static)
                assert "queued for" in str(notice.render())
        finally:
            store.close()

    run(scenario())


def test_broken_store_is_reported_but_not_fatal():
    """A bad database must not cost you the live view."""

    class Exploding:
        def record(self, snapshot, now=None):
            raise StoreError("disk on fire")

    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600, store=Exploding())
        async with app.run_test(size=(200, 40)) as pilot:
            for _ in range(30):
                if app.store_error:
                    break
                await pilot.pause()
            assert app.store_error is not None
            assert app.last_error is None
            # Live data still rendered.
            assert app.query_one(DataTable).row_count == expected_row_count()
            assert "disk on fire" in str(app.query_one("#notice", Static).render())

    run(scenario())


def test_no_store_means_no_alerts_and_no_error():
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600, store=None)
        async with app.run_test(size=(200, 40)) as pilot:
            for _ in range(30):
                if app.snapshot is not None:
                    break
                await pilot.pause()
            assert app.alerts == []
            assert app.store_error is None

    run(scenario())


# ------------------------------------------------------------------ views


def _headers(app) -> list[str]:
    return [str(column.label) for column in app.query_one(DataTable).columns.values()]


def test_every_view_has_a_digit_key():
    """Derived from VIEWS, so a new view cannot be left unreachable.

    Hard-coding this is exactly how the accounts and qos views shipped with no key
    bound to them while "3" still selected history.
    """
    from textual.binding import Binding

    from lampter.render import VIEWS

    keys = {
        binding.key: binding.action
        for binding in SlurmMonitorApp.BINDINGS
        if isinstance(binding, Binding) and binding.key.isdigit()
    }
    assert len(keys) == len(VIEWS)
    for index, view in enumerate(VIEWS):
        assert keys[str(index + 1)] == f"show_view('{view}')"


def test_number_keys_select_the_matching_view():
    async def scenario():
        from lampter.render import VIEWS

        app = SlurmMonitorApp(StubTransport(), interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            for _ in range(30):
                if app.snapshot is not None:
                    break
                await pilot.pause()

            for index, view in enumerate(VIEWS):
                await pilot.press(str(index + 1))
                await pilot.pause()
                assert app.view == view, f"key {index + 1} should select {view}"

            # And each view actually renders its own rows.
            await pilot.press("1")
            await pilot.pause()
            assert "WAIT" in _headers(app)
            assert app.query_one(DataTable).row_count == expected_row_count()

    run(scenario())


def test_tab_cycles_through_every_view():
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            for _ in range(30):
                if app.snapshot is not None:
                    break
                await pilot.pause()
            from lampter.render import VIEWS

            seen = [app.view]
            for _ in range(len(VIEWS)):
                await pilot.press("tab")
                await pilot.pause()
                seen.append(app.view)
            # One full cycle, back to where it started.
            assert seen == [*VIEWS, VIEWS[0]]

    run(scenario())


def test_mine_key_restricts_the_capacity_view():
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            for _ in range(30):
                if app.snapshot is not None:
                    break
                await pilot.pause()

            await pilot.press("l")
            await pilot.pause()
            assert app.view == "partitions"
            assert app.only_mine is True
            assert app.query_one(DataTable).row_count == len(
                app.snapshot.partitions_for_my_jobs()
            )

            await pilot.press("l")
            await pilot.pause()
            assert app.only_mine is False
            assert app.query_one(DataTable).row_count == len(app.snapshot.partitions)

    run(scenario())


def test_view_switch_survives_before_the_first_snapshot():
    async def scenario():
        # A payload with no partitions or history at all: the views must still be
        # switchable, because the first fetch may not have landed yet.
        app = SlurmMonitorApp(
            StubTransport(payload={"schema": 3, "sections": ["jobs"]}),
            interval=600,
        )
        async with app.run_test(size=(200, 40)) as pilot:
            from lampter.render import VIEWS

            # Switch immediately, while the first fetch may still be in flight.
            for index, view in enumerate(VIEWS[:4], start=1):
                await pilot.press(str(index))
                await pilot.pause()
                assert app.view == view

    run(scenario())


def test_partition_view_re_lays_out_on_resize():
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600)
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(30):
                if app.snapshot is not None:
                    break
                await pilot.pause()
            await pilot.press("2")
            await pilot.pause()
            narrow = _headers(app)
            assert "NODES idle/tot" not in narrow

            await pilot.resize_terminal(200, 40)
            await pilot.pause()
            assert "NODES idle/tot" in _headers(app)

    run(scenario())


def test_sort_keys_only_affect_the_jobs_view():
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            for _ in range(30):
                if app.snapshot is not None:
                    break
                await pilot.pause()
            await pilot.press("2")
            await pilot.pause()
            # Cycling the sort while on the capacity view must not crash or rebuild
            # the partition rows into job rows.
            await pilot.press("s")
            await pilot.pause()
            assert app.view == "partitions"
            assert app.query_one(DataTable).row_count == len(app.snapshot.partitions)

    run(scenario())


def test_account_column_is_shown_in_the_jobs_view():
    """Which account a job was submitted under is a primary question, not a detail."""
    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            for _ in range(30):
                if app.snapshot is not None:
                    break
                await pilot.pause()

            headers = _headers(app)
            assert "ACCOUNT" in headers
            index = headers.index("ACCOUNT")

            # Textual sizes columns to their content, so the full name is on screen
            # rather than an ellipsis -- which matters, because two accounts can
            # differ only in the middle.
            accounts = {job.account for job in app.snapshot.jobs if job.account}
            assert accounts
            rendered = {
                str(app.query_one(DataTable).get_row_at(row)[index].plain)
                for row in range(app.query_one(DataTable).row_count)
            }
            assert accounts <= rendered

    run(scenario())
