"""CLI wiring: subcommand routing, option position handling, and output modes.

The option-position tests exist because of a real argparse trap: a subparser parses
into its own namespace and copies it over the parent's, so ordinary defaults on
shared flags silently erase values given *before* the subcommand. The fix is
``argparse.SUPPRESS``, and these tests pin it.
"""

from __future__ import annotations

import json

import pytest
from fixture_data import payload as fixture_payload

from lampter import cli
from lampter.models import Snapshot
from lampter.store import HistoryStore


class StubTransport:
    def describe(self) -> str:
        return "stub-host"

    def fetch(self):  # pragma: no cover - only used if a test forgets to patch
        raise AssertionError("unexpected fetch")


def fixture_snapshot() -> Snapshot:
    raw = fixture_payload()
    return Snapshot.from_payload(
        raw,
        fetched_at=float(raw["generated_at"]),
        probe_elapsed_sec=0.5,
    )


def capture_command(monkeypatch, name: str) -> dict:
    """Replace a command function, recording the args and config it received."""
    seen: dict = {}

    def fake(args, config):
        seen["args"] = args
        seen["config"] = config
        return 0

    monkeypatch.setattr(cli, name, fake)
    # Keep the real environment out of the picture.
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    return seen


# ------------------------------------------------------------------ option position


def test_connection_flags_before_the_subcommand(monkeypatch):
    seen = capture_command(monkeypatch, "cmd_status")
    assert cli.main(["--host", "before", "--interval", "42", "status"]) == 0
    assert seen["config"].host == "before"
    assert seen["config"].refresh_interval == 42.0


def test_connection_flags_after_the_subcommand(monkeypatch):
    seen = capture_command(monkeypatch, "cmd_status")
    assert cli.main(["status", "--host", "after", "--interval", "42"]) == 0
    assert seen["config"].host == "after"
    assert seen["config"].refresh_interval == 42.0


def test_flags_work_in_both_positions_equivalently(monkeypatch):
    first = capture_command(monkeypatch, "cmd_status")
    cli.main(["--host", "h", "--user", "u", "status"])
    before = (first["config"].host, first["config"].user)

    second = capture_command(monkeypatch, "cmd_status")
    cli.main(["status", "--host", "h", "--user", "u"])
    after = (second["config"].host, second["config"].user)

    assert before == after == ("h", "u")


def test_flags_work_for_every_subcommand(monkeypatch):
    for command in ("status", "doctor", "tui"):
        target = {"status": "cmd_status", "doctor": "cmd_doctor", "tui": "cmd_tui"}[command]
        seen = capture_command(monkeypatch, target)
        assert cli.main([command, "--host", f"h-{command}"]) == 0
        assert seen["config"].host == f"h-{command}"


def test_no_batch_mode_flag(monkeypatch):
    seen = capture_command(monkeypatch, "cmd_status")
    cli.main(["status", "--no-batch-mode"])
    assert seen["config"].batch_mode is False

    seen = capture_command(monkeypatch, "cmd_status")
    cli.main(["status"])
    assert seen["config"].batch_mode is True


# ------------------------------------------------------------------ routing


def test_no_subcommand_launches_the_tui(monkeypatch):
    seen = capture_command(monkeypatch, "cmd_tui")
    assert cli.main([]) == 0
    # The default path must not blow up on the subcommand-only flags being absent.
    assert cli.opt(seen["args"], "limit", 0) == 0
    assert cli.opt(seen["args"], "sort", "wait") == "wait"
    assert cli.opt(seen["args"], "no_auto", False) is False


def test_opt_falls_back_for_missing_attributes():
    import argparse

    empty = argparse.Namespace()
    assert cli.opt(empty, "host") is None
    assert cli.opt(empty, "limit", 0) == 0


def test_unknown_flag_exits_with_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["status", "--definitely-not-a-flag"])
    assert excinfo.value.code == 2


def test_tui_accepts_no_auto(monkeypatch):
    seen = capture_command(monkeypatch, "cmd_tui")
    cli.main(["tui", "--no-auto", "--limit", "5", "--sort", "name"])
    assert cli.opt(seen["args"], "no_auto") is True
    assert cli.opt(seen["args"], "limit") == 5
    assert cli.opt(seen["args"], "sort") == "name"


# ------------------------------------------------------------------ status output


def test_status_prints_a_table(monkeypatch, capsys):
    snapshot = fixture_snapshot()
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)
    monkeypatch.setattr(cli, "make_transport", lambda config: StubTransport())
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    # A real job id from the fixture, whatever it happens to be this capture.
    assert snapshot.jobs[0].display_id in out
    assert "WAIT" in out
    counts = snapshot.counts()
    assert f"{counts['running']} running" in out


def test_status_json_is_machine_readable(monkeypatch, capsys):
    snapshot = fixture_snapshot()
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)
    monkeypatch.setattr(cli, "make_transport", lambda config: StubTransport())
    assert cli.main(["status", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["counts"] == snapshot.counts()
    assert document["slurm_version"] == "25.05.4"
    assert len(document["jobs"]) == len(snapshot.jobs)
    assert "queue_wait_sec" in document["jobs"][0]
    assert document["longest_wait_sec"] is not None
    # The other sections are all present in the machine-readable form.
    assert document["partitions"] and document["history"]
    # Derived from the fixture, so adding a collector does not break this.
    assert document["sections"] == list(snapshot.sections)
    # Live usage fields exist on every job, even when nothing was measured.
    for job in document["jobs"]:
        assert "memory_used_gb" in job
        assert "memory_limit_mb" in job
        assert "cpu_seconds" in job


def test_status_limit_truncates(monkeypatch, capsys):
    snapshot = fixture_snapshot()
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)
    monkeypatch.setattr(cli, "make_transport", lambda config: StubTransport())
    assert cli.main(["status", "--json", "--limit", "3"]) == 0
    document = json.loads(capsys.readouterr().out)
    # --limit only affects rendering, so the JSON stays complete.
    assert len(document["jobs"]) == len(snapshot.jobs)


def test_status_reports_transport_failure(monkeypatch, capsys):
    from lampter.transport import TransportError

    def boom(config):
        raise TransportError("ssh to torch failed: connection refused")

    monkeypatch.setattr(cli, "fetch_snapshot", boom)
    assert cli.main(["status"]) == cli.EXIT_FAILURE
    assert "connection refused" in capsys.readouterr().err


def test_status_shows_probe_errors_without_failing(monkeypatch, capsys):
    raw = fixture_payload()
    raw["errors"] = ["slurm warning: odd"]
    snapshot = Snapshot.from_payload(raw, fetched_at=0.0)
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)
    monkeypatch.setattr(cli, "make_transport", lambda config: StubTransport())
    assert cli.main(["status"]) == 0
    assert "slurm warning: odd" in capsys.readouterr().out


# ------------------------------------------------------------------ doctor


def test_doctor_reports_failure(monkeypatch, capsys):
    from lampter.transport import TransportError

    class Failing:
        def describe(self):
            return "torch"

        def fetch(self):
            raise TransportError("ssh to torch failed: no route to host")

    monkeypatch.setattr(cli, "make_transport", lambda config: Failing())
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    assert cli.main(["doctor"]) == cli.EXIT_FAILURE
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "no route to host" in out


def test_doctor_reports_success(monkeypatch, capsys):
    payload = fixture_payload()

    class Ok:
        def describe(self):
            return "torch"

        def fetch(self):
            from lampter.transport import ProbeResult

            return ProbeResult(payload=payload, elapsed_sec=0.42)

    monkeypatch.setattr(cli, "make_transport", lambda config: Ok())
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    assert cli.main(["doctor"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "25.05.4" in out
    assert f"jobs returned      {len(payload['jobs'])}" in out
    assert "no probe errors" in out
    # The polling cadence is part of what doctor explains.
    assert "capacity interval" in out


# ------------------------------------------------------------------ partitions


def _serve(monkeypatch, snapshot):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)
    monkeypatch.setattr(cli, "make_transport", lambda config: StubTransport())


def test_partitions_prints_capacity(monkeypatch, capsys):
    _serve(monkeypatch, fixture_snapshot())
    assert cli.main(["partitions"]) == 0
    out = capsys.readouterr().out
    assert "PARTITION" in out
    assert "GPUS free" in out
    # The overlap warning matters: these rows must not be summed.
    assert "overlap" in out


def test_partitions_json_reports_free_gpus_and_what_i_hold(monkeypatch, capsys):
    snapshot = fixture_snapshot()
    _serve(monkeypatch, snapshot)
    assert cli.main(["partitions", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["partitions"]
    assert document["held_gpus"] == snapshot.held_gpus_by_partition()
    assert document["my_partitions"] == list(snapshot.my_partitions)

    h200 = next(p for p in document["partitions"] if p["name"] == "h200")
    assert h200["gpus_total"] == h200["nodes_total"] * 8
    assert h200["gpus_free"] == h200["gpus_total"] - h200["gpus_used"]


def test_partitions_mine_filters_the_json_too(monkeypatch, capsys):
    snapshot = fixture_snapshot()
    _serve(monkeypatch, snapshot)
    assert cli.main(["partitions", "--mine", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert [p["name"] for p in document["partitions"]] == sorted(snapshot.my_partitions)
    assert len(document["partitions"]) < len(snapshot.partitions)


# ------------------------------------------------------------------ history


def test_history_prints_recent_outcomes(monkeypatch, capsys):
    snapshot = fixture_snapshot()
    _serve(monkeypatch, snapshot)
    assert cli.main(["history"]) == 0
    out = capsys.readouterr().out
    assert "finished in the last" in out
    assert "EXIT" in out
    assert snapshot.finished(1)[0].display_id in out


def test_history_json(monkeypatch, capsys):
    snapshot = fixture_snapshot()
    _serve(monkeypatch, snapshot)
    assert cli.main(["history", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["history_hours"] == 12
    assert document["finished"]
    assert document["finished"][0]["display_id"]
    assert "queue_wait_sec" in document["finished"][0]
    assert isinstance(document["failures"], list)


def test_history_reports_an_empty_window(monkeypatch, capsys):
    raw = fixture_payload()
    raw["history"] = []
    snapshot = Snapshot.from_payload(raw, fetched_at=float(raw["generated_at"]))
    _serve(monkeypatch, snapshot)
    assert cli.main(["history"]) == 0
    assert "no finished jobs" in capsys.readouterr().out


# ------------------------------------------------------------------ record keeping


def test_status_records_into_the_history_database(monkeypatch, capsys, tmp_path):
    """Plain one-shot checks must build history too, not just the dashboard."""
    database = tmp_path / "history.db"
    monkeypatch.setenv("LAMPTER_HISTORY_PATH", str(database))
    snapshot = fixture_snapshot()
    _serve(monkeypatch, snapshot)

    assert cli.main(["status", "--json"]) == 0
    capsys.readouterr()

    with HistoryStore(database) as store:
        stats = store.stats()
        assert stats["jobs"] >= len(snapshot.jobs)
        # One opening transition per live job, and nothing else yet.
        assert stats["transitions"] == len(snapshot.jobs)


def test_repeated_status_does_not_duplicate_transitions(monkeypatch, capsys, tmp_path):
    database = tmp_path / "history.db"
    monkeypatch.setenv("LAMPTER_HISTORY_PATH", str(database))
    snapshot = fixture_snapshot()
    _serve(monkeypatch, snapshot)

    for _ in range(3):
        assert cli.main(["status", "--json"]) == 0
        capsys.readouterr()

    with HistoryStore(database) as store:
        assert store.stats()["transitions"] == len(snapshot.jobs)


def test_no_history_flag_keeps_the_database_out_of_it(monkeypatch, capsys, tmp_path):
    database = tmp_path / "history.db"
    monkeypatch.setenv("LAMPTER_HISTORY_PATH", str(database))
    snapshot = fixture_snapshot()
    _serve(monkeypatch, snapshot)

    assert cli.main(["status", "--json", "--no-history"]) == 0
    capsys.readouterr()
    assert not database.exists()


# ------------------------------------------------------------------ alerts


def test_alerts_reads_the_local_database(monkeypatch, capsys, tmp_path):
    import time as _time

    from lampter.store import HistoryStore as _HistoryStore

    database = tmp_path / "history.db"
    monkeypatch.setenv("LAMPTER_HISTORY_PATH", str(database))
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())

    raw = fixture_payload()
    moment = _time.time()
    for record in raw["jobs"]:
        if record["state"] == "PENDING":
            record["submit_time"] = int(moment) - 30 * 3600
    snapshot = Snapshot.from_payload(raw, fetched_at=moment)
    with _HistoryStore(database, pending_alert_sec=3600) as store:
        assert store.record(snapshot, moment), "expected a starvation alert"

    assert cli.main(["alerts"]) == 0
    out = capsys.readouterr().out
    assert "queued for" in out
    assert "ALERT" in out


def test_alerts_without_history_explains_itself(monkeypatch, capsys):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    assert cli.main(["alerts", "--no-history"]) == cli.EXIT_FAILURE
    assert "disabled" in capsys.readouterr().err


def test_alerts_when_nothing_has_gone_wrong(monkeypatch, capsys):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    assert cli.main(["alerts"]) == 0
    assert "no alerts recorded" in capsys.readouterr().out
