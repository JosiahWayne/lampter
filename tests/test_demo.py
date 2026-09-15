"""Demo mode: the dashboard driven from the bundled dataset.

The point of ``--demo`` is that it is the *same* app, renderer and polling code with a
different data source, so the tests that matter are the ones that pin the boundary:

* it must reach no cluster and no network at all, whatever else happens;
* it must not write to the user's real history database, because sample data raising
  "queued for 22h" alerts in the middle of real ones would be worse than useless;
* it must not be switchable on from a config file, where it could silently shadow a
  live cluster;
* and the payload has to keep parsing, since it is also the test fixture.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fixture_data import FIXTURE_PATH

from lampter import cli
from lampter.demo import DEMO_PAYLOAD_PATH, DemoTransport, load_demo_payload, rebase_times
from lampter.models import Snapshot
from lampter.transport import SSHTransport, TransportError

#: A `--config` pointing at nothing, so a developer's real config cannot leak into
#: these tests through the normal search path.
NO_CONFIG = ("--config", "/nonexistent/lampter.toml")


def run_cli(argv: list[str], capsys) -> tuple[int, str, str]:
    code = cli.main([*argv, *NO_CONFIG])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ------------------------------------------------------------------ the boundary


def test_demo_needs_no_cluster(monkeypatch, capsys):
    """The whole promise: no SSH, no subprocess, no network.

    `subprocess.run` is what the real transport shells out through, so making it fatal
    is a stronger check than asserting a flag was set.
    """
    import subprocess

    def explode(*args, **kwargs):  # pragma: no cover - it must never be called
        raise AssertionError(f"demo mode shelled out: {args!r}")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(subprocess, "Popen", explode)

    code, out, _ = run_cli(["status", "--demo"], capsys)

    assert code == cli.EXIT_OK
    assert "train_gen" in out
    assert "GPU UTIL" in out


def test_demo_is_not_an_option_on_the_real_transport(monkeypatch, capsys):
    """Only `--demo` selects the dataset; nothing else can."""
    from lampter.config import Config

    assert isinstance(cli.make_transport(Config()), SSHTransport)
    assert isinstance(cli.make_transport(Config(demo=True)), DemoTransport)


def test_demo_cannot_be_switched_on_from_a_config_file(tmp_path, monkeypatch, capsys):
    """A config key that shadowed the cluster would be the worst failure mode here.

    The real transport is replaced with one that refuses, which proves the SSH path was
    chosen without any test touching the network.
    """
    config_file = tmp_path / "lampter.toml"
    config_file.write_text('host = "torch"\ndemo = true\n')

    class Refuses:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def describe(self) -> str:
            return "stub"

        def fetch(self, sections=None):
            raise TransportError("stub refuses: the SSH transport was selected")

    monkeypatch.setattr(cli, "SSHTransport", Refuses)

    code = cli.main(["status", "--config", str(config_file), "--json"])
    captured = capsys.readouterr()

    assert code == cli.EXIT_FAILURE
    assert "the SSH transport was selected" in captured.err


def test_demo_is_not_a_config_key_at_all(tmp_path):
    """It is rejected as unknown, rather than silently accepted and ignored."""
    from lampter.config import load_config

    config_file = tmp_path / "lampter.toml"
    config_file.write_text("demo = true\n")

    config = load_config(path=config_file, env={})

    assert config.demo is False
    assert any("unknown config key 'demo'" in warning for warning in config.warnings)


def test_demo_writes_nothing_to_disk(tmp_path, monkeypatch, capsys):
    """No history database, in either the default or the configured location.

    `doctor` used to open the store directly, so this is checked for both commands.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LAMPTER_HISTORY_PATH", raising=False)
    configured = tmp_path / "configured.db"

    assert cli.main(["--demo", "status", "--config", "/nonexistent/x.toml"]) == cli.EXIT_OK
    assert cli.main(["--demo", "doctor", "--config", "/nonexistent/x.toml"]) == cli.EXIT_OK
    capsys.readouterr()

    assert not (home / ".local" / "share" / "lampter" / "history.db").exists()
    assert not configured.exists()
    assert list(home.rglob("*.db")) == []


def test_demo_keeps_no_history_so_it_cannot_raise_alerts(capsys):
    code, _, err = run_cli(["alerts", "--demo"], capsys)
    assert code == cli.EXIT_FAILURE
    # The message must name the real cause: "history is disabled" would send the reader
    # looking for a setting that is not the problem.
    assert "demo mode keeps no history" in err


def test_demo_doctor_says_it_is_demo(capsys):
    """A diagnostic command that hides its data source is worse than no diagnostic."""
    code, out, _ = run_cli(["doctor", "--demo"], capsys)
    assert code == cli.EXIT_OK
    assert "demo" in out
    assert "demo mode writes nothing" in out
    assert "loading sample data" in out
    # The connection settings are not in play, so they must not be presented as if they
    # were the live configuration.
    assert "ssh host" not in out


def test_demo_log_view_refuses_rather_than_inventing_output(capsys):
    code, _, err = run_cli(["logs", "17365703", "--demo"], capsys)
    assert code == cli.EXIT_FAILURE
    assert "needs a real cluster" in err


# ------------------------------------------------------------------ the rebasing


def test_rebase_moves_the_capture_moment_to_now():
    payload = load_demo_payload()
    now = time.time()
    assert payload["generated_at"] == pytest.approx(now, abs=5)
    # The human-readable stamp travels with it, so `doctor` does not print a date from
    # whenever the payload was captured.
    assert payload["generated_at_iso"].startswith(time.strftime("%Y-%m-%d", time.localtime(now)))


def test_rebase_preserves_the_gaps_between_timestamps():
    """Anchoring on the newest timestamp instead of the capture moment broke this.

    The furthest-out timestamp is normally a running job's time limit, so pinning *that*
    to now made every long job claim zero time left. Anchoring on `generated_at` keeps
    every relationship in the file intact.
    """
    original = json.loads(FIXTURE_PATH.read_text())
    original_at = original["generated_at"]
    now = float(original_at + 10 * 365 * 86400)  # a decade later

    rebased = rebase_times(original, now)

    by_id = {job["job_id"]: job for job in original["jobs"]}
    shifted = {job["job_id"]: job for job in rebased["jobs"]}
    for job_id, before in by_id.items():
        after = shifted[job_id]
        for field in ("submit_time", "start_time", "end_time"):
            if not before.get(field):
                continue
            gap_before = before[field] - original_at
            gap_after = after[field] - rebased["generated_at"]
            assert gap_after == pytest.approx(gap_before, abs=1), (job_id, field)

    # Concretely: the long-running CPU job kept the ~6h of its limit that it had left,
    # rather than collapsing to "0m left".
    running = [
        job
        for job in rebased["jobs"]
        if job["state"] == "RUNNING" and job.get("end_time")
    ]
    assert running, "the fixture should still contain a job with a time limit"
    for job in running:
        left = job["end_time"] - rebased["generated_at"]
        assert left > 60, f"{job['name']} reports {left}s left of its limit"


def test_rebase_does_not_mutate_its_input():
    """The payload is read fresh each refresh; a shared mutation would drift."""
    original = json.loads(FIXTURE_PATH.read_text())
    before = original["jobs"][0]["submit_time"]
    rebase_times(original, time.time() + 10**6)
    assert original["jobs"][0]["submit_time"] == before


def test_rebase_survives_a_payload_with_no_timestamps():
    """Degenerate input must not raise: a hand-built payload may have none."""
    payload = rebase_times({"jobs": [{"job_id": 1}], "history": []}, 1000.0)
    assert payload["generated_at"] == 1000


# ------------------------------------------------------------------ the dataset


def test_the_dataset_is_the_shipped_one():
    assert DEMO_PAYLOAD_PATH.name == "demo_payload.json"
    assert DEMO_PAYLOAD_PATH.parent.name == "lampter", "must ship inside the package"
    assert FIXTURE_PATH == DEMO_PAYLOAD_PATH, "the tests and the demo share one file"


def test_the_dataset_covers_every_view():
    payload = load_demo_payload()
    for section in ("jobs", "partitions", "history", "usage", "accounts"):
        assert payload.get(section), f"demo mode would show an empty {section} view"
    assert payload.get("qos"), "the QOS view would be empty"
    assert set(payload["sections"]) >= {"jobs", "partitions", "history", "usage", "accounts"}


def test_the_dataset_parses_into_a_snapshot_that_looks_live():
    snapshot = Snapshot.from_payload(load_demo_payload(), time.time())
    assert snapshot.jobs and snapshot.partitions and snapshot.finished()
    assert snapshot.accounts and snapshot.qos_pressure
    # Queue waits are relative to now, so rebasing is what keeps them believable.
    longest = snapshot.longest_wait(time.time())
    assert longest is not None and 0 < longest < 7 * 86400


def test_the_dataset_exercises_the_gpu_column():
    """A demo that showed one boring value would not document the column."""
    payload = load_demo_payload()
    utils = [entry.get("gpu_util") for entry in payload["usage"]]
    assert any(value is None for value in utils), "a CPU-only job must show '-'"
    present = [value for value in utils if value is not None]
    assert len(present) >= 3
    assert min(present) < 60, "nothing would render below the cancellation line"
    assert max(present) >= 75, "nothing would render as healthy"


def test_demo_transport_honours_the_requested_sections():
    """Staggered polling has to behave as it really does, or the demo misleads."""
    transport = DemoTransport()
    result = transport.fetch(("jobs",))
    assert result.payload["sections"] == ["jobs"]
    # Narrowing the *claim* must not empty the other views: the data is all there, it is
    # simply not claimed to be fresh.
    assert result.payload["partitions"]

    full = transport.fetch()
    assert set(full.payload["sections"]) >= {"jobs", "partitions", "history", "usage"}


def test_demo_transport_names_itself():
    assert "demo" in DemoTransport().describe()
    assert "no SSH" in DemoTransport().describe()


def test_demo_transport_streams_nothing():
    with pytest.raises(TransportError):
        next(iter(DemoTransport().stream_file("/tmp/whatever.out")))


def test_demo_reports_the_policy_it_judges_against(capsys):
    """The GPU verdict and its thresholds must reach `--json`, demo or not."""
    code, out, _ = run_cli(["status", "--demo", "--json"], capsys)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    judged = [job for job in payload["jobs"] if job["gpu_util_per_gpu"] is not None]
    assert judged, "no job carried a utilisation figure"
    for job in judged:
        # These jobs are on `gh*`, so the H100/H200 rule applies.
        assert job["gpu_util_cancel_pct"] == 60.0
        assert job["gpu_util_warn_pct"] == 75.0
        assert job["gpu_util_policy"] == "gh*"


def test_the_demo_payload_has_no_leftover_internal_hostname():
    """The dataset is published now, so an internal FQDN is a leak, not a detail."""
    payload = load_demo_payload()
    hostname = payload["hostname"]
    assert "." not in hostname, f"{hostname!r} looks like an internal domain name"
    assert Path("lampter/demo_payload.json").exists()
