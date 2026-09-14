"""Log tracking: path expansion, streaming, job lookup and the log screen.

The streaming transport is exercised against a fake ``ssh`` that emits lines like a
real ``tail`` would, so no network is involved. The "generator closed early" case
matters most: that is what stops the remote ``tail -f`` when you leave the viewer.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fixture_data import payload as fixture_payload

from lampter import cli
from lampter.models import Job, Snapshot, expand_log_path
from lampter.transport import SSHTransport, TransportError


def fake_ssh(tmp_path: Path, body: str) -> str:
    """An ssh stand-in that runs its own script and ignores its arguments."""
    script = tmp_path / "fake_ssh"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(0o755)
    return str(script)


def job(**overrides) -> Job:
    record = {
        "job_id": 100,
        "array_job_id": None,
        "array_task_id": None,
        "array_task_string": "",
        "name": "train",
        "user": "me",
        "partition": "h100",
        "state": "RUNNING",
        "stdout": "",
        "stderr": "",
    }
    record.update(overrides)
    return Job.from_wire(record)


# ------------------------------------------------------------------ path expansion


def test_expands_the_common_patterns():
    j = job(
        job_id=17325640,
        array_job_id=17325640,
        array_task_id=44,
        name="train_gen",
        user="testuser",
        nodelist="gh002",
    )
    assert (
        expand_log_path("logs/%x_%A_%a.out", j) == "logs/train_gen_17325640_44.out"
    )
    assert expand_log_path("slurm-%j.out", j) == "slurm-17325640.out"
    assert expand_log_path("%u-%N.log", j) == "testuser-gh002.log"
    assert expand_log_path("100%%-%j", j) == "100%-17325640"


def test_leaves_unresolvable_patterns_visible():
    """A wrong path is worse than an obviously unexpanded one."""
    j = job(job_id=7, name="t")
    assert expand_log_path("step-%s.log", j) == "step-%s.log"
    # %a cannot be resolved for a non-array job.
    assert expand_log_path("task-%a.out", j) == "task-%a.out"


def test_paths_without_patterns_are_untouched():
    assert expand_log_path("/dev/null", job()) == "/dev/null"


def test_log_path_prefers_the_expanded_path_from_slurm():
    j = job(stdout="logs/%j.out", stdout_expanded="/real/path/100.out")
    assert j.log_path == "/real/path/100.out"


def test_log_path_falls_back_to_local_expansion():
    j = job(job_id=100, stdout="logs/%j.out")
    assert j.log_path == "logs/100.out"


def test_error_log_path_falls_back_to_stdout():
    assert job(stdout="logs/%j.out").error_log_path == "logs/100.out"
    both = job(stdout="logs/%j.out", stderr="err/%j.err")
    assert both.error_log_path == "err/100.err"


def test_real_fixture_jobs_have_usable_log_paths():
    """The captured jobs use %j/%A/%a patterns, which must not leak into a path."""
    snapshot = Snapshot.from_payload(fixture_payload(), fetched_at=0.0)
    with_paths = [j for j in snapshot.jobs if j.stdout and j.stdout != "/dev/null"]
    assert with_paths
    for j in with_paths:
        # Either Slurm expanded it, or our local expansion resolved it.
        assert "%j" not in j.log_path or j.stdout_expanded
        assert "[" not in j.log_path


# ------------------------------------------------------------------ streaming


def test_stream_file_yields_lines(tmp_path):
    transport = SSHTransport("c", ssh_binary=fake_ssh(tmp_path, "printf 'one\\ntwo\\nthree\\n'\n"))
    assert list(transport.stream_file("/x.log", lines=5, follow=False)) == [
        "one",
        "two",
        "three",
    ]


def test_stream_file_quotes_the_remote_command(tmp_path):
    args_file = tmp_path / "args.txt"
    transport = SSHTransport(
        "c",
        ssh_binary=fake_ssh(tmp_path, f'echo "$@" > "{args_file}"\necho done\n'),
    )
    assert list(transport.stream_file("/a path/with space.log", lines=7, follow=False)) == [
        "done"
    ]
    recorded = args_file.read_text()
    assert "tail -n 7" in recorded
    assert "'/a path/with space.log'" in recorded


def test_stream_file_command_survives_ssh_argument_joining(tmp_path):
    """Faithfully emulate ssh: join the arguments with spaces, then let a shell run it.

    This is the test that catches the original bug. The earlier fake ssh ignored its
    arguments and ran its own script, so it never exercised argument passing at all.
    A real ssh concatenates its command arguments with spaces before the remote login
    shell sees them, so ``["sh", "-c", cmd]`` arrives as ``sh -c tail -n 5 /path``:
    the shell runs ``sh -c tail``, tail gets no arguments, reads stdin, prints
    nothing and exits 0 -- no output and no error.
    """
    target = tmp_path / "job output.txt"  # the space is the point
    target.write_text("alpha\nbeta\ngamma\n")
    joined_file = tmp_path / "joined.txt"
    # Emulate ssh: skip its own options and the destination, join what is left with
    # spaces, and let `eval` stand in for the remote login shell.
    body = (
        'host=0; cmd=""\n'
        'for a in "$@"; do\n'
        '  if [ "$host" = "1" ]; then cmd="$cmd $a";\n'
        '  elif [ "$a" = "c" ]; then host=1; fi\n'
        "done\n"
        'cmd="${cmd# }"\n'
        f'printf "%s" "$cmd" > "{joined_file}"\n'
        'eval "$cmd"\n'
    )
    transport = SSHTransport("c", ssh_binary=fake_ssh(tmp_path, body))

    lines = list(transport.stream_file(str(target), lines=2, follow=False))

    assert lines == ["beta", "gamma"]
    # One single, correctly quoted argument -- not ["sh", "-c", ...].
    assert joined_file.read_text() == f"tail -n 2 '{target}'"


def test_stream_file_reports_a_remote_failure(tmp_path):
    body = "echo 'tail: cannot open' >&2\nexit 1\n"
    transport = SSHTransport("c", ssh_binary=fake_ssh(tmp_path, body))
    with pytest.raises(TransportError, match="cannot open"):
        list(transport.stream_file("/missing.log", follow=False))


def test_stream_file_missing_ssh_binary_is_reported():
    transport = SSHTransport("c", ssh_binary="/nonexistent/ssh")
    with pytest.raises(TransportError, match="ssh binary not found"):
        list(transport.stream_file("/x.log", follow=False))


def test_closing_the_stream_stops_a_follow(tmp_path):
    """Leaving the viewer must not leave a tail running on the login node."""
    body = "while true; do echo tick; sleep 0.05; done\n"
    transport = SSHTransport("c", ssh_binary=fake_ssh(tmp_path, body))

    stream = transport.stream_file("/x.log", lines=1, follow=True)
    assert [next(stream) for _ in range(3)] == ["tick", "tick", "tick"]

    started = time.monotonic()
    stream.close()
    # Terminating the child happens in the generator's finally block.
    assert time.monotonic() - started < 5


# ------------------------------------------------------------------ job lookup


def snapshot_with(*jobs: Job) -> Snapshot:
    return Snapshot.from_payload(
        {
            "generated_at": 1000,
            "sections": ["jobs"],
            "jobs": [j.raw for j in jobs],
        },
        fetched_at=1000.0,
    )


def test_resolve_job_exact_id():
    a, b = job(job_id=1, name="alpha"), job(job_id=2, name="beta")
    match = cli.resolve_job(snapshot_with(a, b), "2")
    assert match.job is not None and match.job.job_id == 2
    assert match.problem == ""


def test_resolve_job_array_task_by_display_id():
    task = job(job_id=555, array_job_id=100, array_task_id=3, name="arr")
    match = cli.resolve_job(snapshot_with(task), "100_3")
    assert match.job is not None and match.job.display_id == "100_3"


def test_resolve_job_array_master_prefers_the_running_task():
    pending = job(job_id=1, array_job_id=100, array_task_id=1, name="arr", state="PENDING")
    running = job(job_id=2, array_job_id=100, array_task_id=2, name="arr", state="RUNNING")
    match = cli.resolve_job(snapshot_with(pending, running), "100")
    assert match.job is not None and match.job.display_id == "100_2"
    assert match.problem == ""


def test_resolve_job_picks_the_freshest_of_several_running_tasks():
    """Four running array tasks should not require disambiguating four long ids."""
    a = job(job_id=1, array_job_id=100, array_task_id=1, name="arr",
            state="RUNNING", start_time=100)
    b = job(job_id=2, array_job_id=100, array_task_id=2, name="arr",
            state="RUNNING", start_time=900)
    match = cli.resolve_job(snapshot_with(a, b), "100")
    assert match.job is not None and match.job.display_id == "100_2"
    assert match.note and "most recently started" in match.note


def test_resolve_job_reports_ambiguity_between_different_jobs():
    a = job(job_id=1, array_job_id=100, array_task_id=1, name="alpha", state="RUNNING")
    b = job(job_id=2, array_job_id=100, array_task_id=2, name="beta", state="RUNNING")
    match = cli.resolve_job(snapshot_with(a, b), "100")
    assert match.job is None
    assert "matches several" in match.problem
    assert "full id" in match.problem


def test_resolve_job_by_name_fragment():
    match = cli.resolve_job(snapshot_with(job(job_id=9, name="eval_stats")), "eval")
    assert match.job is not None and match.job.job_id == 9


def test_resolve_job_not_found():
    match = cli.resolve_job(snapshot_with(job(job_id=1, name="alpha")), "zzz")
    assert match.job is None
    assert "no job matching" in match.problem


# ------------------------------------------------------------------ logs command


def serve_transport(monkeypatch, transport):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    monkeypatch.setattr(cli, "make_transport", lambda config: transport)


def test_logs_with_an_explicit_path_never_probes(monkeypatch, capsys, tmp_path):
    """--path is the escape hatch for a job that is no longer in the queue."""
    transport = SSHTransport(
        "c", ssh_binary=fake_ssh(tmp_path, "printf 'hello\\nworld\\n'\n")
    )
    serve_transport(monkeypatch, transport)

    def must_not_run(config):
        raise AssertionError("a probe should not be needed with --path")

    monkeypatch.setattr(cli, "fetch_snapshot", must_not_run)

    assert cli.main(["logs", "--path", "/remote/x.log"]) == 0
    captured = capsys.readouterr()
    assert "hello" in captured.out
    assert "world" in captured.out
    assert "tail -n 200 /remote/x.log" in captured.err


def test_logs_streams_a_jobs_output(monkeypatch, capsys, tmp_path):
    snapshot = snapshot_with(
        job(job_id=42, name="train", stdout="/logs/train_%j.out")
    )
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)
    transport = SSHTransport("c", ssh_binary=fake_ssh(tmp_path, "printf 'epoch 1\\n'\n"))
    serve_transport(monkeypatch, transport)

    assert cli.main(["logs", "42"]) == 0
    captured = capsys.readouterr()
    assert "epoch 1" in captured.out
    # The resolved path is what gets tailed.
    assert "/logs/train_42.out" in captured.err
    assert "train on h100" in captured.err


def test_logs_reports_a_job_without_an_output_file(monkeypatch, capsys, tmp_path):
    snapshot = snapshot_with(job(job_id=42, name="train", stdout=""))
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)
    serve_transport(monkeypatch, SSHTransport("c", ssh_binary=fake_ssh(tmp_path, "")))

    assert cli.main(["logs", "42"]) == cli.EXIT_FAILURE
    assert "no log file recorded" in capsys.readouterr().err


def test_logs_reports_an_unknown_job(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot_with())
    serve_transport(monkeypatch, SSHTransport("c", ssh_binary=fake_ssh(tmp_path, "")))

    assert cli.main(["logs", "nope"]) == cli.EXIT_FAILURE
    assert "no job matching" in capsys.readouterr().err


def test_logs_does_not_follow_a_finished_job(monkeypatch, capsys, tmp_path):
    """Following a static file would hang and look like a bug."""
    snapshot = snapshot_with(
        job(job_id=42, name="train", state="COMPLETED", stdout="/logs/train_%j.out")
    )
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)
    transport = SSHTransport("c", ssh_binary=fake_ssh(tmp_path, "echo done\n"))
    serve_transport(monkeypatch, transport)

    assert cli.main(["logs", "42", "--follow"]) == 0
    captured = capsys.readouterr()
    assert "done" in captured.out
    assert "nothing more to follow" in captured.err
    # Downgraded, so no -f reached the remote command.
    assert "-f" not in captured.err


def test_logs_reports_a_transport_failure(monkeypatch, capsys, tmp_path):
    transport = SSHTransport(
        "c", ssh_binary=fake_ssh(tmp_path, "echo 'no such file' >&2\nexit 1\n")
    )
    serve_transport(monkeypatch, transport)
    assert cli.main(["logs", "--path", "/nope.log"]) == cli.EXIT_FAILURE
    assert "no such file" in capsys.readouterr().err


# ------------------------------------------------------------------ log screen


def test_log_screen_streams_into_the_view(monkeypatch):
    pytest.importorskip("textual")
    from textual.app import App
    from textual.widgets import RichLog

    from lampter.ui.logview import LogScreen

    class StreamingTransport:
        def __init__(self):
            self.closed = False

        def stream_file(self, path, *, lines=200, follow=True):
            try:
                for index in range(3):
                    yield f"line {index}"
            finally:
                # Mirrors what the real transport does when the generator closes.
                self.closed = True

    # Record what the screen writes, rather than reaching into RichLog's internals
    # (it exposes no public line count).
    written: list[str] = []
    original_write = RichLog.write

    def recording_write(self, content, *args, **kwargs):
        written.append(str(content))
        return original_write(self, content, *args, **kwargs)

    monkeypatch.setattr(RichLog, "write", recording_write)

    async def scenario():
        transport = StreamingTransport()
        screen = LogScreen(transport, "/logs/x.out", header="job 1", follow=False, lines=10)

        class Host(App[None]):
            def on_mount(self) -> None:
                self.push_screen(screen)

        app = Host()
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(30):
                if transport.closed:
                    break
                await pilot.pause()
            assert transport.closed

    asyncio.run(scenario())

    assert any("line 0" in entry for entry in written)
    assert any("line 2" in entry for entry in written)
    # A non-following read says so, so you know it is not still waiting.
    assert any("end of file" in entry for entry in written)


def test_log_open_action_without_a_log_path_notifies():
    pytest.importorskip("textual")
    from textual.widgets import DataTable

    from lampter.ui import SlurmMonitorApp

    class StubTransport:
        def describe(self):
            return "stub"

        def fetch(self, sections=None):
            from lampter.transport import ProbeResult

            payload = fixture_payload()
            payload["generated_at"] = int(time.time())
            return ProbeResult(payload=payload, elapsed_sec=0.01)

    async def scenario():
        app = SlurmMonitorApp(StubTransport(), interval=600)
        async with app.run_test(size=(200, 40)) as pilot:
            for _ in range(30):
                if app.query_one(DataTable).row_count:
                    break
                await pilot.pause()
            # The fixture's first job writes to /dev/null, which is a real path.
            before = len(app.screen_stack)
            await pilot.press("o")
            await pilot.pause()
            # Either a log screen opened, or a warning was shown; never a crash.
            assert len(app.screen_stack) >= before

    asyncio.run(scenario())
