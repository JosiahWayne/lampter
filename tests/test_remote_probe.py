"""The remote probe script itself.

``remote_probe.py`` runs on the cluster, but it is standard-library-only and pure,
so it can be imported and tested locally. The end-to-end test substitutes a fake
``squeue`` on ``PATH`` and checks the whole emission path.
"""

from __future__ import annotations

import json
import os

import pytest

from lampter import remote_probe as rp

# ------------------------------------------------------------------ primitives


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ({"set": True, "infinite": False, "number": 5}, 5),
        ({"set": True, "infinite": False, "number": 0}, 0),
        ({"set": False, "infinite": False, "number": 5}, None),
        ({"set": True, "infinite": True, "number": 0}, None),
        (7, 7),
        (0, 0),
        (None, None),
        ("nonsense", None),
    ],
)
def test_unwrap_number(field, expected):
    assert rp.unwrap_number(field) == expected


def test_unwrap_number_zero_is_none():
    """Timestamps arrive as {"set": true, "number": 0} when still unset."""
    assert rp.unwrap_number({"set": True, "number": 0}, zero_is_none=True) is None


def test_is_infinite():
    assert rp.is_infinite({"set": False, "infinite": True, "number": 0})
    assert not rp.is_infinite({"set": True, "infinite": False, "number": 1})
    assert not rp.is_infinite(None)


def test_loads_lenient():
    assert rp.loads_lenient('{"a": 1}') == {"a": 1}
    assert rp.loads_lenient('warning: something\n{"a": 1}') == {"a": 1}
    assert rp.loads_lenient("") is None
    assert rp.loads_lenient("not json at all") is None
    # A bare array is valid JSON but not a Slurm document.
    assert rp.loads_lenient("[1, 2]") is None


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], None),
        (["--user", "bob"], "bob"),
        (["-u", "bob"], "bob"),
        (["--user=bob"], "bob"),
        (["--user"], None),
    ],
)
def test_parse_args(argv, expected):
    assert rp.parse_args(argv)["user"] == expected


# ------------------------------------------------------------------ normalisation


def slurm_job(**overrides) -> dict:
    """A job record shaped the way ``squeue --json`` emits it."""
    record = {
        "job_id": 42,
        "array_job_id": {"set": True, "infinite": False, "number": 0},
        "array_task_id": {"set": False, "infinite": False, "number": 0},
        "array_task_string": "",
        "name": "train",
        "user_name": "testuser",
        "account": "acct_alpha",
        "partition": "h100_tandon",
        "qos": "gpu48",
        "job_state": ["PENDING"],
        "state_reason": "Priority",
        "state_description": "",
        "priority": {"set": True, "infinite": False, "number": 1234},
        "submit_time": {"set": True, "infinite": False, "number": 1789057843},
        "eligible_time": {"set": True, "infinite": False, "number": 0},
        "start_time": {"set": True, "infinite": False, "number": 0},
        "end_time": {"set": True, "infinite": False, "number": 0},
        "time_limit": {"set": True, "infinite": False, "number": 150},
        "node_count": {"set": True, "infinite": False, "number": 1},
        "nodes": "",
        "cpus": {"set": True, "infinite": False, "number": 8},
        "tres_req_str": "cpu=8,mem=128G,node=1,billing=8,gres/gpu=1",
        "tres_alloc_str": "",
        "dependency": "",
        "standard_output": "/scratch/u/out.log",
        "standard_error": "/scratch/u/err.log",
        "current_working_directory": "/scratch/u",
    }
    record.update(overrides)
    return record


def test_normalize_job_flattens_wrappers():
    out = rp.normalize_job(slurm_job())
    assert out["job_id"] == 42
    assert out["state"] == "PENDING"
    assert out["submit_time"] == 1789057843
    assert out["priority"] == 1234
    assert out["time_limit_min"] == 150
    assert out["cpus"] == 8
    assert out["stdout"] == "/scratch/u/out.log"
    assert out["workdir"] == "/scratch/u"


def test_normalize_job_maps_zero_timestamps_to_none():
    out = rp.normalize_job(slurm_job())
    # Zero means "not yet": both are the signal for a job still waiting.
    assert out["start_time"] is None
    assert out["eligible_time"] is None
    assert out["array_job_id"] is None


def test_normalize_job_handles_infinite_time_limit():
    out = rp.normalize_job(slurm_job(time_limit={"set": False, "infinite": True, "number": 0}))
    assert out["time_limit_min"] is None
    assert out["time_limit_infinite"] is True


def test_normalize_job_survives_an_empty_record():
    """A missing key must degrade, never raise: the probe cannot crash the refresh."""
    out = rp.normalize_job({})
    assert out["state"] == "UNKNOWN"
    assert out["job_id"] is None


def test_normalize_job_reads_array_task_ids():
    out = rp.normalize_job(
        slurm_job(
            job_id=17365570,
            array_job_id={"set": True, "infinite": False, "number": 17325640},
            array_task_id={"set": True, "infinite": False, "number": 42},
        )
    )
    assert out["array_job_id"] == 17325640
    assert out["array_task_id"] == 42


def test_normalize_job_takes_first_state():
    assert rp.normalize_job(slurm_job(job_state=["RUNNING", "RESIZING"]))["state"] == "RUNNING"


# ------------------------------------------------------------------ end to end


def install_fake_tool(directory, name: str, stdout_text: str, returncode: int = 0):
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    body = f"#!/bin/sh\ncat <<'DSH_OUT'\n{stdout_text}\nDSH_OUT\n"
    if returncode:
        body += f"exit {returncode}\n"
    script.write_text(body)
    script.chmod(0o755)
    return script


def prepend_path(monkeypatch, directory) -> None:
    """Put ``directory`` first on PATH without discarding the system one.

    Replacing PATH outright would also hide ``cat``, which the fake tools use, and
    the resulting failures look like probe bugs rather than test bugs.
    """
    monkeypatch.setenv("PATH", os.pathsep.join([str(directory), os.environ.get("PATH", "")]))


def read_emitted(capsys) -> dict:
    out = capsys.readouterr().out
    assert rp.JSON_BEGIN in out and rp.JSON_END in out
    body = out.split(rp.JSON_BEGIN, 1)[1].split(rp.JSON_END, 1)[0]
    return json.loads(body)


def test_main_emits_wrapped_json(tmp_path, monkeypatch, capsys):
    document = {"jobs": [slurm_job()], "warnings": [], "errors": []}
    install_fake_tool(tmp_path / "bin", "squeue", json.dumps(document))
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setenv("LAMPTER_USER", "tester")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    # Only squeue is faked here, so ask for just that section: the default set would
    # correctly also report sinfo and sacct as missing.
    assert rp.main(["--sections", "jobs"]) == 0

    emitted = read_emitted(capsys)
    assert emitted["schema"] == rp.SCHEMA_VERSION
    assert emitted["user"] == "tester"
    assert emitted["hostname"]
    assert len(emitted["jobs"]) == 1
    assert emitted["errors"] == []
    assert emitted["sections"] == ["jobs"]
    assert "squeue" in emitted["timings_ms"]


def test_sections_default_to_everything():
    # Derived, so adding a collector does not break this test.
    assert rp.parse_args([])["sections"] == rp.DEFAULT_SECTIONS
    assert set(rp.DEFAULT_SECTIONS) == set(rp.ALL_SECTIONS)


def test_sections_flag_restricts_the_collectors(tmp_path, monkeypatch, capsys):
    """A jobs-only poll must not invoke sinfo or sacct at all.

    This is the load-limiting behaviour: the expensive collectors are never called,
    rather than called and discarded.
    """
    install_fake_tool(tmp_path / "bin", "squeue", json.dumps({"jobs": []}))
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setenv("LAMPTER_USER", "tester")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    assert rp.main(["--sections", "jobs"]) == 0
    emitted = read_emitted(capsys)
    assert emitted["partitions"] == []
    assert emitted["history"] == []
    # No timings recorded for the skipped collectors means they never ran.
    assert "sinfo" not in emitted["timings_ms"]
    assert "sacct" not in emitted["timings_ms"]
    assert emitted["errors"] == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("jobs", ("jobs",)),
        ("jobs,history", ("jobs", "history")),
        ("JOBS,Partitions", ("jobs", "partitions")),
        ("jobs,nonsense", ("jobs",)),
        ("nonsense", rp.DEFAULT_SECTIONS),
        ("", rp.DEFAULT_SECTIONS),
    ],
)
def test_parse_sections(raw, expected):
    assert rp.parse_sections(raw) == expected


def test_version_is_read_from_the_json_meta_without_calling_sinfo(
    tmp_path, monkeypatch, capsys
):
    """Slurm embeds its version in every --json document, so sinfo -V is redundant."""
    document = {
        "jobs": [],
        "meta": {"Slurm_version": "25.05.4", "plugin": "scheduler"},
        "warnings": [],
    }
    install_fake_tool(tmp_path / "bin", "squeue", json.dumps(document))
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setenv("LAMPTER_USER", "tester")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    assert rp.main(["--sections", "jobs"]) == 0
    emitted = read_emitted(capsys)
    assert emitted["slurm_version"] == "25.05.4"
    assert "sinfo" not in emitted["timings_ms"]


def test_main_records_squeue_failure_without_crashing(tmp_path, monkeypatch, capsys):
    install_fake_tool(tmp_path / "bin", "squeue", "", returncode=1)
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setenv("LAMPTER_USER", "tester")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    assert rp.main([]) == 0

    emitted = read_emitted(capsys)
    assert emitted["jobs"] == []
    assert any("squeue failed" in error for error in emitted["errors"])


def test_main_reports_unparseable_squeue_output(tmp_path, monkeypatch, capsys):
    install_fake_tool(tmp_path / "bin", "squeue", "not json")
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setenv("LAMPTER_USER", "tester")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    assert rp.main([]) == 0
    emitted = read_emitted(capsys)
    assert any("could not parse" in error for error in emitted["errors"])


def test_main_reports_missing_squeue(tmp_path, monkeypatch, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setenv("LAMPTER_USER", "tester")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    assert rp.main([]) == 0
    emitted = read_emitted(capsys)
    assert any("squeue not found" in error for error in emitted["errors"])


def test_main_passes_the_requested_user(tmp_path, monkeypatch, capsys):
    """The fake squeue records its arguments so we can see what the probe sent."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "args.txt"
    script = bin_dir / "squeue"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$*" > "{args_file}"\n'
        "echo '{\"jobs\": []}'\n"
    )
    script.chmod(0o755)
    prepend_path(monkeypatch, bin_dir)
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    assert rp.main(["--user", "someone"]) == 0
    emitted = read_emitted(capsys)
    assert emitted["user"] == "someone"
    assert "--user someone" in args_file.read_text()


# ---------------------------------------------- sacct --json schema tolerance
#
# The payload below is the shape 23.11+ emits. It is spelled out rather than captured
# because the point is the *shape*, and because nothing else in the suite exercises
# `normalize_history_job` at all: the captured fixture is already in the client's wire
# format, so the probe's own sacct flattening has no coverage otherwise. That gap is
# why an `AttributeError` on a scalar `exit_code` could blank the whole dashboard.


def sacct_record(**overrides) -> dict:
    """One realistic ``sacct --json`` record, 23.11-and-later shape."""
    record = {
        "job_id": 17325640,
        "name": "train_gen",
        "partition": "h100_tandon",
        "account": "acct_alpha",
        "qos": "gpu168",
        "state": {"current": ["COMPLETED"], "reason": "None"},
        "exit_code": {
            "status": ["SUCCESS"],
            "return_code": {"set": True, "infinite": False, "number": 0},
            "signal": {"id": {"set": True, "infinite": False, "number": 0}, "name": ""},
        },
        "time": {
            "submission": {"set": True, "infinite": False, "number": 1757000000},
            "start": {"set": True, "infinite": False, "number": 1757000060},
            "end": {"set": True, "infinite": False, "number": 1757003600},
            "elapsed": {"set": True, "infinite": False, "number": 3540},
            "limit": {"set": True, "infinite": False, "number": 2880},
        },
        "tres": {"allocated": [{"type": "gres", "name": "gpu", "count": 2}]},
        "array": {"job_id": 0, "task_id": 0},
        "nodes": "gh007",
        "allocation_nodes": {"set": True, "infinite": False, "number": 1},
    }
    record.update(overrides)
    return record


def test_normalize_history_job_reads_the_23_11_shape():
    job = rp.normalize_history_job(sacct_record())
    assert job["state"] == "COMPLETED"
    assert job["reason"] == "None"
    assert job["exit_status"] == "SUCCESS"
    assert job["exit_code"] == 0
    assert job["gpus"] == 2
    assert job["start_time"] == 1757000060
    assert job["elapsed_sec"] == 3540
    assert job["time_limit_min"] == 2880


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # `exit_code` was a bare integer before 23.11, and some 24.05 builds emit it
        # that way again. `0` is the dangerous one: it is falsy, so `value or {}`
        # silently turned it into "no exit code at all".
        ("exit_code", 1),
        ("exit_code", 0),
        # `state` as a plain string, as other Slurm subcommands report it.
        ("state", "FAILED"),
        # A block that is simply absent.
        ("exit_code", None),
        ("time", None),
        ("tres", None),
        ("array", None),
    ],
)
def test_normalize_history_job_survives_a_schema_shift(field, value):
    """A shape change must cost one field, not the whole snapshot.

    Reading a scalar as a mapping raised `AttributeError` inside `build_payload`, whose
    only guard is `main`'s last-resort handler -- which blanked every section and, by
    omitting `sections`, made the client discard the last good data too.
    """
    job = rp.normalize_history_job(sacct_record(**{field: value}))
    assert job["job_id"] == 17325640  # the record still came through


def test_normalize_history_job_keeps_a_scalar_state_and_exit_code():
    """The information is still meaningful, so it is kept rather than dropped."""
    job = rp.normalize_history_job(sacct_record(state="FAILED", exit_code=2))
    assert job["state"] == "FAILED"
    assert job["exit_code"] == 2
    assert job["exit_status"] == ""  # no status word in the scalar shape


def test_normalize_history_job_keeps_a_zero_exit_code():
    job = rp.normalize_history_job(sacct_record(exit_code=0))
    assert job["exit_code"] == 0


def test_a_crashing_probe_claims_no_sections(monkeypatch, capsys):
    """The crash fallback must not look like a successful refresh.

    `Snapshot.from_payload` infers "what was fetched" from `sections`, and
    `merged_with` carries forward everything else. Omitting the key made a crashed
    probe look like a fresh snapshot of empty tables, so the previous rows were thrown
    away -- the opposite of degrading gracefully.
    """
    monkeypatch.setenv("LAMPTER_USER", "tester")

    def boom(*args, **kwargs):
        raise AttributeError("'int' object has no attribute 'get'")

    monkeypatch.setattr(rp, "build_payload", boom)

    assert rp.main([]) == 0
    emitted = read_emitted(capsys)
    assert emitted["sections"] == []
    assert any("probe crashed" in error for error in emitted["errors"])

    # And the client keeps the data it already had.
    from lampter.models import Snapshot

    previous = Snapshot.from_payload(
        {"sections": ["partitions"], "partitions": [{"name": "h200", "gpus_total": 272}]},
        fetched_at=100.0,
    )
    crashed = Snapshot.from_payload(emitted, fetched_at=200.0)
    assert crashed.partitions == ()
    merged = crashed.merged_with(previous)
    assert [p.name for p in merged.partitions] == ["h200"]
    assert merged.partitions_at == 100.0


def test_an_unknown_username_fallback_claims_no_sections(monkeypatch, capsys):
    monkeypatch.setattr(rp, "resolve_user", lambda _: "")
    assert rp.main([]) == 0
    emitted = read_emitted(capsys)
    assert emitted["sections"] == []
    assert emitted["errors"] == ["could not determine the remote username"]
