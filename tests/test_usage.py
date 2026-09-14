"""Live resource use from ``sstat``.

Two things here are not obvious and were found by looking at real output:

* ``sstat`` must be given the numeric ``job_id`` from ``squeue --json``. Passing the
  ``<array>_<task>`` display form selects every running task of the array at once.
* The ``extern`` step of an array task reports a nonsense ``AveCPU``
  (``213503982334-14:25:51``); displaying that as CPU time would be worse than
  showing nothing.

The cases below use the exact strings the controller produced.
"""

from __future__ import annotations

import json
import os

import pytest

from lampter import remote_probe as rp
from lampter.config import Config, load_config
from lampter.models import Job, Snapshot, Usage
from lampter.render import WIDE_LAYOUT, format_memory_kb, memory_cell
from lampter.tres import memory_mb

# ------------------------------------------------------------------ size parsing


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4491536K", 4491536),
        ("40224628K", 40224628),
        ("2304K", 2304),
        ("1.5G", 1572864),
        ("2M", 2048),
        ("0", 0),
        # sstat reports an absent metric as an empty field.
        ("", None),
        (None, None),
        ("   ", None),
        ("nonsense", None),
        ("12X", None),
    ],
)
def test_parse_size_kb(raw, expected):
    assert rp.parse_size_kb(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("13:40:34", 13 * 3600 + 40 * 60 + 34),
        ("00:51:11", 51 * 60 + 11),
        ("00:00:00", 0),
        ("2-03:00:00", 2 * 86400 + 3 * 3600),
        ("05:30", 5 * 60 + 30),
        ("42", 42),
        ("", None),
        (None, None),
        ("garbage", None),
        ("1:2:3:4", None),
    ],
)
def test_parse_cpu_seconds(raw, expected):
    assert rp.parse_cpu_seconds(raw) == expected


# ------------------------------------------------------------------ aggregation


def test_aggregate_usage_takes_the_maximum_across_steps():
    """Never the sum: `extern` already aggregates the job, so adding is double counting."""
    rows = [
        ["100.extern", "1000K", "00:10:00", "1", "0", "0"],
        ["100.batch", "2000K", "00:05:00", "1", "0", "0"],
    ]
    (record,) = rp.aggregate_usage(rows)
    assert record["job_id"] == 100
    assert record["max_rss_kb"] == 2000
    assert record["cpu_seconds"] == 600
    assert record["steps"] == 2


def test_aggregate_usage_discards_the_bogus_extern_cpu_of_an_array_task():
    """Real output: the extern step reported 213503982334 days of CPU time."""
    rows = [
        ["17365570.extern", "", "213503982334-14:25:51", "1", "0", "0"],
        ["17365570.batch", "40224628K", "00:51:11", "1", "58904", "587"],
    ]
    (record,) = rp.aggregate_usage(rows)
    assert record["job_id"] == 17365570
    # The plausible batch figures survive; the absurd one is dropped.
    assert record["cpu_seconds"] == 51 * 60 + 11
    assert record["max_rss_kb"] == 40224628
    # An empty MaxRSS on the extern step does not erase the batch step's value.
    assert record["max_rss_kb"] is not None


def test_aggregate_usage_keeps_the_extern_figure_when_it_is_the_sane_one():
    """A plain batch job is the mirror image: extern has the real numbers."""
    rows = [
        ["17192918.extern", "4491536K", "13:40:34", "1", "1354974925", "138730467"],
        ["17192918.batch", "2304K", "00:00:00", "1", "58904", "587"],
    ]
    (record,) = rp.aggregate_usage(rows)
    assert record["max_rss_kb"] == 4491536
    assert record["cpu_seconds"] == 13 * 3600 + 40 * 60 + 34


def test_aggregate_usage_separates_jobs():
    rows = [
        ["1.batch", "100K", "00:00:01", "1", "0", "0"],
        ["2.batch", "200K", "00:00:02", "1", "0", "0"],
    ]
    records = rp.aggregate_usage(rows)
    assert [r["job_id"] for r in records] == [1, 2]


def test_aggregate_usage_ignores_unparseable_rows():
    rows = [
        ["not-a-job.batch", "100K", "00:00:01", "1", "0", "0"],
        ["short"],
        ["3.batch", "300K", "00:00:03", "1", "0", "0"],
    ]
    (record,) = rp.aggregate_usage(rows)
    assert record["job_id"] == 3


def test_aggregate_usage_on_no_rows():
    assert rp.aggregate_usage([]) == []


def test_aggregate_usage_records_disk_io_in_bytes():
    rows = [["1.batch", "100K", "00:00:01", "1", "58904", "587"]]
    (record,) = rp.aggregate_usage(rows)
    assert record["disk_read_bytes"] == 58904 * 1024
    assert record["disk_write_bytes"] == 587 * 1024


# ------------------------------------------------------------------ collect_usage


def install_fake_tool(directory, name: str, stdout_text: str, returncode: int = 0, record=None):
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    body = "#!/bin/sh\n"
    if record is not None:
        body += f'echo "$*" > "{record}"\n'
    body += f"cat <<'DSH_OUT'\n{stdout_text}\nDSH_OUT\n"
    if returncode:
        body += f"exit {returncode}\n"
    script.write_text(body)
    script.chmod(0o755)
    return script


def prepend_path(monkeypatch, directory) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join([str(directory), os.environ.get("PATH", "")]))


def test_collect_usage_skips_the_call_when_nothing_is_running(monkeypatch, tmp_path):
    """No running jobs means the invocation is not just useless, it is avoided."""
    args_file = tmp_path / "args.txt"
    install_fake_tool(tmp_path / "bin", "sstat", "", record=args_file)
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    timings: dict = {}
    assert rp.collect_usage([], [], timings) == []
    assert "sstat" not in timings
    assert not args_file.exists()


def test_collect_usage_requests_all_steps(monkeypatch, tmp_path):
    """Without -a, sstat prints only a header and looks like 'no data'."""
    args_file = tmp_path / "args.txt"
    install_fake_tool(tmp_path / "bin", "sstat", "1.batch|100K|00:00:01|1|0|0", record=args_file)
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    timings: dict = {}
    records = rp.collect_usage(["1", "2"], [], timings)

    assert records and records[0]["job_id"] == 1
    assert "sstat" in timings
    recorded = args_file.read_text()
    assert " -a " in f" {recorded} "
    assert "-j 1,2" in recorded
    assert "-P" in recorded


def test_collect_usage_reports_a_failure(monkeypatch, tmp_path):
    install_fake_tool(tmp_path / "bin", "sstat", "", returncode=1)
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    errors: list[str] = []
    assert rp.collect_usage(["1"], errors, {}) == []
    assert any("sstat failed" in error for error in errors)


# ------------------------------------------------------------------ end to end


def slurm_job(**overrides) -> dict:
    record = {
        "job_id": 17365570,
        "array_job_id": {"set": True, "infinite": False, "number": 17325640},
        "array_task_id": {"set": True, "infinite": False, "number": 42},
        "array_task_string": "",
        "name": "train_gen",
        "user_name": "testuser",
        "partition": "h100_tandon",
        "job_state": ["RUNNING"],
        "state_reason": "",
        "submit_time": {"set": True, "number": 1000},
        "start_time": {"set": True, "number": 2000},
        "time_limit": {"set": True, "number": 150},
        "node_count": {"set": True, "number": 1},
        "cpus": {"set": True, "number": 8},
        "tres_req_str": "cpu=8,mem=128G,gres/gpu=1",
        "tres_alloc_str": "cpu=8,mem=128G,gres/gpu=1",
        "stdout_expanded": "/logs/train_gen_17325640_42.out",
    }
    record.update(overrides)
    return record


def read_emitted(capsys) -> dict:
    out = capsys.readouterr().out
    body = out.split(rp.JSON_BEGIN, 1)[1].split(rp.JSON_END, 1)[0]
    return json.loads(body)


def test_probe_uses_the_numeric_job_id_for_sstat(monkeypatch, tmp_path, capsys):
    """The array display form would select every running task of the array."""
    squeue_doc = {"jobs": [slurm_job()], "warnings": []}
    install_fake_tool(tmp_path / "bin", "squeue", json.dumps(squeue_doc))
    sstat_args = tmp_path / "sstat_args.txt"
    install_fake_tool(
        tmp_path / "bin",
        "sstat",
        "17365570.batch|40224628K|00:51:11|1|0|0",
        record=sstat_args,
    )
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setenv("LAMPTER_USER", "tester")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    assert rp.main(["--sections", "jobs,usage"]) == 0
    emitted = read_emitted(capsys)

    assert emitted["sections"] == ["jobs", "usage"]
    assert len(emitted["usage"]) == 1
    assert emitted["usage"][0]["job_id"] == 17365570
    # 17365570, not 17325640_42.
    assert "17365570" in sstat_args.read_text()
    assert "17325640_42" not in sstat_args.read_text()


def test_probe_skips_sstat_when_no_job_is_running(monkeypatch, tmp_path, capsys):
    squeue_doc = {"jobs": [slurm_job(job_state=["PENDING"])], "warnings": []}
    install_fake_tool(tmp_path / "bin", "squeue", json.dumps(squeue_doc))
    sstat_args = tmp_path / "sstat_args.txt"
    install_fake_tool(tmp_path / "bin", "sstat", "", record=sstat_args)
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setenv("LAMPTER_USER", "tester")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    assert rp.main(["--sections", "jobs,usage"]) == 0
    emitted = read_emitted(capsys)
    assert emitted["usage"] == []
    assert "sstat" not in emitted["timings_ms"]
    assert not sstat_args.exists()


# ------------------------------------------------------------------ model


def test_usage_from_wire_and_gb():
    usage = Usage.from_wire({"job_id": 5, "max_rss_kb": 4 * 1024 * 1024, "steps": 2})
    assert usage.max_rss_gb == pytest.approx(4.0)
    assert Usage.from_wire({}).max_rss_gb is None


def snapshot_with(jobs=(), usage=(), sections=("jobs", "usage")):
    return Snapshot.from_payload(
        {
            "generated_at": 1000,
            "sections": list(sections),
            "jobs": list(jobs),
            "usage": list(usage),
        },
        fetched_at=1000.0,
    )


def job(**overrides) -> Job:
    record = {
        "job_id": 17365570,
        "array_job_id": 17325640,
        "array_task_id": 42,
        "name": "train_gen",
        "partition": "h100_tandon",
        "state": "RUNNING",
        "tres_req": "cpu=8,mem=128G,gres/gpu=1",
        "tres_alloc": "cpu=8,mem=128G,gres/gpu=1",
    }
    record.update(overrides)
    return Job.from_wire(record)


def test_usage_lookup_uses_the_numeric_job_id_not_the_display_id():
    """An array task's display id is 17325640_42 but sstat reports 17365570."""
    task = job()
    assert task.display_id == "17325640_42"
    snapshot = snapshot_with([], [{"job_id": 17365570, "max_rss_kb": 40224628}])
    found = snapshot.usage_for(task)
    assert found is not None
    assert found.max_rss_kb == 40224628


def test_usage_lookup_returns_none_when_unmeasured():
    assert snapshot_with().usage_for(job()) is None
    assert snapshot_with().usage_for(job(job_id=None)) is None


def test_merged_with_carries_usage_forward():
    previous = snapshot_with(usage=[{"job_id": 1, "max_rss_kb": 1024}])
    fresh = Snapshot.from_payload(
        {"generated_at": 2000, "sections": ["jobs"], "jobs": []}, fetched_at=2000.0
    ).merged_with(previous)
    assert fresh.usage == previous.usage
    assert fresh.usage_at == 1000.0
    assert fresh.usage_age_sec(1060.0) == 60.0


def test_usage_age_is_none_when_never_fetched():
    assert snapshot_with(sections=("jobs",)).usage_age_sec(1.0) is None


# ------------------------------------------------------------------ memory limit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("128G", 128 * 1024),
        ("96G", 96 * 1024),
        ("1500M", 1500),
        ("64000", 64000),
        ("1T", 1024 * 1024),
        ("", None),
        (None, None),
        ("bogus", None),
    ],
)
def test_memory_mb(raw, expected):
    assert memory_mb(raw) == expected


def test_job_memory_limit_mb():
    # Both TRES strings must agree: for a running job the allocation is authoritative,
    # and `memory_limit_mb` prefers it over the request.
    both = job(tres_req="cpu=8,mem=96G", tres_alloc="cpu=8,mem=96G")
    assert both.memory_limit_mb == 96 * 1024
    # A job that has not started yet only has a request.
    requested = job(tres_req="cpu=8,mem=96G", tres_alloc="")
    assert requested.memory_limit_mb == 96 * 1024


# ------------------------------------------------------------------ rendering


def test_format_memory_kb():
    assert format_memory_kb(None) == "-"
    assert format_memory_kb(512) == "512K"
    assert format_memory_kb(4096) == "4M"
    assert format_memory_kb(4491536) == "4.3G"


def test_memory_cell_is_dash_without_a_measurement():
    assert memory_cell(job(), None).plain == "-"
    assert memory_cell(job(), Usage.from_wire({"job_id": 1})).plain == "-"


def test_memory_cell_warns_as_the_request_is_approached():
    """Slurm kills a job that exceeds its --mem, so nearing it matters."""

    def with_limit(gb: int) -> Job:
        # Both TRES strings, because the allocation wins for a running job.
        return job(tres_req=f"cpu=8,mem={gb}G", tres_alloc=f"cpu=8,mem={gb}G")

    # 40 GB of 48 GB requested: over 75%, under 90%.
    near = memory_cell(with_limit(48), Usage.from_wire({"job_id": 1, "max_rss_kb": 40 * 1024**2}))
    assert near.plain == "40.0G"
    assert "yellow" in str(near.style)

    # 46 GB of 48 GB: over 90%.
    critical = memory_cell(
        with_limit(48), Usage.from_wire({"job_id": 1, "max_rss_kb": 46 * 1024**2})
    )
    assert critical.plain.endswith("!")
    assert "bold red" in str(critical.style)

    # 10 GB of 48 GB: unremarkable.
    fine = memory_cell(with_limit(48), Usage.from_wire({"job_id": 1, "max_rss_kb": 10 * 1024**2}))
    assert fine.plain == "10.0G"
    assert "cyan" in str(fine.style)


def test_memory_cell_without_a_known_limit():
    plain = memory_cell(
        job(tres_req="cpu=8"), Usage.from_wire({"job_id": 1, "max_rss_kb": 1024**2})
    )
    assert plain.plain == "1.0G"


def test_mem_column_is_wide_only():
    assert any(column.key == "mem" for column in WIDE_LAYOUT)


# ------------------------------------------------------------------ config


def test_usage_interval_default_and_env(monkeypatch):
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    # A positive default, without pinning the exact number here -- the example-config
    # test is what keeps the documented default honest.
    assert Config().usage_interval > 0
    assert load_config(env={"LAMPTER_USAGE_INTERVAL": "45"}).usage_interval == 45.0


# ------------------------------------------------------------------ real fixture


def test_real_captured_usage():
    """Regression guard over genuine `sstat` output."""
    from fixture_data import snapshot as fixture_snapshot

    snapshot = fixture_snapshot()
    assert snapshot.usage, "the fixture should carry live usage"

    running = [job for job in snapshot.jobs if job.is_running]
    assert running
    measured = [job for job in running if snapshot.usage_for(job) is not None]
    assert measured, "every running job should have been measured"

    for job in measured:
        usage = snapshot.usage_for(job)
        assert usage is not None
        assert usage.steps >= 1
        if usage.max_rss_kb is not None:
            assert usage.max_rss_kb > 0
            # Plausible for a GPU node; the bogus values are filtered upstream.
            assert usage.max_rss_gb < 1000
        if usage.cpu_seconds is not None:
            assert 0 <= usage.cpu_seconds < 10 * 365 * 86400

    # Pending jobs have nothing for sstat to report.
    for job in snapshot.jobs:
        if job.is_pending:
            assert snapshot.usage_for(job) is None


def test_real_captured_usage_is_keyed_by_numeric_job_id():
    """The fixture contains an array, whose display id is not its job id."""
    from fixture_data import snapshot as fixture_snapshot

    snapshot = fixture_snapshot()
    array_tasks = [job for job in snapshot.jobs if job.array_task_id is not None]
    if not array_tasks:
        pytest.skip("fixture has no array tasks")
    for job in array_tasks:
        assert job.display_id != str(job.job_id)
        if job.is_running:
            assert snapshot.usage_for(job) is not None
