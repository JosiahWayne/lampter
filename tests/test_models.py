"""Job metrics: the derived queue wait and run time, plus identity and sorting.

The fixture in ``tests/fixtures/squeue_payload.json`` is a real (trimmed) probe
payload captured from the Torch cluster, so the regression test at the bottom
pins behaviour against genuine controller output rather than my assumptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fixture_data import payload as fixture_payload
from fixture_data import snapshot as fixture_snapshot

from lampter.models import Job, Snapshot, sort_jobs

FIXTURE = Path(__file__).parent / "fixtures" / "squeue_payload.json"


def wire(**overrides) -> dict:
    """A minimal probe record, as ``squeue --json`` would normalise to."""
    record = {
        "job_id": 100,
        "array_job_id": None,
        "array_task_id": None,
        "array_task_string": "",
        "name": "job",
        "user": "testuser",
        "account": "acct",
        "partition": "h100_tandon",
        "qos": "gpu48",
        "state": "PENDING",
        "reason": "",
        "state_description": "",
        "priority": 0,
        "submit_time": 1000,
        "eligible_time": 1000,
        "start_time": None,
        "end_time": None,
        "time_limit_min": 60,
        "time_limit_infinite": False,
        "node_count": 1,
        "nodelist": "",
        "cpus": 4,
        "tres_req": "cpu=4,mem=64G,node=1,billing=4,gres/gpu=1",
        "tres_alloc": "",
        "dependency": "",
        "stdout": "",
        "stderr": "",
        "workdir": "",
    }
    record.update(overrides)
    return record


def job(**overrides) -> Job:
    return Job.from_wire(wire(**overrides))


# ------------------------------------------------------------------ identity


def test_display_id_plain():
    assert job(job_id=17192918).display_id == "17192918"


def test_display_id_array_task():
    assert job(job_id=17365570, array_job_id=17325640, array_task_id=42).display_id == (
        "17325640_42"
    )


def test_display_id_array_master_restores_brackets():
    """The JSON API strips the brackets that squeue prints."""
    master = job(job_id=17325640, array_job_id=17325640, array_task_string="44-95%4")
    assert master.display_id == "17325640_[44-95%4]"


def test_display_id_array_master_keeps_existing_brackets():
    master = job(job_id=1, array_job_id=1, array_task_string="[0-9]")
    assert master.display_id == "1_[0-9]"


def test_display_id_handles_missing_job_id():
    assert job(job_id=None, array_job_id=None).display_id == "?"


def test_is_array():
    assert job(array_job_id=1, array_task_id=2).is_array
    assert not job().is_array


# ------------------------------------------------------------------ null handling


@pytest.mark.parametrize("value", ["None", "none", "null", "N/A", "", None])
def test_null_like_text_becomes_empty(value):
    """Slurm's JSON encoder writes the string "None" for absent reason codes."""
    assert job(reason=value).reason == ""


def test_real_reason_survives():
    assert job(reason="QOSMaxGRESPerUser").reason == "QOSMaxGRESPerUser"


# ------------------------------------------------------------------ state


def test_state_helpers():
    assert job(state="PENDING").is_pending
    assert job(state="RUNNING").is_running
    assert job(state="FAILED").is_terminal
    assert not job(state="RUNNING").is_pending


def test_state_is_upper_cased():
    assert job(state="running").state == "RUNNING"


def test_reason_text_glosses_known_codes():
    assert "QOS" in job(reason="QOSMaxGRESPerUser").reason_text
    assert job(reason="SomethingNew").reason_text == "SomethingNew"


def test_reason_text_glosses_account_wide_qos_limits():
    """``QOSGrpGRES`` is what Torch reports once the project account's GPUs are spent."""
    gloss = job(reason="QOSGrpGRES").reason_text
    assert gloss != "QOSGrpGRES"
    assert "account" in gloss


# ------------------------------------------------------------------ queue wait


def test_queue_wait_for_pending_counts_up_to_now():
    pending = job(state="PENDING", submit_time=1000)
    assert pending.queue_wait_sec(5000) == 4000


def test_queue_wait_for_running_is_submit_to_start():
    """A started job reports its final wait, so the column stays comparable."""
    running = job(state="RUNNING", submit_time=1000, start_time=6000)
    assert running.queue_wait_sec(99999) == 5000


def test_queue_wait_without_submit_time_is_unknown():
    assert job(submit_time=None).queue_wait_sec(5000) is None


def test_queue_wait_never_negative():
    """Clock skew between the login node and here must not produce "-3s"."""
    assert job(state="PENDING", submit_time=9000).queue_wait_sec(5000) == 0


# ------------------------------------------------------------------ run time


def test_run_time_is_unknown_until_started():
    assert job(state="PENDING", start_time=None).run_time_sec(5000) is None
    # Even if the controller has filled in a start time, a pending job is not running.
    assert job(state="PENDING", start_time=4000).run_time_sec(5000) is None


def test_run_time_for_running():
    running = job(state="RUNNING", start_time=4000)
    assert running.run_time_sec(5000) == 1000


def test_run_time_for_finished_uses_end_time():
    done = job(state="COMPLETED", start_time=4000, end_time=9000)
    assert done.run_time_sec(99999) == 5000


# ------------------------------------------------------------------ time limit


def test_time_left_from_end_time():
    running = job(state="RUNNING", start_time=1000, end_time=7000)
    assert running.time_left_sec(5000) == 2000


def test_time_left_derived_from_start_plus_limit():
    running = job(state="RUNNING", start_time=1000, time_limit_min=60)
    # Deadline is start + 60 minutes = 4600; at t=2000 there are 2600s left.
    assert running.time_left_sec(2000) == 1000 + 60 * 60 - 2000


def test_time_left_is_negative_once_overdue():
    running = job(state="RUNNING", start_time=1000, end_time=2000)
    assert running.time_left_sec(5000) < 0


def test_time_left_unlimited():
    unlimited = job(state="RUNNING", start_time=1000, time_limit_min=None)
    assert unlimited.time_left_sec(5000) is None


def test_used_fraction():
    running = job(state="RUNNING", start_time=1000, time_limit_min=10)
    # 5 minutes consumed out of 10.
    assert running.used_fraction(1000 + 300) == pytest.approx(0.5)


def test_used_fraction_none_when_unlimited():
    running = job(state="RUNNING", start_time=1000, time_limit_min=None)
    assert running.used_fraction(5000) is None


# ------------------------------------------------------------------ wait phase


def test_wait_phase_blocked_when_not_eligible():
    assert job(state="PENDING", eligible_time=None).wait_phase(5000) == "blocked"


def test_wait_phase_competing_when_eligible_and_waiting():
    assert job(state="PENDING", eligible_time=4000).wait_phase(5000) == "competing"


def test_wait_phase_scheduled_when_eligibility_is_in_the_future():
    assert job(state="PENDING", eligible_time=9000).wait_phase(5000) == "scheduled"


def test_wait_phase_not_applicable_when_running():
    assert job(state="RUNNING", start_time=1000).wait_phase(5000) == "-"


# ------------------------------------------------------------------ resources


def test_gpu_label_from_request():
    assert job(tres_req="cpu=8,gres/gpu=2").gpu_label == "2"


def test_gpu_label_prefers_allocation_over_request():
    running = job(state="RUNNING", tres_req="cpu=8,gres/gpu=2", tres_alloc="cpu=8,gres/gpu=1")
    assert running.gpu_count == 1


def test_gpu_label_includes_type_when_typed():
    assert job(tres_req="gres/gpu:h100=4").gpu_label == "4xh100"


def test_gpu_label_dash_without_gpus():
    assert job(tres_req="cpu=4,mem=64G").gpu_label == "-"


def test_memory_label():
    assert job(tres_req="cpu=4,mem=96G").memory_label == "96G"


# ------------------------------------------------------------------ snapshot


def test_snapshot_from_payload_and_counts():
    payload = {
        "generated_at": 1000,
        "hostname": "h",
        "user": "u",
        "slurm_version": "25.05.4",
        "jobs": [
            wire(job_id=1, state="RUNNING", start_time=1500),
            wire(job_id=2, state="PENDING"),
            wire(job_id=3, state="COMPLETED"),
        ],
        "errors": ["a warning"],
        "timings_ms": {"squeue": 90},
    }
    snapshot = Snapshot.from_payload(payload, fetched_at=1234.5, probe_elapsed_sec=0.9)
    assert snapshot.counts() == {"running": 1, "pending": 1, "other": 1}
    assert snapshot.errors == ("a warning",)
    assert snapshot.probe_elapsed_sec == 0.9
    assert snapshot.slurm_version == "25.05.4"
    assert snapshot.age_sec(1500) == 500
    assert snapshot.by_id("2") is not None


def test_snapshot_tolerates_missing_fields():
    snapshot = Snapshot.from_payload({}, fetched_at=10.0)
    assert snapshot.jobs == ()
    assert snapshot.generated_at == 10
    assert snapshot.counts() == {"running": 0, "pending": 0, "other": 0}
    assert snapshot.longest_wait(50) is None


def test_longest_wait():
    snapshot = Snapshot.from_payload(
        {"jobs": [wire(job_id=1, submit_time=100), wire(job_id=2, submit_time=500)]},
        fetched_at=1000.0,
    )
    assert snapshot.longest_wait(1000) == 900


# ------------------------------------------------------------------ sorting


def test_sort_by_wait_puts_running_first_then_longest_wait():
    now = 5000
    jobs = [
        job(job_id=1, state="PENDING", name="c", submit_time=1000),
        job(job_id=2, state="RUNNING", name="a", submit_time=1000, start_time=2000),
        job(job_id=3, state="RUNNING", name="b", submit_time=1000, start_time=3000),
    ]
    ordered = sort_jobs(jobs, "wait", now)
    assert [j.job_id for j in ordered] == [3, 2, 1]


def test_sort_by_name():
    jobs = [job(job_id=1, name="zeta"), job(job_id=2, name="alpha")]
    assert [j.job_id for j in sort_jobs(jobs, "name", 5000)] == [2, 1]


def test_sort_by_priority_descending():
    jobs = [
        job(job_id=1, priority=10),
        job(job_id=2, priority=99),
    ]
    assert [j.job_id for j in sort_jobs(jobs, "priority", 5000)] == [2, 1]


def test_sort_unknown_mode_falls_back_to_wait():
    jobs = [job(job_id=1), job(job_id=2)]
    assert len(sort_jobs(jobs, "nonsense", 5000)) == 2


def test_sort_reverse_inverts_order():
    jobs = [job(job_id=1, name="a"), job(job_id=2, name="b")]
    forward = [j.job_id for j in sort_jobs(jobs, "name", 5000)]
    backward = [j.job_id for j in sort_jobs(jobs, "name", 5000, reverse=True)]
    assert forward == list(reversed(backward))


# ------------------------------------------------------------------ real payload


def test_real_captured_payload_parses():
    """Regression guard against changes in the wire format or the maths."""
    raw = fixture_payload()
    snapshot = Snapshot.from_payload(raw, fetched_at=float(raw["generated_at"]))

    assert snapshot.slurm_version == "25.05.4"
    assert snapshot.errors == ()
    # Derived from the file, not hard-coded: the fixture is live cluster data.
    assert len(snapshot.jobs) == len(raw["jobs"])
    # Whatever the capture contains; adding a collector must not break this test.
    assert snapshot.sections == tuple(raw["sections"])

    counts = snapshot.counts()
    assert counts["running"] + counts["pending"] + counts["other"] == len(snapshot.jobs)

    for job in snapshot.jobs:
        assert job.display_id
        assert job.partition
        if job.is_pending:
            # Pending jobs have no start time, so wait is measured against "now".
            assert job.start_time is None
            assert job.queue_wait_sec(snapshot.generated_at) is not None
        if job.is_running:
            # Running jobs have started, so their wait is final and exact.
            assert job.start_time is not None
            assert job.queue_wait_sec(snapshot.generated_at) == (
                job.start_time - job.submit_time
            )
            assert job.run_time_sec(snapshot.generated_at) is not None


def test_real_captured_payload_array_naming():
    snapshot = fixture_snapshot()
    ids = [job.display_id for job in snapshot.jobs]
    # Every id is usable and distinct, which is what the store keys on.
    assert all(ids)
    assert len(ids) == len(set(ids))
    for job in snapshot.jobs:
        if job.array_task_id is not None:
            assert job.display_id == f"{job.array_job_id}_{job.array_task_id}"
        elif job.array_task_string:
            # A pending array's task range keeps its brackets.
            assert job.display_id.endswith("]")


def test_real_captured_partitions():
    """The sinfo aggregation must scale per-node gres by the group's node count.

    Getting this wrong is the difference between "h200 has 272 GPUs" and "h200 has
    48 GPUs", which is exactly the bug that was found against `scontrol show nodes`.
    """
    snapshot = fixture_snapshot()
    assert snapshot.partitions

    h200 = snapshot.partition_by_name("h200")
    assert h200 is not None
    # 34 nodes x 8 GPUs, and it is a busy partition, not an empty one.
    assert h200.gpus_total == h200.nodes_total * 8
    assert h200.gpu_type == "h200"
    assert 0 < h200.gpus_used <= h200.gpus_total
    assert h200.gpus_free == h200.gpus_total - h200.gpus_used

    # `all` pools every generation on the cluster, so no single model is honest.
    meta = snapshot.partition_by_name("all")
    if meta is not None:
        assert meta.gpu_type == "mixed"
        assert len(meta.gpu_types) > 1


def test_real_captured_partitions_are_self_consistent():
    snapshot = fixture_snapshot()
    for partition in snapshot.partitions:
        assert partition.nodes_idle + partition.nodes_mixed + partition.nodes_allocated + (
            partition.nodes_other
        ) == partition.nodes_total
        assert partition.gpus_used <= partition.gpus_total
        assert partition.cpus_allocated <= partition.cpus_total
        assert partition.mem_allocated_mb <= partition.mem_total_mb


def test_real_captured_history():
    snapshot = fixture_snapshot()
    assert snapshot.history
    assert snapshot.history_hours == 12
    finished = snapshot.finished()
    assert finished
    # Newest first.
    ends = [record.end_time or 0 for record in finished]
    assert ends == sorted(ends, reverse=True)
    for record in snapshot.history:
        assert record.display_id
        if record.is_terminal:
            assert record.end_time
            assert record.elapsed_sec is not None


# ------------------------------------------------------------------ partitions


def partition_wire(**overrides) -> dict:
    record = {
        "name": "h200",
        "nodes_total": 34,
        "nodes_idle": 0,
        "nodes_mixed": 26,
        "nodes_allocated": 7,
        "nodes_other": 1,
        "cpus_total": 4352,
        "cpus_allocated": 2953,
        "mem_total_mb": 2061000 * 34,
        "mem_allocated_mb": 262144,
        "gpus_total": 272,
        "gpus_used": 255,
        "gpu_type": "h200",
        "gpu_types": ["h200"],
        "max_time_min": None,
        "max_time_infinite": True,
    }
    record.update(overrides)
    return record


def test_partition_properties():
    from lampter.models import Partition

    p = Partition.from_wire(partition_wire())
    assert p.gpus_free == 17
    assert p.has_gpus
    assert p.cpus_free == 4352 - 2953
    assert p.mem_free_gb == (2061000 * 34 - 262144) // 1024
    assert p.gpu_type_label == "h200"
    assert p.is_healthy is False  # one node is down/drained
    assert p.gpu_use_fraction == pytest.approx(255 / 272)


def test_partition_free_gpus_never_go_negative():
    """Two probes can disagree mid-refresh; a negative free count would be nonsense."""
    from lampter.models import Partition

    p = Partition.from_wire(partition_wire(gpus_used=300))
    assert p.gpus_free == 0


def test_partition_labels_mixed_and_cpu_only():
    from lampter.models import Partition

    mixed = Partition.from_wire(
        partition_wire(name="all", gpu_type="mixed", gpu_types=["h200", "l40s"])
    )
    assert mixed.gpu_type_label == "mixed"
    assert mixed.gpu_types == ("h200", "l40s")

    cpu = Partition.from_wire(
        partition_wire(name="cpu_short", gpus_total=0, gpus_used=0, gpu_type=None, gpu_types=[])
    )
    assert cpu.gpu_type_label == "-"
    assert not cpu.has_gpus
    assert cpu.gpu_use_fraction is None


def test_partition_healthy_when_no_node_is_down():
    from lampter.models import Partition

    assert Partition.from_wire(partition_wire(nodes_other=0)).is_healthy


# ------------------------------------------------------------------ history jobs


def history_wire(**overrides) -> dict:
    record = {
        "job_id": 17325640,
        "array_job_id": None,
        "array_task_id": None,
        "name": "train_gen",
        "partition": "h100_tandon",
        "account": "acct",
        "qos": "gpu48",
        "state": "COMPLETED",
        "reason": "",
        "exit_status": "SUCCESS",
        "exit_code": 0,
        "exit_signal": 0,
        "submit_time": 1000,
        "eligible_time": 1000,
        "start_time": 4000,
        "end_time": 9000,
        "elapsed_sec": 5000,
        "time_limit_min": 150,
        "nodelist": "gh002",
        "node_count": 1,
        "gpus": 1,
    }
    record.update(overrides)
    return record


def test_history_job_identity_and_metrics():
    from lampter.models import HistoryJob

    h = HistoryJob.from_wire(history_wire())
    assert h.display_id == "17325640"
    assert h.queue_wait_sec(0) == 3000  # start - submit
    assert h.exit_label == "0:0"
    assert not h.is_failed
    assert h.is_terminal


def test_history_job_array_identity():
    from lampter.models import HistoryJob

    h = HistoryJob.from_wire(history_wire(array_job_id=17325640, array_task_id=5))
    assert h.display_id == "17325640_5"


def test_history_job_failure_recognition():
    from lampter.models import HistoryJob

    for state, phrase in (
        ("FAILED", "failed"),
        ("TIMEOUT", "time limit"),
        ("PREEMPTED", "preempted"),
        ("OUT_OF_MEMORY", "memory"),
        ("NODE_FAIL", "node"),
    ):
        h = HistoryJob.from_wire(history_wire(state=state, exit_code=1))
        assert h.is_failed, state
        assert phrase in h.outcome


def test_history_job_cancelled_is_terminal_but_not_a_failure():
    from lampter.models import HistoryJob

    h = HistoryJob.from_wire(history_wire(state="CANCELLED"))
    assert h.is_terminal
    assert not h.is_failed


def test_history_job_missing_exit_is_dash():
    from lampter.models import HistoryJob

    h = HistoryJob.from_wire(history_wire(exit_code=None, exit_signal=None))
    assert h.exit_label == "-"


def test_history_job_pending_wait_counts_up_to_now():
    from lampter.models import HistoryJob

    h = HistoryJob.from_wire(history_wire(state="PENDING", start_time=None))
    assert h.queue_wait_sec(1500) == 500


# ------------------------------------------------------------------ sections


def full_snapshot(jobs=(), history=(), partitions=(), fetched_at=100.0):
    return Snapshot.from_payload(
        {
            "generated_at": int(fetched_at),
            "sections": ["jobs", "partitions", "history"],
            "jobs": list(jobs),
            "history": list(history),
            "partitions": list(partitions),
        },
        fetched_at=fetched_at,
    )


def jobs_only_snapshot(jobs=(), fetched_at=200.0):
    return Snapshot.from_payload(
        {"generated_at": int(fetched_at), "sections": ["jobs"], "jobs": list(jobs)},
        fetched_at=fetched_at,
    )


def test_sections_are_inferred_for_hand_built_payloads():
    """Older or hand-written payloads without a `sections` key stay usable."""
    snap = Snapshot.from_payload({"jobs": [], "partitions": [], "history": []}, fetched_at=1.0)
    assert snap.sections == ("jobs", "partitions", "history")


def test_merged_with_carries_forward_unfetched_sections():
    previous = full_snapshot(
        partitions=[partition_wire()],
        history=[history_wire()],
        fetched_at=100.0,
    )
    fresh = jobs_only_snapshot(jobs=[wire(job_id=1)], fetched_at=200.0).merged_with(previous)

    assert len(fresh.jobs) == 1
    # Capacity and history survive, with their original timestamps intact.
    assert fresh.partitions == previous.partitions
    assert fresh.history == previous.history
    assert fresh.partitions_at == 100.0
    assert fresh.history_at == 100.0
    # ...but the snapshot itself is the new one.
    assert fresh.fetched_at == 200.0
    assert fresh.sections == ("jobs",)


def test_merged_with_replaces_sections_that_were_fetched():
    previous = full_snapshot(partitions=[partition_wire(name="old")], fetched_at=100.0)
    fresh = full_snapshot(partitions=[partition_wire(name="new")], fetched_at=200.0)
    merged = fresh.merged_with(previous)
    assert [p.name for p in merged.partitions] == ["new"]
    assert merged.partitions_at == 200.0


def test_merged_with_without_previous_returns_self():
    fresh = jobs_only_snapshot()
    assert fresh.merged_with(None) is fresh


def test_merged_with_does_not_invent_empty_sections():
    """If nothing was ever fetched, there is nothing to carry forward."""
    previous = jobs_only_snapshot(fetched_at=100.0)
    fresh = jobs_only_snapshot(fetched_at=200.0).merged_with(previous)
    assert fresh.partitions == ()
    assert fresh.partitions_at is None


def test_section_ages():
    snap = full_snapshot(partitions=[partition_wire()], fetched_at=100.0)
    assert snap.partitions_age_sec(160.0) == 60.0
    assert snap.history_age_sec(160.0) == 60.0
    assert jobs_only_snapshot().partitions_age_sec(1.0) is None


# ------------------------------------------------------------------ snapshot helpers


def test_partitions_for_my_jobs_and_held_gpus():
    jobs = [
        wire(job_id=1, partition="h100_tandon", state="RUNNING", tres_alloc="cpu=8,gres/gpu=2"),
        wire(job_id=2, partition="h100_tandon", state="RUNNING", tres_alloc="cpu=8,gres/gpu=1"),
        # Both TRES strings must be GPU-free here: `gpu_count` falls back from the
        # allocation to the request, and the helper's default request has a GPU.
        wire(
            job_id=3,
            partition="cpu_short",
            state="RUNNING",
            tres_alloc="cpu=4",
            tres_req="cpu=4",
        ),
        wire(job_id=4, partition="l40s", state="PENDING", tres_req="cpu=4,gres/gpu=1"),
    ]
    partitions = [
        partition_wire(name="h100_tandon"),
        partition_wire(name="cpu_short"),
        partition_wire(name="l40s"),
        partition_wire(name="unrelated"),
    ]
    snap = full_snapshot(jobs=jobs, partitions=partitions)

    assert snap.my_partitions == ("cpu_short", "h100_tandon", "l40s")
    assert [p.name for p in snap.partitions_for_my_jobs()] == [
        "h100_tandon",
        "cpu_short",
        "l40s",
    ]
    # Only running jobs hold GPUs, so the pending one contributes nothing.
    assert snap.held_gpus_by_partition() == {"h100_tandon": 3, "cpu_short": 0}


def test_finished_and_failures_are_newest_first():
    history = [
        history_wire(job_id=1, state="COMPLETED", end_time=100),
        history_wire(job_id=2, state="FAILED", end_time=300),
        history_wire(job_id=3, state="COMPLETED", end_time=200),
    ]
    snap = full_snapshot(history=history)
    assert [h.job_id for h in snap.finished()] == [2, 3, 1]
    assert [h.job_id for h in snap.recent_failures()] == [2]
    assert [h.job_id for h in snap.finished(1)] == [2]
    assert snap.history_by_id("2") is not None
