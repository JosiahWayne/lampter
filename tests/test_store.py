"""History store: transitions, alerts, deduplication and queue-wait statistics.

These use synthetic snapshots rather than the captured fixture, because the
interesting cases -- a job failing between refreshes, an alert firing exactly once,
a five-hour wait crossing the threshold -- are ones the live fixture rarely contains.
"""

from __future__ import annotations

import pytest

from lampter.models import Snapshot
from lampter.store import HistoryStore, StoreError


def job_wire(**overrides) -> dict:
    record = {
        "job_id": 100,
        "array_job_id": None,
        "array_task_id": None,
        "array_task_string": "",
        "name": "train",
        "user": "me",
        "account": "acct",
        "partition": "h100",
        "qos": "gpu48",
        "state": "PENDING",
        "reason": "Priority",
        "state_description": "",
        "priority": 10,
        "submit_time": 1000,
        "eligible_time": 1000,
        "start_time": None,
        "end_time": None,
        "time_limit_min": 60,
        "time_limit_infinite": False,
        "node_count": 1,
        "nodelist": "",
        "cpus": 4,
        "tres_req": "cpu=4,gres/gpu=1",
        "tres_alloc": "",
        "dependency": "",
        "stdout": "",
        "stderr": "",
        "workdir": "",
    }
    record.update(overrides)
    return record


def history_wire(**overrides) -> dict:
    record = {
        "job_id": 100,
        "array_job_id": None,
        "array_task_id": None,
        "name": "train",
        "partition": "h100",
        "account": "acct",
        "qos": "gpu48",
        "state": "COMPLETED",
        "reason": "",
        "exit_status": "SUCCESS",
        "exit_code": 0,
        "exit_signal": 0,
        "submit_time": 1000,
        "eligible_time": 1000,
        "start_time": 2000,
        "end_time": 3000,
        "elapsed_sec": 1000,
        "time_limit_min": 60,
        "nodelist": "gh001",
        "node_count": 1,
        "gpus": 1,
    }
    record.update(overrides)
    return record


def snapshot(jobs=(), history=(), *, generated_at=1000, sections=("jobs", "partitions", "history")):
    return Snapshot.from_payload(
        {
            "generated_at": generated_at,
            "jobs": list(jobs),
            "history": list(history),
            "sections": list(sections),
        },
        fetched_at=float(generated_at),
    )


@pytest.fixture
def store(tmp_path):
    with HistoryStore(tmp_path / "history.db", pending_alert_sec=3600, limit_soon_sec=600) as s:
        yield s


# ------------------------------------------------------------------ transitions


def test_first_sighting_records_a_transition(store):
    store.record(snapshot([job_wire(state="PENDING")]), now=1000)
    transitions = store.recent_transitions()
    assert len(transitions) == 1
    assert transitions[0]["job_key"] == "100"
    assert transitions[0]["from_state"] is None
    assert transitions[0]["to_state"] == "PENDING"
    assert transitions[0]["reason"] == "Priority"


def test_repeated_identical_refreshes_do_not_duplicate(store):
    """Refreshing every 15 seconds must not append a transition each time."""
    for moment in range(1000, 1010):
        store.record(snapshot([job_wire(state="PENDING")]), now=moment)
    assert store.stats()["jobs"] == 1
    assert store.stats()["transitions"] == 1


def test_state_change_is_recorded_once(store):
    store.record(snapshot([job_wire(state="PENDING")]), now=1000)
    store.record(snapshot([job_wire(state="RUNNING", start_time=1500)]), now=1010)
    store.record(snapshot([job_wire(state="RUNNING", start_time=1500)]), now=1020)

    transitions = store.recent_transitions()
    assert [t["to_state"] for t in transitions] == ["RUNNING", "PENDING"]
    assert transitions[0]["from_state"] == "PENDING"


def test_jobs_are_keyed_by_display_id(store):
    store.record(
        snapshot([job_wire(job_id=7, array_job_id=7, array_task_id=3)]), now=1000
    )
    assert store.recent_transitions()[0]["job_key"] == "7_3"


# ------------------------------------------------------------------ failure alerts


def test_failure_alert_for_a_job_we_watched(store):
    """squeue never shows FAILED, so sacct is the only way to learn this."""
    store.record(snapshot([job_wire(state="RUNNING", start_time=1500)]), now=1000)
    alerts = store.record(
        snapshot(
            [],
            [history_wire(state="TIMEOUT", exit_code=0, exit_signal=9)],
        ),
        now=1010,
    )
    assert len(alerts) == 1
    assert alerts[0].severity == "error"
    assert alerts[0].kind == "failed:TIMEOUT"
    assert "hit its time limit" in alerts[0].message
    assert "0:9" in alerts[0].message


def test_failure_alert_is_not_repeated(store):
    store.record(snapshot([job_wire(state="RUNNING", start_time=1500)]), now=1000)
    first = store.record(snapshot([], [history_wire(state="FAILED", exit_code=1)]), now=1010)
    second = store.record(snapshot([], [history_wire(state="FAILED", exit_code=1)]), now=1020)
    assert len(first) == 1
    assert second == []


def test_no_alert_for_history_we_never_watched(store):
    """A job that finished before the monitor started is not news."""
    alerts = store.record(snapshot([], [history_wire(state="FAILED", exit_code=1)]), now=1000)
    assert alerts == []
    # It is still recorded, so the history view and wait statistics can use it.
    assert store.stats()["jobs"] == 1
    assert store.finished_jobs()


def test_completed_alert_only_for_non_array_jobs(store):
    """A 50-task array must not bury everything else in completion messages."""
    store.record(
        snapshot([job_wire(job_id=1), job_wire(job_id=2, array_job_id=2, array_task_id=0)]),
        now=1000,
    )
    alerts = store.record(
        snapshot(
            [],
            [
                history_wire(job_id=1, state="COMPLETED"),
                history_wire(job_id=2, array_job_id=2, array_task_id=0, state="COMPLETED"),
            ],
        ),
        now=1010,
    )
    kinds = [a.kind for a in alerts]
    assert kinds == ["completed"]


def test_final_state_and_exit_are_persisted(store):
    store.record(snapshot([job_wire(state="RUNNING", start_time=1500)]), now=1000)
    store.record(snapshot([], [history_wire(state="FAILED", exit_code=2)]), now=1010)
    row = store.finished_jobs()[0]
    assert row["final_state"] == "FAILED"
    assert row["final_exit"] == "2:0"


def test_live_history_record_does_not_finalise_a_running_job(store):
    store.record(snapshot([job_wire(state="RUNNING", start_time=1500)]), now=1000)
    store.record(snapshot([], [history_wire(state="RUNNING", end_time=None)]), now=1010)
    assert store.finished_jobs() == []


# ------------------------------------------------------------------ threshold alerts


def test_pending_alert_fires_once_past_the_threshold(store):
    pending = job_wire(state="PENDING", submit_time=1000)
    # 30 minutes in: below the one-hour threshold.
    assert store.record(snapshot([pending]), now=1000 + 1800) == []
    # 90 minutes in: crosses it.
    raised = store.record(snapshot([pending]), now=1000 + 5400)
    assert len(raised) == 1
    assert raised[0].kind == "pending_long"
    assert "1h30m" in raised[0].message
    # And not again.
    assert store.record(snapshot([pending]), now=1000 + 9000) == []


def test_pending_alert_includes_the_scheduler_reason(store):
    pending = job_wire(state="PENDING", submit_time=1000, reason="QOSGrpGRES")
    raised = store.record(snapshot([pending]), now=1000 + 7200)
    assert "account's QOS GPU limit" in raised[0].message


def test_no_pending_alert_for_a_running_job(store):
    running = job_wire(state="RUNNING", submit_time=1000, start_time=1000)
    assert store.record(snapshot([running]), now=1000 + 7200) == []


def test_limit_soon_alert_fires_once(store):
    # 60 minute limit, started 55 minutes ago: 5 minutes left, under the 10 min bar.
    running = job_wire(
        state="RUNNING", start_time=1000, end_time=1000 + 3600, time_limit_min=60
    )
    raised = store.record(snapshot([running]), now=1000 + 3300)
    assert len(raised) == 1
    assert raised[0].kind == "limit_soon"
    assert "5m left" in raised[0].message
    assert store.record(snapshot([running]), now=1000 + 3400) == []


def test_no_limit_alert_when_plenty_of_time_remains(store):
    running = job_wire(
        state="RUNNING", start_time=1000, end_time=1000 + 3600, time_limit_min=60
    )
    assert store.record(snapshot([running]), now=1000 + 60) == []


# ------------------------------------------------------------------ statistics


def test_queue_wait_stats_uses_start_minus_submit(store):
    jobs = [
        job_wire(job_id=1, partition="h100", submit_time=0, start_time=100, state="RUNNING"),
        job_wire(job_id=2, partition="h100", submit_time=0, start_time=300, state="RUNNING"),
        job_wire(job_id=3, partition="h100", submit_time=0, start_time=500, state="RUNNING"),
        job_wire(job_id=4, partition="l40s", submit_time=0, start_time=50, state="RUNNING"),
    ]
    store.record(snapshot(jobs), now=1000)
    stats = {s.partition: s for s in store.queue_wait_stats()}

    assert stats["h100"].samples == 3
    assert stats["h100"].average_sec == pytest.approx(300.0)
    assert stats["h100"].median_sec == 300
    assert stats["h100"].max_sec == 500
    assert stats["l40s"].samples == 1
    assert stats["l40s"].median_sec == 50


def test_queue_wait_stats_ignores_jobs_that_never_started(store):
    store.record(
        snapshot(
            [
                job_wire(job_id=1, submit_time=0, start_time=None),
                job_wire(job_id=2, submit_time=0, start_time=10, state="RUNNING"),
            ]
        ),
        now=1000,
    )
    stats = store.queue_wait_stats()
    assert sum(s.samples for s in stats) == 1


# ------------------------------------------------------------------ durability


def test_alerts_survive_reopening_the_database(tmp_path):
    path = tmp_path / "history.db"
    with HistoryStore(path, pending_alert_sec=60) as store:
        store.record(snapshot([job_wire(state="PENDING", submit_time=0)]), now=1000)
    # A second process (say `lampter status`) must not re-raise it.
    with HistoryStore(path, pending_alert_sec=60) as store:
        assert store.record(snapshot([job_wire(state="PENDING", submit_time=0)]), now=2000) == []
        assert len(store.recent_alerts()) == 1


def test_unacknowledged_alert_count(store):
    # submit_time=0 and now=5000 is past the 3600s pending threshold.
    store.record(snapshot([job_wire(state="PENDING", submit_time=0)]), now=5000)
    assert store.unacknowledged_alert_count(0) == 1
    assert store.unacknowledged_alert_count(99999) == 0


def test_unusable_path_raises_store_error(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("this is a file, not a directory")
    with pytest.raises(StoreError):
        HistoryStore(blocker / "history.db")


def test_disappearance_does_not_lose_the_job(store):
    """A job leaving squeue keeps its row until sacct reports the outcome."""
    store.record(snapshot([job_wire(state="RUNNING", start_time=1500)]), now=1000)
    store.record(snapshot([]), now=1010)
    assert store.stats()["jobs"] == 1
    assert store.finished_jobs() == []
