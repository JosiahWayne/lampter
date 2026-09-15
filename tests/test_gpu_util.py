"""GPU utilisation and the idle-GPU policy.

Torch cancels jobs whose GPUs sit idle, and emails about it first. The published
thresholds are per node family, and the utilisation Slurm records is pooled across a
job's GPUs -- both of which the column has to get right, and neither of which is
obvious from the outside.

Two regressions are pinned here in particular:

* ``gres/gpuutil`` is **pooled**, so a four-GPU job pegged on every card reads 400.
  Printing that as a percentage would be nonsense.
* Passing ``usage`` only for the wide layout once made the whole column blank below the
  wide threshold, which is where most users actually are.
"""

from __future__ import annotations

import io

import pytest
from fixture_data import payload as fixture_payload
from rich.console import Console

from lampter import remote_probe as rp
from lampter.gpu_policy import (
    DEFAULT_PATTERN,
    DEFAULT_THRESHOLDS,
    THRESHOLDS,
    VERDICT_CANCELLED,
    VERDICT_OK,
    VERDICT_WARNING,
    policy_for_prefixes,
    thresholds_for,
)
from lampter.models import Job, Snapshot, Usage
from lampter.render import (
    build_table,
    columns_for_view,
    gpu_cell,
    gpu_utilisation,
)


def job(**overrides) -> Job:
    record = {
        "job_id": 1,
        "name": "train",
        "state": "RUNNING",
        "tres_req": "cpu=8,mem=96G,gres/gpu=1",
        "tres_alloc": "cpu=8,mem=96G,gres/gpu=1",
        "nodelist": "ga005",
    }
    record.update(overrides)
    return Job.from_wire(record)


# ------------------------------------------------------------------ policy table


def test_thresholds_match_the_published_table():
    """Values are quoted from NYU's own documentation, so pin them."""
    assert THRESHOLDS["gl"] == (50.0, 70.0)
    assert THRESHOLDS["gh"] == (60.0, 75.0)
    assert THRESHOLDS["ga"] == (50.0, 70.0)
    assert THRESHOLDS["gr"] == (50.0, 70.0)
    assert DEFAULT_THRESHOLDS == (10.0, 50.0)


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("ga", (50.0, 70.0)),
        ("gh", (60.0, 75.0)),
        ("gl", (50.0, 70.0)),
        ("cl", (10.0, 50.0)),
        (None, (10.0, 50.0)),
    ],
)
def test_thresholds_for(prefix, expected):
    assert thresholds_for(prefix) == expected


def test_policy_uses_the_strictest_family_when_a_job_spans_several():
    """One strict node is enough to lose the job, so the strictest rule governs."""
    policy = policy_for_prefixes(("ga", "gh"))
    assert policy.pattern == "gh*"
    assert (policy.cancel_pct, policy.warn_pct) == (60.0, 75.0)

    # Order must not matter.
    assert policy_for_prefixes(("gh", "ga")).pattern == "gh*"


def test_policy_falls_back_to_the_default_and_says_so():
    policy = policy_for_prefixes(())
    assert policy.pattern == DEFAULT_PATTERN
    assert policy.is_specific is False
    assert policy_for_prefixes(("ga",)).is_specific is True


@pytest.mark.parametrize(
    ("pct", "expected"),
    [
        (0.0, VERDICT_CANCELLED),
        (49.9, VERDICT_CANCELLED),
        (50.0, VERDICT_WARNING),  # the line itself is not "below" it
        (69.9, VERDICT_WARNING),
        (70.0, VERDICT_OK),
        (100.0, VERDICT_OK),
        (None, ""),
    ],
)
def test_verdict_boundaries(pct, expected):
    assert policy_for_prefixes(("ga",)).verdict(pct) == expected


# ------------------------------------------------------------------ the arithmetic


def test_per_gpu_util_divides_the_pooled_figure():
    """Slurm pools the job's GPUs, so 200 on two cards is 100% each."""
    usage = Usage.from_wire({"job_id": 1, "gpu_util": 200})
    assert usage.per_gpu_util(2) == pytest.approx(100.0)
    assert Usage.from_wire({"job_id": 1, "gpu_util": 55}).per_gpu_util(1) == pytest.approx(55.0)


def test_per_gpu_util_refuses_an_impossible_result():
    """Above 100% per GPU means the pooling assumption is wrong for this job.

    Reporting the number anyway would be worse than reporting nothing, because the whole
    point of the column is deciding whether to trust it.
    """
    usage = Usage.from_wire({"job_id": 1, "gpu_util": 200})
    assert usage.per_gpu_util(1) is None
    # The raw value is still available, so the caller can show that something is off.
    assert usage.gpu_util == 200


def test_per_gpu_util_without_enough_information():
    usage = Usage.from_wire({"job_id": 1, "gpu_util": 55})
    assert usage.per_gpu_util(None) is None
    assert usage.per_gpu_util(0) is None
    assert Usage.from_wire({"job_id": 1}).per_gpu_util(1) is None


# ------------------------------------------------------------------ node prefixes


def test_node_prefixes_from_a_compacted_nodelist():
    """Slurm compacts ranges, so the prefix has to be read off the leading letters."""
    assert job(nodelist="ga005").node_prefixes == ("ga",)
    assert job(nodelist="gh[105-112,114-124]").node_prefixes == ("gh",)
    assert job(nodelist="gl001,gl002").node_prefixes == ("gl",)


def test_node_prefixes_across_families():
    assert job(nodelist="ga005,gl003").node_prefixes == ("ga", "gl")


def test_node_prefixes_when_not_started():
    assert job(nodelist="").node_prefixes == ()
    assert job(nodelist=None).node_prefixes == ()


# ------------------------------------------------------------------ the cell


def test_gpu_cell_without_a_gpu():
    assert gpu_cell(job(tres_req="cpu=4", tres_alloc=""), None).plain == "-"


def test_gpu_cell_without_a_measurement_shows_only_the_count():
    """Absent data must not read as 0% -- that would look like an idle GPU."""
    cell = gpu_cell(job(), None)
    assert cell.plain == "1"
    assert cell.spans == []  # nothing to colour


def test_gpu_cell_flags_an_uninterpretable_measurement():
    """A `?` says "measured, but I cannot trust the number", not "0".

    It is styled on the Text itself rather than on a span, because there is no
    value substring to colour -- the whole cell is the caveat.
    """
    cell = gpu_cell(job(), Usage.from_wire({"job_id": 1, "gpu_util": 200}))
    assert cell.plain == "1 ?"
    assert "yellow" in str(cell.style)
    # Neither green nor red: an unusable number must not be read as a verdict.
    assert "green" not in str(cell.style)
    assert "red" not in str(cell.style)


def test_gpu_cell_is_judged_against_the_job_node_family():
    """The same utilisation is safe on one node family and not on another."""
    low = 55.0
    on_a100 = gpu_cell(job(nodelist="ga005"), Usage.from_wire({"job_id": 1, "gpu_util": int(low)}))
    on_h100 = gpu_cell(job(nodelist="gh007"), Usage.from_wire({"job_id": 1, "gpu_util": int(low)}))
    assert "bold red" in str(on_h100.spans[0].style)  # below gh's 60% cancellation line
    assert "yellow" in str(on_a100.spans[0].style)  # above ga's 50%, below its 70%


def test_gpu_cell_marks_a_pooled_figure():
    """`~` warns that a single idle GPU is hidden inside the mean."""
    cell = gpu_cell(
        job(tres_req="gres/gpu=4", tres_alloc="gres/gpu=4"),
        Usage.from_wire({"job_id": 1, "gpu_util": 248}),
    )
    assert cell.plain == "4 ~62%"
    single = gpu_cell(job(), Usage.from_wire({"job_id": 1, "gpu_util": 62}))
    assert single.plain == "1 62%"


def test_gpu_utilisation_reports_both_halves():
    per_gpu, policy = gpu_utilisation(job(), Usage.from_wire({"job_id": 1, "gpu_util": 55}))
    assert per_gpu == pytest.approx(55.0)
    assert policy.pattern == "ga*"
    assert gpu_utilisation(job(), None)[0] is None


# ------------------------------------------------------------------ the column


def fixture_with_gpu_util(pooled: int = 248, gpus: int = 4, node: str = "gh007") -> Snapshot:
    """The captured fixture with realistic GPU figures grafted on.

    The fixture predates this column, so it carries no ``gpuutil``; inventing some is
    the only way to render the column with the content it will really hold.
    """
    raw = fixture_payload()
    for entry in raw.get("usage") or []:
        entry["gpu_util"] = pooled
        entry["gpu_memory_mb"] = 86748
    for record in raw["jobs"]:
        if "gres/gpu" in (record.get("tres_req") or ""):
            tres = f"cpu=16,mem=128G,node=1,gres/gpu={gpus}"
            record["tres_req"] = record["tres_alloc"] = tres
            record["nodelist"] = node
    return Snapshot.from_payload(raw, fetched_at=0.0)


def render(table, width: int) -> str:
    buffer = io.StringIO()
    Console(width=width, file=buffer, force_terminal=False).print(table)
    return buffer.getvalue()


@pytest.mark.parametrize("width", [80, 124, 125, 140, 189, 215, 240])
def test_gpu_utilisation_survives_every_layout(width):
    """Regression: passing `usage` only for the wide layout blanked this column.

    Anyone below the wide threshold saw a bare GPU count with no utilisation at all,
    which is exactly the number they need.
    """
    snapshot = fixture_with_gpu_util()
    out = render(build_table(snapshot, 0.0, width=width), width)
    assert "GPU UTIL" in out
    assert "~62%" in out, f"GPU utilisation missing at width {width}"


@pytest.mark.parametrize("width", [80, 124, 125, 140, 189, 215, 240])
def test_the_gpu_column_does_not_pay_for_itself_with_wait(width):
    """Widening GPU must not cost the column the tool exists to show."""
    snapshot = fixture_with_gpu_util()
    out = render(build_table(snapshot, 0.0, width=width), width)
    assert "WAIT" in out
    widest = max((len(line) for line in out.splitlines()), default=0)
    assert widest <= width


def test_gpu_column_has_a_header_that_says_what_it_holds():
    headers = [column.header for column in columns_for_view("jobs", 240)]
    assert "GPU UTIL" in headers


# ------------------------------------------------------------------ probe parsing


def test_probe_parses_gpuutil_and_gpumem():
    rows = [
        ["1.extern", "", "", "1", "0", "0", "energy=0"],
        ["1.batch", "1000K", "00:10:00", "1", "0", "0",
         "cpu=8,gres/gpumem=8268M,gres/gpuutil=55,mem=1G"],
    ]
    (record,) = rp.aggregate_usage(rows)
    assert record["gpu_util"] == 55
    assert record["gpu_memory_mb"] == 8268


def test_probe_requests_the_tres_usage_column():
    """The GPU figures ride along on the sstat call that was already being made."""
    assert "TRESUsageInAve" in rp.SSTAT_FIELDS


def test_probe_ignores_an_absurd_gpuutil():
    rows = [["1.batch", "1K", "00:00:01", "1", "0", "0", "gres/gpuutil=999999999"]]
    (record,) = rp.aggregate_usage(rows)
    assert record["gpu_util"] is None


def test_probe_leaves_gpuutil_absent_when_unreported():
    rows = [["1.batch", "1K", "00:00:01", "1", "0", "0", "energy=0"]]
    (record,) = rp.aggregate_usage(rows)
    assert record["gpu_util"] is None
    assert record["gpu_memory_mb"] is None


# ------------------------------------------------------------------ cli


def test_cli_exposes_the_policy_and_the_raw_value():
    """Scripts (and the reader) need both halves to judge the derivation themselves."""
    from lampter import cli

    entry = cli.gpu_to_dict(
        job(nodelist="gh007"), Usage.from_wire({"job_id": 1, "gpu_util": 55})
    )
    assert entry["gpu_util_raw"] == 55
    assert entry["gpu_util_per_gpu"] == pytest.approx(55.0)
    assert entry["gpu_util_verdict"] == VERDICT_CANCELLED
    assert entry["gpu_util_policy"] == "gh*"
    assert (entry["gpu_util_cancel_pct"], entry["gpu_util_warn_pct"]) == (60.0, 75.0)


def test_cli_reports_no_verdict_without_a_measurement():
    from lampter import cli

    entry = cli.gpu_to_dict(job(), None)
    assert entry["gpu_util_raw"] is None
    assert entry["gpu_util_per_gpu"] is None
    assert entry["gpu_util_verdict"] == ""
