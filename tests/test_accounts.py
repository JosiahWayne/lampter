"""Accounts, QOS limits and fairshare.

Three things here were learned by looking at real output rather than assuming:

* ``squeue %b`` uses the gres **colon** form (``gres/gpu:1``) while ``sacctmgr`` and
  ``sacct`` use ``key=value`` (``gres/gpu=16``). Parsing only one reported zero GPUs.
* In squeue's format language ``%a`` is the **account** and ``%A`` the array job id.
* ``sacctmgr show qos`` without a ``where name=`` filter returns *inconsistent*
  ``MaxTRESPerUser`` values, so the collector filters -- that is a correctness
  requirement, not an optimisation.
"""

from __future__ import annotations

import json
import os

import pytest
from fixture_data import snapshot as fixture_snapshot

from lampter import remote_probe as rp
from lampter.models import AccountUsage, QosPressure, Snapshot
from lampter.render import (
    ACCOUNT_COLUMNS,
    QOS_COLUMNS,
    account_cell_map,
    build_account_table,
    build_qos_table,
    qos_cell_map,
)

# ------------------------------------------------------------------ TRES parsing


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cpu=1792,gres/gpu=112,mem=28000G", 112),  # sacctmgr GrpTRES
        ("billing=8,cpu=8,gres/gpu=1,mem=96G,node=1", 1),  # sacct AllocTRES
        ("billing=4,cpu=4,mem=64G,node=1", 0),  # a CPU-only job
        ("gres/gpu:1", 1),  # squeue %b, untyped
        ("gres/gpu:h200:1", 1),  # squeue %b, typed
        ("gres/gpu:l40s:1", 1),
        ("N/A", 0),  # squeue %b for a job with no GPU
        ("gres/gpu=0", 0),
        ("gres/gpu:h100:2,gres/gpu=1", 3),  # both spellings in one string
        ("", 0),
        (None, 0),
    ],
)
def test_count_gpus_handles_both_slurm_spellings(raw, expected):
    """``%b`` and ``sacctmgr`` do not spell TRES the same way; both must work."""
    assert rp.count_gpus(raw) == expected


def test_split_tres_is_not_the_gres_colon_form():
    """`gres/gpu:1` has no `=`, so split_tres must leave it alone rather than guess."""
    assert rp.split_tres("gres/gpu:1") == {}
    assert rp.split_tres("cpu=8,gres/gpu=1") == {"cpu": "8", "gres/gpu": "1"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cpu=1792,gres/gpu=112", 1792),
        ("billing=8,cpu=8", 8),
        ("gres/gpu=1", 0),
        ("", 0),
    ],
)
def test_tres_cpus(raw, expected):
    assert rp.tres_cpus(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cpu=1792,gres/gpu=112,mem=28000G", 28000 * 1024),
        ("mem=512000M", 512000),
        ("cpu=1", None),
        ("", None),
    ],
)
def test_tres_memory_mb(raw, expected):
    assert rp.tres_memory_mb(raw) == expected


# ------------------------------------------------------------------ fake tools


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


def only_tool(monkeypatch, tmp_path, **tools):
    """Put just these fake tools on PATH and disable the fallback bin dirs."""
    for name, output in tools.items():
        install_fake_tool(tmp_path / "bin", name, output)
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())
    return tmp_path / "bin"


# ------------------------------------------------------------------ collectors


def test_collect_qos_limits_filters_and_separates_the_two_caps(monkeypatch, tmp_path):
    """`where name=` is required: the unfiltered listing is not trustworthy."""
    args_file = tmp_path / "qos_args.txt"
    install_fake_tool(
        tmp_path / "bin",
        "sacctmgr",
        "gpu48||gres/gpu=16|||2-00:00:00",
        record=args_file,
    )
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    limits = rp.collect_qos_limits(["gpu48"], [], {})
    assert limits["gpu48"]["user_gpus"] == 16
    assert limits["gpu48"]["grp_gpus"] is None
    assert limits["gpu48"]["max_wall"] == "2-00:00:00"
    assert "where name=gpu48" in args_file.read_text()


def test_collect_qos_limits_reports_a_group_cap_separately(monkeypatch, tmp_path):
    only_tool(
        monkeypatch,
        tmp_path,
        sacctmgr="h100_tandon|cpu=1440,gres/gpu=60,mem=22500G||||",
    )
    limits = rp.collect_qos_limits(["h100_tandon"], [], {})
    assert limits["h100_tandon"]["grp_gpus"] == 60
    assert limits["h100_tandon"]["user_gpus"] is None
    assert limits["h100_tandon"]["grp_memory_mb"] == 22500 * 1024


def test_collect_qos_limits_skips_the_call_without_names(monkeypatch, tmp_path):
    args_file = tmp_path / "qos_args.txt"
    install_fake_tool(tmp_path / "bin", "sacctmgr", "", record=args_file)
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())
    assert rp.collect_qos_limits([], [], {}) == {}
    assert not args_file.exists()


def test_collect_associations(monkeypatch, tmp_path):
    only_tool(
        monkeypatch,
        tmp_path,
        # Order matches the requested format=Account,QOS,Partition,Share,...
        sacctmgr=(
            "acct_alpha_general|normal||1||||\n"
            "acct_alpha|normal||1||||\n"
            "users|normal||1||||\n"
        ),
    )
    accounts = rp.collect_associations("testuser", [], {})
    assert set(accounts) == {
        "acct_alpha_general",
        "acct_alpha",
        "users",
    }
    assert accounts["acct_alpha_general"] == {"normal"}


def test_collect_sshare(monkeypatch, tmp_path):
    only_tool(
        monkeypatch,
        tmp_path,
        sshare=(
            "acct_alpha|testuser|0.125000|1482153|0.942054|0.131315"
            "|cpu=2175,gres/gpu=271|0.132689|\n"
        ),
    )
    records = rp.collect_sshare(["acct_alpha"], [], {})
    entry = records["acct_alpha"]
    assert entry["effectv_usage"] == pytest.approx(0.942054)
    assert entry["fairshare"] == pytest.approx(0.131315)
    assert entry["norm_shares"] == pytest.approx(0.125)
    # TRESRunMins is a burn integral, kept for reference but never shown as a count.
    assert entry["tres_run_mins"]["gres/gpu"] == "271"
    assert entry["budget_configured"] is False


def test_collect_current_usage_buckets_by_account_qos_and_me(monkeypatch, tmp_path):
    """`%a` is the account; getting that wrong silently yields job numbers."""
    only_tool(
        monkeypatch,
        tmp_path,
        squeue=(
            "alice|acct_one|gpu48|gres/gpu:2|16|1\n"
            "bob|acct_one|gpu48|gres/gpu:1|8|1\n"
            "alice|acct_two|gpu168|gres/gpu:4|32|2\n"
            "carol|acct_two|cpu48|N/A|8|1\n"
        ),
    )
    by_account, by_qos, mine = rp.collect_current_usage("alice", [], {})

    assert by_account["acct_one"]["gpus"] == 3
    assert by_account["acct_one"]["jobs"] == 2
    assert by_account["acct_one"]["users"] == 2
    assert by_account["acct_two"]["gpus"] == 4

    assert by_qos["gpu48"]["gpus"] == 3
    assert by_qos["cpu48"]["gpus"] == 0  # N/A is a real answer, not an error

    # Only alice's own usage, per QOS.
    assert mine["gpu48"]["gpus"] == 2
    assert mine["gpu168"]["gpus"] == 4
    assert "cpu48" not in mine


def test_collect_account_window_sums_gpu_and_cpu_hours(monkeypatch, tmp_path):
    only_tool(
        monkeypatch,
        tmp_path,
        sacct=(
            "acct_one|3600|billing=8,cpu=8,gres/gpu=2,mem=96G,node=1\n"
            "acct_one|1800|billing=4,cpu=4,mem=64G,node=1\n"
            "acct_two|60|billing=8,cpu=8,gres/gpu=1,mem=96G,node=1\n"
        ),
    )
    totals = rp.collect_account_window(["acct_one", "acct_two"], 7, [], {})
    # 3600s x 2 GPUs = 2 GPU-hours; the CPU-only job adds none.
    assert totals["acct_one"]["gpu_seconds"] == 3600 * 2
    assert totals["acct_one"]["cpu_seconds"] == 3600 * 8 + 1800 * 4
    assert totals["acct_one"]["jobs"] == 2
    assert totals["acct_two"]["gpu_seconds"] == 60


def test_collect_accounts_reports_no_associations_rather_than_inventing_them(
    monkeypatch, tmp_path
):
    """sacctmgr is often restricted off-site; say so instead of guessing."""
    only_tool(monkeypatch, tmp_path, sacctmgr="", returncode=1)
    errors: list[str] = []
    accounts, qos = rp.collect_accounts([], "me", 7, errors, {})
    assert accounts == [] and qos == []
    assert any("associations" in error for error in errors)


def test_qos_reachability_includes_qos_seen_on_jobs(monkeypatch, tmp_path):
    """Torch puts jobs in QOSes that never appear on an association."""
    install_fake_tool(
        tmp_path / "bin",
        "sacctmgr",
        "acct_one|normal||1||||",
        record=None,
    )
    # sacctmgr is used for both associations and qos; the fake returns the same body,
    # so the qos parse sees no limits but the reachability set still matters.
    install_fake_tool(tmp_path / "bin", "sshare", "acct_one|me|0.1|0|0.5|0.5|x|1|")
    install_fake_tool(
        tmp_path / "bin", "squeue", "me|acct_one|gpu48|gres/gpu:1|8|1"
    )
    install_fake_tool(tmp_path / "bin", "sacct", "")
    prepend_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setattr(rp, "SLURM_BIN_DIRS", ())

    jobs = [{"qos": "gpu48"}]
    _accounts, qos = rp.collect_accounts(jobs, "me", 7, [], {})
    names = {entry["name"] for entry in qos}
    assert "normal" in names  # from the association
    assert "gpu48" in names  # from the job in flight


# ------------------------------------------------------------------ models


def test_qos_pressure_headroom_and_blocking():
    blocked = QosPressure.from_wire(
        {"name": "gpu168", "user_gpu_limit": 4, "my_gpus": 4, "group_gpu_limit": 60,
         "running_gpus": 53}
    )
    assert blocked.my_gpu_headroom == 0
    assert blocked.group_gpu_headroom == 7
    assert blocked.blocks_new_gpu_job is True
    assert blocked.is_limited is True

    roomy = QosPressure.from_wire({"name": "gpu48", "user_gpu_limit": 16, "my_gpus": 5})
    assert roomy.my_gpu_headroom == 11
    assert roomy.blocks_new_gpu_job is False

    uncapped = QosPressure.from_wire({"name": "normal"})
    assert uncapped.my_gpu_headroom is None
    assert uncapped.blocks_new_gpu_job is False
    assert uncapped.is_limited is False


def test_qos_pressure_never_reports_negative_headroom():
    over = QosPressure.from_wire({"name": "q", "user_gpu_limit": 2, "my_gpus": 5})
    assert over.my_gpu_headroom == 0
    assert over.blocks_new_gpu_job is True


def test_account_usage_priority_note_explains_fairshare():
    """FairShare is only meaningful next to the share it was computed from."""
    over = AccountUsage.from_wire(
        {"account": "a", "norm_shares": 0.125, "effectv_usage": 0.942, "fairshare": 0.131}
    )
    assert "over its share" in over.priority_note

    under = AccountUsage.from_wire(
        {"account": "b", "norm_shares": 0.5, "effectv_usage": 0.01, "fairshare": 0.9}
    )
    assert "under its share" in under.priority_note

    between = AccountUsage.from_wire(
        {"account": "c", "norm_shares": 0.5, "effectv_usage": 0.4, "fairshare": 0.6}
    )
    assert between.priority_note == "near its share"

    unknown = AccountUsage.from_wire({"account": "d"})
    assert unknown.priority_note == ""


def test_account_usage_shared_flag():
    assert AccountUsage.from_wire({"running_users": 1}).is_shared is False
    assert AccountUsage.from_wire({"running_users": 3}).is_shared is True


def accounts_payload(accounts=(), qos=(), sections=("jobs", "accounts")):
    return Snapshot.from_payload(
        {
            "generated_at": 1000,
            "sections": list(sections),
            "jobs": [],
            "accounts": list(accounts),
            "qos": list(qos),
            "account_days": 7,
        },
        fetched_at=1000.0,
    )


def test_snapshot_accounts_and_blocking_qos():
    snapshot = accounts_payload(
        accounts=[{"account": "a", "running_gpus": 5}],
        qos=[
            {"name": "gpu48", "user_gpu_limit": 16, "my_gpus": 5},
            {"name": "gpu168", "user_gpu_limit": 4, "my_gpus": 4},
        ],
    )
    assert snapshot.account_by_name("a") is not None
    assert snapshot.account_by_name("zzz") is None
    assert [q.name for q in snapshot.blocking_qos()] == ["gpu168"]
    assert snapshot.account_days == 7
    assert snapshot.qos_by_name("gpu48") is not None


def test_merged_with_carries_accounts_forward():
    previous = accounts_payload(accounts=[{"account": "a"}], qos=[{"name": "q"}])
    fresh = Snapshot.from_payload(
        {"generated_at": 2000, "sections": ["jobs"], "jobs": []}, fetched_at=2000.0
    ).merged_with(previous)
    assert [a.account for a in fresh.accounts] == ["a"]
    assert [q.name for q in fresh.qos_pressure] == ["q"]
    assert fresh.accounts_at == 1000.0
    assert fresh.accounts_age_sec(1060.0) == 60.0


def test_accounts_age_none_when_never_fetched():
    assert accounts_payload(sections=("jobs",)).accounts_age_sec(1.0) is None


# ------------------------------------------------------------------ rendering


def render(table, width: int = 200) -> str:
    import io

    from rich.console import Console

    buffer = io.StringIO()
    Console(width=width, file=buffer, force_terminal=False).print(table)
    return buffer.getvalue()


def test_account_and_qos_tables_fit_the_console():
    snapshot = accounts_payload(
        accounts=[{"account": "acct_alpha", "running_gpus": 5}],
        qos=[{"name": "gpu168", "user_gpu_limit": 4, "my_gpus": 4}],
    )
    for width in (80, 100, 150, 200):
        tables = (
            build_account_table(snapshot, width=width),
            build_qos_table(snapshot, width=width),
        )
        for table in tables:
            widest = max((len(line) for line in render(table, width).splitlines()), default=0)
            assert widest <= width


def test_qos_cell_verdicts():
    blocked = qos_cell_map(
        QosPressure.from_wire({"name": "q", "user_gpu_limit": 4, "my_gpus": 4})
    )
    assert "refuse" in blocked["verdict"].plain
    assert "bold red" in str(blocked["verdict"].style)
    assert blocked["user_free"].plain == "0"

    last = qos_cell_map(
        QosPressure.from_wire({"name": "q", "user_gpu_limit": 4, "my_gpus": 3})
    )
    assert "one GPU left" in last["verdict"].plain

    roomy = qos_cell_map(
        QosPressure.from_wire({"name": "q", "user_gpu_limit": 16, "my_gpus": 5})
    )
    assert roomy["verdict"].plain == "room to submit"

    uncapped = qos_cell_map(QosPressure.from_wire({"name": "normal"}))
    assert "no GPU cap" in uncapped["verdict"].plain


def test_account_cell_flags_a_shared_account():
    shared = account_cell_map(
        AccountUsage.from_wire({"account": "a", "running_users": 3, "running_gpus": 9})
    )
    assert "yellow" in str(shared["users"].style)
    solo = account_cell_map(AccountUsage.from_wire({"account": "b", "running_users": 1}))
    assert "dim" in str(solo["users"].style)


def test_account_and_qos_column_sets_agree_on_keys():
    account_cells = account_cell_map(AccountUsage.from_wire({"account": "a"}))
    for column in ACCOUNT_COLUMNS:
        assert column.key in account_cells, column.key
    qos_cells = qos_cell_map(QosPressure.from_wire({"name": "q"}))
    for column in QOS_COLUMNS:
        assert column.key in qos_cells, column.key


def test_real_fixture_has_no_accounts_but_still_parses():
    """The captured fixture predates this section; the model must not care."""
    snapshot = fixture_snapshot()
    assert snapshot.accounts == ()
    assert snapshot.qos_pressure == ()
    assert snapshot.blocking_qos() == []


# ------------------------------------------------------------------ cli


def test_accounts_command_renders(monkeypatch, capsys):
    from lampter import cli

    snapshot = accounts_payload(
        accounts=[
            {
                "account": "acct_alpha",
                "running_gpus": 5,
                "window_gpu_hours": 187.3,
                "norm_shares": 0.125,
                "effectv_usage": 0.942,
                "fairshare": 0.131,
            }
        ],
        qos=[{"name": "gpu168", "user_gpu_limit": 4, "my_gpus": 4}],
    )
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)

    assert cli.main(["accounts"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    # At 80 columns the name is truncated with an ellipsis, so match the prefix.
    assert "acct_alpha" in out or "acct_alph" in out
    assert "ACCOUNT" in out
    # The blocking QOS is called out explicitly.
    assert "per-user GPU cap" in out
    assert "gpu168" in out


def test_accounts_qos_view(monkeypatch, capsys):
    from lampter import cli

    snapshot = accounts_payload(
        accounts=[{"account": "a"}],
        qos=[{"name": "gpu48", "user_gpu_limit": 16, "my_gpus": 5}],
    )
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)

    assert cli.main(["accounts", "--qos"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "PER-USER" in out
    assert "GROUP" in out
    assert "gpu48" in out


def test_accounts_json(monkeypatch, capsys):
    from lampter import cli

    snapshot = accounts_payload(
        accounts=[{"account": "a", "running_gpus": 5}],
        qos=[{"name": "gpu168", "user_gpu_limit": 4, "my_gpus": 4}],
    )
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)

    assert cli.main(["accounts", "--json"]) == cli.EXIT_OK
    document = json.loads(capsys.readouterr().out)
    assert document["account_days"] == 7
    assert document["accounts"][0]["account"] == "a"
    assert document["qos"][0]["my_gpu_headroom"] == 0
    assert document["qos"][0]["blocks_new_gpu_job"] is True


def test_accounts_without_access_explains_itself(monkeypatch, capsys):
    from lampter import cli

    snapshot = accounts_payload()
    monkeypatch.setattr("lampter.config.CONFIG_SEARCH_PATHS", ())
    monkeypatch.setattr(cli, "fetch_snapshot", lambda config: snapshot)

    assert cli.main(["accounts"]) == cli.EXIT_FAILURE
    assert "no account data" in capsys.readouterr().out
