"""SSH transport: argv construction, sentinel slicing and failure reporting.

These tests never touch the network. They substitute a fake ``ssh`` executable, so
they exercise the real ``subprocess`` path -- including stdin piping -- without
needing a cluster.
"""

from __future__ import annotations

import json

import pytest

from lampter.remote_probe import JSON_BEGIN, JSON_END, SCHEMA_VERSION
from lampter.transport import SSHTransport, TransportError


def write_script(path, body: str) -> str:
    path.write_text(body)
    path.chmod(0o755)
    return str(path)


def fake_ssh(tmp_path, stdout_text: str, returncode: int = 0, stderr_text: str = "") -> str:
    """A stand-in for ssh that consumes stdin and prints a canned response."""
    body = "#!/bin/sh\ncat > /dev/null\n"
    body += f"cat <<'DSH_OUT'\n{stdout_text}\nDSH_OUT\n"
    if stderr_text:
        body += f"cat >&2 <<'DSH_ERR'\n{stderr_text}\nDSH_ERR\n"
    body += f"exit {returncode}\n"
    return write_script(tmp_path / "fake_ssh", body)


def envelope(payload: dict) -> str:
    return f"{JSON_BEGIN}\n{json.dumps(payload)}\n{JSON_END}"


def payload(**overrides) -> dict:
    base = {
        "schema": SCHEMA_VERSION,
        "generated_at": 1789115388,
        "hostname": "torch-login-a-0",
        "user": "testuser",
        "slurm_version": "25.05.4",
        "jobs": [],
        "errors": [],
        "timings_ms": {"squeue": 90},
    }
    base.update(overrides)
    return base


def transport_for(tmp_path, stdout_text: str, **kwargs) -> SSHTransport:
    return SSHTransport(
        "cluster",
        ssh_binary=fake_ssh(tmp_path, stdout_text, **kwargs),
        command_timeout=10,
    )


# ------------------------------------------------------------------ argv


def test_argv_defaults_are_non_interactive():
    argv = SSHTransport("torch", user="bob")._argv()
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert any(arg.startswith("ConnectTimeout=") for arg in argv)
    # The remote program reads the probe from stdin, so the host and `python3 -`
    # must appear in that order, ahead of the probe's own flags.
    index = argv.index("torch")
    assert argv[index : index + 3] == ["torch", "python3", "-"]


def test_argv_requests_all_sections_by_default():
    from lampter.remote_probe import DEFAULT_SECTIONS

    argv = SSHTransport("torch", user="bob")._argv()
    assert argv[-2:] == ["--sections", ",".join(DEFAULT_SECTIONS)]


def test_argv_passes_a_restricted_section_set():
    """A jobs-only poll is what keeps a fast refresh cheap on the controller."""
    argv = SSHTransport("torch")._argv(("jobs",))
    assert argv[-2:] == ["--sections", "jobs"]


def test_argv_passes_a_non_default_history_window():
    argv = SSHTransport("torch", history_hours=48)._argv()
    index = argv.index("--history-hours")
    assert argv[index + 1] == "48"
    # The default is left off the command line entirely.
    assert "--history-hours" not in SSHTransport("torch", history_hours=12)._argv()


def test_argv_passes_a_non_default_accounting_window():
    """`--account-days` was documented, configurable and silently inert.

    The transport forwarded `--history-hours` but never this, so the probe always
    totalled its own seven-day default while `doctor` printed the configured value.
    """
    argv = SSHTransport("torch", account_days=30)._argv(("accounts",))
    index = argv.index("--account-days")
    assert argv[index + 1] == "30"
    assert "--account-days" not in SSHTransport("torch", account_days=7)._argv(("accounts",))


def test_argv_omits_user_when_unset():
    argv = SSHTransport("torch")._argv()
    assert "--user" not in argv
    index = argv.index("torch")
    assert argv[index : index + 3] == ["torch", "python3", "-"]


def test_argv_can_allow_password_prompts():
    argv = SSHTransport("torch", batch_mode=False)._argv()
    assert "BatchMode=yes" not in argv


def test_describe():
    assert SSHTransport("torch").describe() == "torch"
    assert SSHTransport("torch", user="bob").describe() == "torch (user bob)"


def test_missing_probe_script_is_reported(tmp_path):
    transport = SSHTransport("torch", probe_path=tmp_path / "nope.py")
    with pytest.raises(TransportError, match="cannot read probe script"):
        transport.fetch()


# ------------------------------------------------------------------ success


def test_fetch_returns_payload(tmp_path):
    transport = transport_for(tmp_path, envelope(payload()))
    result = transport.fetch()
    assert result.payload["slurm_version"] == "25.05.4"
    assert result.schema == SCHEMA_VERSION
    assert result.elapsed_sec >= 0


def test_fetch_ignores_login_banner_noise(tmp_path):
    """A motd banner on stdout must not corrupt the payload."""
    noisy = "\n".join(
        [
            "Welcome to NYU Torch!",
            "Your conda env is out of date.",
            "conda: command not found",
            envelope(payload(jobs=[{"job_id": 1}])),
        ]
    )
    result = transport_for(tmp_path, noisy).fetch()
    assert len(result.payload["jobs"]) == 1


def test_fetch_ignores_trailing_noise(tmp_path):
    noisy = envelope(payload()) + "\nlogout\nConnection to torch closed.\n"
    assert transport_for(tmp_path, noisy).fetch().schema == SCHEMA_VERSION


def test_fetch_surfaces_stderr_without_failing(tmp_path):
    result = transport_for(tmp_path, envelope(payload()), stderr_text="minor warning").fetch()
    assert result.stderr == "minor warning"


# ------------------------------------------------------------------ payload slicing


def test_extract_payload_requires_sentinels():
    assert SSHTransport._extract_payload("no markers here") is None


def test_extract_payload_accepts_missing_end_marker():
    """If the connection drops after the JSON, we still salvage the snapshot."""
    truncated = f"{JSON_BEGIN}\n{json.dumps(payload())}"
    assert SSHTransport._extract_payload(truncated)["user"] == "testuser"


def test_extract_payload_rejects_non_dict():
    assert SSHTransport._extract_payload(f"{JSON_BEGIN}\n[1,2]\n{JSON_END}") is None


# ------------------------------------------------------------------ failures


def test_ssh_connection_failure_is_explicit(tmp_path):
    transport = transport_for(tmp_path, "", returncode=255, stderr_text="Permission denied")
    with pytest.raises(TransportError, match="ssh to cluster failed"):
        transport.fetch()


def test_unparseable_output_reports_what_it_saw(tmp_path):
    transport = transport_for(tmp_path, "totally unexpected output")
    with pytest.raises(TransportError, match="no JSON payload"):
        transport.fetch()


def test_schema_mismatch_is_rejected(tmp_path):
    transport = transport_for(tmp_path, envelope(payload(schema=SCHEMA_VERSION + 1)))
    with pytest.raises(TransportError, match="schema"):
        transport.fetch()


def test_timeout_is_reported(tmp_path):
    slow = write_script(
        tmp_path / "slow_ssh",
        "#!/bin/sh\ncat > /dev/null\nsleep 10\n",
    )
    transport = SSHTransport("cluster", ssh_binary=slow, command_timeout=1)
    with pytest.raises(TransportError, match="timed out"):
        transport.fetch()


def test_missing_ssh_binary_is_reported():
    transport = SSHTransport("cluster", ssh_binary="/nonexistent/ssh")
    with pytest.raises(TransportError, match="ssh binary not found"):
        transport.fetch()
