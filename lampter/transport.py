"""SSH transport: exactly one round trip per refresh.

Latency dominates everything here. The cluster is reached over SSH, so issuing a
dozen small commands (``squeue``, then ``sinfo``, then ``sacct`` ...) would cost a
dozen round trips and make a 5-second refresh interval impossible on a WAN link.

Instead the *entire* probe script is piped to the login node's ``python3`` in a
single connection, and one JSON document comes back::

    ssh <host> python3 - --user <name>   <  remote_probe.py

The user's ``~/.ssh/config`` already sets up ControlMaster/ControlPersist for the
``torch`` host, so after the first connection subsequent refreshes reuse the
multiplexed socket and complete in milliseconds.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .remote_probe import (
    DEFAULT_SECTIONS,
    JSON_BEGIN,
    JSON_END,
    SCHEMA_VERSION,
    loads_lenient,
)

#: ssh reserves exit code 255 for its own failures (DNS, auth, refused, timeout),
#: which is worth distinguishing from a probe that ran but reported no data.
SSH_FAILURE_EXIT = 255

#: How much of a surprising stdout we keep for the error message.
DIAGNOSTIC_CHARS = 400


class TransportError(RuntimeError):
    """Raised when the cluster could not be reached or answered unintelligibly."""


@dataclass(frozen=True)
class ProbeResult:
    """A successful probe: the parsed payload plus how long it took."""

    payload: dict
    elapsed_sec: float
    raw_stdout: str = ""
    stderr: str = ""

    @property
    def schema(self) -> int | None:
        value = self.payload.get("schema")
        return value if isinstance(value, int) else None


def _probe_source_path() -> Path:
    return Path(__file__).with_name("remote_probe.py")


class SSHTransport:
    """Runs the remote probe on the cluster and returns its JSON document."""

    def __init__(
        self,
        host: str,
        *,
        user: str | None = None,
        ssh_binary: str = "ssh",
        connect_timeout: int = 15,
        command_timeout: int = 60,
        batch_mode: bool = True,
        history_hours: int = 12,
        probe_path: Path | None = None,
    ) -> None:
        self.host = host
        #: ``None`` means "let the remote login decide" (i.e. the SSH user).
        self.user = user or None
        self.ssh_binary = ssh_binary
        self.connect_timeout = connect_timeout
        self.command_timeout = command_timeout
        self.batch_mode = batch_mode
        self.history_hours = history_hours
        self._probe_path = probe_path or _probe_source_path()
        self._source_cache: str | None = None

    # ---------------------------------------------------------------- helpers

    def probe_source(self) -> str:
        """The remote script's text, cached after the first read."""
        if self._source_cache is None:
            try:
                self._source_cache = self._probe_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise TransportError(f"cannot read probe script {self._probe_path}: {exc}") from exc
        return self._source_cache

    def _ssh_prefix(self) -> list[str]:
        """The ssh binary plus its options, without a destination."""
        argv = [self.ssh_binary]
        if self.batch_mode:
            # Fail fast instead of blocking the dashboard on a password prompt.
            argv += ["-o", "BatchMode=yes"]
        argv += [
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            # Bound a connection that goes silent mid-transfer.
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
        ]
        return argv

    def _argv(self, sections: tuple[str, ...] | None = None) -> list[str]:
        argv = self._ssh_prefix()
        argv.append(self.host)
        argv.append("python3")
        argv.append("-")
        if self.user:
            argv += ["--user", self.user]
        # Selecting sections is how a fast refresh stays cheap: a jobs-only poll
        # skips the sinfo and sacct invocations entirely, without adding a round trip.
        argv += ["--sections", ",".join(sections or DEFAULT_SECTIONS)]
        if self.history_hours and self.history_hours != 12:
            argv += ["--history-hours", str(self.history_hours)]
        return argv

    def describe(self) -> str:
        """A one-line summary of the target, for the status bar."""
        suffix = f" (user {self.user})" if self.user else ""
        return f"{self.host}{suffix}"

    # ---------------------------------------------------------------- fetch

    def fetch(self, sections: tuple[str, ...] | None = None) -> ProbeResult:
        """Run the probe once and return the parsed payload.

        ``sections`` selects which collectors the remote probe runs; omitted means
        all of them. ``jobs`` is always cheap, while ``partitions`` and ``history``
        cost an extra SLURM invocation each.

        Raises :class:`TransportError` with a message meant to be shown to the user
        verbatim -- it is the only place that knows *why* a refresh failed.
        """
        argv = self._argv(sections)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                input=self.probe_source(),
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
                # Return codes are inspected below rather than raised on, because a
                # probe that ran but reported a problem still yields usable data.
                check=False,
            )
        except FileNotFoundError as exc:
            raise TransportError(
                f"ssh binary not found ({self.ssh_binary!r}); set ssh_binary in your config"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TransportError(
                f"ssh to {self.host} timed out after {self.command_timeout}s "
                "(is the VPN up and the login node reachable?)"
            ) from exc
        elapsed = time.monotonic() - started

        stderr = (completed.stderr or "").strip()

        if completed.returncode == SSH_FAILURE_EXIT:
            detail = stderr or "no diagnostic output"
            raise TransportError(f"ssh to {self.host} failed: {detail}")

        payload = self._extract_payload(completed.stdout or "")

        if completed.returncode != 0 and payload is None:
            raise TransportError(
                f"probe on {self.host} exited {completed.returncode}: {stderr or 'no output'}"
            )

        if payload is None:
            head = (completed.stdout or "").strip()[:DIAGNOSTIC_CHARS]
            raise TransportError(
                f"probe on {self.host} returned no JSON payload"
                + (f"; stdout began with: {head!r}" if head else "")
            )

        version = payload.get("schema")
        if isinstance(version, int) and version != SCHEMA_VERSION:
            raise TransportError(
                f"probe schema {version} does not match this client "
                f"({SCHEMA_VERSION}); update lampter on both sides"
            )

        return ProbeResult(
            payload=payload,
            elapsed_sec=elapsed,
            raw_stdout=completed.stdout or "",
            stderr=stderr,
        )

    @staticmethod
    def _extract_payload(stdout: str) -> dict | None:
        """Slice the JSON out from between the sentinels.

        Login nodes are noisy: motd banners, conda notices and welcome messages all
        land on stdout. Rather than pretending that never happens, the probe fences
        its output and we take only what is inside the fence.
        """
        begin = stdout.find(JSON_BEGIN)
        if begin < 0:
            return None
        body = stdout[begin + len(JSON_BEGIN) :]
        end = body.find(JSON_END)
        if end >= 0:
            body = body[:end]
        return loads_lenient(body)

    # ---------------------------------------------------------------- log streaming

    def stream_file(
        self,
        path: str,
        *,
        lines: int = 200,
        follow: bool = True,
    ) -> Iterator[str]:
        """Yield a remote file's lines, optionally following it as it grows.

        Deliberately *not* part of :meth:`fetch`. Following a log holds a channel
        open indefinitely, which is a different shape from the one-round-trip-per-
        refresh model everything else uses, so it is a separate call made only when
        somebody actually asks to watch a log. Closing the generator stops the
        remote ``tail``.

        Raises :class:`TransportError` if the file cannot be read.
        """
        command = ["tail", "-n", str(max(1, int(lines)))]
        if follow:
            command.append("-f")
        command.append(path)
        # ssh joins its command arguments with spaces and hands the result to the
        # remote login shell, so the quoting has to be baked into a SINGLE argument.
        # Passing ["sh", "-c", cmd] instead arrives as `sh -c tail -n 5 /path`, where
        # the shell runs `sh -c tail` and tail silently reads stdin instead of the
        # file -- producing no output and no error, which is exactly how this was
        # found. shlex.join makes it one argument the remote shell reassembles.
        argv = [*self._ssh_prefix(), self.host, shlex.join(command)]

        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # Line buffering: a follow needs each line as it arrives rather
                # than an 8 KB block at a time.
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise TransportError(
                f"ssh binary not found ({self.ssh_binary!r}); set ssh_binary in your config"
            ) from exc

        try:
            if process.stdout is None:  # pragma: no cover - defensive
                raise TransportError("could not read the remote stream")
            for line in process.stdout:
                yield line.rstrip("\n")
        finally:
            self._stop_process(process)

        # Only reached when the remote command ended on its own: either a
        # non-following read, or a failure such as a missing file.
        if process.returncode not in (0, None):
            detail = ""
            if process.stderr is not None:
                first = process.stderr.read().strip().splitlines()
                detail = first[0] if first else ""
            raise TransportError(
                f"could not read {path}: "
                f"{detail or f'remote command exited {process.returncode}'}"
            )

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        """Terminate a streaming subprocess politely, then firmly."""
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn ssh
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
