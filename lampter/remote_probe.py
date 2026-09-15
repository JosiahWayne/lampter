#!/usr/bin/env python3
"""Remote-side data collector for lampter.

This module is **never imported** by the local application. Its *source text* is
piped over stdin to the cluster's own ``python3``::

    ssh <host> python3 - --user <name>

Two consequences shape everything below:

* It must run on whatever ``python3`` the login node happens to ship, so it uses
  the standard library only -- no third-party imports, ever.
* It must survive login-shell noise. Cluster login nodes frequently print motd
  banners or warnings to stdout, which would corrupt a naive JSON payload. So the
  document is wrapped in sentinels and the reader on the Mac side slices between
  them. Diagnostics go to stderr, never stdout.

Output contract (stdout)::

    <<<LAMPTER_JSON>>>
    {"schema": 1, "jobs": [...], ...}
    <<<LAMPTER_JSON_END>>>

The probe is deliberately tolerant: every collector records its failure into
``errors`` and the probe still exits 0 with whatever it managed to gather, so a
partial outage degrades the dashboard instead of blanking it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

#: Bumped whenever the payload gains or renames keys. The client refuses a payload
#: whose schema it does not understand rather than silently showing partial data.
SCHEMA_VERSION = 5

#: How far back `sacct` is asked for job outcomes.
HISTORY_HOURS = 12

#: Default window for per-account consumption totals.
ACCOUNT_DAYS = 7

#: Collectors that can be requested individually.
#:
#: This exists because of load, not tidiness. A 15-second refresh running all four
#: collectors costs the shared controller roughly a thousand SLURM invocations an
#: hour from a single user. Splitting them up lets the client poll the job list
#: often (it changes constantly and is cheap) while asking for capacity, history and
#: live resource use far less often. The result is still one SSH round trip.
ALL_SECTIONS = ("jobs", "partitions", "history", "usage", "accounts")
DEFAULT_SECTIONS = ALL_SECTIONS

#: States in which a job is holding resources, and so has something for `sstat` to
#: report. Mirrors ``models.STARTED_STATES``.
RUNNING_STATES = frozenset(
    {
        "RUNNING",
        "COMPLETING",
        "CONFIGURING",
        "RESIZING",
        "SUSPENDED",
        "STOPPED",
        "STAGE_OUT",
    }
)

#: `sstat` fields to request. Kept to the ones with a clear meaning and a stable
#: format; the full default set is dozens of columns wide.
SSTAT_FIELDS = (
    "JobID,MaxRSS,AveCPU,NTasks,MaxDiskRead,MaxDiskWrite,"
    # TRESUsageInAve carries gres/gpuutil and gres/gpumem, i.e. the very numbers
    # NYU's low-utilisation policy is enforced on. Requesting one more field on the
    # call we already make costs nothing.
    "TRESUsageInAve"
)

#: Values past these bounds are discarded rather than displayed. Torch's `extern`
#: step reports nonsense for some array tasks -- an `AveCPU` of
#: ``213503982334-14:25:51`` (about 1.8e16 seconds) was observed -- and showing that
#: as "CPU time" would be worse than showing nothing.
MAX_PLAUSIBLE_CPU_SEC = 10 * 365 * 86400
MAX_PLAUSIBLE_RSS_KB = 100 * 1024**3  # 100 TB, in KB

JSON_BEGIN = "<<<LAMPTER_JSON>>>"
JSON_END = "<<<LAMPTER_JSON_END>>>"

#: Where Slurm lives when it is not on the login node's PATH.
SLURM_BIN_DIRS = ("/opt/slurm/bin", "/usr/local/slurm/bin", "/usr/bin")

#: Per-command ceiling. `squeue` on a busy controller can take a moment, but if
#: it takes longer than this something is wrong and we would rather report that
#: than hang the user's terminal.
COMMAND_TIMEOUT = 30


def log(message: str) -> None:
    """Write a diagnostic line to stderr (stdout is reserved for the payload)."""
    print(f"[probe] {message}", file=sys.stderr)


def find_tool(name: str) -> str | None:
    """Locate a Slurm binary, falling back to the well-known install dirs.

    The login node's non-interactive PATH usually contains /opt/slurm/bin, but
    that depends on how the site wires up /etc/profile, so we do not rely on it.
    """
    found = shutil.which(name)
    if found:
        return found
    for directory in SLURM_BIN_DIRS:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def run_command(argv: list[str], timeout: int = COMMAND_TIMEOUT) -> tuple[int, str, str]:
    """Run a command, returning ``(returncode, stdout, stderr)``.

    A timeout is reported as return code 124, mirroring GNU ``timeout``.
    """
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            # The caller decides what a failure means; see collect_jobs.
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except OSError as exc:  # binary vanished, permission denied, ...
        return 127, "", str(exc)


def loads_lenient(raw: str) -> dict | None:
    """Parse Slurm's JSON, tolerating junk printed before it.

    Some wrappers emit a warning line ahead of the document. We first try a
    straight parse, then fall back to decoding the first JSON object we can find.
    Only a JSON *object* is accepted: a bare array or scalar is valid JSON but is
    not a Slurm document, and returning it would break callers that expect ``.get``.
    """
    raw = raw.strip()
    if not raw:
        return None
    candidate: object
    try:
        candidate = json.loads(raw)
    except ValueError:
        start = raw.find("{")
        if start < 0:
            return None
        try:
            candidate, _ = json.JSONDecoder().raw_decode(raw[start:])
        except ValueError:
            return None
    return candidate if isinstance(candidate, dict) else None


def unwrap_number(field: object, *, zero_is_none: bool = False) -> int | None:
    """Flatten Slurm's ``{"set": bool, "infinite": bool, "number": int}`` fields.

    ``zero_is_none`` handles the awkward convention where timestamps for pending
    jobs come back as ``{"set": true, "number": 0}`` -- i.e. "set, but not really".
    """
    if isinstance(field, dict):
        if field.get("infinite"):
            return None
        if not field.get("set", False):
            return None
        value = field.get("number")
    elif isinstance(field, (int, float)):
        value = field
    else:
        return None
    if value is None:
        return None
    value = int(value)
    if zero_is_none and value == 0:
        return None
    return value


def is_infinite(field: object) -> bool:
    return isinstance(field, dict) and bool(field.get("infinite"))


def normalize_job(raw: dict) -> dict:
    """Flatten one ``squeue --json`` job record into the dashboard's wire format."""
    states = raw.get("job_state") or []
    state = states[0] if isinstance(states, list) and states else str(states or "UNKNOWN")
    return {
        "job_id": unwrap_number(raw.get("job_id"), zero_is_none=True),
        "array_job_id": unwrap_number(raw.get("array_job_id"), zero_is_none=True),
        "array_task_id": unwrap_number(raw.get("array_task_id"), zero_is_none=True),
        "array_task_string": raw.get("array_task_string") or "",
        "name": raw.get("name") or "",
        "user": raw.get("user_name") or "",
        "account": raw.get("account") or "",
        "partition": raw.get("partition") or "",
        "qos": raw.get("qos") or "",
        "state": state,
        "reason": raw.get("state_reason") or "",
        "state_description": raw.get("state_description") or "",
        "priority": unwrap_number(raw.get("priority")) or 0,
        # Timestamps are epoch seconds; 0 means "not applicable yet".
        "submit_time": unwrap_number(raw.get("submit_time"), zero_is_none=True),
        "eligible_time": unwrap_number(raw.get("eligible_time"), zero_is_none=True),
        "start_time": unwrap_number(raw.get("start_time"), zero_is_none=True),
        "end_time": unwrap_number(raw.get("end_time"), zero_is_none=True),
        # Time limit is in minutes; None means unlimited.
        "time_limit_min": unwrap_number(raw.get("time_limit"), zero_is_none=True),
        "time_limit_infinite": is_infinite(raw.get("time_limit")),
        "node_count": unwrap_number(raw.get("node_count")) or 0,
        "nodelist": raw.get("nodes") or "",
        "cpus": unwrap_number(raw.get("cpus")) or 0,
        "tres_req": raw.get("tres_req_str") or "",
        "tres_alloc": raw.get("tres_alloc_str") or "",
        "dependency": raw.get("dependency") or "",
        "stdout": raw.get("standard_output") or "",
        "stderr": raw.get("standard_error") or "",
        # Slurm reports both the raw `--output=log/slurm-%j.out` pattern and the
        # path it actually expanded to. Only the latter can be opened, so both are
        # carried across and the client prefers the expanded one.
        "stdout_expanded": raw.get("stdout_expanded") or "",
        "stderr_expanded": raw.get("stderr_expanded") or "",
        "workdir": raw.get("current_working_directory") or "",
    }


def note_version(document: object, meta: dict) -> None:
    """Record the Slurm version that every ``--json`` document reports for free.

    Slurm puts its version inside each document's ``meta`` block, so asking
    ``sinfo -V`` separately would be a wasted invocation against a shared
    controller on every single refresh.
    """
    if meta.get("slurm_version"):
        return
    block = (document or {}).get("meta") if isinstance(document, dict) else None
    if not isinstance(block, dict):
        return
    for key in ("Slurm_version", "slurm_version", "version"):
        value = block.get(key)
        if value:
            meta["slurm_version"] = str(value)
            return


def collect_jobs(
    user: str,
    errors: list[str],
    timings: dict[str, int],
    meta: dict | None = None,
) -> list[dict]:
    """Collect the user's active jobs via ``squeue --json``."""
    squeue = find_tool("squeue")
    if not squeue:
        errors.append(
            "squeue not found on the login node (looked on PATH and in "
            + ", ".join(SLURM_BIN_DIRS)
            + ")"
        )
        return []

    started = time.time()
    code, out, err = run_command([squeue, "--json", "--user", user])
    timings["squeue"] = int((time.time() - started) * 1000)

    if code != 0:
        detail = (err or out).strip().splitlines()
        errors.append(f"squeue failed (rc={code}): {detail[0] if detail else 'no output'}")
        return []

    document = loads_lenient(out)
    if document is None:
        errors.append("could not parse squeue JSON output (unexpected Slurm version?)")
        return []

    if meta is not None:
        note_version(document, meta)

    for warning in document.get("warnings") or []:
        errors.append(f"slurm warning: {warning}")

    jobs = document.get("jobs")
    if not isinstance(jobs, list):
        errors.append("squeue returned no 'jobs' array")
        return []
    return [normalize_job(job) for job in jobs if isinstance(job, dict)]


def collect_version(errors: list[str], meta: dict) -> str | None:
    """Report the controller's Slurm version, for context when parsing breaks.

    Prefers the version already embedded in a ``--json`` document's ``meta`` block
    (see :func:`note_version`). Only when no document was fetched does this fall back
    to invoking ``sinfo``, so the common refresh costs no extra SLURM call.
    """
    cached = meta.get("slurm_version")
    if cached:
        return str(cached)

    sinfo = find_tool("sinfo")
    if not sinfo:
        return None
    code, out, _ = run_command([sinfo, "-V"], timeout=15)
    if code != 0:
        return None
    text = out.strip()
    if not text:
        return None
    # "slurm 25.05.4"
    return text.split()[-1]


# --------------------------------------------------------------------- partitions

#: Node states that mean the node cannot run anything, however free its CPUs look.
#: Slurm reports these alongside scheduling state, e.g.
#: ``["DOWN", "DRAIN", "NOT_RESPONDING"]``.
BAD_NODE_STATES = frozenset(
    {
        "DOWN",
        "DRAIN",
        "FAIL",
        "FAILING",
        "NOT_RESPONDING",
        "UNKNOWN",
        "MAINT",
        "INVAL",
        "ERROR",
        "POWERED_DOWN",
        "POWERING_DOWN",
        "REBOOT_ISSUED",
        "REBOOT_REQUESTED",
        "CANCELLED",
        "COMPLETING",
        "FUTURE",
    }
)

#: Matches the several spellings of a GPU gres entry: a bare ``gpu:4``, a typed
#: ``gpu:h200:8``, and an allocation listing ``gpu:h200:4(IDX:2-3,5,7)``.
_GRES_GPU = re.compile(r"gpu:(?:(?P<type>[A-Za-z0-9_]+):)?(?P<count>\d+)")


def gres_gpu_counts(value: str | None) -> tuple[int, str | None]:
    """Sum GPUs in a Slurm gres string: ``gpu:h200:8(S:0-1)`` -> ``(8, "h200")``.

    Returns ``(0, None)`` for the empty and ``(null)`` values that GPU-less
    partitions report.
    """
    if not value or value == "(null)":
        return 0, None
    total = 0
    gpu_type: str | None = None
    for match in _GRES_GPU.finditer(value):
        if match.group("type"):
            gpu_type = gpu_type or match.group("type")
        total += int(match.group("count"))
    return total, gpu_type


def classify_node_states(states: object) -> str:
    """Reduce a node's state list to ``idle`` / ``mixed`` / ``allocated`` / ``other``."""
    lowered = {str(state).upper() for state in (states or [])}  # type: ignore[union-attr]
    if lowered & BAD_NODE_STATES:
        return "other"
    if "MIXED" in lowered:
        return "mixed"
    if "ALLOCATED" in lowered:
        return "allocated"
    if "IDLE" in lowered:
        return "idle"
    return "other"


def collect_partitions(
    errors: list[str],
    timings: dict[str, int],
    meta: dict | None = None,
) -> list[dict]:
    """Aggregate ``sinfo --json`` into one capacity record per partition.

    ``sinfo`` emits one entry per *group* of identical nodes (same partition, state
    and features), so Torch's 220 entries collapse into a couple of dozen partitions
    by summing over the entries sharing a name. Doing that here rather than on the
    Mac keeps the payload small, which is the entire point of the probe.
    """
    sinfo = find_tool("sinfo")
    if not sinfo:
        errors.append("sinfo not found on the login node")
        return []

    started = time.time()
    code, out, err = run_command([sinfo, "--json"])
    timings["sinfo"] = int((time.time() - started) * 1000)
    if code != 0:
        detail = (err or out).strip().splitlines()
        errors.append(f"sinfo failed (rc={code}): {detail[0] if detail else 'no output'}")
        return []

    document = loads_lenient(out)
    if document is None:
        errors.append("could not parse sinfo JSON output")
        return []

    if meta is not None:
        note_version(document, meta)

    buckets: dict[str, dict] = {}
    for entry in document.get("sinfo") or []:
        if not isinstance(entry, dict):
            continue
        partition = entry.get("partition") or {}
        name = partition.get("name")
        if not name:
            continue

        bucket = buckets.get(name)
        if bucket is None:
            bucket = buckets[name] = {
                "name": name,
                "nodes_total": 0,
                "nodes_idle": 0,
                "nodes_mixed": 0,
                "nodes_allocated": 0,
                "nodes_other": 0,
                "cpus_total": 0,
                "cpus_allocated": 0,
                "mem_total_mb": 0,
                "mem_allocated_mb": 0,
                "gpus_total": 0,
                "gpus_used": 0,
                # Keyed by GPU model and scaled GPU count. Meta-partitions such as
                # `all` pool every generation on the cluster, so a single label
                # would be a lie.
                "gpu_type_counts": {},
                "max_time_min": None,
                "max_time_infinite": False,
            }

        nodes = entry.get("nodes") or {}
        node_count = int(nodes.get("total") or 0)
        bucket["nodes_total"] += node_count
        bucket["nodes_" + classify_node_states((entry.get("node") or {}).get("state"))] += (
            node_count
        )

        cpus = entry.get("cpus") or {}
        bucket["cpus_total"] += int(cpus.get("total") or 0)
        bucket["cpus_allocated"] += int(cpus.get("allocated") or 0)

        memory = entry.get("memory") or {}
        bucket["mem_allocated_mb"] += int(memory.get("allocated") or 0)
        # `maximum` is per-node memory, so it has to be scaled by the node count.
        bucket["mem_total_mb"] += int(memory.get("maximum") or 0) * node_count

        gres = entry.get("gres") or {}
        total_gpus, gpu_type = gres_gpu_counts(gres.get("total"))
        used_gpus, _ = gres_gpu_counts(gres.get("used"))
        # `gres.total` and `gres.used` are BOTH per-node values for the group (the
        # used string lists one representative node's allocated GPU indices), so
        # both must be scaled by the group size. Verified against
        # `scontrol show nodes` on Torch: h200 is 34 nodes x 8 = 272 GPUs with 256
        # in use, which only adds up when scaled this way.
        bucket["gpus_total"] += total_gpus * node_count
        bucket["gpus_used"] += used_gpus * node_count
        if gpu_type and total_gpus:
            scaled = total_gpus * node_count
            bucket["gpu_type_counts"][gpu_type] = (
                bucket["gpu_type_counts"].get(gpu_type, 0) + scaled
            )

        # Partition-wide limits: identical on every entry, so overwriting is fine.
        limit = (partition.get("maximums") or {}).get("time")
        if isinstance(limit, dict):
            bucket["max_time_infinite"] = bool(limit.get("infinite"))
            number = limit.get("number")
            if isinstance(number, (int, float)) and number:
                bucket["max_time_min"] = int(number)

    records = []
    for name in sorted(buckets):
        bucket = buckets[name]
        counts = bucket.pop("gpu_type_counts")
        # One model present is a fact worth stating; several means "mixed".
        if len(counts) == 1:
            bucket["gpu_type"] = next(iter(counts))
        elif counts:
            bucket["gpu_type"] = "mixed"
        else:
            bucket["gpu_type"] = None
        bucket["gpu_types"] = sorted(counts)
        records.append(bucket)
    return records


# --------------------------------------------------------------------- history


def _block(value: object) -> dict:
    """Read a JSON field as a mapping, whatever shape the controller sent.

    This exists because ``--json`` output is *versioned*, not stable: Slurm's
    ``data_parser`` schema moves between releases, and the same field changes shape.
    ``exit_code`` is an object on 23.11+ and a bare integer before it. Reading a scalar
    as if it were a mapping raises ``AttributeError``, and because history is collected
    inside :func:`build_payload` with no per-collector guard, that used to abort the
    *entire* probe: every section came back empty, and the client then treated them as
    freshly fetched and threw away the last good snapshot.
    """
    return value if isinstance(value, dict) else {}


def normalize_history_job(raw: dict) -> dict:
    """Flatten one ``sacct --json`` record into the dashboard's wire format.

    Note that sacct nests every timestamp under ``time`` (unlike ``squeue``, which
    puts them at the top level), and reports state as ``{"current": [...],
    "reason": ...}``.

    Every nested field is read defensively through :func:`_block`, because the schema
    is versioned and a wrong guess must degrade one field rather than lose the snapshot.
    Where the controller sends a *scalar* instead of the object -- an integer
    ``exit_code``, a plain string ``state`` -- the value is kept rather than discarded,
    since the information is still meaningful.
    """
    state_block = _block(raw.get("state"))
    if state_block:
        states = state_block.get("current") or []
        state = states[0] if isinstance(states, list) and states else str(states or "UNKNOWN")
    else:
        state = str(raw.get("state") or "UNKNOWN")

    time_block = _block(raw.get("time"))
    array = _block(raw.get("array"))

    exit_block = _block(raw.get("exit_code"))
    if exit_block:
        statuses = exit_block.get("status") or []
        exit_status = statuses[0] if isinstance(statuses, list) and statuses else ""
        # A return code of 0 is meaningful, so it is not mapped to None.
        exit_code = unwrap_number(exit_block.get("return_code"))
        exit_signal = unwrap_number(_block(exit_block.get("signal")).get("id"))
    else:
        # A bare integer is the whole exit code: no status word, no signal.
        exit_status = ""
        exit_code = unwrap_number(raw.get("exit_code"))
        exit_signal = None

    gpus = 0
    allocated = _block(raw.get("tres")).get("allocated") or []
    for item in allocated if isinstance(allocated, list) else []:
        if isinstance(item, dict) and item.get("type") == "gres" and item.get("name") == "gpu":
            gpus += int(item.get("count") or 0)

    return {
        "job_id": unwrap_number(raw.get("job_id"), zero_is_none=True),
        # `array.job_id` is the array master even for a single task's record, so
        # this mirrors how squeue reports array tasks.
        "array_job_id": unwrap_number(array.get("job_id"), zero_is_none=True),
        "array_task_id": unwrap_number(array.get("task_id")),
        "name": raw.get("name") or "",
        "partition": raw.get("partition") or "",
        "account": raw.get("account") or "",
        "qos": raw.get("qos") or "",
        "state": state,
        "reason": state_block.get("reason") or "",
        "exit_status": exit_status,
        "exit_code": exit_code,
        "exit_signal": exit_signal,
        "submit_time": unwrap_number(time_block.get("submission"), zero_is_none=True),
        "eligible_time": unwrap_number(time_block.get("eligible"), zero_is_none=True),
        "start_time": unwrap_number(time_block.get("start"), zero_is_none=True),
        "end_time": unwrap_number(time_block.get("end"), zero_is_none=True),
        "elapsed_sec": unwrap_number(time_block.get("elapsed")),
        "time_limit_min": unwrap_number(time_block.get("limit"), zero_is_none=True),
        "nodelist": raw.get("nodes") or "",
        "node_count": unwrap_number(raw.get("allocation_nodes")) or 0,
        "gpus": gpus,
    }


def collect_history(
    user: str,
    hours: int,
    errors: list[str],
    timings: dict[str, int],
) -> list[dict]:
    """Collect recent job outcomes via ``sacct --json``.

    This is what tells us *how* a job ended. ``squeue`` only ever shows live jobs, so
    without sacct a job that fails between two refreshes would simply disappear with
    no indication that anything went wrong.
    """
    sacct = find_tool("sacct")
    if not sacct:
        errors.append("sacct not found on the login node")
        return []

    started = time.time()
    code, out, err = run_command(
        [
            sacct,
            "--json",
            "--user",
            user,
            "--starttime",
            f"now-{hours}hours",
            # Allocations only. Without this every job *step* is returned as well,
            # which multiplies the payload for no extra information.
            "--allocations",
        ]
    )
    timings["sacct"] = int((time.time() - started) * 1000)
    if code != 0:
        detail = (err or out).strip().splitlines()
        errors.append(f"sacct failed (rc={code}): {detail[0] if detail else 'no output'}")
        return []

    document = loads_lenient(out)
    if document is None:
        errors.append("could not parse sacct JSON output")
        return []
    return [
        normalize_history_job(job)
        for job in (document.get("jobs") or [])
        if isinstance(job, dict)
    ]


# --------------------------------------------------------------------- live usage

#: Suffix multipliers for `sstat` size fields, which arrive as ``4491536K``.
_SIZE_UNITS = {
    "K": 1,
    "M": 1024,
    "G": 1024**2,
    "T": 1024**3,
    "P": 1024**4,
}


def parse_size_kb(value: str | None) -> int | None:
    """Parse an ``sstat`` size such as ``4491536K``, ``1.5G`` or ``0`` into KB.

    Returns ``None`` for an empty field, which is how ``sstat`` reports a metric the
    step does not have (the ``extern`` step of an array task reports no ``MaxRSS``).
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    multiplier = 1
    if text[-1].isalpha():
        multiplier = _SIZE_UNITS.get(text[-1].upper(), 0)
        if not multiplier:
            return None
        text = text[:-1]

    try:
        number = float(text)
    except ValueError:
        return None
    return int(number * multiplier)


def parse_cpu_seconds(value: str | None) -> int | None:
    """Parse an ``sstat`` CPU time in Slurm's ``[[DD-]HH:]MM:SS`` notation.

    ``13:40:34`` is 49234 seconds; ``2-03:00:00`` is two days and three hours.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    days = 0
    if "-" in text:
        head, _, remainder = text.partition("-")
        try:
            days = int(head)
        except ValueError:
            return None
        text = remainder

    parts = text.split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None

    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours, minutes, seconds = 0, numbers[0], numbers[1]
    elif len(numbers) == 1:
        hours, minutes, seconds = 0, 0, numbers[0]
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _to_int(raw: object) -> int | None:
    try:
        return int(float(str(raw)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _plausible(value: int | None, limit: int, *, keep_zero: bool = True) -> int | None:
    """Drop an impossible measurement rather than showing it as fact."""
    if value is None:
        return None
    if value < 0 or value > limit:
        return None
    if value == 0 and not keep_zero:
        return None
    return value


def aggregate_usage(rows: list[list[str]]) -> list[dict]:
    """Fold ``sstat`` step rows into one record per job.

    Each job reports several *steps* (``.extern``, ``.batch``, and one per ``srun``),
    and the per-job figure is the **maximum across steps, not the sum**: the
    ``extern`` step already aggregates the whole job, so adding the ``batch`` step on
    top would double count. Measurements outside a plausible range are discarded
    because the controller occasionally reports garbage (see
    :data:`MAX_PLAUSIBLE_CPU_SEC`).

    ``rows`` are the ``JobID|MaxRSS|AveCPU|NTasks|MaxDiskRead|MaxDiskWrite`` tuples
    that ``sstat -P -n`` produces.
    """
    totals: dict[int, dict] = {}

    for row in rows:
        if len(row) < 6:
            continue
        # "17365570.batch" -> job 17365570, step "batch".
        step_id, _, _step_name = row[0].partition(".")
        try:
            job_id = int(step_id)
        except ValueError:
            continue

        record = totals.setdefault(
            job_id,
            {
                "job_id": job_id,
                "max_rss_kb": None,
                "cpu_seconds": None,
                "tasks": None,
                "disk_read_bytes": None,
                "disk_write_bytes": None,
                "gpu_util": None,
                "gpu_memory_mb": None,
                "steps": 0,
            },
        )
        record["steps"] += 1

        rss = _plausible(parse_size_kb(row[1]), MAX_PLAUSIBLE_RSS_KB)
        if rss is not None:
            record["max_rss_kb"] = max(record["max_rss_kb"] or 0, rss)

        cpu = _plausible(parse_cpu_seconds(row[2]), MAX_PLAUSIBLE_CPU_SEC)
        if cpu is not None:
            record["cpu_seconds"] = max(record["cpu_seconds"] or 0, cpu)

        try:
            tasks = int(row[3])
        except (TypeError, ValueError):
            tasks = None
        if tasks is not None:
            record["tasks"] = max(record["tasks"] or 0, tasks)

        for index, key in ((4, "disk_read_bytes"), (5, "disk_write_bytes")):
            size = _plausible(parse_size_kb(row[index]), MAX_PLAUSIBLE_RSS_KB * 1024)
            if size is not None:
                record[key] = max(record[key] or 0, size * 1024)

        # TRESUsageInAve is a `key=value,...` TRES string. gpuutil is POOLED over the
        # step's GPUs rather than per-GPU -- the client divides by the GPU count, and
        # flags the result when dividing yields an impossible figure. Keeping the raw
        # value here means the display can change without touching the probe.
        tres = split_tres(row[6]) if len(row) > 6 else {}
        gpu_util = _to_int(tres.get("gres/gpuutil"))
        if gpu_util is not None:
            # A deliberately loose ceiling: 100 per GPU, with room for a whole node's
            # worth. Its only job is to drop the garbage the controller sometimes
            # reports, and anything that survives is checked properly by the client.
            gpu_util = _plausible(gpu_util, 100 * 1024)
        if gpu_util is not None:
            record["gpu_util"] = max(record["gpu_util"] or 0, gpu_util)

        gpu_mem_kb = _plausible(parse_size_kb(tres.get("gres/gpumem")), MAX_PLAUSIBLE_RSS_KB)
        if gpu_mem_kb is not None:
            record["gpu_memory_mb"] = max(record["gpu_memory_mb"] or 0, gpu_mem_kb // 1024)

    return [totals[job_id] for job_id in sorted(totals)]


def collect_usage(
    job_ids: list[str],
    errors: list[str],
    timings: dict[str, int],
) -> list[dict]:
    """Live per-job resource use from ``sstat``.

    Three things about ``sstat`` shape this:

    * There is no ``--json`` (unlike ``squeue``, ``sinfo`` and ``sacct``), so the
      output is parsed from ``-P -n`` pipe-delimited text.
    * There is no ``-u``, so the job ids have to come from the ``squeue`` results the
      probe already collected.
    * ``-a`` is **required**: plain ``sstat -j <id>`` prints only a header, which
      looks like "no data" rather than "wrong invocation".

    The ids passed in must be the ``squeue --json`` ``job_id`` values, not the
    ``<array>_<task>`` display form: the display form matches every running task of
    the array at once and merges their steps together.
    """
    if not job_ids:
        # Nothing is running, so there is nothing to ask about. Skipping the call
        # entirely is the point of this check.
        return []

    sstat = find_tool("sstat")
    if not sstat:
        errors.append("sstat not found on the login node")
        return []

    started = time.time()
    code, out, err = run_command(
        [sstat, "-a", "-j", ",".join(job_ids), "-P", "-n", "-o", SSTAT_FIELDS]
    )
    timings["sstat"] = int((time.time() - started) * 1000)
    if code != 0:
        detail = (err or out).strip().splitlines()
        errors.append(f"sstat failed (rc={code}): {detail[0] if detail else 'no output'}")
        return []

    rows = [line.split("|") for line in out.splitlines() if line.strip()]
    return aggregate_usage(rows)


# --------------------------------------------------------------------- accounts

#: TRES keys that describe a GPU. Slurm spells these ``gres/gpu`` for an untyped
#: request and ``gres/gpu:h100`` when the model is pinned, and a job may have both.
_TRES_GPU_PREFIX = "gres/gpu"


def split_tres(value: str | None) -> dict[str, str]:
    """Split a ``key=value`` TRES string into a mapping.

    Used by ``sacctmgr GrpTRES``, ``sacctmgr MaxTRESPerUser`` and ``sacct AllocTRES``.
    Not to be confused with the *gres colon form* (``gres/gpu:1``) that ``squeue %b``
    and ``sinfo`` produce -- see :func:`count_gpus`, which accepts both because
    getting this wrong silently reports zero GPUs.
    """
    result: dict[str, str] = {}
    for chunk in (value or "").split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, _, raw = chunk.partition("=")
        key = key.strip()
        if key:
            result[key] = raw.strip()
    return result


def count_gpus(value: str | None) -> int:
    """GPUs in either of Slurm's two TRES spellings.

    ``sacctmgr`` and ``sacct`` use ``key=value`` (``gres/gpu=112``); ``squeue %b``
    uses the gres colon form (``gres/gpu:1``, ``gres/gpu:h200:1``) and ``N/A`` for a
    job with no GPU. Handling only one of them is how an earlier version of this
    section reported "0 GPUs" for jobs that were plainly holding them.
    """
    if not value:
        return 0
    total = 0
    for key, raw in split_tres(value).items():
        if key.startswith(_TRES_GPU_PREFIX):
            try:
                total += int(float(raw))
            except ValueError:
                continue
    # Add the gres colon form rather than falling back to it. A string is normally one
    # spelling or the other, and the colon regex cannot match `gres/gpu=1` (it requires
    # a colon), so summing cannot double count -- but it does cope with a mixed string.
    colon_total, _ = gres_gpu_counts(value)
    return total + colon_total


def tres_cpus(value: str | None) -> int:
    raw = split_tres(value).get("cpu")
    try:
        return int(float(raw)) if raw else 0
    except ValueError:
        return 0


#: Suffix multipliers for TRES memory values, which arrive as ``28000G`` or ``512000M``.
_TRES_MEMORY_UNITS = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0**2}


def tres_memory_mb(value: str | None) -> float | None:
    raw = split_tres(value).get("mem")
    if not raw:
        return None
    text = raw.strip()
    multiplier = _TRES_MEMORY_UNITS.get(text[-1:].upper())
    if multiplier is not None:
        text = text[:-1]
    else:
        multiplier = 1.0
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def collect_qos_limits(
    qos_names: list[str],
    errors: list[str],
    timings: dict[str, int],
) -> dict[str, dict]:
    """Per-QOS limits, from ``sacctmgr show qos where name=...``.

    The ``where`` filter is a correctness requirement, not an optimisation.
    Querying the full QOS list returns *inconsistent* values for
    ``MaxTRESPerUser``: three successive unfiltered reads reported ``gpu48`` as
    ``gres/gpu=2``, ``gres/gpu=0`` and ``gres/gpu=16``, while the filtered query
    reproducibly returns ``gres/gpu=16``. Since this field is what explains
    ``QOSMaxGRESPerUser``, a wrong number here would be worse than no number.

    Two different fields answer two different questions, and conflating them is the
    mistake to avoid:

    * ``GrpTRES`` -- the limit for the QOS as a **whole**, shared by every user in
      it. Hitting it produces ``QOSGrpGRES``.
    * ``MaxTRESPerUser`` -- the limit for **one user** inside the QOS. Hitting it
      produces ``QOSMaxGRESPerUser``, which on Torch is by far the more common block
      (``gpu48`` allows just 2 GPUs per user).

    ``GrpTRESMins`` is the classic "your project has N GPU-hours" budget. Torch does
    not configure one, so it is reported as a flag and the client hides the column
    rather than showing an empty one.
    """
    if not qos_names:
        return {}
    sacctmgr = find_tool("sacctmgr")
    if not sacctmgr:
        errors.append("sacctmgr not found on the login node")
        return {}

    started = time.time()
    code, out, err = run_command(
        [
            sacctmgr,
            "-n",
            "-P",
            "show",
            "qos",
            "where",
            f"name={','.join(sorted(set(qos_names)))}",
            "format=Name,GrpTRES,MaxTRESPerUser,GrpTRESMins,MaxTRES,MaxWall",
        ]
    )
    timings["qos"] = int((time.time() - started) * 1000)
    if code != 0:
        detail = (err or out).strip().splitlines()
        errors.append(f"sacctmgr show qos failed (rc={code}): {detail[0] if detail else ''}")
        return {}

    def field(parts: list[str], index: int) -> str:
        return parts[index].strip() if len(parts) > index else ""

    limits: dict[str, dict] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        name = parts[0].strip()
        if not name:
            continue
        group = field(parts, 1)
        per_user = field(parts, 2)
        budget = field(parts, 3)
        max_tres = field(parts, 4)
        limits[name] = {
            "name": name,
            "grp_gpus": count_gpus(group) or None,
            "grp_cpus": tres_cpus(group) or None,
            "grp_memory_mb": tres_memory_mb(group),
            "user_gpus": count_gpus(per_user) or None,
            "user_cpus": tres_cpus(per_user) or None,
            "user_memory_mb": tres_memory_mb(per_user),
            "max_gpus": count_gpus(max_tres) or None,
            "budget_configured": bool(budget and "=" in budget),
            "max_wall": field(parts, 5) or None,
        }
    return limits


def collect_associations(
    user: str,
    errors: list[str],
    timings: dict[str, int],
) -> dict[str, set[str]]:
    """The accounts I may submit to, and the QOS on each association."""
    sacctmgr = find_tool("sacctmgr")
    if not sacctmgr:
        return {}

    started = time.time()
    code, out, err = run_command(
        [
            sacctmgr,
            "-n",
            "-P",
            "show",
            "associations",
            "where",
            f"user={user}",
            "format=Account,QOS,Partition,Share,GrpTRES,GrpTRESMins,MaxTRES",
        ]
    )
    timings["associations"] = int((time.time() - started) * 1000)
    if code != 0:
        detail = (err or out).strip().splitlines()
        errors.append(
            f"sacctmgr show associations failed (rc={code}): "
            f"{detail[0] if detail else 'no output'}"
        )
        return {}

    accounts: dict[str, set[str]] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        account = parts[0].strip()
        if not account:
            continue
        qos_field = parts[1].strip() if len(parts) > 1 else ""
        qoses = accounts.setdefault(account, set())
        for name in qos_field.replace(",", " ").split():
            qoses.add(name)
    return accounts


def collect_sshare(
    accounts: list[str],
    errors: list[str],
    timings: dict[str, int],
) -> dict[str, dict]:
    """Per-account fairshare signals for this user, from ``sshare``.

    ``RawUsage`` is deliberately *not* surfaced: it is Slurm's decayed usage in
    arbitrary units, meaningless without the site's half-life and billing weights.
    ``EffectvUsage`` (0-1, comparable within the tree) and ``FairShare`` (0-1, the
    actual priority factor) say the same thing interpretably.
    """
    if not accounts:
        return {}
    sshare = find_tool("sshare")
    if not sshare:
        errors.append("sshare not found on the login node")
        return {}

    started = time.time()
    code, out, err = run_command(
        [
            sshare,
            "-A",
            ",".join(accounts),
            "-P",
            "-n",
            "-o",
            "Account,User,NormShares,RawUsage,EffectvUsage,FairShare,"
            "TRESRunMins,LevelFS,GrpTRESMins",
        ]
    )
    timings["sshare"] = int((time.time() - started) * 1000)
    if code != 0:
        detail = (err or out).strip().splitlines()
        errors.append(f"sshare failed (rc={code}): {detail[0] if detail else 'no output'}")
        return {}

    def number(raw: str) -> float | None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    records: dict[str, dict] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 9:
            continue
        account = parts[0].strip()
        if not account:
            continue
        records[account] = {
            "account": account,
            "norm_shares": number(parts[2]),
            "effectv_usage": number(parts[4]),
            "fairshare": number(parts[5]),
            # TRES x minutes for the jobs running *right now*. It grows with elapsed
            # time, so it is a burn rate rather than a concurrency count; the current
            # allocation comes from squeue instead.
            "tres_run_mins": {
                key: raw for key, raw in split_tres(parts[6]).items() if raw
            },
            "level_fs": number(parts[7]),
            "budget_configured": bool(parts[8].strip()),
        }
    return records


def collect_current_usage(
    user: str,
    errors: list[str],
    timings: dict[str, int],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Current cluster-wide running allocation, by account, by QOS, and by my QOS.

    One unfiltered ``squeue`` for running jobs answers all three questions, which is
    why this is a single call rather than one per account or per QOS. It is also the
    only honest source for "how much is in use right now": ``sshare``'s ``TRESRunMins``
    is an integral over time, not a count.

    Note that in squeue's format language ``%a`` is the **account** and ``%A`` is the
    array job id. Confusing them yields a table of job numbers.
    """
    squeue = find_tool("squeue")
    if not squeue:
        return {}, {}, {}

    started = time.time()
    code, out, err = run_command(
        [squeue, "-h", "-t", "RUNNING", "-o", "%u|%a|%q|%b|%C|%D"]
    )
    timings["squeue_running"] = int((time.time() - started) * 1000)
    if code != 0:
        detail = (err or out).strip().splitlines()
        errors.append(
            f"squeue -t RUNNING failed (rc={code}): {detail[0] if detail else 'no output'}"
        )
        return {}, {}, {}

    def bucket() -> dict:
        return {"jobs": 0, "gpus": 0, "cpus": 0, "nodes": 0, "users": set()}

    def add(table: dict[str, dict], key: str, gpus: int, cpus: int, nodes: int, who: str) -> None:
        entry = table.setdefault(key, bucket())
        entry["jobs"] += 1
        entry["gpus"] += gpus
        entry["cpus"] += cpus
        entry["nodes"] += nodes
        entry["users"].add(who)

    by_account: dict[str, dict] = {}
    by_qos: dict[str, dict] = {}
    my_by_qos: dict[str, dict] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        who, account, qos, tres, cpus, nodes = (part.strip() for part in parts[:6])
        try:
            cpu_count = int(cpus)
            node_count = int(nodes)
        except ValueError:
            continue
        # `%b` is TRES **per node**, so it has to be scaled by the node count: a two-node
        # job with `--gres=gpu:1` reports `gres/gpu:1` here and holds two GPUs. Getting
        # this wrong undercounts every per-account and per-QOS GPU figure, and therefore
        # the headroom derived from them, on any multi-node GPU job.
        #
        # The field is only documented in the long form (`-O tres-per-node`); the short
        # `%b` is marked internally as a "vestigial option" that "could be removed", per
        # SchedMD ticket 11239. `%C` and `%D` are job totals, so only the TRES needs it.
        gpus = count_gpus(tres) * max(1, node_count)

        if account:
            add(by_account, account, gpus, cpu_count, node_count, who)
        if qos:
            add(by_qos, qos, gpus, cpu_count, node_count, who)
            if user and who == user:
                add(my_by_qos, qos, gpus, cpu_count, node_count, who)

    for table in (by_account, by_qos, my_by_qos):
        for entry in table.values():
            entry["users"] = len(entry["users"])
    return by_account, by_qos, my_by_qos


def collect_account_window(
    accounts: list[str],
    days: int,
    errors: list[str],
    timings: dict[str, int],
) -> dict[str, dict]:
    """GPU-hours and CPU-hours consumed per account over a window, from ``sacct``.

    Parsable text rather than ``--json`` on purpose: three columns are needed and the
    JSON document for a week of accounting is an order of magnitude larger. With
    ``AllocTRES`` last, a stray space inside it cannot shift the earlier fields.
    """
    if not accounts:
        return {}
    sacct = find_tool("sacct")
    if not sacct:
        return {}

    started = time.time()
    code, out, err = run_command(
        [
            sacct,
            "-A",
            ",".join(accounts),
            "--starttime",
            f"now-{days}days",
            "--allocations",
            "-P",
            "-n",
            "-o",
            "Account,ElapsedRaw,AllocTRES",
        ]
    )
    timings["sacct_accounts"] = int((time.time() - started) * 1000)
    if code != 0:
        detail = (err or out).strip().splitlines()
        errors.append(
            f"sacct per account failed (rc={code}): {detail[0] if detail else 'no output'}"
        )
        return {}

    totals: dict[str, dict] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        account = parts[0].strip()
        try:
            elapsed = int(parts[1])
        except ValueError:
            continue
        entry = totals.setdefault(account, {"gpu_seconds": 0.0, "cpu_seconds": 0.0, "jobs": 0})
        entry["jobs"] += 1
        entry["gpu_seconds"] += elapsed * count_gpus(parts[2])
        entry["cpu_seconds"] += elapsed * tres_cpus(parts[2])
    return totals


def collect_accounts(
    jobs: list[dict],
    user: str,
    usage_days: int,
    errors: list[str],
    timings: dict[str, int],
) -> tuple[list[dict], list[dict]]:
    """Accounts I can use, what they consume, and the QOS pressure affecting them.

    Split deliberately in two, because the two kinds of limit are scoped differently
    and mixing them would be misleading: "how much does this account use" belongs to
    the account, while "how close is this QOS to its cap" belongs to the QOS.
    """
    associations = collect_associations(user, errors, timings)
    if not associations:
        # sacctmgr is often restricted or absent off-site. Say so and return nothing
        # rather than inventing an account list.
        errors.append("no account associations found; is sacctmgr available?")
        return [], []

    names = sorted(associations)

    # The QOSes worth reporting on are the ones this user can reach: those on the
    # associations, plus whatever the jobs in flight actually asked for. The second
    # half matters because Torch puts jobs in QOSes (gpu48, gpu168) that never appear
    # on an association -- and those are exactly the ones with per-user GPU caps.
    reachable = {qos for qoses in associations.values() for qos in qoses}
    reachable |= {str(job.get("qos")) for job in jobs if job.get("qos")}

    qos_limits = collect_qos_limits(sorted(reachable), errors, timings)
    shares = collect_sshare(names, errors, timings)
    by_account, by_qos, my_by_qos = collect_current_usage(user, errors, timings)
    window = collect_account_window(names, usage_days, errors, timings)

    accounts: list[dict] = []
    for name in names:
        share = shares.get(name, {})
        current = by_account.get(name, {})
        used = window.get(name, {})
        accounts.append(
            {
                "account": name,
                "qos": sorted(associations.get(name, set())),
                "running_jobs": current.get("jobs", 0),
                "running_gpus": current.get("gpus", 0),
                "running_cpus": current.get("cpus", 0),
                "running_nodes": current.get("nodes", 0),
                "running_users": current.get("users", 0),
                "window_days": usage_days,
                "window_gpu_hours": round(used.get("gpu_seconds", 0.0) / 3600, 2),
                "window_cpu_hours": round(used.get("cpu_seconds", 0.0) / 3600, 2),
                "window_jobs": used.get("jobs", 0),
                "norm_shares": share.get("norm_shares"),
                "effectv_usage": share.get("effectv_usage"),
                "fairshare": share.get("fairshare"),
                "level_fs": share.get("level_fs"),
                "tres_run_mins": share.get("tres_run_mins", {}),
                "budget_configured": bool(share.get("budget_configured")),
            }
        )

    pressure: list[dict] = []
    for name in sorted(reachable):
        limit = qos_limits.get(name, {})
        current = by_qos.get(name, {})
        mine = my_by_qos.get(name, {})
        pressure.append(
            {
                "name": name,
                "max_wall": limit.get("max_wall"),
                "group_gpu_limit": limit.get("grp_gpus"),
                "group_cpu_limit": limit.get("grp_cpus"),
                "group_memory_limit_mb": limit.get("grp_memory_mb"),
                "user_gpu_limit": limit.get("user_gpus"),
                "user_cpu_limit": limit.get("user_cpus"),
                "user_memory_limit_mb": limit.get("user_memory_mb"),
                "budget_configured": bool(limit.get("budget_configured")),
                "running_gpus": current.get("gpus", 0),
                "running_cpus": current.get("cpus", 0),
                "running_jobs": current.get("jobs", 0),
                "running_users": current.get("users", 0),
                "my_gpus": mine.get("gpus", 0),
                "my_cpus": mine.get("cpus", 0),
                "my_jobs": mine.get("jobs", 0),
            }
        )
    return accounts, pressure


def build_payload(
    user: str,
    history_hours: int = HISTORY_HOURS,
    account_days: int = ACCOUNT_DAYS,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
) -> dict:
    errors: list[str] = []
    timings: dict[str, int] = {}
    meta: dict = {}
    now = int(time.time())

    wanted = tuple(section for section in sections if section in ALL_SECTIONS)
    if not wanted:
        wanted = DEFAULT_SECTIONS

    # Every collector is isolated: one missing or unhappy tool degrades the
    # dashboard instead of blanking it. Sections the caller did not ask for are
    # skipped entirely, which is what keeps a fast refresh cheap on the controller.
    jobs = collect_jobs(user, errors, timings, meta) if "jobs" in wanted else []
    partitions = collect_partitions(errors, timings, meta) if "partitions" in wanted else []
    history = (
        collect_history(user, history_hours, errors, timings) if "history" in wanted else []
    )
    # `sstat` has no `-u`, so it needs the ids squeue just produced. This is why the
    # usage section depends on the jobs section rather than standing alone.
    usage = []
    if "usage" in wanted and jobs:
        running_ids = [
            str(job["job_id"])
            for job in jobs
            if job.get("job_id") and str(job.get("state", "")).upper() in RUNNING_STATES
        ]
        usage = collect_usage(running_ids, errors, timings)
    accounts: list[dict] = []
    qos_pressure: list[dict] = []
    if "accounts" in wanted:
        accounts, qos_pressure = collect_accounts(
            jobs, user, account_days, errors, timings
        )
    # Free in the common case: the version rides along in the JSON already fetched.
    version = collect_version(errors, meta)

    return {
        "schema": SCHEMA_VERSION,
        "generated_at": now,
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "hostname": socket.gethostname(),
        "user": user,
        "slurm_version": version,
        "history_hours": history_hours,
        "sections": list(wanted),
        "jobs": jobs,
        "partitions": partitions,
        "history": history,
        "usage": usage,
        "accounts": accounts,
        "qos": qos_pressure,
        "account_days": account_days,
        "errors": errors,
        "timings_ms": timings,
    }


def _to_hours(raw: object) -> int:
    """Clamp a requested history window to something that will not flood the wire."""
    try:
        value = int(float(str(raw)))
    except (TypeError, ValueError):
        return HISTORY_HOURS
    return max(1, min(value, 24 * 7))


def _to_days(raw: object) -> int:
    """Clamp the per-account accounting window to something the controller can serve."""
    try:
        value = int(float(str(raw)))
    except (TypeError, ValueError):
        return ACCOUNT_DAYS
    return max(1, min(value, 90))


def parse_sections(raw: str) -> tuple[str, ...]:
    """Parse ``--sections``. Unknown names are dropped rather than fatal."""
    names = tuple(
        part.strip().lower() for part in raw.split(",") if part.strip()
    )
    known = tuple(name for name in names if name in ALL_SECTIONS)
    return known or DEFAULT_SECTIONS


def parse_args(argv: list[str]) -> dict:
    """Tiny hand-rolled parser -- argparse output would land on stdout."""
    options = {
        "user": None,
        "history_hours": HISTORY_HOURS,
        "account_days": ACCOUNT_DAYS,
        "sections": DEFAULT_SECTIONS,
    }
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("--user", "-u"):
            index += 1
            if index < len(argv):
                options["user"] = argv[index]
        elif arg.startswith("--user="):
            options["user"] = arg.split("=", 1)[1]
        elif arg == "--history-hours":
            index += 1
            if index < len(argv):
                options["history_hours"] = _to_hours(argv[index])
        elif arg.startswith("--history-hours="):
            options["history_hours"] = _to_hours(arg.split("=", 1)[1])
        elif arg == "--account-days":
            index += 1
            if index < len(argv):
                options["account_days"] = _to_days(argv[index])
        elif arg.startswith("--account-days="):
            options["account_days"] = _to_days(arg.split("=", 1)[1])
        elif arg == "--sections":
            index += 1
            if index < len(argv):
                options["sections"] = parse_sections(argv[index])
        elif arg.startswith("--sections="):
            options["sections"] = parse_sections(arg.split("=", 1)[1])
        index += 1
    return options


def resolve_user(explicit: str | None) -> str:
    """Explicit flag, then env override, then whoever the SSH session logs in as."""
    if explicit:
        return explicit
    for key in ("LAMPTER_USER", "USER", "LOGNAME"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        import getpass

        return getpass.getuser()
    except Exception:  # pragma: no cover - exotic environments only
        return ""


def emit(payload: dict) -> None:
    """Write the sentinel-wrapped payload to stdout."""
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    sys.stdout.write(f"{JSON_BEGIN}\n{body}\n{JSON_END}\n")
    sys.stdout.flush()


def main(argv: list[str]) -> int:
    options = parse_args(argv)
    user = resolve_user(options["user"])
    if not user:
        emit(
            {
                "schema": SCHEMA_VERSION,
                "generated_at": int(time.time()),
                "hostname": socket.gethostname(),
                "user": "",
                "slurm_version": None,
                # An empty list, not a missing key: the client infers "what was
                # fetched" from this field, and a missing one would make it believe
                # these empty tables were fresh data and drop the last good snapshot.
                "sections": [],
                "jobs": [],
                "partitions": [],
                "history": [],
                "usage": [],
                "accounts": [],
                "qos": [],
                "errors": ["could not determine the remote username"],
                "timings_ms": {},
            }
        )
        return 0

    try:
        payload = build_payload(
            user,
            options["history_hours"],
            options["account_days"],
            options["sections"],
        )
    except Exception as exc:  # last-resort guard: always emit valid JSON
        payload = {
            "schema": SCHEMA_VERSION,
            "generated_at": int(time.time()),
            "hostname": socket.gethostname(),
            "user": user,
            "slurm_version": None,
            # `sections` is deliberately empty, so a crash costs a refresh rather than
            # the table: the client keeps carrying the previous data, and the error
            # below is what the user sees.
            "sections": [],
            "jobs": [],
            "partitions": [],
            "history": [],
            "usage": [],
            "accounts": [],
            "qos": [],
            "errors": [f"probe crashed: {type(exc).__name__}: {exc}"],
            "timings_ms": {},
        }

    emit(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
