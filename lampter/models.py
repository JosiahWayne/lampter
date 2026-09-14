"""Domain model: turn the probe's wire records into jobs with derived metrics.

The two numbers this tool exists to report are the **queue wait** ("排队时长") and
the **run time** of each job. Slurm's own output does not carry a run-time field,
and its ``--json`` document only exposes epoch timestamps, so both are derived
here from ``submit_time`` / ``start_time`` / ``end_time``.

Keeping that arithmetic in one testable place (rather than in the widgets) is the
point of this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .tres import gpu_count, gpu_type, memory, memory_mb

#: States in which a job occupies resources and therefore has a run time.
STARTED_STATES = frozenset(
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

#: States that mean the job is finished and will drop out of squeue shortly.
TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "OUT_OF_MEMORY",
    }
)

#: Terminal states that mean something went wrong, as opposed to a clean finish or
#: a deliberate cancellation. These are what raise alerts.
FAILURE_STATES = frozenset(
    {
        "FAILED",
        "NODE_FAIL",
        "TIMEOUT",
        "PREEMPTED",
        "OUT_OF_MEMORY",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
        "SPECIAL_EXIT",
    }
)

#: One-line explanation per failure state, used in the alert text.
FAILURE_HELP = {
    "FAILED": "failed",
    "NODE_FAIL": "lost its node",
    "TIMEOUT": "hit its time limit",
    "PREEMPTED": "was preempted",
    "OUT_OF_MEMORY": "ran out of memory",
    "BOOT_FAIL": "failed to boot",
    "DEADLINE": "missed its deadline",
    "REVOKED": "was revoked",
    "SPECIAL_EXIT": "exited specially",
}

#: Human-readable explanations for the ``REASON`` codes most often seen at NYU Torch.
REASON_HELP = {
    "Priority": "eligible, waiting behind higher-priority work",
    "Resources": "eligible, waiting for free CPUs/GPUs/memory",
    "Dependency": "blocked by --dependency on another job",
    "QOSMaxGRESPerUser": "you already hold your QOS GPU limit",
    "QOSMaxCpuPerUser": "you already hold your QOS CPU limit",
    "QOSMaxJobsPerUserLimit": "you already hold your QOS job-count limit",
    "QOSMaxSubmitJobPerUserLimit": "you have too many submitted jobs for your QOS",
    # QOSGrp* limits are account-wide, unlike the per-user QOSMax* limits above.
    # QOSGrpGRES shows up on Torch when the project account has spent its GPUs.
    "QOSGrpGRES": "your account's QOS GPU limit is reached",
    "QOSGrpCPULimit": "your account's QOS CPU limit is reached",
    "QOSGrpMemLimit": "your account's QOS memory limit is reached",
    "QOSGrpCpuMinutesLimit": "your account's CPU-minute budget is exhausted",
    "AssocMaxJobsLimit": "account job limit reached",
    "AssocGrpGRES": "account GPU limit reached",
    "AssocGrpCPUMinutesLimit": "account CPU-minute budget exhausted",
    "JobHeldUser": "held by you (scontrol hold)",
    "JobHeldAdmin": "held by an administrator",
    "BeginTime": "waiting for its --begin time",
    "Reservation": "waiting for a reservation window",
    "NodeDown": "requested node is down",
    "PartitionDown": "requested partition is down",
    "PartitionNodeLimit": "partition node limit reached",
    "None": "",
    "": "",
}


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


#: Slurm's JSON encoder writes the *string* ``"None"`` where the C API has a null
#: reason code, so a running job otherwise reports its reason as "None".
_NULL_WORDS = {"", "none", "null", "n/a", "(null)"}


def _text_or_empty(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _NULL_WORDS else text


def format_job_id(
    job_id: int | None,
    array_job_id: int | None,
    array_task_id: int | None,
    array_task_string: str = "",
) -> str:
    """Render a job identifier the way ``squeue`` does.

    Three shapes occur, and both ``squeue`` and ``sacct`` need the same rendering:

    * ``17192918``            -- an ordinary job
    * ``17325640_42``         -- one task of an array
    * ``17325640_[44-95%4]``  -- a pending array, before tasks are handed out

    The JSON APIs report the array's task list with its brackets stripped, so they
    are restored here.
    """
    if array_task_id is not None and array_job_id:
        return f"{array_job_id}_{array_task_id}"
    if array_job_id:
        spec = array_task_string.strip()
        if spec and not spec.startswith("["):
            spec = f"[{spec}]"
        return f"{array_job_id}_{spec}" if spec else str(array_job_id)
    return str(job_id) if job_id is not None else "?"


#: Slurm filename pattern substitutions that can be resolved from the job record
#: alone. Anything else (step ids, node indices) is deliberately left alone: a
#: plausible-looking wrong path is worse than an obviously unexpanded one.
_LOG_PATTERNS = {
    "%j": lambda job: job.job_id,
    "%J": lambda job: job.job_id,
    "%A": lambda job: job.array_job_id,
    "%a": lambda job: job.array_task_id,
    "%x": lambda job: job.name,
    "%u": lambda job: job.user,
    "%N": lambda job: (job.nodelist.split(",")[0] if job.nodelist else None),
}


def expand_log_path(pattern: str, job: Job) -> str:
    """Resolve Slurm's ``%``-patterns in an ``--output``/``--error`` path.

    ``squeue`` normally reports the already-expanded path alongside the raw pattern,
    which is preferred. This is the fallback for when it does not, and it only
    substitutes patterns it can be sure about.

    ``train_gen_%A_%a.out`` on task 44 of array 17325640 becomes
    ``train_gen_17325640_44.out``; ``%s`` is left verbatim.
    """
    if "%" not in pattern:
        return pattern

    parts: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char != "%" or index + 1 >= len(pattern):
            parts.append(char)
            index += 1
            continue

        token = pattern[index : index + 2]
        index += 2
        if token == "%%":
            parts.append("%")
            continue

        resolver = _LOG_PATTERNS.get(token)
        if resolver is None:
            parts.append(token)
            continue

        value = resolver(job)
        # An unresolvable pattern (say %a on a non-array job) stays visible.
        parts.append(token if value is None else str(value))
    return "".join(parts)


@dataclass(frozen=True)
class Job:
    """One row of ``squeue`` for the monitored user."""

    job_id: int | None = None
    array_job_id: int | None = None
    array_task_id: int | None = None
    array_task_string: str = ""
    name: str = ""
    user: str = ""
    account: str = ""
    partition: str = ""
    qos: str = ""
    state: str = "UNKNOWN"
    reason: str = ""
    state_description: str = ""
    priority: int = 0
    submit_time: int | None = None
    eligible_time: int | None = None
    start_time: int | None = None
    end_time: int | None = None
    time_limit_min: int | None = None
    time_limit_infinite: bool = False
    node_count: int = 0
    nodelist: str = ""
    cpus: int = 0
    tres_req: str = ""
    tres_alloc: str = ""
    dependency: str = ""
    stdout: str = ""
    stderr: str = ""
    stdout_expanded: str = ""
    stderr_expanded: str = ""
    workdir: str = ""
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_wire(cls, record: dict) -> Job:
        """Build a job from one probe record, tolerating missing keys.

        Text fields go through :func:`_text_or_empty` because Slurm's JSON encoder
        stringifies absent values as ``"None"``.
        """
        return cls(
            job_id=_int_or_none(record.get("job_id")),
            array_job_id=_int_or_none(record.get("array_job_id")),
            array_task_id=_int_or_none(record.get("array_task_id")),
            array_task_string=_text_or_empty(record.get("array_task_string")),
            name=_text_or_empty(record.get("name")),
            user=_text_or_empty(record.get("user")),
            account=_text_or_empty(record.get("account")),
            partition=_text_or_empty(record.get("partition")),
            qos=_text_or_empty(record.get("qos")),
            state=(_text_or_empty(record.get("state")) or "UNKNOWN").upper(),
            reason=_text_or_empty(record.get("reason")),
            state_description=_text_or_empty(record.get("state_description")),
            priority=_int_or_none(record.get("priority")) or 0,
            submit_time=_int_or_none(record.get("submit_time")),
            eligible_time=_int_or_none(record.get("eligible_time")),
            start_time=_int_or_none(record.get("start_time")),
            end_time=_int_or_none(record.get("end_time")),
            time_limit_min=_int_or_none(record.get("time_limit_min")),
            time_limit_infinite=bool(record.get("time_limit_infinite")),
            node_count=_int_or_none(record.get("node_count")) or 0,
            nodelist=_text_or_empty(record.get("nodelist")),
            cpus=_int_or_none(record.get("cpus")) or 0,
            tres_req=_text_or_empty(record.get("tres_req")),
            tres_alloc=_text_or_empty(record.get("tres_alloc")),
            dependency=_text_or_empty(record.get("dependency")),
            stdout=_text_or_empty(record.get("stdout")),
            stderr=_text_or_empty(record.get("stderr")),
            stdout_expanded=_text_or_empty(record.get("stdout_expanded")),
            stderr_expanded=_text_or_empty(record.get("stderr_expanded")),
            workdir=_text_or_empty(record.get("workdir")),
            raw=record,
        )

    # ---------------------------------------------------------------- identity

    @property
    def display_id(self) -> str:
        """Slurm's own rendering: ``17192918``, ``17325640_42``, ``17325640_[44-95%4]``."""
        return format_job_id(
            self.job_id,
            self.array_job_id,
            self.array_task_id,
            self.array_task_string,
        )

    @property
    def is_array(self) -> bool:
        return bool(self.array_job_id) or bool(self.array_task_string)

    # ---------------------------------------------------------------- state

    @property
    def is_pending(self) -> bool:
        return self.state in {"PENDING", "CONFIGURING"}

    @property
    def is_running(self) -> bool:
        return self.state in STARTED_STATES

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def reason_text(self) -> str:
        """Plain-language gloss for the scheduler's reason code."""
        return REASON_HELP.get(self.reason, self.reason)

    # ---------------------------------------------------------------- resources

    @property
    def gpu_count(self) -> int | None:
        """GPUs actually held (running) or requested (pending)."""
        return gpu_count(self.tres_alloc) or gpu_count(self.tres_req)

    @property
    def gpu_type(self) -> str | None:
        return gpu_type(self.tres_req) or gpu_type(self.tres_alloc)

    @property
    def gpu_label(self) -> str:
        count = self.gpu_count
        if not count:
            return "-"
        kind = self.gpu_type
        return f"{count}x{kind}" if kind else str(count)

    @property
    def memory_label(self) -> str:
        return memory(self.tres_alloc) or memory(self.tres_req) or "-"

    @property
    def memory_limit_mb(self) -> float | None:
        """Memory the job asked for, in MB.

        Compared against the live RSS from ``sstat`` this is what turns a number into
        a warning: 40 GB means nothing until you know the job asked for 48 GB.
        """
        return memory_mb(memory(self.tres_alloc) or memory(self.tres_req))

    # ---------------------------------------------------------------- log files

    @property
    def log_path(self) -> str:
        """Where Slurm actually wrote stdout, with ``%j``-style patterns resolved.

        squeue reports both the raw ``--output=log/slurm-%j.out`` pattern and the
        expanded path; the expanded one wins because only it can be opened. When the
        API omits it, the common patterns are expanded locally.
        """
        if self.stdout_expanded:
            return self.stdout_expanded
        return expand_log_path(self.stdout, self)

    @property
    def error_log_path(self) -> str:
        """Same for stderr, falling back to stdout because ``--error`` is optional."""
        if self.stderr_expanded:
            return self.stderr_expanded
        if self.stderr:
            return expand_log_path(self.stderr, self)
        return self.log_path

    # ---------------------------------------------------------------- time metrics

    def queue_wait_sec(self, now: float) -> int | None:
        """How long the job waited in the queue.

        For a job that has started this is the *final* wait (``start - submit``);
        for one still pending it is the wait *so far* (``now - submit``). This is
        the number the dashboard exists to show, and it is intentionally the same
        quantity in both cases so the column is comparable across rows.
        """
        if self.submit_time is None:
            return None
        if self.start_time is not None and not self.is_pending:
            return max(0, self.start_time - self.submit_time)
        if self.is_pending:
            return max(0, int(now) - self.submit_time)
        return None

    def run_time_sec(self, now: float) -> int | None:
        """Elapsed run time; ``None`` until the job has started."""
        if self.start_time is None or self.is_pending:
            return None
        end = self.end_time if self.is_terminal and self.end_time else int(now)
        return max(0, end - self.start_time)

    def deadline(self) -> int | None:
        """When Slurm will kill the job, if that is knowable."""
        if self.end_time:
            return self.end_time
        if self.start_time and self.time_limit_min:
            return self.start_time + self.time_limit_min * 60
        return None

    def time_left_sec(self, now: float) -> int | None:
        """Seconds until the time limit expires; negative once it is overdue."""
        deadline = self.deadline()
        if deadline is None:
            return None
        return deadline - int(now)

    def used_fraction(self, now: float) -> float | None:
        """Fraction of the time limit consumed, for the progress bar."""
        if self.time_limit_infinite or not self.time_limit_min:
            return None
        run = self.run_time_sec(now)
        if run is None:
            return None
        return run / (self.time_limit_min * 60)

    def wait_phase(self, now: float) -> str:
        """Why a pending job is not running yet, in one word.

        ``blocked``    -- not eligible at all (dependency, hold, begin time).
        ``competing``  -- eligible and queued behind other work; Slurm will start
                          it the moment resources free up.
        ``scheduled``  -- has a start estimate in the future.
        ``-``          -- job is not pending.
        """
        if not self.is_pending:
            return "-"
        if self.eligible_time is None:
            return "blocked"
        if self.eligible_time > int(now):
            return "scheduled"
        return "competing"


@dataclass(frozen=True)
class Partition:
    """Capacity of one partition, aggregated from ``sinfo``.

    Note that Torch's partitions deliberately overlap: ``all`` and ``cpu_prem``
    pool nodes from every other partition, and the per-project partitions
    (``h200_tandon``, ``h200_public``, ...) all point at the same physical hardware.
    That is why there is no cluster-wide total anywhere in this model -- summing
    partitions would count the same GPUs several times. Read the rows, not a sum.
    """

    name: str = ""
    nodes_total: int = 0
    nodes_idle: int = 0
    nodes_mixed: int = 0
    nodes_allocated: int = 0
    nodes_other: int = 0
    cpus_total: int = 0
    cpus_allocated: int = 0
    mem_total_mb: int = 0
    mem_allocated_mb: int = 0
    gpus_total: int = 0
    gpus_used: int = 0
    #: A single GPU model, ``"mixed"`` when the partition pools several, else None.
    gpu_type: str | None = None
    gpu_types: tuple[str, ...] = ()
    max_time_min: int | None = None
    max_time_infinite: bool = False
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_wire(cls, record: dict) -> Partition:
        gpu_types = record.get("gpu_types") or ()
        return cls(
            name=_text_or_empty(record.get("name")),
            nodes_total=_int_or_none(record.get("nodes_total")) or 0,
            nodes_idle=_int_or_none(record.get("nodes_idle")) or 0,
            nodes_mixed=_int_or_none(record.get("nodes_mixed")) or 0,
            nodes_allocated=_int_or_none(record.get("nodes_allocated")) or 0,
            nodes_other=_int_or_none(record.get("nodes_other")) or 0,
            cpus_total=_int_or_none(record.get("cpus_total")) or 0,
            cpus_allocated=_int_or_none(record.get("cpus_allocated")) or 0,
            mem_total_mb=_int_or_none(record.get("mem_total_mb")) or 0,
            mem_allocated_mb=_int_or_none(record.get("mem_allocated_mb")) or 0,
            gpus_total=_int_or_none(record.get("gpus_total")) or 0,
            gpus_used=_int_or_none(record.get("gpus_used")) or 0,
            gpu_type=_text_or_empty(record.get("gpu_type")) or None,
            gpu_types=tuple(str(t) for t in gpu_types),
            max_time_min=_int_or_none(record.get("max_time_min")),
            max_time_infinite=bool(record.get("max_time_infinite")),
            raw=record,
        )

    @property
    def has_gpus(self) -> bool:
        return self.gpus_total > 0

    @property
    def gpus_free(self) -> int:
        """Free GPUs, floored at zero: two probes can disagree mid-refresh."""
        return max(0, self.gpus_total - self.gpus_used)

    @property
    def gpu_use_fraction(self) -> float | None:
        if not self.gpus_total:
            return None
        return min(1.0, self.gpus_used / self.gpus_total)

    @property
    def cpus_free(self) -> int:
        return max(0, self.cpus_total - self.cpus_allocated)

    @property
    def mem_free_gb(self) -> int:
        return max(0, self.mem_total_mb - self.mem_allocated_mb) // 1024

    @property
    def gpu_type_label(self) -> str:
        """``h200`` / ``mixed`` / ``-`` for a CPU-only partition."""
        if not self.has_gpus:
            return "-"
        return self.gpu_type or "gpu"

    @property
    def is_healthy(self) -> bool:
        """True when no node of the partition is down, drained or unreachable."""
        return self.nodes_other == 0


@dataclass(frozen=True)
class HistoryJob:
    """One record from ``sacct``: a job that ran, is running, or is still queued.

    This is the only source of *how* a job ended. ``squeue`` drops a job the moment
    it finishes, so without sacct a job that fails between two refreshes vanishes
    without a trace.
    """

    job_id: int | None = None
    array_job_id: int | None = None
    array_task_id: int | None = None
    name: str = ""
    partition: str = ""
    account: str = ""
    qos: str = ""
    state: str = "UNKNOWN"
    reason: str = ""
    #: Slurm's textual status: ``SUCCESS``, ``FAILED``, ``SIGNALED``.
    exit_status: str = ""
    exit_code: int | None = None
    exit_signal: int | None = None
    submit_time: int | None = None
    eligible_time: int | None = None
    start_time: int | None = None
    end_time: int | None = None
    elapsed_sec: int | None = None
    time_limit_min: int | None = None
    nodelist: str = ""
    node_count: int = 0
    gpus: int = 0
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_wire(cls, record: dict) -> HistoryJob:
        return cls(
            job_id=_int_or_none(record.get("job_id")),
            array_job_id=_int_or_none(record.get("array_job_id")),
            array_task_id=_int_or_none(record.get("array_task_id")),
            name=_text_or_empty(record.get("name")),
            partition=_text_or_empty(record.get("partition")),
            account=_text_or_empty(record.get("account")),
            qos=_text_or_empty(record.get("qos")),
            state=(_text_or_empty(record.get("state")) or "UNKNOWN").upper(),
            reason=_text_or_empty(record.get("reason")),
            exit_status=_text_or_empty(record.get("exit_status")),
            exit_code=_int_or_none(record.get("exit_code")),
            exit_signal=_int_or_none(record.get("exit_signal")),
            submit_time=_int_or_none(record.get("submit_time")),
            eligible_time=_int_or_none(record.get("eligible_time")),
            start_time=_int_or_none(record.get("start_time")),
            end_time=_int_or_none(record.get("end_time")),
            elapsed_sec=_int_or_none(record.get("elapsed_sec")),
            time_limit_min=_int_or_none(record.get("time_limit_min")),
            nodelist=_text_or_empty(record.get("nodelist")),
            node_count=_int_or_none(record.get("node_count")) or 0,
            gpus=_int_or_none(record.get("gpus")) or 0,
            raw=record,
        )

    @property
    def display_id(self) -> str:
        return format_job_id(self.job_id, self.array_job_id, self.array_task_id)

    @property
    def is_pending(self) -> bool:
        return self.state in {"PENDING", "CONFIGURING"}

    @property
    def is_running(self) -> bool:
        return self.state in STARTED_STATES

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_failed(self) -> bool:
        """Went wrong, as opposed to finishing cleanly or being cancelled."""
        return self.state in FAILURE_STATES

    @property
    def outcome(self) -> str:
        """Short human phrase for how the job ended, for alerts."""
        return FAILURE_HELP.get(self.state, self.state.lower().replace("_", " "))

    @property
    def exit_label(self) -> str:
        """Slurm's ``return_code:signal`` notation, e.g. ``0:0`` or ``1:0``."""
        if self.exit_code is None and self.exit_signal is None:
            return "-"
        return f"{self.exit_code or 0}:{self.exit_signal or 0}"

    def queue_wait_sec(self, now: float) -> int | None:
        """Same definition as :meth:`Job.queue_wait_sec`, so the two agree."""
        if self.submit_time is None:
            return None
        if self.start_time is not None and not self.is_pending:
            return max(0, self.start_time - self.submit_time)
        if self.is_pending:
            return max(0, int(now) - self.submit_time)
        return None


@dataclass(frozen=True)
class Alert:
    """Something worth telling the user about, raised by the history store.

    ``kind`` is the deduplication key alongside the job, so the same job never
    raises the same warning twice.
    """

    job_key: str
    at: float
    kind: str
    #: ``"error"`` for failures, ``"warning"`` for things worth noticing,
    #: ``"info"`` for confirmations such as a job finishing cleanly.
    severity: str
    message: str


@dataclass(frozen=True)
class Usage:
    """Live resource use for one running job, from ``sstat``.

    Every field is optional because ``sstat`` reports what the step has, and the
    ``extern`` step of an array task reports no memory at all. The figures are the
    maximum across the job's steps, never the sum: the ``extern`` step already
    aggregates the whole job, so adding the ``batch`` step would double count.
    """

    job_id: int | None = None
    max_rss_kb: int | None = None
    cpu_seconds: int | None = None
    tasks: int | None = None
    disk_read_bytes: int | None = None
    disk_write_bytes: int | None = None
    #: How many steps contributed, i.e. how much of the job ``sstat`` could see.
    steps: int = 0

    @classmethod
    def from_wire(cls, record: dict) -> Usage:
        return cls(
            job_id=_int_or_none(record.get("job_id")),
            max_rss_kb=_int_or_none(record.get("max_rss_kb")),
            cpu_seconds=_int_or_none(record.get("cpu_seconds")),
            tasks=_int_or_none(record.get("tasks")),
            disk_read_bytes=_int_or_none(record.get("disk_read_bytes")),
            disk_write_bytes=_int_or_none(record.get("disk_write_bytes")),
            steps=_int_or_none(record.get("steps")) or 0,
        )

    @property
    def max_rss_gb(self) -> float | None:
        """Peak resident memory in GiB, the number that predicts an OOM kill."""
        if self.max_rss_kb is None:
            return None
        return self.max_rss_kb / (1024 * 1024)


@dataclass(frozen=True)
class AccountUsage:
    """One account I can submit to, and what it is consuming.

    Scoped to the *account*, so it says nothing about limits: the caps that actually
    refuse a job live on the QOS, and are reported separately in
    :class:`QosPressure`. Keeping the two apart matters because one QOS can serve many
    accounts, so a QOS cap is not an account's cap.
    """

    account: str = ""
    qos: tuple[str, ...] = ()
    #: Current allocation across the whole account, not just my own jobs.
    running_jobs: int = 0
    running_gpus: int = 0
    running_cpus: int = 0
    running_nodes: int = 0
    running_users: int = 0
    #: Consumption over the accounting window, from ``sacct``.
    window_days: int = 0
    window_gpu_hours: float = 0.0
    window_cpu_hours: float = 0.0
    window_jobs: int = 0
    #: Fairshare signals. ``RawUsage`` is deliberately absent: it is Slurm's decayed
    #: usage in arbitrary units, not comparable across sites or interpretable alone.
    norm_shares: float | None = None
    effectv_usage: float | None = None
    fairshare: float | None = None
    level_fs: float | None = None
    #: True when the site configures a TRES-minutes allocation. Torch does not.
    budget_configured: bool = False
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_wire(cls, record: dict) -> AccountUsage:
        return cls(
            account=_text_or_empty(record.get("account")),
            qos=tuple(str(q) for q in (record.get("qos") or ())),
            running_jobs=_int_or_none(record.get("running_jobs")) or 0,
            running_gpus=_int_or_none(record.get("running_gpus")) or 0,
            running_cpus=_int_or_none(record.get("running_cpus")) or 0,
            running_nodes=_int_or_none(record.get("running_nodes")) or 0,
            running_users=_int_or_none(record.get("running_users")) or 0,
            window_days=_int_or_none(record.get("window_days")) or 0,
            window_gpu_hours=_float_or_none(record.get("window_gpu_hours")) or 0.0,
            window_cpu_hours=_float_or_none(record.get("window_cpu_hours")) or 0.0,
            window_jobs=_int_or_none(record.get("window_jobs")) or 0,
            norm_shares=_float_or_none(record.get("norm_shares")),
            effectv_usage=_float_or_none(record.get("effectv_usage")),
            fairshare=_float_or_none(record.get("fairshare")),
            level_fs=_float_or_none(record.get("level_fs")),
            budget_configured=bool(record.get("budget_configured")),
            raw=record,
        )

    @property
    def qos_label(self) -> str:
        return ",".join(self.qos) if self.qos else "-"

    @property
    def is_shared(self) -> bool:
        """True when someone else is also using this account right now."""
        return self.running_users > 1

    @property
    def priority_note(self) -> str:
        """Why this account wins or loses in the queue, in a few words.

        Slurm's fairshare factor falls as an association consumes more than its
        share, so comparing effective usage against the share it was given is the
        whole explanation for a low ``FairShare``.
        """
        if self.fairshare is None or self.effectv_usage is None or not self.norm_shares:
            return ""
        if self.effectv_usage > self.norm_shares * 2:
            return "over its share; priority suppressed"
        if self.effectv_usage < self.norm_shares / 2:
            return "under its share; priority boosted"
        return "near its share"


@dataclass(frozen=True)
class QosPressure:
    """A QOS my jobs use, and how close it is to its caps.

    Two caps matter and they are easy to confuse:

    * ``group_*`` -- the cap for the QOS as a whole, shared by every user. Reaching it
      is what Slurm reports as ``QOSGrpGRES``.
    * ``user_*`` -- the cap for a single user. Reaching it is ``QOSMaxGRESPerUser``,
      which is the far more common block on Torch (``gpu168`` allows 4 GPUs per user).
    """

    name: str = ""
    max_wall: str | None = None
    group_gpu_limit: int | None = None
    group_cpu_limit: int | None = None
    group_memory_limit_mb: float | None = None
    user_gpu_limit: int | None = None
    user_cpu_limit: int | None = None
    user_memory_limit_mb: float | None = None
    budget_configured: bool = False
    #: Cluster-wide usage inside this QOS, all users.
    running_gpus: int = 0
    running_cpus: int = 0
    running_jobs: int = 0
    running_users: int = 0
    #: My own usage inside this QOS.
    my_gpus: int = 0
    my_cpus: int = 0
    my_jobs: int = 0
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_wire(cls, record: dict) -> QosPressure:
        return cls(
            name=_text_or_empty(record.get("name")),
            max_wall=_text_or_empty(record.get("max_wall")) or None,
            group_gpu_limit=_int_or_none(record.get("group_gpu_limit")),
            group_cpu_limit=_int_or_none(record.get("group_cpu_limit")),
            group_memory_limit_mb=_float_or_none(record.get("group_memory_limit_mb")),
            user_gpu_limit=_int_or_none(record.get("user_gpu_limit")),
            user_cpu_limit=_int_or_none(record.get("user_cpu_limit")),
            user_memory_limit_mb=_float_or_none(record.get("user_memory_limit_mb")),
            budget_configured=bool(record.get("budget_configured")),
            running_gpus=_int_or_none(record.get("running_gpus")) or 0,
            running_cpus=_int_or_none(record.get("running_cpus")) or 0,
            running_jobs=_int_or_none(record.get("running_jobs")) or 0,
            running_users=_int_or_none(record.get("running_users")) or 0,
            my_gpus=_int_or_none(record.get("my_gpus")) or 0,
            my_cpus=_int_or_none(record.get("my_cpus")) or 0,
            my_jobs=_int_or_none(record.get("my_jobs")) or 0,
            raw=record,
        )

    @property
    def my_gpu_headroom(self) -> int | None:
        """GPUs I could still request in this QOS, from the per-user cap."""
        if self.user_gpu_limit is None:
            return None
        return max(0, self.user_gpu_limit - self.my_gpus)

    @property
    def group_gpu_headroom(self) -> int | None:
        """GPUs left in the QOS overall, from the group cap."""
        if self.group_gpu_limit is None:
            return None
        return max(0, self.group_gpu_limit - self.running_gpus)

    @property
    def blocks_new_gpu_job(self) -> bool:
        """True when asking for one more GPU here would be refused outright.

        This is the answer to "why did my job come back as QOSMaxGRESPerUser?", which
        no amount of staring at the job's own row can tell you.
        """
        return self.user_gpu_limit is not None and self.my_gpus >= self.user_gpu_limit

    @property
    def is_limited(self) -> bool:
        return self.user_gpu_limit is not None or self.group_gpu_limit is not None


@dataclass(frozen=True)
class Snapshot:
    """A complete, timestamped view of one probe run."""

    jobs: tuple[Job, ...]
    generated_at: int
    fetched_at: float
    hostname: str = ""
    user: str = ""
    slurm_version: str | None = None
    errors: tuple[str, ...] = ()
    timings_ms: dict = field(default_factory=dict)
    #: Wall-clock cost of the whole SSH round trip, shown in the status bar so a
    #: slow refresh is visibly the network's fault rather than the cluster's.
    probe_elapsed_sec: float | None = None
    #: Per-partition capacity, from ``sinfo``.
    partitions: tuple[Partition, ...] = ()
    #: Recent job records from ``sacct``, including jobs that have already ended.
    history: tuple[HistoryJob, ...] = ()
    #: How far back ``history`` reaches, echoed from the probe.
    history_hours: int | None = None
    #: Live resource use per running job, keyed by the numeric job id -- which is
    #: *not* the display id for an array task.
    usage: dict[int, Usage] = field(default_factory=dict)
    #: Accounts I can submit to, and what they are consuming.
    accounts: tuple[AccountUsage, ...] = ()
    #: How close the QOSes my jobs use are to their caps.
    qos_pressure: tuple[QosPressure, ...] = ()
    #: Accounting window behind ``accounts``; echoed from the probe.
    account_days: int | None = None
    #: Which collectors actually ran for this snapshot. With staggered polling most
    #: refreshes carry only ``jobs``.
    sections: tuple[str, ...] = ()
    #: When each optional section was really fetched. Separate from ``fetched_at``
    #: because :meth:`merged_with` can carry older data forward.
    partitions_at: float | None = None
    history_at: float | None = None
    usage_at: float | None = None
    accounts_at: float | None = None

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        fetched_at: float,
        probe_elapsed_sec: float | None = None,
    ) -> Snapshot:
        job_records = payload.get("jobs") or []
        partition_records = payload.get("partitions") or []
        history_records = payload.get("history") or []

        raw_sections = payload.get("sections")
        if raw_sections is None:
            # Hand-built or pre-sections payloads: infer from which keys exist.
            sections = tuple(
                name
                for name in ("jobs", "partitions", "history", "usage")
                if name in payload
            )
        else:
            sections = tuple(str(name) for name in raw_sections)

        accounts = tuple(
            AccountUsage.from_wire(r)
            for r in (payload.get("accounts") or [])
            if isinstance(r, dict)
        )
        qos_pressure = tuple(
            QosPressure.from_wire(r)
            for r in (payload.get("qos") or [])
            if isinstance(r, dict)
        )

        usage: dict[int, Usage] = {}
        for record in payload.get("usage") or []:
            if not isinstance(record, dict):
                continue
            entry = Usage.from_wire(record)
            if entry.job_id is not None:
                usage[entry.job_id] = entry

        return cls(
            jobs=tuple(Job.from_wire(r) for r in job_records if isinstance(r, dict)),
            generated_at=_int_or_none(payload.get("generated_at")) or int(fetched_at),
            fetched_at=fetched_at,
            hostname=payload.get("hostname") or "",
            user=payload.get("user") or "",
            slurm_version=payload.get("slurm_version"),
            errors=tuple(payload.get("errors") or ()),
            timings_ms=dict(payload.get("timings_ms") or {}),
            probe_elapsed_sec=probe_elapsed_sec,
            partitions=tuple(
                Partition.from_wire(r) for r in partition_records if isinstance(r, dict)
            ),
            history=tuple(
                HistoryJob.from_wire(r) for r in history_records if isinstance(r, dict)
            ),
            history_hours=_int_or_none(payload.get("history_hours")),
            usage=usage,
            accounts=accounts,
            qos_pressure=qos_pressure,
            account_days=_int_or_none(payload.get("account_days")),
            sections=sections,
            partitions_at=fetched_at if "partitions" in sections else None,
            history_at=fetched_at if "history" in sections else None,
            usage_at=fetched_at if "usage" in sections else None,
            accounts_at=fetched_at if "accounts" in sections else None,
        )

    def merged_with(self, previous: Snapshot | None) -> Snapshot:
        """Carry forward sections this snapshot did not ask for.

        Staggered polling means most refreshes request only the job list, leaving
        the partition, history and usage fields empty. Empty here means "not
        fetched", not "nothing there", so the last known values are carried across
        *with their own timestamps* -- that way the UI can report how stale they are
        instead of pretending they are current.
        """
        if previous is None:
            return self
        missing = {"jobs", "partitions", "history", "usage", "accounts"} - set(
            self.sections
        )
        if not missing:
            return self

        carried: dict = {}
        if "partitions" in missing and previous.partitions:
            carried["partitions"] = previous.partitions
            carried["partitions_at"] = previous.partitions_at
        if "history" in missing and previous.history:
            carried["history"] = previous.history
            carried["history_at"] = previous.history_at
        if "usage" in missing and previous.usage:
            carried["usage"] = previous.usage
            carried["usage_at"] = previous.usage_at
        if "accounts" in missing and previous.accounts:
            carried["accounts"] = previous.accounts
            carried["qos_pressure"] = previous.qos_pressure
            carried["account_days"] = previous.account_days
            carried["accounts_at"] = previous.accounts_at
        if not carried:
            return self
        return replace(self, **carried)

    # ---------------------------------------------------------------- aggregates

    def counts(self) -> dict[str, int]:
        """Job counts by coarse state group, used for the status line."""
        result = {"running": 0, "pending": 0, "other": 0}
        for job in self.jobs:
            if job.is_running:
                result["running"] += 1
            elif job.is_pending:
                result["pending"] += 1
            else:
                result["other"] += 1
        return result

    def longest_wait(self, now: float) -> int | None:
        """The longest queue wait in this snapshot; ``None`` when there are no jobs."""
        waits = [w for w in (job.queue_wait_sec(now) for job in self.jobs) if w is not None]
        return max(waits) if waits else None

    def age_sec(self, now: float) -> float:
        """Seconds since the cluster produced this snapshot."""
        return max(0.0, now - self.generated_at)

    def partitions_age_sec(self, now: float) -> float | None:
        """Seconds since partition capacity was fetched; ``None`` if never."""
        return None if self.partitions_at is None else max(0.0, now - self.partitions_at)

    def history_age_sec(self, now: float) -> float | None:
        """Seconds since job history was fetched; ``None`` if never."""
        return None if self.history_at is None else max(0.0, now - self.history_at)

    def usage_age_sec(self, now: float) -> float | None:
        """Seconds since live resource use was fetched; ``None`` if never."""
        return None if self.usage_at is None else max(0.0, now - self.usage_at)

    def accounts_age_sec(self, now: float) -> float | None:
        """Seconds since account and QOS data was fetched; ``None`` if never."""
        return None if self.accounts_at is None else max(0.0, now - self.accounts_at)

    def account_by_name(self, name: str) -> AccountUsage | None:
        for account in self.accounts:
            if account.account == name:
                return account
        return None

    def qos_by_name(self, name: str) -> QosPressure | None:
        for entry in self.qos_pressure:
            if entry.name == name:
                return entry
        return None

    def blocking_qos(self) -> list[QosPressure]:
        """QOSes where a further GPU request of mine would be refused.

        Reading this alongside a job's ``QOSMaxGRESPerUser`` reason turns an opaque
        block into a number: the cap, and how much of it I have already spent.
        """
        return [entry for entry in self.qos_pressure if entry.blocks_new_gpu_job]

    def usage_for(self, job: Job) -> Usage | None:
        """Live resource use for a job, if it is running and was measured.

        Keyed by the numeric job id, because an array task's display id is *not* its
        job id -- ``sstat`` reports the latter, and passing the display form selects
        every running task of the array at once.
        """
        if job.job_id is None:
            return None
        return self.usage.get(job.job_id)

    def by_id(self, display_id: str) -> Job | None:
        for job in self.jobs:
            if job.display_id == display_id:
                return job
        return None

    # ---------------------------------------------------------------- partitions

    def partition_by_name(self, name: str) -> Partition | None:
        for partition in self.partitions:
            if partition.name == name:
                return partition
        return None

    @property
    def my_partitions(self) -> tuple[str, ...]:
        """Names of the partitions my own jobs occupy, in sorted order."""
        return tuple(sorted({job.partition for job in self.jobs if job.partition}))

    def partitions_for_my_jobs(self) -> list[Partition]:
        """Capacity rows for the partitions I actually use.

        This is the view that answers "can my next job start soon?" without making
        the reader scan all 35 partitions, most of which I have no access to.
        """
        mine = set(self.my_partitions)
        return [p for p in self.partitions if p.name in mine]

    def held_gpus_by_partition(self) -> dict[str, int]:
        """GPUs my running jobs hold, per partition.

        Comparing this against the partition's free GPUs is what makes a
        ``QOSMaxGRESPerUser`` wait legible: the queue reason is my own quota, not
        cluster congestion.
        """
        held: dict[str, int] = {}
        for job in self.jobs:
            if job.is_running and job.partition:
                held[job.partition] = held.get(job.partition, 0) + (job.gpu_count or 0)
        return held

    # ---------------------------------------------------------------- history

    def finished(self, limit: int = 0) -> list[HistoryJob]:
        """Terminal history records, most recently ended first."""
        ended = [h for h in self.history if h.is_terminal and h.end_time]
        ended.sort(key=lambda h: h.end_time or 0, reverse=True)
        return ended[:limit] if limit and limit > 0 else ended

    def recent_failures(self, limit: int = 0) -> list[HistoryJob]:
        """History records that ended badly, most recent first."""
        failed = [h for h in self.history if h.is_failed]
        failed.sort(key=lambda h: h.end_time or 0, reverse=True)
        return failed[:limit] if limit and limit > 0 else failed

    def history_by_id(self, display_id: str) -> HistoryJob | None:
        for record in self.history:
            if record.display_id == display_id:
                return record
        return None


# ---------------------------------------------------------------- sorting

#: Presets cycled through with the ``s`` key. Ordered by usefulness on this cluster:
#: pending work sorted by how long it has been starving is the default question.
SORT_MODES: tuple[tuple[str, str], ...] = (
    ("wait", "queue wait"),
    ("state", "state"),
    ("runtime", "run time"),
    ("submit", "submit time"),
    ("partition", "partition"),
    ("name", "name"),
    ("priority", "priority"),
)


def sort_jobs(jobs: list[Job], mode: str, now: float, reverse: bool = False) -> list[Job]:
    """Sort jobs for display.

    Every mode falls back to a stable, meaningful tie-break so the table does not
    jitter between refreshes, and pending jobs always sort ahead of finished ones
    unless the mode says otherwise (``state`` and ``submit`` own that ordering).
    """

    def wait_key(job: Job) -> float:
        return job.queue_wait_sec(now) or -1

    def runtime_key(job: Job) -> float:
        return job.run_time_sec(now) or -1

    def state_rank(job: Job) -> int:
        if job.is_running:
            return 0
        if job.is_pending:
            return 1
        return 2

    keys = {
        "wait": lambda job: (state_rank(job), -wait_key(job), job.name),
        "state": lambda job: (state_rank(job), -wait_key(job), job.name),
        "runtime": lambda job: (state_rank(job), -runtime_key(job), job.name),
        "submit": lambda job: (-(job.submit_time or 0), job.name),
        "partition": lambda job: (job.partition, state_rank(job), -wait_key(job)),
        "name": lambda job: (job.name, state_rank(job), -wait_key(job)),
        "priority": lambda job: (state_rank(job), -(job.priority or 0), job.name),
    }
    key = keys.get(mode, keys["wait"])
    return sorted(jobs, key=key, reverse=reverse)
